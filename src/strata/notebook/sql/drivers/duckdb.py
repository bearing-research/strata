"""DuckDB driver adapter (embedded, file-backed).

Backed by the ``duckdb`` Python package's native DBAPI surface — not
ADBC. DuckDB ships its own first-class DBAPI driver that's stricter
about types than the ADBC bridge and exposes ``read_only=True`` at
``connect`` time directly. Skipping the ADBC translation layer
removes a moving part for a driver that already speaks Arrow
natively.

Scope: embedded mode only (path-based ``ConnectionSpec``). MotherDuck
and other remote modes are out of scope for this slice — they need
token-based auth in identity hashing and different freshness
semantics. They'll plug in as a separate adapter when needed.

Read-only enforcement is layered, mirroring SQLite:

1. **File-handle level** — file-backed connections open with
   ``read_only=True``; the engine refuses ``INSERT``/``UPDATE``/
   ``DELETE``/DDL.
2. **Session level** — every read-only open also opens a
   ``BEGIN TRANSACTION READ ONLY`` so an in-memory database (where
   ``read_only=True`` doesn't apply because there's no file to lock)
   still rejects writes at statement time.

Freshness is DB-wide (mirrors SQLite). DuckDB doesn't expose a
per-table change counter natively; we hash ``PRAGMA database_size``
output which advances when blocks are written. Acceptable for the
common "one DB per notebook" case; documented as a limitation.

Schema fingerprint is per-table via the ``duckdb_columns()``
function — catches metadata-only changes the freshness probe would
miss.

See ``docs/internal/design-sql-cells.md`` for the broader rationale.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import Any

from strata.notebook.sql.adapter import (
    AdapterCapabilities,
    ColumnInfo,
    FreshnessToken,
    QualifiedTable,
    SchemaFingerprint,
    TableSchema,
    hash_connection_identity,
)
from strata.notebook.sql.registry import register_adapter

_CAPABILITIES = AdapterCapabilities(
    # DuckDB exposes ``estimated_size`` per table, but it's a
    # coarse approximation that doesn't change reliably for small
    # writes. Treat freshness as DB-wide (same as SQLite) rather
    # than promise per-table semantics we can't keep.
    per_table_freshness=False,
    supports_snapshot=False,
    # ``PRAGMA database_size`` and ``duckdb_columns()`` are
    # statement-level reads — they don't get frozen inside an
    # active transaction the way ``pg_stat_*`` does. Same
    # connection can probe and query.
    needs_separate_probe_conn=False,
)

# DuckDB unquoted identifier — same shape as Postgres. Used to
# splice the schema/database name into ``duckdb_columns()`` filter
# predicates. Identifier validation is what keeps the splice
# injection-safe; bind parameters work for the values, but DuckDB's
# system functions complain less when the schema name is an
# identifier literal.
_DUCKDB_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class DuckDBAdapter:
    """Native-DBAPI driver adapter for DuckDB (embedded mode)."""

    name = "duckdb"
    sqlglot_dialect = "duckdb"
    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        connect_fn: Callable[..., Any] | None = None,
    ) -> None:
        # Test seam: pass a fake connect callable to bypass the real
        # duckdb import in unit tests. Signature matches
        # ``duckdb.connect(database, read_only=...)``.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # ``read_only`` is part of the Protocol so adapters that
        # route reads vs writes through different principals
        # (Snowflake's ``write_role``, BigQuery's
        # ``write_credentials_path``) can include only the
        # relevant fields. DuckDB embedded has no read/write
        # principal split, so the flag is a no-op here.
        del read_only
        return hash_connection_identity(self.name, self._extract_identity(spec))

    def _extract_identity(self, spec: Any) -> dict[str, Any]:
        """Identity for embedded DuckDB is just the absolute path.

        ``:memory:`` connections produce a stable id that's distinct
        from any on-disk path. Two specs that resolve to the same
        absolute path produce the same id; relative-path differences
        canonicalize away.
        """
        identity: dict[str, Any] = {}
        path = getattr(spec, "path", None)
        if path:
            if path == ":memory:":
                identity["path"] = ":memory:"
            else:
                identity["path"] = os.path.abspath(path)
        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open a DuckDB connection in the requested mode.

        Read-only enforcement is layered:

        1. **File-handle level** — for file-backed connections, we
           pass ``read_only=True`` to ``duckdb.connect``; the engine
           refuses to open a writable handle and rejects any DML/DDL
           before it touches storage. Cursors spawned from this
           connection inherit the file flag, so the RO guarantee
           propagates without extra work.
        2. **Session level** — every read-only open also issues
           ``BEGIN TRANSACTION READ ONLY`` on the parent and on
           every cursor it spawns (see ``_ReadOnlyDuckDB``). This
           catches in-memory databases (``:memory:``) where
           ``read_only=True`` cannot apply because the database is
           created on demand, plus any future quirk where the file
           flag might not propagate.

        Both layers together mean a SQL cell can't write to the
        database regardless of how the connection was specified.
        This is the security boundary, not SQL-text keyword
        filtering.
        """
        path = self._build_path(spec)
        is_memory = path == ":memory:"
        # ``duckdb.connect`` with ``read_only=True`` requires the
        # file to exist; a brand-new path can't be opened RO. For
        # memory connections, the flag would also fail. Fall back
        # to a writable handle and let the RO transaction enforce.
        connect_ro = read_only and not is_memory and os.path.exists(path)
        conn = self._invoke_connect(path, read_only=connect_ro)
        if not read_only:
            return conn
        # DuckDB's ``conn.cursor()`` returns a *separate* child
        # connection — one that doesn't share transaction state
        # with the parent. So issuing ``BEGIN TRANSACTION READ
        # ONLY`` on ``conn`` wouldn't apply to cursor-side
        # statements. Wrap so every spawned cursor enters its own
        # RO transaction. For file-backed RO opens this is
        # belt-and-suspenders; for ``:memory:`` it's the only thing
        # blocking writes.
        conn.execute("BEGIN TRANSACTION READ ONLY")
        return _ReadOnlyDuckDB(conn)

    def _build_path(self, spec: Any) -> str:
        path = getattr(spec, "path", None)
        if not path:
            raise RuntimeError("DuckDB connection requires ``path`` to be set")
        if path == ":memory:":
            return ":memory:"
        return os.path.abspath(path)

    def _invoke_connect(self, path: str, *, read_only: bool) -> Any:
        if self._connect_fn is not None:
            return self._connect_fn(path, read_only=read_only)
        try:
            import duckdb
        except ImportError as exc:
            raise RuntimeError(
                "duckdb is not installed; install with "
                "`uv pip install 'strata-notebook[sql-duckdb]'`"
            ) from exc
        return duckdb.connect(path, read_only=read_only)

    # --- probes -----------------------------------------------------------

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """DB-wide freshness via ``PRAGMA database_size``.

        ``tables`` is intentionally ignored — DuckDB doesn't expose
        per-table change counters (``estimated_size`` is too
        approximate for small writes), so every cell against this
        connection sees the same token. The pragma reports
        ``used_blocks`` and ``free_blocks`` which advance when DuckDB
        flushes block-aligned writes; a checkpoint or transaction
        commit makes the change visible to subsequent probes.

        Caveat: between block-aligned flushes, two distinct row
        states can produce the same token. Acceptable for the
        common "one DB per notebook" case; users running a notebook
        against a shared DB during active mutations should pin
        ``# @cache forever`` or ``# @cache off`` accordingly.
        """
        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            try:
                cursor.execute("PRAGMA database_size")
                rows = cursor.fetchall() or []
            except Exception:  # noqa: BLE001
                # In-memory connections may not have a database_size
                # row in some DuckDB versions. Fall back to a
                # session-only token rather than crash.
                return FreshnessToken(
                    value=b"duckdb-no-database-size",
                    is_session_only=True,
                )

        if not rows:
            return FreshnessToken(
                value=b"duckdb-empty-database-size",
                is_session_only=True,
            )

        # ``PRAGMA database_size`` columns vary slightly across DuckDB
        # versions but the row identity (db_name, total bytes,
        # used_blocks, free_blocks) is stable. Hash the whole row
        # rather than indexing — robust to additive schema changes.
        h.update(b"database_size:")
        for row in sorted(rows, key=lambda r: str(r[0]) if r else ""):
            for cell in row:
                h.update(str(cell).encode())
                h.update(b"\x00")
            h.update(b"\x00")
        return FreshnessToken(value=h.digest())

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``duckdb_columns()``.

        DuckDB's ``duckdb_columns()`` is a system-table view that
        carries column name, data type, and nullability for every
        column visible to the current session. Filtering by
        ``database_name`` / ``schema_name`` / ``table_name``
        scopes the read.

        Per-table fingerprint catches metadata-only changes
        (ADD COLUMN, type changes, nullability flips) that the
        DB-wide freshness probe would miss.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            for table in sorted(tables, key=lambda t: t.render()):
                rows = self._fetch_columns_for_table(cursor, table)
                h.update(table.render().encode())
                h.update(b"\x00")
                # Sort columns by name for deterministic ordering;
                # ``duckdb_columns()`` returns them in declaration
                # order which is what we want for *display*, but
                # for fingerprinting we need a canonical order so
                # ``ALTER TABLE ... REORDER`` (if it lands) doesn't
                # flip the token without a real schema change.
                for col_name, data_type, nullable in sorted(rows, key=lambda r: r[0]):
                    h.update(str(col_name).encode())
                    h.update(b":")
                    h.update(str(data_type).encode())
                    h.update(b":")
                    h.update(str(nullable).encode())
                    h.update(b"\x00")
                h.update(b"\x00")

        return SchemaFingerprint(value=h.digest())

    def _fetch_columns_for_table(
        self,
        cursor: Any,
        table: QualifiedTable,
    ) -> list[tuple[str, str, bool]]:
        """Return ``[(name, data_type, nullable), ...]`` for a table.

        DuckDB system functions accept bind parameters for filter
        values, so injection-safe scoping is possible without
        identifier splicing. We still validate the identifiers if
        present to fail fast on garbage input rather than silently
        returning zero rows.
        """
        sql_parts = [
            "SELECT column_name, data_type, is_nullable FROM duckdb_columns() WHERE table_name = ?",
        ]
        params: list[Any] = [table.name]
        if table.schema:
            if not _DUCKDB_IDENT_RE.fullmatch(table.schema):
                raise RuntimeError(
                    f"DuckDB schema name {table.schema!r} is not a valid "
                    "identifier; must match [a-zA-Z_][a-zA-Z0-9_]*"
                )
            sql_parts.append("AND schema_name = ?")
            params.append(table.schema)
        if table.catalog:
            if not _DUCKDB_IDENT_RE.fullmatch(table.catalog):
                raise RuntimeError(
                    f"DuckDB database name {table.catalog!r} is not a valid "
                    "identifier; must match [a-zA-Z_][a-zA-Z0-9_]*"
                )
            sql_parts.append("AND database_name = ?")
            params.append(table.catalog)
        sql = " ".join(sql_parts) + " ORDER BY column_index"
        cursor.execute(sql, params)
        rows = cursor.fetchall() or []
        out: list[tuple[str, str, bool]] = []
        for row in rows:
            if len(row) < 3:
                continue
            name = str(row[0])
            data_type = str(row[1])
            # ``is_nullable`` is BOOLEAN in duckdb_columns() (unlike
            # information_schema which uses 'YES'/'NO'). Coerce
            # defensively so either shape works.
            nullable = self._coerce_nullable(row[2])
            out.append((name, data_type, nullable))
        return out

    @staticmethod
    def _coerce_nullable(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.upper() in {"YES", "TRUE", "T", "1"}
        return bool(value)

    def list_schema(self, conn: Any) -> list[TableSchema]:
        """Enumerate user tables and views via ``duckdb_tables()`` /
        ``duckdb_views()`` joined to ``duckdb_columns()``.

        The ``system`` and ``temp`` databases plus the
        ``information_schema`` and ``pg_catalog`` schemas are
        filtered out — those are DuckDB's internal surface area,
        not anything a notebook author wrote.

        Each ``TableSchema`` carries the table's catalog (DuckDB
        ``database_name``), schema (``schema_name``), name, and a
        tuple of ``ColumnInfo`` (column name + data type +
        nullable).
        """
        out: list[TableSchema] = []
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT database_name, schema_name, table_name "
                "FROM duckdb_tables() "
                "WHERE NOT internal "
                "AND database_name NOT IN ('system', 'temp') "
                "AND schema_name NOT IN ('information_schema', 'pg_catalog') "
                "UNION ALL "
                "SELECT database_name, schema_name, view_name "
                "FROM duckdb_views() "
                "WHERE NOT internal "
                "AND database_name NOT IN ('system', 'temp') "
                "AND schema_name NOT IN ('information_schema', 'pg_catalog') "
                "ORDER BY 1, 2, 3"
            )
            objects = list(cursor.fetchall() or [])

            for catalog, schema, name in objects:
                col_rows = self._fetch_columns_for_table(
                    cursor,
                    QualifiedTable(catalog=catalog, schema=schema, name=name),
                )
                cols = tuple(ColumnInfo(name=c[0], type=c[1], nullable=c[2]) for c in col_rows)
                out.append(
                    TableSchema(
                        catalog=str(catalog) if catalog else None,
                        schema=str(schema) if schema else None,
                        name=str(name),
                        columns=cols,
                    )
                )
        return out


class _ReadOnlyDuckDB:
    """Proxy wrapping a DuckDB connection in read-only mode.

    DuckDB's ``conn.cursor()`` returns an independent child
    connection that does NOT inherit the parent's active
    transaction. So a ``BEGIN TRANSACTION READ ONLY`` on the parent
    is silently ignored by cursor-side statements — the executor's
    write attempts would succeed against an in-memory DB.

    This proxy intercepts ``cursor()`` so each new cursor
    immediately enters its own RO transaction. Everything else
    (``execute``, ``fetchall``, ``fetch_arrow_table``, ``close``,
    context-manager protocol) is forwarded transparently. Tests
    and integration callers don't need to know the proxy is here.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> Any:
        cur = self._conn.cursor()
        cur.execute("BEGIN TRANSACTION READ ONLY")
        return cur

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Any:
        self._conn.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._conn.__exit__(exc_type, exc, tb)

    def __getattr__(self, name: str) -> Any:
        # Forward anything we don't override directly to the
        # underlying handle. ``__getattr__`` is only consulted on
        # AttributeError lookups, so the explicit overrides above
        # take precedence.
        return getattr(self._conn, name)


_ADAPTER = DuckDBAdapter()


def register() -> None:
    """Register this adapter in the global SQL driver registry."""
    register_adapter(_ADAPTER)


# Auto-register on first import.
register()

"""SQLite driver adapter.

Backed by ``adbc-driver-sqlite``. The freshness probe combines
``PRAGMA data_version`` (database-wide write counter) with
``PRAGMA schema_version`` (DDL counter); both are DB-wide rather than
per-table because SQLite doesn't expose per-table change tracking.
The schema fingerprint is still per-table — column structure lives
in ``pragma_table_info``.

Read-only enforcement uses SQLite's URI ``mode=ro`` so the connection
is opened against an immutable DB handle; any ``INSERT`` / ``UPDATE``
/ ``DELETE`` is rejected at the engine before reaching disk.

See ``docs/internal/design-sql-cells.md`` for the SQLite-specific
gotcha list, especially: ``data_version`` only increments on writes
from *other* connections (Phase 1 SQL cells are read-only, so
self-writes that would defeat the probe aren't possible from cells).
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
    # SQLite has no per-table change counter — ``data_version`` is
    # database-wide, so any write to any table flips the freshness
    # token. Acceptable for the common "one DB per notebook" case;
    # documented as a known limitation for shared-DB setups.
    per_table_freshness=False,
    supports_snapshot=False,
    # Pragmas don't have transaction-frozen semantics like Postgres
    # ``pg_stat_*``; the same connection can probe and query.
    needs_separate_probe_conn=False,
)

# SQLite attached-database identifier — same shape as Postgres unquoted
# identifiers. Used to splice the schema name into the qualified pragma
# form (``"<schema>".pragma_table_info(?)``); pragma functions don't
# accept bind parameters in the schema position, so identifier
# validation is what keeps the splice injection-safe.
_SQLITE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ``pragma_table_info`` is the table-valued form of ``PRAGMA table_info(...)``;
# it accepts bind parameters and projects standard columns. The default
# query targets ``main``; the qualified form (``"<schema>".pragma_table_info(?)``)
# is built on demand for attached-database schemas.
_SCHEMA_QUERY_DEFAULT = """
SELECT name, type, "notnull", dflt_value, pk
FROM pragma_table_info(?)
ORDER BY cid
"""

# URI query parameters that don't change which objects the connection
# sees. Stripped from ``connection_id`` canonicalization so toggling
# read-only or asserting immutability doesn't perturb the cache key.
# Everything else (``cache``, ``mode=memory``, ``vfs``, named memory
# DBs, etc.) IS identity-shaping and MUST be preserved.
_NON_IDENTITY_ACCESS_MODES = frozenset({"ro", "rw", "rwc"})


def _split_query(query: str) -> list[tuple[str, str]]:
    """Split a URI query string into (key, raw_value) pairs."""
    if not query:
        return []
    out: list[tuple[str, str]] = []
    for part in query.split("&"):
        if not part:
            continue
        key, _, value = part.partition("=")
        out.append((key, value))
    return out


def _strip_non_identity_params(query: str) -> str:
    """Return the query string with non-identity params removed.

    Strips ONLY ``mode=ro|rw|rwc`` and ``immutable=1``. Other params
    (``cache``, ``mode=memory``, ``vfs``, ``psow``, etc.) stay because
    they affect either visibility (``cache=shared`` exposes a different
    object set) or how the file is read at a level that callers
    treat as identity. Output is sorted so two equivalent inputs
    produce the same canonical string.
    """
    kept: list[tuple[str, str]] = []
    for key, value in _split_query(query):
        if key == "mode" and value in _NON_IDENTITY_ACCESS_MODES:
            continue
        if key == "immutable":
            continue
        kept.append((key, value))
    if not kept:
        return ""
    return "&".join(f"{k}={v}" if v else k for k, v in sorted(kept))


def _build_schema_probe_query(table: QualifiedTable) -> tuple[str, tuple[str]]:
    """Build the schema-probe SQL for ``pragma_table_info`` on ``table``.

    SQLite's "schema" is the attached-database name (default ``main``).
    A qualified table — e.g. ``aux.events`` after ``ATTACH DATABASE
    'aux.db' AS aux`` — must run the pragma against ``aux``, not the
    default search order, otherwise we'd fingerprint a same-named
    table in the wrong DB.

    Pragma functions don't accept bind parameters in the schema
    position, so the schema name is splice-inlined. ``_SQLITE_IDENT_RE``
    rejects anything that isn't a plain identifier — that's the
    injection guard.
    """
    if not table.schema:
        return _SCHEMA_QUERY_DEFAULT, (table.name,)
    if not _SQLITE_IDENT_RE.match(table.schema):
        raise RuntimeError(
            f"SQLite attached-database name {table.schema!r} is not a "
            "valid identifier; must match [a-zA-Z_][a-zA-Z0-9_]*"
        )
    sql = (
        'SELECT name, type, "notnull", dflt_value, pk\n'
        f'FROM "{table.schema}".pragma_table_info(?)\n'
        "ORDER BY cid"
    )
    return sql, (table.name,)


def _force_mode_ro_in_uri(uri: str) -> str:
    """Replace any ``mode=ro|rw|rwc`` with ``mode=ro`` and append it
    when no ``mode=`` is present, except for ``mode=memory`` URIs.

    ``mode=memory`` is mutually exclusive with the access-mode values
    in SQLite's URI syntax — combining them would error. Memory DBs
    rely on the post-open ``PRAGMA query_only = ON`` for read-only
    enforcement.
    """
    if uri == ":memory:":
        return uri
    base, sep, query = uri.partition("?")
    if not sep:
        return f"{uri}?mode=ro"
    pairs = _split_query(query)
    if any(k == "mode" and v == "memory" for k, v in pairs):
        # Can't combine mode=memory with mode=ro; PRAGMA query_only
        # is the read-only enforcement for these URIs.
        return uri
    other = [(k, v) for k, v in pairs if k != "mode"]
    other.append(("mode", "ro"))
    return f"{base}?{'&'.join(f'{k}={v}' if v else k for k, v in other)}"


class SqliteAdapter:
    """ADBC-backed driver adapter for SQLite."""

    name = "sqlite"
    sqlglot_dialect = "sqlite"
    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        # Test seam: pass a fake connect callable to bypass the real
        # ADBC import in unit tests.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # ``read_only`` is part of the Protocol so adapters that
        # route reads vs writes through different principals
        # (Snowflake's ``write_role``, BigQuery's
        # ``write_credentials_path``) can include only the
        # relevant fields. SQLite has no read/write principal
        # split, so the flag is a no-op here.
        del read_only
        """Hash the absolute DB path (or URI / ``:memory:`` literal).

        Identity-shaping for SQLite is just "which file." Two specs
        that resolve to the same absolute path produce the same id;
        relative-path differences canonicalize away. ``:memory:``
        connections produce a stable id that's distinct from any
        on-disk path.
        """
        return hash_connection_identity(self.name, self._extract_identity(spec))

    def _extract_identity(self, spec: Any) -> dict[str, Any]:
        identity: dict[str, Any] = {}
        path = getattr(spec, "path", None)
        uri = getattr(spec, "uri", None)
        if uri:
            identity["uri"] = self._canonicalize_uri(uri)
        elif path:
            if path == ":memory:":
                identity["path"] = ":memory:"
            else:
                identity["path"] = os.path.abspath(path)
        return identity

    def _canonicalize_uri(self, uri: str) -> str:
        """Canonicalize a SQLite URI for identity hashing.

        Three things happen:
        1. ``mode=ro|rw|rwc`` and ``immutable=1`` are stripped — they
           change *how* we open, not which objects we see.
        2. All other query params (``cache``, ``mode=memory``,
           ``vfs``, ``psow``, named memory DBs) are PRESERVED. They
           affect visibility, locking, or which physical DB is opened
           — collapsing them onto the same id would alias distinct
           connections.
        3. The path portion of a non-memory ``file:`` URI is
           canonicalized to absolute. Memory-backed URIs (``mode=memory``)
           keep their bare name because it's a logical identifier,
           not a filesystem path.
        """
        if uri == ":memory:":
            return uri
        if not uri.startswith("file:"):
            # Non-URI form; treat as a path.
            return os.path.abspath(uri)

        rest = uri[len("file:") :]
        if "?" in rest:
            path_part, query = rest.split("?", 1)
        else:
            path_part, query = rest, ""

        # Drop leading "//" authority form.
        if path_part.startswith("//"):
            path_part = path_part[2:]
            if "/" in path_part:
                path_part = "/" + path_part.split("/", 1)[1]

        is_memory_uri = any(k == "mode" and v == "memory" for k, v in _split_query(query))

        if path_part and not is_memory_uri:
            path_part = os.path.abspath(path_part)

        canonical_query = _strip_non_identity_params(query)
        if canonical_query:
            return f"file:{path_part}?{canonical_query}"
        return f"file:{path_part}"

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC SQLite connection.

        Read-only enforcement is layered:

        1. **File-handle level** — for file-backed connections, the
           URI carries ``mode=ro`` so the SQLite engine refuses to
           open a writable handle. Any user-supplied access-mode
           (``mode=rwc``, ``mode=rw``) is overridden when the
           executor asks for read-only.
        2. **Session level** — every read-only open also issues
           ``PRAGMA query_only = ON``, which the engine consults on
           every statement. This catches in-memory databases
           (``:memory:`` and ``mode=memory`` URIs) where ``mode=ro``
           can't apply, plus any future quirk where ``mode=ro``
           might not propagate.

        Both layers together mean a SQL cell can't write to the
        database regardless of how the connection URI was specified.
        This is the security boundary, not SQL-text keyword filtering.
        """
        uri = self._build_uri(spec, read_only=read_only)
        conn = self._invoke_connect(uri)
        if read_only:
            with conn.cursor() as cursor:
                cursor.execute("PRAGMA query_only = ON")
        return conn

    def _build_uri(self, spec: Any, *, read_only: bool) -> str:
        path = getattr(spec, "path", None)
        existing_uri = getattr(spec, "uri", None)

        if existing_uri:
            if read_only:
                return _force_mode_ro_in_uri(existing_uri)
            return existing_uri

        if not path:
            raise RuntimeError("SQLite connection requires either ``path`` or ``uri`` to be set")

        if path == ":memory:":
            # ``mode=ro`` doesn't apply to in-memory DBs; the
            # ``PRAGMA query_only = ON`` in ``open()`` is the
            # enforcement.
            return ":memory:"

        abspath = os.path.abspath(path)
        if read_only:
            return f"file:{abspath}?mode=ro"
        return abspath

    def _invoke_connect(self, uri: str) -> Any:
        if self._connect_fn is not None:
            return self._connect_fn(uri)
        try:
            from adbc_driver_sqlite import dbapi as adbc_sqlite
        except ImportError as exc:
            raise RuntimeError(
                "adbc-driver-sqlite is not installed; install with "
                "`uv pip install 'strata-notebook[sql-sqlite]'`"
            ) from exc
        return adbc_sqlite.connect(uri)

    # --- probes -----------------------------------------------------------

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """DB-wide freshness via ``PRAGMA data_version`` + ``schema_version``.

        ``tables`` is intentionally ignored — SQLite doesn't expose
        per-table change counters, so every cell against this
        connection sees the same token. A write to any table (from
        another connection or process) increments ``data_version``;
        any DDL increments ``schema_version``.

        Caveat: ``data_version`` does NOT increment for writes on the
        connection it's queried on. Phase 1 SQL cells are read-only
        and the probe runs on its own usage path, so this gotcha
        doesn't bite us in practice — but the limitation is real and
        documented.
        """
        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            cursor.execute("PRAGMA data_version")
            data_row = cursor.fetchone()
            cursor.execute("PRAGMA schema_version")
            schema_row = cursor.fetchone()

        if data_row is None or schema_row is None:
            # Pragmas always return a row on a healthy SQLite handle;
            # missing here means the connection is broken. Surface as
            # a session-only token rather than crashing the executor.
            return FreshnessToken(value=b"sqlite-pragma-missing", is_session_only=True)

        h.update(b"data_version:")
        h.update(str(data_row[0]).encode())
        h.update(b":schema_version:")
        h.update(str(schema_row[0]).encode())
        return FreshnessToken(value=h.digest())

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``pragma_table_info``.

        Each ``QualifiedTable.schema`` (when set) is treated as the
        attached-database name. ``main`` is implicit when schema is
        None. This matters for queries against ``ATTACH DATABASE``
        targets — without the qualified pragma form, a probe of
        ``aux.events`` would silently fingerprint ``main.events``
        (or whatever the search-order resolution turns up).

        Pragma functions don't accept bind parameters in the schema
        position, so the schema name is splice-inlined and validated
        against the SQLite identifier pattern; an unsafe value raises
        before any SQL hits the connection.

        Per-table schema fingerprint catches metadata-only changes
        (ADD COLUMN, type changes, nullability flips) that the
        DB-wide freshness probe would miss.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            for table in sorted(tables, key=lambda t: t.render()):
                sql, params = _build_schema_probe_query(table)
                cursor.execute(sql, params)
                rows = cursor.fetchall() or []

                h.update(table.render().encode())
                h.update(b"\x00")
                for col_name, col_type, notnull, _dflt, _pk in rows:
                    h.update(str(col_name).encode())
                    h.update(b":")
                    h.update(str(col_type).encode())
                    h.update(b":")
                    h.update(str(notnull).encode())
                    h.update(b"\x00")
                h.update(b"\x00")

        return SchemaFingerprint(value=h.digest())

    def list_schema(self, conn: Any) -> list[TableSchema]:
        """Enumerate tables and views via ``sqlite_master`` + ``pragma_table_info``.

        SQLite has a single namespace per attached database; we
        report the implicit ``main`` schema only (attached
        databases would need an extra pass per attachment, deferred
        until users hit it). Views surface alongside tables so a
        notebook author sees the full readable surface; the
        ``type`` column on the result distinguishes them.
        """
        out: list[TableSchema] = []
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            names = [row[0] for row in cursor.fetchall() or []]

            for name in names:
                if not _SQLITE_IDENT_RE.fullmatch(name):
                    # Skip pathological names rather than splice
                    # them into a pragma — same defense as the
                    # schema-fingerprint probe.
                    continue
                cursor.execute(f'SELECT * FROM pragma_table_info("{name}")')
                cols: list[ColumnInfo] = []
                for row in cursor.fetchall() or []:
                    # pragma_table_info: cid, name, type, notnull, dflt_value, pk
                    col_name = str(row[1]) if len(row) > 1 else ""
                    col_type = str(row[2]) if len(row) > 2 else ""
                    notnull = bool(row[3]) if len(row) > 3 else False
                    cols.append(ColumnInfo(name=col_name, type=col_type, nullable=not notnull))
                out.append(
                    TableSchema(
                        catalog=None,
                        schema=None,
                        name=name,
                        columns=tuple(cols),
                    )
                )
        return out


_ADAPTER = SqliteAdapter()


def register() -> None:
    """Register this adapter in the global SQL driver registry."""
    register_adapter(_ADAPTER)


# Auto-register on first import.
register()

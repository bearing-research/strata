"""DuckDB driver adapter (embedded, file-backed).

Uses duckdb's native DBAPI rather than ADBC: it is stricter about types, takes
``read_only=True`` at ``connect``, and already speaks Arrow. Embedded mode only;
MotherDuck and other remote modes would need token auth in identity hashing and
different freshness semantics.

Read-only is enforced in two layers: ``read_only=True`` on file-backed
connections, and ``BEGIN TRANSACTION READ ONLY`` on every read-only open, which
also covers ``:memory:``. Freshness is DB-wide (``PRAGMA database_size``), since
DuckDB has no per-table change counter; the schema fingerprint is per table
(``duckdb_columns()``) and catches metadata-only changes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
    # DuckDB's per-table ``estimated_size`` doesn't change reliably for small
    # writes, so freshness is DB-wide (as in SQLite).
    per_table_freshness=False,
    supports_snapshot=False,
    # ``PRAGMA database_size`` and ``duckdb_columns()`` aren't frozen inside a
    # transaction (unlike ``pg_stat_*``), so one connection can probe and query.
    needs_separate_probe_conn=False,
)

# Spliced into ``duckdb_columns()`` predicates; this validation is what keeps
# the splice injection-safe.
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
        # Test seam; matches ``duckdb.connect(database, read_only=...)``.
        self._connect_fn = connect_fn

    # --- identity ---

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # DuckDB embedded has no read/write principal split, so ``read_only`` is a no-op.
        del read_only
        return hash_connection_identity(self.name, self._extract_identity(spec))

    def _extract_identity(self, spec: Any) -> dict[str, Any]:
        """Identity for embedded DuckDB: the absolute path.

        ``:memory:`` gets a stable id distinct from any path; relative-path
        differences canonicalize away.
        """
        identity: dict[str, Any] = {}
        path = getattr(spec, "path", None)
        if path:
            if path == ":memory:":
                identity["path"] = ":memory:"
            else:
                identity["path"] = os.path.abspath(path)
        # What a lake connection sees beyond the file: the catalog by name and
        # each mount by name. Their contents are in the freshness inputs.
        catalog = getattr(spec, "catalog", None)
        if catalog:
            identity["catalog"] = catalog
        mounts = getattr(spec, "mounts", None)
        if mounts:
            identity["mounts"] = sorted(mounts)
        return identity

    # --- connection lifecycle ---

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open a DuckDB connection in the requested mode.

        Read-only is the security boundary (not SQL keyword filtering) and is layered:
        file-backed connections pass ``read_only=True`` to ``duckdb.connect``, and every
        read-only open issues ``BEGIN TRANSACTION READ ONLY`` on the parent and on each
        cursor (see ``_ReadOnlyDuckDB``), which also covers ``:memory:``.
        """
        path = self._build_path(spec)
        catalog = getattr(spec, "catalog_properties", None)
        mounts = getattr(spec, "mount_sources", None)
        confine = getattr(spec, "confine_to", None)
        if catalog or mounts:
            if not read_only:
                raise RuntimeError("a connection's catalog and mounts are read-only")
            return self._open_lake(
                path, getattr(spec, "catalog", None), catalog, mounts or [], confine
            )
        is_memory = path == ":memory:"
        # ``read_only=True`` needs an existing file and fails for memory connections;
        # fall back to a writable handle and let the RO transaction enforce.
        connect_ro = read_only and not is_memory and os.path.exists(path)
        conn = self._invoke_connect(path, read_only=connect_ro)
        if confine is not None:
            _confine(conn, confine)
        if not read_only:
            return conn
        # ``conn.cursor()`` is a separate child connection without the parent's
        # transaction, so each cursor opens its own RO transaction. For ``:memory:``
        # this is the only thing blocking writes.
        conn.execute("BEGIN TRANSACTION READ ONLY")
        return _ReadOnlyDuckDB(conn)

    def _open_lake(
        self,
        path: str,
        catalog_name: str | None,
        catalog: dict[str, str] | None,
        mounts: list[dict[str, Any]],
        confine: list[str] | None = None,
    ) -> Any:
        """A read-only handle with the catalog attached and each mount a view.

        Views must exist before the read-only transaction starts and a read-only file
        cannot hold them, so the handle is an in-memory database with the file attached
        as default; views live in ``memory`` and resolve after the file's own tables.
        The executor resolves ``catalog_properties`` and ``mount_sources``.
        """
        conn = self._invoke_connect(":memory:", read_only=False)
        setup: list[str] = []
        if path != ":memory:":
            # The file's own name, as a plain connection calls it, unless
            # DuckDB reserves it or the catalog has it.
            stem = Path(path).stem
            taken = {"memory", "main", "system", "temp", catalog_name}
            alias = _ident(f"{stem}_file" if stem in taken else stem)
            mode = " (READ_ONLY)" if os.path.exists(path) else ""
            conn.execute(f"ATTACH {_literal(path)} AS {alias}{mode}")
            setup.append(f"SET search_path = {_literal(f'{alias}.main,memory.main')}")
        if any(_mount_scheme(m["uri"]) == "s3" for m in mounts) or catalog:
            conn.execute("INSTALL httpfs; LOAD httpfs")
        if catalog:
            assert catalog_name is not None
            conn.execute("INSTALL iceberg; LOAD iceberg")
            _attach_catalog(conn, catalog_name, catalog)
        for mount in mounts:
            _create_mount_view(conn, mount)
        for statement in setup:
            conn.execute(statement)
        if confine is not None:
            _confine(conn, confine)
        conn.execute("BEGIN TRANSACTION READ ONLY")
        return _ReadOnlyDuckDB(conn, setup)

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

    # --- probes ---

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """DB-wide freshness via ``PRAGMA database_size``; ``tables`` is ignored.

        DuckDB has no per-table change counter. ``used_blocks``/``free_blocks`` advance
        on block-aligned flushes, so between flushes two row states can share a token.
        Notebooks against a shared DB under active writes should use ``# @cache
        forever`` or ``# @cache off``.
        """
        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            try:
                cursor.execute("PRAGMA database_size")
                rows = cursor.fetchall() or []
            except Exception:  # noqa: BLE001
                # Some DuckDB versions have no database_size row for in-memory connections.
                return FreshnessToken(
                    value=b"duckdb-no-database-size",
                    is_session_only=True,
                )

        if not rows:
            return FreshnessToken(
                value=b"duckdb-empty-database-size",
                is_session_only=True,
            )

        # Columns vary across DuckDB versions; hash the whole row, robust to additive changes.
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

        Catches metadata-only changes (ADD COLUMN, type or nullability changes) that
        the DB-wide freshness probe misses.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            for table in sorted(tables, key=lambda t: t.render()):
                rows = self._fetch_columns_for_table(cursor, table)
                h.update(table.render().encode())
                h.update(b"\x00")
                # Canonical name order so a column reorder doesn't flip the token without a
                # real schema change.
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

        Filters are bind parameters; identifiers are still validated so garbage input
        fails fast instead of returning zero rows.
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
            # BOOLEAN here, 'YES'/'NO' in information_schema; coerce either.
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
        """Enumerate user tables and views with their columns.

        Skips DuckDB's internal surface: the ``system`` and ``temp`` databases and the
        ``information_schema`` and ``pg_catalog`` schemas.
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
    """Proxy that keeps a DuckDB connection's cursors read-only.

    ``conn.cursor()`` returns a child connection that does not inherit the parent's
    transaction, so a read-only ``BEGIN`` on the parent would not stop cursor-side
    writes to an in-memory DB. The proxy starts a read-only transaction on each new
    cursor and forwards everything else.
    """

    def __init__(self, conn: Any, setup: list[str] | None = None) -> None:
        self._conn = conn
        # Session settings (a lake handle's search path) do not carry to a
        # cursor's child connection either.
        self._setup = setup or []

    def cursor(self) -> Any:
        cur = self._conn.cursor()
        for statement in self._setup:
            cur.execute(statement)
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
        # ``__getattr__`` only runs on failed lookups, so the overrides above win.
        return getattr(self._conn, name)


_MOUNT_FORMATS = (("parquet", "read_parquet"), ("csv", "read_csv"), ("json", "read_json"))


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _confine(conn: Any, locations: list[str]) -> None:
    """Limit what SQL on *conn* can reach to *locations*, and lock the configuration.

    In service mode a SQL cell runs in the server process and could otherwise read
    any server file (e.g. ``/proc/self/environ``) or ``COPY ... TO`` / ``ATTACH``
    any path. Runs after the handle's own setup, which needed that access. A
    location ending in ``/`` admits everything under it; any other is one file.
    ``SET search_path``, re-issued by every cursor, still works under the lock.
    """
    directories = [location for location in locations if location.endswith("/")]
    files = [location for location in locations if not location.endswith("/")]
    conn.execute(f"SET allowed_directories = [{', '.join(map(_literal, directories))}]")
    conn.execute(f"SET allowed_paths = [{', '.join(map(_literal, files))}]")
    conn.execute("SET enable_external_access = false")
    conn.execute("SET lock_configuration = true")


def _mount_scheme(uri: str) -> str:
    from strata.notebook.mounts import parse_mount_uri

    return parse_mount_uri(uri)[0]


def _s3_secret(name: str, fields: dict[str, Any], scope: str | None = None) -> str | None:
    """``CREATE SECRET`` for S3 from fsspec or pyiceberg names, or None without any.

    Mount storage options use s3fs names (``key``, ``endpoint_url``); catalog
    properties use pyiceberg names (``s3.access-key-id``, ``s3.endpoint``).
    """
    client_kwargs = fields.get("client_kwargs") or {}
    values = {
        "KEY_ID": fields.get("key") or fields.get("s3.access-key-id"),
        "SECRET": fields.get("secret") or fields.get("s3.secret-access-key"),
        "SESSION_TOKEN": fields.get("token") or fields.get("s3.session-token"),
        "REGION": (
            fields.get("region_name") or client_kwargs.get("region_name") or fields.get("s3.region")
        ),
    }
    endpoint = fields.get("endpoint_url") or fields.get("s3.endpoint")
    if not endpoint and not any(values.values()):
        return None
    options = [f"{key} {_literal(str(value))}" for key, value in values.items() if value]
    if endpoint:
        parsed = urlparse(str(endpoint))
        options += [
            f"ENDPOINT {_literal(parsed.netloc or parsed.path)}",
            f"USE_SSL {'false' if parsed.scheme == 'http' else 'true'}",
            "URL_STYLE 'path'",
        ]
    if scope:
        options.append(f"SCOPE {_literal(scope)}")
    return f"CREATE OR REPLACE SECRET {_ident(name)} (TYPE s3, {', '.join(options)})"


def _attach_catalog(conn: Any, name: str, properties: dict[str, str]) -> None:
    """Attach an Iceberg REST catalog, from its pyiceberg properties, as *name*."""
    kind = properties.get("type", "rest")
    if kind != "rest":
        raise RuntimeError(f"DuckDB attaches REST catalogs; catalog {name!r} is {kind!r}")
    uri = properties.get("uri")
    if not uri:
        raise RuntimeError(f"catalog {name!r} has no uri")
    secret = f"strata_catalog_{name}"
    if properties.get("token"):
        conn.execute(
            f"CREATE OR REPLACE SECRET {_ident(secret)} "
            f"(TYPE iceberg, TOKEN {_literal(properties['token'])})"
        )
        auth = f"SECRET {_ident(secret)}"
    elif properties.get("credential"):
        client_id, _, client_secret = properties["credential"].rpartition(":")
        server = properties.get("oauth2-server-uri") or f"{uri.rstrip('/')}/v1/oauth/tokens"
        scope = f", OAUTH2_SCOPE {_literal(properties['scope'])}" if properties.get("scope") else ""
        conn.execute(
            f"CREATE OR REPLACE SECRET {_ident(secret)} (TYPE iceberg, "
            f"CLIENT_ID {_literal(client_id)}, CLIENT_SECRET {_literal(client_secret)}, "
            f"OAUTH2_SERVER_URI {_literal(server)}{scope})"
        )
        auth = f"SECRET {_ident(secret)}"
    else:
        auth = "AUTHORIZATION_TYPE 'none'"
    # Scoped to the warehouse, so it neither reaches other buckets nor answers
    # for a mount's; without an s3 warehouse to scope to, the catalog's vended
    # credentials are all the tables get.
    warehouse = properties.get("warehouse", "")
    s3 = (
        _s3_secret(f"strata_catalog_{name}_s3", properties, warehouse)
        if warehouse.startswith("s3://")
        else None
    )
    if s3:
        conn.execute(s3)
    conn.execute(
        f"ATTACH {_literal(properties.get('warehouse', ''))} AS {_ident(name)} "
        f"(TYPE iceberg, ENDPOINT {_literal(uri)}, {auth}, READ_ONLY)"
    )


def _create_mount_view(conn: Any, mount: dict[str, Any]) -> None:
    """``memory.main.<name>``: a view over the mount's Parquet, CSV or JSON files."""
    from strata.notebook.mounts import parse_mount_uri

    name, uri = mount["name"], mount["uri"]
    scheme, path = parse_mount_uri(uri)
    if scheme == "file":
        root = path
    elif scheme == "s3":
        root = f"s3://{path}"
        secret = _s3_secret(f"strata_mount_{name}", mount.get("storage_options") or {}, root)
        if secret:
            conn.execute(secret)
    else:
        raise RuntimeError(f"DuckDB reads file and s3 mounts; mount {name!r} is {scheme}")
    root = root.rstrip("/")
    for extension, reader in _MOUNT_FORMATS:
        if root.endswith(f".{extension}"):
            files = root
            break
        pattern = f"{root}/**/*.{extension}"
        (count,) = conn.execute(f"SELECT count(*) FROM glob({_literal(pattern)})").fetchone()
        if count:
            files = pattern
            break
    else:
        raise RuntimeError(f"mount {name!r} has no Parquet, CSV or JSON files under {uri}")
    conn.execute(
        f"CREATE VIEW memory.main.{_ident(name)} AS "
        f"SELECT * FROM {reader}({_literal(files)}, union_by_name = true)"
    )


_ADAPTER = DuckDBAdapter()


def register() -> None:
    """Register this adapter in the global SQL driver registry."""
    register_adapter(_ADAPTER)


register()

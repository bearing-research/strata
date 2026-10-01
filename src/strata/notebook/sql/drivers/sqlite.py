"""SQLite driver adapter, backed by ``adbc-driver-sqlite``.

SQLite has no per-table change tracking, so freshness is database-wide (file
state plus pragmas) while the schema fingerprint is per table. Read-only opens
use ``mode=ro`` plus ``PRAGMA query_only``. See ``docs/internal/design-sql-cells.md``.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path
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
    # ``data_version`` is database-wide, so any write flips the token. Fine for
    # one DB per notebook; a known limitation for shared DBs.
    per_table_freshness=False,
    supports_snapshot=False,
    # Pragmas aren't transaction-frozen, so one connection can probe and query.
    needs_separate_probe_conn=False,
)

# Pragma functions don't accept a bound schema, so identifier validation is
# what keeps the ``"<schema>".pragma_table_info(?)`` splice injection-safe.
_SQLITE_IDENT_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Targets ``main``; the qualified form is built on demand for attached schemas.
_SCHEMA_QUERY_DEFAULT = """
SELECT name, type, "notnull", dflt_value, pk
FROM pragma_table_info(?)
ORDER BY cid
"""

# Access modes don't change which objects the connection sees, so they are
# stripped from ``connection_id``. Everything else (``cache``,
# ``mode=memory``, ``vfs``, named memory DBs) shapes identity.
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
    """Return the query string with non-identity params removed, sorted.

    Strips only ``mode=ro|rw|rwc`` and ``immutable=1``. Others (``cache``,
    ``mode=memory``, ``vfs``, ``psow``) stay: they change which objects are visible.
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
    """Build the ``pragma_table_info`` schema-probe SQL for *table*.

    A qualified table's schema is the attached-database name, so ``aux.events``
    probes ``aux``, not whatever the search order finds. The schema name is inlined
    (pragmas take no bind there); ``_SQLITE_IDENT_RE`` is the injection guard.
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
    """Force ``mode=ro`` into a SQLite URI, except for ``mode=memory`` URIs.

    ``mode=memory`` cannot combine with an access mode; memory DBs rely on
    ``PRAGMA query_only = ON`` instead.
    """
    if uri == ":memory:":
        return uri
    base, sep, query = uri.partition("?")
    if not sep:
        return f"{uri}?mode=ro"
    pairs = _split_query(query)
    if any(k == "mode" and v == "memory" for k, v in pairs):
        # mode=memory can't combine with mode=ro; PRAGMA query_only enforces read-only.
        return uri
    other = [(k, v) for k, v in pairs if k != "mode"]
    other.append(("mode", "ro"))
    return f"{base}?{'&'.join(f'{k}={v}' if v else k for k, v in other)}"


def _database_path(probe_conn) -> str | None:
    """The file behind the connection's ``main`` database, or None in memory."""
    try:
        with probe_conn.cursor() as cursor:
            cursor.execute("PRAGMA database_list")
            rows = cursor.fetchall() or []
    except Exception:  # noqa: BLE001 — a broken handle is the caller's problem
        return None
    for row in rows:
        if len(row) >= 3 and str(row[1]) == "main":
            path = str(row[2] or "")
            return path or None
    return None


def _file_signals(path: str | None) -> list[bytes]:
    """What the database file says about how recently it changed."""
    if not path:
        return [b"memory"]
    signals: list[bytes] = []
    for candidate in (Path(path), Path(f"{path}-wal")):
        try:
            stat = candidate.stat()
        except OSError:
            signals.append(f"{candidate.name}:absent".encode())
            continue
        signals.append(f"{candidate.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    try:
        with open(path, "rb") as database:
            header = database.read(28)
    except OSError:
        return signals
    if len(header) >= 28:
        # Bumped on every commit reaching the main file: how a rollback-journal DB signals change.
        signals.append(b"change-counter:" + header[24:28])
    return signals


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
        # Test seam: a fake connect callable bypasses the real ADBC import.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # SQLite has no read/write principal split.
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

        Strips ``mode=ro|rw|rwc`` and ``immutable=1`` (how we open, not what we see);
        keeps every other param, since they change visibility, locking or which DB opens;
        makes a non-memory ``file:`` path absolute. Memory names stay as they are.
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

        Read-only is enforced in two layers, which together are the security boundary:
        ``mode=ro`` in the URI for file-backed databases (overriding any user access
        mode), and ``PRAGMA query_only = ON`` on every read-only open, which also covers
        in-memory databases.
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
            # ``mode=ro`` doesn't apply in memory; ``PRAGMA query_only`` in ``open()`` enforces it.
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
        """DB-wide freshness: the database file's own state, plus the pragmas.

        *tables* is ignored; SQLite has no per-table change counters. ``PRAGMA
        data_version`` alone is silently useless: it tracks other connections' writes
        as seen by this one, and the probe opens a new connection each run. So the file
        is asked directly: size, mtime, the header change counter, and the WAL's size
        and mtime. ``:memory:`` has only the pragmas, and nothing there can go stale.
        """
        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            cursor.execute("PRAGMA data_version")
            data_row = cursor.fetchone()
            cursor.execute("PRAGMA schema_version")
            schema_row = cursor.fetchone()

        if data_row is None or schema_row is None:
            # Missing means a broken connection: return a session-only token, don't crash.
            return FreshnessToken(value=b"sqlite-pragma-missing", is_session_only=True)

        h.update(b"data_version:")
        h.update(str(data_row[0]).encode())
        h.update(b":schema_version:")
        h.update(str(schema_row[0]).encode())
        for part in _file_signals(_database_path(probe_conn)):
            h.update(b":")
            h.update(part)
        return FreshnessToken(value=h.digest())

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``pragma_table_info``.

        ``QualifiedTable.schema`` is the attached-database name (``main`` when None);
        an unsafe name raises before any SQL runs. Catches metadata-only changes (ADD
        COLUMN, type or nullability) the DB-wide freshness probe misses.
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

        Reports only the ``main`` database, not attached ones; the ``type`` column tells
        tables from views.
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
                    # Skip unsafe names rather than splice them, as in the schema-fingerprint probe.
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


register()

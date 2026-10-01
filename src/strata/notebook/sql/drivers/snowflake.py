"""Snowflake driver adapter, backed by ``adbc-driver-snowflake``.

Freshness reads ``LAST_ALTERED`` from each touched database's own
``INFORMATION_SCHEMA.TABLES`` (one query per database); the schema fingerprint
reads ``INFORMATION_SCHEMA.COLUMNS``.

Read-only enforcement is role-based: Snowflake has no session read-only flag,
so the boundary is the grants of the connection's role. A read connection
should use a SELECT-only role; write cells need a role with DML grants.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from datetime import timedelta
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

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
from strata.notebook.sql.time_travel import iso_utc, pin_tables, plus

_CAPABILITIES = AdapterCapabilities(
    per_table_freshness=True,
    # No per-table snapshot id, but Time Travel queries a table as of a
    # timestamp, so a snapshot is a timestamp (``time_travel.py``).
    supports_snapshot=True,
    # Snowflake INFORMATION_SCHEMA isn't frozen inside a transaction, so the
    # probe can share the query connection.
    needs_separate_probe_conn=False,
)

# Validates role / warehouse / database names spliced into ``USE``
# statements, which don't accept bind parameters.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _spec_attr(spec: Any, key: str) -> Any:
    """Read a top-level field off a ``ConnectionSpec`` safely.

    Pydantic reserves names such as ``schema`` (a bound method), so this reads
    ``model_extra`` first and falls back to ``getattr``.
    """
    extras = getattr(spec, "model_extra", None) or {}
    if key in extras:
        return extras.get(key)
    value = getattr(spec, key, None)
    # Reject Pydantic-bound-method shadows (``spec.schema``).
    if callable(value) and getattr(value, "__self__", None) is not None:
        return None
    return value


def _resolve_session_defaults(cursor: Any) -> tuple[str | None, str | None]:
    """Return ``(CURRENT_DATABASE(), CURRENT_SCHEMA())``; either may be None.

    Probes resolve unqualified names against these defaults, as the query
    connection does, rather than assume ``PUBLIC``.
    """
    cursor.execute("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()")
    row = cursor.fetchone()
    if not row:
        return None, None
    db = str(row[0]) if row[0] else None
    sch = str(row[1]) if len(row) > 1 and row[1] else None
    return db, sch


def _parse_snowflake_uri(uri: str) -> dict[str, Any]:
    """Pull identity-shaping fields out of a gosnowflake URI.

    Shape: ``snowflake://<user>:<password>@<account>/<database>/<schema>?warehouse=…&role=…``.
    Returns whichever of ``account``, ``user``, ``database``, ``schema``,
    ``warehouse`` and ``role`` are non-empty; never the password.
    """
    out: dict[str, Any] = {}
    try:
        parsed = urlparse(uri)
    except Exception:  # noqa: BLE001
        return out

    if parsed.username:
        out["user"] = parsed.username

    # ``urlparse.hostname`` lowercases, but Snowflake account identifiers can
    # be case-sensitive; parse ``netloc`` to preserve case.
    netloc = parsed.netloc or ""
    if "@" in netloc:
        host_part = netloc.rsplit("@", 1)[1]
    else:
        host_part = netloc
    if ":" in host_part:
        host_part = host_part.split(":", 1)[0]
    if host_part:
        out["account"] = host_part

    # gosnowflake encodes db/schema in the path: /DB/SCHEMA.
    if parsed.path and parsed.path.startswith("/"):
        path_parts = [p for p in parsed.path.split("/") if p]
        if len(path_parts) >= 1:
            out["database"] = path_parts[0]
        if len(path_parts) >= 2:
            out["schema"] = path_parts[1]

    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=False)
        for key in ("warehouse", "role"):
            values = params.get(key)
            if values:
                out[key] = values[0]

    return out


def _resolve_var(value: str) -> str:
    """Resolve a single ``${VAR}`` indirection to its env-var value.

    Literals pass through: the writer scrubs them on save, so one can only come
    from unsaved in-memory state.
    """
    import os

    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        env_val = os.environ.get(var)
        if env_val is None:
            raise RuntimeError(
                f"Connection auth references ${{{var}}} but the environment variable is not set"
            )
        return env_val
    return value


class SnowflakeAdapter:
    """ADBC-backed driver adapter for Snowflake."""

    name = "snowflake"
    sqlglot_dialect = "snowflake"
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
        """Hash identity-shaping fields: account, user, role, warehouse, database, schema.

        Read cells fold ``role`` only, so changing ``write_role`` does not churn
        their caches; write cells fold ``write_role`` (falling back to ``role``),
        the role actually applied at open. Secrets are excluded.
        """
        return hash_connection_identity(
            self.name, self._extract_identity(spec, read_only=read_only)
        )

    def _extract_identity(self, spec: Any, *, read_only: bool = True) -> dict[str, Any]:
        identity: dict[str, Any] = {}

        # URI components are part of identity, or two uri-only connections to
        # different DBs or roles would share cache entries. Discrete fields below
        # override URI-derived ones.
        uri = _spec_attr(spec, "uri")
        if uri:
            identity.update(_parse_snowflake_uri(str(uri)))

        for key in (
            "account",
            "user",
            "role",
            "warehouse",
            "database",
            "schema",
        ):
            value = _spec_attr(spec, key)
            if value is not None:
                identity[key] = value

        # A read-only open applies only ``role``, so ``write_role`` joins only
        # write-cell identity.
        if not read_only:
            write_role = _spec_attr(spec, "write_role")
            if write_role is not None:
                identity["write_role"] = write_role

        # ``auth.user`` shapes object visibility. Resolve ${VAR} so specs naming the
        # same effective user get the same connection_id.
        auth = getattr(spec, "auth", None) or {}
        auth_user = auth.get("user")
        if auth_user:
            try:
                identity["user"] = _resolve_var(auth_user)
            except RuntimeError:
                # Env var unset: use the raw value so identity stays stable.
                identity["user"] = auth_user

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC Snowflake connection.

        Read-only is role-based: ``read_only=True`` applies the spec's ``role``
        (the user must make it SELECT-only); ``read_only=False`` applies
        ``write_role``, falling back to ``role``. Then ``USE`` sets warehouse,
        database and schema; ``USE`` takes no bind parameters, so identifiers
        are validated against ``_IDENTIFIER_RE`` first.
        """
        uri = self._build_uri(spec)
        conn = self._invoke_connect(uri)

        ro_role = _spec_attr(spec, "role")
        rw_role = _spec_attr(spec, "write_role") or ro_role
        chosen_role = ro_role if read_only else rw_role

        applied_any = False
        with conn.cursor() as cursor:
            for kw, value in (
                ("ROLE", chosen_role),
                ("WAREHOUSE", _spec_attr(spec, "warehouse")),
                ("DATABASE", _spec_attr(spec, "database")),
                ("SCHEMA", _spec_attr(spec, "schema")),
            ):
                if not value:
                    continue
                value_str = str(value)
                if not _IDENTIFIER_RE.match(value_str):
                    raise RuntimeError(
                        f"Connection {kw.lower()} {value!r} is not a valid Snowflake "
                        "identifier; must match [A-Za-z_][A-Za-z0-9_$]*"
                    )
                cursor.execute(f'USE {kw} "{value_str}"')
                applied_any = True

        if applied_any:
            commit = getattr(conn, "commit", None)
            if callable(commit):
                try:
                    commit()
                except Exception:
                    # USE statements autocommit, so an explicit commit may report "no
                    # transaction in progress"; no writes are lost.
                    pass

        return conn

    def _invoke_connect(self, uri: str) -> Any:
        if self._connect_fn is not None:
            return self._connect_fn(uri)
        try:
            from adbc_driver_snowflake import dbapi as adbc_snowflake
        except ImportError as exc:
            raise RuntimeError(
                "adbc-driver-snowflake is not installed; install with "
                "`uv pip install 'strata-notebook[sql-snowflake]'`"
            ) from exc
        return adbc_snowflake.connect(uri)

    def _build_uri(self, spec: Any) -> str:
        """Construct the gosnowflake connection URI from the spec.

        An explicit ``spec.uri`` wins over the discrete fields. ``${VAR}`` in
        ``auth.user`` / ``auth.password`` is resolved from the environment.
        """
        existing = _spec_attr(spec, "uri")
        if existing:
            return existing

        account = _spec_attr(spec, "account")
        if not account:
            raise RuntimeError(
                "Snowflake connection requires either ``uri`` or ``account`` to be set"
            )

        auth_raw = getattr(spec, "auth", None) or {}
        auth_user = auth_raw.get("user")
        auth_password = auth_raw.get("password")
        if auth_user:
            auth_user = _resolve_var(auth_user)
        if auth_password:
            auth_password = _resolve_var(auth_password)

        userinfo = ""
        if auth_user:
            userinfo = quote(auth_user, safe="")
            if auth_password:
                userinfo += ":" + quote(auth_password, safe="")
            userinfo += "@"

        path_parts = [str(account)]
        database = _spec_attr(spec, "database")
        if database:
            path_parts.append(str(database))
            schema = _spec_attr(spec, "schema")
            if schema:
                path_parts.append(str(schema))

        query_pairs: list[str] = []
        for key in ("warehouse", "role"):
            value = _spec_attr(spec, key)
            if value:
                query_pairs.append(f"{key}={quote(str(value), safe='')}")

        query = ("?" + "&".join(query_pairs)) if query_pairs else ""
        return f"snowflake://{userinfo}{'/'.join(path_parts)}{query}"

    # --- probes ----------------------------------------------------------

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """Per-table freshness via ``INFORMATION_SCHEMA.TABLES.LAST_ALTERED``.

        One round-trip per database; tables without a catalog use
        ``CURRENT_DATABASE()``. ``LAST_ALTERED`` moves even on 0-row DML, which
        over-invalidates but never under-. A missing table contributes a
        sentinel, distinct from any found table.
        """
        if not tables:
            return FreshnessToken(value=b"")

        by_catalog: dict[str | None, list[QualifiedTable]] = {}
        for t in tables:
            by_catalog.setdefault(t.catalog, []).append(t)

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            current_db, current_schema = _resolve_session_defaults(cursor)

            for catalog, group in sorted(
                by_catalog.items(),
                key=lambda kv: (kv[0] or "") + ":",
            ):
                effective_db = catalog or current_db
                if not effective_db:
                    for table in sorted(group, key=lambda t: t.render()):
                        h.update(b"no-database:")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                    continue

                if not _IDENTIFIER_RE.match(effective_db):
                    raise RuntimeError(
                        f"Snowflake database identifier {effective_db!r} is not valid"
                    )

                query = (
                    f"SELECT TABLE_SCHEMA, TABLE_NAME, LAST_ALTERED "
                    f'FROM "{effective_db}".INFORMATION_SCHEMA.TABLES '
                    f"WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?"
                )
                for table in sorted(group, key=lambda t: t.render()):
                    schema_arg = table.schema or current_schema
                    if not schema_arg:
                        # No schema on table or session: fold a sentinel rather than assume
                        # PUBLIC, so the key reflects the unresolved name.
                        h.update(b"no-schema:")
                        h.update(effective_db.encode())
                        h.update(b".")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                        continue
                    cursor.execute(query, (schema_arg, table.name))
                    row = cursor.fetchone()
                    h.update(effective_db.encode())
                    h.update(b".")
                    h.update(schema_arg.encode())
                    h.update(b".")
                    h.update(table.name.encode())
                    h.update(b":")
                    if row is None:
                        h.update(b"missing")
                    else:
                        # row = (schema, name, last_altered)
                        h.update(str(row[0]).encode())
                        h.update(b".")
                        h.update(str(row[1]).encode())
                        h.update(b":")
                        h.update(str(row[2]).encode())
                    h.update(b"\x00")

        return FreshnessToken(value=h.digest())

    def snapshot_timestamp(self, conn: Any) -> str:
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_TIMESTAMP()")
            row = cursor.fetchone()
        return iso_utc(row[0])

    def retention_until(self, conn: Any, tables: list[QualifiedTable], at: str) -> str | None:
        """``at`` plus the shortest ``RETENTION_TIME`` (days) of the tables.

        None when a table's retention cannot be read: a horizon that cannot be
        stated is not guessed.
        """
        if not tables:
            return None
        days: list[int] = []
        with conn.cursor() as cursor:
            current_db, current_schema = _resolve_session_defaults(cursor)
            for table in sorted(tables, key=lambda t: t.render()):
                database = table.catalog or current_db
                schema = table.schema or current_schema
                if not database or not schema:
                    return None
                if not _IDENTIFIER_RE.match(database):
                    raise RuntimeError(f"Snowflake database identifier {database!r} is not valid")
                cursor.execute(
                    f'SELECT RETENTION_TIME FROM "{database}".INFORMATION_SCHEMA.TABLES '
                    "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
                    (schema, table.name),
                )
                row = cursor.fetchone()
                if row is None or row[0] is None:
                    return None
                days.append(int(row[0]))
        return plus(at, timedelta(days=min(days)))

    def pin_query(self, sql: str, at: str) -> str:
        return pin_tables(
            sql,
            "snowflake",
            f"SELECT * FROM t AT (TIMESTAMP => '{at}'::TIMESTAMP_TZ)",
            "when",
        )

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``INFORMATION_SCHEMA.COLUMNS``.

        Mostly redundant with ``LAST_ALTERED``; kept for a metadata-only change
        that might not bump it.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        by_catalog: dict[str | None, list[QualifiedTable]] = {}
        for t in tables:
            by_catalog.setdefault(t.catalog, []).append(t)

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            current_db, current_schema = _resolve_session_defaults(cursor)

            for catalog, group in sorted(
                by_catalog.items(),
                key=lambda kv: (kv[0] or "") + ":",
            ):
                effective_db = catalog or current_db
                if not effective_db:
                    for table in sorted(group, key=lambda t: t.render()):
                        h.update(b"no-database:")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                    continue

                if not _IDENTIFIER_RE.match(effective_db):
                    raise RuntimeError(
                        f"Snowflake database identifier {effective_db!r} is not valid"
                    )

                query = (
                    f"SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
                    f'FROM "{effective_db}".INFORMATION_SCHEMA.COLUMNS '
                    f"WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? "
                    f"ORDER BY ORDINAL_POSITION"
                )
                for table in sorted(group, key=lambda t: t.render()):
                    schema_arg = table.schema or current_schema
                    if not schema_arg:
                        h.update(b"no-schema:")
                        h.update(effective_db.encode())
                        h.update(b".")
                        h.update(table.render().encode())
                        h.update(b"\x00")
                        continue
                    cursor.execute(query, (schema_arg, table.name))
                    rows = cursor.fetchall() or []

                    h.update(effective_db.encode())
                    h.update(b".")
                    h.update(schema_arg.encode())
                    h.update(b".")
                    h.update(table.name.encode())
                    h.update(b":")
                    for col_name, data_type, is_nullable in rows:
                        h.update(str(col_name).encode())
                        h.update(b":")
                        h.update(str(data_type).encode())
                        h.update(b":")
                        h.update(str(is_nullable).encode())
                        h.update(b"\x00")
                    h.update(b"\x00")

        return SchemaFingerprint(value=h.digest())

    def list_schema(self, conn: Any) -> list[TableSchema]:
        """Enumerate tables and views in the connection's default database only.

        Each extra database would cost an ``INFORMATION_SCHEMA`` query billed in
        cloud-services credits.
        """
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_DATABASE()")
            row = cursor.fetchone()
            if not row or not row[0]:
                return []
            db = str(row[0])

            if not _IDENTIFIER_RE.match(db):
                # ADBC drivers should quote their own identifiers; skip rather than splice
                # an unsafe value.
                return []

            query = (
                f"SELECT t.TABLE_CATALOG, t.TABLE_SCHEMA, t.TABLE_NAME, "
                f"       c.COLUMN_NAME, c.DATA_TYPE, c.IS_NULLABLE "
                f'  FROM "{db}".INFORMATION_SCHEMA.TABLES t '
                f'  JOIN "{db}".INFORMATION_SCHEMA.COLUMNS c '
                f"       ON c.TABLE_CATALOG = t.TABLE_CATALOG "
                f"      AND c.TABLE_SCHEMA  = t.TABLE_SCHEMA "
                f"      AND c.TABLE_NAME    = t.TABLE_NAME "
                f" WHERE t.TABLE_SCHEMA <> 'INFORMATION_SCHEMA' "
                f"   AND t.TABLE_TYPE IN ('BASE TABLE', 'VIEW') "
                f" ORDER BY t.TABLE_SCHEMA, t.TABLE_NAME, c.ORDINAL_POSITION"
            )
            cursor.execute(query)
            rows = cursor.fetchall() or []

        grouped: dict[tuple[str | None, str | None, str], list[ColumnInfo]] = {}
        order: list[tuple[str | None, str | None, str]] = []
        for cat, sch, name, col_name, data_type, nullable_str in rows:
            key = (cat or None, sch or None, str(name))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(
                ColumnInfo(
                    name=str(col_name),
                    type=str(data_type),
                    nullable=(str(nullable_str).upper() == "YES"),
                )
            )

        return [
            TableSchema(
                catalog=cat,
                schema=sch,
                name=name,
                columns=tuple(grouped[(cat, sch, name)]),
            )
            for (cat, sch, name) in order
        ]


_ADAPTER = SnowflakeAdapter()


def register() -> None:
    """Register this adapter; re-registering replaces the entry, so it is idempotent."""
    register_adapter(_ADAPTER)


register()

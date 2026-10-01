"""PostgreSQL driver adapter, backed by ``adbc-driver-postgresql``.

Freshness combines ``pg_stat_user_tables`` DML counters with
``pg_class.relfilenode`` to catch data changes and rewrite-style DDL. Read-only
mode sets ``default_transaction_read_only = on`` so the engine rejects writes.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from typing import Any
from urllib.parse import quote, urlparse, urlunparse

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
    per_table_freshness=True,
    supports_snapshot=False,
    needs_separate_probe_conn=True,
)

# ``SET ROLE`` and ``SET search_path`` take no bind parameters, so spliced
# values must match this to rule out injection.
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# ``to_regclass($1)`` resolves unqualified names through the real
# ``search_path``; the resolved ``nspname`` goes into the fingerprint so
# different schemas give different tokens.
_FRESHNESS_QUERY = """
SELECT
    COALESCE(s.n_tup_ins, 0)
        + COALESCE(s.n_tup_upd, 0)
        + COALESCE(s.n_tup_del, 0) AS dml,
    c.relfilenode,
    n.nspname AS resolved_schema
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE c.oid = to_regclass($1)
"""

# ``pg_attribute`` rather than ``information_schema.columns``: filters by the
# ``to_regclass`` OID without a second round-trip.
_SCHEMA_QUERY = """
SELECT
    a.attname,
    format_type(a.atttypid, a.atttypmod) AS data_type,
    NOT a.attnotnull AS is_nullable
FROM pg_attribute a
WHERE a.attrelid = to_regclass($1)
  AND a.attnum > 0
  AND NOT a.attisdropped
ORDER BY a.attnum
"""


def _resolve_var(value: str) -> str:
    """Resolve a single ``${VAR}`` indirection to its env-var value.

    Literal strings pass through: the writer scrubs literals on save, so one can
    only come from unsaved in-memory state.
    """
    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        env_val = os.environ.get(var)
        if env_val is None:
            raise RuntimeError(
                f"Connection auth references ${{{var}}} but the environment variable is not set"
            )
        return env_val
    return value


class PostgresAdapter:
    """ADBC-backed driver adapter for PostgreSQL."""

    name = "postgresql"
    sqlglot_dialect = "postgres"
    capabilities = _CAPABILITIES

    def __init__(
        self,
        *,
        connect_fn: Callable[[str], Any] | None = None,
    ) -> None:
        # Test seam; by default ``open()`` lazy-imports the ADBC driver.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # Postgres has no read/write principal split (the role governs both).
        del read_only
        """Hash identity-shaping fields, excluding secrets and runtime tunables.

        Identity-shaping for Postgres: host, port, database, user, role,
        search_path. Excluded: password (secret), application_name and
        ``connect_timeout`` (runtime tunables — they don't change which
        objects the connection sees).
        """
        return hash_connection_identity(self.name, self._extract_identity(spec))

    def _extract_identity(self, spec: Any) -> dict[str, Any]:
        identity: dict[str, Any] = {}

        # The URI's password is never identity-shaping.
        uri = getattr(spec, "uri", None)
        if uri:
            parsed = urlparse(uri)
            if parsed.hostname:
                identity["host"] = parsed.hostname
            if parsed.port is not None:
                identity["port"] = parsed.port
            if parsed.path and parsed.path.startswith("/") and len(parsed.path) > 1:
                identity["database"] = parsed.path[1:]
            if parsed.username:
                identity["user"] = parsed.username

        # Discrete keys override URI components.
        for key in ("host", "port", "database", "user", "role"):
            value = getattr(spec, key, None)
            if value is not None:
                identity[key] = value

        # `auth.user` changes object visibility. Resolve ${VAR} so specs with
        # the same effective user share a connection_id. Password excluded.
        auth = getattr(spec, "auth", None) or {}
        auth_user = auth.get("user")
        if auth_user:
            try:
                identity["user"] = _resolve_var(auth_user)
            except RuntimeError:
                # Env var unset: the raw value keeps the identity stable.
                identity["user"] = auth_user

        # `search_path` changes which schema unqualified names resolve to.
        options = getattr(spec, "options", None) or {}
        sp = options.get("search_path")
        if sp:
            identity["search_path"] = sp

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC PostgreSQL connection.

        ``read_only=True`` sets ``default_transaction_read_only`` so the engine
        rejects writes; that, not keyword filtering, is the security boundary.
        Then applies ``role`` and ``options.search_path``, the same fields
        ``canonicalize_connection_id`` hashes, so cache identity matches the
        live session.
        """
        uri = self._build_uri(spec)
        conn = self._invoke_connect(uri)

        applied_any = False
        with conn.cursor() as cursor:
            if read_only:
                cursor.execute("SET default_transaction_read_only = on")
                applied_any = True

            role = getattr(spec, "role", None)
            if role:
                if not _IDENTIFIER_RE.match(str(role)):
                    raise RuntimeError(
                        f"Connection role {role!r} is not a valid Postgres "
                        "identifier; must match [a-zA-Z_][a-zA-Z0-9_]*"
                    )
                # Validated identifier; double-quoted to preserve case.
                cursor.execute(f'SET ROLE "{role}"')
                applied_any = True

            options = getattr(spec, "options", None) or {}
            search_path = options.get("search_path")
            if search_path:
                cursor.execute(f"SET search_path TO {_format_search_path(search_path)}")
                applied_any = True

        if applied_any:
            conn.commit()

        return conn

    def _invoke_connect(self, uri: str) -> Any:
        if self._connect_fn is not None:
            return self._connect_fn(uri)
        try:
            from adbc_driver_postgresql import dbapi as adbc_postgres
        except ImportError as exc:
            raise RuntimeError(
                "adbc-driver-postgresql is not installed; install with "
                "`uv pip install 'strata-notebook[sql-postgres]'`"
            ) from exc
        return adbc_postgres.connect(uri)

    def _build_uri(self, spec: Any) -> str:
        """Construct the ADBC connection URI from the spec.

        Uses ``spec.uri`` with resolved auth spliced into its userinfo, or else
        builds from ``host``/``port``/``database``. Raises ``RuntimeError`` when
        a ``${VAR}`` in ``auth`` names a missing env var.
        """
        auth_raw = getattr(spec, "auth", None) or {}
        auth_user = auth_raw.get("user")
        auth_password = auth_raw.get("password")
        if auth_user:
            auth_user = _resolve_var(auth_user)
        if auth_password:
            auth_password = _resolve_var(auth_password)

        uri = getattr(spec, "uri", None)
        if uri:
            if auth_user or auth_password:
                return _splice_userinfo(uri, auth_user, auth_password)
            return uri

        host = getattr(spec, "host", None) or "localhost"
        port = getattr(spec, "port", None) or 5432
        database = getattr(spec, "database", None) or "postgres"
        user = auth_user or getattr(spec, "user", None) or "postgres"
        password = auth_password or ""

        userinfo = quote(user, safe="")
        if password:
            userinfo += ":" + quote(password, safe="")
        return f"postgresql://{userinfo}@{host}:{port}/{quote(database, safe='')}"

    # --- probes -----------------------------------------------------------

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """Per-table freshness via DML counters, relfilenode and resolved schema.

        ``to_regclass`` resolves unqualified names through the live
        ``search_path``, and the resolved schema is part of the digest. An
        unresolvable name makes the token ``is_session_only``.
        """
        if not tables:
            return FreshnessToken(value=b"")

        any_missing = False
        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            for table in sorted(tables, key=lambda t: t.render()):
                cursor.execute(_FRESHNESS_QUERY, (_to_regclass_arg(table),))
                row = cursor.fetchone()

                h.update(table.render().encode())
                h.update(b"\x00")
                if row is None:
                    any_missing = True
                    h.update(b"missing")
                else:
                    dml, relfilenode, resolved_schema = row
                    h.update(str(resolved_schema).encode())
                    h.update(b":")
                    h.update(str(dml).encode())
                    h.update(b":")
                    h.update(str(relfilenode).encode())
                h.update(b"\x00")

        return FreshnessToken(value=h.digest(), is_session_only=any_missing)

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``pg_attribute`` and ``to_regclass``.

        Catches ``ADD COLUMN``, type changes and nullability flips that the
        freshness probe misses.
        """
        if not tables:
            return SchemaFingerprint(value=b"")

        h = hashlib.sha256()
        with probe_conn.cursor() as cursor:
            for table in sorted(tables, key=lambda t: t.render()):
                cursor.execute(_SCHEMA_QUERY, (_to_regclass_arg(table),))
                rows = cursor.fetchall() or []

                h.update(table.render().encode())
                h.update(b"\x00")
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
        """Enumerate user tables and views via ``information_schema`` in one round-trip.

        System schemas (``pg_catalog``, ``information_schema``) are excluded.
        """
        query = (
            "SELECT t.table_catalog, t.table_schema, t.table_name, "
            "       c.column_name, c.data_type, c.is_nullable "
            "  FROM information_schema.tables t "
            "  JOIN information_schema.columns c "
            "       ON c.table_catalog = t.table_catalog "
            "      AND c.table_schema  = t.table_schema "
            "      AND c.table_name    = t.table_name "
            " WHERE t.table_schema NOT IN ('pg_catalog', 'information_schema') "
            "   AND t.table_type IN ('BASE TABLE', 'VIEW') "
            " ORDER BY t.table_schema, t.table_name, c.ordinal_position"
        )
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall() or []

        # Keep the query's ordinal_position order.
        grouped: dict[tuple[str | None, str | None, str], list[ColumnInfo]] = {}
        order: list[tuple[str | None, str | None, str]] = []
        for row in rows:
            cat, sch, name, col_name, col_type, nullable_str = row
            key = (cat or None, sch or None, str(name))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(
                ColumnInfo(
                    name=str(col_name),
                    type=str(col_type),
                    nullable=(str(nullable_str).upper() == "YES"),
                )
            )

        return [
            TableSchema(
                catalog=cat, schema=sch, name=name, columns=tuple(grouped[(cat, sch, name)])
            )
            for (cat, sch, name) in order
        ]


def _to_regclass_arg(table: QualifiedTable) -> str:
    """Build the ``to_regclass($1)`` argument: ``"schema"."name"`` or ``"name"``.

    Quoting preserves identifier case; embedded double quotes are escaped.
    """
    parts = []
    if table.schema:
        parts.append(table.schema)
    parts.append(table.name)
    return ".".join(f'"{p.replace(chr(34), chr(34) * 2)}"' for p in parts)


def _format_search_path(value: Any) -> str:
    """Format a ``search_path`` option (comma string or list) as ``SET search_path TO ...``.

    ``SET search_path`` takes no bind parameters, so validating each name against
    the unquoted-identifier pattern is the injection defense.
    """
    if isinstance(value, str):
        names = [s.strip() for s in value.split(",") if s.strip()]
    elif isinstance(value, (list, tuple)):
        names = [str(s).strip() for s in value if str(s).strip()]
    else:
        raise RuntimeError(
            f"options.search_path must be a string or list, got {type(value).__name__}"
        )
    if not names:
        raise RuntimeError("options.search_path is empty after parsing")
    for name in names:
        if not _IDENTIFIER_RE.match(name):
            raise RuntimeError(
                f"search_path entry {name!r} is not a valid Postgres "
                "identifier; must match [a-zA-Z_][a-zA-Z0-9_]*"
            )
    return ", ".join(f'"{name}"' for name in names)


def _splice_userinfo(
    uri: str,
    user: str | None,
    password: str | None,
) -> str:
    """Return ``uri`` with the user/password portion of the userinfo replaced.

    Host, port and path stay intact. ``user=None`` keeps the URI's existing user.
    """
    parsed = urlparse(uri)
    host = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port is not None else ""
    final_user = user if user else parsed.username

    if final_user:
        userinfo = quote(final_user, safe="")
        if password:
            userinfo += ":" + quote(password, safe="")
        netloc = f"{userinfo}@{host}{port}"
    else:
        netloc = f"{host}{port}"

    return urlunparse(parsed._replace(netloc=netloc))


_ADAPTER = PostgresAdapter()


def register() -> None:
    """Register this adapter in the global SQL driver registry.

    A callable so tests can re-register after ``_reset_for_tests`` without
    reloading the module.
    """
    register_adapter(_ADAPTER)


register()

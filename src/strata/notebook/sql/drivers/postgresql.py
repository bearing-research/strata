"""PostgreSQL driver adapter.

Backed by ``adbc-driver-postgresql``. The freshness probe combines
``pg_stat_user_tables`` cumulative DML counters with
``pg_class.relfilenode`` to catch both data changes and rewrite-style
DDL. Read-only enforcement uses
``SET default_transaction_read_only = on`` so every transaction the
session opens rejects ``INSERT``/``UPDATE``/``DELETE`` at the engine.

See ``docs/internal/design-sql-cells.md`` for the full design rationale
and the gotcha list (stats-collector lag, frozen-inside-transaction
semantics, replica counter divergence).
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

# Postgres unquoted identifier — letters, digits, underscores, with a
# non-digit first char. ``SET ROLE`` and ``SET search_path`` don't
# accept bind parameters, so any value we splice in must be validated
# against this pattern to avoid SQL injection.
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Probe queries use ``to_regclass($1)`` so unqualified table names
# resolve through the connection's actual ``search_path`` instead of
# being pinned to ``public``. The resolved ``nspname`` is selected
# back out and folded into the fingerprint, so an unqualified name
# resolving to different schemas across connections produces
# different tokens.
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

# Per-table column structure via ``pg_attribute``. Walking
# ``pg_attribute`` directly (rather than ``information_schema.columns``)
# lets us filter the ``to_regclass``-resolved OID and avoid a second
# round-trip just to translate names back to OIDs.
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

    Literal strings pass through unchanged. The writer scrubs literals
    before saving, so the only way a literal reaches this point is from
    in-memory state that hasn't been persisted yet — kept usable for
    test/dev scenarios where the user is iterating before save.
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
        # Test seam: pass a fake connect callable to bypass the real
        # ADBC import in unit tests. Production code uses the default
        # path and lets ``open()`` lazy-import the driver.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        # ``read_only`` is part of the Protocol so adapters that
        # route reads vs writes through different principals
        # (Snowflake's ``write_role``, BigQuery's
        # ``write_credentials_path``) can include only the
        # relevant fields. Postgres has no read/write principal
        # split — both are governed by the connection's role —
        # so the flag is a no-op here.
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

        # If a URI is set, parse out host/port/database/user from it.
        # `password` lives in the URI for some forms but is never
        # identity-shaping, so we drop it.
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

        # Discrete top-level keys override URI components and supply
        # values when no URI is set.
        for key in ("host", "port", "database", "user", "role"):
            value = getattr(spec, key, None)
            if value is not None:
                identity[key] = value

        # `auth.user` is identity-shaping (different user → different
        # object visibility). Resolve ${VAR} so two specs that resolve
        # to the same effective user produce the same connection_id.
        # `auth.password` is intentionally not included.
        auth = getattr(spec, "auth", None) or {}
        auth_user = auth.get("user")
        if auth_user:
            try:
                identity["user"] = _resolve_var(auth_user)
            except RuntimeError:
                # Env var not set yet — fall back to the raw spec value
                # so the identity is still stable across calls.
                identity["user"] = auth_user

        # `search_path` lives in options; it's identity-shaping
        # (changes which schema unqualified names resolve to).
        options = getattr(spec, "options", None) or {}
        sp = options.get("search_path")
        if sp:
            identity["search_path"] = sp

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC PostgreSQL connection.

        ``read_only=True`` flips the session into Postgres'
        ``default_transaction_read_only`` mode so any subsequent
        ``INSERT``/``UPDATE``/``DELETE`` is rejected by the engine
        before reaching disk. This is the security boundary; SQL-text
        keyword filtering is *not* relied on.

        After read-only mode, applies any identity-shaping session
        overrides — ``role`` (via ``SET ROLE``) and
        ``options.search_path`` (via ``SET search_path``). These are
        the same fields ``canonicalize_connection_id`` folds into the
        cache key; applying them here keeps cache identity and live
        session state consistent.
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
                # SET ROLE doesn't accept bind parameters, so we splice the
                # validated identifier in. Double-quoted to preserve case.
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

        Strategy:
        - If ``spec.uri`` is set, use it as the base. Splice
          ``auth.user``/``auth.password`` (resolved from env) into the
          userinfo portion if either is present.
        - Otherwise build from discrete ``host``/``port``/``database``
          components and resolved auth.

        ``${VAR}`` indirections in ``auth`` are resolved here, raising
        ``RuntimeError`` when an env var is missing.
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
        """Per-table freshness via DML counters + relfilenode + resolved schema.

        Uses ``to_regclass`` so unqualified table names resolve through
        the connection's actual ``search_path``. The resolved ``nspname``
        is folded into the digest, so an unqualified name pointing at
        different schemas across connections produces different tokens.

        Returns a token whose value digests every (qualified-input,
        resolved-schema, dml_count, relfilenode) tuple in sorted order.
        A name that ``to_regclass`` can't resolve flips the token to
        ``is_session_only=True``.
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
        """Per-table schema fingerprint via ``pg_attribute`` + ``to_regclass``.

        Catches metadata-only changes that the freshness probe would
        miss: ``ADD COLUMN``, type changes, nullability flips. Uses
        ``to_regclass`` for the same reason as ``probe_freshness`` —
        unqualified names resolve through the connection's actual
        ``search_path``, not a hardcoded ``public``.
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
        """Enumerate user-visible tables and views via ``information_schema``.

        Filters out system schemas (``pg_catalog``, ``information_schema``)
        because those clutter the discovery surface and the user's
        own queries almost never touch them. The query joins
        ``tables`` to ``columns`` so we make one round-trip
        regardless of how many tables the connection sees.

        Read-only by construction: the connection arrives in
        ``READ ONLY`` mode courtesy of ``open(read_only=True)``.
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

        # Group columns by (catalog, schema, name) preserving the
        # ordinal_position order from the query.
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
    """Build the string argument for ``to_regclass($1)``.

    Schema-qualified tables become ``"schema"."name"``; unqualified
    tables become ``"name"``. Double-quoting preserves identifier case
    (Postgres folds unquoted identifiers to lowercase). Embedded
    double quotes are escaped per the SQL standard.
    """
    parts = []
    if table.schema:
        parts.append(table.schema)
    parts.append(table.name)
    return ".".join(f'"{p.replace(chr(34), chr(34) * 2)}"' for p in parts)


def _format_search_path(value: Any) -> str:
    """Format a ``search_path`` option into a ``SET search_path TO ...`` clause.

    Accepts either a comma-separated string (``"analytics, public"``)
    or a list (``["analytics", "public"]``). Each schema name is
    validated against the unquoted-identifier pattern and double-
    quoted in the output. ``SET search_path`` doesn't accept bind
    parameters, so the validation is what prevents injection.
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

    Used when ``spec.auth`` carries credentials separately from the
    base URI. The host/port/database/path stay intact; only the
    authority's user[:password] portion changes. When ``user`` is
    None, the URI's existing user is preserved — common when the
    user is in the URI and only the password comes from
    ``${VAR}`` indirection.
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

    Exposed as a callable (not just a side effect of import) so
    ``_restore_defaults_for_tests`` can re-register after a
    ``_reset_for_tests`` without having to reload modules — Python's
    import cache makes module-level registration only run once.
    """
    register_adapter(_ADAPTER)


# Auto-register on first import.
register()

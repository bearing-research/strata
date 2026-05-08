"""Snowflake driver adapter.

Backed by ``adbc-driver-snowflake``. The freshness probe reads
``LAST_ALTERED`` from each touched database's per-DB
``INFORMATION_SCHEMA.TABLES`` view (Snowflake scopes
``INFORMATION_SCHEMA`` per database, so probing tables in
multiple databases means one query per database). The schema
fingerprint walks ``INFORMATION_SCHEMA.COLUMNS``.

Read-only enforcement is **role-based**, not session-flag-based.
Snowflake has no equivalent of Postgres's
``SET default_transaction_read_only = on`` — the security
boundary lives in the role's grants. Strata trusts the role
specified on the connection: a connection that should only be
used for read cells should reference a role with USAGE +
SELECT grants, no DML. Write cells should reference a role
that includes the necessary INSERT/UPDATE/DELETE grants.

See ``docs/internal/design-sql-cells.md`` for the full design
rationale and the gotcha list (cloud-services-credit cost,
``LAST_ALTERED`` updates on 0-row DML).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
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

_CAPABILITIES = AdapterCapabilities(
    per_table_freshness=True,
    # Time Travel exposes snapshot queries via SELECT … AT (TIMESTAMP),
    # but the per-table snapshot ID is not exposed as a stable token
    # the way Iceberg's snapshot_id is. Treat as equality-only.
    supports_snapshot=False,
    # Snowflake INFORMATION_SCHEMA isn't frozen inside a transaction
    # the way Postgres's pg_stat_* views are; the probe can share
    # the query connection.
    needs_separate_probe_conn=False,
)

# Snowflake unquoted identifier pattern. Used to validate role /
# warehouse / database names before splicing them into ``USE ROLE``
# statements (Snowflake doesn't accept bind parameters in those
# positions, so any value we splice must come from a known-safe
# pattern).
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _spec_attr(spec: Any, key: str) -> Any:
    """Read a top-level field off a ``ConnectionSpec`` safely.

    Pydantic v2's ``BaseModel`` reserves a few attribute names —
    ``schema`` is the most painful one for SQL drivers because
    ``BaseModel.schema()`` is a bound method that shadows
    user-provided extras. Reading via ``model_extra`` (where
    ``extra='allow'`` stashes unknown fields) and falling back to
    ``getattr`` correctly recovers user-supplied values without
    confusing them with Pydantic-internal methods.
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
    """Read ``CURRENT_DATABASE()`` and ``CURRENT_SCHEMA()``.

    Returns ``(database, schema)`` — either may be None if the
    role/warehouse has no defaults configured. Used by the
    freshness and schema probes to resolve unqualified table
    names against the same defaults the executor's query
    connection would use, instead of hardcoding ``PUBLIC`` and
    risking a probe that fingerprints the wrong schema.
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

    Returns a dict with whichever of ``account`` / ``user`` /
    ``database`` / ``schema`` / ``warehouse`` / ``role`` were
    present. Password is intentionally not extracted (secret).
    Empty strings are dropped so a bare ``snowflake://account/``
    doesn't introduce phantom keys.
    """
    out: dict[str, Any] = {}
    try:
        parsed = urlparse(uri)
    except Exception:  # noqa: BLE001
        return out

    if parsed.username:
        out["user"] = parsed.username

    # ``urlparse.hostname`` lowercases per RFC, but Snowflake
    # account identifiers can carry case-sensitive segments
    # (legacy account-locator format, region suffixes). Parse
    # ``netloc`` directly to preserve the original case.
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

    Literals pass through unchanged. The writer scrubs literal
    secrets before notebook.toml is saved, but in-memory specs
    can still carry literals between edit + save.
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
        # Test seam: pass a fake connect callable to bypass the real
        # ADBC import in unit tests. Production code uses the default
        # path and lets ``open()`` lazy-import the driver.
        self._connect_fn = connect_fn

    # --- identity ---------------------------------------------------------

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        """Hash identity-shaping fields, excluding secrets.

        Identity-shaping for Snowflake: account, user, role
        (when ``read_only=True``) or write_role (when
        ``read_only=False``), warehouse, default database,
        default schema.

        ``read_only`` controls which role joins the identity:
        read cells fold ``role`` only, so swapping ``write_role``
        on a connection used by both read and write cells doesn't
        churn read-cell caches. Write cells fold ``write_role``
        (or ``role`` as the fallback) since that's the role
        actually applied at open time.

        Excluded: password / private_key (secret).
        """
        return hash_connection_identity(
            self.name, self._extract_identity(spec, read_only=read_only)
        )

    def _extract_identity(self, spec: Any, *, read_only: bool = True) -> dict[str, Any]:
        identity: dict[str, Any] = {}

        # If a URI is set, its components are part of the
        # connection's identity — without this, two connections
        # whose only configured field is ``uri`` (different DBs,
        # different roles) would collapse onto the same
        # connection_id and share cache entries. Discrete fields
        # below override URI-derived components so the user can
        # build on a base URI with overrides.
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

        # ``write_role`` only joins write-cell identity. When
        # the cell is read-only, the role applied at open time
        # is just ``role``, so the cache key shouldn't depend
        # on the write_role.
        if not read_only:
            write_role = _spec_attr(spec, "write_role")
            if write_role is not None:
                identity["write_role"] = write_role

        # ``auth.user`` is identity-shaping (different user →
        # different object visibility). Resolve ${VAR} so two
        # specs that point at the same effective user produce the
        # same connection_id.
        auth = getattr(spec, "auth", None) or {}
        auth_user = auth.get("user")
        if auth_user:
            try:
                identity["user"] = _resolve_var(auth_user)
            except RuntimeError:
                # Env var not set yet — fall back to the raw spec
                # value so the identity stays stable across calls.
                identity["user"] = auth_user

        return identity

    # --- connection lifecycle --------------------------------------------

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC Snowflake connection.

        Read-only enforcement for Snowflake is **role-based**:
        Snowflake has no session-level read-only flag like
        Postgres's ``default_transaction_read_only``, so the
        security boundary lives in the role's grants. The adapter
        wires ``read_only`` through to which role gets applied:

        - ``read_only=True`` (read cells): apply the spec's
          ``role``. The user is responsible for picking a role
          whose grants are SELECT-only on the touched objects.
        - ``read_only=False`` (write cells, ``# @sql write=true``):
          apply ``write_role`` if the spec sets it; otherwise
          fall back to ``role``. ``write_role`` is the per-cell
          handle to a DML-capable role; without it, write cells
          inherit the same role as read cells (which the user's
          warehouse access policy then decides whether to allow).

        After role selection, applies the spec's warehouse,
        default database, and default schema (each via the
        corresponding ``USE …`` statement). All identifiers are
        validated against ``_IDENTIFIER_RE`` before splicing —
        Snowflake's ``USE`` statements don't accept bind
        parameters.
        """
        uri = self._build_uri(spec)
        conn = self._invoke_connect(uri)

        # Pick the role for this open() call based on the
        # cell's read/write intent.
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
                    # USE statements are autocommit on Snowflake;
                    # an explicit commit may surface "no
                    # transaction in progress." Ignore that
                    # specific class — applied_any is true so we
                    # don't lose any writes.
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
        """Construct the ADBC connection URI from the spec.

        Snowflake's gosnowflake URI shape:
        ``<user>:<password>@<account>/<database>/<schema>?warehouse=…&role=…``.

        Honors ``${VAR}`` indirection in ``auth.user`` /
        ``auth.password`` (env vars). Either an explicit
        ``spec.uri`` or the discrete fields can be the source —
        explicit URI wins when both are set.
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

        Snowflake scopes ``INFORMATION_SCHEMA`` per-database, so
        tables grouped by their catalog (database) get one
        round-trip per database. Tables without a catalog fall
        back to the connection's current database via
        ``CURRENT_DATABASE()``.

        ``LAST_ALTERED`` updates on any DML touching the table,
        even a 0-row update — this is the safe direction
        (potentially over-invalidating, never under-).

        Tables not found in any of the queried databases
        contribute a sentinel "missing" entry so two probes
        agree on "table doesn't exist" but disagree from a
        successfully-found table.
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
                        # Neither the table nor the session has
                        # a schema. Fold a sentinel rather than
                        # silently picking PUBLIC — the cache key
                        # must reflect the unresolved name.
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

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Per-table schema fingerprint via ``INFORMATION_SCHEMA.COLUMNS``.

        Same per-database scoping as ``probe_freshness``. Catches
        ADD COLUMN / type changes / nullability flips that
        ``LAST_ALTERED`` would also catch — the schema fingerprint
        is finer-grained but redundant most of the time.
        ``LAST_ALTERED`` does cover schema changes, so this is
        belt-and-suspenders for the rare case where a metadata-
        only event might not bump it.
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
        """Enumerate tables and views in the connection's database.

        Scopes to the connection's default database — schema
        discovery across multiple databases would mean one
        ``INFORMATION_SCHEMA`` query per database, which is
        cloud-services-credit-billed; we keep the surface tight
        for v1.
        """
        with conn.cursor() as cursor:
            cursor.execute("SELECT CURRENT_DATABASE()")
            row = cursor.fetchone()
            if not row or not row[0]:
                return []
            db = str(row[0])

            if not _IDENTIFIER_RE.match(db):
                # Pathological, but defend: ADBC drivers should
                # quote their own identifiers safely. Skip
                # enumeration rather than splice an unsafe value.
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
    """Idempotent registration entry point.

    Module imports are cached, so import side effects don't fire
    twice — but the registry's caller (drivers/__init__.py) calls
    each driver's ``register()`` explicitly to make the wiring
    visible. Mirrors the pattern in postgresql.py / sqlite.py.
    """
    register_adapter(_ADAPTER)


register()

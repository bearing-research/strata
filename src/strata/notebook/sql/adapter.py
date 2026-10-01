"""DriverAdapter protocol and supporting types for SQL cells.

Strata-shaped, not a generic SQL abstraction: each method exists because the
executor or cache layer needs it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AdapterCapabilities:
    """Capability flags for a ``DriverAdapter``.

    The cache policy resolver reads these to decide whether ``# @cache snapshot``
    can be honored, whether ``fingerprint`` yields a real per-table token or falls
    back to session scope, and whether a separate probe connection is needed.
    """

    per_table_freshness: bool
    """True when ``probe_freshness`` returns a per-table token; False
    when only a database-wide token is available (SQLite) or no token
    at all (DuckDB native)."""

    supports_snapshot: bool
    """True when ``probe_freshness`` can return a durable snapshot
    identity (Iceberg ``snapshot_id``, BigQuery time-travel target)."""

    needs_separate_probe_conn: bool
    """True when the freshness probe must run on a connection separate
    from the query connection. Postgres requires this because
    ``pg_stat_*`` views are frozen inside an open transaction."""


@dataclass(frozen=True)
class QualifiedTable:
    """Fully qualified table reference.

    ``catalog`` and ``schema`` are None for backends without those layers.
    """

    catalog: str | None
    schema: str | None
    name: str

    def render(self) -> str:
        """Render as a dotted string for diagnostics and probe queries."""
        parts = [p for p in (self.catalog, self.schema, self.name) if p]
        return ".".join(parts)


@dataclass(frozen=True)
class FreshnessToken:
    """Opaque equality token for the database state of touched tables.

    Equal tokens mean the touched tables give the same result; the cache layer
    only compares the bytes. ``is_session_only`` means no real fingerprint was
    available (e.g. DuckDB native) and a session-unique salt was used, so reuse
    is session-scoped. ``is_snapshot`` means a durable, queryable snapshot ID
    (e.g. Iceberg ``snapshot_id``), which ``# @cache snapshot`` requires.
    """

    value: bytes
    is_session_only: bool = False
    is_snapshot: bool = False


@dataclass(frozen=True)
class ColumnInfo:
    """One column of a table, as the driver reports it.

    ``type`` is the driver's own SQL-text type label, not normalized across drivers.
    """

    name: str
    type: str
    nullable: bool | None = None


@dataclass(frozen=True)
class TableSchema:
    """A table's identity plus its columns, for schema discovery."""

    catalog: str | None
    schema: str | None
    name: str
    columns: tuple[ColumnInfo, ...] = ()

    def render(self) -> str:
        parts = [p for p in (self.catalog, self.schema, self.name) if p]
        return ".".join(parts)


@dataclass(frozen=True)
class SchemaFingerprint:
    """Opaque equality token for the column structure of touched tables.

    Catches schema changes the freshness token misses (metadata-only ADD COLUMN,
    type changes, renames); folded into the provenance hash.
    """

    value: bytes


class DriverAdapter(Protocol):
    """Per-driver glue between Strata's SQL pipeline and ADBC.

    Implementations in ``strata.notebook.sql.drivers.*`` call ``register_adapter``
    at import time; the executor finds them with ``get_adapter(driver)``.
    """

    name: str
    """Driver identifier, matched against ``ConnectionSpec.driver``."""

    sqlglot_dialect: str
    """Dialect name passed to ``sqlglot.parse(..., dialect=...)``."""

    capabilities: AdapterCapabilities

    def canonicalize_connection_id(self, spec: Any, *, read_only: bool = True) -> str:
        """Return a stable hash of the connection's identity-shaping config.

        Includes everything that changes object visibility (host, port,
        database, role, warehouse, search path) and excludes secrets and
        tunables like ``connect_timeout``, so equal ids see the same objects as
        the same principal and caches never bleed between differing ones.

        ``read_only`` selects the principal: adapters with a separate write
        principal (Snowflake ``write_role``) include it only when ``False``, so
        changing it does not invalidate read-cell caches.
        """
        ...

    def open(self, spec: Any, *, read_only: bool) -> Any:
        """Open an ADBC connection; the handle is opaque to the executor.

        ``read_only=True`` requires an enforceable read-only mode (Postgres
        ``READ ONLY`` transaction, SQLite immutable). An adapter that cannot
        provide one raises ``RuntimeError`` rather than fall back to keyword-based
        DML rejection.
        """
        ...

    def probe_freshness(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> FreshnessToken:
        """Return a freshness token for the given tables.

        Called once per cell execution for the cache key, so it must be cheap
        (one metadata round-trip per database).
        """
        ...

    def probe_schema(
        self,
        probe_conn: Any,
        tables: list[QualifiedTable],
    ) -> SchemaFingerprint:
        """Return a schema fingerprint for the given tables (one metadata read each)."""
        ...

    def list_schema(self, conn: Any) -> list[TableSchema]:
        """Enumerate the tables and columns visible on this connection.

        Side-effect free; called on a read-only connection. Raises when the
        driver cannot enumerate, and the route surfaces the error verbatim.
        """
        ...


def hash_connection_identity(
    driver: str,
    identity: dict[str, Any],
) -> str:
    """Hash ``{"driver": <driver>, **identity}`` as sorted JSON.

    The adapter chooses the identity-shaping keys and must exclude secrets.
    """
    payload = {"driver": driver, **identity}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()

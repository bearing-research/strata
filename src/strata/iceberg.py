"""Iceberg snapshot resolution using pyiceberg."""

import contextlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

import pyarrow as pa
from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError, ValidationError
from pyiceberg.schema import Schema
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation, Snapshot

from strata.config import StrataConfig
from strata.types import ACL_STORE_NAMES, TableIdentity

logger = logging.getLogger(__name__)


class CatalogProvider(Protocol):
    """Catalog-provider interface, to allow alternative backends."""

    def load_table(self, table_uri: str) -> Table:
        """Load the Iceberg table named by ``table_uri``."""
        ...

    def get_snapshot_id(self, table: Table, snapshot_id: int | None) -> int:
        """Resolve the snapshot id to read (the current snapshot if ``None``)."""
        ...


_NAMED = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):(?!//)(.+)$")


def named_catalog(table_uri: str, config: StrataConfig) -> tuple[str | None, str]:
    """Split ``<name>:<namespace>.<table>`` into the catalog name and table id.

    Returns ``(None, table_uri)`` unless ``<name>`` is a configured catalog and the
    URI has no ``#``, so a Windows path or a scheme is never taken for a catalog.
    """
    if "#" in table_uri:
        return None, table_uri
    match = _NAMED.match(table_uri)
    if match is None or match.group(1) not in (getattr(config, "catalogs", None) or {}):
        return None, table_uri
    return match.group(1), match.group(2)


def table_identity_for(table_uri: str, config: StrataConfig) -> TableIdentity:
    """Return the canonical identity of *table_uri*, a named catalog's table included.

    A table in a configured catalog belongs to that catalog; one in a warehouse
    the URI carries belongs to ``strata``, matching the cache keys.
    """
    named, table_id = named_catalog(table_uri, config)
    if named is not None:
        return TableIdentity.from_table_id(table_id, catalog=named)
    warehouse_path, table_id = PyIcebergCatalog.parse_table_uri(table_uri)
    catalog = config.catalog_name if warehouse_path is None else "strata"
    return TableIdentity.from_table_id(table_id, catalog=catalog)


def shared_catalog_stores(table_uri: str, config: StrataConfig) -> tuple[str, ...]:
    """Return every store name an ACL rule can give the table *table_uri* reads.

    With ``catalog_properties["uri"]`` set, every warehouse URI (``s3:``, ``gs:``,
    ``az:``, ``file:``) builds ``SqlCatalog("strata")`` over one database, so all
    name the same table; a bare ``namespace.table`` joins them when the default
    catalog is also ``strata``. ``()`` means the table has only its requested name.
    """
    if "uri" not in config.catalog_properties:
        return ()
    if named_catalog(table_uri, config)[0] is not None:
        return ()
    warehouse_path, _ = PyIcebergCatalog.parse_table_uri(table_uri)
    if warehouse_path is None and config.catalog_name != "strata":
        return ()
    return ACL_STORE_NAMES


def _is_connection_io_error(exc: BaseException) -> bool:
    """Return whether *exc* is a dead catalog connection (SQLITE_IOERR or its ADBC fallout).

    Nothing else matches, so a missing table or malformed URI still surfaces on the first attempt.
    """
    text = str(exc).lower()
    return "disk i/o error" in text or "sqlite_ioerr" in text or "adbcstatement" in text


class PyIcebergCatalog:
    """Default catalog provider backed by pyiceberg.

    Catalogs are built lazily per warehouse and cached under a lock, so concurrent
    planning threads do not build duplicates.
    """

    def __init__(self, config: StrataConfig) -> None:
        """Initialize the provider from server config (catalog properties, S3 credentials)."""
        self.config = config
        self._catalogs: dict[str, Catalog] = {}
        self._lock = Lock()

    def _get_default_catalog_uri(self, warehouse_path: str | None = None) -> str:
        """Return the catalog URI for a warehouse.

        A configured ``catalog_properties["uri"]`` (e.g. PostgreSQL) wins; otherwise
        SQLite keyed off the warehouse path, or in-memory for ``None``.
        """
        # A configured URI may name PostgreSQL, MySQL, etc.
        if "uri" in self.config.catalog_properties:
            return self.config.catalog_properties["uri"]

        if warehouse_path and warehouse_path.startswith("s3://"):
            return f"sqlite:///{self.config.metadata_db}"
        elif warehouse_path:
            return f"sqlite:///{Path(warehouse_path) / 'catalog.db'}"
        else:
            return "sqlite:///:memory:"

    def _s3_catalog_props(self) -> dict[str, str]:
        """Return the ``s3.*`` catalog properties for the credentials that are configured."""
        props: dict[str, str] = {}
        if self.config.s3_region:
            props["s3.region"] = self.config.s3_region
        if self.config.s3_access_key:
            props["s3.access-key-id"] = self.config.s3_access_key
        if self.config.s3_secret_key:
            props["s3.secret-access-key"] = self.config.s3_secret_key
        if self.config.s3_endpoint_url:
            props["s3.endpoint"] = self.config.s3_endpoint_url
        return props

    def _build_catalog(self, warehouse_path: str | None) -> Catalog:
        """Construct a catalog for a warehouse, uncached.

        A warehouse path yields a ``SqlCatalog`` (with ``s3.*`` props for ``s3://``);
        ``None`` yields the configured default catalog, else in-memory SQLite.
        """
        if warehouse_path:
            props: dict = {
                "uri": self._get_default_catalog_uri(warehouse_path),
                "warehouse": warehouse_path,
            }
            if warehouse_path.startswith("s3://"):
                props.update(self._s3_catalog_props())
            props.update(self.config.catalog_properties)
            return SqlCatalog("strata", **props)

        if self.config.catalog_properties:
            return load_catalog(self.config.catalog_name, **self.config.catalog_properties)
        return SqlCatalog(
            self.config.catalog_name,
            uri="sqlite:///:memory:",
            warehouse=str(self.config.cache_dir / "warehouse"),
        )

    def _get_named_catalog(self, name: str) -> Catalog:
        """Return the configured catalog *name*, built on first use and cached."""
        key = f"catalog:{name}"
        catalog = self._catalogs.get(key)
        if catalog is not None:
            return catalog
        with self._lock:
            catalog = self._catalogs.get(key)
            if catalog is None:
                from strata.notebook.credentials import (
                    CredentialResolver,
                    resolve_catalog_properties,
                )

                properties = resolve_catalog_properties(
                    self.config.catalogs[name], CredentialResolver.from_config(self.config)
                )
                catalog = load_catalog(name, **properties)
                self._catalogs[key] = catalog
        return catalog

    def _get_catalog(self, warehouse_path: str | None = None) -> Catalog:
        """Return the cached catalog for a warehouse, building it once under a lock on a miss."""
        cache_key = warehouse_path or "default"

        cached = self._catalogs.get(cache_key)
        if cached is not None:
            return cached

        with self._lock:
            cached = self._catalogs.get(cache_key)
            if cached is not None:
                return cached
            catalog = self._build_catalog(warehouse_path)
            self._catalogs[cache_key] = catalog
            return catalog

    def _invalidate_catalog(self, warehouse_path: str | None) -> None:
        """Drop a cached catalog so the next access rebuilds its connection."""
        with self._lock:
            self._catalogs.pop(warehouse_path or "default", None)

    @staticmethod
    def parse_table_uri(table_uri: str) -> tuple[str | None, str]:
        """Split a table URI into ``(warehouse_path, table_id)``.

        Accepts ``file:///wh#ns.table``, ``/wh#ns.table``, ``s3://bucket/wh#ns.table``,
        or ``ns.table`` (default catalog; warehouse ``None``). ``s3://`` is kept and
        ``file://`` stripped. Named-catalog URIs are resolved by :func:`named_catalog` first.
        """
        if "#" in table_uri:
            path_part, table_id = table_uri.rsplit("#", 1)
            if path_part.startswith("s3://"):
                warehouse_path = path_part
            else:
                warehouse_path = path_part.replace("file://", "")
            return warehouse_path, table_id
        else:
            return None, table_uri

    def load_table(self, table_uri: str) -> Table:
        """Load an Iceberg table from any URI form :meth:`parse_table_uri` accepts."""
        name, named_table_id = named_catalog(table_uri, self.config)
        if name is not None:
            return self._get_named_catalog(name).load_table(named_table_id)
        warehouse_path, table_id = self.parse_table_uri(table_uri)
        catalog = self._get_catalog(warehouse_path)
        try:
            return catalog.load_table(table_id)
        except Exception as exc:
            # A catalog whose connection has gone bad (e.g. SqlCatalog on SQLite returning
            # SQLITE_IOERR) stays cached, and retrying through it can never succeed. Drop the
            # poisoned entry and rebuild once.
            #
            # Only connection-level I/O failures retry, so a missing table or bad URI still raises
            # immediately.
            if not _is_connection_io_error(exc):
                raise
            logger.warning(
                "Iceberg catalog connection for %s failed with %s; rebuilding",
                warehouse_path or "default",
                exc,
            )
            self._invalidate_catalog(warehouse_path)
            return self._get_catalog(warehouse_path).load_table(table_id)

    def get_snapshot_id(self, table: Table, snapshot_id: int | None) -> int:
        """Resolve the snapshot id to read (the current one when ``snapshot_id`` is ``None``).

        Raises
        ------
        ValueError
            If ``snapshot_id`` is absent from the table, or the table has no snapshots.
        """
        if snapshot_id is not None:
            snapshot = table.snapshot_by_id(snapshot_id)
            if snapshot is None:
                raise ValueError(f"Snapshot {snapshot_id} not found in table")
            return snapshot_id

        current = table.current_snapshot()
        if current is None:
            raise ValueError("Table has no snapshots")
        return current.snapshot_id

    def create_table_if_not_exists(
        self,
        warehouse_path: str,
        namespace: str,
        table_name: str,
        schema: Schema,
    ) -> Table:
        """Load a table, creating it and its namespace if absent (for demos and tests)."""
        catalog = self._get_catalog(warehouse_path)

        try:
            catalog.create_namespace(namespace)
        except NamespaceAlreadyExistsError:
            pass  # idempotent: the namespace already exists

        table_id = f"{namespace}.{table_name}"
        try:
            return catalog.load_table(table_id)
        except NoSuchTableError:
            return catalog.create_table(table_id, schema)


# Snapshot summary properties naming the artifact a snapshot was written from.
SUMMARY_ARTIFACT_ID = "strata.artifact_id"
SUMMARY_VERSION = "strata.version"
SUMMARY_PROVENANCE = "strata.provenance_hash"
SUMMARY_PROMOTED_BY = "strata.promoted_by"
SUMMARY_TENANT = "strata.tenant"


class TableOfAnotherTenant(PermissionError):
    """The table's latest export was written for another tenant."""


@dataclass(frozen=True)
class TableWrite:
    """What writing an artifact into a table produced."""

    table: str
    snapshot_id: int
    created: bool


def _last_strata_write(table: Table) -> Snapshot | None:
    """The latest snapshot of *table* an export wrote, or None if Strata never wrote it.

    Later exports overwrite, so a table Strata never wrote (e.g. a mistyped
    production name) must not be treated as Strata's. Uses the marker ``tag`` recognises.
    """
    written = [
        snapshot
        for snapshot in table.snapshots()
        if snapshot.summary is not None and snapshot.summary.get(SUMMARY_ARTIFACT_ID)
    ]
    if not written:
        return None
    return max(written, key=lambda s: s.timestamp_ms)


class IcebergWriter:
    """Writes artifacts into Iceberg tables as snapshots that name them.

    The first write appends and later ones overwrite, so the current snapshot is
    one artifact version and history is the sequence written. Adding columns and
    widening types evolve the schema; other changes are refused before writing.
    Snapshot summaries carry artifact id, version, provenance hash, writer and
    tenant; an alias is an Iceberg tag on its version's snapshot.
    """

    def __init__(self, catalogs: PyIcebergCatalog) -> None:
        self._catalogs = catalogs

    def _table_id(self, table_uri: str) -> tuple[Catalog, str]:
        named, named_table_id = named_catalog(table_uri, self._catalogs.config)
        if named is not None:
            # A configured catalog holds the table; without this the whole
            # "<name>:<namespace>.<table>" string would be read as a table id
            # in the default catalog, and the write would land elsewhere.
            if "." not in named_table_id:
                raise ValueError(f"{table_uri!r} names no namespace; expected <namespace>.<table>")
            return self._catalogs._get_named_catalog(named), named_table_id
        warehouse_path, table_id = self._catalogs.parse_table_uri(table_uri)
        if "." not in table_id:
            raise ValueError(f"{table_uri!r} names no namespace; expected <namespace>.<table>")
        if warehouse_path and "://" not in warehouse_path:
            # A local warehouse is where its SQLite catalog lives, so the first
            # write to it has to be able to create it.
            Path(warehouse_path).mkdir(parents=True, exist_ok=True)
        return self._catalogs._get_catalog(warehouse_path), table_id

    def write(
        self,
        table_uri: str,
        data: pa.Table,
        *,
        artifact_id: str,
        version: int,
        provenance_hash: str,
        promoted_by: str | None,
        alias: str | None = None,
        tenant: str | None = None,
    ) -> TableWrite:
        """Write *data* as the table's current snapshot.

        ``tenant`` is the tenant the write is scoped to (None: unscoped, as in personal
        mode or for an admin). A scoped write may only replace a table whose latest
        export was written for the same tenant.

        Raises:
            TableOfAnotherTenant: The table's latest export belongs to another tenant.
            ValueError: Strata did not write the table, or the schema cannot evolve.
        """
        catalog, table_id = self._table_id(table_uri)
        properties = {
            SUMMARY_ARTIFACT_ID: artifact_id,
            SUMMARY_VERSION: str(version),
            SUMMARY_PROVENANCE: provenance_hash,
        }
        if promoted_by:
            properties[SUMMARY_PROMOTED_BY] = promoted_by
        if tenant is not None:
            properties[SUMMARY_TENANT] = tenant
        # Schema metadata is the writer's (pandas index layout, Strata's shape
        # tags) and means nothing to a table.
        data = data.replace_schema_metadata(None)

        created = False
        try:
            table = catalog.load_table(table_id)
        except NoSuchTableError:
            with contextlib.suppress(NamespaceAlreadyExistsError):
                catalog.create_namespace(table_id.rsplit(".", 1)[0])
            table = catalog.create_table(table_id, schema=data.schema)
            created = True

        if table.current_snapshot() is None:
            table.append(data, snapshot_properties=properties)
        else:
            last_write = _last_strata_write(table)
            if last_write is None:
                raise ValueError(
                    f"{table_uri} holds data Strata did not write, and a later "
                    f"write replaces the table's contents -- refusing to "
                    f"overwrite it with {artifact_id}@v={version}. Export to a "
                    f"table of its own, or append to this one outside Strata."
                )
            assert last_write.summary is not None
            # The table ACL has no write verb, so the tenant that last exported here owns
            # the table; without this any tenant allowed to read it could replace it.
            if tenant is not None and last_write.summary.get(SUMMARY_TENANT) != tenant:
                raise TableOfAnotherTenant(f"{table_uri} was written for another tenant")
            try:
                with table.update_schema() as update:
                    update.union_by_name(data.schema)
            except ValidationError as exc:
                raise ValueError(
                    f"{artifact_id}@v={version} cannot be written to {table_uri}: "
                    f"its schema is not compatible with the table's ({exc})"
                ) from exc
            table.overwrite(data, snapshot_properties=properties)

        snapshot = table.current_snapshot()
        assert snapshot is not None
        if alias:
            table.manage_snapshots().create_tag(snapshot.snapshot_id, alias).commit()
        return TableWrite(table=table_uri, snapshot_id=snapshot.snapshot_id, created=created)

    def tag(self, table_uri: str, alias: str, *, artifact_id: str, version: int) -> int | None:
        """Point tag *alias* at the latest snapshot written from ``artifact_id@v=version``.

        Returns the snapshot id, or None if the table has none from that version.
        """
        catalog, table_id = self._table_id(table_uri)
        table = catalog.load_table(table_id)
        # An overwrite commits a delete and then an append, both carrying the
        # properties; the append is the snapshot that holds the data.
        written = [
            snapshot
            for snapshot in table.snapshots()
            if snapshot.summary is not None
            and snapshot.summary.operation == Operation.APPEND
            and snapshot.summary.get(SUMMARY_ARTIFACT_ID) == artifact_id
            and snapshot.summary.get(SUMMARY_VERSION) == str(version)
        ]
        if not written:
            return None
        snapshot_id = max(written, key=lambda s: s.timestamp_ms).snapshot_id
        table.manage_snapshots().create_tag(snapshot_id, alias).commit()
        return snapshot_id

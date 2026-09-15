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
from pyiceberg.table.snapshots import Operation

from strata.config import StrataConfig

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

    ``(None, table_uri)`` for anything else: a URI with a ``#`` names a
    warehouse, and a name that is not a configured catalog is not one, so a
    Windows path or a scheme is never taken for a catalog.
    """
    if "#" in table_uri:
        return None, table_uri
    match = _NAMED.match(table_uri)
    if match is None or match.group(1) not in (getattr(config, "catalogs", None) or {}):
        return None, table_uri
    return match.group(1), match.group(2)


def _is_connection_io_error(exc: BaseException) -> bool:
    """Whether *exc* looks like a dead catalog connection rather than a real error.

    Matches the I/O-level failures that leave the connection unusable
    (SQLITE_IOERR and the ADBC cursor-finalizer fallout seen alongside it), and
    deliberately nothing else — a missing table or a malformed URI must still
    surface on the first attempt.
    """
    text = str(exc).lower()
    return "disk i/o error" in text or "sqlite_ioerr" in text or "adbcstatement" in text


class PyIcebergCatalog:
    """Default catalog provider backed by pyiceberg.

    Catalogs are created lazily per warehouse and cached. The cache is guarded
    by a lock so concurrent planning threads don't each build a duplicate
    catalog for the same warehouse.
    """

    def __init__(self, config: StrataConfig) -> None:
        """Initialize the provider.

        Parameters
        ----------
        config : StrataConfig
            Server configuration supplying catalog properties and S3 credentials.
        """
        self.config = config
        self._catalogs: dict[str, Catalog] = {}
        self._lock = Lock()

    def _get_default_catalog_uri(self, warehouse_path: str | None = None) -> str:
        """Return the catalog URI for a warehouse.

        A configured ``catalog_properties["uri"]`` (e.g. PostgreSQL) wins;
        otherwise fall back to SQLite keyed off the warehouse path.

        Parameters
        ----------
        warehouse_path : str or None, optional
            Warehouse location, or ``None`` for the in-memory default.

        Returns
        -------
        str
            A catalog connection URI.
        """
        # Use configured URI if provided (supports PostgreSQL, MySQL, etc.)
        if "uri" in self.config.catalog_properties:
            return self.config.catalog_properties["uri"]

        # Fall back to SQLite based on warehouse path
        if warehouse_path and warehouse_path.startswith("s3://"):
            return f"sqlite:///{self.config.metadata_db}"
        elif warehouse_path:
            return f"sqlite:///{Path(warehouse_path) / 'catalog.db'}"
        else:
            return "sqlite:///:memory:"

    def _s3_catalog_props(self) -> dict[str, str]:
        """Return the ``s3.*`` catalog properties from configured credentials.

        Returns
        -------
        dict
            Only the keys whose corresponding config value is set.
        """
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
        """Construct a catalog for a warehouse (no caching).

        A warehouse path yields a ``SqlCatalog`` over that warehouse (with
        ``s3.*`` props folded in for ``s3://`` paths); ``None`` yields the
        configured default catalog, or an in-memory SQLite catalog as a
        fallback.

        Parameters
        ----------
        warehouse_path : str or None
            Warehouse location, or ``None`` for the default catalog.

        Returns
        -------
        pyiceberg.catalog.Catalog
            The constructed catalog.
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
        """The configured catalog *name*, built on first use and cached."""
        key = f"catalog:{name}"
        catalog = self._catalogs.get(key)
        if catalog is not None:
            return catalog
        with self._lock:
            catalog = self._catalogs.get(key)
            if catalog is None:
                catalog = load_catalog(name, **self.config.catalogs[name])
                self._catalogs[key] = catalog
        return catalog

    def _get_catalog(self, warehouse_path: str | None = None) -> Catalog:
        """Return the cached catalog for a warehouse, building it on first use.

        Uses double-checked locking: the common path is a lock-free cache hit;
        only a miss takes the lock to build (and re-checks under it so two
        threads can't build the same catalog twice).

        Parameters
        ----------
        warehouse_path : str or None, optional
            Warehouse location, or ``None`` for the default catalog.

        Returns
        -------
        pyiceberg.catalog.Catalog
            The cached catalog.
        """
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

        Parameters
        ----------
        table_uri : str
            One of:

            - ``file:///path/to/warehouse#namespace.table``
            - ``/path/to/warehouse#namespace.table``
            - ``s3://bucket/path/to/warehouse#namespace.table``
            - ``namespace.table`` (default catalog)

            ``<name>:namespace.table``, a configured named catalog, is
            resolved by :func:`named_catalog` before this is consulted.

        Returns
        -------
        tuple of (str or None, str)
            The warehouse path (``None`` when the URI carries no ``#`` part)
            and the ``namespace.table`` id. ``s3://`` is preserved; ``file://``
            is stripped.
        """
        if "#" in table_uri:
            path_part, table_id = table_uri.rsplit("#", 1)
            # Preserve s3:// prefix, strip file:// prefix
            if path_part.startswith("s3://"):
                warehouse_path = path_part
            else:
                warehouse_path = path_part.replace("file://", "")
            return warehouse_path, table_id
        else:
            return None, table_uri

    def load_table(self, table_uri: str) -> Table:
        """Load an Iceberg table from a URI.

        Parameters
        ----------
        table_uri : str
            A table URI in any form accepted by :meth:`parse_table_uri`
            (``file://`` / local / ``s3://`` warehouse, or bare
            ``namespace.table`` for the default catalog).

        Returns
        -------
        pyiceberg.table.Table
            The loaded table.
        """
        name, named_table_id = named_catalog(table_uri, self.config)
        if name is not None:
            return self._get_named_catalog(name).load_table(named_table_id)
        warehouse_path, table_id = self.parse_table_uri(table_uri)
        catalog = self._get_catalog(warehouse_path)
        try:
            return catalog.load_table(table_id)
        except Exception as exc:
            # A catalog whose backing connection has gone bad stays cached, so
            # every later read of that warehouse fails the same way until the
            # process restarts. The observed case is a SqlCatalog on SQLite
            # returning "disk I/O error" (SQLITE_IOERR): the connection is done,
            # but the object lives on in ``_catalogs`` and retrying through it
            # can never succeed — which is why a bounded retry loop at the call
            # site did not help.
            #
            # Drop the poisoned entry and rebuild once. Narrow on purpose: only
            # connection-level I/O failures are retried, so a genuinely missing
            # table or a bad URI still raises immediately.
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
        """Resolve the snapshot id to read.

        Parameters
        ----------
        table : pyiceberg.table.Table
            The table to read.
        snapshot_id : int or None
            A specific snapshot id, or ``None`` for the current snapshot.

        Returns
        -------
        int
            The resolved snapshot id.

        Raises
        ------
        ValueError
            If ``snapshot_id`` is given but absent from the table, or the table
            has no snapshots.
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
        """Load a table, creating it (and its namespace) if absent.

        Intended for demos and tests.

        Parameters
        ----------
        warehouse_path : str
            Warehouse to create the table in.
        namespace : str
            Namespace for the table.
        table_name : str
            Table name within the namespace.
        schema : pyiceberg.schema.Schema
            Schema used when the table must be created.

        Returns
        -------
        pyiceberg.table.Table
            The existing or newly created table.
        """
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


@dataclass(frozen=True)
class TableWrite:
    """What writing an artifact into a table produced."""

    table: str
    snapshot_id: int
    created: bool


class IcebergWriter:
    """Writes artifacts into Iceberg tables as snapshots that name them.

    The first write to a table appends; a later one overwrites, so the table's
    current snapshot is always one artifact version and its history is the
    sequence of versions written. A new version may add columns or widen a
    type; a change Iceberg cannot evolve to is refused before anything is
    written. Each snapshot's summary carries the artifact id, version,
    provenance hash and who wrote it, and an alias is an Iceberg tag on the
    snapshot of the version it names.
    """

    def __init__(self, catalogs: PyIcebergCatalog) -> None:
        self._catalogs = catalogs

    def _table_id(self, table_uri: str) -> tuple[Catalog, str]:
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
    ) -> TableWrite:
        catalog, table_id = self._table_id(table_uri)
        properties = {
            SUMMARY_ARTIFACT_ID: artifact_id,
            SUMMARY_VERSION: str(version),
            SUMMARY_PROVENANCE: provenance_hash,
        }
        if promoted_by:
            properties[SUMMARY_PROMOTED_BY] = promoted_by
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
        """Point tag *alias* at the latest snapshot written from
        ``artifact_id@v=version``; its id, or None if the table has none."""
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

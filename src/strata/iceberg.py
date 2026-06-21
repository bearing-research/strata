"""Iceberg snapshot resolution using pyiceberg."""

from pathlib import Path
from threading import Lock
from typing import Protocol

from pyiceberg.catalog import Catalog, load_catalog
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError, NoSuchTableError
from pyiceberg.schema import Schema
from pyiceberg.table import Table

from strata.config import StrataConfig


class CatalogProvider(Protocol):
    """Catalog-provider interface, to allow alternative backends."""

    def load_table(self, table_uri: str) -> Table:
        """Load the Iceberg table named by ``table_uri``."""
        ...

    def get_snapshot_id(self, table: Table, snapshot_id: int | None) -> int:
        """Resolve the snapshot id to read (the current snapshot if ``None``)."""
        ...


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
        warehouse_path, table_id = self.parse_table_uri(table_uri)
        catalog = self._get_catalog(warehouse_path)
        return catalog.load_table(table_id)

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

"""Apache DataFusion helpers over Strata scans.

DataFusion predicates run *after* the data is fetched; pass Strata filters to
the registration functions for server-side row-group pruning.

Example:
    from strata_client.integration.datafusion import register_strata_table

    ctx = register_strata_table("events", "file:///warehouse#db.events")
    result = ctx.sql("SELECT * FROM events WHERE value > 100").collect()
"""

from typing import TYPE_CHECKING

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value

if TYPE_CHECKING:
    import datafusion
    import pyarrow as pa


def _build_scan_transform(
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    snapshot_id: int | None = None,
) -> dict:
    """Build a scan@v1 transform specification.

    ``snapshot_id`` must be passed through, or a pinned scan silently reads
    (and records provenance for) the current snapshot.
    """
    params: dict = {}
    if columns is not None:
        params["columns"] = columns
    if filters is not None:
        params["filters"] = [
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
        ]
    if snapshot_id is not None:
        params["snapshot_id"] = snapshot_id
    return {"executor": "scan@v1", "params": params}


def register_strata_table(
    name: str,
    table_uri: str,
    ctx: "datafusion.SessionContext | None" = None,
    snapshot_id: int | None = None,
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> "datafusion.SessionContext":
    """Fetch a Strata table now and register it in a DataFusion context.

    Args:
        name: Table name in DataFusion's catalog.
        table_uri: Iceberg table URI, e.g. "file:///warehouse#db.table".
        ctx: Existing SessionContext (a new one if None).
        snapshot_id: Snapshot to read (None for latest).
        columns: Columns to project (None for all).
        filters: Filters for Strata-side row-group pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        SessionContext with the table registered.

    Example:
        ctx = register_strata_table(
            "events", "file:///warehouse#db.events", filters=[gt("value", 100)]
        )
        result = ctx.sql("SELECT id, value FROM events WHERE id < 10").collect()
    """
    import datafusion

    if ctx is None:
        ctx = datafusion.SessionContext()

    client = StrataClient(config=config, base_url=base_url)
    try:
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = artifact.to_table()

        ctx.register_record_batches(name, [arrow_table.to_batches()])

        return ctx
    finally:
        client.close()


def strata_query(
    sql: str,
    tables: dict[str, str],
    snapshot_id: int | None = None,
    columns: dict[str, list[str]] | None = None,
    filters: dict[str, list[Filter]] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> list["pa.RecordBatch"]:
    """Register several Strata tables in a fresh DataFusion context and run one SQL query.

    Args:
        sql: SQL query to execute.
        tables: Table name to Strata table URI.
        snapshot_id: Snapshot ID for all tables (None for latest).
        columns: Per-table column projections.
        filters: Per-table Strata filters for row-group pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        Arrow RecordBatches with the query result.

    Example:
        result = strata_query(
            "SELECT e.id, u.name FROM events e JOIN users u ON e.user_id = u.id",
            tables={
                "events": "file:///warehouse#db.events",
                "users": "file:///warehouse#db.users",
            },
            filters={"events": [gt("timestamp", 1700000000)]},
        )
    """
    import datafusion

    ctx = datafusion.SessionContext()
    columns = columns or {}
    filters = filters or {}

    client = StrataClient(config=config, base_url=base_url)
    try:
        for name, uri in tables.items():
            artifact = client.materialize(
                inputs=[uri],
                transform=_build_scan_transform(columns.get(name), filters.get(name), snapshot_id),
            )
            arrow_table = artifact.to_table()
            ctx.register_record_batches(name, [arrow_table.to_batches()])

        df = ctx.sql(sql)
        return df.collect()
    finally:
        client.close()


class StrataDataFusionContext:
    """A DataFusion context plus one Strata client, for registering several tables.

    Example:
        with StrataDataFusionContext() as ctx:
            ctx.register("events", "file:///warehouse#db.events")
            result = ctx.sql("SELECT * FROM events WHERE value > 100").collect()
    """

    def __init__(
        self,
        config: HasServerUrl | None = None,
        base_url: str | None = None,
    ) -> None:
        import datafusion

        self.client = StrataClient(config=config, base_url=base_url)
        self.ctx = datafusion.SessionContext()
        self._tables: dict[str, pa.Table] = {}  # Keep references to prevent GC

    def __enter__(self) -> "StrataDataFusionContext":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the Strata client connection."""
        self.client.close()

    def register(
        self,
        name: str,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> "StrataDataFusionContext":
        """Fetch a Strata table now and register it.

        Args:
            name: Table name in DataFusion's catalog.
            table_uri: Iceberg table URI.
            snapshot_id: Snapshot to read (None for latest).
            columns: Columns to project.
            filters: Filters for row-group pruning.

        Returns:
            self, for chaining.
        """
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = artifact.to_table()

        self._tables[name] = arrow_table

        self.ctx.register_record_batches(name, [arrow_table.to_batches()])
        return self

    def sql(self, query: str) -> "datafusion.DataFrame":
        """Execute a SQL query.

        Args:
            query: SQL query string.

        Returns:
            DataFusion DataFrame with the query results.
        """
        return self.ctx.sql(query)

    def table(self, name: str) -> "datafusion.DataFrame":
        """Get a registered table as a DataFrame.

        Args:
            name: Table name.

        Returns:
            DataFusion DataFrame for the table.
        """
        return self.ctx.table(name)

    def tables(self) -> set[str]:
        """List registered table names."""
        return self.ctx.catalog().schema("public").table_names()

    def deregister(self, name: str) -> None:
        """Remove a registered table.

        Args:
            name: Table name to remove.
        """
        self.ctx.deregister_table(name)
        self._tables.pop(name, None)

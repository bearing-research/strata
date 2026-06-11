"""DataFusion integration for Strata.

Provides helpers to use Strata data with Apache DataFusion, an Arrow-native
query engine. DataFusion's execution model aligns well with Strata's
"plan once, cache results" architecture.

Example:
    from strata_client.integration.datafusion import register_strata_table, strata_query

    # Register a Strata table with DataFusion
    ctx = register_strata_table(
        "events",
        "file:///warehouse#db.events",
        columns=["id", "value", "timestamp"],
    )

    # Query using SQL
    result = ctx.sql("SELECT * FROM events WHERE value > 100").collect()

    # Or use the DataFrame API
    df = ctx.table("events")
    result = df.filter(col("value") > lit(100)).collect()

Important: DataFusion filter operations are applied *after* data is fetched
from Strata. To get Strata-side row-group pruning, pass filters to the
registration functions. For best performance, use Strata filters for coarse
pruning and DataFusion for fine-grained predicates.
"""

from typing import TYPE_CHECKING

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter

if TYPE_CHECKING:
    import datafusion
    import pyarrow as pa


def _build_scan_transform(
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
) -> dict:
    """Build a scan@v1 transform specification."""
    params: dict = {}
    if columns is not None:
        params["columns"] = columns
    if filters is not None:
        params["filters"] = [
            {"column": f.column, "op": f.op.value, "value": f.value} for f in filters
        ]
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
    """Register a Strata table with DataFusion.

    Fetches data from Strata and registers it as a table in DataFusion's
    catalog. The table can then be queried using SQL or the DataFrame API.

    Args:
        name: Table name in DataFusion's catalog
        table_uri: Iceberg table URI (e.g., "file:///warehouse#db.table")
        ctx: Existing SessionContext (creates new one if None)
        snapshot_id: Specific snapshot to read (None for latest)
        columns: Columns to project (None for all)
        filters: Filters for Strata-side row-group pruning
        config: Strata configuration
        base_url: Override server URL

    Returns:
        SessionContext with the table registered

    Example:
        from strata_client.integration.datafusion import register_strata_table
        from strata_client.client import gt

        ctx = register_strata_table(
            "events",
            "file:///warehouse#db.events",
            filters=[gt("value", 100)],  # Strata-side pruning
        )

        # Query with SQL
        result = ctx.sql("SELECT id, value FROM events WHERE id < 10").collect()
    """
    import datafusion

    if ctx is None:
        ctx = datafusion.SessionContext()

    client = StrataClient(config=config, base_url=base_url)
    try:
        # Fetch data as Arrow table using unified materialize API
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters),
        )
        arrow_table = artifact.to_table()

        # Register with DataFusion using from_arrow_table
        # This creates a DataFrame internally, we need to register it as a table
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
    """Execute a SQL query over Strata tables using DataFusion.

    Convenience function that registers multiple tables and executes a query.

    Args:
        sql: SQL query to execute
        tables: Mapping of table name to Strata table URI
        snapshot_id: Snapshot ID for all tables (None for latest)
        columns: Per-table column projections
        filters: Per-table Strata filters for row-group pruning
        config: Strata configuration
        base_url: Override server URL

    Returns:
        List of Arrow RecordBatches containing query results

    Example:
        from strata_client.integration.datafusion import strata_query
        from strata_client.client import gt

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
        # Register all tables using unified materialize API
        for name, uri in tables.items():
            artifact = client.materialize(
                inputs=[uri],
                transform=_build_scan_transform(columns.get(name), filters.get(name)),
            )
            arrow_table = artifact.to_table()
            ctx.register_record_batches(name, [arrow_table.to_batches()])

        # Execute query and collect results
        df = ctx.sql(sql)
        return df.collect()
    finally:
        client.close()


class StrataDataFusionContext:
    """A DataFusion context with Strata table registration.

    Maintains a connection to Strata for registering multiple tables
    and running queries.

    Example:
        from strata_client.integration.datafusion import StrataDataFusionContext

        with StrataDataFusionContext() as ctx:
            ctx.register("events", "file:///warehouse#db.events")
            ctx.register("users", "file:///warehouse#db.users")

            # SQL query
            result = ctx.sql("SELECT * FROM events WHERE value > 100").collect()

            # DataFrame API
            df = ctx.table("events")
            result = df.select("id", "value").collect()
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
        """Register a Strata table.

        Args:
            name: Table name in DataFusion's catalog
            table_uri: Iceberg table URI
            snapshot_id: Specific snapshot to read
            columns: Columns to project
            filters: Filters for row-group pruning

        Returns:
            self for method chaining
        """
        # Fetch data using unified materialize API
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters),
        )
        arrow_table = artifact.to_table()

        # Keep reference to prevent garbage collection
        self._tables[name] = arrow_table

        self.ctx.register_record_batches(name, [arrow_table.to_batches()])
        return self

    def sql(self, query: str) -> "datafusion.DataFrame":
        """Execute a SQL query.

        Args:
            query: SQL query string

        Returns:
            DataFusion DataFrame with query results
        """
        return self.ctx.sql(query)

    def table(self, name: str) -> "datafusion.DataFrame":
        """Get a registered table as a DataFrame.

        Args:
            name: Table name

        Returns:
            DataFusion DataFrame for the table
        """
        return self.ctx.table(name)

    def tables(self) -> set[str]:
        """List registered table names."""
        return self.ctx.catalog().schema("public").table_names()

    def deregister(self, name: str) -> None:
        """Remove a registered table.

        Args:
            name: Table name to remove
        """
        self.ctx.deregister_table(name)
        self._tables.pop(name, None)

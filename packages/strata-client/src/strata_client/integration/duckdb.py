"""DuckDB integration for Strata.

Provides helpers to register Strata scans as DuckDB views. Data is fetched
from Strata once at registration time and materialized as an Arrow table.

Important: DuckDB SQL filters (WHERE clauses) are applied *after* the data
is fetched from Strata. To get Strata-side pruning, pass filters to the
register/scan functions. For example:

    # Strata-side pruning (fast, reduces data transfer):
    scanner.register("events", uri, filters=[gt("value", 100)])

    # DuckDB-side filtering (after full scan):
    scanner.query("SELECT * FROM events WHERE value > 100")

For best performance, use Strata filters for coarse pruning and DuckDB
filters for fine-grained predicates.
"""

from typing import Any, TypedDict

import duckdb
import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value


class StrataTableParams(TypedDict, total=False):
    """Parameters for registering a Strata table.

    Required:
        table_uri: Iceberg table URI (e.g., "file:///warehouse#db.events")

    Optional:
        snapshot_id: Specific snapshot to read (default: latest)
        columns: Columns to project (default: all)
        filters: Filters for Strata-side pruning
    """

    table_uri: str
    snapshot_id: int | None
    columns: list[str] | None
    filters: list[Filter] | None


def _build_scan_transform(
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    snapshot_id: int | None = None,
) -> dict[str, Any]:
    """Build a scan@v1 transform specification."""
    params: dict[str, Any] = {}
    if columns:
        params["columns"] = columns
    if filters:
        params["filters"] = [
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
        ]
    if snapshot_id is not None:
        params["snapshot_id"] = snapshot_id
    return {"executor": "scan@v1", "params": params}


def register_strata_scan(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    table_uri: str,
    snapshot_id: int | None = None,
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> pa.Table:
    """Register a Strata scan as a DuckDB view.

    Fetches data from Strata and registers it as a view that DuckDB can query.
    Returns the Arrow table for caller to hold a reference if needed.

    Note: The `filters` parameter controls Strata-side pruning. Any WHERE
    clauses in subsequent DuckDB queries filter the already-fetched data.

    Args:
        conn: DuckDB connection
        name: Name for the registered view (will overwrite if exists)
        table_uri: Iceberg table URI
        snapshot_id: Specific snapshot to read
        columns: Columns to project
        filters: Filters for Strata-side pruning
        config: Strata configuration
        base_url: Override server URL

    Returns:
        The Arrow table that was registered (caller can hold reference)

    Example:
        import duckdb
        from strata_client.integration.duckdb import register_strata_scan
        from strata_client.client import gt

        conn = duckdb.connect()

        # Strata-side filter: only fetch rows where value > 100
        register_strata_scan(
            conn, "my_table", "file:///warehouse#db.events",
            columns=["id", "value"],
            filters=[gt("value", 100)],
        )

        # DuckDB-side filter: applied to already-fetched data
        result = conn.execute("SELECT * FROM my_table WHERE id < 1000").fetchall()
    """
    client = StrataClient(config=config, base_url=base_url)

    try:
        # Materialize the table data
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        # Fetch the artifact data
        arrow_table = client.fetch(artifact.uri)

        # Register as a view in DuckDB (overwrites if exists)
        conn.register(name, arrow_table)

        return arrow_table

    finally:
        client.close()


def strata_query(
    sql: str,
    tables: dict[str, StrataTableParams],
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> pa.Table:
    """Execute a SQL query over Strata tables.

    Convenience function that registers multiple Strata scans and executes
    a query over them.

    Note: WHERE clauses in the SQL are applied *after* data is fetched from
    Strata. Use the `filters` key in table params for Strata-side pruning.

    Args:
        sql: SQL query to execute
        tables: Dict mapping view names to StrataTableParams
        config: Strata configuration
        base_url: Override server URL

    Returns:
        Arrow Table with query results

    Example:
        from strata_client.integration.duckdb import strata_query
        from strata_client.client import gt

        # Strata-side filter via params, DuckDB-side via SQL
        result = strata_query(
            "SELECT id, value FROM events WHERE id < 1000",
            tables={
                "events": {
                    "table_uri": "file:///warehouse#db.events",
                    "columns": ["id", "value"],
                    "filters": [gt("value", 100)],  # Strata-side
                }
            }
        )
    """
    conn = duckdb.connect(database=":memory:")

    # Keep references to Arrow tables to prevent GC during query
    _table_refs: list[pa.Table] = []

    try:
        for name, params in tables.items():
            arrow_table = register_strata_scan(
                conn=conn,
                name=name,
                table_uri=params["table_uri"],
                snapshot_id=params.get("snapshot_id"),
                columns=params.get("columns"),
                filters=params.get("filters"),
                config=config,
                base_url=base_url,
            )
            _table_refs.append(arrow_table)

        result = conn.execute(sql).fetch_arrow_table()
        return result

    finally:
        conn.close()


class StrataScanner:
    """A reusable scanner for DuckDB integration.

    Maintains a DuckDB connection and allows incremental registration of
    Strata tables as views.

    Note on filtering: The `filters` parameter in `register()` controls
    Strata-side pruning (reduces data transfer). WHERE clauses in `query()`
    are applied by DuckDB after data is fetched.

    Example:
        scanner = StrataScanner()

        # Register with Strata-side filter
        scanner.register(
            "events", "file:///warehouse#db.events",
            filters=[gt("value", 100)]
        )
        scanner.register("users", "file:///warehouse#db.users")

        # DuckDB-side filtering in SQL
        result = scanner.query('''
            SELECT e.*, u.name
            FROM events e
            JOIN users u ON e.user_id = u.id
            WHERE e.id < 1000
        ''')

        scanner.close()
    """

    def __init__(
        self,
        config: HasServerUrl | None = None,
        base_url: str | None = None,
    ) -> None:
        self.config = config
        self.base_url = base_url
        self.conn = duckdb.connect(database=":memory:")
        # Keep references to Arrow tables to prevent GC
        self._tables: dict[str, pa.Table] = {}

    def __enter__(self) -> "StrataScanner":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the DuckDB connection and release table references."""
        self.conn.close()
        self._tables.clear()

    def register(
        self,
        name: str,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
        *,
        replace: bool = True,
    ) -> "StrataScanner":
        """Register a Strata table for querying.

        Args:
            name: View name in DuckDB
            table_uri: Iceberg table URI
            snapshot_id: Specific snapshot (default: latest)
            columns: Columns to project (default: all)
            filters: Filters for Strata-side pruning
            replace: If True, replace existing view with same name

        Returns:
            Self for method chaining

        Raises:
            ValueError: If name exists and replace=False
        """
        if not replace and name in self._tables:
            raise ValueError(f"Table '{name}' already registered. Use replace=True to overwrite.")

        arrow_table = register_strata_scan(
            conn=self.conn,
            name=name,
            table_uri=table_uri,
            snapshot_id=snapshot_id,
            columns=columns,
            filters=filters,
            config=self.config,
            base_url=self.base_url,
        )

        # Keep reference to prevent GC
        self._tables[name] = arrow_table

        return self

    def unregister(self, name: str) -> "StrataScanner":
        """Unregister a table.

        Args:
            name: View name to remove

        Returns:
            Self for method chaining
        """
        if name in self._tables:
            self.conn.unregister(name)
            del self._tables[name]
        return self

    @property
    def registered_tables(self) -> list[str]:
        """List of registered table names."""
        return list(self._tables.keys())

    def query(self, sql: str) -> pa.Table:
        """Execute a SQL query and return Arrow Table."""
        return self.conn.execute(sql).fetch_arrow_table()

    def query_df(self, sql: str):
        """Execute a SQL query and return a pandas DataFrame."""
        return self.conn.execute(sql).fetchdf()

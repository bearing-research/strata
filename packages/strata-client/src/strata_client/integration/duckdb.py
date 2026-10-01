"""Register Strata scans as DuckDB views.

Data is fetched once, at registration, into an Arrow table. DuckDB WHERE
clauses run after the fetch; pass Strata filters at registration for
server-side pruning::

    scanner.register("events", uri, filters=[gt("value", 100)])
"""

from typing import Any, TypedDict

import duckdb
import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value


class StrataTableParams(TypedDict, total=False):
    """Parameters for registering a Strata table; only ``table_uri`` is required.

    Attributes:
        table_uri: Iceberg table URI, e.g. "file:///warehouse#db.events".
        snapshot_id: Snapshot to read (default: latest).
        columns: Columns to project (default: all).
        filters: Filters for Strata-side pruning.
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
    """Fetch a Strata scan now and register it as a DuckDB view.

    Args:
        conn: DuckDB connection.
        name: View name; an existing view of that name is replaced.
        table_uri: Iceberg table URI.
        snapshot_id: Snapshot to read (None for latest).
        columns: Columns to project.
        filters: Filters for Strata-side pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        The registered Arrow table; keep a reference while the view is in use.

    Example:
        conn = duckdb.connect()
        register_strata_scan(
            conn, "my_table", "file:///warehouse#db.events", filters=[gt("value", 100)]
        )
        result = conn.execute("SELECT * FROM my_table WHERE id < 1000").fetchall()
    """
    client = StrataClient(config=config, base_url=base_url)

    try:
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = client.fetch(artifact.uri)

        # Overwrites an existing view of the same name.
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
    """Register several Strata scans in an in-memory DuckDB and run one SQL query.

    Args:
        sql: SQL query to execute; its WHERE runs after the fetch.
        tables: View name to StrataTableParams (``filters`` prune server-side).
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        Arrow Table with the query result.

    Example:
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

        result = conn.execute(sql).to_arrow_table()
        return result

    finally:
        conn.close()


class StrataScanner:
    """An in-memory DuckDB connection with Strata tables registered as views.

    Example:
        with StrataScanner() as scanner:
            scanner.register("events", "file:///warehouse#db.events", filters=[gt("value", 100)])
            result = scanner.query("SELECT * FROM events WHERE id < 1000")
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
        """Fetch a Strata table now and register it as a view.

        Args:
            name: View name in DuckDB.
            table_uri: Iceberg table URI.
            snapshot_id: Snapshot to read (default: latest).
            columns: Columns to project (default: all).
            filters: Filters for Strata-side pruning.
            replace: Replace an existing view with the same name.

        Returns:
            self, for chaining.

        Raises:
            ValueError: If the name exists and replace=False.
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

        self._tables[name] = arrow_table

        return self

    def unregister(self, name: str) -> "StrataScanner":
        """Unregister a table; unknown names are ignored.

        Args:
            name: View name to remove.

        Returns:
            self, for chaining.
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
        return self.conn.execute(sql).to_arrow_table()

    def query_df(self, sql: str):
        """Execute a SQL query and return a pandas DataFrame."""
        return self.conn.execute(sql).fetchdf()

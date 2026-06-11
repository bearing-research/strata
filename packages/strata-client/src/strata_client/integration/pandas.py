"""Pandas integration for Strata.

Provides helpers to convert Strata fetches to pandas DataFrames.
Uses Arrow as the intermediate format, then converts to pandas.

Note on memory: Unlike Polars, Arrow → pandas conversion typically
involves a copy because pandas uses its own memory layout. This is
expected behavior and acceptable for most use cases.

Important: pandas filter operations (e.g., df[df["value"] > 100]) are
applied *after* data is fetched from Strata. To get Strata-side pruning,
pass filters to the fetch functions. For example:

    # Strata-side pruning (fast, reduces data transfer):
    df = fetch_to_pandas(uri, filters=[gt("value", 100)])

    # pandas-side filtering (after full fetch):
    df = fetch_to_pandas(uri)
    df = df[df["value"] > 100]

For best performance, use Strata filters for coarse pruning and pandas
filters for fine-grained predicates.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter

if TYPE_CHECKING:
    import pandas as pd


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
            {"column": f.column, "op": f.op.value, "value": f.value} for f in filters
        ]
    if snapshot_id is not None:
        params["snapshot_id"] = snapshot_id
    return {"executor": "scan@v1", "params": params}


def fetch_to_pandas(
    table_uri: str,
    snapshot_id: int | None = None,
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> "pd.DataFrame":
    """Fetch an Iceberg table via Strata and return a pandas DataFrame.

    This is the simplest way to get Iceberg data into pandas.
    Converts via Arrow (may copy data due to pandas memory layout).

    Args:
        table_uri: Iceberg table URI (e.g., "file:///warehouse#db.table")
        snapshot_id: Specific snapshot to read (None for latest)
        columns: Columns to project (None for all)
        filters: Filters for row-group pruning
        config: Strata configuration
        base_url: Override server URL (default: http://127.0.0.1:8765)

    Returns:
        pandas DataFrame with the fetch results

    Example:
        from strata_client.integration.pandas import fetch_to_pandas
        from strata_client.client import gt

        df = fetch_to_pandas(
            "file:///warehouse#db.events",
            columns=["id", "value", "timestamp"],
            filters=[gt("value", 100.0)],
        )
        print(df.head())
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
        # Convert Arrow table to pandas (may copy due to memory layout)
        return arrow_table.to_pandas()
    finally:
        client.close()


# Backwards compatibility alias
scan_to_pandas = fetch_to_pandas


class StrataPandasScanner:
    """A reusable scanner for pandas integration.

    Maintains a connection to the Strata server for multiple fetches.

    Example:
        from strata_client.integration.pandas import StrataPandasScanner

        with StrataPandasScanner() as scanner:
            events = scanner.fetch("file:///warehouse#db.events")
            users = scanner.fetch("file:///warehouse#db.users")

            # Merge in pandas
            result = events.merge(users, on="user_id")
    """

    def __init__(
        self,
        config: HasServerUrl | None = None,
        base_url: str | None = None,
    ) -> None:
        self.client = StrataClient(config=config, base_url=base_url)

    def __enter__(self) -> "StrataPandasScanner":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the client connection."""
        self.client.close()

    def fetch(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> "pd.DataFrame":
        """Fetch a table and return a pandas DataFrame."""
        # Materialize the table data
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        # Fetch the artifact data
        arrow_table = self.client.fetch(artifact.uri)
        return arrow_table.to_pandas()

    # Backwards compatibility alias
    scan = fetch

    def fetch_batches(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Fetch a table and yield Arrow RecordBatches.

        Uses the unified materialize API and returns all batches from the
        Arrow IPC stream.

        Args:
            table_uri: Iceberg table URI
            snapshot_id: Specific snapshot to read
            columns: Columns to project
            filters: Filters for row-group pruning

        Yields:
            pyarrow.RecordBatch objects from the fetched data

        Example:
            with StrataPandasScanner() as scanner:
                for batch in scanner.fetch_batches("file:///warehouse#db.events"):
                    # Process each batch
                    df = batch.to_pandas()
                    process(df)
        """
        # Materialize and fetch
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = self.client.fetch(artifact.uri)
        yield from arrow_table.to_batches()

    # Backwards compatibility alias
    scan_batches = fetch_batches

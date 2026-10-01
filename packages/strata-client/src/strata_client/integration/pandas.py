"""Fetch Strata scans as pandas DataFrames, via Arrow (usually a copy).

pandas filtering runs after the fetch; pass Strata filters for server-side pruning::

    df = fetch_to_pandas(uri, filters=[gt("value", 100)])
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value

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
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
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
    """Fetch an Iceberg table via Strata as a pandas DataFrame.

    Args:
        table_uri: Iceberg table URI, e.g. "file:///warehouse#db.table".
        snapshot_id: Snapshot to read (None for latest).
        columns: Columns to project (None for all).
        filters: Filters for row-group pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        pandas DataFrame with the scan result.

    Example:
        df = fetch_to_pandas("file:///warehouse#db.events", filters=[gt("value", 100.0)])
    """
    client = StrataClient(config=config, base_url=base_url)

    try:
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = client.fetch(artifact.uri)
        # May copy, depending on memory layout.
        return arrow_table.to_pandas()
    finally:
        client.close()


# Backwards compatibility alias
scan_to_pandas = fetch_to_pandas


class StrataPandasScanner:
    """One Strata client reused across several pandas fetches.

    Example:
        with StrataPandasScanner() as scanner:
            events = scanner.fetch("file:///warehouse#db.events")
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
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
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
        """Fetch a table and yield its Arrow RecordBatches (the whole table is fetched first).

        Args:
            table_uri: Iceberg table URI.
            snapshot_id: Snapshot to read (None for latest).
            columns: Columns to project.
            filters: Filters for row-group pruning.

        Yields:
            pyarrow.RecordBatch objects.

        Example:
            for batch in scanner.fetch_batches("file:///warehouse#db.events"):
                process(batch.to_pandas())
        """
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = self.client.fetch(artifact.uri)
        yield from arrow_table.to_batches()

    # Backwards compatibility alias
    scan_batches = fetch_batches

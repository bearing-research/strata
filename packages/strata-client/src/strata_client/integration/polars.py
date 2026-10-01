"""Fetch Strata scans as Polars frames, usually zero-copy from Arrow.

Polars filtering runs after the fetch; pass Strata filters for server-side pruning::

    df = fetch_to_polars(uri, filters=[gt("value", 100)])
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value

if TYPE_CHECKING:
    import polars as pl


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


def fetch_to_polars(
    table_uri: str,
    snapshot_id: int | None = None,
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> "pl.DataFrame":
    """Fetch an Iceberg table via Strata as a Polars DataFrame.

    Args:
        table_uri: Iceberg table URI, e.g. "file:///warehouse#db.table".
        snapshot_id: Snapshot to read (None for latest).
        columns: Columns to project (None for all).
        filters: Filters for row-group pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        Polars DataFrame with the scan result.

    Example:
        df = fetch_to_polars("file:///warehouse#db.events", filters=[gt("value", 100.0)])
    """
    import polars as pl

    client = StrataClient(config=config, base_url=base_url)

    try:
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = client.fetch(artifact.uri)
        # Usually zero-copy. ``pl.from_arrow`` on a ``pa.Table`` always yields a DataFrame;
        # narrow so the union does not leak into the signature.
        result = pl.from_arrow(arrow_table)
        assert isinstance(result, pl.DataFrame)
        return result
    finally:
        client.close()


# Backwards compatibility alias
scan_to_polars = fetch_to_polars


def fetch_to_lazy(
    table_uri: str,
    snapshot_id: int | None = None,
    columns: list[str] | None = None,
    filters: list[Filter] | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> "pl.LazyFrame":
    """Fetch an Iceberg table via Strata as a Polars LazyFrame.

    The data is fetched eagerly; only the downstream Polars operations are lazy.

    Args:
        table_uri: Iceberg table URI.
        snapshot_id: Snapshot to read (None for latest).
        columns: Columns to project.
        filters: Filters for row-group pruning.
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        Polars LazyFrame over the fetched data.

    Example:
        lf = fetch_to_lazy("file:///warehouse#db.events")
        result = lf.filter(pl.col("value") > 100).collect()
    """
    df = fetch_to_polars(
        table_uri=table_uri,
        snapshot_id=snapshot_id,
        columns=columns,
        filters=filters,
        config=config,
        base_url=base_url,
    )
    return df.lazy()


# Backwards compatibility alias
scan_to_lazy = fetch_to_lazy


class StrataPolarsScanner:
    """One Strata client reused across several Polars fetches.

    Example:
        with StrataPolarsScanner() as scanner:
            events = scanner.fetch("file:///warehouse#db.events")
    """

    def __init__(
        self,
        config: HasServerUrl | None = None,
        base_url: str | None = None,
    ) -> None:
        self.client = StrataClient(config=config, base_url=base_url)

    def __enter__(self) -> "StrataPolarsScanner":
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
    ) -> "pl.DataFrame":
        """Fetch a table and return a Polars DataFrame."""
        import polars as pl

        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = self.client.fetch(artifact.uri)
        # ``pa.Table`` → DataFrame (never Series); narrow to honor the signature.
        result = pl.from_arrow(arrow_table)
        assert isinstance(result, pl.DataFrame)
        return result

    # Backwards compatibility alias
    scan = fetch

    def fetch_lazy(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> "pl.LazyFrame":
        """Fetch a table eagerly and return it as a Polars LazyFrame."""
        return self.fetch(
            table_uri=table_uri,
            snapshot_id=snapshot_id,
            columns=columns,
            filters=filters,
        ).lazy()

    # Backwards compatibility alias
    scan_lazy = fetch_lazy

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
                process(pl.from_arrow(batch))
        """
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        arrow_table = self.client.fetch(artifact.uri)
        yield from arrow_table.to_batches()

    # Backwards compatibility alias
    scan_batches = fetch_batches

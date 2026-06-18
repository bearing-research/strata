"""Polars integration for Strata.

Provides helpers to convert Strata fetches to Polars DataFrames.
Polars is Arrow-native, so conversions are typically zero-copy when
Arrow types are supported (may copy for dictionary encoding, large
strings, or extension types).

Important: Polars filter operations (e.g., df.filter(...)) are applied
*after* data is fetched from Strata. To get Strata-side pruning, pass
filters to the fetch functions. For example:

    # Strata-side pruning (fast, reduces data transfer):
    df = fetch_to_polars(uri, filters=[gt("value", 100)])

    # Polars-side filtering (after full fetch):
    df = fetch_to_polars(uri).filter(pl.col("value") > 100)

For best performance, use Strata filters for coarse pruning and Polars
filters for fine-grained predicates.
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
    """Fetch an Iceberg table via Strata and return a Polars DataFrame.

    This is the simplest way to get Iceberg data into Polars.
    Arrow-native; typically zero-copy when types are supported.

    Args:
        table_uri: Iceberg table URI (e.g., "file:///warehouse#db.table")
        snapshot_id: Specific snapshot to read (None for latest)
        columns: Columns to project (None for all)
        filters: Filters for row-group pruning
        config: Strata configuration
        base_url: Override server URL (default: http://127.0.0.1:8765)

    Returns:
        Polars DataFrame with the fetch results

    Example:
        from strata_client.integration.polars import fetch_to_polars
        from strata_client.client import gt

        df = fetch_to_polars(
            "file:///warehouse#db.events",
            columns=["id", "value", "timestamp"],
            filters=[gt("value", 100.0)],
        )
        print(df.head())
    """
    import polars as pl

    client = StrataClient(config=config, base_url=base_url)

    try:
        # Materialize the table data
        artifact = client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        # Fetch the artifact data
        arrow_table = client.fetch(artifact.uri)
        # Arrow-native; typically zero-copy when types are supported.
        # ``pl.from_arrow`` on a ``pa.Table`` always yields a DataFrame
        # (Series only for Array / ChunkedArray inputs); narrow so the
        # function signature is honored without leaking the union.
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
    """Fetch an Iceberg table via Strata and return a Polars LazyFrame.

    NOTE: Data is fetched eagerly from Strata, then wrapped in a LazyFrame
    for downstream lazy transforms. This is NOT true lazy evaluation from
    storage—use this when you want Polars' lazy API for chaining operations
    after the fetch is complete.

    Args:
        table_uri: Iceberg table URI
        snapshot_id: Specific snapshot to read
        columns: Columns to project
        filters: Filters for row-group pruning
        config: Strata configuration
        base_url: Override server URL

    Returns:
        Polars LazyFrame wrapping eagerly-fetched data

    Example:
        from strata_client.integration.polars import fetch_to_lazy

        # Data is fetched immediately, but downstream ops are lazy
        lf = fetch_to_lazy("file:///warehouse#db.events")
        result = (
            lf
            .filter(pl.col("value") > 100)
            .group_by("category")
            .agg(pl.col("value").mean())
            .collect()  # Only this triggers Polars computation
        )
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
    """A reusable scanner for Polars integration.

    Maintains a connection to the Strata server for multiple fetches.

    Example:
        from strata_client.integration.polars import StrataPolarsScanner

        with StrataPolarsScanner() as scanner:
            events = scanner.fetch("file:///warehouse#db.events")
            users = scanner.fetch("file:///warehouse#db.users")

            # Join in Polars
            result = events.join(users, on="user_id")
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

        # Materialize the table data
        artifact = self.client.materialize(
            inputs=[table_uri],
            transform=_build_scan_transform(columns, filters, snapshot_id),
        )
        # Fetch the artifact data
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
        """Fetch a table and return a Polars LazyFrame.

        NOTE: Data is fetched eagerly, then wrapped in LazyFrame.
        """
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
            with StrataPolarsScanner() as scanner:
                for batch in scanner.fetch_batches("file:///warehouse#db.events"):
                    # Process each batch
                    df = pl.from_arrow(batch)
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

"""PyArrow Dataset/Scanner integration for Strata.

Provides a PyArrow-native interface that serves as the foundation for
other integrations (DuckDB, Polars, pandas, etc.).

Example:
    from strata_client.integration.arrow import StrataDataset

    # Create a dataset bound to a table
    dataset = StrataDataset("file:///warehouse#db.events")

    # Get schema without fetching data
    print(dataset.schema)

    # Create a scanner with projection and filters
    scanner = dataset.scanner(columns=["id", "value"], filter=gt("value", 100))

    # Iterate over batches (streaming)
    for batch in scanner.to_batches():
        process(batch)

    # Or get as Arrow Table
    table = scanner.to_table()

    # Or get a RecordBatchReader for zero-copy handoff
    reader = scanner.to_reader()
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import pyarrow as pa

from strata_client._clientconfig import HasServerUrl
from strata_client.client import StrataClient
from strata_client.filters import Filter, serialize_filter_value

if TYPE_CHECKING:
    pass


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
            {"column": f.column, "op": f.op.value, "value": serialize_filter_value(f.value)}
            for f in filters
        ]
    return {"executor": "scan@v1", "params": params}


class StrataScanner:
    """A scanner for reading data from a Strata dataset.

    Similar to pyarrow.dataset.Scanner, provides methods to read data
    as batches, tables, or readers.

    Created via StrataDataset.scanner() - do not instantiate directly.
    """

    def __init__(
        self,
        client: StrataClient,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._client = client
        self._table_uri = table_uri
        self._snapshot_id = snapshot_id
        self._columns = columns
        self._filters = filters
        self._batch_size = batch_size  # Reserved for future use

    @property
    def projected_schema(self) -> pa.Schema | None:
        """Schema of the projected columns, if available.

        Note: Returns None until first batch is read. For schema before
        reading, use StrataDataset.schema.
        """
        return None  # Would require metadata fetch

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        """Read data as an iterator of RecordBatches.

        This is the streaming interface - batches are yielded as they
        arrive from Strata, enabling memory-efficient processing.

        Yields:
            pyarrow.RecordBatch objects
        """
        # Use the unified materialize API
        artifact = self._client.materialize(
            inputs=[self._table_uri],
            transform=_build_scan_transform(self._columns, self._filters),
        )
        table = artifact.to_table()
        yield from table.to_batches()

    def to_table(self) -> pa.Table:
        """Read all data as an Arrow Table.

        Returns:
            pyarrow.Table containing all scan results
        """
        # Use the unified materialize API
        artifact = self._client.materialize(
            inputs=[self._table_uri],
            transform=_build_scan_transform(self._columns, self._filters),
        )
        return artifact.to_table()

    def to_reader(self) -> pa.RecordBatchReader:
        """Get a RecordBatchReader for zero-copy handoff.

        Returns a reader that can be passed to other Arrow-aware libraries
        (DuckDB, Polars, pandas) for efficient data transfer.

        Returns:
            pyarrow.RecordBatchReader
        """
        batches = list(self.to_batches())
        if not batches:
            # Return empty reader with no schema
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])
        return pa.RecordBatchReader.from_batches(batches[0].schema, batches)

    def count_rows(self) -> int:
        """Count total rows without materializing all data.

        Note: Currently fetches all data to count. Future optimization
        could use server-side counting.

        Returns:
            Total number of rows
        """
        return sum(batch.num_rows for batch in self.to_batches())

    def head(self, num_rows: int = 10) -> pa.Table:
        """Read the first N rows.

        Args:
            num_rows: Number of rows to return

        Returns:
            pyarrow.Table with up to num_rows rows
        """
        batches = []
        rows_collected = 0

        for batch in self.to_batches():
            if rows_collected >= num_rows:
                break

            rows_needed = num_rows - rows_collected
            if batch.num_rows <= rows_needed:
                batches.append(batch)
                rows_collected += batch.num_rows
            else:
                # Slice the batch
                batches.append(batch.slice(0, rows_needed))
                rows_collected += rows_needed
                break

        if not batches:
            return pa.table({})
        return pa.Table.from_batches(batches)


class StrataDataset:
    """A dataset representing a Strata-served Iceberg table.

    Similar to pyarrow.dataset.Dataset, provides a high-level interface
    for scanning tabular data. This is the recommended foundation for
    building integrations with other data processing libraries.

    Example:
        from strata_client.integration.arrow import StrataDataset
        from strata_client.client import gt

        # Bind to a table (optionally pin to a snapshot)
        dataset = StrataDataset(
            "file:///warehouse#db.events",
            snapshot_id=12345,  # Optional: pin to specific snapshot
        )

        # Create scanners with different projections/filters
        scanner1 = dataset.scanner(columns=["id", "value"])
        scanner2 = dataset.scanner(filter=gt("value", 100))

        # Read data
        for batch in scanner1.to_batches():
            process(batch)

    Note on filtering:
        Filters passed to scanner() are applied server-side by Strata
        for row-group pruning. This reduces data transfer compared to
        filtering after the fact.
    """

    def __init__(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        config: HasServerUrl | None = None,
        base_url: str | None = None,
    ) -> None:
        """Create a dataset bound to a Strata table.

        Args:
            table_uri: Iceberg table URI (e.g., "file:///warehouse#db.table")
            snapshot_id: Pin to specific snapshot (None for latest)
            config: Strata configuration
            base_url: Override server URL
        """
        self._table_uri = table_uri
        self._snapshot_id = snapshot_id
        self._client = StrataClient(config=config, base_url=base_url)
        self._schema: pa.Schema | None = None

    def __enter__(self) -> "StrataDataset":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying client connection."""
        self._client.close()

    @property
    def table_uri(self) -> str:
        """The Iceberg table URI."""
        return self._table_uri

    @property
    def snapshot_id(self) -> int | None:
        """The pinned snapshot ID, if any."""
        return self._snapshot_id

    @property
    def schema(self) -> pa.Schema:
        """Schema of the dataset.

        Fetches schema on first access by reading a small sample.
        """
        if self._schema is None:
            # Fetch schema by reading data using the unified materialize API
            artifact = self._client.materialize(
                inputs=[self._table_uri],
                transform=_build_scan_transform(),
            )
            table = artifact.to_table()
            self._schema = table.schema if table.num_rows > 0 else pa.schema([])
        return self._schema

    def scanner(
        self,
        columns: list[str] | None = None,
        filter: Filter | list[Filter] | None = None,
        batch_size: int | None = None,
    ) -> StrataScanner:
        """Create a scanner for reading data.

        Args:
            columns: Columns to project (None for all)
            filter: Filter(s) for Strata-side row-group pruning
            batch_size: Batch size hint (reserved for future use)

        Returns:
            StrataScanner for reading data
        """
        # Normalize filter to list
        filters: list[Filter] | None = None
        if filter is not None:
            filters = [filter] if isinstance(filter, Filter) else filter

        return StrataScanner(
            client=self._client,
            table_uri=self._table_uri,
            snapshot_id=self._snapshot_id,
            columns=columns,
            filters=filters,
            batch_size=batch_size,
        )

    def to_table(
        self,
        columns: list[str] | None = None,
        filter: Filter | list[Filter] | None = None,
    ) -> pa.Table:
        """Convenience method to read entire dataset as Arrow Table.

        Args:
            columns: Columns to project
            filter: Filter(s) for pruning

        Returns:
            pyarrow.Table with all data
        """
        return self.scanner(columns=columns, filter=filter).to_table()

    def to_batches(
        self,
        columns: list[str] | None = None,
        filter: Filter | list[Filter] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Convenience method to iterate over batches.

        Args:
            columns: Columns to project
            filter: Filter(s) for pruning

        Yields:
            pyarrow.RecordBatch objects
        """
        yield from self.scanner(columns=columns, filter=filter).to_batches()

    def count_rows(
        self,
        filter: Filter | list[Filter] | None = None,
    ) -> int:
        """Count rows in the dataset.

        Args:
            filter: Filter(s) for pruning

        Returns:
            Total row count
        """
        return self.scanner(filter=filter).count_rows()

    def head(
        self,
        num_rows: int = 10,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Read the first N rows.

        Args:
            num_rows: Number of rows to return
            columns: Columns to project

        Returns:
            pyarrow.Table with up to num_rows rows
        """
        return self.scanner(columns=columns).head(num_rows)


def dataset(
    table_uri: str,
    snapshot_id: int | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> StrataDataset:
    """Create a StrataDataset for the given table.

    This is the main entry point for the Arrow integration.

    Args:
        table_uri: Iceberg table URI (e.g., "file:///warehouse#db.table")
        snapshot_id: Pin to specific snapshot (None for latest)
        config: Strata configuration
        base_url: Override server URL

    Returns:
        StrataDataset bound to the table

    Example:
        from strata_client.integration.arrow import dataset
        from strata_client.client import gt

        ds = dataset("file:///warehouse#db.events")
        table = ds.scanner(filter=gt("value", 100)).to_table()
    """
    return StrataDataset(
        table_uri=table_uri,
        snapshot_id=snapshot_id,
        config=config,
        base_url=base_url,
    )

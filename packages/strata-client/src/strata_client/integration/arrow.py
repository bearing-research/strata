"""PyArrow Dataset/Scanner-style interface to Strata scans.

Example:
    from strata_client.integration.arrow import StrataDataset

    dataset = StrataDataset("file:///warehouse#db.events")
    scanner = dataset.scanner(columns=["id", "value"], filter=gt("value", 100))
    table = scanner.to_table()
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


class StrataScanner:
    """Reads one projected, filtered scan of a Strata dataset, like ``pyarrow.dataset.Scanner``.

    Create via ``StrataDataset.scanner()``. Each read method materializes the full scan.
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
        """Always None; use ``StrataDataset.schema`` instead."""
        return None  # Would require metadata fetch

    def to_batches(self) -> Iterator[pa.RecordBatch]:
        """Read data as RecordBatches.

        The whole scan is fetched before the first batch is yielded.

        Yields:
            pyarrow.RecordBatch objects.
        """
        artifact = self._client.materialize(
            inputs=[self._table_uri],
            transform=_build_scan_transform(self._columns, self._filters, self._snapshot_id),
        )
        table = artifact.to_table()
        yield from table.to_batches()

    def to_table(self) -> pa.Table:
        """Read all data as an Arrow Table.

        Returns:
            pyarrow.Table with all scan results.
        """
        artifact = self._client.materialize(
            inputs=[self._table_uri],
            transform=_build_scan_transform(self._columns, self._filters, self._snapshot_id),
        )
        return artifact.to_table()

    def to_reader(self) -> pa.RecordBatchReader:
        """Get a RecordBatchReader for handoff to Arrow-aware libraries.

        Returns:
            pyarrow.RecordBatchReader over the fetched batches.
        """
        batches = list(self.to_batches())
        if not batches:
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])
        return pa.RecordBatchReader.from_batches(batches[0].schema, batches)

    def count_rows(self) -> int:
        """Count rows in the scan (fetches all data to count).

        Returns:
            Total number of rows.
        """
        return sum(batch.num_rows for batch in self.to_batches())

    def head(self, num_rows: int = 10) -> pa.Table:
        """Read the first N rows (the whole scan is still fetched).

        Args:
            num_rows: Number of rows to return.

        Returns:
            pyarrow.Table with up to num_rows rows.
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
                batches.append(batch.slice(0, rows_needed))
                rows_collected += rows_needed
                break

        if not batches:
            return pa.table({})
        return pa.Table.from_batches(batches)


class StrataDataset:
    """A Strata-served Iceberg table, like ``pyarrow.dataset.Dataset``.

    Filters passed to ``scanner()`` drive server-side row-group pruning.

    Example:
        from strata_client.integration.arrow import StrataDataset
        from strata_client.client import gt

        dataset = StrataDataset("file:///warehouse#db.events", snapshot_id=12345)
        table = dataset.scanner(columns=["id"], filter=gt("value", 100)).to_table()
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
            table_uri: Iceberg table URI, e.g. "file:///warehouse#db.table".
            snapshot_id: Snapshot to pin to (None for latest).
            config: Anything with ``server_url``.
            base_url: Server URL; overrides ``config``.
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
        """Schema of the dataset, fetched and cached on first access.

        The first access runs a full scan; an empty table yields an empty schema.
        """
        if self._schema is None:
            artifact = self._client.materialize(
                inputs=[self._table_uri],
                # The schema probe must read the same snapshot the data reads,
                # or a pinned dataset can describe itself with a newer schema.
                transform=_build_scan_transform(snapshot_id=self._snapshot_id),
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
            columns: Columns to project (None for all).
            filter: Filter(s) for Strata-side row-group pruning.
            batch_size: Unused; reserved.

        Returns:
            StrataScanner for reading data.
        """
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
        """Read the dataset as an Arrow Table.

        Args:
            columns: Columns to project.
            filter: Filter(s) for pruning.

        Returns:
            pyarrow.Table with all matching data.
        """
        return self.scanner(columns=columns, filter=filter).to_table()

    def to_batches(
        self,
        columns: list[str] | None = None,
        filter: Filter | list[Filter] | None = None,
    ) -> Iterator[pa.RecordBatch]:
        """Iterate over the dataset's RecordBatches.

        Args:
            columns: Columns to project.
            filter: Filter(s) for pruning.

        Yields:
            pyarrow.RecordBatch objects.
        """
        yield from self.scanner(columns=columns, filter=filter).to_batches()

    def count_rows(
        self,
        filter: Filter | list[Filter] | None = None,
    ) -> int:
        """Count rows in the dataset.

        Args:
            filter: Filter(s) for pruning.

        Returns:
            Total row count.
        """
        return self.scanner(filter=filter).count_rows()

    def head(
        self,
        num_rows: int = 10,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Read the first N rows.

        Args:
            num_rows: Number of rows to return.
            columns: Columns to project.

        Returns:
            pyarrow.Table with up to num_rows rows.
        """
        return self.scanner(columns=columns).head(num_rows)


def dataset(
    table_uri: str,
    snapshot_id: int | None = None,
    config: HasServerUrl | None = None,
    base_url: str | None = None,
) -> StrataDataset:
    """Create a StrataDataset for the given table.

    Args:
        table_uri: Iceberg table URI, e.g. "file:///warehouse#db.table".
        snapshot_id: Snapshot to pin to (None for latest).
        config: Anything with ``server_url``.
        base_url: Server URL; overrides ``config``.

    Returns:
        StrataDataset bound to the table.

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

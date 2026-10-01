"""Parquet fetcher: reads row groups into Arrow RecordBatches.

This module provides a clean seam for future Rust acceleration.
The Fetcher protocol defines the interface that any implementation must satisfy.
"""

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, cast

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from strata.iceberg_equality import EqualityDeleteSets, deleted_mask
from strata.iceberg_schema import read_as_snapshot, source_columns
from strata.metrics import MetricsCollector
from strata.types import Task

if TYPE_CHECKING:
    import pyarrow.fs as pafs

# pyarrow.compute registers its kernels at import time, so ty does not know
# members like ``is_in``. Cast through Any.
pc = cast(Any, _pc)

_MAX_FILE_CACHE_SIZE = 128


class Fetcher(Protocol):
    """Protocol for fetching row groups from Parquet files.

    This abstraction allows swapping the Python implementation
    with a Rust-based one without changing the public API.
    """

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a single row group as a RecordBatch.

        Args:
            task: The task describing which row group to fetch

        Returns:
            Arrow RecordBatch containing the row group data
        """
        ...

    def fetch_to_table(self, tasks: list[Task]) -> pa.Table:
        """Fetch multiple row groups and combine into a Table.

        Args:
            tasks: List of tasks to fetch

        Returns:
            Arrow Table containing all row group data
        """
        ...


class PyArrowFetcher:
    """Python implementation of Parquet fetcher using PyArrow.

    Supports both local filesystem and S3 storage backends.
    S3 files are identified by the s3:// prefix.
    """

    def __init__(
        self,
        metrics: MetricsCollector | None = None,
        max_file_cache_size: int = _MAX_FILE_CACHE_SIZE,
        s3_filesystem: "pafs.S3FileSystem | None" = None,
        max_equality_delete_rows: int = 10_000_000,
    ) -> None:
        self.metrics = metrics or MetricsCollector()
        self._max_file_cache_size = max_file_cache_size
        self._s3_filesystem = s3_filesystem
        self._equality_deletes = EqualityDeleteSets(s3_filesystem, max_equality_delete_rows)
        # Parsed footers, least recently used first. Footers, not open handles:
        # the fetch pool reads row groups of one file in parallel, and a
        # ParquetFile is not thread-safe (pyarrow 25 segfaults). An immutable
        # FileMetaData is shared; each read opens its own handle with it.
        self._file_cache: OrderedDict[str, pq.FileMetaData] = OrderedDict()
        self._file_cache_lock = threading.Lock()

    @staticmethod
    def _close_parquet_file(parquet_file: pq.ParquetFile) -> None:
        """Close a ParquetFile handle if it exposes a close method."""
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()

    def _open(self, file_path: str) -> pq.ParquetFile:
        """A handle on *file_path* for one read, opened with its cached footer."""
        from strata.lake_files import open_parquet

        with self._file_cache_lock:
            metadata = self._file_cache.get(file_path)
            if metadata is not None:
                self._file_cache.move_to_end(file_path)

        # Open outside the lock: opening an S3 file is a network round-trip,
        # and holding the lock across it would serialize the whole fetch pool.
        pf = open_parquet(file_path, self._s3_filesystem, metadata=metadata)
        if metadata is None:
            with self._file_cache_lock:
                self._file_cache[file_path] = pf.metadata
                self._file_cache.move_to_end(file_path)
                while len(self._file_cache) > self._max_file_cache_size:
                    self._file_cache.popitem(last=False)
        return pf

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a single row group as a RecordBatch."""
        start_time = time.perf_counter()

        columns = task.columns
        if task.file_columns is not None:
            columns = source_columns(task.file_columns, task.columns)
        # Equality delete keys the caller did not ask for: read, used, dropped.
        keys_only: list[str] = []
        if task.equality_deletes and columns is not None:
            keys_only = [
                name
                for _, name, _ in task.equality_columns
                if name is not None and name not in columns
            ]
            columns = [*columns, *keys_only]
        pf = self._open(task.file_path)
        try:
            table = pf.read_row_group(task.row_group_id, columns=columns)
        finally:
            self._close_parquet_file(pf)
        # One mask over the rows as read, so positions stay the file's.
        deleted = None
        if task.deleted_rows is not None:
            positions = pa.array(range(table.num_rows), pa.int64())
            deleted = pc.is_in(positions, value_set=task.deleted_rows)
        if task.equality_deletes:
            hit = deleted_mask(
                table,
                {field_id: name for field_id, name, _ in task.equality_columns},
                task.equality_deletes,
                self._equality_deletes.keys,
                defaults={field_id: default for field_id, _, default in task.equality_columns},
                downcast_ns=task.downcast_ns,
            )
            deleted = hit if deleted is None else pc.or_(deleted, hit)
        if deleted is not None:
            table = table.filter(pc.invert(deleted))
        if keys_only:
            table = table.drop_columns(keys_only)
        if task.file_columns is not None:
            table = read_as_snapshot(table, task.file_columns, task.columns)

        if table.num_rows == 0:
            batch = pa.RecordBatch.from_pylist([], schema=table.schema)
        else:
            table = table.combine_chunks()
            batch = table.to_batches()[0]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        bytes_read = batch.nbytes
        task.bytes_read = bytes_read

        self.metrics.record_fetch(
            bytes_read=bytes_read,
            rows_read=batch.num_rows,
            elapsed_ms=elapsed_ms,
            from_cache=False,
        )

        return batch

    def fetch_to_table(self, tasks: list[Task]) -> pa.Table:
        """Fetch multiple row groups and combine into a Table."""
        if not tasks:
            return pa.table({})

        batches = [self.fetch(task) for task in tasks]
        return pa.Table.from_batches(batches)

    def close(self) -> None:
        """Forget the cached footers. No handle outlives the read that opened it."""
        with self._file_cache_lock:
            self._file_cache.clear()


def create_fetcher(
    metrics: MetricsCollector | None = None,
    s3_filesystem: "pafs.S3FileSystem | None" = None,
    max_equality_delete_rows: int = 10_000_000,
) -> Fetcher:
    """Factory function to create a Fetcher.

    This provides a clean seam for future Rust integration.
    When a Rust fetcher is available, this function can be
    updated to return it based on configuration.

    Args:
        metrics: Optional metrics collector
        s3_filesystem: Optional S3 filesystem for reading from S3
        max_equality_delete_rows: Equality delete rows kept parsed in memory
            (``StrataConfig.max_equality_delete_rows``)

    Returns:
        A Fetcher instance
    """
    return PyArrowFetcher(
        metrics, s3_filesystem=s3_filesystem, max_equality_delete_rows=max_equality_delete_rows
    )

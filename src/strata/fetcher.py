"""Parquet fetcher: reads row groups into Arrow RecordBatches.

This module provides a clean seam for future Rust acceleration.
The Fetcher protocol defines the interface that any implementation must satisfy.
"""

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from strata.metrics import MetricsCollector
from strata.types import Task

if TYPE_CHECKING:
    import pyarrow.fs as pafs

# Maximum number of ParquetFile handles to cache
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
    ) -> None:
        self.metrics = metrics or MetricsCollector()
        self._max_file_cache_size = max_file_cache_size
        self._s3_filesystem = s3_filesystem
        # OrderedDict for LRU eviction of file handles.
        #
        # One Fetcher instance is shared by the whole fetch thread pool
        # (``max_fetch_workers``, 32 by default), so every mutation of this
        # dict is concurrent and must hold the lock. Unsynchronized
        # ``move_to_end`` / ``__setitem__`` / ``popitem`` on an OrderedDict can
        # also lose or duplicate entries outright, leaking handles.
        self._file_cache: OrderedDict[str, pq.ParquetFile] = OrderedDict()
        self._file_cache_lock = threading.Lock()

    @staticmethod
    def _close_parquet_file(parquet_file: pq.ParquetFile) -> None:
        """Close a ParquetFile handle if it exposes a close method."""
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()

    def _get_parquet_file(self, file_path: str) -> pq.ParquetFile:
        """Get a cached ParquetFile handle with LRU eviction."""
        with self._file_cache_lock:
            if file_path in self._file_cache:
                # Move to end (most recently used)
                self._file_cache.move_to_end(file_path)
                return self._file_cache[file_path]

        # Open outside the lock: opening an S3 file is a network round-trip and
        # holding the lock across it would serialize the whole fetch pool. A
        # concurrent open of the same path just means one redundant handle,
        # which the insert below resolves.

        # Open new file (with S3 filesystem if needed)
        if file_path.startswith("s3://"):
            if self._s3_filesystem is None:
                # Create default S3 filesystem on demand
                import pyarrow.fs as pafs

                self._s3_filesystem = pafs.S3FileSystem()
            # Strip s3:// prefix for PyArrow filesystem
            s3_path = file_path[5:]
            pf = pq.ParquetFile(s3_path, filesystem=self._s3_filesystem)
        else:
            pf = pq.ParquetFile(file_path)
        with self._file_cache_lock:
            existing = self._file_cache.get(file_path)
            if existing is not None:
                # Another thread opened it first; keep one handle and let ours
                # be closed when this frame drops the last reference.
                self._file_cache.move_to_end(file_path)
                return existing
            self._file_cache[file_path] = pf

            # Evict oldest if over limit.
            #
            # Deliberately does NOT close the evicted handle. Another thread may
            # be inside ``pf.read_row_group(...)`` with it right now — closing
            # it there raised "I/O operation on closed file" mid-stream, after
            # the 200 had already been sent. Dropping the reference is enough:
            # CPython closes the file once the last user releases it, so the fd
            # bound is honoured a moment later instead of a moment too early.
            while len(self._file_cache) > self._max_file_cache_size:
                self._file_cache.popitem(last=False)

        return pf

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a single row group as a RecordBatch."""
        start_time = time.perf_counter()

        pf = self._get_parquet_file(task.file_path)

        # Read the specific row group with optional column projection
        table = pf.read_row_group(task.row_group_id, columns=task.columns)

        # Convert to a single RecordBatch
        # combine_chunks() is more efficient than manual concat_arrays
        if table.num_rows == 0:
            batch = pa.RecordBatch.from_pydict({}, schema=table.schema)
        else:
            # Combine chunked arrays, then get single batch
            table = table.combine_chunks()
            batch = table.to_batches()[0]

        # Track metrics
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
        """Close cached file handles.

        Unlike eviction, this is a shutdown path: no fetch is in flight, so
        closing eagerly is safe and releases the fds immediately.
        """
        with self._file_cache_lock:
            handles = list(self._file_cache.values())
            self._file_cache.clear()
        for parquet_file in handles:
            self._close_parquet_file(parquet_file)


def create_fetcher(
    metrics: MetricsCollector | None = None,
    s3_filesystem: "pafs.S3FileSystem | None" = None,
) -> Fetcher:
    """Factory function to create a Fetcher.

    This provides a clean seam for future Rust integration.
    When a Rust fetcher is available, this function can be
    updated to return it based on configuration.

    Args:
        metrics: Optional metrics collector
        s3_filesystem: Optional S3 filesystem for reading from S3

    Returns:
        A Fetcher instance
    """
    return PyArrowFetcher(metrics, s3_filesystem=s3_filesystem)

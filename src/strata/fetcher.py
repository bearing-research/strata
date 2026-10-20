"""Parquet fetcher: reads row groups into Arrow RecordBatches.

The Fetcher protocol is the seam for swapping in another implementation.
"""

import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Protocol, cast

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq

from strata.iceberg.equality import EqualityDeleteSets, deleted_mask
from strata.iceberg.schema import read_as_snapshot, source_columns
from strata.observability.metrics import MetricsCollector
from strata.types import Task

if TYPE_CHECKING:
    import pyarrow.fs as pafs

# pyarrow.compute registers its kernels at import time, so ty does not know
# members like ``is_in``. Cast through Any.
pc = cast(Any, _pc)

_MAX_FILE_CACHE_SIZE = 128


def _chunk_bytes(row_group: pq.RowGroupMetaData, columns: list[str] | None) -> int:
    """The compressed size of the row group's column chunks a read of *columns* fetches."""
    wanted = None if columns is None else set(columns)
    return sum(
        chunk.total_compressed_size
        for chunk in (row_group.column(i) for i in range(row_group.num_columns))
        if wanted is None or chunk.path_in_schema.split(".")[0] in wanted
    )


class Fetcher(Protocol):
    """Protocol for fetching row groups from Parquet files."""

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a single row group as a RecordBatch."""
        ...

    def fetch_to_table(self, tasks: list[Task]) -> pa.Table:
        """Fetch multiple row groups and combine them into a Table."""
        ...


class PyArrowFetcher:
    """PyArrow Parquet fetcher for local files and ``s3://`` paths."""

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
        # Parsed footers by (path, manifest size), least recently used first. Footers,
        # not open handles: the fetch pool reads row groups of one file in parallel, and a
        # ParquetFile is not thread-safe (pyarrow 25 segfaults). An immutable
        # FileMetaData is shared; each read opens its own handle with it.
        self._file_cache: OrderedDict[tuple[str, int], pq.FileMetaData] = OrderedDict()
        self._file_cache_lock = threading.Lock()

    @staticmethod
    def _close_parquet_file(parquet_file: pq.ParquetFile) -> None:
        """Close a ParquetFile handle if it exposes a close method."""
        close = getattr(parquet_file, "close", None)
        if callable(close):
            close()

    def _open(self, file_path: str, file_size: int) -> pq.ParquetFile:
        """Return a handle on *file_path* for one read, opened with its cached footer.

        The footer is cached under the manifest's *file_size* too, so a table recreated at
        the same paths does not read its new files with the old files' footers.
        """
        from strata.iceberg.lake_files import open_parquet

        key = (file_path, file_size)
        with self._file_cache_lock:
            metadata = self._file_cache.get(key)
            if metadata is not None:
                self._file_cache.move_to_end(key)

        # Open outside the lock: opening an S3 file is a network round-trip,
        # and holding the lock across it would serialize the whole fetch pool.
        pf = open_parquet(file_path, self._s3_filesystem, metadata=metadata)
        if metadata is None:
            with self._file_cache_lock:
                self._file_cache[key] = pf.metadata
                self._file_cache.move_to_end(key)
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
        pf = self._open(task.file_path, task.cache_key.file_size)
        try:
            table = pf.read_row_group(task.row_group_id, columns=columns)
            task.storage_bytes = _chunk_bytes(pf.metadata.row_group(task.row_group_id), columns)
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
    """Create a Fetcher.

    ``max_equality_delete_rows`` bounds the equality-delete rows kept parsed in
    memory (``StrataConfig.max_equality_delete_rows``).
    """
    return PyArrowFetcher(
        metrics, s3_filesystem=s3_filesystem, max_equality_delete_rows=max_equality_delete_rows
    )

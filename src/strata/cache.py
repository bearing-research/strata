"""Disk cache for Arrow IPC row group data."""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.ipc as ipc

from strata.cache_metrics import get_eviction_tracker
from strata.cache_stats import get_cache_histogram
from strata.config import StrataConfig
from strata.fetcher import Fetcher, create_fetcher
from strata.metrics import MetricsCollector
from strata.tracing import trace_span
from strata.types import CacheGranularity, CacheKey, ReadPlan, Task

# Cache file extension (Arrow IPC Stream format for zero-copy serving)
CACHE_FILE_EXTENSION = ".arrowstream"
# Metadata sidecar file extension
CACHE_META_EXTENSION = ".meta.json"

# Cache version - bump this when cache format changes to invalidate old caches.
# This is baked into the cache directory structure so old and new caches coexist.
# Version history:
#   1: Initial version (Arrow IPC stream format, SHA-256 keyed)
#   2: Multi-tenancy support (tenant_id in cache key, tenant-prefixed directories)
#   3: created_at / stats timestamps are epoch floats (were ISO-8601 strings)
CACHE_VERSION = 3


@dataclass
class CacheEntryMetadata:
    """Sidecar metadata for one cached row group.

    Attributes
    ----------
    table_id : str
        Identity of the source table.
    snapshot_id : int
        Iceberg snapshot the row group belongs to.
    file_path : str
        Source data file path.
    row_group_id : int
        Row group index within the file.
    columns : list of str or None
        Projected columns, or ``None`` for all columns.
    num_rows : int
        Rows in the cached batch.
    size_bytes : int
        Serialized Arrow IPC stream size in bytes.
    created_at : float
        Unix epoch timestamp (seconds) of when the entry was written.
    """

    table_id: str
    snapshot_id: int
    file_path: str
    row_group_id: int
    columns: list[str] | None
    num_rows: int
    size_bytes: int
    created_at: float


@dataclass
class CacheStats:
    """Aggregate statistics for the disk cache.

    Attributes
    ----------
    total_entries : int
        Number of cached row groups (current version only).
    total_size_bytes : int
        Total on-disk size of cached data.
    max_size_bytes : int
        Configured cache size limit.
    usage_percent : float
        ``total_size_bytes / max_size_bytes * 100``.
    oldest_entry, newest_entry : float or None
        Epoch timestamps of the oldest/newest entries, or ``None`` when empty.
    entries_by_table : dict of str to int
        Entry count per ``table_id``.
    entries_by_snapshot : dict of str to int
        Entry count per ``"table_id:snapshot_id"``.
    """

    total_entries: int
    total_size_bytes: int
    max_size_bytes: int
    usage_percent: float
    oldest_entry: float | None
    newest_entry: float | None
    entries_by_table: dict[str, int]
    entries_by_snapshot: dict[str, int]


class Cache(Protocol):
    """Interface for cache backends."""

    def get(self, key: CacheKey) -> pa.RecordBatch | None:
        """Return the cached record batch for ``key``, or ``None`` on a miss."""
        ...

    def put(self, key: CacheKey, batch: pa.RecordBatch) -> None:
        """Store ``batch`` under ``key``."""
        ...

    def contains(self, key: CacheKey) -> bool:
        """Return whether ``key`` is cached."""
        ...

    def clear(self) -> None:
        """Remove all cached data."""
        ...


class DiskCache:
    """Disk-based cache using the Arrow IPC Stream format.

    Each cached row group is a separate ``.arrowstream`` file named by the
    SHA-256 hash of its cache key. Because the on-disk format is the same as
    the network transfer format, a cache hit is a pure file read with zero
    Arrow parsing — the bytes go straight from disk to the network::

        disk -> read_file_bytes -> network   (no Arrow parsing)
    """

    def __init__(
        self,
        config: StrataConfig,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize the cache and ensure its directory exists.

        Parameters
        ----------
        config : StrataConfig
            Supplies the cache directory, size limit, and granularity.
        metrics : MetricsCollector, optional
            Metrics sink; a fresh collector is created when omitted.
        """
        self.cache_dir = config.cache_dir
        self.max_size_bytes = config.max_cache_size_bytes
        self.granularity = config.cache_granularity
        self.metrics = metrics or MetricsCollector()

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: CacheKey) -> Path:
        """Return (and create the directory for) a cache key's data file.

        The layout is
        ``cache_dir/v{VERSION}/{tenant_prefix}/{hash[:2]}/{hash[2:4]}/{hash}.arrowstream``.
        The ``tenant_prefix`` (first 8 chars of ``SHA-256(tenant_id)``) isolates
        each tenant's entries, and the version prefix lets old and new cache
        formats coexist so a format bump can't corrupt readers.

        Parameters
        ----------
        key : CacheKey
            The cache key.

        Returns
        -------
        pathlib.Path
            Path to the entry's ``.arrowstream`` data file.
        """
        import hashlib

        hex_digest = key.to_hex(self.granularity)
        # Tenant prefix for isolation (hash first 8 chars for consistent directory naming)
        tenant_prefix = hashlib.sha256(key.tenant_id.encode()).hexdigest()[:8]
        # Version prefix + tenant prefix + two-level directory structure
        subdir = (
            self.cache_dir / f"v{CACHE_VERSION}" / tenant_prefix / hex_digest[:2] / hex_digest[2:4]
        )
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{hex_digest}{CACHE_FILE_EXTENSION}"

    def _meta_path(self, data_path: Path) -> Path:
        """Return the metadata sidecar path for a data file."""
        return data_path.with_suffix(CACHE_META_EXTENSION)

    def _data_path_from_meta(self, meta_path: Path) -> Path:
        """Return the data file path for a metadata sidecar path."""
        path_str = str(meta_path)
        if path_str.endswith(CACHE_META_EXTENSION):
            return Path(path_str.removesuffix(CACHE_META_EXTENSION) + CACHE_FILE_EXTENSION)
        return meta_path

    def _delete_entry_files(self, data_path: Path) -> None:
        """Delete an entry's data file and its metadata sidecar."""
        data_path.unlink(missing_ok=True)
        self._meta_path(data_path).unlink(missing_ok=True)

    def get(self, key: CacheKey) -> pa.RecordBatch | None:
        """Return the cached record batch for ``key``, parsing the stream.

        For the zero-parse hot path use :meth:`get_as_stream_bytes`. A file that
        fails to parse is treated as corrupt and removed.

        Parameters
        ----------
        key : CacheKey
            The cache key.

        Returns
        -------
        pyarrow.RecordBatch or None
            The cached batch, or ``None`` on a miss / corrupt entry.
        """
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            # Read stream format and parse
            stream_bytes = path.read_bytes()
            reader = ipc.open_stream(pa.BufferReader(stream_bytes))
            batches = list(reader)
            if not batches:
                return None
            return batches[0]
        except Exception:
            # Corrupted cache file, remove it
            self._delete_entry_files(path)
            return None

    def get_as_stream_bytes(self, key: CacheKey) -> bytes | None:
        """Return cached data as Arrow IPC stream bytes (zero-copy hot path).

        Since data is stored in stream format, a hit needs no Arrow parsing
        (``disk -> mmap -> bytes -> network``); memory-mapped reads via Rust
        (when available) speed up large files and repeated access.

        Parameters
        ----------
        key : CacheKey
            The cache key.

        Returns
        -------
        bytes or None
            The entry's Arrow IPC stream bytes, or ``None`` on a miss / read
            failure.
        """
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            # Use mmap-based read for better performance on large files
            # and OS page cache reuse on repeated access
            from strata import fast_io

            return fast_io.read_file_mmap(str(path))
        except OSError:
            # The file vanished or could not be read (read_file_mmap returns
            # raw bytes, so this is an I/O error, not Arrow corruption). Drop
            # the unreadable entry and treat it as a miss.
            self._delete_entry_files(path)
            return None

    def get_path(self, key: CacheKey) -> Path | None:
        """Return the cache file path for ``key`` if it exists.

        Useful for zero-copy streaming where the caller handles the file
        directly.

        Parameters
        ----------
        key : CacheKey
            The cache key.

        Returns
        -------
        pathlib.Path or None
            The data file path, or ``None`` on a miss.
        """
        path = self._key_path(key)
        if path.exists():
            return path
        return None

    def put(self, key: CacheKey, batch: pa.RecordBatch) -> None:
        """Store ``batch`` under ``key`` crash-safely via atomic rename.

        Writes the Arrow IPC stream + a JSON metadata sidecar to unique temp
        files, then ``os.replace``-es both into place — so concurrent writers
        don't race and a crash never leaves a half-written entry.

        Parameters
        ----------
        key : CacheKey
            The cache key.
        batch : pyarrow.RecordBatch
            The batch to cache.
        """
        import uuid

        path = self._key_path(key)
        # Use unique suffix to avoid race between concurrent writers
        unique_suffix = uuid.uuid4().hex[:8]
        tmp_path = path.with_suffix(f".{unique_suffix}.tmp")
        meta_path = self._meta_path(path)
        meta_tmp_path = meta_path.with_suffix(f".{unique_suffix}.tmp")

        try:
            # Serialize to stream format (same as network transfer format)
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, batch.schema)
            writer.write_batch(batch)
            writer.close()
            stream_bytes = sink.getvalue().to_pybytes()

            # Write to temp file first
            tmp_path.write_bytes(stream_bytes)

            # Write metadata sidecar
            metadata = CacheEntryMetadata(
                table_id=key.table_id,
                snapshot_id=key.snapshot_id,
                file_path=key.file_path,
                row_group_id=key.row_group_id,
                columns=None,  # Could be extracted from projection_fingerprint if needed
                num_rows=batch.num_rows,
                size_bytes=len(stream_bytes),
                created_at=time.time(),
            )
            meta_tmp_path.write_text(json.dumps(asdict(metadata)))

            # Atomic rename both files
            # If another thread already wrote, that's fine - we just overwrite with same data
            os.replace(tmp_path, path)
            os.replace(meta_tmp_path, meta_path)

            self.metrics.record_cache_write(len(stream_bytes))

            # Evict old entries if over size limit
            self._evict_if_needed()
        except Exception:
            # Failed to write, clean up temp files
            tmp_path.unlink(missing_ok=True)
            meta_tmp_path.unlink(missing_ok=True)
            raise

    def contains(self, key: CacheKey) -> bool:
        """Return whether ``key`` is cached."""
        return self._key_path(key).exists()

    def clear(self) -> None:
        """Remove all cached data (preserving ``metadata.sqlite``)."""
        import shutil

        for item in self.cache_dir.iterdir():
            # Skip metadata database - it's managed by MetadataStore
            if item.name == "metadata.sqlite":
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def get_size_bytes(self) -> int:
        """Return the current cache size in bytes (current version only)."""
        total = 0
        versioned_dir = self.cache_dir / f"v{CACHE_VERSION}"
        if versioned_dir.exists():
            for path in versioned_dir.rglob(f"*{CACHE_FILE_EXTENSION}"):
                total += path.stat().st_size
        return total

    def get_stats(self) -> CacheStats:
        """Compute aggregate cache statistics (current version only).

        Walks the metadata sidecars to count entries by table/snapshot, total
        size, and the oldest/newest timestamps. Corrupt sidecars are skipped,
        and a sidecar with no data file is pruned.

        Returns
        -------
        CacheStats
            The aggregate statistics.
        """
        total_entries = 0
        total_size = 0
        timestamps: list[float] = []
        by_table: dict[str, int] = {}
        by_snapshot: dict[str, int] = {}

        versioned_dir = self.cache_dir / f"v{CACHE_VERSION}"
        if not versioned_dir.exists():
            return CacheStats(
                total_entries=0,
                total_size_bytes=0,
                max_size_bytes=self.max_size_bytes,
                usage_percent=0.0,
                oldest_entry=None,
                newest_entry=None,
                entries_by_table={},
                entries_by_snapshot={},
            )

        for meta_path in versioned_dir.rglob(f"*{CACHE_META_EXTENSION}"):
            try:
                data_path = self._data_path_from_meta(meta_path)
                if not data_path.exists():
                    meta_path.unlink(missing_ok=True)
                    continue
                meta = CacheEntryMetadata(**json.loads(meta_path.read_text()))
                total_entries += 1
                total_size += data_path.stat().st_size
                timestamps.append(meta.created_at)

                # Count by table
                by_table[meta.table_id] = by_table.get(meta.table_id, 0) + 1

                # Count by snapshot
                snap_key = f"{meta.table_id}:{meta.snapshot_id}"
                by_snapshot[snap_key] = by_snapshot.get(snap_key, 0) + 1
            except Exception:
                # Skip corrupted metadata files
                continue

        # Sort timestamps to find oldest/newest
        timestamps.sort()
        oldest = timestamps[0] if timestamps else None
        newest = timestamps[-1] if timestamps else None

        usage_pct = (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0

        return CacheStats(
            total_entries=total_entries,
            total_size_bytes=total_size,
            max_size_bytes=self.max_size_bytes,
            usage_percent=usage_pct,
            oldest_entry=oldest,
            newest_entry=newest,
            entries_by_table=by_table,
            entries_by_snapshot=by_snapshot,
        )

    def list_entries(self) -> list[CacheEntryMetadata]:
        """Return every cached entry's metadata (current version only).

        Returns
        -------
        list of CacheEntryMetadata
            One entry per readable sidecar; corrupt sidecars are skipped.
        """
        entries = []
        versioned_dir = self.cache_dir / f"v{CACHE_VERSION}"
        if not versioned_dir.exists():
            return entries
        for meta_path in versioned_dir.rglob(f"*{CACHE_META_EXTENSION}"):
            try:
                meta = CacheEntryMetadata(**json.loads(meta_path.read_text()))
                entries.append(meta)
            except Exception:
                continue
        return entries

    def _evict_if_needed(self) -> None:
        """Evict oldest entries (by mtime) when the cache exceeds its limit.

        Eviction is oldest-first by file mtime (write time), not LRU — ``get``
        doesn't touch mtime. Evicts down to 80% of the limit to avoid running
        on every ``put``. Only the current-version directory is touched.
        """
        current_size = self.get_size_bytes()
        if current_size <= self.max_size_bytes:
            return

        size_before = current_size

        # Get all cache files sorted by modification time (oldest first)
        versioned_dir = self.cache_dir / f"v{CACHE_VERSION}"
        if not versioned_dir.exists():
            return
        files = []
        for path in versioned_dir.rglob(f"*{CACHE_FILE_EXTENSION}"):
            files.append((path, path.stat().st_mtime, path.stat().st_size))
        files.sort(key=lambda x: x[1])

        # Evict until under limit (target 80% to avoid evicting on every put)
        target_size = int(self.max_size_bytes * 0.8)
        evicted_count = 0
        evicted_bytes = 0
        while current_size > target_size and files:
            path, _, size = files.pop(0)
            path.unlink(missing_ok=True)
            # Also remove metadata sidecar
            self._meta_path(path).unlink(missing_ok=True)
            current_size -= size
            evicted_count += 1
            evicted_bytes += size

        # Record eviction metrics
        if evicted_count > 0:
            self.metrics.record_cache_eviction(evicted_count, evicted_bytes)
            # Record detailed eviction event
            tracker = get_eviction_tracker()
            tracker.record_eviction(
                files_evicted=evicted_count,
                bytes_evicted=evicted_bytes,
                cache_size_before=size_before,
                cache_size_after=current_size,
                reason="size_limit",
            )


class CachedFetcher:
    """A :class:`~strata.fetcher.Fetcher` wrapper that caches results.

    Composes a fetcher and a cache so callers get transparent caching.
    """

    def __init__(
        self,
        config: StrataConfig,
        fetcher: Fetcher | None = None,
        cache: Cache | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Compose the fetcher and cache.

        Parameters
        ----------
        config : StrataConfig
            Server configuration (S3 filesystem, cache settings).
        fetcher : Fetcher, optional
            Backing fetcher; one is created (with the configured S3 filesystem)
            when omitted.
        cache : Cache, optional
            Cache backend; a :class:`DiskCache` is created when omitted.
        metrics : MetricsCollector, optional
            Metrics sink; a fresh collector is created when omitted.
        """
        self.config = config
        self.metrics = metrics or MetricsCollector()

        # Create fetcher with S3 filesystem if configured
        if fetcher is None:
            s3_filesystem = None
            if config.s3_region or config.s3_access_key or config.s3_anonymous:
                s3_filesystem = config.get_s3_filesystem()
            self.fetcher = create_fetcher(self.metrics, s3_filesystem=s3_filesystem)
        else:
            self.fetcher = fetcher

        self.cache = cache or DiskCache(config, self.metrics)

    @staticmethod
    def _project_batch(batch: pa.RecordBatch, columns: list[str] | None) -> pa.RecordBatch:
        """Return ``batch`` projected to ``columns`` (or unchanged if ``None``)."""
        if columns is None:
            return batch
        if batch.schema.names == columns:
            return batch
        return pa.RecordBatch.from_arrays(
            [batch.column(batch.schema.get_field_index(name)) for name in columns],
            names=columns,
        )

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a row group, serving from cache when possible.

        On a miss, fetches from storage, caches the (full) row group, and
        returns the requested projection.

        Parameters
        ----------
        task : Task
            The row-group fetch task; its ``cached`` / ``bytes_read`` are updated.

        Returns
        -------
        pyarrow.RecordBatch
            The (projected) row group.
        """
        histogram = get_cache_histogram()
        cache_full_row_groups = self.config.cache_granularity == CacheGranularity.ROW_GROUP

        # Check cache first
        cached_batch = self.cache.get(task.cache_key)
        if cached_batch is not None:
            result_batch = self._project_batch(cached_batch, task.columns)
            task.cached = True
            task.bytes_read = result_batch.nbytes
            self.metrics.record_fetch(
                bytes_read=result_batch.nbytes,
                rows_read=result_batch.num_rows,
                elapsed_ms=0.0,
                from_cache=True,
            )
            # Record hit in histogram
            histogram.record_hit(
                bytes_accessed=result_batch.nbytes,
                table_id=task.cache_key.table_id,
            )
            return result_batch

        # Fetch from storage with tracing
        with trace_span(
            "fetch_row_group",
            file_path=task.file_path,
            row_group_id=task.row_group_id,
            cache_hit=False,
        ) as span:
            fetch_task = task
            if cache_full_row_groups and task.columns is not None:
                fetch_task = Task(
                    file_path=task.file_path,
                    row_group_id=task.row_group_id,
                    cache_key=task.cache_key,
                    num_rows=task.num_rows,
                    columns=None,
                    estimated_bytes=task.estimated_bytes,
                )
            batch = self.fetcher.fetch(fetch_task)
            span.set_attribute("bytes_read", batch.nbytes)
            span.set_attribute("num_rows", batch.num_rows)

        # Record miss in histogram
        histogram.record_miss(
            bytes_accessed=batch.nbytes,
            table_id=task.cache_key.table_id,
        )

        # Store in cache
        self.cache.put(task.cache_key, batch)

        result_batch = self._project_batch(batch, task.columns)
        task.bytes_read = result_batch.nbytes
        return result_batch

    def execute_plan(self, plan: ReadPlan) -> list[pa.RecordBatch]:
        """Execute a read plan and return all batches.

        Parameters
        ----------
        plan : ReadPlan
            The plan to execute.

        Returns
        -------
        list of pyarrow.RecordBatch
            One batch per task, in plan order.
        """
        batches = []
        for task in plan.tasks:
            batch = self.fetch(task)
            batches.append(batch)
        return batches

    def stream_plan(self, plan: ReadPlan):
        """Execute a read plan, yielding one batch at a time.

        Parameters
        ----------
        plan : ReadPlan
            The plan to execute.

        Yields
        ------
        pyarrow.RecordBatch
            Each task's batch, in plan order.
        """
        for task in plan.tasks:
            yield self.fetch(task)

    def stream_plan_as_ipc(self, plan: ReadPlan):
        """Execute a read plan, yielding Arrow IPC stream bytes per batch.

        Parameters
        ----------
        plan : ReadPlan
            The plan to execute.

        Yields
        ------
        bytes
            Each batch serialized to Arrow IPC stream format.
        """
        for task in plan.tasks:
            batch = self.fetch(task)
            # Serialize to IPC stream format
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, batch.schema)
            writer.write_batch(batch)
            writer.close()
            yield sink.getvalue().to_pybytes()

    def fetch_as_stream_bytes(self, task: Task) -> bytes:
        """Fetch a row group as Arrow IPC stream bytes (optimized hot path).

        On a cache hit (full-row-group granularity), uses the Rust-accelerated
        zero-parse path (``disk -> mmap -> bytes``). On a miss, fetches, caches,
        and serializes.

        Parameters
        ----------
        task : Task
            The fetch task; its ``cached`` / ``bytes_read`` are updated.

        Returns
        -------
        bytes
            Arrow IPC stream bytes ready for network transfer.
        """
        histogram = get_cache_histogram()
        cache_full_row_groups = self.config.cache_granularity == CacheGranularity.ROW_GROUP

        # Check if DiskCache (not just Cache protocol) for optimized path
        if isinstance(self.cache, DiskCache) and not (
            cache_full_row_groups and task.columns is not None
        ):
            stream_bytes = self.cache.get_as_stream_bytes(task.cache_key)
            if stream_bytes is not None:
                task.cached = True
                task.bytes_read = len(stream_bytes)
                self.metrics.record_fetch(
                    bytes_read=len(stream_bytes),
                    rows_read=0,  # We don't parse the batch, so row count unknown
                    elapsed_ms=0.0,
                    from_cache=True,
                )
                # Record hit in histogram
                histogram.record_hit(
                    bytes_accessed=len(stream_bytes),
                    table_id=task.cache_key.table_id,
                )
                return stream_bytes

        # Cache miss or non-DiskCache: fetch, cache, serialize
        # Note: self.fetch() may record its own metrics with batch.nbytes,
        # but we override task.bytes_read below to reflect actual IPC stream size.
        batch = self.fetch(task)

        # Serialize to IPC stream format
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        stream_bytes = sink.getvalue().to_pybytes()

        # Set task metrics to reflect actual output (IPC stream bytes)
        # This overrides bytes_read set by fetch() to use stream size for consistency
        task.bytes_read = len(stream_bytes)
        # task.cached already set by fetch() (True if cache hit, False if miss)

        return stream_bytes

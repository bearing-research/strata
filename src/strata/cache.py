"""Disk cache for Arrow IPC row group data."""

import json
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.ipc as ipc

from strata.cache_metrics import get_eviction_tracker
from strata.cache_stats import get_cache_histogram
from strata.config import StrataConfig
from strata.fetcher import Fetcher, create_fetcher
from strata.file_modes import private_dir
from strata.metrics import MetricsCollector
from strata.tracing import trace_span
from strata.types import CacheGranularity, CacheKey, ReadPlan, Task

# Arrow IPC stream format, served zero-copy.
CACHE_FILE_EXTENSION = ".arrowstream"
CACHE_META_EXTENSION = ".meta.json"

# Bump when the cache format changes. Baked into the directory layout; a DiskCache deletes other
# versions' directories at startup, since nothing else would evict them.
CACHE_VERSION = 4

# A cache directory of some version, current or not: ``v`` and digits only.
_VERSION_DIR = re.compile(r"v\d+")
# What every version has written under its directory: hex-named directories
# (a tenant prefix of 8, then two levels of 2; version 1 had no tenant prefix)
# holding entries, their metadata sidecars and write temp files.
_CACHE_SUBDIR = re.compile(r"[0-9a-f]{2}|[0-9a-f]{8}")
_CACHE_FILE_SUFFIXES = (CACHE_FILE_EXTENSION, CACHE_META_EXTENSION, ".tmp")

logger = logging.getLogger(__name__)

# Every Arrow IPC stream opens with a continuation marker and, once its writer
# is closed (``put`` always closes), ends with an end-of-stream marker. Both are
# fixed, so a head+tail check validates an entry Strata wrote itself for free.
_IPC_STREAM_HEAD = b"\xff\xff\xff\xff"
_IPC_STREAM_TAIL = b"\xff\xff\xff\xff\x00\x00\x00\x00"


def _write_durably(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` and fsync it before returning.

    The later ``os.replace`` is atomic but not durable: without the flush a crash can leave
    the rename applied and the data blocks unwritten.
    """
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass
class CacheEntryMetadata:
    """Sidecar metadata for one cached row group.

    ``columns`` is ``None`` for all columns; ``created_at`` is epoch seconds.
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
    """Aggregate statistics for the disk cache (current cache version only).

    Timestamps are epoch seconds, ``None`` when empty; ``entries_by_snapshot`` is keyed by
    ``"table_id:snapshot_id"``.
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


def _not_a_cache_entry(tree: Path) -> Path | None:
    """The first thing under *tree* that a cache does not write, or ``None``."""
    for root, dirs, files in os.walk(tree):
        for name in dirs:
            if not _CACHE_SUBDIR.fullmatch(name):
                return Path(root) / name
        for name in files:
            if not name.endswith(_CACHE_FILE_SUFFIXES):
                return Path(root) / name
    return None


class DiskCache:
    """Disk cache storing each row group as an Arrow IPC stream file named by its key hash.

    The on-disk format is the network format, so a hit is a file read with no Arrow parsing.
    """

    def __init__(
        self,
        config: StrataConfig,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Create the cache directory and remove other cache versions' directories."""
        self.cache_dir = config.cache_dir
        self.max_size_bytes = config.max_cache_size_bytes
        self.granularity = config.cache_granularity
        self.metrics = metrics or MetricsCollector()

        # Row groups of every tenant's tables: no other account on the host reads them.
        private_dir(self.cache_dir)
        self._remove_other_versions()

    def _remove_other_versions(self) -> None:
        """Delete the directories of other cache versions.

        Size accounting and eviction only walk the current version, so old ones would never be
        evicted. Only ``v<N>`` directories holding nothing but cache files are removed, since
        ``cache_dir`` may hold the user's own files.
        """
        current = f"v{CACHE_VERSION}"
        for item in self.cache_dir.iterdir():
            if item.name == current or not _VERSION_DIR.fullmatch(item.name):
                continue
            if not item.is_dir():
                continue
            foreign = _not_a_cache_entry(item)
            if foreign is not None:
                logger.warning(
                    "Left %s in the cache directory: it holds %s, which the cache did not write",
                    item,
                    foreign,
                )
                continue
            try:
                shutil.rmtree(item)
            except OSError as e:
                # Another process sharing the cache may be removing it too; a
                # leftover directory is not worth failing startup over.
                logger.warning("Could not remove cache directory %s: %s", item, e)
                continue
            logger.info("Removed cache directory %s left by another cache version", item)

    def _key_path(self, key: CacheKey) -> Path:
        """Return (and create the directory for) a cache key's data file.

        Layout: ``cache_dir/v{VERSION}/{tenant_prefix}/{hash[:2]}/{hash[2:4]}/{hash}.arrowstream``,
        where ``tenant_prefix`` is the first 8 hex chars of ``SHA-256(tenant_id)``.
        """
        import hashlib

        hex_digest = key.to_hex(self.granularity)
        # Hashed tenant prefix for isolation.
        tenant_prefix = hashlib.sha256(key.tenant_id.encode()).hexdigest()[:8]
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
        """Return the cached record batch for ``key``, or ``None`` on a miss.

        Parses the stream; use :meth:`get_as_stream_bytes` for the zero-parse path. An entry that
        fails to parse is treated as corrupt and removed.
        """
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            stream_bytes = path.read_bytes()
            reader = ipc.open_stream(pa.BufferReader(stream_bytes))
            batches = list(reader)
            if not batches:
                return None
            return batches[0]
        except Exception:
            # Corrupted: remove it.
            self._delete_entry_files(path)
            return None

    def get_as_stream_bytes(self, key: CacheKey) -> bytes | None:
        """Return cached Arrow IPC stream bytes with no parsing, or ``None`` on a miss/read failure.

        Uses the Rust mmap reader when available.
        """
        path = self._key_path(key)
        if not path.exists():
            return None

        try:
            # mmap for large files and OS page-cache reuse on repeated access.
            from strata import fast_io

            data = fast_io.read_file_mmap(str(path))
        except OSError:
            # Vanished or unreadable (an I/O error, not Arrow corruption, since this returns raw
            # bytes): drop the entry and treat it as a miss.
            self._delete_entry_files(path)
            return None

        # This path does not parse Arrow, and keys are never invalidated, so a damaged entry would
        # be served on every later request (``get`` self-heals by parsing). The fixed head and tail
        # markers are already in memory: checking them costs nothing and catches a zeroed, empty or
        # truncated entry.
        if not (data.startswith(_IPC_STREAM_HEAD) and data.endswith(_IPC_STREAM_TAIL)):
            self._delete_entry_files(path)
            return None

        return data

    def get_path(self, key: CacheKey) -> Path | None:
        """Return the cache file path for ``key``, or ``None`` on a miss."""
        path = self._key_path(key)
        if path.exists():
            return path
        return None

    def put(self, key: CacheKey, batch: pa.RecordBatch) -> None:
        """Store ``batch`` under ``key`` crash-safely.

        Data and metadata sidecar go to unique temp files, then ``os.replace`` into place, so
        concurrent writers do not race and a crash never leaves a half-written entry.
        """
        import uuid

        path = self._key_path(key)
        # Unique suffix so concurrent writers do not race on one temp file.
        unique_suffix = uuid.uuid4().hex[:8]
        tmp_path = path.with_suffix(f".{unique_suffix}.tmp")
        meta_path = self._meta_path(path)
        meta_tmp_path = meta_path.with_suffix(f".{unique_suffix}.tmp")

        try:
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, batch.schema)
            writer.write_batch(batch)
            writer.close()
            stream_bytes = sink.getvalue().to_pybytes()

            # fsync before the rename: os.replace is atomic for observers, not durable. After a
            # power loss the rename can survive while its data blocks do not, and with immutable
            # keys nothing would ever invalidate that entry. Costs one flush, on the miss path only.
            _write_durably(tmp_path, stream_bytes)

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
            _write_durably(meta_tmp_path, json.dumps(asdict(metadata)).encode())

            # Another thread may already have written this entry; overwriting it with the same data
            # is fine.
            os.replace(tmp_path, path)
            os.replace(meta_tmp_path, meta_path)

            self.metrics.record_cache_write(len(stream_bytes))

            self._evict_if_needed()
        except Exception:
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
            # Skip the metadata database (MetadataStore's) and its -wal / -shm sidecars. In WAL mode
            # those hold committed, uncheckpointed transactions and the index every live connection
            # maps; deleting them can lose metadata and raise SQLITE_IOERR.
            if item.name.startswith("metadata.sqlite"):
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

        Corrupt sidecars are skipped; a sidecar with no data file is pruned.
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

                by_table[meta.table_id] = by_table.get(meta.table_id, 0) + 1

                snap_key = f"{meta.table_id}:{meta.snapshot_id}"
                by_snapshot[snap_key] = by_snapshot.get(snap_key, 0) + 1
            except Exception:
                continue

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

        Corrupt sidecars are skipped.
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
        """Evict the oldest entries by mtime when the cache exceeds its limit.

        Write-time order, not LRU (``get`` does not touch mtime). Evicts down to 80% of the limit
        so it does not run on every ``put``.
        """
        current_size = self.get_size_bytes()
        if current_size <= self.max_size_bytes:
            return

        size_before = current_size

        versioned_dir = self.cache_dir / f"v{CACHE_VERSION}"
        if not versioned_dir.exists():
            return
        files = []
        for path in versioned_dir.rglob(f"*{CACHE_FILE_EXTENSION}"):
            files.append((path, path.stat().st_mtime, path.stat().st_size))
        files.sort(key=lambda x: x[1])

        # Target 80% to avoid evicting on every put.
        target_size = int(self.max_size_bytes * 0.8)
        evicted_count = 0
        evicted_bytes = 0
        while current_size > target_size and files:
            path, _, size = files.pop(0)
            path.unlink(missing_ok=True)
            self._meta_path(path).unlink(missing_ok=True)
            current_size -= size
            evicted_count += 1
            evicted_bytes += size

        if evicted_count > 0:
            self.metrics.record_cache_eviction(evicted_count, evicted_bytes)
            tracker = get_eviction_tracker()
            tracker.record_eviction(
                files_evicted=evicted_count,
                bytes_evicted=evicted_bytes,
                cache_size_before=size_before,
                cache_size_after=current_size,
                reason="size_limit",
            )


class _Flight:
    """One storage read of a row group, which concurrent misses on its key share."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.batch: pa.RecordBatch | None = None


class CachedFetcher:
    """A :class:`~strata.fetcher.Fetcher` wrapper that adds transparent caching."""

    def __init__(
        self,
        config: StrataConfig,
        fetcher: Fetcher | None = None,
        cache: Cache | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Compose the fetcher and cache, creating defaults for any omitted."""
        self.config = config
        self.metrics = metrics or MetricsCollector()

        if fetcher is None:
            s3_filesystem = None
            if config.s3_region or config.s3_access_key or config.s3_anonymous:
                s3_filesystem = config.get_s3_filesystem()
            self.fetcher = create_fetcher(
                self.metrics,
                s3_filesystem=s3_filesystem,
                max_equality_delete_rows=config.max_equality_delete_rows,
            )
        else:
            self.fetcher = fetcher

        self.cache = cache or DiskCache(config, self.metrics)
        # Row groups being read from storage right now, by cache key, so concurrent scans that miss
        # on one row group read it once.
        self._flights: dict[CacheKey, _Flight] = {}
        self._flights_lock = threading.Lock()

    @staticmethod
    def _project_batch(batch: pa.RecordBatch, columns: list[str] | None) -> pa.RecordBatch:
        """Return ``batch`` projected to ``columns`` (or unchanged if ``None``)."""
        if columns is None:
            return batch
        if batch.schema.names == columns:
            return batch
        # Index by NAME: ``get_field_index`` returns -1 for an unknown name, and
        # ``batch.column(-1)`` is the last column, so an unknown column would silently hold another
        # column's values.
        return pa.RecordBatch.from_arrays(
            [batch.column(name) for name in columns],
            names=columns,
        )

    def fetch(self, task: Task) -> pa.RecordBatch:
        """Fetch a row group, serving from cache when possible.

        On a miss, caches the full row group and returns the requested projection. Concurrent
        misses on one key share a single storage read. Sets ``task.cached`` and ``task.bytes_read``.
        """
        cached_batch = self.cache.get(task.cache_key)
        if cached_batch is not None:
            return self._serve_hit(task, cached_batch)

        with self._flights_lock:
            in_flight = self._flights.get(task.cache_key)
            if in_flight is None:
                flight = _Flight()
                self._flights[task.cache_key] = flight

        if in_flight is not None:
            in_flight.done.wait()
            if in_flight.batch is not None:
                return self._serve_hit(task, in_flight.batch)
            # The shared read failed. Try again rather than every scan failing
            # on one transient error, and still one read at a time: one waiter
            # leads the retry and the rest wait on it.
            return self.fetch(task)

        try:
            # A read of this key may have landed between our miss and taking
            # the lead; it is in the cache now.
            cached_batch = self.cache.get(task.cache_key)
            if cached_batch is not None:
                flight.batch = cached_batch
                return self._serve_hit(task, cached_batch)
            return self._fetch_from_storage(task, flight)
        finally:
            with self._flights_lock:
                del self._flights[task.cache_key]
            flight.done.set()

    def _serve_hit(self, task: Task, batch: pa.RecordBatch) -> pa.RecordBatch:
        """Answer *task* from a row group already read, by the cache or a peer."""
        result_batch = self._project_batch(batch, task.columns)
        task.cached = True
        task.bytes_read = result_batch.nbytes
        self.metrics.record_fetch(
            bytes_read=result_batch.nbytes,
            rows_read=result_batch.num_rows,
            elapsed_ms=0.0,
            from_cache=True,
        )
        get_cache_histogram().record_hit(
            bytes_accessed=result_batch.nbytes,
            table_id=task.cache_key.table_id,
        )
        return result_batch

    def _fetch_from_storage(self, task: Task, flight: _Flight) -> pa.RecordBatch:
        """Read *task*'s row group from storage, cache it, and hand it to *flight*."""
        histogram = get_cache_histogram()
        cache_full_row_groups = self.config.cache_granularity == CacheGranularity.ROW_GROUP

        with trace_span(
            "fetch_row_group",
            file_path=task.file_path,
            row_group_id=task.row_group_id,
            cache_hit=False,
        ) as span:
            fetch_task = task
            if cache_full_row_groups and task.columns is not None:
                fetch_task = replace(task, columns=None)
            batch = self.fetcher.fetch(fetch_task)
            span.set_attribute("bytes_read", batch.nbytes)
            span.set_attribute("num_rows", batch.num_rows)

        histogram.record_miss(
            bytes_accessed=batch.nbytes,
            table_id=task.cache_key.table_id,
        )

        self.cache.put(task.cache_key, batch)
        flight.batch = batch

        result_batch = self._project_batch(batch, task.columns)
        task.bytes_read = result_batch.nbytes
        return result_batch

    def execute_plan(self, plan: ReadPlan) -> list[pa.RecordBatch]:
        """Execute a read plan and return one batch per task, in plan order."""
        batches = []
        for task in plan.tasks:
            batch = self.fetch(task)
            batches.append(batch)
        return batches

    def stream_plan(self, plan: ReadPlan):
        """Execute a read plan, yielding one batch per task in plan order."""
        for task in plan.tasks:
            yield self.fetch(task)

    def stream_plan_as_ipc(self, plan: ReadPlan):
        """Execute a read plan, yielding each batch as Arrow IPC stream bytes."""
        for task in plan.tasks:
            batch = self.fetch(task)
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, batch.schema)
            writer.write_batch(batch)
            writer.close()
            yield sink.getvalue().to_pybytes()

    def fetch_as_stream_bytes(self, task: Task) -> bytes:
        """Fetch a row group as Arrow IPC stream bytes (the hot path).

        A hit whose cached bytes already match the requested projection is returned without
        parsing; otherwise this fetches, projects and serializes. Sets ``task.cached`` and
        ``task.bytes_read`` (the IPC size).
        """
        histogram = get_cache_histogram()
        cache_full_row_groups = self.config.cache_granularity == CacheGranularity.ROW_GROUP

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
                histogram.record_hit(
                    bytes_accessed=len(stream_bytes),
                    table_id=task.cache_key.table_id,
                )
                return stream_bytes

        batch = self.fetch(task)

        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, batch.schema)
        writer.write_batch(batch)
        writer.close()
        stream_bytes = sink.getvalue().to_pybytes()

        # Report the IPC stream size, overriding the bytes_read that fetch() set.
        task.bytes_read = len(stream_bytes)
        # task.cached was already set by fetch().

        return stream_bytes

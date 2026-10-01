"""Structured metrics logging for Strata."""

import atexit
import json
import math
import queue
import sys
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any, TextIO

# Logs are dropped when the queue is full rather than block.
DEFAULT_LOG_QUEUE_SIZE = 1000

MAX_TRACKED_TABLES = 100

# In milliseconds.
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]


@dataclass
class TableMetrics:
    """Per-table aggregated scan metrics (latencies in milliseconds)."""

    table_id: str
    scan_count: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_from_cache: int = 0
    bytes_from_storage: int = 0
    rows_returned: int = 0
    row_groups_pruned: int = 0

    # Bounded buffer of recent values for percentiles.
    _latencies: list[float] = field(default_factory=list, repr=False)
    _max_latency_samples: int = field(default=1000, repr=False)

    last_access: float = field(default_factory=time.time, repr=False)

    def record_scan(self, metrics: "ScanMetrics") -> None:
        """Fold a completed scan's metrics into this table's aggregates."""
        self.scan_count += 1
        self.total_latency_ms += metrics.total_time_ms
        self.cache_hits += metrics.cache_hits
        self.cache_misses += metrics.cache_misses
        self.bytes_from_cache += metrics.bytes_from_cache
        self.bytes_from_storage += metrics.bytes_from_storage
        self.rows_returned += metrics.rows_returned
        self.row_groups_pruned += metrics.pruned_row_groups
        self.last_access = time.time()

        if len(self._latencies) >= self._max_latency_samples:
            self._latencies.pop(0)
        self._latencies.append(metrics.total_time_ms)

    def get_latency_percentiles(self) -> dict[str, float]:
        """Return ``{p50_ms, p95_ms, p99_ms}`` from the recent-sample buffer.

        Unrounded; zeros when no samples have been recorded.
        """
        if not self._latencies:
            return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = max(0, math.ceil(n * p) - 1)
            return sorted_latencies[idx]

        return {
            "p50_ms": percentile(0.50),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
        }

    def get_latency_histogram(self) -> dict[str, int]:
        """Return counts per ``le_{bucket}ms`` threshold plus ``le_inf``."""
        buckets = {f"le_{b}ms": 0 for b in LATENCY_BUCKETS}
        buckets["le_inf"] = 0

        for latency in self._latencies:
            for bucket in LATENCY_BUCKETS:
                if latency <= bucket:
                    buckets[f"le_{bucket}ms"] += 1
                    break
            else:
                buckets["le_inf"] += 1

        return buckets

    def to_dict(self) -> dict[str, Any]:
        """Return the API-facing projection with derived average latency, hit rate and percentiles.

        The sample buffer and ``last_access`` are omitted; values are unrounded.
        """
        total_requests = self.cache_hits + self.cache_misses
        avg_latency = self.total_latency_ms / self.scan_count if self.scan_count > 0 else 0.0

        return {
            "table_id": self.table_id,
            "scan_count": self.scan_count,
            "avg_latency_ms": avg_latency,
            "cache_hit_rate": (self.cache_hits / total_requests if total_requests > 0 else 0.0),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_from_cache": self.bytes_from_cache,
            "bytes_from_storage": self.bytes_from_storage,
            "rows_returned": self.rows_returned,
            "row_groups_pruned": self.row_groups_pruned,
            **self.get_latency_percentiles(),
        }


@dataclass
class ScanMetrics:
    """Metrics for a single scan operation (timings in milliseconds).

    ``request_id`` is omitted from ``to_dict`` when empty.
    """

    scan_id: str
    snapshot_id: int
    table_id: str = ""
    request_id: str = ""
    planning_time_ms: float = 0.0
    fetch_time_ms: float = 0.0
    total_time_ms: float = 0.0

    cache_hits: int = 0
    cache_misses: int = 0
    bytes_from_cache: int = 0
    bytes_from_storage: int = 0

    total_row_groups: int = 0
    pruned_row_groups: int = 0
    rows_returned: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the API/log projection, with derived ``cache_hit_rate`` (unrounded)."""
        total_requests = self.cache_hits + self.cache_misses
        result = {
            "scan_id": self.scan_id,
            "table_id": self.table_id,
            "snapshot_id": self.snapshot_id,
            "planning_time_ms": self.planning_time_ms,
            "fetch_time_ms": self.fetch_time_ms,
            "total_time_ms": self.total_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_from_cache": self.bytes_from_cache,
            "bytes_from_storage": self.bytes_from_storage,
            "cache_hit_rate": (self.cache_hits / total_requests if total_requests > 0 else 0.0),
            "total_row_groups": self.total_row_groups,
            "pruned_row_groups": self.pruned_row_groups,
            "rows_returned": self.rows_returned,
        }
        if self.request_id:
            result["request_id"] = self.request_id
        return result


@dataclass
class MetricsCollector:
    """Collects metrics and writes structured log entries.

    Logging is non-blocking: entries go through a bounded queue to a background
    thread, and are dropped (counted in ``dropped_logs``) when it is full.
    """

    output: TextIO = field(default_factory=lambda: sys.stdout)
    enabled: bool = True
    log_queue_size: int = DEFAULT_LOG_QUEUE_SIZE

    # Lock only protects aggregate counters, NOT log writing
    _counter_lock: Lock = field(default_factory=Lock, repr=False)

    _log_queue: queue.Queue = field(init=False, repr=False)
    _writer_thread: Thread = field(init=False, repr=False)
    _shutdown: Event = field(default_factory=Event, repr=False)

    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_bytes_from_cache: int = 0
    total_bytes_from_storage: int = 0
    total_bytes_written_to_cache: int = 0
    total_fetches: int = 0
    total_rows_fetched: int = 0
    total_scans: int = 0
    total_row_groups_pruned: int = 0

    stream_aborts_timeout: int = 0
    stream_aborts_size: int = 0
    client_disconnects: int = 0

    cache_evictions_count: int = 0
    cache_evicted_bytes: int = 0

    dropped_logs: int = 0

    _table_metrics: dict[str, TableMetrics] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Create the log queue and start the background writer thread."""
        self._log_queue = queue.Queue(maxsize=self.log_queue_size)
        self._writer_thread = Thread(
            target=self._writer_loop,
            name="MetricsWriter",
            daemon=True,
        )
        self._writer_thread.start()
        atexit.register(self.shutdown)

    def _writer_loop(self) -> None:
        """Drain the queue to ``output`` until shutdown, then flush the rest.

        A write or serialization failure drops that entry; the writer keeps running.
        """
        while not self._shutdown.is_set():
            try:
                # Timeout so the loop can check the shutdown flag.
                entry = self._log_queue.get(timeout=0.1)
                try:
                    json.dump(entry, self.output)
                    self.output.write("\n")
                    self.output.flush()
                except (OSError, TypeError, ValueError):
                    # Broken pipe, closed stream or non-serializable entry: drop it; the writer must
                    # survive one bad log.
                    pass
                finally:
                    self._log_queue.task_done()
            except queue.Empty:
                continue

        while True:
            try:
                entry = self._log_queue.get_nowait()
                try:
                    json.dump(entry, self.output)
                    self.output.write("\n")
                    self.output.flush()
                except (OSError, TypeError, ValueError):
                    pass
                finally:
                    self._log_queue.task_done()
            except queue.Empty:
                break

    def shutdown(self) -> None:
        """Shutdown the background writer thread gracefully."""
        self._shutdown.set()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=1.0)

    def record_fetch(
        self,
        bytes_read: int,
        rows_read: int,
        elapsed_ms: float,
        from_cache: bool,
    ) -> None:
        """Record one fetch's bytes, rows, duration (ms) and cache outcome."""
        with self._counter_lock:
            self.total_fetches += 1
            self.total_rows_fetched += rows_read

            if from_cache:
                self.total_cache_hits += 1
                self.total_bytes_from_cache += bytes_read
            else:
                self.total_cache_misses += 1
                self.total_bytes_from_storage += bytes_read

    def record_cache_write(self, bytes_written: int) -> None:
        """Record bytes written to the cache."""
        with self._counter_lock:
            self.total_bytes_written_to_cache += bytes_written

    def record_stream_abort_timeout(self) -> None:
        """Record a stream abort due to timeout."""
        with self._counter_lock:
            self.stream_aborts_timeout += 1

    def record_stream_abort_size(self) -> None:
        """Record a stream abort due to size limit."""
        with self._counter_lock:
            self.stream_aborts_size += 1

    def record_client_disconnect(self) -> None:
        """Record a client disconnect during streaming."""
        with self._counter_lock:
            self.client_disconnects += 1

    def record_cache_eviction(self, count: int, bytes_evicted: int) -> None:
        """Record ``count`` cache evictions freeing ``bytes_evicted`` bytes."""
        with self._counter_lock:
            self.cache_evictions_count += count
            self.cache_evicted_bytes += bytes_evicted

    def log_scan_complete(self, metrics: ScanMetrics) -> None:
        """Update aggregate and per-table metrics and emit a ``scan_complete`` log."""
        with self._counter_lock:
            self.total_scans += 1
            self.total_row_groups_pruned += metrics.pruned_row_groups

            if metrics.table_id:
                self._record_table_metrics(metrics)

        if not self.enabled:
            return

        log_entry = {
            "event": "scan_complete",
            "timestamp": time.time(),
            **metrics.to_dict(),
        }
        self._write_log(log_entry)

    def _record_table_metrics(self, metrics: ScanMetrics) -> None:
        """Fold a scan into its table's metrics, evicting the LRU table if full.

        Caller must hold ``_counter_lock``.
        """
        table_id = metrics.table_id

        if table_id not in self._table_metrics:
            if len(self._table_metrics) >= MAX_TRACKED_TABLES:
                oldest_table = min(
                    self._table_metrics.keys(),
                    key=lambda t: self._table_metrics[t].last_access,
                )
                del self._table_metrics[oldest_table]

            self._table_metrics[table_id] = TableMetrics(table_id=table_id)

        self._table_metrics[table_id].record_scan(metrics)

    def get_table_metrics(self, table_id: str) -> TableMetrics | None:
        """Return the metrics for ``table_id``, or ``None`` if untracked."""
        with self._counter_lock:
            return self._table_metrics.get(table_id)

    def get_all_table_metrics(self) -> list[dict[str, Any]]:
        """Return every tracked table's ``to_dict``, most scanned first."""
        with self._counter_lock:
            tables = list(self._table_metrics.values())

        tables.sort(key=lambda t: t.scan_count, reverse=True)
        return [t.to_dict() for t in tables]

    def get_top_tables(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the ``limit`` most-scanned tables' projections."""
        all_tables = self.get_all_table_metrics()
        return all_tables[:limit]

    def log_event(self, event: str, **kwargs) -> None:
        """Emit a timestamped log event with extra JSON-serializable fields."""
        if not self.enabled:
            return

        log_entry = {
            "event": event,
            "timestamp": time.time(),
            **kwargs,
        }
        self._write_log(log_entry)

    def _write_log(self, entry: dict) -> None:
        """Queue a log entry for the writer thread, dropping it if the queue is full."""
        try:
            self._log_queue.put_nowait(entry)
        except queue.Full:
            # Drop rather than block.
            with self._counter_lock:
                self.dropped_logs += 1

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Return a snapshot of the lifetime counters, with unrounded ``cache_hit_rate``."""
        with self._counter_lock:
            total_requests = self.total_cache_hits + self.total_cache_misses
            return {
                "scan_count": self.total_scans,
                "total_fetches": self.total_fetches,
                "total_rows_fetched": self.total_rows_fetched,
                "cache_hits": self.total_cache_hits,
                "cache_misses": self.total_cache_misses,
                "cache_hit_rate": (
                    self.total_cache_hits / total_requests if total_requests > 0 else 0.0
                ),
                "bytes_from_cache": self.total_bytes_from_cache,
                "bytes_from_storage": self.total_bytes_from_storage,
                "bytes_written_to_cache": self.total_bytes_written_to_cache,
                "row_groups_pruned": self.total_row_groups_pruned,
                "stream_aborts_timeout": self.stream_aborts_timeout,
                "stream_aborts_size": self.stream_aborts_size,
                "client_disconnects": self.client_disconnects,
                "cache_evictions_count": self.cache_evictions_count,
                "cache_evicted_bytes": self.cache_evicted_bytes,
                "dropped_logs": self.dropped_logs,
            }

    def reset(self) -> None:
        """Reset all counters."""
        with self._counter_lock:
            self.total_cache_hits = 0
            self.total_cache_misses = 0
            self.total_bytes_from_cache = 0
            self.total_bytes_from_storage = 0
            self.total_bytes_written_to_cache = 0
            self.total_fetches = 0
            self.total_rows_fetched = 0
            self.total_scans = 0
            self.total_row_groups_pruned = 0
            self.stream_aborts_timeout = 0
            self.stream_aborts_size = 0
            self.client_disconnects = 0
            self.cache_evictions_count = 0
            self.cache_evicted_bytes = 0
            self.dropped_logs = 0
            self._table_metrics.clear()

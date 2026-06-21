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

# Default queue size - logs are dropped if queue is full to prevent blocking
DEFAULT_LOG_QUEUE_SIZE = 1000

# Max tables to track individually (LRU eviction after this)
MAX_TRACKED_TABLES = 100

# Latency histogram buckets in milliseconds
LATENCY_BUCKETS = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]


@dataclass
class TableMetrics:
    """Per-table aggregated scan metrics.

    Attributes
    ----------
    table_id : str
        Canonical table identity.
    scan_count : int
        Scans recorded against this table.
    total_latency_ms : float
        Sum of scan latencies, in milliseconds.
    cache_hits, cache_misses : int
        Aggregate cache outcomes.
    bytes_from_cache, bytes_from_storage : int
        Aggregate bytes served from each source.
    rows_returned : int
        Total rows returned.
    row_groups_pruned : int
        Total row groups skipped by pruning.
    """

    table_id: str
    scan_count: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_from_cache: int = 0
    bytes_from_storage: int = 0
    rows_returned: int = 0
    row_groups_pruned: int = 0

    # Latency tracking for percentiles (circular buffer of recent values)
    _latencies: list[float] = field(default_factory=list, repr=False)
    _max_latency_samples: int = field(default=1000, repr=False)

    # Last access time for LRU eviction
    last_access: float = field(default_factory=time.time, repr=False)

    def record_scan(self, metrics: "ScanMetrics") -> None:
        """Fold a completed scan's metrics into this table's aggregates.

        Parameters
        ----------
        metrics : ScanMetrics
            The completed scan's metrics.
        """
        self.scan_count += 1
        self.total_latency_ms += metrics.total_time_ms
        self.cache_hits += metrics.cache_hits
        self.cache_misses += metrics.cache_misses
        self.bytes_from_cache += metrics.bytes_from_cache
        self.bytes_from_storage += metrics.bytes_from_storage
        self.rows_returned += metrics.rows_returned
        self.row_groups_pruned += metrics.pruned_row_groups
        self.last_access = time.time()

        # Track latency for percentile calculation
        if len(self._latencies) >= self._max_latency_samples:
            self._latencies.pop(0)
        self._latencies.append(metrics.total_time_ms)

    def get_latency_percentiles(self) -> dict[str, float]:
        """Return p50/p95/p99 latency (ms) from the recent-sample buffer.

        Returns
        -------
        dict
            ``{p50_ms, p95_ms, p99_ms}`` at full precision (zeros when no
            samples have been recorded).
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
        """Return the latency distribution as counts per bucket.

        Returns
        -------
        dict
            Count per ``le_{bucket}ms`` threshold plus ``le_inf``.
        """
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
        """Return the API-facing projection (derived fields, no internals).

        Adds derived ``avg_latency_ms`` / ``cache_hit_rate`` and the latency
        percentiles; the recent-sample buffer and ``last_access`` are omitted.
        Values are full precision; rounding for display is the consumer's
        concern.

        Returns
        -------
        dict
            Counters plus derived rate, average latency, and percentiles.
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
    """Metrics for a single scan operation.

    Attributes
    ----------
    scan_id : str
        Unique id for the scan.
    snapshot_id : int
        Iceberg snapshot scanned.
    table_id : str
        Canonical table identity (``catalog.namespace.table``).
    request_id : str
        Correlation id for request tracing (omitted from ``to_dict`` when empty).
    planning_time_ms, fetch_time_ms, total_time_ms : float
        Phase and total timings, in milliseconds.
    cache_hits, cache_misses : int
        Cache outcomes for the scan.
    bytes_from_cache, bytes_from_storage : int
        Bytes served from each source.
    total_row_groups, pruned_row_groups : int
        Row groups considered and skipped by pruning.
    rows_returned : int
        Rows returned by the scan.
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
        """Return the API/log projection of this scan.

        Adds the derived ``cache_hit_rate`` and includes ``request_id`` only
        when set. Values are full precision; rounding for display is the
        consumer's concern.

        Returns
        -------
        dict
            Scan identity, timings, cache/row-group counters, and hit rate.
        """
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
        # Include request_id only when set (for correlation)
        if self.request_id:
            result["request_id"] = self.request_id
        return result


@dataclass
class MetricsCollector:
    """Collects and logs metrics for Strata operations.

    Logging is non-blocking: log entries are queued and written by a background
    thread. If the queue is full, logs are dropped (not blocked) to prevent
    request latency impact. The dropped_logs counter tracks how many were dropped.
    """

    output: TextIO = field(default_factory=lambda: sys.stdout)
    enabled: bool = True
    log_queue_size: int = DEFAULT_LOG_QUEUE_SIZE

    # Lock only protects aggregate counters, NOT log writing
    _counter_lock: Lock = field(default_factory=Lock, repr=False)

    # Background writer thread and queue (initialized in __post_init__)
    _log_queue: queue.Queue = field(init=False, repr=False)
    _writer_thread: Thread = field(init=False, repr=False)
    _shutdown: Event = field(default_factory=Event, repr=False)

    # Aggregate counters
    total_cache_hits: int = 0
    total_cache_misses: int = 0
    total_bytes_from_cache: int = 0
    total_bytes_from_storage: int = 0
    total_bytes_written_to_cache: int = 0
    total_fetches: int = 0
    total_rows_fetched: int = 0
    total_scans: int = 0
    total_row_groups_pruned: int = 0

    # Stream abort counters
    stream_aborts_timeout: int = 0
    stream_aborts_size: int = 0
    client_disconnects: int = 0

    # Cache eviction counters
    cache_evictions_count: int = 0
    cache_evicted_bytes: int = 0

    # Logging metrics
    dropped_logs: int = 0

    # Per-table metrics (table_id -> TableMetrics)
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
        # Register shutdown handler
        atexit.register(self.shutdown)

    def _writer_loop(self) -> None:
        """Drain the queue to ``output`` until shutdown, then flush the rest.

        A single bad entry never crashes the writer: a write/serialize failure
        (broken pipe, closed stream, or a non-serializable value) drops that
        entry and continues.
        """
        while not self._shutdown.is_set():
            try:
                # Use timeout so we can check shutdown flag periodically
                entry = self._log_queue.get(timeout=0.1)
                try:
                    json.dump(entry, self.output)
                    self.output.write("\n")
                    self.output.flush()
                except (OSError, TypeError, ValueError):
                    # Broken pipe / closed stream / non-serializable entry —
                    # drop it; the writer must survive one bad log.
                    pass
                finally:
                    self._log_queue.task_done()
            except queue.Empty:
                continue

        # Drain remaining items on shutdown
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
        """Record one fetch's outcome into the aggregate counters.

        Parameters
        ----------
        bytes_read : int
            Bytes read for the fetch.
        rows_read : int
            Rows read for the fetch.
        elapsed_ms : float
            Fetch duration in milliseconds.
        from_cache : bool
            Whether the fetch was served from cache.
        """
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
        """Record bytes written to the cache.

        Parameters
        ----------
        bytes_written : int
            Bytes written.
        """
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
        """Record a batch of cache evictions.

        Parameters
        ----------
        count : int
            Number of entries evicted.
        bytes_evicted : int
            Bytes freed.
        """
        with self._counter_lock:
            self.cache_evictions_count += count
            self.cache_evicted_bytes += bytes_evicted

    def log_scan_complete(self, metrics: ScanMetrics) -> None:
        """Update aggregates/per-table metrics and emit a ``scan_complete`` log.

        Parameters
        ----------
        metrics : ScanMetrics
            The completed scan's metrics.
        """
        # Update aggregate counters and per-table metrics
        with self._counter_lock:
            self.total_scans += 1
            self.total_row_groups_pruned += metrics.pruned_row_groups

            # Update per-table metrics
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

        Must be called with ``_counter_lock`` held.

        Parameters
        ----------
        metrics : ScanMetrics
            The completed scan's metrics.
        """
        table_id = metrics.table_id

        if table_id not in self._table_metrics:
            # Check if we need to evict old entries (LRU)
            if len(self._table_metrics) >= MAX_TRACKED_TABLES:
                # Find and remove the least recently accessed table
                oldest_table = min(
                    self._table_metrics.keys(),
                    key=lambda t: self._table_metrics[t].last_access,
                )
                del self._table_metrics[oldest_table]

            self._table_metrics[table_id] = TableMetrics(table_id=table_id)

        self._table_metrics[table_id].record_scan(metrics)

    def get_table_metrics(self, table_id: str) -> TableMetrics | None:
        """Return the metrics for ``table_id``, or ``None`` if untracked.

        Parameters
        ----------
        table_id : str
            Table to look up.

        Returns
        -------
        TableMetrics or None
            The table's metrics, or ``None``.
        """
        with self._counter_lock:
            return self._table_metrics.get(table_id)

    def get_all_table_metrics(self) -> list[dict[str, Any]]:
        """Return every tracked table's projection, hottest first.

        Returns
        -------
        list of dict
            ``TableMetrics.to_dict`` for each table, sorted by scan count
            descending.
        """
        with self._counter_lock:
            tables = list(self._table_metrics.values())

        # Sort by scan count (hottest tables first)
        tables.sort(key=lambda t: t.scan_count, reverse=True)
        return [t.to_dict() for t in tables]

    def get_top_tables(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the ``limit`` most-scanned tables.

        Parameters
        ----------
        limit : int, optional
            Maximum number of tables to return (default 10).

        Returns
        -------
        list of dict
            The hottest tables' projections.
        """
        all_tables = self.get_all_table_metrics()
        return all_tables[:limit]

    def log_event(self, event: str, **kwargs) -> None:
        """Emit a generic timestamped log event.

        Parameters
        ----------
        event : str
            Event name.
        **kwargs
            Additional JSON-serializable fields to include.
        """
        if not self.enabled:
            return

        log_entry = {
            "event": event,
            "timestamp": time.time(),
            **kwargs,
        }
        self._write_log(log_entry)

    def _write_log(self, entry: dict) -> None:
        """Queue a log entry for the writer thread, dropping it if the queue is full.

        Parameters
        ----------
        entry : dict
            The log entry to enqueue.
        """
        try:
            self._log_queue.put_nowait(entry)
        except queue.Full:
            # Drop the log rather than block - increment counter for observability
            with self._counter_lock:
                self.dropped_logs += 1

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Return a snapshot of the aggregate counters.

        The derived ``cache_hit_rate`` is full precision; rounding for display
        is the consumer's concern.

        Returns
        -------
        dict
            Lifetime cache / fetch / stream-abort / eviction / logging counters.
        """
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
                # Stream abort metrics
                "stream_aborts_timeout": self.stream_aborts_timeout,
                "stream_aborts_size": self.stream_aborts_size,
                "client_disconnects": self.client_disconnects,
                # Cache eviction metrics
                "cache_evictions_count": self.cache_evictions_count,
                "cache_evicted_bytes": self.cache_evicted_bytes,
                # Logging metrics
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


class Timer:
    """Context manager that measures wall-clock duration in milliseconds.

    On exit, ``elapsed_ms`` holds the time spent in the ``with`` block.

    Examples
    --------
    >>> with Timer() as t:
    ...     do_work()
    >>> t.elapsed_ms
    """

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        """Start the timer and return self."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop the timer, recording the elapsed milliseconds."""
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000

"""Thread pool and connection metrics for Strata.

This module provides utilities for tracking thread pool utilization and
connection-related metrics to help diagnose performance bottlenecks.

Key metrics:
- Thread pool active workers vs max workers
- Thread pool queue depth (pending tasks)
- Thread pool utilization percentage
- HTTP connection stats (from uvicorn if available)
"""

import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypedDict


class ThreadPoolStatsDict(TypedDict):
    """JSON-serializable thread pool statistics."""

    name: str
    max_workers: int
    active_workers: int
    queue_depth: int
    utilization_pct: float
    tasks_completed: int
    tasks_submitted: int


class PoolMetricsSummaryDict(TypedDict):
    """Summary payload for all registered thread pools."""

    thread_pools: dict[str, ThreadPoolStatsDict]
    total_pools: int


class ConnectionStatsDict(TypedDict):
    """JSON-serializable HTTP connection metrics."""

    active_requests: int
    total_requests: int
    max_concurrent_requests: int
    request_rate_per_sec: float
    uptime_seconds: float
    keepalive_pct: float
    requests_with_keepalive: int
    requests_without_keepalive: int


@dataclass
class ThreadPoolStats:
    """Statistics for a single thread pool."""

    name: str
    max_workers: int
    active_workers: int = 0
    queue_depth: int = 0
    utilization_pct: float = 0.0
    tasks_completed: int = 0
    tasks_submitted: int = 0

    def to_dict(self) -> ThreadPoolStatsDict:
        return {
            "name": self.name,
            "max_workers": self.max_workers,
            "active_workers": self.active_workers,
            "queue_depth": self.queue_depth,
            "utilization_pct": round(self.utilization_pct, 1),
            "tasks_completed": self.tasks_completed,
            "tasks_submitted": self.tasks_submitted,
        }


@dataclass
class PoolMetricsTracker:
    """Tracks metrics for thread pools used by Strata.

    This class provides instrumentation for ThreadPoolExecutor instances,
    tracking utilization, queue depth, and throughput.

    Usage:
        tracker = PoolMetricsTracker()
        tracker.register_pool("planning", planning_executor)
        tracker.register_pool("fetch", fetch_executor)

        # Later, get stats:
        stats = tracker.get_all_stats()
    """

    _pools: dict[str, ThreadPoolExecutor] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Track cumulative task counts
    _tasks_submitted: dict[str, int] = field(default_factory=dict)
    _tasks_completed: dict[str, int] = field(default_factory=dict)

    # Submitted minus completed: the work the pool is carrying right now.
    _in_flight: dict[str, int] = field(default_factory=dict)

    def register_pool(self, name: str, executor: ThreadPoolExecutor) -> None:
        """Register a thread pool for metrics tracking.

        Args:
            name: Human-readable name for the pool (e.g., "planning", "fetch")
            executor: The ThreadPoolExecutor to track
        """
        with self._lock:
            self._pools[name] = executor
            self._tasks_submitted[name] = 0
            self._tasks_completed[name] = 0
            self._in_flight[name] = 0

    def record_task_submitted(self, pool_name: str) -> None:
        """Record that a task was submitted to a pool."""
        with self._lock:
            if pool_name in self._tasks_submitted:
                self._tasks_submitted[pool_name] += 1
                self._in_flight[pool_name] += 1

    def record_task_completed(self, pool_name: str) -> None:
        """Record that a task completed in a pool."""
        with self._lock:
            if pool_name in self._tasks_completed:
                self._tasks_completed[pool_name] += 1
                self._in_flight[pool_name] = max(0, self._in_flight[pool_name] - 1)

    @contextmanager
    def track(self, pool_name: str) -> Iterator[None]:
        """Count one unit of work against ``pool_name`` for the block's life.

        Wrap the ``await loop.run_in_executor(...)`` that hands work to the
        pool. The pair of counters this maintains is what makes
        ``active_workers`` mean "workers busy now"; without a caller it read
        the pool's thread count instead, which only ever grows.

        If the awaiting coroutine is cancelled while its callable is still
        running in a worker, the block exits and stops counting slightly
        early. That under-reports for the length of one task, which is the
        right direction to be wrong in: the failure it replaces was a
        utilization figure that latched high permanently.
        """
        self.record_task_submitted(pool_name)
        try:
            yield
        finally:
            self.record_task_completed(pool_name)

    def get_pool_stats(self, name: str) -> ThreadPoolStats | None:
        """Get statistics for a specific thread pool.

        Args:
            name: Name of the pool to query

        Returns:
            ThreadPoolStats or None if pool not found
        """
        with self._lock:
            executor = self._pools.get(name)
            if executor is None:
                return None

            max_workers = executor._max_workers

            # Workers busy right now, from the tasks this tracker is carrying.
            #
            # This used to count live threads in ``executor._threads``, which
            # is every thread the pool has ever created. Pool threads park on
            # the work queue instead of exiting, so that number never falls:
            # once a pool had seen ``max_workers`` concurrent submissions it
            # reported 100% utilization for the rest of the process's life,
            # and ``check_thread_pools`` (DEGRADED above 90%) latched the whole
            # /health/dependencies report to degraded after a single burst.
            #
            # In-flight includes tasks still queued, so it is capped at
            # ``max_workers`` to keep utilization a percentage; ``queue_depth``
            # reports the backlog separately.
            in_flight = self._in_flight.get(name, 0)
            active_workers = min(in_flight, max_workers)

            # Queue depth from the work queue
            # This is the number of pending tasks waiting for a worker
            queue_depth = executor._work_queue.qsize()

            # Calculate utilization
            utilization_pct = (active_workers / max_workers * 100) if max_workers > 0 else 0.0

            return ThreadPoolStats(
                name=name,
                max_workers=max_workers,
                active_workers=active_workers,
                queue_depth=queue_depth,
                utilization_pct=utilization_pct,
                tasks_submitted=self._tasks_submitted.get(name, 0),
                tasks_completed=self._tasks_completed.get(name, 0),
            )

    def get_all_stats(self) -> dict[str, ThreadPoolStats]:
        """Get statistics for all registered thread pools.

        Returns:
            Dictionary mapping pool names to their stats
        """
        result = {}
        with self._lock:
            pool_names = list(self._pools.keys())

        for name in pool_names:
            stats = self.get_pool_stats(name)
            if stats:
                result[name] = stats

        return result

    def get_summary(self) -> PoolMetricsSummaryDict:
        """Get a summary of all pool metrics for the /metrics endpoint.

        Returns:
            Dictionary with pool metrics suitable for JSON serialization
        """
        stats = self.get_all_stats()
        return {
            "thread_pools": {name: s.to_dict() for name, s in stats.items()},
            "total_pools": len(stats),
        }


@dataclass
class ConnectionMetrics:
    """Tracks HTTP connection-related metrics.

    Since Strata uses FastAPI/Uvicorn, connection management is handled
    by the ASGI server. This class tracks what we can observe:
    - Active requests (from middleware)
    - Request rate
    - Connection reuse hints
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Request tracking
    _active_requests: int = 0
    _total_requests: int = 0
    _max_concurrent_requests: int = 0

    # Timing for rate calculation
    _start_time: float = field(default_factory=time.time)
    _last_request_time: float = 0.0

    # Connection keep-alive tracking
    _requests_with_keepalive: int = 0
    _requests_without_keepalive: int = 0

    def request_started(self, has_keepalive: bool = True) -> None:
        """Record that a request has started."""
        with self._lock:
            self._active_requests += 1
            self._total_requests += 1
            self._last_request_time = time.time()

            if self._active_requests > self._max_concurrent_requests:
                self._max_concurrent_requests = self._active_requests

            if has_keepalive:
                self._requests_with_keepalive += 1
            else:
                self._requests_without_keepalive += 1

    def request_completed(self) -> None:
        """Record that a request has completed."""
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def get_stats(self) -> ConnectionStatsDict:
        """Get connection statistics."""
        with self._lock:
            elapsed = time.time() - self._start_time
            request_rate = self._total_requests / elapsed if elapsed > 0 else 0.0

            total_keepalive = self._requests_with_keepalive + self._requests_without_keepalive
            keepalive_pct = (
                self._requests_with_keepalive / total_keepalive * 100
                if total_keepalive > 0
                else 0.0
            )

            return {
                "active_requests": self._active_requests,
                "total_requests": self._total_requests,
                "max_concurrent_requests": self._max_concurrent_requests,
                "request_rate_per_sec": round(request_rate, 2),
                "uptime_seconds": round(elapsed, 1),
                "keepalive_pct": round(keepalive_pct, 1),
                "requests_with_keepalive": self._requests_with_keepalive,
                "requests_without_keepalive": self._requests_without_keepalive,
            }

    def reset(self) -> None:
        """Reset all counters (for testing)."""
        with self._lock:
            self._active_requests = 0
            self._total_requests = 0
            self._max_concurrent_requests = 0
            self._start_time = time.time()
            self._last_request_time = 0.0
            self._requests_with_keepalive = 0
            self._requests_without_keepalive = 0


# Global instances for server-wide tracking
_pool_tracker: PoolMetricsTracker | None = None
_connection_metrics: ConnectionMetrics | None = None


def get_pool_tracker() -> PoolMetricsTracker:
    """Get or create the global pool metrics tracker."""
    global _pool_tracker
    if _pool_tracker is None:
        _pool_tracker = PoolMetricsTracker()
    return _pool_tracker


def get_connection_metrics() -> ConnectionMetrics:
    """Get or create the global connection metrics tracker."""
    global _connection_metrics
    if _connection_metrics is None:
        _connection_metrics = ConnectionMetrics()
    return _connection_metrics


def reset_metrics() -> None:
    """Reset all metrics (for testing)."""
    global _pool_tracker, _connection_metrics
    _pool_tracker = None
    _connection_metrics = None

"""Cache statistics with time-windowed histograms.

Tracks cache hit/miss rates over configurable time windows,
providing insight into cache effectiveness over time.
"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import TypedDict


class WindowStatsDict(TypedDict):
    """JSON-serializable cache statistics for a single time window."""

    window_seconds: int
    covered_seconds: int
    hits: int
    misses: int
    total: int
    hit_rate: float
    miss_rate: float
    bytes_from_cache: int
    bytes_from_storage: int


class LifetimeStatsDict(TypedDict):
    """JSON-serializable cache lifetime statistics."""

    hits: int
    misses: int
    total: int
    hit_rate: float
    bytes_from_cache: int
    bytes_from_storage: int


class TableStatsDict(TypedDict):
    """JSON-serializable per-table cache statistics."""

    table_id: str
    hits: int
    misses: int
    total: int
    hit_rate: float


class CacheSummaryDict(TypedDict):
    """Top-level cache metrics payload."""

    lifetime: LifetimeStatsDict
    windows: list[WindowStatsDict]
    top_tables: list[TableStatsDict]


@dataclass
class WindowStats:
    """Statistics for a single time window.

    ``covered_seconds`` is how much of ``window_seconds`` the counters
    actually span. It equals ``window_seconds`` for every configured window;
    it is smaller only when a caller asks ``get_window_stats`` for a window
    deeper than the retained history.
    """

    window_seconds: int
    covered_seconds: int
    hits: int
    misses: int
    bytes_from_cache: int
    bytes_from_storage: int

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    @property
    def miss_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.misses / self.total

    def to_dict(self) -> WindowStatsDict:
        return {
            "window_seconds": self.window_seconds,
            "covered_seconds": self.covered_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "miss_rate": round(self.miss_rate, 4),
            "bytes_from_cache": self.bytes_from_cache,
            "bytes_from_storage": self.bytes_from_storage,
        }


class CacheStatsHistogram:
    """Tracks cache statistics over multiple time windows.

    Maintains rolling statistics for configurable time windows
    (e.g., 1 minute, 5 minutes, 1 hour) to show cache hit rate trends.

    Counts are aggregated into one bucket per second, and a window sums the
    buckets it spans. The previous implementation retained the last 10,000
    individual events and answered every window by scanning whatever was still
    in that buffer. One event is recorded per *row group* rather than per
    request (``cache.py``), so the buffer drains in a handful of scans: at a
    steady 33 row groups/sec the "1 hour" window reported 8% of the hour's
    accesses, and the 5-minute and 1-hour windows returned byte-identical
    numbers because both were simply "everything still buffered". The bucket
    ring makes each window exact and its memory a function of the largest
    window rather than of traffic.
    """

    def __init__(
        self,
        windows: list[int] | None = None,
    ) -> None:
        """Initialize the histogram.

        Args:
            windows: List of window sizes in seconds. Default: [60, 300, 3600]
        """
        self.windows = windows or [60, 300, 3600]  # 1m, 5m, 1h
        self._lock = Lock()

        # One bucket per second, indexed by ``epoch_second % depth``. A bucket
        # carries the second it holds so a wrapped-around slot reads as absent
        # rather than as stale counts.
        self._depth = max(self.windows)
        self._bucket_second = [-1] * self._depth
        self._bucket_hits = [0] * self._depth
        self._bucket_misses = [0] * self._depth
        self._bucket_bytes_cache = [0] * self._depth
        self._bucket_bytes_storage = [0] * self._depth

        # Lifetime counters
        self._total_hits = 0
        self._total_misses = 0
        self._total_bytes_cache = 0
        self._total_bytes_storage = 0

        # Per-table stats (table_id -> {hits, misses})
        self._table_stats: dict[str, dict[str, int]] = {}

    def _record(
        self,
        *,
        is_hit: bool,
        bytes_accessed: int,
        table_id: str | None,
    ) -> None:
        """Fold one access into the current second's bucket and the totals."""
        second = int(time.time())
        with self._lock:
            slot = second % self._depth
            if self._bucket_second[slot] != second:
                self._bucket_second[slot] = second
                self._bucket_hits[slot] = 0
                self._bucket_misses[slot] = 0
                self._bucket_bytes_cache[slot] = 0
                self._bucket_bytes_storage[slot] = 0

            if is_hit:
                self._bucket_hits[slot] += 1
                self._bucket_bytes_cache[slot] += bytes_accessed
                self._total_hits += 1
                self._total_bytes_cache += bytes_accessed
            else:
                self._bucket_misses[slot] += 1
                self._bucket_bytes_storage[slot] += bytes_accessed
                self._total_misses += 1
                self._total_bytes_storage += bytes_accessed

            if table_id:
                if table_id not in self._table_stats:
                    self._table_stats[table_id] = {"hits": 0, "misses": 0}
                self._table_stats[table_id]["hits" if is_hit else "misses"] += 1

    def record_hit(
        self,
        bytes_accessed: int,
        table_id: str | None = None,
    ) -> None:
        """Record a cache hit."""
        self._record(is_hit=True, bytes_accessed=bytes_accessed, table_id=table_id)

    def record_miss(
        self,
        bytes_accessed: int,
        table_id: str | None = None,
    ) -> None:
        """Record a cache miss."""
        self._record(is_hit=False, bytes_accessed=bytes_accessed, table_id=table_id)

    def get_window_stats(self, window_seconds: int) -> WindowStats:
        """Get statistics for a specific time window.

        A window deeper than the retained history is answered with what is
        retained, and ``covered_seconds`` says how much that was.
        """
        now_second = int(time.time())
        covered = min(window_seconds, self._depth)

        hits = 0
        misses = 0
        bytes_cache = 0
        bytes_storage = 0

        with self._lock:
            for offset in range(covered):
                second = now_second - offset
                slot = second % self._depth
                if self._bucket_second[slot] != second:
                    continue
                hits += self._bucket_hits[slot]
                misses += self._bucket_misses[slot]
                bytes_cache += self._bucket_bytes_cache[slot]
                bytes_storage += self._bucket_bytes_storage[slot]

        return WindowStats(
            window_seconds=window_seconds,
            covered_seconds=covered,
            hits=hits,
            misses=misses,
            bytes_from_cache=bytes_cache,
            bytes_from_storage=bytes_storage,
        )

    def get_all_window_stats(self) -> list[WindowStats]:
        """Get statistics for all configured windows."""
        return [self.get_window_stats(w) for w in self.windows]

    def get_lifetime_stats(self) -> LifetimeStatsDict:
        """Get lifetime statistics."""
        with self._lock:
            total = self._total_hits + self._total_misses
            hit_rate = self._total_hits / total if total > 0 else 0.0
            return {
                "hits": self._total_hits,
                "misses": self._total_misses,
                "total": total,
                "hit_rate": round(hit_rate, 4),
                "bytes_from_cache": self._total_bytes_cache,
                "bytes_from_storage": self._total_bytes_storage,
            }

    def get_table_stats(self, limit: int = 10) -> list[TableStatsDict]:
        """Get per-table statistics, sorted by total accesses."""
        with self._lock:
            table_list: list[TableStatsDict] = []
            for table_id, stats in self._table_stats.items():
                total = stats["hits"] + stats["misses"]
                hit_rate = stats["hits"] / total if total > 0 else 0.0
                table_list.append(
                    {
                        "table_id": table_id,
                        "hits": stats["hits"],
                        "misses": stats["misses"],
                        "total": total,
                        "hit_rate": round(hit_rate, 4),
                    }
                )

        # Sort by total accesses descending
        table_list.sort(key=lambda stats: stats["total"], reverse=True)
        return table_list[:limit]

    def get_summary(self) -> CacheSummaryDict:
        """Get a comprehensive summary of cache statistics."""
        return {
            "lifetime": self.get_lifetime_stats(),
            "windows": [w.to_dict() for w in self.get_all_window_stats()],
            "top_tables": self.get_table_stats(limit=5),
        }

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._bucket_second = [-1] * self._depth
            self._bucket_hits = [0] * self._depth
            self._bucket_misses = [0] * self._depth
            self._bucket_bytes_cache = [0] * self._depth
            self._bucket_bytes_storage = [0] * self._depth
            self._total_hits = 0
            self._total_misses = 0
            self._total_bytes_cache = 0
            self._total_bytes_storage = 0
            self._table_stats.clear()


# Global histogram instance
_cache_histogram: CacheStatsHistogram | None = None


def get_cache_histogram() -> CacheStatsHistogram:
    """Get the global cache statistics histogram."""
    global _cache_histogram
    if _cache_histogram is None:
        _cache_histogram = CacheStatsHistogram()
    return _cache_histogram


def reset_cache_histogram() -> None:
    """Reset the global cache histogram (for testing)."""
    global _cache_histogram
    _cache_histogram = None

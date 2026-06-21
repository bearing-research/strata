"""Cache eviction metrics: event tracking, rates, and pressure level."""

import time
from collections import deque
from dataclasses import asdict, dataclass
from enum import StrEnum
from threading import Lock
from typing import Any


class EvictionPressure(StrEnum):
    """Cache-eviction load band derived from the per-minute eviction rate."""

    LOW = "low"  # < 1 eviction per minute
    MEDIUM = "medium"  # 1–5 evictions per minute
    HIGH = "high"  # 5–10 evictions per minute
    CRITICAL = "critical"  # 10+ evictions per minute


@dataclass
class EvictionEvent:
    """A single cache eviction.

    Attributes
    ----------
    timestamp : float
        Unix timestamp when the eviction happened.
    files_evicted : int
        Number of files removed.
    bytes_evicted : int
        Total bytes freed.
    cache_size_before : int
        Cache size in bytes immediately before the eviction.
    cache_size_after : int
        Cache size in bytes immediately after the eviction.
    reason : str
        Why the eviction ran: ``"size_limit"``, ``"manual"``, or ``"ttl"``.
    """

    timestamp: float
    files_evicted: int
    bytes_evicted: int
    cache_size_before: int
    cache_size_after: int
    reason: str = "size_limit"


@dataclass
class EvictionStats:
    """Aggregate eviction statistics over the tracked window.

    Attributes
    ----------
    total_evictions : int
        Lifetime eviction count.
    total_files_evicted : int
        Lifetime files removed.
    total_bytes_evicted : int
        Lifetime bytes freed.
    evictions_last_minute : int
        Evictions in the last 60 seconds.
    evictions_last_hour : int
        Evictions in the last hour.
    bytes_evicted_last_minute : int
        Bytes freed in the last 60 seconds.
    bytes_evicted_last_hour : int
        Bytes freed in the last hour.
    eviction_rate_per_minute : float
        Evictions per minute, averaged over the last hour.
    last_eviction_at : float or None
        Timestamp of the most recent eviction, or ``None`` if none recorded.
    pressure_level : EvictionPressure
        Derived load band (``low`` / ``medium`` / ``high`` / ``critical``).
    """

    total_evictions: int
    total_files_evicted: int
    total_bytes_evicted: int
    evictions_last_minute: int
    evictions_last_hour: int
    bytes_evicted_last_minute: int
    bytes_evicted_last_hour: int
    eviction_rate_per_minute: float
    last_eviction_at: float | None
    pressure_level: EvictionPressure


class CacheEvictionTracker:
    """Records cache eviction events and computes aggregate metrics."""

    def __init__(self, max_events: int = 1000) -> None:
        """Initialize the tracker.

        Parameters
        ----------
        max_events : int, optional
            Maximum recent events retained for rate/window calculations
            (default 1000). Older events are dropped.
        """
        self._lock = Lock()
        self._events: deque[EvictionEvent] = deque(maxlen=max_events)
        self._total_evictions = 0
        self._total_files_evicted = 0
        self._total_bytes_evicted = 0

    def record_eviction(
        self,
        files_evicted: int,
        bytes_evicted: int,
        cache_size_before: int,
        cache_size_after: int,
        reason: str = "size_limit",
    ) -> None:
        """Record one eviction event and update the lifetime totals.

        Parameters
        ----------
        files_evicted : int
            Number of files removed.
        bytes_evicted : int
            Total bytes freed.
        cache_size_before : int
            Cache size in bytes before the eviction.
        cache_size_after : int
            Cache size in bytes after the eviction.
        reason : str, optional
            Why the eviction ran (default ``"size_limit"``).
        """
        event = EvictionEvent(
            timestamp=time.time(),
            files_evicted=files_evicted,
            bytes_evicted=bytes_evicted,
            cache_size_before=cache_size_before,
            cache_size_after=cache_size_after,
            reason=reason,
        )
        with self._lock:
            self._events.append(event)
            self._total_evictions += 1
            self._total_files_evicted += files_evicted
            self._total_bytes_evicted += bytes_evicted

    def get_stats(self) -> EvictionStats:
        """Compute aggregate statistics over the retained events.

        Returns
        -------
        EvictionStats
            Lifetime totals, last-minute and last-hour counts, the
            per-minute rate, and the derived pressure level.
        """
        now = time.time()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with self._lock:
            events = list(self._events)

        # Count recent evictions
        evictions_minute = 0
        evictions_hour = 0
        bytes_minute = 0
        bytes_hour = 0
        last_eviction = None

        for event in events:
            if event.timestamp >= one_minute_ago:
                evictions_minute += 1
                bytes_minute += event.bytes_evicted
            if event.timestamp >= one_hour_ago:
                evictions_hour += 1
                bytes_hour += event.bytes_evicted
            if last_eviction is None or event.timestamp > last_eviction:
                last_eviction = event.timestamp

        # Calculate rate (evictions per minute over last hour)
        rate = evictions_hour / 60.0 if evictions_hour > 0 else 0.0

        # Determine pressure level based on eviction rate
        if rate >= 10:
            pressure = EvictionPressure.CRITICAL
        elif rate >= 5:
            pressure = EvictionPressure.HIGH
        elif rate >= 1:
            pressure = EvictionPressure.MEDIUM
        else:
            pressure = EvictionPressure.LOW

        return EvictionStats(
            total_evictions=self._total_evictions,
            total_files_evicted=self._total_files_evicted,
            total_bytes_evicted=self._total_bytes_evicted,
            evictions_last_minute=evictions_minute,
            evictions_last_hour=evictions_hour,
            bytes_evicted_last_minute=bytes_minute,
            bytes_evicted_last_hour=bytes_hour,
            eviction_rate_per_minute=rate,
            last_eviction_at=last_eviction,
            pressure_level=pressure,
        )

    def get_recent_events(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return the most recent eviction events, newest first.

        Parameters
        ----------
        limit : int, optional
            Maximum number of events to return (default 10).

        Returns
        -------
        list of dict
            One mapping per event (all ``EvictionEvent`` fields).
        """
        with self._lock:
            events = list(self._events)

        events = events[-limit:][::-1]
        return [asdict(e) for e in events]

    def reset(self) -> None:
        """Clear all events and lifetime totals."""
        with self._lock:
            self._events.clear()
            self._total_evictions = 0
            self._total_files_evicted = 0
            self._total_bytes_evicted = 0


# Global tracker instance
_eviction_tracker: CacheEvictionTracker | None = None


def get_eviction_tracker() -> CacheEvictionTracker:
    """Return the process-wide eviction tracker, creating it on first use.

    Returns
    -------
    CacheEvictionTracker
        The shared tracker instance.
    """
    global _eviction_tracker
    if _eviction_tracker is None:
        _eviction_tracker = CacheEvictionTracker()
    return _eviction_tracker


def reset_eviction_tracker() -> None:
    """Drop the process-wide eviction tracker (for testing)."""
    global _eviction_tracker
    _eviction_tracker = None

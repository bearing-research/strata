"""Cache eviction metrics: event tracking, rates, and pressure level."""

import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from threading import Lock
from typing import Any


class EvictionPressure(StrEnum):
    """Cache-eviction load band derived from the per-minute eviction rate.

    The rate is the worse of the last minute and the mean so far, so a burst
    reaches its band as it happens rather than being averaged away.
    """

    LOW = "low"  # < 1 eviction per minute
    MEDIUM = "medium"  # 1–5 evictions per minute
    HIGH = "high"  # 5–10 evictions per minute
    CRITICAL = "critical"  # 10+ evictions per minute


@dataclass
class EvictionEvent:
    """A single cache eviction; sizes in bytes, ``timestamp`` in Unix seconds.

    ``reason`` is ``"size_limit"``, ``"manual"``, or ``"ttl"``.
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

    ``eviction_rate_per_minute`` is the worse of the last minute and the mean over
    the span observed so far (capped at an hour): the mean alone cannot tell a
    short thrash from a steady trickle. ``last_eviction_at`` is ``None`` if none.
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

    def __init__(self, max_events: int = 1000, clock: Callable[[], float] = time.time) -> None:
        """Initialize the tracker.

        Keeps at most ``max_events`` recent events; ``clock`` is injectable so tests
        can place events in a window without sleeping.
        """
        self._clock = clock
        self._lock = Lock()
        self._events: deque[EvictionEvent] = deque(maxlen=max_events)
        # When rate measurement began, so a young server's hourly rate is not divided by an hour it
        # has not lived.
        self._started_at = clock()
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
        """Record one eviction event and update the lifetime totals."""
        event = EvictionEvent(
            timestamp=self._clock(),
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
        """Compute lifetime totals, minute and hour windows, the rate and the pressure level."""
        now = self._clock()
        one_minute_ago = now - 60
        one_hour_ago = now - 3600

        with self._lock:
            events = list(self._events)

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

        # Per minute over the hour actually observed, not a constant 60, so a young server that
        # evicts fast reads at its real band. Floored at a minute so the first seconds cannot
        # extrapolate one sweep into a crisis.
        observed_seconds = min(max(now - self._started_at, 0.0), 3600.0)
        observed_minutes = max(observed_seconds / 60.0, 1.0)
        hourly_rate = evictions_hour / observed_minutes if evictions_hour > 0 else 0.0

        # The worse of the hour and the last minute: an hourly mean cannot tell a five-minute thrash
        # from a steady trickle. The hour keeps the band raised after a burst, so pressure is quick
        # to fire and slow to clear.
        rate = max(hourly_rate, float(evictions_minute))

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
        """Return up to ``limit`` recent eviction events as dicts, newest first."""
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


_eviction_tracker: CacheEvictionTracker | None = None


def get_eviction_tracker() -> CacheEvictionTracker:
    """Return the process-wide eviction tracker, creating it on first use."""
    global _eviction_tracker
    if _eviction_tracker is None:
        _eviction_tracker = CacheEvictionTracker()
    return _eviction_tracker


def reset_eviction_tracker() -> None:
    """Drop the process-wide eviction tracker (for testing)."""
    global _eviction_tracker
    _eviction_tracker = None

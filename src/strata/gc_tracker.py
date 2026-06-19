"""GC pause duration tracking using gc.callbacks.

This module provides precise measurement of Python garbage collection pause times,
which is critical for understanding latency stalls in the server.

Usage:
    from strata.gc_tracker import install_gc_tracker, get_gc_stats, reset_gc_stats

    # At server startup
    install_gc_tracker()

    # In /metrics endpoint
    stats = get_gc_stats()

The tracker records:
- Pause duration for each GC generation (0, 1, 2)
- Timestamp of each GC event
- Running statistics (min, max, p50, p95, p99)

This differs from gc.get_stats() which only counts collections, not pause duration.
"""

import gc
import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GCPause:
    """A single GC pause event."""

    timestamp: float  # Unix timestamp when GC completed
    generation: int  # 0, 1, or 2
    duration_ms: float  # Pause duration in milliseconds


@dataclass
class GCStats:
    """Aggregated GC statistics."""

    # Recent pauses (last N events)
    recent_pauses: list[GCPause] = field(default_factory=list)

    # Per-generation stats
    gen0_count: int = 0
    gen0_total_ms: float = 0.0
    gen0_max_ms: float = 0.0

    gen1_count: int = 0
    gen1_total_ms: float = 0.0
    gen1_max_ms: float = 0.0

    gen2_count: int = 0
    gen2_total_ms: float = 0.0
    gen2_max_ms: float = 0.0

    # Overall stats
    total_pauses: int = 0
    total_pause_ms: float = 0.0
    max_pause_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "total_pauses": self.total_pauses,
            "total_pause_ms": round(self.total_pause_ms, 3),
            "max_pause_ms": round(self.max_pause_ms, 3),
            "gen0": {
                "count": self.gen0_count,
                "total_ms": round(self.gen0_total_ms, 3),
                "max_ms": round(self.gen0_max_ms, 3),
                "avg_ms": round(self.gen0_total_ms / self.gen0_count, 3) if self.gen0_count else 0,
            },
            "gen1": {
                "count": self.gen1_count,
                "total_ms": round(self.gen1_total_ms, 3),
                "max_ms": round(self.gen1_max_ms, 3),
                "avg_ms": round(self.gen1_total_ms / self.gen1_count, 3) if self.gen1_count else 0,
            },
            "gen2": {
                "count": self.gen2_count,
                "total_ms": round(self.gen2_total_ms, 3),
                "max_ms": round(self.gen2_max_ms, 3),
                "avg_ms": round(self.gen2_total_ms / self.gen2_count, 3) if self.gen2_count else 0,
            },
        }

        # Calculate percentiles from recent pauses if we have enough data
        if len(self.recent_pauses) >= 10:
            durations = sorted(p.duration_ms for p in self.recent_pauses)
            n = len(durations)
            result["recent"] = {
                "count": n,
                "p50_ms": round(durations[n // 2], 3),
                "p95_ms": round(durations[int(n * 0.95)], 3),
                "p99_ms": round(durations[int(n * 0.99)], 3),
            }

        return result


class GCTracker:
    """Tracks GC pause durations using gc.callbacks."""

    def __init__(self, max_recent: int = 1000):
        """Initialize the tracker.

        Args:
            max_recent: Maximum number of recent pauses to keep for percentile calculation.
        """
        self._lock = threading.Lock()
        self._gc_start_time: float = 0.0
        self._current_generation: int = 0
        self._max_recent = max_recent

        # Recent pauses (bounded deque for memory safety)
        self._recent: deque[GCPause] = deque(maxlen=max_recent)

        # Per-generation counters
        self._gen_counts = [0, 0, 0]
        self._gen_total_ms = [0.0, 0.0, 0.0]
        self._gen_max_ms = [0.0, 0.0, 0.0]

        # Overall counters
        self._total_pauses = 0
        self._total_pause_ms = 0.0
        self._max_pause_ms = 0.0

        self._installed = False

    def _gc_callback(self, phase: str, info: dict) -> None:
        """Callback invoked by the GC on start/stop of collection.

        Args:
            phase: "start" or "stop"
            info: Dict with "generation" key (0, 1, or 2)
        """
        if phase == "start":
            self._gc_start_time = time.perf_counter()
            self._current_generation = info.get("generation", 0)
        elif phase == "stop":
            if self._gc_start_time == 0:
                return  # Missed the start, skip

            duration_ms = (time.perf_counter() - self._gc_start_time) * 1000
            generation = self._current_generation
            timestamp = time.time()

            # Reset for next collection
            self._gc_start_time = 0.0

            # Best-effort, NON-BLOCKING update. This callback fires *during* a GC
            # — which can be triggered by an allocation made while a thread holds
            # ``_lock`` (e.g. ``get_stats``/``reset`` building their result). With
            # a non-reentrant lock a blocking acquire here would self-deadlock
            # against that holder on the same thread (observed as an intermittent
            # py3.12/macOS hang on the ``/metrics`` endpoint). Skip recording this
            # pause when the lock is contended — dropping the occasional sample is
            # acceptable for best-effort GC observability; a deadlock is not.
            if not self._lock.acquire(blocking=False):
                return
            try:
                pause = GCPause(
                    timestamp=timestamp,
                    generation=generation,
                    duration_ms=duration_ms,
                )
                self._recent.append(pause)

                # Update per-generation stats
                if 0 <= generation <= 2:
                    self._gen_counts[generation] += 1
                    self._gen_total_ms[generation] += duration_ms
                    self._gen_max_ms[generation] = max(self._gen_max_ms[generation], duration_ms)

                # Update overall stats
                self._total_pauses += 1
                self._total_pause_ms += duration_ms
                self._max_pause_ms = max(self._max_pause_ms, duration_ms)
            finally:
                self._lock.release()

    def install(self) -> None:
        """Install the GC callback. Safe to call multiple times."""
        if self._installed:
            return

        gc.callbacks.append(self._gc_callback)
        self._installed = True

    def uninstall(self) -> None:
        """Remove the GC callback."""
        if not self._installed:
            return

        try:
            gc.callbacks.remove(self._gc_callback)
        except ValueError:
            pass  # Already removed
        self._installed = False

    def get_stats(self) -> GCStats:
        """Get current GC statistics."""
        with self._lock:
            return GCStats(
                recent_pauses=list(self._recent),
                gen0_count=self._gen_counts[0],
                gen0_total_ms=self._gen_total_ms[0],
                gen0_max_ms=self._gen_max_ms[0],
                gen1_count=self._gen_counts[1],
                gen1_total_ms=self._gen_total_ms[1],
                gen1_max_ms=self._gen_max_ms[1],
                gen2_count=self._gen_counts[2],
                gen2_total_ms=self._gen_total_ms[2],
                gen2_max_ms=self._gen_max_ms[2],
                total_pauses=self._total_pauses,
                total_pause_ms=self._total_pause_ms,
                max_pause_ms=self._max_pause_ms,
            )

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._recent.clear()
            self._gen_counts = [0, 0, 0]
            self._gen_total_ms = [0.0, 0.0, 0.0]
            self._gen_max_ms = [0.0, 0.0, 0.0]
            self._total_pauses = 0
            self._total_pause_ms = 0.0
            self._max_pause_ms = 0.0

    def get_recent_pauses(self, limit: int = 100) -> list[dict]:
        """Get recent GC pauses for detailed analysis.

        Args:
            limit: Maximum number of pauses to return (most recent first).

        Returns:
            List of pause dictionaries with timestamp, generation, duration_ms.
        """
        with self._lock:
            pauses = list(self._recent)[-limit:]
            return [
                {
                    "timestamp": p.timestamp,
                    "generation": p.generation,
                    "duration_ms": round(p.duration_ms, 3),
                }
                for p in reversed(pauses)  # Most recent first
            ]


# Global tracker instance
_tracker: GCTracker | None = None


def install_gc_tracker(max_recent: int = 1000) -> GCTracker:
    """Install the global GC tracker.

    Args:
        max_recent: Maximum recent pauses to keep for percentile calculation.

    Returns:
        The GCTracker instance.
    """
    global _tracker
    if _tracker is None:
        _tracker = GCTracker(max_recent=max_recent)
    _tracker.install()
    return _tracker


def get_gc_tracker() -> GCTracker | None:
    """Get the global GC tracker instance, or None if not installed."""
    return _tracker


def get_gc_stats() -> dict:
    """Get GC statistics as a dictionary.

    Returns empty dict if tracker not installed.
    """
    if _tracker is None:
        return {}
    return _tracker.get_stats().to_dict()


def get_recent_gc_pauses(limit: int = 100) -> list[dict]:
    """Get recent GC pauses for detailed analysis.

    Returns empty list if tracker not installed.
    """
    if _tracker is None:
        return []
    return _tracker.get_recent_pauses(limit)


def reset_gc_stats() -> None:
    """Reset GC statistics."""
    if _tracker is not None:
        _tracker.reset()

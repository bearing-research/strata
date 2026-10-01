"""Adaptive concurrency control for Strata QoS.

Resizes the interactive and bulk tier limiters from p95 latency and queue-wait
signals fed by ``strata.streaming.qos.QoSAdmission``; without those feeds it never
adjusts. It steers only the ``_default`` tenant's limiters, which is why
``adaptive_enabled`` is rejected at startup alongside multi-tenancy.
"""

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

logger = logging.getLogger("strata.adaptive")


class ResizableLimiter:
    """Concurrency limiter whose capacity can change while slots are held.

    Unlike asyncio.Semaphore it tracks capacity and in-use separately: a larger
    capacity admits waiters at once, a smaller one takes effect as holders release.
    """

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._in_use = 0
        self._pending = 0  # acquire() calls not yet returned
        self._lock = asyncio.Lock()
        self._cv = asyncio.Condition(self._lock)

    @property
    def capacity(self) -> int:
        """Current capacity (max concurrent requests)."""
        return self._capacity

    @property
    def in_use(self) -> int:
        """Current number of active requests."""
        return self._in_use

    @property
    def available(self) -> int:
        """Number of available slots."""
        return max(0, self._capacity - self._in_use)

    @property
    def idle(self) -> bool:
        """No slot is held and no acquire() is waiting for one."""
        return self._in_use == 0 and self._pending == 0

    async def acquire(self, timeout: float | None = None) -> bool:
        """Acquire a slot, waiting up to ``timeout`` seconds (``None``: forever).

        Return ``False`` if the timeout expired.
        """
        self._pending += 1
        try:
            return await self._acquire(timeout)
        finally:
            self._pending -= 1

    async def _acquire(self, timeout: float | None) -> bool:
        async with self._cv:
            try:
                if timeout is None:
                    while self._in_use >= self._capacity:
                        await self._cv.wait()
                    self._in_use += 1
                    return True

                loop = asyncio.get_running_loop()
                end = loop.time() + timeout
                while self._in_use >= self._capacity:
                    remaining = end - loop.time()
                    if remaining <= 0:
                        return False
                    try:
                        await asyncio.wait_for(self._cv.wait(), timeout=remaining)
                    except TimeoutError:
                        # A notify may have landed just before the timeout.
                        if self._in_use >= self._capacity:
                            return False
                self._in_use += 1
                return True
            except BaseException:
                # Before 3.13, asyncio.Condition drops a notify whose waiter is
                # cancelled before it runs. Pass it on so a free slot isn't left
                # idle; a spurious wakeup is harmless since waiters re-check.
                self._cv.notify(1)
                raise

    async def release(self) -> None:
        """Release a slot, waking one waiting acquirer."""
        async with self._cv:
            if self._in_use <= 0:
                raise RuntimeError("release() called without matching acquire()")
            self._in_use -= 1
            self._cv.notify(1)

    async def resize(self, new_capacity: int) -> None:
        """Set the capacity; a decrease takes effect as active requests complete.

        Raises
        ------
        ValueError
            If ``new_capacity`` is less than 1.
        """
        if new_capacity < 1:
            raise ValueError("capacity must be >= 1")
        async with self._cv:
            old_capacity = self._capacity
            self._capacity = new_capacity
            if new_capacity > old_capacity:
                self._cv.notify_all()

    def get_stats(self) -> dict[str, int]:
        """Return ``{capacity, in_use, available}`` without awaiting (for metrics)."""
        return {
            "capacity": self._capacity,
            "in_use": self._in_use,
            "available": max(0, self._capacity - self._in_use),
        }


@dataclass
class AdaptiveConfig:
    """Configuration for the adaptive concurrency controller (opt-in).

    Latencies are in milliseconds, intervals and ages in seconds. ``hysteresis_count``
    is the number of consecutive signals required before adjusting.
    ``sample_max_age_seconds`` expires old samples so an idle period after a slow
    burst does not keep walking a tier down to ``min_slots``.
    """

    enabled: bool = False
    adjustment_interval_seconds: float = 5.0
    latency_target_p95_ms: float = 500.0
    queue_wait_threshold_ms: float = 100.0  # 100ms queue wait = pressure
    min_slots_interactive: int = 4
    max_slots_interactive: int = 64
    min_slots_bulk: int = 2
    max_slots_bulk: int = 32
    increase_step: int = 1
    decrease_step: int = 1
    hysteresis_count: int = 3
    window_size: int = 100  # Keep last 100 samples for p95
    sample_max_age_seconds: float = 60.0  # Ignore samples older than this


class RollingLatencyWindow:
    """Thread-safe rolling latency window for percentiles.

    Bounded by count and, when ``max_age_seconds`` is set, by age: a count-only
    window would keep adjusting an idle tier off the last burst it saw.
    """

    def __init__(
        self,
        size: int = 100,
        max_age_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Create the window; ``clock`` is injectable so tests can age samples without sleeping."""
        self._size = size
        self._max_age_seconds = max_age_seconds
        self._clock = clock
        self._lock = Lock()
        # (monotonic timestamp, latency_ms); monotonic so a clock step can't
        # make a sample look infinitely old or fresh.
        self._samples: deque[tuple[float, float]] = deque(maxlen=size)
        self._count = 0  # Total samples seen (for metrics)

    def record(self, latency_ms: float) -> None:
        """Record a latency observation."""
        with self._lock:
            self._samples.append((self._clock(), latency_ms))
            self._count += 1

    def _live_values(self) -> list[float]:
        """Sorted latencies still inside the age bound. Caller holds no lock."""
        with self._lock:
            if self._max_age_seconds is None:
                return sorted(value for _, value in self._samples)
            cutoff = self._clock() - self._max_age_seconds
            # Samples are appended in time order, so expiry is a prefix drop.
            while self._samples and self._samples[0][0] < cutoff:
                self._samples.popleft()
            return sorted(value for _, value in self._samples)

    def get_p95(self) -> float | None:
        """Return the p95 latency in ms, or ``None`` with fewer than 10 live samples."""
        sorted_samples = self._live_values()
        if len(sorted_samples) < 10:
            return None

        n = len(sorted_samples)
        idx = max(0, math.ceil(n * 0.95) - 1)
        return sorted_samples[idx]

    def get_stats(self) -> dict[str, Any]:
        """Return count, window size, p50/p95/p99 and min/max/avg in ms, unrounded.

        Latency values are ``None`` when the window is empty.
        """
        sorted_samples = self._live_values()
        if not sorted_samples:
            return {
                "count": self._count,
                "window_size": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "min_ms": None,
                "max_ms": None,
                "avg_ms": None,
            }
        count = len(sorted_samples)

        def pct(p: float) -> float:
            idx = max(0, math.ceil(count * p) - 1)
            return sorted_samples[idx]

        return {
            "count": self._count,
            "window_size": count,
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "min_ms": sorted_samples[0],
            "max_ms": sorted_samples[-1],
            "avg_ms": sum(sorted_samples) / count,
        }

    def reset(self) -> None:
        """Reset the window."""
        with self._lock:
            self._samples.clear()
            self._count = 0


@dataclass
class TierState:
    """Control-loop state for one tier (interactive or bulk)."""

    name: str
    current_slots: int
    min_slots: int
    max_slots: int
    latency_window: RollingLatencyWindow = field(default_factory=RollingLatencyWindow)
    queue_wait_window: RollingLatencyWindow = field(default_factory=RollingLatencyWindow)

    # Positive = increase signals, negative = decrease signals.
    consecutive_increase_signals: int = 0
    consecutive_decrease_signals: int = 0

    last_adjustment_time: float = 0.0
    last_adjustment_direction: str = ""  # "increase", "decrease", ""
    last_p95_ms: float | None = None
    last_queue_wait_p95_ms: float | None = None

    # Event counts, not slot counts.
    increase_events: int = 0
    decrease_events: int = 0


class AdaptiveConcurrencyController:
    """Adjusts tier slot counts from latency and queue-wait signals.

    Per tier: p95 over target signals a decrease; p95 under 80% of target with
    queue-wait p95 over threshold signals an increase; anything else resets the
    signals. Slots change only after ``hysteresis_count`` consecutive signals.
    """

    def __init__(
        self,
        config: AdaptiveConfig,
        interactive_limiter: ResizableLimiter,
        bulk_limiter: ResizableLimiter,
    ):
        self.config = config
        self._interactive_limiter = interactive_limiter
        self._bulk_limiter = bulk_limiter

        max_age = config.sample_max_age_seconds
        self._interactive = TierState(
            name="interactive",
            current_slots=interactive_limiter.capacity,
            min_slots=config.min_slots_interactive,
            max_slots=config.max_slots_interactive,
            latency_window=RollingLatencyWindow(config.window_size, max_age),
            queue_wait_window=RollingLatencyWindow(config.window_size, max_age),
        )
        self._bulk = TierState(
            name="bulk",
            current_slots=bulk_limiter.capacity,
            min_slots=config.min_slots_bulk,
            max_slots=config.max_slots_bulk,
            latency_window=RollingLatencyWindow(config.window_size, max_age),
            queue_wait_window=RollingLatencyWindow(config.window_size, max_age),
        )

        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    def record_latency(self, tier: str, latency_ms: float) -> None:
        """Record a completed request's latency in ms for ``"interactive"`` or ``"bulk"``."""
        if tier == "interactive":
            self._interactive.latency_window.record(latency_ms)
        elif tier == "bulk":
            self._bulk.latency_window.record(latency_ms)
        else:
            logger.warning("Unknown tier for latency recording", extra={"tier": tier})

    def record_queue_wait(self, tier: str, wait_ms: float) -> None:
        """Record how long a request waited for a slot, in ms.

        High queue wait means demand exceeds capacity: the increase signal when
        latency is good.
        """
        if tier == "interactive":
            self._interactive.queue_wait_window.record(wait_ms)
        elif tier == "bulk":
            self._bulk.queue_wait_window.record(wait_ms)
        else:
            logger.warning("Unknown tier for queue wait recording", extra={"tier": tier})

    async def start(self) -> None:
        """Start the adaptive control background loop."""
        if not self.config.enabled:
            logger.info("Adaptive concurrency control is disabled")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._control_loop())
        logger.info(
            "Adaptive concurrency control started",
            extra={
                "interval_seconds": self.config.adjustment_interval_seconds,
                "target_p95_ms": self.config.latency_target_p95_ms,
                "hysteresis": self.config.hysteresis_count,
            },
        )

    async def stop(self) -> None:
        """Stop the adaptive control background loop."""
        if self._task is not None:
            self._stop_event.set()
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Adaptive concurrency control stopped")

    async def _control_loop(self) -> None:
        """Background loop that periodically checks and adjusts concurrency."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(self.config.adjustment_interval_seconds)

                await self._evaluate_and_adjust(self._interactive, self._interactive_limiter)
                await self._evaluate_and_adjust(self._bulk, self._bulk_limiter)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Adaptive control loop error: {e}", exc_info=True)

    async def _evaluate_and_adjust(
        self,
        tier: TierState,
        limiter: ResizableLimiter,
    ) -> None:
        """Evaluate one tier and adjust its slots after enough consecutive signals.

        Increases also require queue pressure, so a fast but memory-hungry period with
        low latency does not open more concurrent work.
        """
        p95 = tier.latency_window.get_p95()
        queue_wait_p95 = tier.queue_wait_window.get_p95()
        tier.last_p95_ms = p95
        tier.last_queue_wait_p95_ms = queue_wait_p95

        if p95 is None:
            return

        target = self.config.latency_target_p95_ms
        queue_threshold = self.config.queue_wait_threshold_ms

        if p95 > target:
            tier.consecutive_decrease_signals += 1
            tier.consecutive_increase_signals = 0

            if tier.consecutive_decrease_signals >= self.config.hysteresis_count:
                await self._adjust_slots(tier, limiter, -self.config.decrease_step)
                tier.consecutive_decrease_signals = 0

        elif p95 < target * 0.8 and queue_wait_p95 is not None and queue_wait_p95 > queue_threshold:
            # Open up only when there is headroom (p95 under 80% of target, to
            # avoid oscillation) AND demand (queue wait over threshold).
            tier.consecutive_increase_signals += 1
            tier.consecutive_decrease_signals = 0

            if tier.consecutive_increase_signals >= self.config.hysteresis_count:
                await self._adjust_slots(tier, limiter, self.config.increase_step)
                tier.consecutive_increase_signals = 0

        else:
            tier.consecutive_increase_signals = 0
            tier.consecutive_decrease_signals = 0

    async def _adjust_slots(
        self,
        tier: TierState,
        limiter: ResizableLimiter,
        delta: int,
    ) -> None:
        """Change limiter capacity by ``delta``, bounded to [min_slots, max_slots].

        The bound never reverses the requested direction: capacity seeded outside the
        bounds (e.g. ``interactive_slots=2`` with ``min_slots_interactive=4``) must not
        turn a decrease into a doubling of concurrency on an overloaded tier.
        """
        new_slots = tier.current_slots + delta
        new_slots = max(tier.min_slots, min(tier.max_slots, new_slots))

        if (delta < 0 and new_slots > tier.current_slots) or (
            delta > 0 and new_slots < tier.current_slots
        ):
            logger.warning(
                f"Adaptive concurrency refused a clamp that reverses direction on {tier.name}",
                extra={
                    "tier": tier.name,
                    "current_slots": tier.current_slots,
                    "requested_delta": delta,
                    "clamped_to": new_slots,
                    "min_slots": tier.min_slots,
                    "max_slots": tier.max_slots,
                },
            )
            return

        if new_slots == tier.current_slots:
            return

        await limiter.resize(new_slots)

        direction = "increase" if new_slots > tier.current_slots else "decrease"
        if direction == "increase":
            tier.increase_events += 1
        else:
            tier.decrease_events += 1

        tier.current_slots = new_slots
        tier.last_adjustment_time = time.time()
        tier.last_adjustment_direction = direction

        logger.info(
            f"Adaptive concurrency adjusted {tier.name}",
            extra={
                "tier": tier.name,
                "direction": direction,
                "new_slots": tier.current_slots,
                "p95_ms": tier.last_p95_ms,
                "target_p95_ms": self.config.latency_target_p95_ms,
            },
        )

    def get_metrics(self) -> dict[str, Any]:
        """Return controller config plus per-tier slots, signals, events and latency stats."""
        return {
            "enabled": self.config.enabled,
            "target_p95_ms": self.config.latency_target_p95_ms,
            "queue_wait_threshold_ms": self.config.queue_wait_threshold_ms,
            "hysteresis_count": self.config.hysteresis_count,
            "interactive": {
                "current_slots": self._interactive.current_slots,
                "min_slots": self._interactive.min_slots,
                "max_slots": self._interactive.max_slots,
                "last_p95_ms": self._interactive.last_p95_ms,
                "last_queue_wait_p95_ms": self._interactive.last_queue_wait_p95_ms,
                "consecutive_increase_signals": self._interactive.consecutive_increase_signals,
                "consecutive_decrease_signals": self._interactive.consecutive_decrease_signals,
                "increase_events": self._interactive.increase_events,
                "decrease_events": self._interactive.decrease_events,
                "latency_stats": self._interactive.latency_window.get_stats(),
                "queue_wait_stats": self._interactive.queue_wait_window.get_stats(),
            },
            "bulk": {
                "current_slots": self._bulk.current_slots,
                "min_slots": self._bulk.min_slots,
                "max_slots": self._bulk.max_slots,
                "last_p95_ms": self._bulk.last_p95_ms,
                "last_queue_wait_p95_ms": self._bulk.last_queue_wait_p95_ms,
                "consecutive_increase_signals": self._bulk.consecutive_increase_signals,
                "consecutive_decrease_signals": self._bulk.consecutive_decrease_signals,
                "increase_events": self._bulk.increase_events,
                "decrease_events": self._bulk.decrease_events,
                "latency_stats": self._bulk.latency_window.get_stats(),
                "queue_wait_stats": self._bulk.queue_wait_window.get_stats(),
            },
        }

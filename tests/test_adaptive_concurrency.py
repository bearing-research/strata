"""Tests for adaptive concurrency control."""

import asyncio

import pytest
from pydantic import ValidationError

from strata.adaptive_concurrency import (
    AdaptiveConcurrencyController,
    AdaptiveConfig,
    ResizableLimiter,
    RollingLatencyWindow,
    TierState,
)
from strata.config import StrataConfig


class TestResizableLimiter:
    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        limiter = ResizableLimiter(2)
        assert limiter.capacity == 2
        assert limiter.in_use == 0
        assert limiter.available == 2

        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 1
        assert limiter.available == 1

        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 2
        assert limiter.available == 0

        await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 1

        await limiter.release()
        assert limiter.in_use == 0
        assert limiter.available == 2

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        limiter = ResizableLimiter(1)

        await limiter.acquire()
        assert limiter.available == 0

        result = await limiter.acquire(timeout=0.05)
        assert result is False
        assert limiter.in_use == 1  # Still just one

    @pytest.mark.asyncio
    async def test_resize_increase(self):
        limiter = ResizableLimiter(2)

        await limiter.acquire()
        await limiter.acquire()
        assert limiter.available == 0

        await limiter.resize(4)
        assert limiter.capacity == 4
        assert limiter.available == 2  # 4 - 2 in use

        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 3

    @pytest.mark.asyncio
    async def test_resize_decrease(self):
        limiter = ResizableLimiter(4)

        await limiter.acquire()
        await limiter.acquire()
        assert limiter.in_use == 2

        await limiter.resize(3)
        assert limiter.capacity == 3
        assert limiter.available == 1  # 3 - 2 in use

        await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 2  # 3 - 1 in use

    @pytest.mark.asyncio
    async def test_resize_below_in_use(self):
        """Resize below in_use succeeds and just blocks new acquires."""
        limiter = ResizableLimiter(4)

        for _ in range(4):
            await limiter.acquire()
        assert limiter.in_use == 4

        # Resize below current in_use.
        await limiter.resize(2)
        assert limiter.capacity == 2
        assert limiter.available == 0  # max(0, 2 - 4) = 0

        result = await limiter.acquire(timeout=0.01)
        assert result is False

        for _ in range(3):
            await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 1  # 2 - 1

    @pytest.mark.asyncio
    async def test_resize_wakes_waiters(self):
        """A capacity increase wakes waiting acquirers."""
        limiter = ResizableLimiter(1)
        await limiter.acquire()

        acquired = False

        async def try_acquire():
            nonlocal acquired
            acquired = await limiter.acquire(timeout=1.0)

        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.01)  # Let it start waiting

        await limiter.resize(2)
        await asyncio.sleep(0.01)  # Let it acquire

        await task
        assert acquired is True
        assert limiter.in_use == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("timeout", [None, 60.0])
    async def test_a_cancelled_waiter_passes_its_wakeup_on(self, timeout):
        """A waiter cancelled after release() picked it must not strand the slot.

        On CPython 3.12, asyncio.Condition drops a notify whose waiter is cancelled before it runs,
        so the next waiter slept with the slot free until its deadline.
        """
        limiter = ResizableLimiter(1)
        await limiter.acquire()
        picked = asyncio.create_task(limiter.acquire(timeout=timeout))
        next_in_line = asyncio.create_task(limiter.acquire(timeout=timeout))
        for _ in range(3):
            await asyncio.sleep(0)  # both queue on the condition

        await limiter.release()  # notify(1) picks the first waiter
        picked.cancel()  # its client disconnects before it runs
        for _ in range(10):
            if next_in_line.done():
                break
            await asyncio.sleep(0)

        try:
            assert picked.cancelled()
            assert next_in_line.done(), "the free slot was left idle"
            assert next_in_line.result() is True
            assert limiter.in_use == 1
        finally:
            next_in_line.cancel()

    @pytest.mark.asyncio
    async def test_release_without_acquire_raises(self):
        limiter = ResizableLimiter(2)
        with pytest.raises(RuntimeError, match="release.*without"):
            await limiter.release()

    @pytest.mark.asyncio
    async def test_resize_to_zero_raises(self):
        limiter = ResizableLimiter(2)
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            await limiter.resize(0)

    def test_get_stats(self):
        limiter = ResizableLimiter(5)
        stats = limiter.get_stats()
        assert stats["capacity"] == 5
        assert stats["in_use"] == 0
        assert stats["available"] == 5


class TestRollingLatencyWindow:
    def test_empty_window_returns_none(self):
        window = RollingLatencyWindow(size=100)
        assert window.get_p95() is None

    def test_few_samples_returns_none(self):
        """At least 10 samples are needed for a meaningful percentile."""
        window = RollingLatencyWindow(size=100)
        for i in range(9):
            window.record(float(i))
        assert window.get_p95() is None

    def test_exactly_10_samples(self):
        window = RollingLatencyWindow(size=100)
        for i in range(10):
            window.record(float(i))
        p95 = window.get_p95()
        assert p95 is not None
        # With 10 samples [0-9], p95 should be the 9th or 10th value
        assert p95 >= 8.0

    def test_rolling_behavior(self):
        """A full window drops its oldest values."""
        window = RollingLatencyWindow(size=10)

        for _ in range(10):
            window.record(10.0)

        p95_low = window.get_p95()
        assert p95_low == 10.0

        for _ in range(10):
            window.record(100.0)

        p95_high = window.get_p95()
        assert p95_high == 100.0  # Old values should be gone

    def test_get_stats(self):
        window = RollingLatencyWindow(size=100)
        for i in range(1, 101):
            window.record(float(i))

        stats = window.get_stats()
        assert stats["count"] == 100
        assert stats["window_size"] == 100
        assert stats["min_ms"] == 1.0
        assert stats["max_ms"] == 100.0
        assert stats["avg_ms"] == 50.5  # Sum 1..100 / 100
        assert stats["p50_ms"] == pytest.approx(50.0, abs=1)
        assert stats["p95_ms"] == pytest.approx(95.0, abs=1)
        assert stats["p99_ms"] == pytest.approx(99.0, abs=1)

    def test_reset(self):
        window = RollingLatencyWindow(size=100)
        for i in range(50):
            window.record(float(i))

        window.reset()
        assert window.get_p95() is None
        stats = window.get_stats()
        assert stats["count"] == 0
        assert stats["window_size"] == 0


class TestAdaptiveConfig:
    def test_default_values(self):
        config = AdaptiveConfig()
        assert config.enabled is False  # Disabled by default
        assert config.adjustment_interval_seconds == 5.0
        assert config.latency_target_p95_ms == 500.0
        assert config.hysteresis_count == 3
        assert config.min_slots_interactive == 4
        assert config.max_slots_interactive == 64
        assert config.min_slots_bulk == 2
        assert config.max_slots_bulk == 32

    def test_custom_values(self):
        config = AdaptiveConfig(
            enabled=True,
            latency_target_p95_ms=200.0,
            hysteresis_count=5,
        )
        assert config.enabled is True
        assert config.latency_target_p95_ms == 200.0
        assert config.hysteresis_count == 5


class TestAdaptiveConcurrencyController:
    @pytest.fixture
    def limiters(self):
        """Create test limiters."""
        interactive = ResizableLimiter(10)
        bulk = ResizableLimiter(4)
        return interactive, bulk

    @pytest.fixture
    def controller(self, limiters):
        """Create a controller with test config."""
        interactive, bulk = limiters
        config = AdaptiveConfig(
            enabled=True,
            adjustment_interval_seconds=0.1,  # Fast for tests
            latency_target_p95_ms=100.0,  # 100ms target
            hysteresis_count=2,  # Only need 2 consecutive signals
            min_slots_interactive=4,
            max_slots_interactive=20,
            min_slots_bulk=2,
            max_slots_bulk=10,
            window_size=20,
        )
        return AdaptiveConcurrencyController(
            config=config,
            interactive_limiter=interactive,
            bulk_limiter=bulk,
        )

    def test_record_latency(self, controller):
        controller.record_latency("interactive", 50.0)
        controller.record_latency("bulk", 150.0)

        interactive_stats = controller._interactive.latency_window.get_stats()
        bulk_stats = controller._bulk.latency_window.get_stats()

        assert interactive_stats["count"] == 1
        assert bulk_stats["count"] == 1

    def test_get_metrics(self, controller):
        for i in range(20):
            controller.record_latency("interactive", float(50 + i))
            controller.record_latency("bulk", float(100 + i))

        metrics = controller.get_metrics()

        assert metrics["enabled"] is True
        assert metrics["target_p95_ms"] == 100.0
        assert metrics["hysteresis_count"] == 2

        assert metrics["interactive"]["current_slots"] == 10
        assert metrics["interactive"]["min_slots"] == 4
        assert metrics["interactive"]["max_slots"] == 20
        assert metrics["interactive"]["latency_stats"]["count"] == 20

        assert metrics["bulk"]["current_slots"] == 4
        assert metrics["bulk"]["min_slots"] == 2
        assert metrics["bulk"]["max_slots"] == 10
        assert metrics["bulk"]["latency_stats"]["count"] == 20

    @pytest.mark.asyncio
    async def test_disabled_controller_does_nothing(self, limiters):
        """A disabled controller starts no background task."""
        interactive, bulk = limiters
        config = AdaptiveConfig(enabled=False)
        controller = AdaptiveConcurrencyController(
            config=config,
            interactive_limiter=interactive,
            bulk_limiter=bulk,
        )

        await controller.start()
        assert controller._task is None
        await controller.stop()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, controller):
        await controller.start()
        assert controller._task is not None

        await controller.stop()
        assert controller._task is None

    @pytest.mark.asyncio
    async def test_decrease_slots_on_high_latency(self, controller, limiters):
        """Slots decrease when p95 exceeds the target."""
        interactive, bulk = limiters

        # Latencies above the 100ms target.
        for _ in range(20):
            controller.record_latency("interactive", 200.0)

        # Manually trigger evaluation (don't wait for control loop)
        await controller._evaluate_and_adjust(controller._interactive, interactive)
        # First signal
        assert controller._interactive.consecutive_decrease_signals == 1

        await controller._evaluate_and_adjust(controller._interactive, interactive)
        # The second signal triggers the adjustment (hysteresis=2): 10 -> 9.
        assert controller._interactive.current_slots == 9
        assert controller._interactive.decrease_events == 1

    @pytest.mark.asyncio
    async def test_increase_slots_on_low_latency_with_queue_pressure(self, controller, limiters):
        """Slots increase only when p95 < 80% of target and there is queue pressure."""
        interactive, bulk = limiters

        # Record low latencies (below 80ms = 80% of 100ms target)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)

        # Record queue wait above threshold (100ms default) to indicate demand
        for _ in range(20):
            controller.record_queue_wait("interactive", 150.0)

        await controller._evaluate_and_adjust(controller._interactive, interactive)
        await controller._evaluate_and_adjust(controller._interactive, interactive)

        assert controller._interactive.current_slots == 11
        assert controller._interactive.increase_events == 1

    @pytest.mark.asyncio
    async def test_no_increase_without_queue_pressure(self, controller, limiters):
        interactive, bulk = limiters

        # Record low latencies (below 80ms = 80% of 100ms target)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)

        # Record LOW queue wait (below 100ms threshold) - no pressure
        for _ in range(20):
            controller.record_queue_wait("interactive", 10.0)

        for _ in range(5):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # No queue pressure means no demand, so no increase.
        assert controller._interactive.current_slots == 10
        assert controller._interactive.increase_events == 0

    @pytest.mark.asyncio
    async def test_slots_bounded_by_min(self, controller, limiters):
        interactive, bulk = limiters

        # Set slots near minimum - also resize the limiter
        controller._interactive.current_slots = 5
        await interactive.resize(5)

        for _ in range(20):
            controller.record_latency("interactive", 200.0)

        for _ in range(10):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Should not go below min_slots_interactive=4
        assert controller._interactive.current_slots >= 4

    @pytest.mark.asyncio
    async def test_slots_bounded_by_max(self, controller, limiters):
        interactive, bulk = limiters

        # Set slots near maximum - also resize the limiter
        controller._interactive.current_slots = 19
        await interactive.resize(19)

        # Record low latencies and high queue wait (to trigger increase attempts)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)
            controller.record_queue_wait("interactive", 150.0)

        for _ in range(10):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Should not go above max_slots_interactive=20
        assert controller._interactive.current_slots <= 20

    @pytest.mark.asyncio
    async def test_hysteresis_prevents_flapping(self, controller, limiters):
        interactive, bulk = limiters

        for _ in range(20):
            controller.record_latency("interactive", 200.0)

        # Single evaluation shouldn't change slots
        await controller._evaluate_and_adjust(controller._interactive, interactive)
        assert controller._interactive.current_slots == 10  # No change yet
        assert controller._interactive.consecutive_decrease_signals == 1

        # Now record low latency (resets signals)
        controller._interactive.latency_window.reset()
        for _ in range(20):
            controller.record_latency("interactive", 85.0)  # Between 80% and 100% of target

        await controller._evaluate_and_adjust(controller._interactive, interactive)
        # Signals should reset (latency in acceptable range)
        assert controller._interactive.consecutive_decrease_signals == 0
        assert controller._interactive.consecutive_increase_signals == 0
        assert controller._interactive.current_slots == 10  # Still no change

    @pytest.mark.asyncio
    async def test_bulk_tier_independent(self, controller, limiters):
        interactive, bulk = limiters

        # Only record high latencies for bulk
        for _ in range(20):
            controller.record_latency("bulk", 200.0)

        await controller._evaluate_and_adjust(controller._interactive, interactive)
        await controller._evaluate_and_adjust(controller._bulk, bulk)
        await controller._evaluate_and_adjust(controller._bulk, bulk)

        assert controller._interactive.current_slots == 10  # No change
        assert controller._bulk.current_slots == 3  # Decreased from 4


class TestTierState:
    def test_default_values(self):
        state = TierState(
            name="interactive",
            current_slots=10,
            min_slots=4,
            max_slots=20,
        )
        assert state.consecutive_increase_signals == 0
        assert state.consecutive_decrease_signals == 0
        assert state.increase_events == 0
        assert state.decrease_events == 0
        assert state.last_adjustment_direction == ""


class TestSampleAging:
    """A control loop must not steer off traffic that is over.

    A count-only window let stale slow samples keep signalling decreases through an idle period.
    Time is stepped, not slept: expiry compares timestamps, and Windows' ~15ms timer granularity
    made sleeping past a 50ms window flaky.
    """

    @staticmethod
    def _window(**kwargs):
        """A window whose clock the test advances by hand."""
        now = [1000.0]

        def advance(seconds: float) -> None:
            now[0] += seconds

        return RollingLatencyWindow(clock=lambda: now[0], **kwargs), advance

    def test_stale_samples_stop_counting(self):
        window, advance = self._window(size=100, max_age_seconds=60.0)
        for _ in range(10):
            window.record(900.0)
        assert window.get_p95() == 900.0

        advance(61.0)

        assert window.get_p95() is None
        stats = window.get_stats()
        assert stats["window_size"] == 0
        assert stats["p95_ms"] is None
        # The cumulative count is history, not a live signal, so it stays.
        assert stats["count"] == 10

    def test_fresh_samples_survive_alongside_expired_ones(self):
        window, advance = self._window(size=100, max_age_seconds=60.0)
        for _ in range(10):
            window.record(900.0)
        advance(61.0)
        for _ in range(10):
            window.record(10.0)

        # The p95 describes current traffic, not the old burst.
        assert window.get_p95() == 10.0

    def test_a_sample_inside_the_window_is_still_live(self):
        """Just under the age bound, the sample is still live."""
        window, advance = self._window(size=100, max_age_seconds=60.0)
        for _ in range(10):
            window.record(900.0)
        advance(59.0)
        assert window.get_p95() == 900.0

    def test_no_max_age_keeps_every_sample(self):
        window, advance = self._window(size=100)
        for _ in range(10):
            window.record(900.0)
        advance(3600.0)
        assert window.get_p95() == 900.0

    def test_controller_windows_inherit_the_configured_age(self):
        config = AdaptiveConfig(sample_max_age_seconds=7.0)
        controller = AdaptiveConcurrencyController(
            config=config,
            interactive_limiter=ResizableLimiter(10),
            bulk_limiter=ResizableLimiter(4),
        )
        assert controller._interactive.latency_window._max_age_seconds == 7.0
        assert controller._interactive.queue_wait_window._max_age_seconds == 7.0
        assert controller._bulk.latency_window._max_age_seconds == 7.0


class TestClampDoesNotReverseDirection:
    """Clamping bounds the result but must not reverse the direction.

    ``current_slots`` is seeded from the limiter capacity, which may start below ``min_slots``; a
    decrease request then clamped up to the minimum and raised concurrency on an overloaded tier.
    """

    @pytest.mark.asyncio
    async def test_a_decrease_below_min_does_not_raise_capacity(self):
        limiter = ResizableLimiter(2)
        controller = AdaptiveConcurrencyController(
            config=AdaptiveConfig(min_slots_interactive=4, max_slots_interactive=20),
            interactive_limiter=limiter,
            bulk_limiter=ResizableLimiter(4),
        )
        tier = controller._interactive
        assert tier.current_slots == 2  # below min_slots, as configured

        await controller._adjust_slots(tier, limiter, -1)

        assert limiter.capacity == 2
        assert tier.current_slots == 2
        assert tier.increase_events == 0
        assert tier.last_adjustment_direction == ""

    @pytest.mark.asyncio
    async def test_an_increase_above_max_does_not_drop_capacity(self):
        limiter = ResizableLimiter(100)
        controller = AdaptiveConcurrencyController(
            config=AdaptiveConfig(min_slots_interactive=4, max_slots_interactive=64),
            interactive_limiter=limiter,
            bulk_limiter=ResizableLimiter(4),
        )
        tier = controller._interactive

        await controller._adjust_slots(tier, limiter, 1)

        assert limiter.capacity == 100
        assert tier.decrease_events == 0

    @pytest.mark.asyncio
    async def test_a_decrease_within_bounds_still_decreases(self):
        limiter = ResizableLimiter(10)
        controller = AdaptiveConcurrencyController(
            config=AdaptiveConfig(min_slots_interactive=4, max_slots_interactive=20),
            interactive_limiter=limiter,
            bulk_limiter=ResizableLimiter(4),
        )
        tier = controller._interactive

        await controller._adjust_slots(tier, limiter, -1)

        assert limiter.capacity == 9
        assert tier.current_slots == 9
        assert tier.last_adjustment_direction == "decrease"


class TestAdaptiveStartupValidation:
    """The controller's starting slot count must lie within its own bounds.

    Otherwise the first adjustment jumps to a bound. Refusing the config tells the operator instead
    of silently overriding slot counts they chose.
    """

    def test_slots_below_the_adaptive_floor_are_rejected(self):
        with pytest.raises(ValidationError, match="interactive_slots"):
            StrataConfig(adaptive_enabled=True, interactive_slots=2)

    def test_slots_above_the_adaptive_ceiling_are_rejected(self):
        with pytest.raises(ValidationError, match="interactive_slots"):
            StrataConfig(adaptive_enabled=True, interactive_slots=200)

    def test_bulk_slots_outside_the_adaptive_range_are_rejected(self):
        with pytest.raises(ValidationError, match="bulk_slots"):
            StrataConfig(adaptive_enabled=True, bulk_slots=100)

    def test_default_slot_counts_are_inside_the_default_bounds(self):
        # Otherwise merely flipping the flag on would fail to boot.
        config = StrataConfig(adaptive_enabled=True)
        assert config.adaptive_min_interactive <= config.interactive_slots
        assert config.interactive_slots <= config.adaptive_max_interactive

    def test_adaptive_with_multi_tenancy_is_rejected(self, tmp_path):
        # The controller holds the default tenant's limiters, a tier no multi-tenant
        # request acquires, so it would run as a no-op.
        # Spelled out as a valid service-mode config: personal mode already rejects
        # multi-tenancy, and matching that error would pass whether or not this rule exists.
        with pytest.raises(ValidationError, match="adaptive_enabled cannot be combined"):
            StrataConfig(
                deployment_mode="service",
                adaptive_enabled=True,
                multi_tenant_enabled=True,
                auth_mode="trusted_proxy",
                proxy_token="x" * 32,
                artifact_dir=str(tmp_path),
            )

    def test_inverted_bounds_are_still_rejected(self):
        with pytest.raises(ValidationError, match="adaptive_min_interactive"):
            StrataConfig(
                adaptive_enabled=True,
                adaptive_min_interactive=40,
                adaptive_max_interactive=8,
            )

    def test_none_of_this_applies_when_adaptive_is_off(self):
        config = StrataConfig(interactive_slots=2, bulk_slots=100)
        assert config.adaptive_enabled is False

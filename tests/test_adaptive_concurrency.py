"""Tests for adaptive concurrency control."""

import asyncio
import time

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
    """Tests for ResizableLimiter."""

    @pytest.mark.asyncio
    async def test_basic_acquire_release(self):
        """Basic acquire and release should work."""
        limiter = ResizableLimiter(2)
        assert limiter.capacity == 2
        assert limiter.in_use == 0
        assert limiter.available == 2

        # Acquire first slot
        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 1
        assert limiter.available == 1

        # Acquire second slot
        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 2
        assert limiter.available == 0

        # Release one
        await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 1

        # Release second
        await limiter.release()
        assert limiter.in_use == 0
        assert limiter.available == 2

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Acquire should respect timeout."""
        limiter = ResizableLimiter(1)

        # Fill the limiter
        await limiter.acquire()
        assert limiter.available == 0

        # Try to acquire with short timeout - should fail
        result = await limiter.acquire(timeout=0.05)
        assert result is False
        assert limiter.in_use == 1  # Still just one

    @pytest.mark.asyncio
    async def test_resize_increase(self):
        """Resize should allow increasing capacity."""
        limiter = ResizableLimiter(2)

        # Fill the limiter
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.available == 0

        # Resize to allow more
        await limiter.resize(4)
        assert limiter.capacity == 4
        assert limiter.available == 2  # 4 - 2 in use

        # Now we can acquire more
        result = await limiter.acquire()
        assert result is True
        assert limiter.in_use == 3

    @pytest.mark.asyncio
    async def test_resize_decrease(self):
        """Resize should allow decreasing capacity."""
        limiter = ResizableLimiter(4)

        # Use 2 slots
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.in_use == 2

        # Resize down to 3
        await limiter.resize(3)
        assert limiter.capacity == 3
        assert limiter.available == 1  # 3 - 2 in use

        # Release one
        await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 2  # 3 - 1 in use

    @pytest.mark.asyncio
    async def test_resize_below_in_use(self):
        """Resize below in_use should work, just block new acquires."""
        limiter = ResizableLimiter(4)

        # Use all 4 slots
        for _ in range(4):
            await limiter.acquire()
        assert limiter.in_use == 4

        # Resize down to 2 (below current in_use)
        await limiter.resize(2)
        assert limiter.capacity == 2
        assert limiter.available == 0  # max(0, 2 - 4) = 0

        # Can't acquire more
        result = await limiter.acquire(timeout=0.01)
        assert result is False

        # Release 3 - now we have capacity again
        for _ in range(3):
            await limiter.release()
        assert limiter.in_use == 1
        assert limiter.available == 1  # 2 - 1

    @pytest.mark.asyncio
    async def test_resize_wakes_waiters(self):
        """Resize increase should wake waiting acquirers."""
        limiter = ResizableLimiter(1)
        await limiter.acquire()

        acquired = False

        async def try_acquire():
            nonlocal acquired
            acquired = await limiter.acquire(timeout=1.0)

        # Start a waiter
        task = asyncio.create_task(try_acquire())
        await asyncio.sleep(0.01)  # Let it start waiting

        # Resize to allow the waiter through
        await limiter.resize(2)
        await asyncio.sleep(0.01)  # Let it acquire

        await task
        assert acquired is True
        assert limiter.in_use == 2

    @pytest.mark.asyncio
    async def test_release_without_acquire_raises(self):
        """Release without acquire should raise."""
        limiter = ResizableLimiter(2)
        with pytest.raises(RuntimeError, match="release.*without"):
            await limiter.release()

    @pytest.mark.asyncio
    async def test_resize_to_zero_raises(self):
        """Resize to zero should raise."""
        limiter = ResizableLimiter(2)
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            await limiter.resize(0)

    def test_get_stats(self):
        """get_stats should return accurate info."""
        limiter = ResizableLimiter(5)
        stats = limiter.get_stats()
        assert stats["capacity"] == 5
        assert stats["in_use"] == 0
        assert stats["available"] == 5


class TestRollingLatencyWindow:
    """Tests for RollingLatencyWindow percentile calculation."""

    def test_empty_window_returns_none(self):
        """Empty window should return None for p95."""
        window = RollingLatencyWindow(size=100)
        assert window.get_p95() is None

    def test_few_samples_returns_none(self):
        """Need at least 10 samples for meaningful percentile."""
        window = RollingLatencyWindow(size=100)
        for i in range(9):
            window.record(float(i))
        assert window.get_p95() is None

    def test_exactly_10_samples(self):
        """10 samples should give a valid p95."""
        window = RollingLatencyWindow(size=100)
        for i in range(10):
            window.record(float(i))
        p95 = window.get_p95()
        assert p95 is not None
        # With 10 samples [0-9], p95 should be the 9th or 10th value
        assert p95 >= 8.0

    def test_rolling_behavior(self):
        """Window should drop old values when full."""
        window = RollingLatencyWindow(size=10)

        # Fill with low latencies
        for _ in range(10):
            window.record(10.0)

        p95_low = window.get_p95()
        assert p95_low == 10.0

        # Now add high latencies
        for _ in range(10):
            window.record(100.0)

        p95_high = window.get_p95()
        assert p95_high == 100.0  # Old values should be gone

    def test_get_stats(self):
        """Test comprehensive stats output."""
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
        """Reset should clear all samples."""
        window = RollingLatencyWindow(size=100)
        for i in range(50):
            window.record(float(i))

        window.reset()
        assert window.get_p95() is None
        stats = window.get_stats()
        assert stats["count"] == 0
        assert stats["window_size"] == 0


class TestAdaptiveConfig:
    """Tests for AdaptiveConfig defaults."""

    def test_default_values(self):
        """Test sensible defaults."""
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
        """Test custom configuration."""
        config = AdaptiveConfig(
            enabled=True,
            latency_target_p95_ms=200.0,
            hysteresis_count=5,
        )
        assert config.enabled is True
        assert config.latency_target_p95_ms == 200.0
        assert config.hysteresis_count == 5


class TestAdaptiveConcurrencyController:
    """Tests for AdaptiveConcurrencyController."""

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
        """Test latency recording to appropriate tier."""
        controller.record_latency("interactive", 50.0)
        controller.record_latency("bulk", 150.0)

        # Check latencies are recorded in correct windows
        interactive_stats = controller._interactive.latency_window.get_stats()
        bulk_stats = controller._bulk.latency_window.get_stats()

        assert interactive_stats["count"] == 1
        assert bulk_stats["count"] == 1

    def test_get_metrics(self, controller):
        """Test metrics output."""
        # Record some latencies
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
        """Disabled controller should not start background task."""
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
        """Test controller start/stop lifecycle."""
        await controller.start()
        assert controller._task is not None

        await controller.stop()
        assert controller._task is None

    @pytest.mark.asyncio
    async def test_decrease_slots_on_high_latency(self, controller, limiters):
        """Controller should decrease slots when p95 > target."""
        interactive, bulk = limiters

        # Record high latencies (above 100ms target)
        for _ in range(20):
            controller.record_latency("interactive", 200.0)

        # Manually trigger evaluation (don't wait for control loop)
        await controller._evaluate_and_adjust(controller._interactive, interactive)
        # First signal
        assert controller._interactive.consecutive_decrease_signals == 1

        await controller._evaluate_and_adjust(controller._interactive, interactive)
        # Second signal should trigger adjustment (hysteresis=2)
        # Slots should decrease from 10 to 9
        assert controller._interactive.current_slots == 9
        assert controller._interactive.decrease_events == 1

    @pytest.mark.asyncio
    async def test_increase_slots_on_low_latency_with_queue_pressure(self, controller, limiters):
        """Controller should increase slots when p95 < 80% of target AND queue pressure exists."""
        interactive, bulk = limiters

        # Record low latencies (below 80ms = 80% of 100ms target)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)

        # Record queue wait above threshold (100ms default) to indicate demand
        for _ in range(20):
            controller.record_queue_wait("interactive", 150.0)

        # Trigger evaluations
        await controller._evaluate_and_adjust(controller._interactive, interactive)
        await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Slots should increase from 10 to 11
        assert controller._interactive.current_slots == 11
        assert controller._interactive.increase_events == 1

    @pytest.mark.asyncio
    async def test_no_increase_without_queue_pressure(self, controller, limiters):
        """Controller should NOT increase slots when latency is low but no queue pressure."""
        interactive, bulk = limiters

        # Record low latencies (below 80ms = 80% of 100ms target)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)

        # Record LOW queue wait (below 100ms threshold) - no pressure
        for _ in range(20):
            controller.record_queue_wait("interactive", 10.0)

        # Trigger evaluations multiple times
        for _ in range(5):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Slots should NOT increase - no queue pressure means no demand
        assert controller._interactive.current_slots == 10
        assert controller._interactive.increase_events == 0

    @pytest.mark.asyncio
    async def test_slots_bounded_by_min(self, controller, limiters):
        """Slots should not go below minimum."""
        interactive, bulk = limiters

        # Set slots near minimum - also resize the limiter
        controller._interactive.current_slots = 5
        await interactive.resize(5)

        # Record high latencies
        for _ in range(20):
            controller.record_latency("interactive", 200.0)

        # Try to decrease multiple times
        for _ in range(10):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Should not go below min_slots_interactive=4
        assert controller._interactive.current_slots >= 4

    @pytest.mark.asyncio
    async def test_slots_bounded_by_max(self, controller, limiters):
        """Slots should not go above maximum."""
        interactive, bulk = limiters

        # Set slots near maximum - also resize the limiter
        controller._interactive.current_slots = 19
        await interactive.resize(19)

        # Record low latencies and high queue wait (to trigger increase attempts)
        for _ in range(20):
            controller.record_latency("interactive", 50.0)
            controller.record_queue_wait("interactive", 150.0)

        # Try to increase multiple times
        for _ in range(10):
            await controller._evaluate_and_adjust(controller._interactive, interactive)

        # Should not go above max_slots_interactive=20
        assert controller._interactive.current_slots <= 20

    @pytest.mark.asyncio
    async def test_hysteresis_prevents_flapping(self, controller, limiters):
        """Hysteresis should prevent rapid changes."""
        interactive, bulk = limiters

        # Record high latency
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
        """Bulk tier should be adjusted independently."""
        interactive, bulk = limiters

        # Only record high latencies for bulk
        for _ in range(20):
            controller.record_latency("bulk", 200.0)

        # Trigger evaluations for both tiers
        await controller._evaluate_and_adjust(controller._interactive, interactive)
        await controller._evaluate_and_adjust(controller._bulk, bulk)
        await controller._evaluate_and_adjust(controller._bulk, bulk)

        # Only bulk should have decreased
        assert controller._interactive.current_slots == 10  # No change
        assert controller._bulk.current_slots == 3  # Decreased from 4


class TestTierState:
    """Tests for TierState dataclass."""

    def test_default_values(self):
        """Test TierState defaults."""
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

    The window was count-only, so a burst of slow requests followed by an idle
    period kept re-triggering decrease signals from stale samples until the
    tier sat at ``min_slots`` with nothing running.
    """

    def test_stale_samples_stop_counting(self):
        window = RollingLatencyWindow(size=100, max_age_seconds=0.05)
        for _ in range(10):
            window.record(900.0)
        assert window.get_p95() == 900.0

        time.sleep(0.06)

        assert window.get_p95() is None
        stats = window.get_stats()
        assert stats["window_size"] == 0
        assert stats["p95_ms"] is None
        # The cumulative count is history, not a live signal — it stays.
        assert stats["count"] == 10

    def test_fresh_samples_survive_alongside_expired_ones(self):
        window = RollingLatencyWindow(size=100, max_age_seconds=0.05)
        for _ in range(10):
            window.record(900.0)
        time.sleep(0.06)
        for _ in range(10):
            window.record(10.0)

        # The p95 describes current traffic, not the old burst.
        assert window.get_p95() == 10.0

    def test_no_max_age_keeps_every_sample(self):
        window = RollingLatencyWindow(size=100)
        for _ in range(10):
            window.record(900.0)
        time.sleep(0.06)
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
    """Clamping bounded the result but not the *direction*.

    ``current_slots`` is seeded from the limiter's configured capacity, which
    could start outside ``[min_slots, max_slots]``. With capacity 2 and
    ``min_slots_interactive=4``, a latency breach asked for -1, the clamp
    answered 4, and concurrency doubled on an overloaded tier — logged as an
    "increase" nothing had requested.
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
    """The controller's starting point has to be inside its own bounds.

    It seeds each tier from the configured slot count and clamps from there,
    so a slot count outside ``[min, max]`` means the first adjustment jumps to
    a bound instead of nudging. Refusing the config tells the operator; the
    alternative silently overrides slot counts they chose deliberately.
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
        # The controller holds the default tenant's limiters. Under
        # multi-tenancy that is a tier no request acquires, so it would run as
        # a no-op again — the exact failure #549 is about.
        #
        # Spelled out in full as a valid service-mode config: multi-tenancy is
        # already incoherent in personal mode, and matching that error instead
        # would pass whether or not this rule exists.
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

"""Tests for build QoS (Quality of Service) admission control."""

from __future__ import annotations

import asyncio

import pytest

from strata.transforms.build_qos import (
    BuildPriority,
    BuildQoS,
    BuildQoSConfig,
    GlobalCapacityError,
    TenantAtCapacityError,
    TenantQuotaExceededError,
    get_build_qos,
    reset_build_qos,
    set_build_qos,
)


class TestBuildClassification:
    """Tests for build priority classification."""

    def test_classify_default_is_interactive(self):
        """Default classification is interactive."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        priority = qos.classify_build()
        assert priority == BuildPriority.INTERACTIVE

    def test_classify_explicit_priority(self):
        """Explicit priority overrides all other signals."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        # Even with many inputs, explicit priority wins
        priority = qos.classify_build(
            input_count=100,
            explicit_priority=BuildPriority.INTERACTIVE,
        )
        assert priority == BuildPriority.INTERACTIVE

        priority = qos.classify_build(
            estimated_output_bytes=1,
            input_count=1,
            explicit_priority=BuildPriority.BULK,
        )
        assert priority == BuildPriority.BULK

    def test_classify_by_output_bytes(self):
        """Large output estimates classify as bulk."""
        config = BuildQoSConfig(classify_by_estimated_bytes=1024 * 1024)  # 1MB
        qos = BuildQoS(config)

        # Below threshold = interactive
        priority = qos.classify_build(estimated_output_bytes=512 * 1024)
        assert priority == BuildPriority.INTERACTIVE

        # Above threshold = bulk
        priority = qos.classify_build(estimated_output_bytes=2 * 1024 * 1024)
        assert priority == BuildPriority.BULK

    def test_classify_by_input_count(self):
        """Many inputs classify as bulk."""
        config = BuildQoSConfig(classify_by_input_count=3)
        qos = BuildQoS(config)

        # Below threshold = interactive
        priority = qos.classify_build(input_count=2)
        assert priority == BuildPriority.INTERACTIVE

        # At threshold = interactive
        priority = qos.classify_build(input_count=3)
        assert priority == BuildPriority.INTERACTIVE

        # Above threshold = bulk
        priority = qos.classify_build(input_count=4)
        assert priority == BuildPriority.BULK


class TestTenantQuota:
    """Tests for per-tenant byte quota tracking."""

    @pytest.mark.asyncio
    async def test_quota_not_enforced_when_disabled(self):
        """Quota check passes when bytes_per_day_limit is None."""
        config = BuildQoSConfig(bytes_per_day_limit=None)
        qos = BuildQoS(config)

        # Should not raise even with huge bytes
        await qos.check_quota("tenant1", 10 * 1024 * 1024 * 1024)  # 10GB

    @pytest.mark.asyncio
    async def test_quota_enforced_when_enabled(self):
        """Quota check raises when limit exceeded."""
        config = BuildQoSConfig(bytes_per_day_limit=1024 * 1024)  # 1MB
        qos = BuildQoS(config)

        # First check should pass
        await qos.check_quota("tenant1", 512 * 1024)

        # Record some usage
        await qos.record_bytes("tenant1", 900 * 1024)

        # Now exceeding quota should raise
        with pytest.raises(TenantQuotaExceededError) as exc_info:
            await qos.check_quota("tenant1", 200 * 1024)

        assert exc_info.value.tenant_id == "tenant1"
        assert exc_info.value.status_code == 429
        assert exc_info.value.used_bytes == 900 * 1024
        assert exc_info.value.limit_bytes == 1024 * 1024

    @pytest.mark.asyncio
    async def test_quota_per_tenant_isolation(self):
        """Each tenant has separate quota tracking."""
        config = BuildQoSConfig(bytes_per_day_limit=1024 * 1024)  # 1MB
        qos = BuildQoS(config)

        # Record usage for tenant1
        await qos.record_bytes("tenant1", 900 * 1024)

        # tenant2 should still be able to use quota
        await qos.check_quota("tenant2", 500 * 1024)

        # tenant1 should be near limit
        with pytest.raises(TenantQuotaExceededError):
            await qos.check_quota("tenant1", 200 * 1024)


class TestSlotAcquisition:
    """Tests for build slot acquisition with timeouts."""

    @pytest.mark.asyncio
    async def test_acquire_and_release(self):
        """Basic acquire and release flow."""
        config = BuildQoSConfig(
            interactive_slots=2,
            per_tenant_interactive=2,
        )
        qos = BuildQoS(config)

        # Should acquire successfully
        slot = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Check metrics
        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 1

        # Release
        await slot.release()

        # Check slot was released
        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 0

    @pytest.mark.asyncio
    async def test_acquire_releases_tenant_slot_on_cancel(self):
        """Cancelling acquire while queued for a global slot must release the
        per-tenant slot it already grabbed.

        Regression: the cleanup was `except Exception`, which does not catch
        asyncio.CancelledError (a BaseException). A client disconnect / shutdown
        while a build was queued for a global slot leaked the per-tenant slot
        permanently, eventually returning TenantAtCapacityError for every later
        build by that tenant (a per-tenant DoS).
        """
        config = BuildQoSConfig(
            interactive_slots=1,  # one global slot
            per_tenant_interactive=5,  # tenant slots are not the constraint
            interactive_queue_timeout=10.0,  # long, so the 2nd acquire queues
        )
        qos = BuildQoS(config)

        # Fill the single global slot.
        slot_a = await qos.acquire("t1", BuildPriority.INTERACTIVE)
        tenant_limiter = qos._tenant_limiters["t1"].interactive
        assert tenant_limiter.in_use == 1

        # Second acquire grabs a tenant slot, then blocks on the global slot.
        task = asyncio.create_task(qos.acquire("t1", BuildPriority.INTERACTIVE))
        await asyncio.sleep(0.05)  # let it reach the global-slot await
        assert tenant_limiter.in_use == 2  # tenant slot grabbed, now queued

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The queued acquire's tenant slot must be released, not leaked.
        assert tenant_limiter.in_use == 1

        await slot_a.release()

    @pytest.mark.asyncio
    async def test_per_tenant_limit_enforced(self):
        """Per-tenant limit returns 429 when exceeded."""
        config = BuildQoSConfig(
            interactive_slots=10,
            per_tenant_interactive=2,
            per_tenant_timeout=0.01,  # Very short timeout
        )
        qos = BuildQoS(config)

        # Acquire 2 slots (at limit)
        slot1 = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)
        slot2 = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Third should fail quickly
        with pytest.raises(TenantAtCapacityError) as exc_info:
            await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        assert exc_info.value.tenant_id == "tenant1"
        assert exc_info.value.limit == 2
        assert exc_info.value.status_code == 429

        # Different tenant should still succeed
        slot3 = await qos.acquire("tenant2", BuildPriority.INTERACTIVE)

        # Clean up
        await slot1.release()
        await slot2.release()
        await slot3.release()

    @pytest.mark.asyncio
    async def test_global_capacity_enforced(self):
        """Global capacity limit returns 429 when exceeded."""
        config = BuildQoSConfig(
            interactive_slots=2,
            per_tenant_interactive=10,
            interactive_queue_timeout=0.1,  # Short but reasonable timeout
        )
        qos = BuildQoS(config)

        # Acquire all global slots
        slot1 = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)
        slot2 = await qos.acquire("tenant2", BuildPriority.INTERACTIVE)

        try:
            # Third should fail (global capacity)
            with pytest.raises(GlobalCapacityError) as exc_info:
                await qos.acquire("tenant3", BuildPriority.INTERACTIVE)

            assert exc_info.value.tier == "interactive"
            assert exc_info.value.slots == 2
            assert exc_info.value.status_code == 429
        finally:
            # Clean up
            await slot1.release()
            await slot2.release()

    @pytest.mark.asyncio
    async def test_interactive_and_bulk_separate_pools(self):
        """Interactive and bulk have separate slot pools."""
        config = BuildQoSConfig(
            interactive_slots=1,
            bulk_slots=1,
            per_tenant_interactive=10,
            per_tenant_bulk=10,
        )
        qos = BuildQoS(config)

        # Acquire interactive slot
        slot1 = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Should still be able to acquire bulk slot
        slot2 = await qos.acquire("tenant1", BuildPriority.BULK)

        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 1
        assert metrics["bulk"]["active"] == 1

        await slot1.release()
        await slot2.release()

    @pytest.mark.asyncio
    async def test_slot_release_is_idempotent(self):
        """Releasing a slot multiple times is safe."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        slot = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Multiple releases should be safe
        await slot.release()
        await slot.release()
        await slot.release()

        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 0


class TestContextManager:
    """Tests for async context manager usage."""

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_success(self):
        """Context manager releases slot on normal exit."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        async with await qos.acquire("tenant1", BuildPriority.INTERACTIVE):
            metrics = qos.get_metrics()
            assert metrics["interactive"]["active"] == 1

        # Should be released after exiting context
        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 0

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_error(self):
        """Context manager releases slot on exception."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        with pytest.raises(ValueError):
            async with await qos.acquire("tenant1", BuildPriority.INTERACTIVE):
                metrics = qos.get_metrics()
                assert metrics["interactive"]["active"] == 1
                raise ValueError("test error")

        # Should be released after exception
        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 0


class TestMetrics:
    """Tests for QoS metrics."""

    @pytest.mark.asyncio
    async def test_rejection_metrics(self):
        """Rejection metrics are tracked."""
        config = BuildQoSConfig(
            interactive_slots=1,
            per_tenant_interactive=10,
            interactive_queue_timeout=0.1,  # Short but reasonable timeout
        )
        qos = BuildQoS(config)

        # Acquire all slots
        slot = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        try:
            # Trigger rejection
            with pytest.raises(GlobalCapacityError):
                await qos.acquire("tenant2", BuildPriority.INTERACTIVE)

            metrics = qos.get_metrics()
            assert metrics["interactive"]["rejected"] >= 1
        finally:
            await slot.release()

    @pytest.mark.asyncio
    async def test_per_tenant_metrics(self):
        """Per-tenant metrics are available."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        # Create some activity
        slot = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Get tenant metrics
        tenant_metrics = qos.get_tenant_metrics("tenant1")
        assert tenant_metrics is not None
        assert tenant_metrics["tenant_id"] == "tenant1"
        assert tenant_metrics["interactive"]["active"] == 1

        await slot.release()

    def test_tenant_metrics_not_found(self):
        """Unknown tenant returns None."""
        config = BuildQoSConfig()
        qos = BuildQoS(config)

        assert qos.get_tenant_metrics("nonexistent") is None


class TestErrorResponses:
    """Tests for error response formatting."""

    def test_tenant_at_capacity_error_dict(self):
        """TenantAtCapacityError formats correctly."""
        error = TenantAtCapacityError(
            tenant_id="acme",
            limit=5,
            active=5,
            retry_after=3.0,
        )

        d = error.to_dict()
        assert d["error"] == "tenant_at_capacity"
        assert d["tenant_id"] == "acme"
        assert d["limit"] == 5
        assert d["active"] == 5
        assert d["retry_after_seconds"] == 3.0

    def test_global_capacity_error_dict(self):
        """GlobalCapacityError formats correctly."""
        error = GlobalCapacityError(
            tier="interactive",
            slots=16,
            queue_timeout_seconds=5.0,
            queue_wait_ms=4500.0,
            retry_after=5.0,
        )

        d = error.to_dict()
        assert d["error"] == "too_many_requests"
        assert d["tier"] == "interactive"
        assert d["slots"] == 16
        assert d["queue_timeout_seconds"] == 5.0
        assert d["queue_wait_ms"] == 4500.0
        assert d["retry_after_seconds"] == 5.0

    def test_quota_exceeded_error_dict(self):
        """TenantQuotaExceededError formats correctly."""
        error = TenantQuotaExceededError(
            tenant_id="acme",
            used_bytes=1000000,
            limit_bytes=900000,
            reset_in_seconds=3600.0,
        )

        d = error.to_dict()
        assert d["error"] == "quota_exceeded"
        assert d["tenant_id"] == "acme"
        assert d["used_bytes"] == 1000000
        assert d["limit_bytes"] == 900000
        assert d["reset_in_seconds"] == 3600.0


class TestSingleton:
    """Tests for module-level singleton management."""

    def test_get_set_reset(self):
        """Singleton get/set/reset work correctly."""
        # Should start as None
        reset_build_qos()
        assert get_build_qos() is None

        # Set a QoS
        config = BuildQoSConfig()
        qos = BuildQoS(config)
        set_build_qos(qos)
        assert get_build_qos() is qos

        # Reset
        reset_build_qos()
        assert get_build_qos() is None


class TestConcurrentAcquisition:
    """Tests for concurrent slot acquisition behavior."""

    @pytest.mark.asyncio
    async def test_many_concurrent_requests(self):
        """Many concurrent requests correctly share slots."""
        config = BuildQoSConfig(
            interactive_slots=4,
            per_tenant_interactive=2,
            interactive_queue_timeout=1.0,
        )
        qos = BuildQoS(config)

        slots_acquired = []
        errors = []

        async def try_acquire(tenant_id: str, n: int):
            try:
                slot = await qos.acquire(tenant_id, BuildPriority.INTERACTIVE)
                slots_acquired.append((tenant_id, n))
                # Hold slot briefly
                await asyncio.sleep(0.01)
                await slot.release()
            except TenantAtCapacityError:
                errors.append(("tenant_cap", tenant_id, n))
            except GlobalCapacityError:
                errors.append(("global_cap", tenant_id, n))

        # Launch many concurrent requests across tenants
        tasks = []
        for tenant in ["t1", "t2", "t3"]:
            for i in range(5):
                tasks.append(try_acquire(tenant, i))

        await asyncio.gather(*tasks)

        # Some should have succeeded, some may have failed
        assert len(slots_acquired) > 0

        # All slots should be released
        metrics = qos.get_metrics()
        assert metrics["interactive"]["active"] == 0
        assert metrics["bulk"]["active"] == 0

    @pytest.mark.asyncio
    async def test_queue_wait_time_recorded(self):
        """Queue wait time is tracked in metrics."""
        config = BuildQoSConfig(
            interactive_slots=1,
            per_tenant_interactive=10,
            interactive_queue_timeout=1.0,
        )
        qos = BuildQoS(config)

        # Hold one slot
        slot1 = await qos.acquire("tenant1", BuildPriority.INTERACTIVE)

        # Start a waiter
        async def wait_for_slot():
            slot = await qos.acquire("tenant2", BuildPriority.INTERACTIVE)
            await slot.release()

        waiter = asyncio.create_task(wait_for_slot())

        # Let it wait a bit
        await asyncio.sleep(0.05)

        # Release first slot so waiter can proceed
        await slot1.release()

        # Wait for waiter to complete
        await waiter

        # Check queue wait was recorded
        metrics = qos.get_metrics()
        assert metrics["interactive"]["queue_wait_count"] >= 1

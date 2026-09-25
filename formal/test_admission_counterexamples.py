"""Replay Admission.tla's counterexamples against TenantRegistry.

Each test passes while its bug exists (asserts the violating outcome).

    uv run pytest formal/ -v
"""

from __future__ import annotations

import asyncio

import pytest


def _registry():
    registry_mod = pytest.importorskip("strata.tenant_registry")
    return registry_mod.TenantRegistry(default_interactive_slots=1, default_bulk_slots=1)


def _evict(registry, tenant_id):
    """Touch MAX_TRACKED_TENANTS other tenants, as ordinary traffic would."""
    from strata.tenant_registry import MAX_TRACKED_TENANTS

    for i in range(MAX_TRACKED_TENANTS):
        registry.get_or_create_limiters(f"other-{i}")
    assert tenant_id not in registry._quotas


def test_evicted_limiter_lets_a_tenant_exceed_its_quota():
    """Admission_Eviction / WithinQuota.

    Tenant "a" has 1 interactive slot and uses it. LRU pressure evicts its
    quotas while the stream is live, and its next request gets a fresh
    limiter, so "a" now holds 2 slots on a quota of 1.
    """
    registry = _registry()

    async def scenario():
        first, _ = registry.get_or_create_limiters("a")
        assert await first.acquire(timeout=0.1)
        _evict(registry, "a")
        second, _ = registry.get_or_create_limiters("a")
        return await second.acquire(timeout=0.1), first.in_use + second.in_use

    acquired_again, held = asyncio.run(scenario())
    assert acquired_again
    assert held == 2


def test_evicted_limiter_is_invisible_to_the_shutdown_drain():
    """Admission_Eviction / DrainSeesAll.

    _graceful_shutdown waits while aggregate_limiter_usage() reports
    in-flight scans, then cancels stream tasks. A stream on an evicted
    limiter is not counted, so shutdown can cancel it mid-stream.
    """
    registry = _registry()

    async def scenario():
        limiter, _ = registry.get_or_create_limiters("a")
        assert await limiter.acquire(timeout=0.1)
        _evict(registry, "a")
        interactive_in_use, _, bulk_in_use, _ = registry.aggregate_limiter_usage()
        return limiter.in_use, interactive_in_use + bulk_in_use

    live, counted = asyncio.run(scenario())
    assert live == 1
    assert counted == 0

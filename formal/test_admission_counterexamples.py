"""Replay Admission.tla's counterexamples against ResizableLimiter / TenantRegistry.

Each test passes while its bug exists (asserts the violating outcome).

The lost-wakeup test needs CPython 3.12, whose asyncio.Condition does not
re-notify when a notified waiter is cancelled (fixed in 3.13). It loads
adaptive_concurrency.py by path, which needs only the standard library,
so it runs without the project environment:

    uv run --no-project --python 3.12 --with pytest \
        pytest formal/test_admission_counterexamples.py -k wakeup
    uv run pytest formal/ -v     # everything else
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "strata" / "adaptive_concurrency.py"


def _resizable_limiter():
    spec = importlib.util.spec_from_file_location("_strata_adaptive", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ResizableLimiter


@pytest.mark.skipif(
    sys.version_info >= (3, 13),
    reason="asyncio.Condition re-notifies on cancel since CPython 3.13",
)
def test_cancelled_waiter_swallows_the_wakeup_on_python_312():
    """Admission_Py312 / NoLostWakeup.

    Capacity 1. B and C queue behind the holder. The holder releases, and
    notify(1) picks B, but B's request is cancelled (client disconnect)
    before it runs. Nobody else is notified: C sleeps out its whole
    deadline while the slot is free, and only the timeout handler's
    re-check lets it in. A newer request arriving meanwhile would take the
    slot and C would get a 429.
    """
    ResizableLimiter = _resizable_limiter()
    deadline = 0.5

    async def scenario():
        limiter = ResizableLimiter(1)
        assert await limiter.acquire()
        b = asyncio.create_task(limiter.acquire(timeout=5.0))
        c = asyncio.create_task(limiter.acquire(timeout=deadline))
        await asyncio.sleep(0.01)  # both queued

        started = time.monotonic()
        await limiter.release()  # notify(1) -> B
        b.cancel()  # B's client disconnects before B runs
        await asyncio.sleep(deadline / 2)
        idle_while_queued = limiter.in_use == 0 and not c.done()
        acquired = await c
        return idle_while_queued, acquired, time.monotonic() - started

    idle_while_queued, acquired, waited = asyncio.run(scenario())
    assert idle_while_queued
    assert acquired
    assert waited >= deadline * 0.9  # the full deadline, not ~0


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

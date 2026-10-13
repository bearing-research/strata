"""Pre-warm hints: a machine started before the job that needs it.

A hint starts one machine for the tenant unless one is warm or starting, under the same
caps as a job's machine, and an unused one is reaped like any idle machine.
"""

import asyncio

import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import MachineType
from strata_pool.types import WorkerState


async def _settled(pool) -> None:
    """Wait for the background starts and boots a hint left running."""
    while pool._tasks:
        await asyncio.gather(*list(pool._tasks), return_exceptions=True)


def _states(pool, tenant_id: str) -> list[WorkerState]:
    return [w.state for w in pool.store.list_workers() if w.tenant_id == tenant_id]


async def test_a_hint_starts_a_warm_machine_for_its_tenant(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend)

    assert await pool.prewarm("acme", "cpu") == "reserved"
    await _settled(pool)

    assert backend.started == ["machine-1"]
    assert _states(pool, "acme") == [WorkerState.WARM]


async def test_a_second_hint_starts_nothing_while_one_is_starting_or_warm(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend)

    assert await pool.prewarm("acme", "cpu") == "reserved"
    # Still starting: the first hint's machine absorbs the second.
    assert await pool.prewarm("acme", "cpu") == "no_demand"
    await _settled(pool)
    assert await pool.prewarm("acme", "cpu") == "no_demand"

    assert backend.started == ["machine-1"]


async def test_another_tenants_warm_machine_does_not_satisfy_a_hint(make_pool):
    """A machine belongs to one tenant for its life, so each tenant warms its own."""
    backend = FakeBackend()
    pool = make_pool(backend)

    await pool.prewarm("acme", "cpu")
    assert await pool.prewarm("globex", "cpu") == "reserved"
    await _settled(pool)

    assert _states(pool, "acme") == [WorkerState.WARM]
    assert _states(pool, "globex") == [WorkerState.WARM]


async def test_the_next_job_runs_on_the_warmed_machine(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend)
    await pool.prewarm("acme", "cpu")
    await _settled(pool)

    job = await pool.submit("acme", "cpu", b"work")
    done = await pool.wait(job.id)

    assert done.result == b"done:work"
    assert backend.started == ["machine-1"]


async def test_a_hint_respects_the_tenants_cap(make_pool):
    """A tenant at its type's cap gets no extra machine from a hint, however busy it is."""
    release = asyncio.Event()

    async def held(request):
        await release.wait()
        return FakeWorkers._echo(request)

    backend = FakeBackend()
    pool = make_pool(
        backend,
        FakeWorkers(held),
        [MachineType(name="cpu", image="worker:latest", max_workers=1)],
    )
    job = await pool.submit("acme", "cpu", b"work")
    for _ in range(200):
        if _states(pool, "acme") == [WorkerState.BUSY]:
            break
        await asyncio.sleep(0.01)
    assert _states(pool, "acme") == [WorkerState.BUSY]

    assert await pool.prewarm("acme", "cpu") == "tenant_cap"

    release.set()
    await pool.wait(job.id)
    assert backend.started == ["machine-1"]


async def test_a_hint_respects_the_fleet_cap(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend, max_workers_total=1)

    assert await pool.prewarm("acme", "cpu") == "reserved"
    assert await pool.prewarm("globex", "cpu") == "fleet_cap"
    await _settled(pool)

    assert backend.started == ["machine-1"]


async def test_an_unused_warmed_machine_is_reaped_after_its_cool_down(make_pool):
    """No job came: the machine ages from its boot and goes like any idle one."""
    now = [1000.0]
    backend = FakeBackend()
    pool = make_pool(
        backend,
        machine_types=[MachineType(name="cpu", image="worker:latest", cool_down_seconds=60)],
        wall=lambda: now[0],
    )
    await pool.prewarm("acme", "cpu")
    await _settled(pool)

    now[0] += 30
    assert await pool.reap_idle_workers() == 0
    now[0] += 31
    assert await pool.reap_idle_workers() == 1

    assert backend.stopped == ["machine-1"]


async def test_a_hint_for_an_unknown_machine_type_is_refused(make_pool):
    pool = make_pool()

    with pytest.raises(ValueError, match="unknown machine type"):
        await pool.prewarm("acme", "h100")

    assert pool.store.list_workers() == []

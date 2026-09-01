"""A ceiling on the whole fleet, not just on one tenant.

`MachineType.max_workers` caps a single tenant. Without a global cap the
fleet is that number times however many tenants show up, which is the shape
of a cloud bill nobody predicted.
"""

import logging

from conftest import FakeBackend
from strata_pool import JobState, MachineType


def _spec(**kwargs) -> MachineType:
    return MachineType(
        name=kwargs.pop("name", "cpu"),
        image=kwargs.pop("image", "w"),
        **kwargs,
    )


async def test_tenants_together_cannot_exceed_the_fleet_ceiling(make_pool):
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=5, boot_timeout_seconds=60)],
        max_workers_total=2,
    )

    for tenant in ("a", "b", "c", "d"):
        await pool.submit(tenant_id=tenant, machine_type="cpu", payload=b"work")

    assert len(backend.started) == 2
    assert len(pool.store.list_workers()) == 2


async def test_the_ceiling_does_not_lose_the_work_it_defers(make_pool):
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=5, boot_timeout_seconds=60)],
        max_workers_total=1,
    )

    await pool.submit(tenant_id="a", machine_type="cpu", payload=b"one")
    deferred = await pool.submit(tenant_id="b", machine_type="cpu", payload=b"two")

    assert pool.store.get_job(deferred.id).state is JobState.QUEUED
    assert pool.store.count_queued("cpu", "b") == 1


async def test_capacity_freed_by_one_tenant_becomes_available_to_another(make_pool):
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=5, cool_down_seconds=0.0)],
        max_workers_total=1,
    )

    first = await pool.submit(tenant_id="a", machine_type="cpu", payload=b"one")
    await pool.wait(first.id)

    blocked = await pool.submit(tenant_id="b", machine_type="cpu", payload=b"two")
    assert pool.store.get_job(blocked.id).state is JobState.QUEUED

    await pool.reap_idle_workers()  # tenant a's machine goes away
    await pool.submit(tenant_id="b", machine_type="cpu", payload=b"three")

    assert (await pool.wait(blocked.id)).state is JobState.COMPLETED


async def test_hitting_the_ceiling_is_never_silent(make_pool, caplog):
    """A capped fleet looks exactly like a slow queue from the outside."""
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=5, boot_timeout_seconds=60)],
        max_workers_total=1,
    )

    await pool.submit(tenant_id="a", machine_type="cpu", payload=b"one")
    with caplog.at_level(logging.WARNING):
        await pool.submit(tenant_id="b", machine_type="cpu", payload=b"two")

    assert "global cap" in caplog.text


async def test_no_ceiling_is_the_default(make_pool):
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=1, boot_timeout_seconds=60)],
    )

    for tenant in ("a", "b", "c"):
        await pool.submit(tenant_id=tenant, machine_type="cpu", payload=b"work")

    assert len(backend.started) == 3, "each tenant still gets its own machine"

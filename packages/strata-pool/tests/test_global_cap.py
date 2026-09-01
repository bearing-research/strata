"""A ceiling on the whole fleet, not just on one tenant.

`MachineType.max_workers` caps a single tenant. Without a global cap the
fleet is that number times however many tenants show up, which is the shape
of a cloud bill nobody predicted.
"""

import asyncio
import logging

from conftest import FakeBackend
from strata_pool import JobState, MachineType
from strata_pool.types import Job


class SlowStartBackend(FakeBackend):
    """Yields inside start(), which is where a real provider spends its time."""

    async def start(self, spec, env=None):
        await asyncio.sleep(0)
        return await super().start(spec, env)


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

    # Asserting on the queue alone would pass with no cap at all: the machine
    # never boots either way. The cap is what withheld the second machine.
    assert backend.started == ["machine-1"]
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
    assert backend.started == ["machine-1"], "the cap, not a slow boot, is what deferred it"

    # Reaping tenant a's idle machine frees the only slot. Nothing else will
    # ever run for tenant b: its jobs are queued and capacity is only
    # reconsidered when a tenant submits. The freed slot has to be offered.
    await pool.reap_idle_workers()

    assert (await pool.wait(blocked.id)).state is JobState.COMPLETED
    assert backend.started == ["machine-1", "machine-2"]


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


async def test_two_tenants_starting_at_once_cannot_overshoot_the_ceiling(make_pool):
    """Starting a machine awaits the backend, and another tenant can start its
    own in that window. Headroom decided once, before the loop, lets two
    callers each spend the same last slot.

    Driven through `_ensure_capacity` directly with the queues pre-loaded,
    because that is the only way to get two provisioning loops overlapping on
    purpose rather than by luck.
    """
    backend = SlowStartBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=5, boot_timeout_seconds=60)],
        max_workers_total=2,
    )

    for tenant in ("a", "b"):
        for n in range(2):
            pool.store.save_job(
                Job(
                    id=f"job-{tenant}-{n}",
                    tenant_id=tenant,
                    machine_type="cpu",
                    payload=b"work",
                    state=JobState.QUEUED,
                    submitted_at=100.0,
                )
            )

    await asyncio.gather(
        pool._ensure_capacity("cpu", "a"),
        pool._ensure_capacity("cpu", "b"),
    )

    assert len(backend.started) <= 2, (
        f"started {len(backend.started)} machines for a fleet cap of 2"
    )


async def test_no_warning_when_nothing_was_wanted(make_pool, caplog):
    """The fleet can sit at or over its ceiling with an empty queue. Warning
    then trains operators to ignore the one warning that matters."""
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], max_workers_total=1)

    job = await pool.submit(tenant_id="a", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    with caplog.at_level(logging.WARNING):
        await pool._ensure_capacity("cpu", "a")

    assert "global cap" not in caplog.text

"""Placing jobs on machines: reuse, affinity, and when to start one."""

import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, WorkerState


async def test_a_job_with_no_fleet_starts_a_machine_and_runs(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend=backend)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.COMPLETED
    assert done.result == b"done:work"
    assert backend.started == ["machine-1"]


async def test_a_warm_machine_is_reused_rather_than_a_second_one_started(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend=backend)

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    ran_first = await pool.wait(first.id)
    second = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    done = await pool.wait(second.id)

    assert done.state is JobState.COMPLETED
    assert done.worker_id == ran_first.worker_id
    assert backend.started == ["machine-1"]


async def test_a_session_returns_to_the_machine_that_served_it(make_pool):
    pool = make_pool(machine_types=[MachineType(name="cpu", image="w", max_workers=2)])

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one", session_id="s1")
    second = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"two", session_id="s2"
    )
    await pool.wait(first.id)
    await pool.wait(second.id)

    again = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"three", session_id="s1"
    )
    done = await pool.wait(again.id)

    assert done.worker_id == pool.store.get_job(first.id).worker_id


async def test_an_affinity_miss_falls_back_to_any_warm_machine(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend=backend)

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one", session_id="s1")
    await pool.wait(first.id)

    stranger = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"two", session_id="s2"
    )
    done = await pool.wait(stranger.id)

    assert done.state is JobState.COMPLETED
    assert backend.started == ["machine-1"], "an unseen session must not force a cold start"


async def test_the_fleet_stops_growing_at_max_workers(make_pool):
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", max_workers=2, boot_timeout_seconds=60)],
    )

    for _ in range(5):
        await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")

    assert len(backend.started) == 2
    assert pool.store.count_queued("cpu") == 5


async def test_queued_work_drains_onto_a_machine_as_it_warms(make_pool):
    backend = FakeBackend(healthy_after_polls=1)
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1)],
    )

    jobs = [
        await pool.submit(tenant_id="acme", machine_type="cpu", payload=f"j{i}".encode())
        for i in range(3)
    ]

    for job in jobs:
        assert (await pool.wait(job.id)).state is JobState.COMPLETED
    assert backend.started == ["machine-1"]


async def test_a_backend_that_cannot_provision_leaves_no_worker_behind(make_pool):
    backend = FakeBackend(fail_start=True)
    pool = make_pool(backend=backend)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")

    assert pool.store.list_workers() == []
    assert pool.store.get_job(job.id).state is JobState.QUEUED


async def test_a_machine_that_never_boots_is_stopped_and_the_work_stays_queued(make_pool):
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1, boot_timeout_seconds=0)],
    )

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    with pytest.raises(TimeoutError):
        await pool.wait(job.id, timeout=0.2)

    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []
    assert pool.store.get_job(job.id).state is JobState.QUEUED

    # No scaler yet: the next submit is what retries the start.
    await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"more")
    assert backend.started == ["machine-1", "machine-2"]


async def test_an_unknown_machine_type_is_rejected(make_pool):
    pool = make_pool()
    with pytest.raises(ValueError, match="unknown machine type"):
        await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"work")


async def test_a_busy_machine_does_not_take_a_second_job(make_pool):
    """The worker is released only when its job is done."""
    workers = FakeWorkers()
    pool = make_pool(
        workers=workers,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1)],
    )

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    ran = await pool.wait(first.id)

    worker = pool.store.get_worker(ran.worker_id)
    worker.state = WorkerState.BUSY
    pool.store.save_worker(worker)

    second = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    assert pool.store.get_job(second.id).state is JobState.QUEUED
    assert len(workers.requests) == 1

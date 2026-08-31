"""What the pool does with each way a job can end."""

import asyncio
import sqlite3

import httpx
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, WorkerState


async def test_the_payload_reaches_the_worker_verbatim(make_pool):
    workers = FakeWorkers()
    pool = make_pool(workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"\x00binary\xff")
    await pool.wait(job.id)

    request = workers.requests[0]
    assert request.url.path == "/execute"
    assert request.content == b"\x00binary\xff"


async def test_a_worker_error_fails_the_job_and_keeps_the_machine(make_pool):
    backend = FakeBackend()
    workers = FakeWorkers(lambda request: httpx.Response(500, text="boom"))
    pool = make_pool(backend=backend, workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert "500" in done.error
    assert backend.stopped == [], "a job that fails on a healthy machine is not the machine's fault"
    assert pool.store.get_worker(done.worker_id).state is WorkerState.WARM


async def test_an_unreachable_worker_is_stopped_and_forgotten(make_pool):
    backend = FakeBackend()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    pool = make_pool(backend=backend, workers=FakeWorkers(refuse))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert "unreachable" in done.error
    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_job_that_runs_too_long_times_out_and_the_machine_is_retired(make_pool):
    """Nothing can tell the worker to stop, so it must not take the next job."""
    backend = FakeBackend()

    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    pool = make_pool(
        backend=backend,
        workers=FakeWorkers(stall),
        machine_types=[MachineType(name="cpu", image="w", job_timeout_seconds=0.05)],
    )

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.TIMED_OUT
    assert "0.05s timeout" in done.error
    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_worker_that_stops_answering_is_not_mistaken_for_a_slow_job(make_pool):
    """A wedged machine surfaces as ConnectTimeout, which is a dead machine.

    Calling it a slow job would return it to the fleet, and every job after it
    would be handed to a corpse one timeout at a time.
    """
    backend = FakeBackend()

    def black_hole(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route", request=request)

    pool = make_pool(backend=backend, workers=FakeWorkers(black_hole))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert "unreachable" in done.error
    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_job_outlives_its_deadline_even_if_bytes_keep_arriving(make_pool):
    """httpx's read timeout measures gaps between bytes, not total runtime."""

    async def dribble(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5)
        return httpx.Response(200, content=b"ok")

    pool = make_pool(
        workers=FakeWorkers(dribble),
        machine_types=[MachineType(name="cpu", image="w", job_timeout_seconds=0.05)],
    )

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id, timeout=2.0)

    assert done.state is JobState.TIMED_OUT


async def test_an_unexpected_failure_releases_the_machine_instead_of_wedging_it(make_pool):
    """A worker that misbehaves in a way we cannot classify still cannot leak.

    A job stuck RUNNING with a worker stuck BUSY would bill forever and hold a
    slot against max_workers with nothing left alive to release it.
    """
    backend = FakeBackend()

    def garbled(request: httpx.Request) -> httpx.Response:
        raise httpx.DecodingError("corrupt body", request=request)

    pool = make_pool(backend=backend, workers=FakeWorkers(garbled))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id, timeout=2.0)

    assert done.state is JobState.FAILED
    assert "pool failed to run the job" in done.error
    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_store_that_rejects_the_bill_still_releases_the_machine(make_pool):
    """A store that refuses a write is a problem; a machine stuck BUSY with
    nothing left alive to release it is a permanent one."""
    pool = make_pool()

    original = pool.store.record_usage

    def explode(event):
        pool.store.record_usage = original
        raise sqlite3.OperationalError("database is locked")

    pool.store.record_usage = explode

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id, timeout=2.0)

    assert done.state is JobState.COMPLETED
    assert pool.store.list_usage() == [], "the bill is the write that failed"
    assert [w.state for w in pool.store.list_workers()] == [WorkerState.WARM]


async def test_a_per_job_timeout_overrides_the_machine_default(make_pool):
    seen: list[float | None] = []

    def record_timeout(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, content=b"ok")

    pool = make_pool(
        workers=FakeWorkers(record_timeout),
        machine_types=[MachineType(name="cpu", image="w", job_timeout_seconds=300.0)],
    )

    job = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"work", timeout_seconds=7.0
    )
    await pool.wait(job.id)

    assert seen == [7.0]


async def test_a_finished_job_records_when_it_ran(make_pool):
    pool = make_pool()

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.submitted_at <= done.started_at <= done.completed_at
    assert done.worker_id is not None

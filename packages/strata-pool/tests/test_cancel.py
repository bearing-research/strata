"""Cancelling a job: it stops the job, never the machine."""

import asyncio

import httpx
import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, WorkerState
from strata_pool.types import Job, Worker, new_auth_token


async def _until(predicate) -> None:
    for _ in range(4000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


class StoppableWorker:
    """A worker whose long job runs until its cancel route is called, as `strata-worker`'s does."""

    def __init__(self, *, honours_cancel: bool = True):
        self.honours_cancel = honours_cancel
        self.executing = asyncio.Event()
        self.killed = asyncio.Event()
        self.finish = asyncio.Event()
        self.cancels: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/execute":
            if request.content != b"long":
                return httpx.Response(200, content=b"done:" + request.content)
            self.executing.set()
            await self.finish.wait()
            if self.killed.is_set():
                return httpx.Response(500, text="harness killed")
            return httpx.Response(200, content=b"a result nobody wants")
        if request.url.path.endswith("/cancel"):
            self.cancels.append(request)
            if not self.honours_cancel:
                return httpx.Response(404)
            self.killed.set()
            self.finish.set()
            return httpx.Response(200, json={"cancelled": True})
        return httpx.Response(404)


async def test_a_queued_job_cancelled_never_starts_a_machine(make_pool):
    release = asyncio.Event()

    async def run_then_vanish(request: httpx.Request) -> httpx.Response:
        await release.wait()
        raise httpx.ConnectError("machine gone", request=request)

    backend = FakeBackend()
    workers = FakeWorkers(run_then_vanish)
    pool = make_pool(
        backend=backend,
        workers=workers,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1)],
    )

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"first")
    queued = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"second")
    assert pool.store.get_job(queued.id).state is JobState.QUEUED

    cancelled = await pool.cancel(queued.id, "build-2")
    assert cancelled.state is JobState.CANCELLED

    # The first machine dies, freeing the tenant's one slot: a job still queued would start
    # a second machine for itself here.
    release.set()
    await pool.wait(first.id)

    assert backend.started == ["machine-1"]
    assert [request.content for request in workers.requests] == [b"first"]
    assert pool.store.get_job(queued.id).state is JobState.CANCELLED
    assert [event.job_id for event in pool.store.list_usage()] == [first.id], (
        "a job that never ran is not metered"
    )


async def test_a_running_jobs_cancel_reaches_its_machine_and_the_machine_stays_warm(make_pool):
    worker = StoppableWorker()
    backend = FakeBackend()
    pool = make_pool(backend=backend, workers=FakeWorkers(worker))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"long")
    await worker.executing.wait()

    cancelled = await pool.cancel(job.id, "build-1")

    assert cancelled.state is JobState.CANCELLED
    machine = pool.store.get_worker(cancelled.worker_id)
    [request] = worker.cancels
    assert str(request.url) == f"{machine.endpoint}/v1/executions/build-1/cancel"
    assert request.headers["Authorization"] == f"Bearer {machine.auth_token}"

    await _until(lambda: pool.store.get_worker(machine.id).state is WorkerState.WARM)
    assert backend.stopped == []
    assert pool.store.get_job(job.id).state is JobState.CANCELLED
    [event] = pool.store.list_usage("acme")
    assert event.job_id == job.id
    assert event.terminal_state is JobState.CANCELLED
    assert event.worker_id == machine.id
    assert event.duration_ms > 0

    after = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"next")
    done = await pool.wait(after.id)
    assert done.state is JobState.COMPLETED
    assert done.worker_id == machine.id, "the next job runs on the machine the cancel kept"
    assert backend.started == ["machine-1"]


async def test_a_job_cancelled_before_its_machine_got_it_never_runs(make_pool):
    workers = FakeWorkers()
    pool = make_pool(workers=workers)
    warm_up = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"warm-up")
    await pool.wait(warm_up.id)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"never")
    assert pool.store.get_job(job.id).state is JobState.DISPATCHED

    cancelled = await pool.cancel(job.id, "build-1")

    assert cancelled.state is JobState.CANCELLED
    await _until(lambda: pool.store.get_worker(cancelled.worker_id).state is WorkerState.WARM)
    assert [request.content for request in workers.requests] == [b"warm-up"]
    assert [event.job_id for event in pool.store.list_usage()] == [warm_up.id]
    assert pool.store.get_job(job.id).lease_owner is None


async def test_the_cancel_stands_when_the_machine_ignores_it(make_pool):
    """A worker without the cancel route runs the job out; the job is still cancelled."""
    worker = StoppableWorker(honours_cancel=False)
    backend = FakeBackend()
    pool = make_pool(backend=backend, workers=FakeWorkers(worker))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"long")
    await worker.executing.wait()
    await pool.cancel(job.id, "build-1")
    worker.finish.set()

    machine_id = pool.store.get_job(job.id).worker_id
    await _until(lambda: pool.store.get_worker(machine_id).state is WorkerState.WARM)
    after = pool.store.get_job(job.id)
    assert after.state is JobState.CANCELLED
    assert after.result is None
    assert [event.terminal_state for event in pool.store.list_usage()] == [JobState.CANCELLED]
    assert backend.stopped == []


async def test_cancelling_a_finished_job_changes_nothing(make_pool):
    workers = FakeWorkers()
    pool = make_pool(workers=workers)
    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    after = await pool.cancel(job.id, "build-1")

    assert after.state is JobState.COMPLETED
    assert after.result == b"done:work"
    assert [request.url.path for request in workers.requests] == ["/execute"]
    assert len(pool.store.list_usage()) == 1


async def test_a_build_id_that_could_leave_the_cancel_path_is_refused(make_pool):
    pool = make_pool(backend=FakeBackend(never_healthy=True))
    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")

    with pytest.raises(ValueError, match="invalid build id"):
        await pool.cancel(job.id, "../execute")

    assert pool.store.get_job(job.id).state is JobState.QUEUED


async def test_a_cancel_never_reaches_a_machine_that_moved_on_to_another_job(make_pool):
    """Between the cancel and the post, the machine may have answered and taken the next job."""
    workers = FakeWorkers()
    pool = make_pool(workers=workers)
    pool.store.save_worker(
        Worker(
            id="w1",
            machine_type="cpu",
            tenant_id="acme",
            backend="fake",
            state=WorkerState.BUSY,
            created_at=1.0,
            endpoint="http://w1.test",
            auth_token=new_auth_token(),
            current_job_id="the-next-job",
        )
    )
    pool.store.save_job(
        Job(
            id="j",
            tenant_id="acme",
            machine_type="cpu",
            payload=b"x",
            state=JobState.RUNNING,
            submitted_at=1.0,
            started_at=2.0,
            worker_id="w1",
        )
    )

    pool.store.save_worker(
        Worker(
            id="w2",
            machine_type="cpu",
            tenant_id="acme",
            backend="fake",
            state=WorkerState.BUSY,
            created_at=1.0,
            endpoint="http://w2.test",
            auth_token=new_auth_token(),
            current_job_id="k",
        )
    )
    pool.store.save_job(
        Job(
            id="k",
            tenant_id="acme",
            machine_type="cpu",
            payload=b"y",
            state=JobState.RUNNING,
            submitted_at=1.0,
            started_at=2.0,
            worker_id="w2",
        )
    )

    cancelled = await pool.cancel("j", "build-1")
    assert cancelled.state is JobState.CANCELLED
    assert workers.requests == []

    # The same cancel does reach a machine still on its job.
    assert (await pool.cancel("k", "build-2")).state is JobState.CANCELLED
    assert [str(r.url) for r in workers.requests] == ["http://w2.test/v1/executions/build-2/cancel"]

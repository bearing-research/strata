"""What the pool does with each way a job can end."""

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


async def test_a_job_that_runs_too_long_times_out(make_pool):
    def stall(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    pool = make_pool(
        workers=FakeWorkers(stall),
        machine_types=[MachineType(name="cpu", image="w", job_timeout_seconds=0.05)],
    )

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.TIMED_OUT
    assert "0.05s timeout" in done.error


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

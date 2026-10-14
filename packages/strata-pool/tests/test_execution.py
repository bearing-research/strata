"""What the pool does with each way a job can end."""

import asyncio
import json
import sqlite3

import httpx
import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, WorkerState
from strata_pool.types import STATUS_IN_BODY_HEADER


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


def _streamed(status_code: int, content: str) -> httpx.Response:
    """What a heartbeat worker answers: 200, spaces, then the real status and body."""
    body = b"   " + json.dumps({"status_code": status_code, "content": content}).encode()
    return httpx.Response(200, content=body, headers={STATUS_IN_BODY_HEADER: "1"})


async def test_a_streamed_answer_completes_with_the_body_inside(make_pool):
    workers = FakeWorkers(lambda request: _streamed(200, '{"success": true}'))
    pool = make_pool(workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.COMPLETED
    assert done.result == b'{"success": true}'


async def test_a_streamed_failure_fails_the_job_and_keeps_the_machine(make_pool):
    """The worker sent 200 to keep the proxy open; the status inside is what counts."""
    backend = FakeBackend()
    workers = FakeWorkers(lambda request: _streamed(502, '{"detail": "upload failed"}'))
    pool = make_pool(backend=backend, workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert done.error == 'worker returned 502: {"detail": "upload failed"}'
    assert backend.stopped == []
    assert pool.store.get_worker(done.worker_id).state is WorkerState.WARM


@pytest.mark.parametrize("body", [b"   not json", b'  {"content": "no status"}'])
async def test_a_malformed_streamed_answer_fails_the_job_and_retires_the_machine(make_pool, body):
    backend = FakeBackend()
    workers = FakeWorkers(
        lambda request: httpx.Response(200, content=body, headers={STATUS_IN_BODY_HEADER: "1"})
    )
    pool = make_pool(backend=backend, workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert "pool failed to run the job" in done.error
    assert backend.stopped == ["machine-1"]


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
    """A ConnectTimeout means a dead machine; treating it as slow would feed it every next job."""
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
    """An unclassified failure must not leave the worker BUSY, billing and holding a slot."""
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
    """A failed usage write must not leave the machine stuck BUSY."""
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


async def test_a_per_job_timeout_cannot_extend_the_machine_default(make_pool):
    """Otherwise a caller holds a machine, and a fleet slot, past the operator's limit."""
    seen: list[float | None] = []

    def record_timeout(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, content=b"ok")

    pool = make_pool(
        workers=FakeWorkers(record_timeout),
        machine_types=[MachineType(name="cpu", image="w", job_timeout_seconds=300.0)],
    )

    job = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"work", timeout_seconds=1e12
    )
    await pool.wait(job.id)

    assert seen == [300.0]


async def test_a_finished_job_records_when_it_ran(make_pool):
    pool = make_pool()

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.submitted_at <= done.started_at <= done.completed_at
    assert done.worker_id is not None

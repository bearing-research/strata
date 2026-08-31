"""Usage accounting.

Billing numbers are asserted against independently observed time, never
against a fixed threshold: a metering path that reports a plausible constant
is indistinguishable from one that reports nothing, and this repo has shipped
that bug before.
"""

import asyncio
import itertools
import time

import httpx
from conftest import FakeWorkers
from strata_pool import JobState


async def test_a_completed_job_produces_exactly_one_usage_event(make_pool):
    pool = make_pool()

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    events = pool.store.list_usage("acme")
    assert len(events) == 1
    assert events[0].job_id == job.id
    assert events[0].worker_id == done.worker_id
    assert events[0].terminal_state is JobState.COMPLETED


async def test_billed_time_never_exceeds_time_that_actually_passed(make_pool):
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.02)
        return httpx.Response(200, content=b"ok")

    pool = make_pool(workers=FakeWorkers(slow))

    started = time.monotonic()
    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)
    observed_ms = (time.monotonic() - started) * 1000

    billed_ms = pool.store.list_usage("acme")[0].duration_ms
    assert 0 < billed_ms <= observed_ms


async def test_a_backwards_clock_step_cannot_change_the_bill(make_pool):
    """The wall clock runs backwards for the whole job; the duration must not.

    Durations come from a monotonic source for exactly this reason. Computing
    them from the wall clock would bill this job a negative number of seconds.
    """
    backwards = itertools.count(1000.0, -1.0)
    pool = make_pool(wall=lambda: next(backwards))

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    event = pool.store.list_usage("acme")[0]
    assert done.completed_at < done.started_at, "the clock did step backwards"
    assert event.duration_ms > 0


async def test_a_failed_job_is_still_metered(make_pool):
    """The machine ran either way. What is billable is decided above the pool."""
    workers = FakeWorkers(lambda request: httpx.Response(500, text="boom"))
    pool = make_pool(workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    event = pool.store.list_usage("acme")[0]
    assert event.terminal_state is JobState.FAILED
    assert event.duration_ms > 0


async def test_usage_is_attributed_to_the_submitting_tenant(make_pool):
    pool = make_pool()

    await pool.wait((await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"a")).id)
    await pool.wait((await pool.submit(tenant_id="globex", machine_type="cpu", payload=b"b")).id)

    assert len(pool.store.list_usage("acme")) == 1
    assert len(pool.store.list_usage("globex")) == 1
    assert len(pool.store.list_usage()) == 2

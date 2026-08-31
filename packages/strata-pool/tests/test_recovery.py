"""Restarting the pool process.

The claim is that pool state lives on disk and a restart resumes rather than
forgets. These tests build a second `Pool` over the first one's database,
which is what a restart actually is.
"""

from conftest import FakeBackend
from strata_pool import JobState, WorkerState


async def _one_completed_job(pool) -> str:
    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)
    return job.id


async def test_a_machine_that_survived_the_restart_is_kept_and_reused(make_pool):
    first = make_pool(db_name="shared.sqlite")
    await _one_completed_job(first)
    await first.aclose()

    backend = FakeBackend()
    restarted = make_pool(backend=backend, db_name="shared.sqlite")
    await restarted.recover()

    assert [w.state for w in restarted.store.list_workers()] == [WorkerState.WARM]

    job = await restarted.submit(tenant_id="acme", machine_type="cpu", payload=b"more")
    assert (await restarted.wait(job.id)).state is JobState.COMPLETED
    assert backend.started == [], "the surviving machine should have taken the work"


async def test_a_machine_that_did_not_survive_is_stopped_and_dropped(make_pool):
    first = make_pool(db_name="shared.sqlite")
    await _one_completed_job(first)
    await first.aclose()

    backend = FakeBackend()
    backend.dead = {"http://machine-1.test"}
    restarted = make_pool(backend=backend, db_name="shared.sqlite")
    await restarted.recover()

    assert restarted.store.list_workers() == []
    assert backend.stopped == ["machine-1"]


async def test_a_machine_still_running_our_orphaned_job_is_retired(make_pool):
    """The pool died, not the worker: it may still be computing.

    Nothing can tell it to stop, so returning it to the fleet would run the
    next job alongside the orphan on hardware sized for one.
    """
    first = make_pool(db_name="shared.sqlite")
    await _one_completed_job(first)

    worker = first.store.list_workers()[0]
    worker.state = WorkerState.BUSY
    worker.current_job_id = "job-orphan"
    first.store.save_worker(worker)
    await first.aclose()

    backend = FakeBackend()
    restarted = make_pool(backend=backend, db_name="shared.sqlite")
    await restarted.recover()

    assert restarted.store.list_workers() == []
    assert backend.stopped == ["machine-1"]


async def test_a_machine_that_finished_booting_while_we_were_down_is_put_to_work(make_pool):
    """Answering a health check is the promotion criterion, and no boot task
    survived the restart to apply it. Left STARTING it bills forever and holds
    a slot against max_workers without ever accepting a job."""
    first = make_pool(db_name="shared.sqlite")
    await _one_completed_job(first)

    worker = first.store.list_workers()[0]
    worker.state = WorkerState.STARTING
    first.store.save_worker(worker)
    await first.aclose()

    backend = FakeBackend()
    restarted = make_pool(backend=backend, db_name="shared.sqlite")
    await restarted.recover()

    assert [w.state for w in restarted.store.list_workers()] == [WorkerState.WARM]

    job = await restarted.submit(tenant_id="acme", machine_type="cpu", payload=b"more")
    assert (await restarted.wait(job.id)).state is JobState.COMPLETED
    assert backend.started == []


async def test_a_job_that_was_in_flight_is_failed_rather_than_re_run(make_pool):
    first = make_pool(db_name="shared.sqlite")
    done_id = await _one_completed_job(first)

    in_flight = first.store.get_job(done_id)
    in_flight.id = "job-in-flight"
    in_flight.state = JobState.RUNNING
    in_flight.result = None
    first.store.save_job(in_flight)
    await first.aclose()

    restarted = make_pool(db_name="shared.sqlite")
    await restarted.recover()

    recovered = restarted.store.get_job("job-in-flight")
    assert recovered.state is JobState.FAILED
    assert "restarted" in recovered.error
    billed = [event.job_id for event in restarted.store.list_usage()]
    assert "job-in-flight" not in billed, "our own crash is not billable"


async def test_work_still_queued_at_restart_gets_dispatched(make_pool):
    first = make_pool(db_name="shared.sqlite")
    await _one_completed_job(first)

    queued = first.store.get_job(await _one_completed_job(first))
    queued.id = "job-queued"
    queued.state = JobState.QUEUED
    queued.worker_id = None
    queued.result = None
    first.store.save_job(queued)
    await first.aclose()

    restarted = make_pool(db_name="shared.sqlite")
    await restarted.recover()

    assert (await restarted.wait("job-queued")).state is JobState.COMPLETED

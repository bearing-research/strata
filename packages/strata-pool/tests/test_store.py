"""Persistence: the pool's only authoritative state."""

import sqlite3

import pytest
from strata_pool import JobState, PoolStore, WorkerState
from strata_pool.types import Job, UsageEvent, Worker, new_id


def _worker(store: PoolStore, **overrides) -> Worker:
    worker = Worker(
        id=overrides.pop("id", new_id("worker")),
        machine_type=overrides.pop("machine_type", "cpu"),
        backend=overrides.pop("backend", "fake"),
        state=overrides.pop("state", WorkerState.WARM),
        created_at=overrides.pop("created_at", 100.0),
        **overrides,
    )
    store.save_worker(worker)
    return worker


def _job(store: PoolStore, **overrides) -> Job:
    job = Job(
        id=overrides.pop("id", new_id("job")),
        tenant_id=overrides.pop("tenant_id", "acme"),
        machine_type=overrides.pop("machine_type", "cpu"),
        payload=overrides.pop("payload", b"work"),
        state=overrides.pop("state", JobState.QUEUED),
        submitted_at=overrides.pop("submitted_at", 100.0),
        **overrides,
    )
    store.save_job(job)
    return job


@pytest.fixture
def store(tmp_path):
    store = PoolStore(tmp_path / "pool.sqlite")
    yield store
    store.close()


class TestWorkers:
    def test_round_trip(self, store):
        worker = _worker(store, endpoint="http://a.test", session_id="s1")
        loaded = store.get_worker(worker.id)
        assert loaded == worker

    def test_save_is_an_upsert(self, store):
        worker = _worker(store, state=WorkerState.STARTING)
        worker.state = WorkerState.WARM
        worker.backend_id = "machine-1"
        store.save_worker(worker)
        assert len(store.list_workers()) == 1
        assert store.get_worker(worker.id).state is WorkerState.WARM

    def test_two_workers_may_await_their_backend_ids(self, store):
        """The uniqueness index is partial, so NULL backend_ids don't collide."""
        _worker(store, state=WorkerState.STARTING)
        _worker(store, state=WorkerState.STARTING)
        assert len(store.list_workers()) == 2

    def test_the_same_machine_cannot_be_claimed_twice(self, store):
        _worker(store, backend_id="machine-1")
        with pytest.raises(sqlite3.IntegrityError):
            _worker(store, backend_id="machine-1")

    def test_find_warm_ignores_busy_and_starting(self, store):
        _worker(store, state=WorkerState.BUSY)
        _worker(store, state=WorkerState.STARTING)
        assert store.find_warm_worker("cpu") is None

    def test_find_warm_by_session_only_matches_that_session(self, store):
        _worker(store, session_id="other")
        assert store.find_warm_worker("cpu", session_id="mine") is None

    def test_count_workers_filters_by_type_and_state(self, store):
        _worker(store, state=WorkerState.WARM)
        _worker(store, state=WorkerState.BUSY)
        _worker(store, machine_type="gpu", state=WorkerState.WARM)
        assert store.count_workers("cpu", [WorkerState.WARM, WorkerState.BUSY]) == 2
        assert store.count_workers("gpu", [WorkerState.WARM]) == 1


class TestJobs:
    def test_round_trip(self, store):
        job = _job(store, priority=3, session_id="s1", timeout_seconds=12.0)
        assert store.get_job(job.id) == job

    def test_queue_is_priority_then_fifo(self, store):
        _job(store, id="low-early", priority=0, submitted_at=100.0)
        _job(store, id="high-late", priority=5, submitted_at=200.0)
        _job(store, id="high-early", priority=5, submitted_at=150.0)
        assert store.next_queued_job("cpu").id == "high-early"

    def test_queue_only_holds_queued_jobs(self, store):
        _job(store, state=JobState.RUNNING)
        assert store.next_queued_job("cpu") is None
        assert store.count_queued("cpu") == 0

    def test_job_history_outlives_its_worker(self, store):
        worker = _worker(store)
        job = _job(store, state=JobState.COMPLETED, worker_id=worker.id)
        store.delete_worker(worker.id)
        assert store.get_job(job.id).worker_id == worker.id


class TestUsage:
    def _event(self, **overrides) -> UsageEvent:
        return UsageEvent(
            id=overrides.pop("id", new_id("usage")),
            tenant_id=overrides.pop("tenant_id", "acme"),
            job_id=overrides.pop("job_id", "job-1"),
            machine_type=overrides.pop("machine_type", "cpu"),
            duration_ms=overrides.pop("duration_ms", 250.0),
            started_at=overrides.pop("started_at", 100.0),
            completed_at=overrides.pop("completed_at", 100.25),
            terminal_state=overrides.pop("terminal_state", JobState.COMPLETED),
            **overrides,
        )

    def test_round_trip(self, store):
        event = self._event(worker_id="worker-1")
        store.record_usage(event)
        assert store.list_usage("acme") == [event]

    def test_a_job_cannot_be_billed_twice(self, store):
        store.record_usage(self._event(job_id="job-1"))
        with pytest.raises(sqlite3.IntegrityError):
            store.record_usage(self._event(job_id="job-1"))

    def test_usage_is_scoped_by_tenant(self, store):
        store.record_usage(self._event(tenant_id="acme", job_id="job-1"))
        store.record_usage(self._event(tenant_id="other", job_id="job-2"))
        assert [e.tenant_id for e in store.list_usage("acme")] == ["acme"]

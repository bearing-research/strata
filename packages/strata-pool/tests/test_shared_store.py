"""Several pool processes over one store.

Runs on SQLite (two connections to one file) and, when `STRATA_POOL_POSTGRES_DSN`
is set, Postgres. Two pools with separate connections, backends and instance ids
stand in for two processes; killing one cancels its tasks before they write.
"""

import asyncio
import os
import threading
import time

import httpx
import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, Pool, PoolStore, WorkerState
from strata_pool.store import PostgresPoolStore
from strata_pool.types import Job, Worker, new_auth_token, new_id

_DSN = os.environ.get("STRATA_POOL_POSTGRES_DSN")
_ENGINES = ["sqlite", "postgres"]


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture(params=_ENGINES)
def open_store(request, tmp_path):
    """A factory for connections to one database."""
    if request.param == "postgres":
        if not _DSN:
            if os.environ.get("STRATA_POOL_REQUIRE_POSTGRES"):
                pytest.fail("STRATA_POOL_REQUIRE_POSTGRES is set but no DSN was given")
            pytest.skip("STRATA_POOL_POSTGRES_DSN is not set")
        import psycopg

        with psycopg.connect(_DSN, autocommit=True) as conn:
            conn.execute("DROP TABLE IF EXISTS workers, jobs, usage_events, catalogue")
    opened = []

    def _open():
        store = (
            PostgresPoolStore(_DSN)
            if request.param == "postgres"
            else PoolStore(tmp_path / "shared.sqlite")
        )
        opened.append(store)
        return store

    yield _open
    for store in opened:
        store.close()


@pytest.fixture
async def make_pool(open_store):
    built: list[tuple[Pool, httpx.AsyncClient]] = []

    def _make(instance_id: str, *, workers: FakeWorkers, **kwargs) -> Pool:
        client = httpx.AsyncClient(transport=httpx.MockTransport(workers.handle))
        kwargs.setdefault("backend", FakeBackend(id_prefix=instance_id))
        kwargs.setdefault("machine_types", [MachineType(name="cpu", image="w", max_workers=2)])
        pool = Pool(
            open_store(),
            kwargs.pop("backend"),
            kwargs.pop("machine_types"),
            client=client,
            health_poll_seconds=0,
            instance_id=instance_id,
            **kwargs,
        )
        built.append((pool, client))
        return pool

    yield _make
    for pool, client in built:
        await pool.aclose()
        await client.aclose()


async def _until(predicate) -> None:
    for _ in range(4000):
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true")


async def _kill(pool: Pool) -> None:
    """Stop a pool the way a dead process stops: nothing further is written."""
    for task in list(pool._tasks):
        task.cancel()
    await asyncio.gather(*pool._tasks, return_exceptions=True)


class TestTwoProcesses:
    async def test_a_stream_of_jobs_is_run_once_each_on_no_more_machines_than_allowed(
        self, make_pool
    ):
        async def slow_echo(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.01)
            return httpx.Response(200, content=b"done:" + request.content)

        workers = FakeWorkers(slow_echo)
        a = make_pool("a", workers=workers)
        b = make_pool("b", workers=workers)

        jobs = await asyncio.gather(
            *(
                (a if i % 2 else b).submit(
                    tenant_id="acme", machine_type="cpu", payload=f"job-{i}".encode()
                )
                for i in range(30)
            )
        )
        finished = [await a.wait(job.id, timeout=30) for job in jobs]

        assert {job.state for job in finished} == {JobState.COMPLETED}
        ran = sorted(request.content.decode() for request in workers.requests)
        assert ran == sorted(f"job-{i}" for i in range(30)), "every job ran exactly once"
        assert len(a.store.list_usage()) == 30
        started = a.backend.started + b.backend.started
        assert 1 <= len(started) <= 2, f"the tenant's cap is 2, the pools started {started}"

    async def test_two_processes_placing_the_same_queue_start_one_machine_per_job(self, make_pool):
        gate = asyncio.Event()

        class SlowStart(FakeBackend):
            async def start(self, spec, env=None):
                await gate.wait()
                return await super().start(spec, env)

        workers = FakeWorkers()
        spec = MachineType(name="cpu", image="w", max_workers=10)
        a = make_pool("a", workers=workers, backend=SlowStart(id_prefix="a"), machine_types=[spec])
        b = make_pool("b", workers=workers, backend=SlowStart(id_prefix="b"), machine_types=[spec])
        for i in range(3):
            a.store.save_job(
                Job(
                    id=f"job-{i}",
                    tenant_id="acme",
                    machine_type="cpu",
                    payload=b"work",
                    state=JobState.QUEUED,
                    submitted_at=float(i),
                )
            )

        placing = asyncio.gather(
            a._ensure_capacity("cpu", "acme"), b._ensure_capacity("cpu", "acme")
        )
        await asyncio.sleep(0.05)
        gate.set()
        await placing

        assert len(a.backend.started) + len(b.backend.started) == 3

    async def test_a_cancel_through_one_process_stops_a_job_another_runs(self, make_pool):
        """The process running the job settles it: it meters the time and keeps the machine."""
        killed = asyncio.Event()

        async def worker(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/cancel"):
                killed.set()
                return httpx.Response(200, json={"cancelled": True})
            await killed.wait()
            return httpx.Response(500, text="harness killed")

        workers = FakeWorkers(worker)
        a = make_pool("a", workers=workers)
        b = make_pool("b", workers=workers)

        job = await a.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        await _until(lambda: a.store.get_job(job.id).state is JobState.RUNNING)
        cancelled = await b.cancel(job.id, "build-1")

        assert cancelled.state is JobState.CANCELLED
        machine = b.store.get_worker(cancelled.worker_id)
        cancel = workers.requests[-1]
        assert cancel.url.path == "/v1/executions/build-1/cancel"
        assert cancel.headers["Authorization"] == f"Bearer {machine.auth_token}"

        await _until(lambda: b.store.get_worker(machine.id).state is WorkerState.WARM)
        assert a.backend.stopped == [] and b.backend.stopped == []
        [event] = b.store.list_usage()
        assert (event.job_id, event.terminal_state) == (job.id, JobState.CANCELLED)
        assert b.store.get_job(job.id).lease_owner is None

    async def test_a_dead_process_s_job_fails_and_its_machine_stops_once_its_lease_expires(
        self, make_pool
    ):
        clock = Clock()
        never_answers = asyncio.Event()

        async def hang(request: httpx.Request) -> httpx.Response:
            await never_answers.wait()
            return httpx.Response(200)

        workers = FakeWorkers(hang)
        a = make_pool("a", workers=workers, wall=clock, lease_seconds=30)
        b = make_pool("b", workers=workers, wall=clock, lease_seconds=30)

        job = await a.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        await _until(lambda: a.store.get_job(job.id).state is JobState.RUNNING)
        await _kill(a)

        # Inside the lease, the machine and the job are still a's.
        assert await b.reclaim_expired_leases() == 0
        assert b.store.get_job(job.id).state is JobState.RUNNING
        assert [w.state for w in b.store.list_workers()] == [WorkerState.BUSY]

        clock.now += 31
        assert await b.reclaim_expired_leases() == 1

        failed = b.store.get_job(job.id)
        assert failed.state is JobState.FAILED
        assert "'a' stopped renewing" in failed.error
        assert b.backend.stopped == ["a-1"]
        assert b.store.list_workers() == []
        assert b.store.list_usage() == [], "a job nobody saw finish is not billed"

    async def test_a_live_process_renews_its_lease_and_keeps_its_job(self, make_pool):
        clock = Clock()
        release = asyncio.Event()

        async def wait_for_release(request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(200, content=b"done")

        workers = FakeWorkers(wait_for_release)
        a = make_pool("a", workers=workers, wall=clock, lease_seconds=0.03)
        b = make_pool("b", workers=workers, wall=clock, lease_seconds=0.03)

        job = await a.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        await _until(lambda: a.store.get_job(job.id).state is JobState.RUNNING)
        clock.now += 60
        await _until(lambda: a.store.get_job(job.id).lease_expires_at > clock.now)
        await _until(lambda: all(w.lease_expires_at > clock.now for w in a.store.list_workers()))

        assert await b.reclaim_expired_leases() == 0
        release.set()
        assert (await a.wait(job.id)).state is JobState.COMPLETED
        # Now warm, with no lease at all: another process's to use, not to stop.
        assert await b.reclaim_expired_leases() == 0
        assert [w.state for w in b.store.list_workers()] == [WorkerState.WARM]
        assert b.backend.stopped == []

    async def test_a_restart_does_not_take_what_another_live_process_holds(self, make_pool):
        clock = Clock()
        release = asyncio.Event()

        async def wait_for_release(request: httpx.Request) -> httpx.Response:
            await release.wait()
            return httpx.Response(200, content=b"done")

        workers = FakeWorkers(wait_for_release)
        a = make_pool("a", workers=workers, wall=clock)
        job = await a.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        await _until(lambda: a.store.get_job(job.id).state is JobState.RUNNING)

        restarted_b = make_pool("b", workers=workers, wall=clock)
        await restarted_b.recover()

        assert restarted_b.store.get_job(job.id).state is JobState.RUNNING
        assert restarted_b.backend.stopped == []
        release.set()
        assert (await a.wait(job.id)).state is JobState.COMPLETED

    async def test_a_restart_under_the_same_name_takes_its_own_rows_back_at_once(self, make_pool):
        clock = Clock()
        hang = asyncio.Event()

        async def never(request: httpx.Request) -> httpx.Response:
            await hang.wait()
            return httpx.Response(200)

        workers = FakeWorkers(never)
        a = make_pool("a", workers=workers, wall=clock)
        job = await a.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        await _until(lambda: a.store.get_job(job.id).state is JobState.RUNNING)
        await _kill(a)

        again = make_pool("a", workers=workers, wall=clock, backend=FakeBackend(id_prefix="a2"))
        await again.recover()

        assert again.store.get_job(job.id).state is JobState.FAILED
        assert again.backend.stopped == ["a-1"]


class TestTheStore:
    def test_of_two_claims_on_one_warm_machine_one_wins(self, open_store):
        first, second = open_store(), open_store()
        worker = Worker(
            id="w1",
            machine_type="cpu",
            tenant_id="acme",
            backend="fake",
            state=WorkerState.WARM,
            created_at=1.0,
        )
        first.save_worker(worker)
        jobs = []
        for i in range(2):
            job = Job(
                id=f"j{i}",
                tenant_id="acme",
                machine_type="cpu",
                payload=b"x",
                state=JobState.QUEUED,
                submitted_at=float(i),
            )
            first.save_job(job)
            jobs.append(job)

        assert first.claim_dispatch(worker, jobs[0], "a", 100.0) is True
        assert second.claim_dispatch(worker, jobs[1], "b", 100.0) is False
        assert second.get_job("j1").state is JobState.QUEUED, "the losing claim wrote nothing"
        assert second.get_worker("w1").current_job_id == "j0"

    def test_a_job_claimed_through_two_machines_goes_to_one_and_frees_the_other(self, open_store):
        first, second = open_store(), open_store()
        machines = []
        for name in ("w1", "w2"):
            worker = Worker(
                id=name,
                machine_type="cpu",
                tenant_id="acme",
                backend="fake",
                state=WorkerState.WARM,
                created_at=1.0,
            )
            first.save_worker(worker)
            machines.append(worker)
        job = Job(
            id="j",
            tenant_id="acme",
            machine_type="cpu",
            payload=b"x",
            state=JobState.QUEUED,
            submitted_at=1.0,
        )
        first.save_job(job)

        assert first.claim_dispatch(machines[0], job, "a", 100.0) is True
        assert second.claim_dispatch(machines[1], job, "b", 100.0) is False
        assert second.get_job("j").worker_id == "w1"
        assert second.get_worker("w2").state is WorkerState.WARM, "the losing claim rolled back"

    def test_a_job_cancelled_through_one_connection_cannot_be_claimed_through_another(
        self, open_store
    ):
        first, second = open_store(), open_store()
        worker = Worker(
            id="w1",
            machine_type="cpu",
            tenant_id="acme",
            backend="fake",
            state=WorkerState.WARM,
            created_at=1.0,
        )
        first.save_worker(worker)
        job = Job(
            id="j",
            tenant_id="acme",
            machine_type="cpu",
            payload=b"x",
            state=JobState.QUEUED,
            submitted_at=1.0,
        )
        first.save_job(job)

        assert second.cancel_job("j", "cancelled", 2.0) is True
        assert first.claim_dispatch(worker, job, "a", 100.0) is False
        assert first.get_worker("w1").state is WorkerState.WARM, "the losing claim rolled back"
        assert first.get_job("j").state is JobState.CANCELLED
        assert second.cancel_job("j", "cancelled", 3.0) is False, "a finished job stays as it is"
        assert first.get_job("j").completed_at == 2.0

    def test_concurrent_reservations_never_exceed_demand(self, open_store):
        """Threads with their own connections, so the store's serialization is under test."""
        setup = open_store()
        for i in range(5):
            setup.save_job(
                Job(
                    id=f"j{i}",
                    tenant_id="acme",
                    machine_type="cpu",
                    payload=b"x",
                    state=JobState.QUEUED,
                    submitted_at=float(i),
                )
            )
        stores = [open_store() for _ in range(4)]
        for store in stores:
            # Widen the window between reading demand and inserting the
            # machine, which is where an unserialized reservation races.
            read = store.count_queued

            def slow_count(*args, _read=read):
                count = _read(*args)
                time.sleep(0.01)
                return count

            store.count_queued = slow_count
        outcomes: list[str] = []
        lock = threading.Lock()

        def reserve(store):
            for _ in range(5):
                outcome = _reserve_one(store)
                with lock:
                    outcomes.append(outcome)

        def _reserve_one(store):
            worker = Worker(
                id=new_id("worker"),
                machine_type="cpu",
                tenant_id="acme",
                backend="fake",
                state=WorkerState.STARTING,
                created_at=1.0,
                auth_token=new_auth_token(),
                image="w",
            )
            try:
                return store.reserve_worker(worker, max_workers=10, max_workers_total=None)
            except Exception as exc:
                # An unserialized store surfaces its race as a lock error as
                # often as a double reservation; either is a failure.
                return repr(exc)

        threads = [threading.Thread(target=reserve, args=(store,)) for store in stores]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(set(outcomes)) == ["no_demand", "reserved"], outcomes
        assert outcomes.count("reserved") == 5
        assert len(setup.list_workers()) == 5

    def test_a_shared_store_needs_an_instance_id(self, open_store):
        store = open_store()
        if not store.shared:
            pytest.skip("a SQLite store is not shared across hosts")
        with pytest.raises(ValueError, match="instance_id"):
            Pool(store, FakeBackend(), [MachineType(name="cpu", image="w")])


class TestTwoProcessesAreTwoNames:
    """Leases match by name, so two processes under one name would take each other's rows."""

    def test_a_pool_without_a_name_does_not_share_one(self, tmp_path):
        types = [MachineType(name="cpu", image="w")]
        first = Pool(PoolStore(tmp_path / "a.sqlite"), FakeBackend(), types)
        second = Pool(PoolStore(tmp_path / "a.sqlite"), FakeBackend(), types)

        assert first.instance_id != second.instance_id
        assert "pool" != first.instance_id, "a constant name is every other process's name too"

    @pytest.mark.asyncio
    async def test_a_second_pool_does_not_stop_the_first_ones_busy_machine(
        self, open_store, make_pool
    ):
        """A machine another live process holds is left alone."""
        workers = FakeWorkers()
        clock = Clock()
        first = make_pool("a", workers=workers, wall=clock)
        store = first.store
        worker = Worker(
            id=new_id("w"),
            machine_type="cpu",
            tenant_id="t",
            backend="fake",
            state=WorkerState.BUSY,
            backend_id="a-1",
            endpoint="http://a-1",
            auth_token=new_auth_token(),
            created_at=clock(),
            lease_owner=first.instance_id,
            lease_expires_at=clock() + first.lease_seconds,
        )
        store.save_worker(worker)

        second = make_pool("b", workers=workers, wall=clock)
        stopped = await second._stop_worker(store.get_worker(worker.id))

        assert stopped is False
        assert store.get_worker(worker.id).state is WorkerState.BUSY


class TestAJobTakenOverStaysTakenOver:
    @pytest.mark.asyncio
    async def test_a_stalled_process_does_not_resurrect_a_failed_job(self, open_store, make_pool):
        """After another process failed the job, the stalled task must not write ``running``."""
        workers = FakeWorkers()
        clock = Clock()
        pool = make_pool("a", workers=workers, wall=clock)
        store = pool.store
        worker = Worker(
            id=new_id("w"),
            machine_type="cpu",
            tenant_id="t",
            backend="fake",
            state=WorkerState.BUSY,
            backend_id="a-1",
            endpoint="http://a-1",
            auth_token=new_auth_token(),
            created_at=clock(),
        )
        store.save_worker(worker)
        job = Job(
            id=new_id("j"),
            machine_type="cpu",
            tenant_id="t",
            payload=b"{}",
            state=JobState.FAILED,
            error="reclaimed: the pool process holding it stopped answering",
            submitted_at=clock(),
            completed_at=clock(),
        )
        store.save_job(job)

        dispatched = Job(**{**job.__dict__, "state": JobState.DISPATCHED})
        await pool._execute(worker, dispatched)

        after = store.get_job(job.id)
        assert after.state is JobState.FAILED
        assert after.error.startswith("reclaimed:")


class TestAMachineBeingStoppedStillCounts:
    @pytest.mark.asyncio
    async def test_the_fleet_cap_holds_while_a_stop_is_in_flight(self, open_store, make_pool):
        """A machine whose provider stop is in flight is still allocated and
        still billing, so it holds its place against the cap."""
        workers = FakeWorkers()
        clock = Clock()
        pool = make_pool("a", workers=workers, wall=clock, max_workers_total=1)
        store = pool.store
        stopping = Worker(
            id=new_id("w"),
            machine_type="cpu",
            tenant_id="t",
            backend="fake",
            state=WorkerState.STOPPING,
            backend_id="a-1",
            endpoint="http://a-1",
            auth_token=new_auth_token(),
            created_at=clock(),
        )
        store.save_worker(stopping)
        # Demand from another tenant, so the only thing that can refuse the
        # reservation is the fleet cap.
        store.save_job(
            Job(
                id=new_id("j"),
                machine_type="cpu",
                tenant_id="t2",
                payload=b"{}",
                state=JobState.QUEUED,
                submitted_at=clock(),
            )
        )

        reserved = store.reserve_worker(
            Worker(
                id=new_id("w"),
                machine_type="cpu",
                tenant_id="t2",
                backend="fake",
                state=WorkerState.STARTING,
                created_at=clock(),
            ),
            max_workers=2,
            max_workers_total=1,
        )

        assert reserved == "fleet_cap", "the cap was exceeded while a machine was still stopping"


class TestAStopThatFailsKeepsTheRow:
    @pytest.mark.asyncio
    async def test_a_provider_error_does_not_orphan_the_machine(self, open_store, make_pool):
        """A failed provider stop keeps the row in ``stopping`` to be retried, not deleted."""

        class _Failing(FakeBackend):
            async def stop(self, backend_id: str) -> None:
                raise RuntimeError("provider 503")

        workers = FakeWorkers()
        clock = Clock()
        pool = make_pool("a", workers=workers, wall=clock, backend=_Failing())
        store = pool.store
        worker = Worker(
            id=new_id("w"),
            machine_type="cpu",
            tenant_id="t",
            backend="fake",
            state=WorkerState.WARM,
            backend_id="a-1",
            endpoint="http://a-1",
            auth_token=new_auth_token(),
            created_at=clock(),
        )
        store.save_worker(worker)

        stopped = await pool._stop_worker(worker)

        assert stopped is False
        kept = store.get_worker(worker.id)
        assert kept is not None and kept.state is WorkerState.STOPPING


class TestAStopThatFailedIsTriedAgain:
    @pytest.mark.asyncio
    async def test_the_fleet_slot_comes_back_once_the_provider_answers(self, open_store, make_pool):
        """A retried stop frees the fleet slot, so a provider blip does not cost it forever."""

        class _FlakyBackend(FakeBackend):
            def __init__(self):
                super().__init__()
                self.failures = 1

            async def stop(self, backend_id: str) -> None:
                if self.failures:
                    self.failures -= 1
                    raise RuntimeError("provider 503")
                await super().stop(backend_id)

        workers = FakeWorkers()
        clock = Clock()
        backend = _FlakyBackend()
        pool = make_pool("a", workers=workers, wall=clock, backend=backend)
        store = pool.store
        worker = Worker(
            id=new_id("w"),
            machine_type="cpu",
            tenant_id="t",
            backend="fake",
            state=WorkerState.WARM,
            backend_id="a-1",
            endpoint="http://a-1",
            auth_token=new_auth_token(),
            created_at=clock(),
        )
        store.save_worker(worker)

        assert await pool._stop_worker(worker) is False
        assert store.get_worker(worker.id).state is WorkerState.STOPPING

        clock.now += 1000  # the claim it holds has long expired
        stopped = await pool.reap_idle_workers()

        assert stopped == 1
        assert store.get_worker(worker.id) is None, "the fleet slot never came back"

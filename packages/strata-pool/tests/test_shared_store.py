"""Several pool processes over one store. Item 37.

Each test runs against SQLite (two connections to one file) and, when
`STRATA_POOL_POSTGRES_DSN` names a database, Postgres. The two pools stand in
for two processes: separate store connections, separate backends, separate
instance ids, one database. Killing a process is cancelling its tasks without
letting them write anything, which is what a process that dies leaves behind.
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

    def test_concurrent_reservations_never_exceed_demand(self, open_store):
        """Threads with their own connections, so the store's own
        serialization is what is under test, not the event loop's."""
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

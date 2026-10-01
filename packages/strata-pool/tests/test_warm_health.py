"""Finding a dead warm machine before a job does (and fails on it)."""

from conftest import FakeBackend
from strata_pool import JobState, MachineType, WorkerState


def _spec(**kwargs) -> MachineType:
    return MachineType(name=kwargs.pop("name", "cpu"), image=kwargs.pop("image", "w"), **kwargs)


async def _warm_worker(pool, backend):
    """Run a job so a machine exists and is warm afterwards."""
    job = await pool.submit(tenant_id="t", machine_type="cpu", payload=b"work")
    assert (await pool.wait(job.id)).state is JobState.COMPLETED
    warm = pool.store.list_workers("cpu", [WorkerState.WARM])
    assert len(warm) == 1
    return warm[0]


class TestRetiringADeadMachine:
    async def test_a_machine_that_stops_answering_is_retired(self, make_pool):
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        worker = await _warm_worker(pool, backend)

        backend.dead.add(worker.endpoint)
        stopped = await pool.probe_warm_workers()

        assert stopped == 1
        assert pool.store.list_workers("cpu", [WorkerState.WARM]) == []
        assert worker.backend_id in backend.stopped

    async def test_a_healthy_machine_is_left_alone(self, make_pool):
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        await _warm_worker(pool, backend)

        assert await pool.probe_warm_workers() == 0
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 1


class TestOneMissIsNotDeath:
    """A single missed probe is a hiccup, not a reason to retire the machine."""

    async def test_it_takes_consecutive_failures(self, make_pool):
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=3)])
        worker = await _warm_worker(pool, backend)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0
        assert await pool.probe_warm_workers() == 0
        assert await pool.probe_warm_workers() == 1

    async def test_recovering_resets_the_count(self, make_pool):
        """A hit resets the miss count, or intermittent misses would retire a healthy machine."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=3)])
        worker = await _warm_worker(pool, backend)

        backend.dead.add(worker.endpoint)
        await pool.probe_warm_workers()
        await pool.probe_warm_workers()
        backend.dead.discard(worker.endpoint)
        await pool.probe_warm_workers()
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0, "the count should have restarted"


class TestScope:
    async def test_probing_can_be_switched_off(self, make_pool):
        """For a backend whose health check costs something."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=0)])
        worker = await _warm_worker(pool, backend)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 1

    async def test_a_freed_slot_is_offered_to_whoever_is_waiting(self, make_pool):
        """Retiring a machine offers the freed slot to a waiting job of another tenant.

        It must be another tenant: a same-tenant job would dispatch onto the
        dying machine before the probe, and the assertion would pass regardless.
        """
        backend = FakeBackend()
        pool = make_pool(
            backend,
            machine_types=[_spec(health_check_failures=1)],
            max_workers_total=1,
        )
        worker = await _warm_worker(pool, backend)
        backend.dead.add(worker.endpoint)

        queued = await pool.submit(tenant_id="other", machine_type="cpu", payload=b"next")
        assert pool.store.get_worker(worker.id).state is WorkerState.WARM
        assert await pool.probe_warm_workers() == 1

        assert (await pool.wait(queued.id, timeout=5)).state is JobState.COMPLETED


class TestPoolSideFaults:
    """Every probe failing at once is more likely a pool-side fault (DNS, proxy) than the fleet."""

    async def test_a_whole_fleet_failing_at_once_is_not_acted_on(self, make_pool):
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        await _warm_worker(pool, backend)
        second_job = await pool.submit(tenant_id="u", machine_type="cpu", payload=b"w")
        await pool.wait(second_job.id)
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 2

        backend.never_healthy = True  # the pool's own network, in effect

        assert await pool.probe_warm_workers() == 0
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 2

    async def test_a_single_machine_is_still_retired(self, make_pool):
        """With one candidate the fleet-wide guard does not apply."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        worker = await _warm_worker(pool, backend)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 1

    async def test_one_dead_among_healthy_is_still_retired(self, make_pool):
        """The guard is about the whole fleet failing, not about any failure."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        first = await _warm_worker(pool, backend)
        second_job = await pool.submit(tenant_id="u", machine_type="cpu", payload=b"w")
        await pool.wait(second_job.id)

        backend.dead.add(first.endpoint)

        assert await pool.probe_warm_workers() == 1
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 1


class TestProbeBudget:
    async def test_machines_are_probed_concurrently(self, make_pool):
        """Serial probes would cost a timeout per machine and delay the reaper behind them."""
        import asyncio

        class SlowBackend(FakeBackend):
            async def health(self, endpoint: str) -> bool:
                await asyncio.sleep(0.2)
                return await super().health(endpoint)

        backend = SlowBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        await _warm_worker(pool, backend)
        for tenant in ("u", "v", "w"):
            job = await pool.submit(tenant_id=tenant, machine_type="cpu", payload=b"x")
            await pool.wait(job.id)
        assert len(pool.store.list_workers("cpu", [WorkerState.WARM])) == 4

        started = asyncio.get_running_loop().time()
        await pool.probe_warm_workers()
        elapsed = asyncio.get_running_loop().time() - started

        # Four probes of 0.2s each: ~0.2s together, ~0.8s one after another.
        assert elapsed < 0.5, f"probes look serial ({elapsed:.2f}s for four)"


class TestNoCounterLeak:
    async def test_stopping_a_machine_forgets_its_probe_count(self, make_pool):
        """Any stop path clears the probe count; the probe loop never sees the machine again."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=3)])
        worker = await _warm_worker(pool, backend)

        backend.dead.add(worker.endpoint)
        await pool.probe_warm_workers()
        assert pool._probe_failures.get(worker.id) == 1

        await pool._stop_worker(pool.store.get_worker(worker.id))

        assert worker.id not in pool._probe_failures


class TestBusyMachines:
    async def test_a_busy_machine_is_never_probed_out_from_under_its_job(self, make_pool):
        """Only warm machines are probed; a slow probe must not retire a machine mid-job."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        worker = await _warm_worker(pool, backend)

        busy = pool.store.get_worker(worker.id)
        busy.state = WorkerState.BUSY
        pool.store.save_worker(busy)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0
        assert backend.stopped == []

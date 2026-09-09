"""Finding a dead warm machine before a job does.

`backend.health` was called while awaiting boot and in `recover()`, and
nowhere else. A machine that died while warm was discovered by the next job
being sent to it — and that job failed. The machine was already unusable; the
delay only bought a user watching a cell fail for reasons unrelated to their
code.
"""

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
    """A single missed probe is a slow machine, a restarting agent, or a
    dropped packet. Retiring on that trades a cold start for every hiccup."""

    async def test_it_takes_consecutive_failures(self, make_pool):
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=3)])
        worker = await _warm_worker(pool, backend)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0
        assert await pool.probe_warm_workers() == 0
        assert await pool.probe_warm_workers() == 1

    async def test_recovering_resets_the_count(self, make_pool):
        """Two misses and a hit is a machine that is alive, not two-thirds
        dead — otherwise an intermittent probe retires a healthy machine
        eventually, however long it stays up."""
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
        """Retiring a machine frees capacity, and the queue may hold work for
        a different tenant than the one whose machine died.

        The second job is for a *different* tenant and the fleet cap is 1.
        Submitting for the same tenant would dispatch straight onto the dying
        machine — flipping it to BUSY before the probe runs, so the probe
        would list nothing, stop nothing, and the job would still complete
        over the mock transport. The assertion would then hold whatever this
        code did.
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
    """Every machine failing at once is more likely this process than the fleet.

    DNS, a proxy, an exhausted connection pool: any of them fails every probe
    simultaneously. Acting on that retires the entire warm fleet and hands
    every user a cold start for a fault that was never on the machines.
    """

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
        """With one candidate there is no evidence either way, and never
        noticing is worse than one cold start."""
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
        """Serially, a provider black-holing packets costs the health timeout
        times the fleet size — and reap_idle_workers, the only thing that
        stops a machine billing, does not run until the pass finishes."""
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
        """A machine stopped by any other path is never listed WARM again, so
        the probe loop that would have popped it never sees it."""
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
        """Only warm machines are listed. A probe that loses a race against a
        long-running cell must not retire the machine running it."""
        backend = FakeBackend()
        pool = make_pool(backend, machine_types=[_spec(health_check_failures=1)])
        worker = await _warm_worker(pool, backend)

        busy = pool.store.get_worker(worker.id)
        busy.state = WorkerState.BUSY
        pool.store.save_worker(busy)
        backend.dead.add(worker.endpoint)

        assert await pool.probe_warm_workers() == 0
        assert backend.stopped == []

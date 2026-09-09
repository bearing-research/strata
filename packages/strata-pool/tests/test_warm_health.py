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
        a different tenant than the one whose machine died."""
        backend = FakeBackend()
        pool = make_pool(
            backend,
            machine_types=[_spec(max_workers=1, health_check_failures=1)],
        )
        worker = await _warm_worker(pool, backend)

        backend.dead.add(worker.endpoint)
        queued = await pool.submit(tenant_id="t", machine_type="cpu", payload=b"next")
        await pool.probe_warm_workers()

        assert (await pool.wait(queued.id)).state is JobState.COMPLETED


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

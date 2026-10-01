"""Stopping idle machines; nothing else in the pool ever does.

The clock is injected throughout rather than proved by sleeping.
"""

import asyncio
import itertools

from conftest import FakeBackend
from strata_pool import JobState, MachineType, WorkerState
from strata_pool.types import Worker, new_auth_token, new_id


class Clock:
    """A wall clock the test moves by hand."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def _start_idle_machine(pool, tenant_id: str = "acme") -> None:
    """A machine nobody queued work for, as a job that failed leaves one."""
    spec = pool.machine_types["cpu"]
    worker = Worker(
        id=new_id("worker"),
        machine_type=spec.name,
        tenant_id=tenant_id,
        backend=pool.backend.name,
        state=WorkerState.STARTING,
        created_at=pool._wall(),
        auth_token=new_auth_token(),
        image=spec.image,
        lease_owner=pool.instance_id,
        lease_expires_at=pool._wall() + pool.lease_seconds,
    )
    pool.store.save_worker(worker)
    await pool._start_worker(spec, worker)


def _spec(**kwargs) -> MachineType:
    return MachineType(
        name=kwargs.pop("name", "cpu"),
        image=kwargs.pop("image", "w"),
        cool_down_seconds=kwargs.pop("cool_down_seconds", 300.0),
        **kwargs,
    )


async def test_a_machine_idle_past_its_cool_down_is_stopped(make_pool):
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    clock.advance(301)
    assert await pool.reap_idle_workers() == 1

    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_machine_inside_its_cool_down_is_left_alone(make_pool):
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    clock.advance(299)
    assert await pool.reap_idle_workers() == 0
    assert backend.stopped == []


async def test_cool_down_is_per_machine_type(make_pool):
    """An idle H100 and an idle CPU worker are not the same money."""
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[
            _spec(name="cpu", cool_down_seconds=300.0),
            _spec(name="gpu", cool_down_seconds=60.0),
        ],
        wall=clock,
    )

    for machine_type in ("cpu", "gpu"):
        job = await pool.submit(tenant_id="acme", machine_type=machine_type, payload=b"work")
        await pool.wait(job.id)

    clock.advance(61)
    assert await pool.reap_idle_workers() == 1

    survivors = {w.machine_type for w in pool.store.list_workers()}
    assert survivors == {"cpu"}, "the expensive one goes first, because it was told to"


async def test_a_machine_that_never_ran_anything_still_ages_out(make_pool):
    """With no last_active_at, a machine ages from when it booted."""
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    await _start_idle_machine(pool)
    await asyncio.sleep(0)  # let it boot to warm
    assert [w.state for w in pool.store.list_workers()] == [WorkerState.WARM]

    clock.advance(301)
    assert await pool.reap_idle_workers() == 1


async def test_a_busy_machine_is_never_reaped(make_pool):
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    ran = await pool.wait(job.id)
    worker = pool.store.get_worker(ran.worker_id)
    worker.state = WorkerState.BUSY
    pool.store.save_worker(worker)

    clock.advance(10_000)
    assert await pool.reap_idle_workers() == 0
    assert backend.stopped == []


async def test_a_backwards_clock_step_cannot_make_a_machine_immortal(make_pool):
    """A machine last active in the future is clamped to now, or it would be unreapable."""
    clock = Clock()
    pool = make_pool(machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    clock.now -= 86_400  # the clock steps back a day
    assert await pool.reap_idle_workers() == 0, "it is not idle yet, it is skewed"

    clock.advance(301)
    assert await pool.reap_idle_workers() == 1, "and it ages from the clamp, not the skew"


async def test_reaping_frees_capacity_for_the_next_job(make_pool):
    """A reaped machine stops counting against max_workers, so a capped tenant can replace it."""
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[_spec(max_workers=1)],
        wall=clock,
    )

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    await pool.wait(first.id)

    clock.advance(301)
    await pool.reap_idle_workers()

    second = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    assert (await pool.wait(second.id)).state is JobState.COMPLETED
    assert backend.started == ["machine-1", "machine-2"]


class SlowStopBackend(FakeBackend):
    """Holds a machine open mid-teardown so the race window is observable."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stop_entered = asyncio.Event()
        self.release = asyncio.Event()

    async def stop(self, backend_id: str) -> None:
        self.stop_entered.set()
        await self.release.wait()
        await super().stop(backend_id)


async def test_a_machine_being_stopped_is_not_offered_to_a_dispatcher(make_pool):
    """The stop claim lands before the backend await, so no job is dispatched to it meanwhile."""
    clock = Clock()
    backend = SlowStopBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    dying = (await pool.wait(job.id)).worker_id

    clock.advance(301)
    reaping = asyncio.create_task(pool.reap_idle_workers())
    await asyncio.wait_for(backend.stop_entered.wait(), timeout=2)

    # The machine is still alive on the backend, and must already be invisible.
    assert pool.store.find_warm_worker("cpu", "acme") is None
    assert pool.store.get_worker(dying).state is WorkerState.STOPPING

    arriving = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"more")
    assert pool.store.get_job(arriving.id).worker_id != dying

    backend.release.set()
    await reaping
    assert backend.started == ["machine-1", "machine-2"], "the new job got a new machine"


async def test_a_failing_pass_does_not_kill_the_loop(make_pool):
    """A raising pass must not end the scaler loop."""
    clock = Clock()
    pool = make_pool(machine_types=[_spec()], wall=clock)

    calls = itertools.count()
    original = pool.reap_idle_workers

    async def explode_once():
        if next(calls) == 0:
            raise RuntimeError("transient store failure")
        return await original()

    pool.reap_idle_workers = explode_once
    pool.start_scaler(interval_seconds=0.01)
    await asyncio.sleep(0.1)

    assert next(calls) > 1, "the loop kept running after a pass raised"


async def test_the_scaler_actually_runs_on_its_own(make_pool):
    """The loop runs passes by itself; every other test here calls the pass by hand."""
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)
    clock.advance(301)

    pool.start_scaler(interval_seconds=0.01)
    for _ in range(200):
        if backend.stopped:
            break
        await asyncio.sleep(0.01)

    assert backend.stopped == ["machine-1"]
    assert pool.store.list_workers() == []


async def test_a_machine_that_takes_a_job_mid_pass_is_not_stopped_under_it(make_pool):
    """A listed idle machine that takes a job during an earlier stop's await is left running."""
    clock = Clock()
    backend = SlowStopBackend()
    pool = make_pool(backend=backend, machine_types=[_spec(max_workers=2)], wall=clock)

    # Two idle machines for the same tenant, both past their cool-down.
    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    await pool.wait(first.id)
    await _start_idle_machine(pool)
    await asyncio.sleep(0)
    assert len(pool.store.list_workers()) == 2
    clock.advance(301)

    reaping = asyncio.create_task(pool.reap_idle_workers())
    await asyncio.wait_for(backend.stop_entered.wait(), timeout=2)

    # While the first stop is in flight, work arrives for the other machine.
    arriving = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    claimed = pool.store.get_job(arriving.id).worker_id
    assert claimed is not None, "the second machine was still warm and took the job"

    backend.release.set()
    await reaping

    assert (await pool.wait(arriving.id)).state is JobState.COMPLETED
    assert pool.store.get_worker(claimed) is not None, "the working machine survived the pass"
    assert len(backend.stopped) == 1


async def test_a_machine_left_mid_teardown_by_a_crash_is_finished_off(make_pool):
    """A machine left in ``stopping`` by a crash is stopped by the next scaler pass."""
    clock = Clock()
    first = make_pool(machine_types=[_spec()], wall=clock, db_name="shared.sqlite")
    job = await first.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    ran = await first.wait(job.id)

    abandoned = first.store.get_worker(ran.worker_id)
    abandoned.state = WorkerState.STOPPING
    first.store.save_worker(abandoned)
    await first.aclose()

    backend = FakeBackend(id_prefix="after-restart")
    restarted = make_pool(
        backend=backend, machine_types=[_spec()], wall=clock, db_name="shared.sqlite"
    )
    await restarted.recover()

    assert restarted.store.get_worker(abandoned.id) is None
    assert backend.stopped == ["machine-1"], "the machine the last process claimed is gone"

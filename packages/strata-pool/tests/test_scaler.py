"""Stopping machines that are no longer earning their keep.

Nothing else in the pool ever stops a machine that finished its work, so
without this a pool bills for every machine it ever started, forever. On a
CPU worker that is waste; on a GPU it is the most expensive bug the system
can produce.

The clock is injected throughout: a cost control proved by sleeping is a cost
control proved by hope.
"""

import asyncio
import itertools

from conftest import FakeBackend
from strata_pool import JobState, MachineType, WorkerState


class Clock:
    """A wall clock the test moves by hand."""

    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


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
    """A machine booted for a job that then failed would otherwise idle
    forever with no last_active_at to age from."""
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(backend=backend, machine_types=[_spec()], wall=clock)

    await pool._start_worker(pool.machine_types["cpu"], "acme")
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
    """Idleness is wall-clock, because it has to survive a restart. A machine
    last active in the future would be unreapable until the clock caught up,
    which is unbounded idle billing — so it is clamped to now and ages from
    there."""
    clock = Clock()
    pool = make_pool(machine_types=[_spec()], wall=clock)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    clock.now -= 86_400  # the clock steps back a day
    assert await pool.reap_idle_workers() == 0, "it is not idle yet, it is skewed"

    clock.advance(301)
    assert await pool.reap_idle_workers() == 1, "and it ages from the clamp, not the skew"


async def test_reaping_frees_capacity_for_the_next_job(make_pool):
    """A machine on its way out must not count against max_workers, or a
    tenant at its cap cannot start the replacement."""
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
    """Stopping a machine awaits the backend. In that window the dispatcher
    must not find it warm and hand it a job we are about to kill — so the
    claim has to land before the await, not after it."""
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
    """A dead scaler loop is indistinguishable from no scaler at all, and
    that difference is measured in dollars per hour."""
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
    """The one test that distinguishes a scaler from a scaler nobody started.

    Every other test here calls the pass by hand, which a control loop that
    is written and never wired would also survive.
    """
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

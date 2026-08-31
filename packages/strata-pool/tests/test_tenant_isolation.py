"""A machine belongs to one tenant for its life.

Reuse across tenants is the cheap thing to do and the wrong thing to do: even
scrubbed of files, a process that ran one tenant's code is not a boundary the
next tenant should have to trust, and GPU memory is not reliably zeroed
between processes at all.
"""

from conftest import FakeBackend
from strata_pool import JobState, MachineType, WorkerState


async def test_a_warm_machine_is_never_handed_to_another_tenant(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend=backend)

    mine = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"mine")
    ran_mine = await pool.wait(mine.id)

    theirs = await pool.submit(tenant_id="globex", machine_type="cpu", payload=b"theirs")
    ran_theirs = await pool.wait(theirs.id)

    assert ran_theirs.state is JobState.COMPLETED
    assert ran_theirs.worker_id != ran_mine.worker_id
    assert backend.started == ["machine-1", "machine-2"], "a second tenant means a second machine"


async def test_each_machine_records_who_it_belongs_to(make_pool):
    pool = make_pool()

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert pool.store.get_worker(done.worker_id).tenant_id == "acme"


async def test_one_tenant_cannot_exhaust_another_tenants_capacity(make_pool):
    """max_workers is a per-tenant cap, so a busy tenant cannot crowd out a
    quiet one by filling the fleet."""
    backend = FakeBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1, boot_timeout_seconds=60)],
    )

    for _ in range(3):
        await pool.submit(tenant_id="loud", machine_type="cpu", payload=b"work")
    assert len(backend.started) == 1

    await pool.submit(tenant_id="quiet", machine_type="cpu", payload=b"work")
    assert len(backend.started) == 2, "the quiet tenant still gets its own machine"

    owners = {w.tenant_id for w in pool.store.list_workers()}
    assert owners == {"loud", "quiet"}


async def test_a_tenant_that_cannot_run_does_not_block_another_tenants_queue(make_pool):
    """Draining is per tenant. A global queue walk would stop at the first job
    the freed machine is not allowed to run and starve everything behind it."""
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", max_workers=1)],
    )

    # Machine 1 belongs to "blocked" and is busy; its queue keeps growing.
    first = await pool.submit(tenant_id="blocked", machine_type="cpu", payload=b"one")
    ran_first = await pool.wait(first.id)
    stuck = pool.store.get_worker(ran_first.worker_id)
    stuck.state = WorkerState.BUSY
    pool.store.save_worker(stuck)
    await pool.submit(tenant_id="blocked", machine_type="cpu", payload=b"two")

    # A different tenant arrives and is served on its own machine.
    other = await pool.submit(tenant_id="other", machine_type="cpu", payload=b"three")
    assert (await pool.wait(other.id)).state is JobState.COMPLETED


async def test_session_affinity_still_applies_within_a_tenant(make_pool):
    pool = make_pool(machine_types=[MachineType(name="cpu", image="w", max_workers=2)])

    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one", session_id="s1")
    second = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"two", session_id="s2"
    )
    ran_first = await pool.wait(first.id)
    await pool.wait(second.id)

    again = await pool.submit(
        tenant_id="acme", machine_type="cpu", payload=b"three", session_id="s1"
    )
    assert (await pool.wait(again.id)).worker_id == ran_first.worker_id


async def test_recovery_places_queued_work_for_every_waiting_tenant(make_pool):
    """After a restart there is no submit to drive placement, so recovery has
    to ask who is waiting rather than assume one tenant."""
    first = make_pool(db_name="shared.sqlite")
    seed = await first.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await first.wait(seed.id)

    for tenant in ("acme", "globex"):
        job = first.store.get_job(done.id)
        job.id = f"job-{tenant}"
        job.tenant_id = tenant
        job.state = JobState.QUEUED
        job.worker_id = None
        job.result = None
        first.store.save_job(job)
    await first.aclose()

    restarted = make_pool(backend=FakeBackend(id_prefix="after-restart"), db_name="shared.sqlite")
    await restarted.recover()

    assert (await restarted.wait("job-acme")).state is JobState.COMPLETED
    assert (await restarted.wait("job-globex")).state is JobState.COMPLETED

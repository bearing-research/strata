"""A machine belongs to one tenant for its life.

A machine that ran one tenant's code is no boundary for the next, and GPU
memory is not reliably zeroed between processes.
"""

import asyncio
import threading

import pytest
from conftest import FakeBackend
from strata_pool import JobState, MachineType, WorkerState
from strata_pool.pool import WORKER_TOKEN_ENV


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
    """max_workers is per tenant, so a busy tenant cannot crowd out a quiet one."""
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
    """Draining is per tenant; a global walk would stall on a job the machine may not run."""
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
    """With no submit after a restart, recovery places queued work for every tenant."""
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


async def test_each_tenants_machine_boots_with_that_tenants_own_variables(
    make_pool, tmp_path, caplog
):
    """Amber item 92: `tenant_env` adds a tenant's variables over the type's, the token last."""
    secrets = {"acme": "acme-key-5f1c", "globex": "globex-key-9e2a"}

    async def tenant_env(tenant_id: str, machine_type: str) -> dict[str, str]:
        assert machine_type == "cpu"
        return {
            "STRATA_NOTEBOOK_CREDENTIALS": secrets[tenant_id],
            "REGION": tenant_id,
            "STRATA_WORKER_TOKEN": "forged",
        }

    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w", env={"REGION": "default", "A": "1"})],
        tenant_env=tenant_env,
    )
    caplog.set_level("DEBUG")

    for tenant_id in ("acme", "globex"):
        job = await pool.submit(tenant_id=tenant_id, machine_type="cpu", payload=b"work")
        assert (await pool.wait(job.id)).state is JobState.COMPLETED

    tokens = [w.auth_token for w in pool.store.list_workers()]
    assert [{k: v for k, v in env.items() if k != WORKER_TOKEN_ENV} for env in backend.envs] == [
        {"STRATA_NOTEBOOK_CREDENTIALS": "acme-key-5f1c", "REGION": "acme", "A": "1"},
        {"STRATA_NOTEBOOK_CREDENTIALS": "globex-key-9e2a", "REGION": "globex", "A": "1"},
    ]
    assert sorted(env[WORKER_TOKEN_ENV] for env in backend.envs) == sorted(tokens)
    stored = b"".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    for secret in secrets.values():
        assert secret.encode() not in stored
        assert secret not in caplog.text


async def test_a_sync_tenant_env_is_used_too(make_pool):
    backend = FakeBackend()
    pool = make_pool(backend=backend, tenant_env=lambda tenant_id, _type: {"T": tenant_id})

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    assert backend.envs[0]["T"] == "acme"


async def test_a_tenant_env_that_raises_fails_the_start_and_the_job_waits(make_pool):
    def tenant_env(tenant_id: str, machine_type: str) -> dict[str, str]:
        raise RuntimeError("the organization's sources are unreachable")

    backend = FakeBackend()
    pool = make_pool(backend=backend, tenant_env=tenant_env)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")

    assert backend.started == []
    assert pool.store.list_workers() == []
    assert pool.store.get_job(job.id).state is JobState.QUEUED


async def test_a_tenant_env_that_never_answers_gives_up_and_the_job_waits(make_pool, caplog):
    """A hung secrets lookup must not hold the submit, nor a scaler pass serving every tenant."""
    never = asyncio.Event()

    async def tenant_env(tenant_id: str, machine_type: str) -> dict[str, str]:
        await never.wait()
        return {}

    backend = FakeBackend()
    pool = make_pool(backend=backend, tenant_env=tenant_env, tenant_env_timeout_seconds=0.05)

    # wait_for only turns a hang into a failure; the pool's own timeout is what returns.
    submit = pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    job = await asyncio.wait_for(submit, 10)

    assert backend.started == []
    assert pool.store.list_workers() == []
    assert pool.store.get_job(job.id).state is JobState.QUEUED
    assert "tenant_env gave no usable environment" in caplog.text


async def test_a_sync_tenant_env_does_not_block_the_event_loop(make_pool):
    """A sync callable may wait on a secrets store; the loop keeps serving meanwhile."""
    entered = threading.Event()
    released = threading.Event()

    def tenant_env(tenant_id: str, machine_type: str) -> dict[str, str]:
        entered.set()
        # Only the event loop sets it, so this is True only if the loop kept running.
        return {"LOOP_RAN": str(released.wait(timeout=10))}

    backend = FakeBackend()
    pool = make_pool(backend=backend, tenant_env=tenant_env)

    submit = asyncio.create_task(pool.submit(tenant_id="acme", machine_type="cpu", payload=b"x"))
    while not entered.is_set():
        await asyncio.sleep(0.01)
    released.set()
    await pool.wait((await submit).id)

    assert backend.envs[0]["LOOP_RAN"] == "True"


@pytest.mark.parametrize(
    "answer",
    [None, ["KEY=leaked-value"], {"KEY": b"leaked-value"}, {"KEY": 1}, {1: "leaked-value"}],
)
async def test_a_tenant_env_answer_that_is_not_strings_starts_nothing(make_pool, caplog, answer):
    """A machine must not boot on an environment the provider would mangle or refuse."""
    backend = FakeBackend()
    pool = make_pool(backend=backend, tenant_env=lambda tenant_id, machine_type: answer)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")

    assert backend.started == []
    assert pool.store.list_workers() == []
    assert pool.store.get_job(job.id).state is JobState.QUEUED
    assert "tenant_env gave no usable environment" in caplog.text
    assert "backend failed" not in caplog.text
    assert "leaked-value" not in caplog.text

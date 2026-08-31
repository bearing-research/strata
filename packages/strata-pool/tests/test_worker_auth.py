"""The credential the pool presents to a machine.

Without it, `POST /execute` is an unauthenticated remote-code-execution
endpoint. That is survivable while the only backend publishes on loopback,
and stops being survivable the moment a machine has a routable address.
"""

import logging

import httpx
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState
from strata_pool.pool import WORKER_TOKEN_ENV


class RecordingBackend(FakeBackend):
    """A backend that remembers the environment each machine was booted with."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.envs: list[dict[str, str]] = []

    async def start(self, spec, env=None):
        self.envs.append(dict(env or {}))
        return await super().start(spec, env)


async def test_the_machine_is_booted_with_a_credential(make_pool):
    backend = RecordingBackend()
    pool = make_pool(backend=backend)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    token = backend.envs[0][WORKER_TOKEN_ENV]
    assert token
    assert pool.store.get_worker(done.worker_id).auth_token == token


async def test_the_pool_presents_that_credential_when_it_dispatches(make_pool):
    backend = RecordingBackend()
    workers = FakeWorkers()
    pool = make_pool(backend=backend, workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await pool.wait(job.id)

    sent = workers.requests[0].headers["authorization"]
    assert sent == f"Bearer {backend.envs[0][WORKER_TOKEN_ENV]}"


async def test_each_machine_gets_its_own_credential(make_pool):
    """One machine's token must not open another's."""
    backend = RecordingBackend(never_healthy=True)
    pool = make_pool(
        backend=backend,
        machine_types=[_two_machine_type()],
    )

    await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")

    tokens = {env[WORKER_TOKEN_ENV] for env in backend.envs}
    assert len(tokens) == 2


async def test_the_credential_survives_a_restart(make_pool):
    """Recovery keeps talking to machines it booted before the crash."""
    backend = RecordingBackend()
    first = make_pool(backend=backend, db_name="shared.sqlite")
    job = await first.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await first.wait(job.id)
    await first.aclose()

    workers = FakeWorkers()
    restarted = make_pool(workers=workers, db_name="shared.sqlite")
    await restarted.recover()

    again = await restarted.submit(tenant_id="acme", machine_type="cpu", payload=b"more")
    await restarted.wait(again.id)

    assert workers.requests[0].headers["authorization"] == (
        f"Bearer {backend.envs[0][WORKER_TOKEN_ENV]}"
    )
    assert restarted.store.get_worker(done.worker_id).auth_token is not None


async def test_a_worker_that_rejects_the_credential_fails_the_job(make_pool):
    workers = FakeWorkers(lambda request: httpx.Response(401, text="unauthorized"))
    pool = make_pool(workers=workers)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    assert done.state is JobState.FAILED
    assert "401" in done.error


async def test_the_credential_stays_out_of_logs(make_pool, caplog):
    """A worker reaches a log line via repr, and a repr'd token is a leak."""
    backend = RecordingBackend()
    pool = make_pool(backend=backend)

    job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    done = await pool.wait(job.id)

    worker = pool.store.get_worker(done.worker_id)
    token = worker.auth_token
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test").info("worker state: %r", worker)

    assert token not in caplog.text
    assert token not in repr(worker)


def _two_machine_type():
    from strata_pool import MachineType

    return MachineType(name="cpu", image="w", max_workers=2, boot_timeout_seconds=60)

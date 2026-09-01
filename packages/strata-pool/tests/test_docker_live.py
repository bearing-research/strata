"""The pool against a real Docker daemon.

This is the reason Docker is the first backend: the whole path — provision,
boot, dispatch, execute, meter, stop — runs against real containers in CI
before any of it produces a number someone pays for. The fakes elsewhere
cannot catch a wrong port binding or a container that never becomes healthy.

Skipped when there is no daemon, which is the common local case.
"""

import asyncio
import os
import time
from dataclasses import replace

import httpx
import pytest
from strata_pool import DockerBackend, JobState, MachineType, Pool, PoolStore
from strata_pool.backends.docker import DEFAULT_SOCKET

IMAGE = "python:3.12-slim"

# A worker: 200 on any GET (the health check), and POST /execute echoes its
# body back so the test can prove the payload made the round trip. It enforces
# the bearer token, which is what makes these tests prove the credential
# actually reaches the container and matches what the pool sends — a 401 here
# fails the job, so every passing test below is an assertion about auth.
WORKER_SCRIPT = """
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKEN = os.environ["STRATA_WORKER_TOKEN"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        if self.headers.get("Authorization") != "Bearer " + TOKEN:
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"unauthorized")
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ran:" + body)

    def log_message(self, *args):
        pass


HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
"""

pytestmark = pytest.mark.docker


def _daemon_is_reachable() -> bool:
    socket_path = os.environ.get("STRATA_POOL_DOCKER_SOCKET", DEFAULT_SOCKET)
    if not os.path.exists(socket_path):
        return False
    try:
        with httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path), base_url="http://docker"
        ) as client:
            return client.get("/_ping", timeout=5.0).status_code == 200
    except httpx.HTTPError:
        return False


_DAEMON_IS_REACHABLE = _daemon_is_reachable()

if os.environ.get("STRATA_POOL_REQUIRE_DOCKER") == "1" and not _DAEMON_IS_REACHABLE:
    # CI sets this. Without it, a runner whose socket moved would skip every
    # test in this file and report green — coverage that examined nothing,
    # which is the exact failure mode this package keeps guarding against.
    raise RuntimeError(
        "STRATA_POOL_REQUIRE_DOCKER=1 but no Docker daemon is reachable at "
        f"{os.environ.get('STRATA_POOL_DOCKER_SOCKET', DEFAULT_SOCKET)}"
    )

requires_docker = pytest.mark.skipif(not _DAEMON_IS_REACHABLE, reason="no reachable Docker daemon")


def _socket() -> str:
    return os.environ.get("STRATA_POOL_DOCKER_SOCKET", DEFAULT_SOCKET)


def _daemon_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=_socket()),
        base_url="http://docker",
        timeout=600.0,
    )


@pytest.fixture
async def docker_pool(tmp_path):
    """A pool wired to a real daemon, with every container it started removed.

    Cleanup goes by label rather than by what the store remembers: a test that
    exercises a machine dying deletes the row, and the container still has to
    go.
    """
    async with _daemon_client() as daemon:
        # /containers/create does not pull, and the pool does not either yet.
        pull = await daemon.post(
            "/images/create", params={"fromImage": "python", "tag": "3.12-slim"}
        )
        assert pull.status_code == 200, pull.text[:300]

        backend = DockerBackend(
            socket_path=_socket(),
            command=["python", "-c", WORKER_SCRIPT],
        )
        store = PoolStore(tmp_path / "pool.sqlite")
        pool = Pool(
            store,
            backend,
            [
                MachineType(
                    name="cpu",
                    image=IMAGE,
                    max_workers=2,
                    boot_timeout_seconds=90,
                    cpus=1.0,
                    memory_mb=512,
                )
            ],
        )

        try:
            yield pool
        finally:
            await pool.aclose()
            leftovers = await daemon.get(
                "/containers/json",
                params={"all": "true", "filters": '{"label":["strata.pool.machine-type"]}'},
            )
            for container in leftovers.json():
                await backend.stop(container["Id"])
            await backend.aclose()
            store.close()


@requires_docker
async def test_the_daemon_applies_the_resource_limits_we_asked_for(docker_pool):
    """The limits are only real if the daemon accepted them; a field name we
    got wrong would be ignored silently."""
    job = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await docker_pool.wait(job.id, timeout=120)

    worker = docker_pool.store.list_workers()[0]
    async with _daemon_client() as daemon:
        inspected = (await daemon.get(f"/containers/{worker.backend_id}/json")).json()

    assert inspected["HostConfig"]["NanoCpus"] == 1_000_000_000
    assert inspected["HostConfig"]["Memory"] == 512 * 1024 * 1024


@requires_docker
async def test_a_job_runs_in_a_real_container_and_is_metered(docker_pool):
    job = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"payload")
    done = await docker_pool.wait(job.id, timeout=120)

    assert done.state is JobState.COMPLETED
    assert done.result == b"ran:payload"

    event = docker_pool.store.list_usage("acme")[0]
    assert event.job_id == job.id
    assert event.duration_ms > 0


@requires_docker
async def test_a_second_job_reuses_the_container_that_is_already_warm(docker_pool):
    first = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    ran_first = await docker_pool.wait(first.id, timeout=120)

    second = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    ran_second = await docker_pool.wait(second.id, timeout=120)

    assert ran_second.state is JobState.COMPLETED
    assert ran_second.worker_id == ran_first.worker_id


async def _eventually_gone(store, worker_id: str, timeout: float = 30.0) -> bool:
    """Wait for a worker row to disappear.

    The job is answered before the machine is torn down — stopping a container
    is a round trip to the daemon, and no caller should wait on it — so
    cleanup lands shortly after `wait()` returns, not before.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if store.get_worker(worker_id) is None:
            return True
        await asyncio.sleep(0.05)
    return False


@requires_docker
async def test_a_container_that_dies_mid_flight_fails_the_job_and_leaves_no_worker(docker_pool):
    """A machine that vanishes must not stay in the fleet taking work."""
    first = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"one")
    await docker_pool.wait(first.id, timeout=120)

    worker = docker_pool.store.list_workers()[0]
    await docker_pool.backend.stop(worker.backend_id)

    second = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"two")
    done = await docker_pool.wait(second.id, timeout=120)

    assert done.state is JobState.FAILED
    assert "unreachable" in done.error
    assert await _eventually_gone(docker_pool.store, worker.id)


@requires_docker
async def test_the_scaler_removes_an_idle_container_from_the_daemon(docker_pool):
    """The row disappearing is not the point; the container disappearing is.

    A stop path that updated the database and left the machine running would
    keep billing while the pool believed it had scaled down.
    """
    job = await docker_pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
    await docker_pool.wait(job.id, timeout=120)

    worker = docker_pool.store.list_workers()[0]
    docker_pool.machine_types["cpu"] = replace(
        docker_pool.machine_types["cpu"], cool_down_seconds=0.0
    )

    assert await docker_pool.reap_idle_workers() == 1

    async with _daemon_client() as daemon:
        inspected = await daemon.get(f"/containers/{worker.backend_id}/json")
    assert inspected.status_code == 404, "the daemon should no longer know this container"
    assert docker_pool.store.list_workers() == []

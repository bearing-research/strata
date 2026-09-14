"""Every request the Fly Machines backend makes.

These assert the shape of what we send, which only a live account can confirm
Fly wants (see ``test_fly_live.py``), and they stop it drifting afterwards.
"""

import json

import httpx
import pytest
from strata_pool import MachineType
from strata_pool.backends.fly import FlyBackend, FlyError, private_url


class FakeFly:
    """Answers the Machines API and records what it was asked."""

    def __init__(self, state: str = "started", **overrides: httpx.Response):
        self.requests: list[httpx.Request] = []
        self.state = state
        self.overrides = overrides

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        if key in self.overrides:
            return self.overrides[key]
        if request.method == "POST" and request.url.path == "/apps/pool/machines":
            return httpx.Response(200, json={"id": "e28650", "region": "sjc", "state": "created"})
        if request.method == "GET":
            return httpx.Response(200, json={"id": "e28650", "state": self.state})
        return httpx.Response(200, json={"ok": True})

    def created_body(self) -> dict:
        for request in self.requests:
            if request.method == "POST":
                return json.loads(request.content)
        raise AssertionError("never asked Fly to create a machine")


def _backend(fake: FakeFly, probe_status: int = 200, **kwargs) -> tuple[FlyBackend, list[str]]:
    probed: list[str] = []

    def _probe(request: httpx.Request) -> httpx.Response:
        probed.append(str(request.url))
        return httpx.Response(probe_status)

    backend = FlyBackend(
        "fly-token",
        app="pool",
        region="sjc",
        api=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handle), base_url="https://fly.test"
        ),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(_probe)),
        **kwargs,
    )
    return backend, probed


def _spec(**kwargs) -> MachineType:
    return MachineType(
        name=kwargs.pop("name", "a100"), image=kwargs.pop("image", "strata-worker:1"), **kwargs
    )


async def test_a_machine_is_created_in_the_app_and_region_and_reached_privately():
    fake = FakeFly()
    backend, _ = _backend(fake)

    worker = await backend.start(
        _spec(), {"STRATA_WORKER_TOKEN": "tok", "STRATA_WORKER_MAX_CONCURRENT": "1"}
    )

    create = fake.requests[0]
    assert (create.method, create.url.path) == ("POST", "/apps/pool/machines")
    assert create.headers["Authorization"] == "Bearer fly-token"
    body = fake.created_body()
    assert body["region"] == "sjc"
    assert body["config"]["image"] == "strata-worker:1"
    assert body["config"]["env"] == {
        "STRATA_WORKER_TOKEN": "tok",
        "STRATA_WORKER_MAX_CONCURRENT": "1",
    }
    # The pool owns the machine's lifetime, not Fly.
    assert body["config"]["restart"] == {"policy": "no"}
    assert body["config"]["auto_destroy"] is False
    assert body["name"].startswith("strata-a100-")
    assert worker.backend_id == "e28650"
    assert worker.endpoint == "http://e28650.vm.pool.internal:8080"
    assert worker.region == "sjc"


async def test_the_hardware_is_asked_for_in_the_guest_block():
    fake = FakeFly()
    backend, _ = _backend(fake)

    await backend.start(_spec(cpus=8, memory_mb=65536, gpu_type="a100-80gb", gpu_count=2))

    assert fake.created_body()["config"]["guest"] == {
        "cpus": 8,
        "cpu_kind": "performance",
        "memory_mb": 65536,
        "gpu_kind": "a100-80gb",
        "gpus": 2,
    }


async def test_provider_options_override_but_cannot_drop_the_credential():
    """A machine without its token runs code for anything on the network."""
    fake = FakeFly()
    backend, _ = _backend(fake)

    await backend.start(
        _spec(provider_options={"env": {"OTHER": "1"}, "auto_destroy": True}),
        {"STRATA_WORKER_TOKEN": "tok"},
    )

    config = fake.created_body()["config"]
    assert config["env"] == {"OTHER": "1", "STRATA_WORKER_TOKEN": "tok"}
    assert config["auto_destroy"] is True


async def test_an_env_override_that_cannot_hold_the_credential_is_refused():
    fake = FakeFly()
    backend, _ = _backend(fake)

    with pytest.raises(FlyError, match="credential"):
        await backend.start(
            _spec(provider_options={"env": ["STRATA_WORKER_TOKEN=x"]}),
            {"STRATA_WORKER_TOKEN": "tok"},
        )

    assert fake.requests == []


async def test_a_created_machine_without_an_id_is_named_in_the_error():
    fake = FakeFly(**{"POST /apps/pool/machines": httpx.Response(200, json={"region": "sjc"})})
    backend, _ = _backend(fake)

    with pytest.raises(FlyError, match="strata-a100-"):
        await backend.start(_spec())


async def test_a_refused_create_says_why():
    fake = FakeFly(
        **{"POST /apps/pool/machines": httpx.Response(422, json={"error": "invalid image"})}
    )
    backend, _ = _backend(fake)

    with pytest.raises(FlyError, match="invalid image"):
        await backend.start(_spec())


async def test_stop_destroys_by_force_and_is_idempotent():
    fake = FakeFly(
        **{"DELETE /apps/pool/machines/gone": httpx.Response(404, json={"error": "not found"})}
    )
    backend, _ = _backend(fake)

    await backend.stop("e28650")
    await backend.stop("gone")

    delete = fake.requests[0]
    assert (delete.method, delete.url.path, delete.url.params["force"]) == (
        "DELETE",
        "/apps/pool/machines/e28650",
        "true",
    )


async def test_stop_that_fly_refuses_raises():
    fake = FakeFly(
        **{"DELETE /apps/pool/machines/e28650": httpx.Response(500, json={"error": "boom"})}
    )
    backend, _ = _backend(fake)

    with pytest.raises(FlyError, match="boom"):
        await backend.stop("e28650")


async def test_healthy_means_started_and_answering():
    fake = FakeFly(state="started")
    backend, probed = _backend(fake)
    endpoint = private_url("e28650", "pool", 8080)

    assert await backend.health(endpoint) is True
    assert fake.requests[0].url.path == "/apps/pool/machines/e28650"
    assert probed == ["http://e28650.vm.pool.internal:8080/health"]


async def test_a_machine_fly_has_not_started_is_not_healthy_and_is_not_probed():
    """Its private address may already belong to another machine."""
    fake = FakeFly(state="stopped")
    backend, probed = _backend(fake)

    assert await backend.health(private_url("e28650", "pool", 8080)) is False
    assert probed == []


async def test_a_started_machine_whose_worker_is_not_serving_yet_is_not_healthy():
    fake = FakeFly(state="started")
    backend, _ = _backend(fake, probe_status=502)

    assert await backend.health(private_url("e28650", "pool", 8080)) is False


async def test_the_pool_boots_a_fly_worker_for_a_job_and_destroys_it_after_cooldown(tmp_path):
    """Item 51's whole loop, with Fly and the worker answered locally: a job
    boots a machine, runs on its private endpoint, and the idle machine is
    destroyed once its cooldown passes."""
    from strata_pool import JobState, Pool, PoolStore

    fake = FakeFly(state="started")
    backend, probed = _backend(fake)
    executed: list[str] = []

    def _worker(request: httpx.Request) -> httpx.Response:
        executed.append(str(request.url))
        return httpx.Response(200, content=b"done")

    store = PoolStore(tmp_path / "pool.sqlite")
    pool = Pool(
        store,
        backend,
        [_spec(name="cpu", cool_down_seconds=0.0)],
        client=httpx.AsyncClient(transport=httpx.MockTransport(_worker)),
        health_poll_seconds=0.01,
    )
    try:
        job = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"work")
        done = await pool.wait(job.id, timeout=10)

        assert done.state is JobState.COMPLETED
        assert executed == ["http://e28650.vm.pool.internal:8080/execute"]
        assert probed and probed[0].endswith(".vm.pool.internal:8080/health")

        assert await pool.reap_idle_workers() == 1
        destroyed = [r for r in fake.requests if r.method == "DELETE"]
        assert [r.url.path for r in destroyed] == ["/apps/pool/machines/e28650"]
        assert store.list_workers() == []
    finally:
        await pool.aclose()
        store.close()

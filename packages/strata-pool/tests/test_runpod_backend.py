"""Every request the RunPod backend makes.

These assert the *shape* of what we send. They cannot tell us the shape is
what RunPod actually wants — only a live account can do that — but they make
correcting it a one-line edit, and they stop it drifting afterwards.
"""

import httpx
import pytest
from strata_pool import MachineType
from strata_pool.backends.runpod import RunPodBackend, RunPodError, proxy_url


class FakeRunPod:
    """Answers RunPod's API and records what it was asked."""

    def __init__(self, **overrides: httpx.Response):
        self.requests: list[httpx.Request] = []
        self.overrides = overrides

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        key = f"{request.method} {request.url.path}"
        if key in self.overrides:
            return self.overrides[key]
        if request.url.path == "/pods" and request.method == "POST":
            return httpx.Response(
                201, json={"id": "pod123", "machine": {"dataCenterId": "US-KS-2"}}
            )
        return httpx.Response(200, json={})

    def body(self) -> dict:
        import json

        for request in self.requests:
            if request.url.path == "/pods" and request.method == "POST":
                return json.loads(request.content)
        raise AssertionError("never asked RunPod to create a pod")


def _backend(fake: FakeRunPod, **kwargs) -> RunPodBackend:
    return RunPodBackend(
        api_key="secret",
        api=httpx.AsyncClient(
            transport=httpx.MockTransport(fake.handle), base_url="https://runpod.test"
        ),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        **kwargs,
    )


def _spec(**kwargs) -> MachineType:
    return MachineType(
        name=kwargs.pop("name", "h100-80gb"),
        image=kwargs.pop("image", "strata-worker:latest"),
        **kwargs,
    )


async def test_starting_a_pod_returns_its_proxy_endpoint():
    fake = FakeRunPod()
    worker = await _backend(fake).start(_spec(gpu_type="NVIDIA H100 80GB PCIe"))

    assert worker.backend_id == "pod123"
    assert worker.endpoint == "https://pod123-8080.proxy.runpod.net"
    assert worker.region == "US-KS-2"


async def test_the_endpoint_is_derived_not_waited_for():
    """It is available the moment the pod exists, so the pool can start
    polling before RunPod reports the pod as running."""
    assert proxy_url("abc", 8080) == "https://abc-8080.proxy.runpod.net"
    assert proxy_url("abc", 9000) == "https://abc-9000.proxy.runpod.net"


async def test_the_gpu_is_asked_for_by_the_providers_own_name():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(gpu_type="NVIDIA H100 80GB PCIe", gpu_count=2))

    body = fake.body()
    assert body["gpuTypeIds"] == ["NVIDIA H100 80GB PCIe"]
    assert body["gpuCount"] == 2


async def test_a_cpu_machine_asks_for_no_gpu_at_all():
    """Sending gpuCount for a CPU pod would either be rejected or, worse,
    quietly rent a GPU."""
    fake = FakeRunPod()
    await _backend(fake).start(_spec())

    body = fake.body()
    assert "gpuTypeIds" not in body
    assert "gpuCount" not in body


async def test_the_worker_credential_reaches_the_pod():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(), {"STRATA_WORKER_TOKEN": "abc123"})

    assert fake.body()["env"]["STRATA_WORKER_TOKEN"] == "abc123"


async def test_the_worker_port_is_published_as_http():
    fake = FakeRunPod()
    await _backend(fake, worker_port=9000).start(_spec())

    assert fake.body()["ports"] == ["9000/http"]


async def test_the_pod_is_named_for_the_pool_that_started_it():
    """The only backstop for an orphaned pod is finding it in the console."""
    fake = FakeRunPod()
    await _backend(fake).start(_spec(name="h100-80gb"))

    assert fake.body()["name"].startswith("strata-h100-80gb-")


async def test_provider_options_can_correct_anything_above_them():
    """A deployment should not have to wait for a release to fix a field we
    got wrong."""
    fake = FakeRunPod()
    await _backend(fake).start(
        _spec(
            gpu_type="NVIDIA H100 80GB PCIe",
            provider_options={"cloudType": "SECURE", "gpuTypeIds": ["corrected"]},
        )
    )

    body = fake.body()
    assert body["cloudType"] == "SECURE"
    assert body["gpuTypeIds"] == ["corrected"]


async def test_disk_is_only_sent_when_asked_for():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(disk_gb=40))
    assert fake.body()["containerDiskInGb"] == 40

    bare = FakeRunPod()
    await _backend(bare).start(_spec())
    assert "containerDiskInGb" not in bare.body()


async def test_a_refusal_surfaces_runpods_own_message():
    fake = FakeRunPod(**{"POST /pods": httpx.Response(400, json={"error": "no h100 capacity"})})
    with pytest.raises(RunPodError, match="no h100 capacity"):
        await _backend(fake).start(_spec(gpu_type="NVIDIA H100 80GB PCIe"))


async def test_a_pod_created_without_an_id_is_an_error_not_a_bad_endpoint():
    """Building a proxy URL out of None produces a worker that can never be
    reached and never be stopped."""
    fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json={})})
    with pytest.raises(RunPodError, match="without an id"):
        await _backend(fake).start(_spec())


async def test_stopping_terminates_rather_than_pauses():
    """A stopped RunPod pod keeps its disk, and keeps charging for it."""
    fake = FakeRunPod()
    await _backend(fake).stop("pod123")

    assert [(r.method, r.url.path) for r in fake.requests] == [("DELETE", "/pods/pod123")]


async def test_terminating_a_pod_that_is_already_gone_is_not_an_error():
    fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(404, json={"error": "not found"})})
    await _backend(fake).stop("pod123")  # idempotent, per the Backend protocol


async def test_a_pod_that_will_not_terminate_is_loud():
    """Silence here is a GPU nobody is watching."""
    fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(500, json={"error": "busy"})})
    with pytest.raises(RunPodError, match="busy"):
        await _backend(fake).stop("pod123")


async def test_the_api_key_travels_as_a_bearer_token():
    fake = FakeRunPod()
    await _backend(fake).start(_spec())

    assert fake.requests[0].headers["authorization"] == "Bearer secret"


class TestHealth:
    async def test_a_booting_pod_is_not_healthy_rather_than_an_error(self):
        """RunPod's proxy answers 502 until the pod serves, and the pool polls
        this in a loop."""
        backend = RunPodBackend(
            api_key="secret",
            api=httpx.AsyncClient(transport=httpx.MockTransport(FakeRunPod().handle)),
            probe=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(502))),
        )
        assert await backend.health("https://pod123-8080.proxy.runpod.net") is False

    async def test_an_unreachable_proxy_is_not_healthy_rather_than_an_error(self):
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure", request=request)

        backend = RunPodBackend(
            api_key="secret",
            api=httpx.AsyncClient(transport=httpx.MockTransport(FakeRunPod().handle)),
            probe=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
        )
        assert await backend.health("https://pod123-8080.proxy.runpod.net") is False

    async def test_health_asks_the_worker_not_runpod(self):
        seen: list[str] = []

        def probe(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200)

        backend = RunPodBackend(
            api_key="secret",
            api=httpx.AsyncClient(transport=httpx.MockTransport(FakeRunPod().handle)),
            probe=httpx.AsyncClient(transport=httpx.MockTransport(probe)),
        )
        assert await backend.health("https://pod123-8080.proxy.runpod.net") is True
        assert seen == ["https://pod123-8080.proxy.runpod.net/health"]

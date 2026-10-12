"""The shape of every request the RunPod backend makes.

Only a live account confirms RunPod wants it (``test_runpod_live.py``); these stop drift.
"""

import re

import httpx
import pytest
from strata_pool import MachineType
from strata_pool.backends.runpod import (
    DEFAULT_BASE_URL,
    RunPodBackend,
    RunPodError,
    proxy_url,
)


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
            return httpx.Response(201, json={"id": "pod123", "dataCenterId": "US-KS-2"})
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
    """The proxy URL exists with the pod, so polling can start before RunPod reports it running."""
    assert proxy_url("abc", 8080) == "https://abc-8080.proxy.runpod.net"
    assert proxy_url("abc", 9000) == "https://abc-9000.proxy.runpod.net"


async def test_the_gpu_is_asked_for_by_the_providers_own_name():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(gpu_type="NVIDIA H100 80GB PCIe", gpu_count=2))

    body = fake.body()
    assert body["gpu"] == {"id": "NVIDIA H100 80GB PCIe", "count": 2}
    assert "cpu" not in body
    assert body["image"] == "strata-worker:latest"


async def test_a_cpu_machine_asks_for_a_cpu_flavor_and_no_gpu():
    """v2 refuses a pod with neither `gpu` nor `cpu`; a stray `gpu` would rent one."""
    fake = FakeRunPod()
    await _backend(fake).start(_spec())

    body = fake.body()
    assert body["cpu"] == {"id": "cpu3g", "vcpuCount": 2}
    assert "gpu" not in body


@pytest.mark.parametrize(
    ("cpus", "vcpus"), [(None, 2), (1, 2), (2, 2), (3, 4), (8, 8), (12, 16), (64, 32)]
)
async def test_cpus_round_up_to_a_power_of_two_runpod_accepts(cpus, vcpus):
    fake = FakeRunPod()
    await _backend(fake).start(_spec(cpus=cpus))
    assert fake.body()["cpu"]["vcpuCount"] == vcpus


async def test_the_default_base_url_is_rest_v2():
    """v1 (rest.runpod.io/v1) is retired on 2026-11-15."""
    seen: list[str] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(201, json={"id": "pod123"})

    api = httpx.AsyncClient(transport=httpx.MockTransport(record), base_url=DEFAULT_BASE_URL)
    backend = RunPodBackend(api_key="secret", api=api)
    try:
        await backend.start(_spec())
    finally:
        await backend.aclose()
        await api.aclose()
    assert seen == ["https://api.runpod.io/v2/pods"]


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
    """A deployment can fix a wrong field without waiting for a release."""
    fake = FakeRunPod()
    await _backend(fake).start(
        _spec(
            gpu_type="NVIDIA H100 80GB PCIe",
            provider_options={"cloud": "COMMUNITY", "image": "corrected:1"},
        )
    )

    body = fake.body()
    assert body["cloud"] == "COMMUNITY"
    assert body["image"] == "corrected:1"


async def test_a_gpu_override_merges_rather_than_drops_the_gpu_type():
    """Pinning a CUDA floor must not leave a pod with no GPU type."""
    fake = FakeRunPod()
    await _backend(fake).start(
        _spec(
            gpu_type="NVIDIA H100 80GB PCIe", provider_options={"gpu": {"minCudaVersion": "12.8"}}
        )
    )

    assert fake.body()["gpu"] == {
        "id": "NVIDIA H100 80GB PCIe",
        "count": 1,
        "minCudaVersion": "12.8",
    }


async def test_a_cpu_override_can_pick_another_flavor():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(cpus=4, provider_options={"cpu": {"id": "cpu5c"}}))

    assert fake.body()["cpu"] == {"id": "cpu5c", "vcpuCount": 4}


async def test_disk_is_only_sent_when_asked_for():
    fake = FakeRunPod()
    await _backend(fake).start(_spec(disk_gb=40))
    assert fake.body()["disk"] == 40

    bare = FakeRunPod()
    await _backend(bare).start(_spec())
    assert "disk" not in bare.body()


async def test_a_refusal_surfaces_runpods_own_message():
    refusal = {"title": "Bad Request", "status": 400, "detail": "no h100 capacity"}
    fake = FakeRunPod(**{"POST /pods": httpx.Response(400, json=refusal)})
    with pytest.raises(RunPodError, match="HTTP 400: no h100 capacity$"):
        await _backend(fake).start(_spec(gpu_type="NVIDIA H100 80GB PCIe"))


async def test_a_validation_failure_names_each_rejected_field():
    """v2's `detail` only summarizes a 422; the field it rejected is in `errors`."""
    refusal = {
        "title": "Unprocessable Entity",
        "status": 422,
        "detail": "Request validation failed.",
        "errors": ["$: additional properties 'cloudType' not allowed"],
    }
    fake = FakeRunPod(**{"POST /pods": httpx.Response(422, json=refusal)})
    expected = (
        "HTTP 422: Request validation failed. ($: additional properties 'cloudType' not allowed)"
    )
    with pytest.raises(RunPodError, match=re.escape(expected)):
        await _backend(fake).start(_spec())


async def test_a_pod_created_without_an_id_is_an_error_not_a_bad_endpoint():
    """A proxy URL built from None would be a worker nobody can reach or stop."""
    fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json={})})
    with pytest.raises(RunPodError, match="without an id"):
        await _backend(fake).start(_spec())


async def test_stopping_terminates_rather_than_pauses():
    """A stopped RunPod pod keeps its disk, and keeps charging for it."""
    fake = FakeRunPod()
    await _backend(fake).stop("pod123")

    assert [(r.method, r.url.path) for r in fake.requests] == [("DELETE", "/pods/pod123")]


async def test_terminating_a_pod_that_is_already_gone_is_not_an_error():
    gone = {"title": "Not Found", "status": 404, "detail": "pod not found"}
    fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(404, json=gone)})
    await _backend(fake).stop("pod123")  # idempotent, per the Backend protocol


async def test_a_pod_that_will_not_terminate_is_loud():
    """Silence here is a GPU nobody is watching."""
    busy = {"title": "Internal Server Error", "status": 500, "detail": "busy"}
    fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(500, json=busy)})
    with pytest.raises(RunPodError, match="busy"):
        await _backend(fake).stop("pod123")


async def test_the_api_key_travels_as_a_bearer_token():
    fake = FakeRunPod()
    await _backend(fake).start(_spec())

    assert fake.requests[0].headers["authorization"] == "Bearer secret"


class TestHealth:
    async def test_a_booting_pod_is_not_healthy_rather_than_an_error(self):
        """RunPod's proxy answers 502 until the pod serves."""
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


class TestResponseShapes:
    """Response shapes that must not raise once a pod exists and bills.

    A raising parse is a failed start, which deletes the row holding the backend_id.
    """

    async def test_a_pod_not_yet_placed_has_no_region(self):
        """`dataCenterId` is null until the scheduler places the pod."""
        fake = FakeRunPod(
            **{"POST /pods": httpx.Response(201, json={"id": "pod123", "dataCenterId": None})}
        )
        worker = await _backend(fake).start(_spec())

        assert worker.backend_id == "pod123"
        assert worker.region is None

    async def test_a_non_json_success_body_is_an_error_not_a_crash(self):
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, text="<html>gateway</html>")})
        with pytest.raises(RunPodError, match="non-JSON"):
            await _backend(fake).start(_spec())

    async def test_an_unidentifiable_pod_is_named_in_the_error(self):
        """The generated name is the only handle left on a billing pod with no id."""
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json={})})
        with pytest.raises(RunPodError, match=r"strata-h100-80gb-[0-9a-f]{8}"):
            await _backend(fake).start(_spec())

    async def test_a_json_array_success_body_is_an_error_not_a_crash(self):
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json=[{"id": "pod123"}])})
        with pytest.raises(RunPodError, match="unexpected body shape"):
            await _backend(fake).start(_spec())

    async def test_a_bare_string_error_body_still_reports_the_status(self):
        """A raising formatter would hide RunPod's real complaint, such as a bad key."""
        fake = FakeRunPod(**{"POST /pods": httpx.Response(401, json="Unauthorized")})
        with pytest.raises(RunPodError, match="401"):
            await _backend(fake).start(_spec())

    async def test_a_bare_string_error_body_on_stop_still_reports_the_status(self):
        fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(403, json="Forbidden")})
        with pytest.raises(RunPodError, match="403"):
            await _backend(fake).stop("pod123")


class TestCredentialCannotBeDropped:
    async def test_adding_an_env_var_does_not_delete_the_worker_credential(self):
        """Overriding env (say, to add HF_TOKEN) must not drop the worker token."""
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": {"HF_TOKEN": "hf_abc"}}),
            {"STRATA_WORKER_TOKEN": "secret-token"},
        )

        env = fake.body()["env"]
        assert env["HF_TOKEN"] == "hf_abc"
        assert env["STRATA_WORKER_TOKEN"] == "secret-token"

    async def test_an_override_cannot_substitute_its_own_credential(self):
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": {"STRATA_WORKER_TOKEN": "attacker"}}),
            {"STRATA_WORKER_TOKEN": "ours"},
        )

        assert fake.body()["env"]["STRATA_WORKER_TOKEN"] == "ours"

    async def test_an_override_can_still_correct_other_env_values(self):
        """`provider_options` overrides everything except the credential, even caller-set values."""
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": {"HF_HOME": "/corrected"}}),
            {"STRATA_WORKER_TOKEN": "ours", "HF_HOME": "/wrong"},
        )

        env = fake.body()["env"]
        assert env["HF_HOME"] == "/corrected"
        assert env["STRATA_WORKER_TOKEN"] == "ours"

    @pytest.mark.parametrize("shape", ["A=1", [{"key": "A", "value": "1"}]])
    async def test_an_env_shape_with_nowhere_to_put_a_credential_is_refused(self, shape):
        """v2 takes env only as a map; GraphQL's list form included, nothing else is sent."""
        fake = FakeRunPod()
        with pytest.raises(RunPodError, match="open execute endpoint"):
            await _backend(fake).start(
                _spec(provider_options={"env": shape}),
                {"STRATA_WORKER_TOKEN": "ours"},
            )
        assert fake.requests == []


class TestClientOwnership:
    async def test_injecting_one_client_does_not_leak_the_other(self):
        """With only `api` injected, the backend still closes the probe client it built."""
        injected = httpx.AsyncClient(transport=httpx.MockTransport(FakeRunPod().handle))
        backend = RunPodBackend(api_key="secret", api=injected)
        probe = backend._probe

        await backend.aclose()

        assert probe.is_closed, "the client we built is ours to close"
        assert not injected.is_closed, (
            "closing a caller's shared client breaks every other user of it"
        )
        await injected.aclose()

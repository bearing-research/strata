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


class TestResponseShapes:
    """Everything here runs with a pod already created and already billing.

    A parse that raises is caught upstream as a failed start, which deletes
    the row holding the backend_id — so nothing could ever terminate the pod.
    These are the shapes that must not do that.
    """

    async def test_a_pod_not_yet_placed_on_a_host_has_a_null_machine(self):
        """The normal shape immediately after create. `.get("machine", {})`
        returns None for it, not the default."""
        fake = FakeRunPod(
            **{"POST /pods": httpx.Response(201, json={"id": "pod123", "machine": None})}
        )
        worker = await _backend(fake).start(_spec())

        assert worker.backend_id == "pod123"
        assert worker.region is None

    async def test_a_machine_field_that_is_not_an_object_is_survivable(self):
        fake = FakeRunPod(
            **{"POST /pods": httpx.Response(201, json={"id": "pod123", "machine": "us-ks-2"})}
        )
        assert (await _backend(fake).start(_spec())).region is None

    async def test_a_non_json_success_body_is_an_error_not_a_crash(self):
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, text="<html>gateway</html>")})
        with pytest.raises(RunPodError, match="non-JSON"):
            await _backend(fake).start(_spec())

    async def test_an_unidentifiable_pod_is_named_in_the_error(self):
        """The pool deletes the row that would have held the id, so the
        generated name is the only handle anyone has left for a pod that is
        already billing."""
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json={})})
        with pytest.raises(RunPodError, match=r"strata-h100-80gb-[0-9a-f]{8}"):
            await _backend(fake).start(_spec())

    async def test_a_json_array_success_body_is_an_error_not_a_crash(self):
        fake = FakeRunPod(**{"POST /pods": httpx.Response(201, json=[{"id": "pod123"}])})
        with pytest.raises(RunPodError, match="unexpected body shape"):
            await _backend(fake).start(_spec())

    async def test_a_bare_string_error_body_still_reports_the_status(self):
        """A formatter that raises replaces RunPod's actual complaint with a
        traceback, and in stop() that is swallowed as "may still be billing"
        while the operator never learns the key is wrong."""
        fake = FakeRunPod(**{"POST /pods": httpx.Response(401, json="Unauthorized")})
        with pytest.raises(RunPodError, match="401"):
            await _backend(fake).start(_spec())

    async def test_a_bare_string_error_body_on_stop_still_reports_the_status(self):
        fake = FakeRunPod(**{"DELETE /pods/pod123": httpx.Response(403, json="Forbidden")})
        with pytest.raises(RunPodError, match="403"):
            await _backend(fake).stop("pod123")


class TestCredentialCannotBeDropped:
    async def test_adding_an_env_var_does_not_delete_the_worker_credential(self):
        """A pod without its token is an open execute endpoint on a public URL,
        and adding one HF_TOKEN is a plausible way to get there by accident."""
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

    async def test_the_list_encoding_is_supported_not_refused(self):
        """RunPod's GraphQL surface takes env as a list of {key, value}. That
        is the field most likely to be wrong here, so refusing the correction
        would close the escape hatch on exactly the case it exists for."""
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": [{"key": "HF_TOKEN", "value": "hf_abc"}]}),
            {"STRATA_WORKER_TOKEN": "ours"},
        )

        env = fake.body()["env"]
        assert {"key": "HF_TOKEN", "value": "hf_abc"} in env
        assert {"key": "STRATA_WORKER_TOKEN", "value": "ours"} in env

    async def test_a_list_encoding_cannot_smuggle_its_own_credential(self):
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": [{"key": "STRATA_WORKER_TOKEN", "value": "attacker"}]}),
            {"STRATA_WORKER_TOKEN": "ours"},
        )

        tokens = [
            entry["value"] for entry in fake.body()["env"] if entry["key"] == "STRATA_WORKER_TOKEN"
        ]
        assert tokens == ["ours"]

    async def test_an_override_can_still_correct_other_env_values(self):
        """`provider_options` overrides everything except the credential —
        including values the caller set."""
        fake = FakeRunPod()
        await _backend(fake).start(
            _spec(provider_options={"env": {"HF_HOME": "/corrected"}}),
            {"STRATA_WORKER_TOKEN": "ours", "HF_HOME": "/wrong"},
        )

        env = fake.body()["env"]
        assert env["HF_HOME"] == "/corrected"
        assert env["STRATA_WORKER_TOKEN"] == "ours"

    async def test_an_env_shape_with_nowhere_to_put_a_credential_is_refused(self):
        fake = FakeRunPod()
        with pytest.raises(RunPodError, match="open execute endpoint"):
            await _backend(fake).start(
                _spec(provider_options={"env": "A=1"}),
                {"STRATA_WORKER_TOKEN": "ours"},
            )


class TestClientOwnership:
    async def test_injecting_one_client_does_not_leak_the_other(self):
        """The documented reason to inject `api` is retries on the control
        plane; the probe is then built here and nobody else can close it."""
        injected = httpx.AsyncClient(transport=httpx.MockTransport(FakeRunPod().handle))
        backend = RunPodBackend(api_key="secret", api=injected)
        probe = backend._probe

        await backend.aclose()

        assert probe.is_closed, "the client we built is ours to close"
        assert not injected.is_closed, (
            "closing a caller's shared client breaks every other user of it"
        )
        await injected.aclose()

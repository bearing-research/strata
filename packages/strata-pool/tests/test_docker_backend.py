"""The Docker Engine API calls, without a daemon.

Every request the backend makes is asserted here against a mock transport;
`test_docker_live.py` runs the same code against a real daemon when there is
one. These stay fast and always run.
"""

import httpx
import pytest
from strata_pool import DockerBackend
from strata_pool.backends import DockerError


class FakeDaemon:
    """Answers the Docker Engine API and records what it was asked."""

    def __init__(self, **overrides: httpx.Response):
        self.requests: list[httpx.Request] = []
        self.overrides = overrides

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.overrides:
            return self.overrides[path]
        if path == "/containers/create":
            return httpx.Response(201, json={"Id": "c0ffee"})
        if path.endswith("/start"):
            return httpx.Response(204)
        if path.endswith("/json"):
            return httpx.Response(
                200,
                json={
                    "NetworkSettings": {
                        "Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "49154"}]}
                    }
                },
            )
        if path.endswith("/stop"):
            return httpx.Response(204)
        return httpx.Response(204)  # DELETE /containers/{id}

    def body(self, path: str) -> dict:
        import json

        for request in self.requests:
            if request.url.path == path:
                return json.loads(request.content)
        raise AssertionError(f"never called {path}")


def _backend(daemon: FakeDaemon, **kwargs) -> DockerBackend:
    return DockerBackend(
        api=httpx.AsyncClient(transport=httpx.MockTransport(daemon.handle), base_url="http://d"),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
        **kwargs,
    )


async def test_start_creates_starts_and_reports_the_published_port():
    daemon = FakeDaemon()
    worker = await _backend(daemon).start("cpu-4x", "worker:latest", {"TOKEN": "abc"})

    assert worker.backend_id == "c0ffee"
    assert worker.endpoint == "http://127.0.0.1:49154"
    assert [r.url.path for r in daemon.requests] == [
        "/containers/create",
        "/containers/c0ffee/start",
        "/containers/c0ffee/json",
    ]


async def test_the_worker_port_is_published_on_loopback_only():
    """A worker on 0.0.0.0 is reachable by anything sharing the host."""
    daemon = FakeDaemon()
    await _backend(daemon).start("cpu-4x", "worker:latest")

    host_config = daemon.body("/containers/create")["HostConfig"]
    assert host_config["PortBindings"]["8080/tcp"] == [{"HostIp": "127.0.0.1", "HostPort": ""}]


async def test_environment_and_labels_reach_the_container():
    daemon = FakeDaemon()
    await _backend(daemon).start("gpu-a100", "worker:latest", {"A": "1", "B": "2"})

    created = daemon.body("/containers/create")
    assert created["Env"] == ["A=1", "B=2"]
    assert created["Labels"]["strata.pool.machine-type"] == "gpu-a100"


async def test_a_custom_worker_port_is_the_one_requested_and_read_back():
    daemon = FakeDaemon(
        **{
            "/containers/c0ffee/json": httpx.Response(
                200,
                json={"NetworkSettings": {"Ports": {"9000/tcp": [{"HostPort": "33001"}]}}},
            )
        }
    )
    worker = await _backend(daemon, worker_port=9000).start("cpu-4x", "worker:latest")

    assert daemon.body("/containers/create")["ExposedPorts"] == {"9000/tcp": {}}
    assert worker.endpoint == "http://127.0.0.1:33001"


async def test_an_image_that_publishes_nothing_says_so():
    daemon = FakeDaemon(
        **{"/containers/c0ffee/json": httpx.Response(200, json={"NetworkSettings": {"Ports": {}}})}
    )
    with pytest.raises(DockerError, match="published no host port"):
        await _backend(daemon).start("cpu-4x", "worker:latest")


async def test_a_missing_image_surfaces_the_daemons_own_message():
    daemon = FakeDaemon(
        **{"/containers/create": httpx.Response(404, json={"message": "No such image: nope"})}
    )
    with pytest.raises(DockerError, match="No such image: nope"):
        await _backend(daemon).start("cpu-4x", "nope")


async def test_a_container_that_will_not_start_is_removed_rather_than_left_behind():
    daemon = FakeDaemon(
        **{"/containers/c0ffee/start": httpx.Response(500, json={"message": "no such device"})}
    )
    with pytest.raises(DockerError, match="no such device"):
        await _backend(daemon).start("cpu-4x", "worker:latest")

    assert "/containers/c0ffee/stop" in [r.url.path for r in daemon.requests]
    assert ("DELETE", "/containers/c0ffee") in [(r.method, r.url.path) for r in daemon.requests]


async def test_stop_removes_the_container_too():
    daemon = FakeDaemon()
    await _backend(daemon).stop("c0ffee")

    assert [(r.method, r.url.path) for r in daemon.requests] == [
        ("POST", "/containers/c0ffee/stop"),
        ("DELETE", "/containers/c0ffee"),
    ]


async def test_stopping_a_container_that_is_already_gone_is_not_an_error():
    daemon = FakeDaemon(
        **{
            "/containers/c0ffee/stop": httpx.Response(404, json={"message": "No such container"}),
            "/containers/c0ffee": httpx.Response(404, json={"message": "No such container"}),
        }
    )
    await _backend(daemon).stop("c0ffee")  # idempotent, per the Backend protocol


async def test_a_container_that_cannot_be_removed_is_loud():
    """Left unreported this is a machine that keeps costing money."""
    daemon = FakeDaemon(
        **{"/containers/c0ffee": httpx.Response(409, json={"message": "removal in progress"})}
    )
    with pytest.raises(DockerError, match="removal in progress"):
        await _backend(daemon).stop("c0ffee")


async def test_health_is_false_while_the_container_refuses_connections():
    """A booting container refuses connections, and the pool polls in a loop."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    backend = DockerBackend(
        api=httpx.AsyncClient(transport=httpx.MockTransport(FakeDaemon().handle)),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(refuse)),
    )
    assert await backend.health("http://127.0.0.1:49154") is False


async def test_health_asks_the_worker_not_the_daemon():
    seen: list[str] = []

    def probe(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200)

    backend = DockerBackend(
        api=httpx.AsyncClient(transport=httpx.MockTransport(FakeDaemon().handle)),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(probe)),
    )
    assert await backend.health("http://127.0.0.1:49154") is True
    assert seen == ["http://127.0.0.1:49154/health"]


async def test_health_is_false_when_the_worker_answers_badly():
    backend = DockerBackend(
        api=httpx.AsyncClient(transport=httpx.MockTransport(FakeDaemon().handle)),
        probe=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))),
    )
    assert await backend.health("http://127.0.0.1:49154") is False

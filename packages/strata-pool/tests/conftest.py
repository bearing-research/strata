"""Fakes for the pool's two outside edges: the backend and the workers.

Everything the pool touches beyond its own SQLite file is one of these, so
the tests exercise the real dispatch code with no containers and no sockets.
"""

from collections.abc import Awaitable, Callable

import httpx
import pytest
from strata_pool import MachineType, Pool, PoolStore
from strata_pool.backend import ProvisionedWorker


class FakeBackend:
    """Hands out endpoints without provisioning anything."""

    name = "fake"

    def __init__(
        self,
        *,
        healthy_after_polls: int = 0,
        never_healthy: bool = False,
        fail_start: bool = False,
        id_prefix: str = "machine",
    ):
        self.id_prefix = id_prefix
        """Distinguishes the machines of two backends sharing one database —
        a real provider never reissues an ID a live machine already holds."""

        self.healthy_after_polls = healthy_after_polls
        self.never_healthy = never_healthy
        self.fail_start = fail_start
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.dead: set[str] = set()
        """Endpoints that stop answering, simulating a machine that vanished."""

        self._polls: dict[str, int] = {}
        self._counter = 0

    async def start(
        self,
        spec: MachineType,
        env: dict[str, str] | None = None,
    ) -> ProvisionedWorker:
        if self.fail_start:
            raise RuntimeError("no capacity")
        self._counter += 1
        backend_id = f"{self.id_prefix}-{self._counter}"
        self.started.append(backend_id)
        return ProvisionedWorker(
            backend_id=backend_id,
            endpoint=f"http://{backend_id}.test",
            region="local",
        )

    async def stop(self, backend_id: str) -> None:
        self.stopped.append(backend_id)

    async def health(self, endpoint: str) -> bool:
        if endpoint in self.dead or self.never_healthy:
            return False
        self._polls[endpoint] = self._polls.get(endpoint, 0) + 1
        return self._polls[endpoint] > self.healthy_after_polls


Responder = Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]


class FakeWorkers:
    """The HTTP side of a worker: what `POST /execute` returns."""

    def __init__(self, responder: Responder | None = None):
        self.responder = responder if responder is not None else self._echo
        self.requests: list[httpx.Request] = []

    @staticmethod
    def _echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"done:" + request.content)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self.responder(request)
        if isinstance(result, httpx.Response):
            return result
        return await result


@pytest.fixture
async def make_pool(tmp_path):
    """Build pools whose background tasks are cancelled at teardown."""
    built: list[tuple[Pool, PoolStore, httpx.AsyncClient]] = []

    def _make(
        backend: FakeBackend | None = None,
        workers: FakeWorkers | None = None,
        machine_types: list[MachineType] | None = None,
        db_name: str = "pool.sqlite",
        **kwargs,
    ) -> Pool:
        store = PoolStore(tmp_path / db_name)
        client = httpx.AsyncClient(transport=httpx.MockTransport((workers or FakeWorkers()).handle))
        pool = Pool(
            store,
            backend or FakeBackend(),
            machine_types or [MachineType(name="cpu", image="worker:latest")],
            client=client,
            health_poll_seconds=0,
            **kwargs,
        )
        built.append((pool, store, client))
        return pool

    yield _make

    for pool, store, client in built:
        await pool.aclose()
        await client.aclose()
        store.close()

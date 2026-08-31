"""Run workers as containers on a local Docker daemon.

The first backend, chosen first so that dispatch, boot handling, and the
metering path are all exercised by CI before any of them produces a number
someone pays for.

It talks to the Docker Engine API over its UNIX socket with httpx rather than
pulling in the Docker SDK: the pool's dependency list stays at one entry, and
every request shape is testable against `httpx.MockTransport` without a
daemon.

The backend does not care what runs inside the container. It starts an image,
finds the host port Docker published, and reports the endpoint; whether that
image is `strata-worker` or something else is the caller's business.
"""

import logging

import httpx

from strata_pool.backend import ProvisionedWorker

logger = logging.getLogger(__name__)

DEFAULT_SOCKET = "/var/run/docker.sock"


class DockerError(RuntimeError):
    """The Docker daemon refused a request."""


class DockerBackend:
    """Provisions workers as containers on the local Docker daemon."""

    name = "docker"

    def __init__(
        self,
        *,
        socket_path: str = DEFAULT_SOCKET,
        worker_port: int = 8080,
        command: list[str] | None = None,
        stop_timeout_seconds: int = 5,
        api: httpx.AsyncClient | None = None,
        probe: httpx.AsyncClient | None = None,
    ):
        """
        Args:
            worker_port: The port the image listens on inside the container.
                Docker publishes it on an arbitrary host port, which is what
                the pool connects to.
            command: Overrides the image's entrypoint. For images that host
                more than one, and for tests.
            api: Client for the Docker Engine API. Defaults to one bound to
                the daemon socket.
            probe: Client for worker health checks, which go over TCP to the
                published port rather than through the daemon.
        """
        self.worker_port = worker_port
        self.command = command
        self.stop_timeout_seconds = stop_timeout_seconds
        self._owns_clients = api is None and probe is None
        self._api = api if api is not None else _socket_client(socket_path)
        self._probe = probe if probe is not None else httpx.AsyncClient()

    async def aclose(self) -> None:
        """Release the HTTP clients. The pool does not own the backend."""
        if self._owns_clients:
            await self._api.aclose()
            await self._probe.aclose()

    async def start(
        self,
        machine_type: str,
        image: str,
        env: dict[str, str] | None = None,
    ) -> ProvisionedWorker:
        port_key = f"{self.worker_port}/tcp"
        create = await self._api.post(
            "/containers/create",
            json={
                "Image": image,
                "Env": [f"{key}={value}" for key, value in (env or {}).items()],
                "Cmd": self.command,
                "ExposedPorts": {port_key: {}},
                "HostConfig": {
                    # Empty HostPort means "pick a free one". Binding to
                    # loopback keeps a worker off the network: the pool is the
                    # only thing that should be able to reach it.
                    "PortBindings": {port_key: [{"HostIp": "127.0.0.1", "HostPort": ""}]},
                },
                # A machine the pool forgot about can still be found by hand,
                # and gives a later reconcile pass something to match on.
                "Labels": {"strata.pool.machine-type": machine_type},
            },
        )
        if create.status_code >= 400:
            raise DockerError(f"could not create a {image} container: {_message(create)}")
        container_id = create.json()["Id"]

        started = await self._api.post(f"/containers/{container_id}/start")
        if started.status_code >= 400:
            # The container exists and would sit there costing disk, so take
            # it back out before reporting the failure.
            await self.stop(container_id)
            raise DockerError(f"could not start {container_id}: {_message(started)}")

        port = await self._published_port(container_id, port_key)
        return ProvisionedWorker(
            backend_id=container_id,
            endpoint=f"http://127.0.0.1:{port}",
            region="local",
            metadata={"machine_type": machine_type},
        )

    async def stop(self, backend_id: str) -> None:
        """Stop and remove a container. Idempotent."""
        stopped = await self._api.post(
            f"/containers/{backend_id}/stop",
            params={"t": self.stop_timeout_seconds},
        )
        # 304 is "already stopped" and 404 is "already gone"; both are the
        # state this method exists to reach.
        if stopped.status_code >= 400 and stopped.status_code != 404:
            logger.warning(
                "docker refused to stop a container; removing it anyway",
                extra={"container_id": backend_id, "detail": _message(stopped)},
            )

        removed = await self._api.delete(f"/containers/{backend_id}", params={"force": "true"})
        if removed.status_code >= 400 and removed.status_code != 404:
            raise DockerError(f"could not remove {backend_id}: {_message(removed)}")

    async def health(self, endpoint: str) -> bool:
        """Report whether the worker answers on its published port.

        Never raises: a refused connection is the normal state of a container
        that is still booting, and the pool polls this in a loop.
        """
        try:
            response = await self._probe.get(f"{endpoint}/health", timeout=2.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400

    async def _published_port(self, container_id: str, port_key: str) -> int:
        inspect = await self._api.get(f"/containers/{container_id}/json")
        if inspect.status_code >= 400:
            raise DockerError(f"could not inspect {container_id}: {_message(inspect)}")

        bindings = inspect.json().get("NetworkSettings", {}).get("Ports") or {}
        published = bindings.get(port_key) or []
        if not published:
            raise DockerError(
                f"{container_id} published no host port for {port_key}. The image must "
                f"listen on {self.worker_port}, or the backend needs a different worker_port."
            )
        return int(published[0]["HostPort"])


def _socket_client(socket_path: str) -> httpx.AsyncClient:
    """A client bound to the daemon socket.

    The host in the URL is ignored for a UNIX-socket transport but httpx still
    requires one, hence the placeholder.
    """
    return httpx.AsyncClient(
        transport=httpx.AsyncHTTPTransport(uds=socket_path),
        base_url="http://docker",
        timeout=30.0,
    )


def _message(response: httpx.Response) -> str:
    """The daemon's own error text, which is JSON when it is well behaved."""
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    return f"HTTP {response.status_code}: {payload.get('message', response.text[:200])}"

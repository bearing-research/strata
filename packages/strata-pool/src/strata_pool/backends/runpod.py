"""Run workers as RunPod pods, published on the public internet through RunPod's proxy.

The worker credential is the only thing between a public URL and remote code
execution. `tests/test_runpod_live.py` verifies the request shapes against a live
account, except the GPU fields and the region, which is probably always None.
"""

import logging
import uuid

import httpx

from strata_pool.backend import ProvisionedWorker
from strata_pool.types import WORKER_TOKEN_ENV, MachineType

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://rest.runpod.io/v1"
PROXY_HOST = "proxy.runpod.net"


class RunPodError(RuntimeError):
    """RunPod refused a request."""


class RunPodBackend:
    """Provisions workers as pods on RunPod."""

    name = "runpod"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        worker_port: int = 8080,
        api: httpx.AsyncClient | None = None,
        probe: httpx.AsyncClient | None = None,
    ):
        """Configure the backend.

        Args:
            base_url: RunPod REST API base; overridable in case RunPod moves it.
            worker_port: Port the image listens on; reachable only through RunPod's proxy.
        """
        self.worker_port = worker_port
        # Tracked per client: injecting one for retries or a proxy must not
        # leave the other's connection pool unreleased.
        self._owns_api = api is None
        self._owns_probe = probe is None
        # Sent per request, not baked into the client, so an injected client (retries,
        # a proxy, a test) cannot silently drop authentication.
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._api = api or httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self._probe = probe or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_api:
            await self._api.aclose()
        if self._owns_probe:
            await self._probe.aclose()

    async def start(
        self,
        spec: MachineType,
        env: dict[str, str] | None = None,
    ) -> ProvisionedWorker:
        body = _create_body(spec, env, self.worker_port)
        created = await self._api.post("/pods", json=body, headers=self._auth)
        if created.status_code >= 400:
            raise RunPodError(f"could not create a {spec.name} pod: {_message(created)}")

        # Past this line a pod exists and is billing. A failure below deletes the pool row
        # that would hold its id, so the generated name is the only handle left: every
        # error from here carries it.
        name = body["name"]

        # Read the body once and assume nothing about its shape.
        try:
            payload = created.json()
        except ValueError as exc:
            raise _orphaned(
                name, f"RunPod returned a non-JSON body for a created pod: {created.text[:200]}"
            ) from exc
        if not isinstance(payload, dict):
            raise _orphaned(name, f"RunPod returned an unexpected body shape: {created.text[:200]}")

        pod_id = payload.get("id")
        if not pod_id:
            # Fail loudly rather than build an endpoint from None.
            raise _orphaned(name, f"RunPod created a pod without an id: {created.text[:200]}")

        # `machine` is null until the pod is placed on a host, and `.get("machine", {})`
        # returns None for a null, not the default.
        machine = payload.get("machine") or {}
        region = machine.get("dataCenterId") if isinstance(machine, dict) else None

        return ProvisionedWorker(
            backend_id=pod_id,
            endpoint=proxy_url(pod_id, self.worker_port),
            region=region,
            metadata={"machine_type": spec.name},
        )

    async def stop(self, backend_id: str) -> None:
        """Terminate (not stop: a stopped pod still bills for its disk) a pod. Idempotent."""
        removed = await self._api.delete(f"/pods/{backend_id}", headers=self._auth)
        if removed.status_code >= 400 and removed.status_code != 404:
            raise RunPodError(f"could not terminate {backend_id}: {_message(removed)}")

    async def health(self, endpoint: str) -> bool:
        """Ask the worker through RunPod's proxy. Never raises.

        The proxy answers 502 while a pod boots, which can take minutes.
        """
        try:
            response = await self._probe.get(f"{endpoint}/health", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400


def proxy_url(pod_id: str, port: int) -> str:
    """Where a pod's HTTP port is reachable; derived, so polling can start before the pod runs."""
    return f"https://{pod_id}-{port}.{PROXY_HOST}"


def _create_body(spec: MachineType, env: dict[str, str] | None, worker_port: int) -> dict:
    """The pod-creation request. Every field RunPod might rename lives here."""
    body: dict[str, object] = {
        # Names are not unique, but one that identifies the pool makes an orphaned pod
        # findable in the console, our only backstop until the backend can list machines.
        "name": f"strata-{spec.name}-{uuid.uuid4().hex[:8]}",
        "imageName": spec.image,
        "ports": [f"{worker_port}/http"],
        "env": dict(env or {}),
    }
    if spec.gpu_type is not None:
        body["gpuTypeIds"] = [spec.gpu_type]
        body["gpuCount"] = spec.gpu_count
    if spec.disk_gb is not None:
        body["containerDiskInGb"] = spec.disk_gb
    # Last, so a deployment can correct any field above (even a wrong one) without a release.
    body.update(spec.provider_options)
    _restore_credential(body, env)
    return body


def _restore_credential(body: dict, env: dict[str, str] | None) -> None:
    """Put the worker credential back after a `provider_options` override, and nothing else.

    The override's `env` stands in either encoding (a dict, or a list of
    ``{key, value}`` as RunPod's GraphQL takes); anything else raises RunPodError.
    """
    token = (env or {}).get(WORKER_TOKEN_ENV)
    if token is None:
        return

    declared = body.get("env")
    if isinstance(declared, dict):
        declared[WORKER_TOKEN_ENV] = token
        return
    if isinstance(declared, list):
        body["env"] = [
            entry
            for entry in declared
            if not (isinstance(entry, dict) and entry.get("key") == WORKER_TOKEN_ENV)
        ] + [{"key": WORKER_TOKEN_ENV, "value": token}]
        return

    raise RunPodError(
        f"provider_options set `env` to a {type(declared).__name__}, which the worker "
        f"credential cannot be added to. A pod without it is an open execute endpoint "
        f"on a public URL."
    )


def _orphaned(name: str, detail: str) -> RunPodError:
    """Log and build the error for a pod that exists and bills but has no usable id.

    The pool drops the row that would have held the id, so the name is the only handle.
    """
    logger.error(
        "a created pod could not be identified and may still be billing",
        extra={"pod_name": name},
    )
    return RunPodError(f"{detail} The pod is named {name!r} and may still be running.")


def _message(response: httpx.Response) -> str:
    """Format a refusal. Must never raise: it runs on the failure path."""
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    if not isinstance(payload, dict):
        return f"HTTP {response.status_code}: {response.text[:200]}"
    detail = payload.get("error") or payload.get("message") or response.text[:200]
    return f"HTTP {response.status_code}: {detail}"

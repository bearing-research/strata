"""Run workers as RunPod pods, published on the public internet through RunPod's proxy.

The worker credential is the only thing between a public URL and remote code
execution. Speaks RunPod's REST v2; `tests/test_runpod_live.py` checks the request
shapes against a live account.
"""

import copy
import logging
import uuid

import httpx

from strata_pool.backend import ProvisionedWorker
from strata_pool.types import WORKER_HEARTBEAT_ENV, MachineType

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.runpod.io/v2"
# v2 requires exactly one of `gpu` or `cpu`; a machine type with no GPU gets this
# flavor unless `provider_options` names another.
DEFAULT_CPU_FLAVOR = "cpu3g"
PROXY_HOST = "proxy.runpod.net"
# The proxy ends a request that sends nothing for 125 s (measured 2026-10); a worker
# sending a byte this often keeps a long job's request open.
HEARTBEAT_SECONDS = 30


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

        # Null until the scheduler places the pod.
        region = payload.get("dataCenterId")

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
        "image": spec.image,
        "ports": [f"{worker_port}/http"],
        "env": dict(env or {}),
    }
    if spec.gpu_type is not None:
        body["gpu"] = {"id": spec.gpu_type, "count": spec.gpu_count}
    else:
        body["cpu"] = {"id": DEFAULT_CPU_FLAVOR, "vcpuCount": _vcpu_count(spec.cpus)}
    if spec.disk_gb is not None:
        body["disk"] = spec.disk_gb
    # Last, so a deployment can correct any field above (even a wrong one) without a release.
    # `gpu` and `cpu` merge, so adding `minCudaVersion` keeps the GPU type, and `env` merges
    # under the pool's. Copied: the shared spec must not alias what is sent.
    for key, value in copy.deepcopy(spec.provider_options).items():
        current = body.get(key)
        if key in ("gpu", "cpu") and isinstance(current, dict) and isinstance(value, dict):
            body[key] = {**current, **value}
        elif key == "env":
            body[key] = _merge_env(value, env)
        else:
            body[key] = value
    # After the override, so it can set another interval: without it a long job fails at the proxy.
    if isinstance(body.get("env"), dict):
        body["env"].setdefault(WORKER_HEARTBEAT_ENV, str(HEARTBEAT_SECONDS))
    return body


def _vcpu_count(cpus: float | None) -> int:
    """RunPod takes a power of two from 2 to 32; round up so a machine never gets less."""
    count = 2
    while count < min(cpus or 2, 32):
        count *= 2
    return count


def _merge_env(declared: object, env: dict[str, str] | None) -> dict[str, str]:
    """`provider_options`' `env` under the pool's: the type's, the tenant's and the credential.

    v2 takes `env` only as a map; anything else raises RunPodError.
    """
    if not isinstance(declared, dict):
        raise RunPodError(
            f"provider_options set `env` to a {type(declared).__name__}, which the worker "
            f"credential cannot be added to. A pod without it is an open execute endpoint "
            f"on a public URL."
        )
    return {**declared, **(env or {})}


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
    detail = payload.get("detail") or response.text[:200]
    # A 422 names each rejected field here, which `detail` only summarizes.
    errors = payload.get("errors")
    if isinstance(errors, list) and errors:
        detail = f"{detail} ({'; '.join(str(e) for e in errors)})"
    return f"HTTP {response.status_code}: {detail}"

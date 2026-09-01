"""Run workers as RunPod pods.

The first backend that rents real hardware, and the first whose machines are
reachable from the public internet: a pod's HTTP port is published through
RunPod's proxy at `https://{pod_id}-{port}.proxy.runpod.net`. The worker
credential from the pool is doing real work here, not defence in depth — an
unauthenticated `/execute` on a public URL is a remote code execution endpoint
anyone can find.

**The request shapes below are written from RunPod's documented API and have
not been run against a live account.** They are deliberately confined to
`_create_body` and the three call sites in `start`/`stop`, and every one is
asserted by a test, so correcting them is a small, visible edit rather than an
excavation. Anything here may be wrong until someone with an account runs
`STRATA_POOL_RUNPOD_LIVE=1 pytest tests/test_runpod_live.py`; that test starts
a real pod and **costs real money**, which is why it is opt-in and never runs
in CI.
"""

import logging
import uuid

import httpx

from strata_pool.backend import ProvisionedWorker
from strata_pool.types import MachineType

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
        """
        Args:
            base_url: Overridable because RunPod has moved its API surface
                before and the shapes here are unverified. Pointing this at a
                corrected endpoint should not require touching the pool.
            worker_port: The port the image listens on. RunPod publishes it
                through its proxy; the pod itself is not directly addressable.
        """
        self.worker_port = worker_port
        self._owns_clients = api is None and probe is None
        # Sent per request rather than baked into the client, so that handing
        # in a client — for retries, a proxy, or a test — cannot silently drop
        # authentication and leave every call failing for a reason nobody
        # would look for.
        self._auth = {"Authorization": f"Bearer {api_key}"}
        self._api = api or httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self._probe = probe or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_clients:
            await self._api.aclose()
            await self._probe.aclose()

    async def start(
        self,
        spec: MachineType,
        env: dict[str, str] | None = None,
    ) -> ProvisionedWorker:
        created = await self._api.post(
            "/pods",
            json=_create_body(spec, env, self.worker_port),
            headers=self._auth,
        )
        if created.status_code >= 400:
            raise RunPodError(f"could not create a {spec.name} pod: {_message(created)}")

        pod_id = created.json().get("id")
        if not pod_id:
            # Better to fail loudly than to hand back an endpoint built from
            # None and let it surface later as an unreachable worker.
            raise RunPodError(f"RunPod created a pod without an id: {created.text[:200]}")

        return ProvisionedWorker(
            backend_id=pod_id,
            endpoint=proxy_url(pod_id, self.worker_port),
            region=created.json().get("machine", {}).get("dataCenterId"),
            metadata={"machine_type": spec.name},
        )

    async def stop(self, backend_id: str) -> None:
        """Terminate a pod. Idempotent.

        Terminate rather than stop: a stopped RunPod pod keeps its disk and
        keeps charging for it, which is not what the pool means when it says
        a machine is gone.
        """
        removed = await self._api.delete(f"/pods/{backend_id}", headers=self._auth)
        if removed.status_code >= 400 and removed.status_code != 404:
            raise RunPodError(f"could not terminate {backend_id}: {_message(removed)}")

    async def health(self, endpoint: str) -> bool:
        """Ask the worker, through RunPod's proxy.

        Never raises. The proxy answers 502 for a pod that has not started
        serving yet, which is the normal state during a boot that can take
        minutes on a large image.
        """
        try:
            response = await self._probe.get(f"{endpoint}/health", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400


def proxy_url(pod_id: str, port: int) -> str:
    """Where a pod's HTTP port is reachable.

    Derived rather than read back from the API: it is available the moment
    the pod exists, so the pool can start polling before RunPod reports the
    pod as running.
    """
    return f"https://{pod_id}-{port}.{PROXY_HOST}"


def _create_body(spec: MachineType, env: dict[str, str] | None, worker_port: int) -> dict:
    """The pod-creation request.

    Every field RunPod might rename lives here, and each one is asserted by a
    test, so a shape that turns out to be wrong is one edit and one test line.
    """
    body: dict[str, object] = {
        # RunPod names are not unique, but a name that identifies the pool
        # makes an orphaned pod findable in the console, which is the only
        # backstop we have until the backend can list its own machines.
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
    # Last, so a deployment can correct anything above it without waiting for
    # a release — including a field this function got wrong.
    body.update(spec.provider_options)
    return body


def _message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    detail = payload.get("error") or payload.get("message") or response.text[:200]
    return f"HTTP {response.status_code}: {detail}"

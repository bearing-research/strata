"""Run workers as RunPod pods.

The first backend that rents real hardware, and the first whose machines are
reachable from the public internet: a pod's HTTP port is published through
RunPod's proxy at `https://{pod_id}-{port}.proxy.runpod.net`. The worker
credential from the pool is doing real work here, not defence in depth — an
unauthenticated `/execute` on a public URL is a remote code execution endpoint
anyone can find.

**Verified against a live account** on 2026-09-02 by
`STRATA_POOL_RUNPOD_LIVE=1 pytest tests/test_runpod_live.py`: the base URL,
`POST /pods` with `imageName` / `ports` / `env` / `containerDiskInGb` / `name`,
the `id` in the create response, the proxy endpoint, health through that proxy,
and `DELETE /pods/{id}` including a second delete of the same pod. A CPU pod
booted, answered, and terminated with nothing left running.

**Two things that run remains silent on**, because a CPU pod does not exercise
them: `gpuTypeIds` + `gpuCount`, and reading a region from `machine`. Every pod
in the account listing carried `machine: {}` with no `dataCenterId`, so
`region` is probably always `None` today — harmless, since it is metadata, but
do not trust it. Point `STRATA_POOL_RUNPOD_GPU` at a GPU type to close the
first gap; it costs real money, which is why the test is opt-in and never runs
in CI.

The request shapes stay confined to `_create_body` and the three call sites in
`start`/`stop`, and every one is asserted by a test, so a correction is still a
small, visible edit rather than an excavation.
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
        """
        Args:
            base_url: Overridable because RunPod has moved its API surface
                before and the shapes here are unverified. Pointing this at a
                corrected endpoint should not require touching the pool.
            worker_port: The port the image listens on. RunPod publishes it
                through its proxy; the pod itself is not directly addressable.
        """
        self.worker_port = worker_port
        # Tracked per client: injecting one for retries or a proxy must not
        # leave the other's connection pool unreleased.
        self._owns_api = api is None
        self._owns_probe = probe is None
        # Sent per request rather than baked into the client, so that handing
        # in a client — for retries, a proxy, or a test — cannot silently drop
        # authentication and leave every call failing for a reason nobody
        # would look for.
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

        # Past this line a pod exists and is billing. Every failure below is
        # an orphan: the pool catches it as a failed start and deletes the row
        # that would have held the id, so nothing can terminate it. Converting
        # the exception type does not change that — naming the pod does. The
        # generated name is the only handle left.
        name = body["name"]

        # Everything below runs with a pod already created and already
        # billing. A parse that raises here is caught upstream as a failed
        # start, which deletes the row holding the backend_id — so nothing
        # could ever terminate the pod. Read the body once, and assume
        # nothing about its shape.
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
            # Better to fail loudly than to hand back an endpoint built from
            # None and let it surface later as an unreachable worker.
            raise _orphaned(name, f"RunPod created a pod without an id: {created.text[:200]}")

        # `machine` is null until the pod is placed on a host, which is the
        # normal shape immediately after create — `.get("machine", {})` would
        # return None for it, not the default.
        machine = payload.get("machine") or {}
        region = machine.get("dataCenterId") if isinstance(machine, dict) else None

        return ProvisionedWorker(
            backend_id=pod_id,
            endpoint=proxy_url(pod_id, self.worker_port),
            region=region,
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
    _restore_credential(body, env)
    return body


def _restore_credential(body: dict, env: dict[str, str] | None) -> None:
    """Put the worker credential back after an override, and nothing else.

    `provider_options` overrides everything on purpose: these request shapes
    were written from documentation and never run live, and `env` is the field
    most likely to be wrong — RunPod's GraphQL surface takes a list of
    ``{key, value}``. Refusing that form would close the escape hatch on
    exactly the field it exists for.

    So the override stands, in either encoding, and only the credential is
    restored. A pod without it is an open execute endpoint on a public URL.
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
    """An error raised with a pod already created and already billing.

    The pool will delete the row that would have held the id, so the name is
    the only way anyone finds this again. Log it as well as raising it.
    """
    logger.error(
        "a created pod could not be identified and may still be billing",
        extra={"pod_name": name},
    )
    return RunPodError(f"{detail} The pod is named {name!r} and may still be running.")


def _message(response: httpx.Response) -> str:
    """Format a refusal. Must never raise.

    This runs on the failure path, so an exception here replaces RunPod's
    actual complaint — "HTTP 401" — with an AttributeError traceback, and
    in stop() that gets swallowed as "may still be billing" while the
    operator never learns their API key is wrong.
    """
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    if not isinstance(payload, dict):
        return f"HTTP {response.status_code}: {response.text[:200]}"
    detail = payload.get("error") or payload.get("message") or response.text[:200]
    return f"HTTP {response.status_code}: {detail}"

"""Run workers as Fly Machines.

A Fly machine lives in an app's private network (6PN): it is reachable from
anything else in the same organization at `{machine_id}.vm.{app}.internal`, and
from nothing on the public internet unless the app publishes a service. So a
pool running next to its notebook servers on Fly can hand them workers that
never face the internet — which the RunPod backend, whose pods are published
through a public proxy, cannot.

The worker credential still matters: every machine in the organization's
network can reach every other, so a worker without one would run code for any
of them.

**Not yet verified against a live account.** The request shapes follow the
Machines API documentation — `POST /apps/{app}/machines` with `name`, `region`
and `config` (`image`, `env`, `guest`, `restart`, `auto_destroy`); `id`,
`region` and `state` in the response; `GET /apps/{app}/machines/{id}` for state;
`DELETE /apps/{app}/machines/{id}?force=true`. As with RunPod, every field lives
in `_create_body` and is asserted by a test, `provider_options` overrides
anything in `config`, and `base_url` can be repointed, so a wrong shape is a
small visible edit. `tests/test_fly_live.py` closes the gap when run with a
token.
"""

import logging
import re
import uuid

import httpx

from strata_pool.backend import ProvisionedWorker
from strata_pool.types import WORKER_TOKEN_ENV, MachineType

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.machines.dev/v1"

# A machine is serving only once Fly has it started; `created` and `starting`
# are the normal states of a boot, and anything else means it is not coming.
_READY_STATE = "started"

_ENDPOINT = re.compile(
    r"^http://(?P<machine>[A-Za-z0-9]+)\.vm\.(?P<app>[A-Za-z0-9-]+)\.internal:\d+$"
)


class FlyError(RuntimeError):
    """Fly refused a request."""


class FlyBackend:
    """Provisions workers as machines in one Fly app and region."""

    name = "fly"

    def __init__(
        self,
        api_token: str,
        *,
        app: str,
        region: str,
        base_url: str = DEFAULT_BASE_URL,
        worker_port: int = 8080,
        api: httpx.AsyncClient | None = None,
        probe: httpx.AsyncClient | None = None,
    ):
        """
        Args:
            app: The Fly app machines are created in. Its private network is
                the one the workers are reachable on, so it belongs to the
                same organization as whatever dispatches to them.
            region: Where machines are placed, e.g. ``"sjc"``. Next to the
                notebook servers, so a job's inputs do not cross regions.
            worker_port: The port the image listens on inside the machine.
        """
        self.app = app
        self.region = region
        self.worker_port = worker_port
        self._owns_api = api is None
        self._owns_probe = probe is None
        # Per request rather than on the client, for the same reason as the
        # RunPod backend: an injected client must not silently drop auth.
        self._auth = {"Authorization": f"Bearer {api_token}"}
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
        body = _create_body(spec, env, self.region)
        created = await self._api.post(f"/apps/{self.app}/machines", json=body, headers=self._auth)
        if created.status_code >= 400:
            raise FlyError(f"could not create a {spec.name} machine: {_message(created)}")

        # Past this line a machine exists and is billing, and a failure below
        # deletes the pool row that would have held its id. The name is the
        # only handle left, so every error from here carries it.
        name = body["name"]
        try:
            payload = created.json()
        except ValueError as exc:
            raise _orphaned(name, f"Fly returned a non-JSON body: {created.text[:200]}") from exc
        if not isinstance(payload, dict) or not payload.get("id"):
            raise _orphaned(name, f"Fly created a machine without an id: {created.text[:200]}")

        machine_id = str(payload["id"])
        return ProvisionedWorker(
            backend_id=machine_id,
            endpoint=private_url(machine_id, self.app, self.worker_port),
            region=str(payload.get("region") or self.region),
            metadata={"machine_type": spec.name, "app": self.app},
        )

    async def stop(self, backend_id: str) -> None:
        """Destroy a machine. Idempotent.

        Destroy rather than stop: a stopped machine keeps its root filesystem
        and its place in the app, and the pool means gone when it says gone.
        ``force`` because the machine may still be running a job the pool has
        already given up on.
        """
        removed = await self._api.delete(
            f"/apps/{self.app}/machines/{backend_id}",
            params={"force": "true"},
            headers=self._auth,
        )
        if removed.status_code >= 400 and removed.status_code != 404:
            raise FlyError(f"could not destroy {backend_id}: {_message(removed)}")

    async def health(self, endpoint: str) -> bool:
        """Ready when Fly reports the machine started and the worker answers.

        Never raises. The machine state comes first because it is cheap and
        final: a machine Fly has stopped or destroyed is not coming back, and
        its private address may already belong to another.
        """
        match = _ENDPOINT.match(endpoint)
        if match is not None:
            try:
                state = await self._api.get(
                    f"/apps/{match['app']}/machines/{match['machine']}", headers=self._auth
                )
            except httpx.HTTPError:
                return False
            if state.status_code >= 400:
                return False
            try:
                if state.json().get("state") != _READY_STATE:
                    return False
            except (ValueError, AttributeError):
                return False
        try:
            response = await self._probe.get(f"{endpoint}/health", timeout=10.0)
        except httpx.HTTPError:
            return False
        return response.status_code < 400


def private_url(machine_id: str, app: str, port: int) -> str:
    """Where a machine's port is reachable on the organization's private network.

    Derived, not read back: Fly's internal DNS answers for a machine as soon as
    it exists, so the pool can start polling before it reports started.
    """
    return f"http://{machine_id}.vm.{app}.internal:{port}"


def _create_body(spec: MachineType, env: dict[str, str] | None, region: str) -> dict:
    """The machine-creation request. Every field Fly might rename lives here."""
    guest: dict[str, object] = {}
    if spec.cpus is not None:
        guest["cpus"] = max(1, int(spec.cpus))
        guest["cpu_kind"] = "performance"
    if spec.memory_mb is not None:
        guest["memory_mb"] = spec.memory_mb
    if spec.gpu_type is not None:
        guest["gpu_kind"] = spec.gpu_type
        guest["gpus"] = spec.gpu_count

    config: dict[str, object] = {
        "image": spec.image,
        "env": dict(env or {}),
        # The pool decides when a machine goes away; Fly restarting a worker
        # that exited, or destroying one on its own, would put a machine the
        # pool believes is warm somewhere it is not.
        "restart": {"policy": "no"},
        "auto_destroy": False,
    }
    if guest:
        config["guest"] = guest
    # Last, so a deployment can correct any field above without a release.
    config.update(spec.provider_options)
    _restore_credential(config, env)

    return {
        # Unique in the app, and findable in `fly machines list` if an error
        # leaves one orphaned.
        "name": f"strata-{spec.name}-{uuid.uuid4().hex[:8]}",
        "region": region,
        "config": config,
    }


def _restore_credential(config: dict, env: dict[str, str] | None) -> None:
    """Put the worker credential back after a `provider_options` override.

    Fly's `env` is a flat string map; an override that replaced it with
    anything else cannot carry the credential, and a machine without one runs
    code for anything on the organization's network.
    """
    token = (env or {}).get(WORKER_TOKEN_ENV)
    if token is None:
        return
    declared = config.get("env")
    if not isinstance(declared, dict):
        raise FlyError(
            f"provider_options set `env` to a {type(declared).__name__}, which the worker "
            "credential cannot be added to."
        )
    declared[WORKER_TOKEN_ENV] = token


def _orphaned(name: str, detail: str) -> FlyError:
    logger.error(
        "a created machine could not be identified and may still be billing",
        extra={"machine_name": name},
    )
    return FlyError(f"{detail} The machine is named {name!r} and may still be running.")


def _message(response: httpx.Response) -> str:
    """Format a refusal. Must never raise: it runs on the failure path."""
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    if not isinstance(payload, dict):
        return f"HTTP {response.status_code}: {response.text[:200]}"
    return f"HTTP {response.status_code}: {payload.get('error') or response.text[:200]}"

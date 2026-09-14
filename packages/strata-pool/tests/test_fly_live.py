"""The Fly Machines backend against a real account.

**This starts a real machine and costs real money.** It is opt-in, never runs
in CI, and destroys what it starts in a ``finally``; a crashed interpreter can
still leave one running, so check ``fly machines list`` afterwards.

    export FLY_API_TOKEN=... STRATA_POOL_FLY_APP=... STRATA_POOL_FLY_REGION=sjc
    STRATA_POOL_FLY_LIVE=1 pytest tests/test_fly_live.py -v -s

Workers are reachable only on the organization's private network. From a
laptop that is not on it (no ``fly wireguard`` peer), this proves the API half —
create, the machine reaching ``started``, destroy, and a repeated destroy — and
skips the worker probe. Set ``STRATA_POOL_FLY_ON_NETWORK=1`` where
``*.vm.<app>.internal`` resolves to probe ``/health`` through the backend too,
which is the half item 51's "answers on the private network" needs.

Never run yet: the shapes in ``backends/fly.py`` come from the Machines API
documentation, and this is what confirms them.
"""

import asyncio
import os
import time

import pytest
from strata_pool import FlyBackend, MachineType

LIVE = os.environ.get("STRATA_POOL_FLY_LIVE") == "1"
TOKEN = os.environ.get("FLY_API_TOKEN")
APP = os.environ.get("STRATA_POOL_FLY_APP")
REGION = os.environ.get("STRATA_POOL_FLY_REGION", "sjc")
ON_NETWORK = os.environ.get("STRATA_POOL_FLY_ON_NETWORK") == "1"

pytestmark = [
    pytest.mark.fly,
    pytest.mark.skipif(
        not (LIVE and TOKEN and APP),
        reason="needs STRATA_POOL_FLY_LIVE=1, FLY_API_TOKEN and STRATA_POOL_FLY_APP "
        "(starts a billed machine)",
    ),
]

if LIVE and not (TOKEN and APP):
    raise RuntimeError("STRATA_POOL_FLY_LIVE=1 but FLY_API_TOKEN or STRATA_POOL_FLY_APP is unset")

# Serves HTTP on port 80 and answers anything, so nothing has to be built.
IMAGE = os.environ.get("STRATA_POOL_FLY_IMAGE", "traefik/whoami:latest")
WORKER_PORT = int(os.environ.get("STRATA_POOL_FLY_PORT", "80"))
BOOT_SECONDS = float(os.environ.get("STRATA_POOL_FLY_BOOT", "300"))


@pytest.fixture
async def fly():
    backend = FlyBackend(TOKEN, app=APP, region=REGION, worker_port=WORKER_PORT)
    started: list[str] = []
    original = backend.start

    async def remember(spec, env=None):
        provisioned = await original(spec, env)
        started.append(provisioned.backend_id)
        print(f"\nstarted Fly machine {provisioned.backend_id} at {provisioned.endpoint}")
        return provisioned

    backend.start = remember
    try:
        yield backend
    finally:
        stranded: list[str] = []
        try:
            for machine_id in started:
                try:
                    await backend.stop(machine_id)
                    print(f"destroyed Fly machine {machine_id}")
                except Exception as exc:
                    stranded.append(f"{machine_id}: {exc!r}")
        finally:
            await backend.aclose()
        if stranded:
            pytest.fail("Fly machines left running, destroy them by hand: " + "; ".join(stranded))


async def test_a_machine_starts_and_is_destroyed(fly):
    spec = MachineType(name="live-test", image=IMAGE, cpus=1, memory_mb=256)
    provisioned = await fly.start(spec, {"STRATA_WORKER_TOKEN": "unused-here"})

    assert provisioned.backend_id
    assert (
        provisioned.endpoint == f"http://{provisioned.backend_id}.vm.{APP}.internal:{WORKER_PORT}"
    )

    deadline = time.monotonic() + BOOT_SECONDS
    state = None
    while time.monotonic() < deadline:
        response = await fly._api.get(
            f"/apps/{APP}/machines/{provisioned.backend_id}", headers=fly._auth
        )
        state = response.json().get("state")
        if state == "started" and (not ON_NETWORK or await fly.health(provisioned.endpoint)):
            break
        await asyncio.sleep(3)
    else:
        pytest.fail(f"machine {provisioned.backend_id} never became ready (last state {state!r})")

    await fly.stop(provisioned.backend_id)
    await fly.stop(provisioned.backend_id)  # idempotent, per the protocol

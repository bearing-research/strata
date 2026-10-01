"""The RunPod backend against a real account.

**This starts a real pod and costs real money.** It is opt-in, never runs in
CI, and terminates what it starts in a `finally`; a crashed interpreter can
still leave a pod running, so check the console afterwards.

    export RUNPOD_API_KEY=...
    STRATA_POOL_RUNPOD_LIVE=1 pytest tests/test_runpod_live.py -v -s

This confirms RunPod accepts what `test_runpod_backend.py` asserts we send, and
prints the pod id for chasing failures. It uses a CPU pod; set
`STRATA_POOL_RUNPOD_GPU` to a GPU type to cover the accelerator fields too.
"""

import asyncio
import os
import time

import pytest
from strata_pool import JobState, MachineType, Pool, PoolStore, RunPodBackend

LIVE = os.environ.get("STRATA_POOL_RUNPOD_LIVE") == "1"
API_KEY = os.environ.get("RUNPOD_API_KEY")

pytestmark = [
    pytest.mark.runpod,
    pytest.mark.skipif(
        not (LIVE and API_KEY),
        reason="needs STRATA_POOL_RUNPOD_LIVE=1 and RUNPOD_API_KEY (starts a billed pod)",
    ),
]

if LIVE and not API_KEY:
    # Opting in without a key is a mistake; say so rather than skip and report green.
    raise RuntimeError("STRATA_POOL_RUNPOD_LIVE=1 but RUNPOD_API_KEY is not set")

# A published image that serves HTTP and echoes, so the test needs nothing
# built or pushed. Override once there is a real strata-worker image.
IMAGE = os.environ.get("STRATA_POOL_RUNPOD_IMAGE", "traefik/whoami:latest")
WORKER_PORT = int(os.environ.get("STRATA_POOL_RUNPOD_PORT", "80"))


def _spec() -> MachineType:
    return MachineType(
        name="live-test",
        image=IMAGE,
        max_workers=1,
        # Pulling an image onto a fresh pod is minutes, not seconds.
        boot_timeout_seconds=float(os.environ.get("STRATA_POOL_RUNPOD_BOOT", "600")),
        job_timeout_seconds=120.0,
        cool_down_seconds=0.0,
        disk_gb=10,
        gpu_type=os.environ.get("STRATA_POOL_RUNPOD_GPU"),
        provider_options={"cloudType": "SECURE"},
    )


@pytest.fixture
async def runpod():
    backend = RunPodBackend(API_KEY, worker_port=WORKER_PORT)
    started: list[str] = []
    original = backend.start

    async def remember(spec, env=None):
        provisioned = await original(spec, env)
        started.append(provisioned.backend_id)
        print(f"\nstarted RunPod pod {provisioned.backend_id} at {provisioned.endpoint}")
        return provisioned

    backend.start = remember
    try:
        yield backend
    finally:
        stranded: list[str] = []
        try:
            # One pod refusing to terminate must not strand the rest with GPUs running.
            for pod_id in started:
                try:
                    await backend.stop(pod_id)
                    print(f"terminated RunPod pod {pod_id}")
                except Exception as exc:
                    stranded.append(f"{pod_id}: {exc!r}")
        finally:
            await backend.aclose()

        # Not a print: output of a passing test is swallowed, and a run that leaves a
        # billing GPU is not a pass.
        if stranded:
            pytest.fail("RunPod pods left running, terminate them by hand: " + "; ".join(stranded))


async def test_a_pod_starts_becomes_healthy_and_terminates(runpod):
    """The backend contract against the real API: provision, proxy endpoint, health, terminate."""
    spec = _spec()
    provisioned = await runpod.start(spec, {"STRATA_WORKER_TOKEN": "unused-here"})

    assert provisioned.backend_id
    assert provisioned.endpoint.endswith(f"-{WORKER_PORT}.proxy.runpod.net")

    deadline = time.monotonic() + spec.boot_timeout_seconds
    while time.monotonic() < deadline:
        if await runpod.health(provisioned.endpoint):
            break
        await asyncio.sleep(5)
    else:
        pytest.fail(
            f"pod {provisioned.backend_id} never became healthy within "
            f"{spec.boot_timeout_seconds}s at {provisioned.endpoint}"
        )

    await runpod.stop(provisioned.backend_id)
    await runpod.stop(provisioned.backend_id)  # idempotent, per the protocol


async def test_the_pool_drives_a_real_pod_end_to_end(tmp_path, runpod):
    """Provision, boot, dispatch, meter and reap on rented hardware.

    Skipped without an image that serves `/execute`; the default one only answers health.
    """
    if "STRATA_POOL_RUNPOD_IMAGE" not in os.environ:
        pytest.skip("set STRATA_POOL_RUNPOD_IMAGE to an image that serves /execute")

    store = PoolStore(tmp_path / "pool.sqlite")
    pool = Pool(store, runpod, [_spec()], health_poll_seconds=5.0)
    try:
        job = await pool.submit(tenant_id="live", machine_type="live-test", payload=b"work")
        done = await pool.wait(job.id, timeout=900)

        assert done.state is JobState.COMPLETED
        assert store.list_usage("live")[0].duration_ms > 0

        assert await pool.reap_idle_workers() == 1
        assert store.list_workers() == []
    finally:
        await pool.aclose()
        store.close()

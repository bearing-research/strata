"""The RunPod backend against a real account.

**This starts a real pod and costs real money.** It is opt-in, it never runs
in CI, and it terminates what it starts in a `finally` — but a crashed
interpreter can still leave a pod running, so check the console afterwards.

    export RUNPOD_API_KEY=...
    STRATA_POOL_RUNPOD_LIVE=1 pytest tests/test_runpod_live.py -v -s

Everything in `test_runpod_backend.py` proves only that we send what we
intended to send — not that RunPod wants it. This test is the one that closes
that gap, and it prints the pod id so a failure can be chased in the console.

Last run 2026-09-02: passed in 25s. A CPU pod booted, answered health through
the proxy, and terminated with nothing left in the account — covering the base
URL, the create body, the `id` in the response, the proxy endpoint, and a
repeated delete. It does **not** cover the accelerator fields.

It uses a CPU pod by default because it is the cheapest thing that exercises
the same path. Point `STRATA_POOL_RUNPOD_GPU` at a GPU type to prove the
accelerator fields too, which is the part most likely to be wrong.
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
    # Opting in without a key is a mistake worth saying out loud, rather than
    # skipping and reporting green.
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
            # One pod refusing to terminate must not strand the rest. The
            # docstring promises this cleans up; a bare loop breaks that
            # promise on the first RunPodError and leaves GPUs running.
            for pod_id in started:
                try:
                    await backend.stop(pod_id)
                    print(f"terminated RunPod pod {pod_id}")
                except Exception as exc:
                    stranded.append(f"{pod_id}: {exc!r}")
        finally:
            await backend.aclose()

        # Not a print. Teardown output is swallowed under default capture when
        # the test passes, so a green run would be hiding a billing GPU — and
        # a run that leaves one is not a passing run.
        if stranded:
            pytest.fail("RunPod pods left running, terminate them by hand: " + "; ".join(stranded))


async def test_a_pod_starts_becomes_healthy_and_terminates(runpod):
    """The whole backend contract, against the real API.

    Deliberately not a full job run: this proves provisioning, the proxy
    endpoint, health polling, and termination, which is everything the pool
    depends on. Running a job needs a worker image that speaks the contract.
    """
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
    """Provision, boot, dispatch, meter, reap — on rented hardware.

    Skipped unless an image that speaks the worker contract is given, since
    the default image answers health but not `/execute`.
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

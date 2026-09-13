"""A worker that cannot be overcommitted, and keeps concurrent cells off each
other's GPU. Item 40.

``strata-worker`` spawned a harness per request with no limit, and pinning a
cell to a GPU was whatever ``CUDA_VISIBLE_DEVICES`` the caller sent. Both were
the dispatcher's to get right, so a caller that skipped the dispatcher could
put every member's cells on one GPU.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from strata.notebook import remote_executor
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    create_notebook_executor_app,
)


class _BlockingHarness:
    """Stands in for the harness: records what each cell was given, and holds
    it running until released, so concurrency is real rather than a race."""

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.started: list[dict[str, Any]] = []
        self.all_started = asyncio.Event()
        self.expected = 0

    async def __call__(self, _harness_path, manifest_path: Path, _timeout, **kwargs):
        manifest = json.loads(manifest_path.read_text())
        env = kwargs.get("env") or {}
        self.started.append(
            {
                "manifest_cuda": manifest["env"].get("CUDA_VISIBLE_DEVICES"),
                "process_cuda": env.get("CUDA_VISIBLE_DEVICES"),
            }
        )
        if len(self.started) >= self.expected:
            self.all_started.set()
        await self.release.wait()
        return {"success": True, "variables": {}, "stdout": "", "stderr": ""}


def _post(client: httpx.AsyncClient, env: dict[str, str] | None = None):
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 1",
        "timeout_seconds": 30,
        "inputs": {},
        "mounts": [],
        "env": env or {},
    }
    return client.post(
        "/v1/notebook-execute",
        files={"metadata": ("metadata.json", json.dumps(metadata), "application/json")},
    )


@pytest.fixture
def harness(monkeypatch):
    fake = _BlockingHarness()
    monkeypatch.setattr(remote_executor, "_run_harness", fake)
    # Packing a bundle from a harness that wrote nothing is not what is under test.
    monkeypatch.setattr(remote_executor, "pack_notebook_output_bundle", _write_empty_bundle)
    return fake


def _write_empty_bundle(bundle_path: Path, _result, _output_dir) -> None:
    bundle_path.write_bytes(b"")


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker")


async def test_one_past_the_limit_is_refused_with_retry_after(harness):
    app = create_notebook_executor_app(max_concurrent=2)
    harness.expected = 2
    async with _client(app) as client:
        running = [asyncio.create_task(_post(client)) for _ in range(2)]
        await asyncio.wait_for(harness.all_started.wait(), timeout=10)

        # Bounded: a worker that admits it would hold it until release, and the
        # test should fail rather than hang.
        third = await asyncio.wait_for(_post(client), timeout=10)
        health = (await client.get("/health")).json()

        harness.release.set()
        finished = await asyncio.gather(*running)

    assert third.status_code == 503
    assert int(third.headers["Retry-After"]) > 0
    assert (health["max_concurrent"], health["active_executions"]) == (2, 2)
    assert [r.status_code for r in finished] == [200, 200]


async def test_a_finished_execution_frees_its_slot_and_its_gpu(harness):
    app = create_notebook_executor_app(max_concurrent=1, gpu_slots=1)
    harness.expected = 1
    harness.release.set()
    async with _client(app) as client:
        first = await _post(client)
        second = await _post(client)
        health = (await client.get("/health")).json()

    assert (first.status_code, second.status_code) == (200, 200)
    assert health["active_executions"] == 0
    assert health["free_gpu_slots"] == 1


async def test_concurrent_cells_get_different_gpus_whatever_they_asked_for(harness):
    """Both callers ask for GPU 0; the worker decides."""
    app = create_notebook_executor_app(gpu_slots=2)
    harness.expected = 2
    async with _client(app) as client:
        running = [
            asyncio.create_task(_post(client, env={"CUDA_VISIBLE_DEVICES": "0"})) for _ in range(2)
        ]
        await asyncio.wait_for(harness.all_started.wait(), timeout=10)
        no_gpu_left = await asyncio.wait_for(_post(client), timeout=10)
        health = (await client.get("/health")).json()
        harness.release.set()
        await asyncio.gather(*running)

    manifest = sorted(s["manifest_cuda"] for s in harness.started)
    process = sorted(s["process_cuda"] for s in harness.started)
    assert manifest == ["0", "1"]
    assert process == ["0", "1"]
    assert no_gpu_left.status_code == 503
    assert (health["gpu_slots"], health["free_gpu_slots"]) == (2, 0)


async def test_unset_is_unlimited_as_before(harness):
    app = create_notebook_executor_app()
    harness.expected = 3
    async with _client(app) as client:
        running = [asyncio.create_task(_post(client)) for _ in range(3)]
        await asyncio.wait_for(harness.all_started.wait(), timeout=10)
        health = (await client.get("/health")).json()
        harness.release.set()
        finished = await asyncio.gather(*running)

    assert [r.status_code for r in finished] == [200, 200, 200]
    assert health["max_concurrent"] is None
    assert all(s["process_cuda"] is None for s in harness.started)


def test_the_limits_can_come_from_the_environment(monkeypatch):
    """``uvicorn --factory`` deployments cannot pass arguments."""
    monkeypatch.setenv("STRATA_WORKER_MAX_CONCURRENT", "3")
    monkeypatch.setenv("STRATA_WORKER_GPU_SLOTS", "4")

    async def _health():
        async with _client(create_notebook_executor_app()) as client:
            return (await client.get("/health")).json()

    health = asyncio.run(_health())

    assert (health["max_concurrent"], health["gpu_slots"]) == (3, 4)


def test_a_zero_limit_is_refused_at_startup():
    with pytest.raises(SystemExit):
        remote_executor.main(["--max-concurrent", "0"])


async def test_a_real_cell_sees_the_gpu_the_worker_chose(tmp_path):
    """Through the real harness: the pinning is what the cell's process sees,
    not only what the manifest says."""
    from strata.notebook.remote_bundle import unpack_notebook_output_bundle

    app = create_notebook_executor_app(gpu_slots=1)
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "import os\ngpu = os.environ['CUDA_VISIBLE_DEVICES']\nprint('GPU=' + gpu)",
        "timeout_seconds": 60,
        "inputs": {},
        "mounts": [],
        "env": {"CUDA_VISIBLE_DEVICES": "7"},
    }
    async with _client(app) as client:
        response = await client.post(
            "/v1/notebook-execute",
            files={"metadata": ("metadata.json", json.dumps(metadata), "application/json")},
            timeout=60,
        )

    assert response.status_code == 200, response.text
    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(response.content)
    result = unpack_notebook_output_bundle(bundle, tmp_path / "out")
    assert result["success"], result
    assert "GPU=0" in result["stdout"]

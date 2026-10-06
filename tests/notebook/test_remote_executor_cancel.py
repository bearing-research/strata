"""Cancelling a remote cell stops the worker, instead of leaving it computing.

The cancelled result could never land anyway; the cost was the machine (a slot,
or a paid GPU) running the cell for a caller that had gone.
"""

from __future__ import annotations

import asyncio
import sys

import pytest
from fastapi.testclient import TestClient

from strata.notebook.remote_executor import _run_harness, create_notebook_executor_app


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    return TestClient(create_notebook_executor_app())


def test_health_advertises_cancel(worker):
    """A server only calls cancel on a worker that says it has it."""
    features = worker.get("/health").json()["capabilities"]["features"]
    assert features["cancel"] is True


def test_cancelling_an_unknown_execution_is_not_an_error(worker):
    """Cancelling an execution that already finished is a normal answer.

    A caller reading it as failure would retire a healthy, idle machine.
    """
    response = worker.post("/v1/executions/nonexistent/cancel")

    assert response.status_code == 200
    assert response.json() == {"build_id": "nonexistent", "cancelled": False}


def test_cancel_requires_the_worker_token_when_one_is_set(monkeypatch):
    """Anyone who can reach the worker could otherwise kill its work."""
    monkeypatch.setenv("STRATA_WORKER_TOKEN", "sekret")
    client = TestClient(create_notebook_executor_app())

    assert client.post("/v1/executions/abc/cancel").status_code == 401
    assert (
        client.post("/v1/executions/abc/cancel", headers={"Authorization": "Bearer sekret"})
    ).status_code == 200


@pytest.mark.asyncio
async def test_a_registered_harness_is_killed_and_deregistered(tmp_path):
    """The process must actually die, not just the response report it."""
    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(300)\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    in_flight: dict[str, object] = {}
    run = asyncio.create_task(
        _run_harness(script, manifest, 300.0, in_flight=in_flight, build_id="b1")
    )

    # Wait for registration rather than sleeping a fixed amount.
    for _ in range(100):
        if "b1" in in_flight:
            break
        await asyncio.sleep(0.01)
    assert "b1" in in_flight, "the harness never registered itself"
    proc = in_flight["b1"]

    from strata.notebook.process_tree import terminate_subprocess_tree

    await terminate_subprocess_tree(proc)

    with pytest.raises(Exception):
        # No harness-result.json, because it was killed rather than finishing.
        await asyncio.wait_for(run, timeout=30)

    assert proc.returncode is not None, "the process is still running"
    assert "b1" not in in_flight, "a finished execution must not stay cancellable"


@pytest.mark.asyncio
async def test_an_unidentified_execution_is_simply_not_cancellable(tmp_path):
    """No build id means not cancellable, which is not an error."""
    script = tmp_path / "quick.py"
    script.write_text("import json,sys,pathlib\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    in_flight: dict[str, object] = {}
    with pytest.raises(Exception):
        await _run_harness(script, manifest, 30.0, in_flight=in_flight, build_id=None)

    assert in_flight == {}


@pytest.mark.asyncio
async def test_the_route_kills_a_running_execution(tmp_path):
    """End to end: POST cancel, and the harness process is actually dead.

    An in-process ASGI client, not TestClient, keeps the subprocess and the route
    on one event loop; an asyncio subprocess is bound to its creating loop.
    """
    import httpx

    from strata.notebook.process_tree import subprocess_kwargs_for_new_group

    app = create_notebook_executor_app()

    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(300)\n")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **subprocess_kwargs_for_new_group(),
    )
    app.state.in_flight["b1"] = proc

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.post("/v1/executions/b1/cancel")

    assert response.status_code == 200
    assert response.json() == {"build_id": "b1", "cancelled": True}

    await asyncio.wait_for(proc.wait(), timeout=30)
    assert proc.returncode is not None, "the harness process survived the cancel"


@pytest.mark.asyncio
async def test_a_cancel_while_the_environment_builds_stops_the_run(monkeypatch):
    """Inputs and the locked env come before the harness; a cancel then must still stop it."""
    import json
    from types import SimpleNamespace

    import httpx

    from strata.notebook import remote_executor, worker_env

    building = asyncio.Event()
    release = asyncio.Event()

    async def _slow_environment(_spec):
        building.set()
        await release.wait()
        return SimpleNamespace(python=None, key="k", installed=False)

    spawned: list[str | None] = []

    async def _spy_run_harness(*_args, build_id=None, **_kwargs):
        spawned.append(build_id)
        return {"success": True, "variables": {}}

    monkeypatch.setattr(worker_env, "ensure_environment", _slow_environment)
    monkeypatch.setattr(remote_executor, "_run_harness", _spy_run_harness)
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    app = create_notebook_executor_app()
    metadata = {
        "protocol_version": remote_executor.NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 1",
        "build_id": "b1",
        "environment": {"lock": "pinned"},
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        run = asyncio.create_task(
            client.post(
                "/v1/notebook-execute",
                files={"metadata": ("metadata.json", json.dumps(metadata), "application/json")},
            )
        )
        await asyncio.wait_for(building.wait(), timeout=30)
        cancel = await client.post("/v1/executions/b1/cancel")
        release.set()
        response = await asyncio.wait_for(run, timeout=30)

    assert cancel.json() == {"build_id": "b1", "cancelled": True}
    assert response.status_code == 409
    assert spawned == [], "the harness ran after the cancel"
    assert "b1" not in app.state.in_flight


@pytest.mark.asyncio
async def test_a_harness_cancelled_while_it_starts_is_killed(tmp_path):
    """The cancel can land between the pre-spawn check and the process registering."""
    from strata.notebook.remote_executor import _PendingRun

    script = tmp_path / "sleeper.py"
    script.write_text("import time\ntime.sleep(300)\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    pending = _PendingRun()
    pending.cancelled = True
    in_flight: dict[str, object] = {"b1": pending}

    with pytest.raises(RuntimeError, match="harness-result.json"):
        await asyncio.wait_for(
            _run_harness(script, manifest, 300.0, in_flight=in_flight, build_id="b1"),
            timeout=30,
        )
    assert in_flight == {}


class TestCancelUrlMapping:
    """The cancel URL lands on the worker the dispatch went to.

    Executor URLs come as a bare base, an endpoint, or a facade's path prefix; a
    cancel to the wrong path is a silent 404 while the machine keeps computing.
    """

    @pytest.mark.parametrize(
        "executor_url",
        [
            "http://worker:9000",
            "http://worker:9000/",
            "http://worker:9000/v1/execute",
            "http://worker:9000/v1/notebook-execute",
            "http://worker:9000/v1/execute-manifest",
        ],
    )
    def test_every_endpoint_shape_maps_to_the_same_cancel_url(self, executor_url):
        from strata.notebook.executor import CellExecutor

        assert (
            CellExecutor._cancel_url(None, executor_url, "b1")
            == "http://worker:9000/v1/executions/b1/cancel"
        )

    def test_a_path_prefix_is_preserved(self):
        """A pool facade fronts many workers under one host."""
        from strata.notebook.executor import CellExecutor

        assert (
            CellExecutor._cancel_url(None, "http://pool/machines/m7/v1/execute", "b1")
            == "http://pool/machines/m7/v1/executions/b1/cancel"
        )

    def test_query_and_fragment_are_dropped(self):
        from strata.notebook.executor import CellExecutor

        assert (
            CellExecutor._cancel_url(None, "http://worker:9000/v1/execute?x=1#f", "b1")
            == "http://worker:9000/v1/executions/b1/cancel"
        )

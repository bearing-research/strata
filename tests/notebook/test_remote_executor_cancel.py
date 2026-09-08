"""Cancelling a remote cell stops the worker, instead of leaving it computing.

The server marks the build failed and both ``finalize`` and the upload route
refuse anything outside the active states, so a cancelled cell's result could
never land. What was lost was the machine: it ran the cell to completion for a
caller that had already gone, holding a slot — and, on a GPU box, the hardware
someone is paying for.
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
    """ "Already gone" is a normal answer.

    The execution may have finished between the server deciding to cancel and
    the request landing. A caller that read that as a failure would retire a
    machine that is healthy and idle.
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
    """The mechanism, not the response body: the process must actually die."""
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
    """No build id is the pre-existing behaviour, not an error."""
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

    Driven with an in-process ASGI client rather than TestClient so the
    subprocess and the route that terminates it share one event loop — an
    asyncio subprocess is bound to the loop that created it.
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


class TestCancelUrlMapping:
    """The cancel URL has to land on the same worker the dispatch went to.

    Executor URLs arrive in several shapes — bare base, a specific endpoint,
    or a path prefix when a facade fronts a pool — and a cancel posted to the
    wrong path is silently a no-op: the worker answers 404, the machine keeps
    computing, and nothing says so.
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

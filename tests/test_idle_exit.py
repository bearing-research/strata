"""The server stops itself once nothing has used it for ``idle_exit_minutes``."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

from strata import server
from strata.notebook import ws


@pytest.fixture
def activity(monkeypatch):
    """A fresh request and socket count, unshared with other tests."""
    fresh = server._Activity()
    monkeypatch.setattr(server, "_activity", fresh)
    return fresh


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["http", "websocket"])
async def test_a_request_or_socket_counts_until_it_ends(activity, kind):
    started, release = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        started.set()
        # The headers went out and the body is still streaming, or the tab is still open.
        await release.wait()

    async def send(message):
        pass

    async def receive():
        return {"type": "http.request"}

    task = asyncio.create_task(server.ActivityMiddleware(app)({"type": kind}, receive, send))
    await started.wait()
    assert activity.idle_seconds(time.monotonic() + 3600) == 0.0
    release.set()
    await task
    # From when it ended (approx: subtracting the clock reading back rounds).
    assert activity.idle_seconds(activity.last_ended + 5) == pytest.approx(5.0)


def test_a_run_counts_from_its_reservation(monkeypatch):
    state = ws.NotebookExecutionState()
    monkeypatch.setattr(ws, "_notebook_execution_state", {"nb": state})
    assert not ws.runs_in_flight()

    # Reserved, its task not yet made.
    state.requested_cell = "c1"
    assert ws.runs_in_flight()

    async def running() -> None:
        state.requested_cell = None
        state.execution_task = asyncio.create_task(asyncio.sleep(10))
        assert ws.runs_in_flight()
        state.execution_task.cancel()

    asyncio.run(running())


def test_builds_and_warm_jobs_are_work_in_flight(monkeypatch):
    from types import SimpleNamespace

    from strata.transforms import runner

    def busy(value):
        return SimpleNamespace(busy=lambda: value)

    monkeypatch.setattr(ws, "_notebook_execution_state", {})
    idle = SimpleNamespace(streams=busy(False), _cache_warmer=None)
    monkeypatch.setattr(runner, "get_build_runner", lambda: busy(False))
    assert not server._work_in_flight(idle)
    assert not server._work_in_flight(None)

    monkeypatch.setattr(runner, "get_build_runner", lambda: busy(True))
    assert server._work_in_flight(idle)
    monkeypatch.setattr(runner, "get_build_runner", lambda: None)
    assert server._work_in_flight(SimpleNamespace(streams=busy(True), _cache_warmer=None))
    assert server._work_in_flight(SimpleNamespace(streams=busy(False), _cache_warmer=busy(True)))


@pytest.mark.asyncio
async def test_the_loop_stops_an_unused_server_and_not_a_busy_one(activity, monkeypatch):
    stopped = asyncio.Event()
    monkeypatch.setattr(server, "_work_in_flight", lambda state: True)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(server._idle_exit_loop(0.05, stop=stopped.set), timeout=0.5)
    assert not stopped.is_set()

    monkeypatch.setattr(server, "_work_in_flight", lambda state: False)
    await asyncio.wait_for(server._idle_exit_loop(0.05, stop=stopped.set), timeout=30)
    assert stopped.is_set()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_an_idle_server_process_shuts_down_cleanly(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env.update(
        HOME=str(tmp_path / "home"),
        STRATA_HOST="127.0.0.1",
        STRATA_PORT=str(_free_port()),
        STRATA_CACHE_DIR=str(tmp_path / "cache"),
        STRATA_ARTIFACT_DIR=str(tmp_path / "artifacts"),
        STRATA_NOTEBOOK_STORAGE_DIR=str(tmp_path / "notebooks"),
        STRATA_NOTEBOOK_WARM_POOL_SIZE="0",
        # 0.6 s.
        STRATA_IDLE_EXIT_MINUTES="0.01",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "strata"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    log = proc.stdout + proc.stderr
    if sys.platform != "win32":
        # uvicorn ends on the signal it caught, as after Ctrl-C.
        assert proc.returncode == -signal.SIGTERM, log
    assert "idle_exit" in log, log
    # The lifespan's shutdown ran, as on Ctrl-C.
    assert "Application shutdown complete" in log, log


@pytest.mark.asyncio
async def test_a_background_build_or_warm_job_is_busy_until_it_ends():
    from types import SimpleNamespace

    from strata.cache.warmer import CacheWarmer, WarmingJob
    from strata.streaming.registry import StreamRegistry, StreamState
    from strata.transforms.runner import BuildRunner

    release = asyncio.Event()
    task = asyncio.create_task(release.wait())

    streams = StreamRegistry(ttl_seconds=60)
    streams.register(
        StreamState(stream_id="s", plan=None, artifact_id="a", artifact_version=1, created_at=0.0)
    )
    assert not streams.busy()
    streams.get("s").background_task = task

    warmer = CacheWarmer(planner=None, fetcher=None, metrics=None)
    warmer._jobs["j"] = WarmingJob(job_id="j", request=None, _task=task)

    runner = BuildRunner(
        config=SimpleNamespace(max_concurrent_builds=1, runner_id="r"),
        artifact_store=None,
        build_store=None,
        transform_registry=None,
        artifact_dir=None,
    )
    runner._running_builds.add("b")

    assert streams.busy() and warmer.busy() and runner.busy()
    release.set()
    await task
    runner._running_builds.clear()
    assert not streams.busy() and not warmer.busy() and not runner.busy()


@pytest.mark.asyncio
async def test_busy_is_work_besides_the_question(activity, monkeypatch):
    from strata.api.routers.metrics_health import health_busy

    monkeypatch.setattr(server, "_work_in_flight", lambda state: False)
    # The question itself is open.
    activity.begin()
    assert await health_busy() == {"busy": False}
    activity.begin()
    assert await health_busy() == {"busy": True}
    activity.end()
    monkeypatch.setattr(server, "_work_in_flight", lambda state: True)
    assert await health_busy() == {"busy": True}

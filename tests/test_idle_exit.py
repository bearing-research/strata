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
from strata.observability.pool_metrics import ConnectionMetrics


def test_a_request_in_flight_is_never_idle(monkeypatch):
    requests = ConnectionMetrics()
    clock = [time.time()]
    monkeypatch.setattr("strata.observability.pool_metrics.time.time", lambda: clock[0])

    requests.request_started()
    clock[0] += 600
    assert requests.idle_seconds() == 0.0
    requests.request_completed()
    clock[0] += 5
    # From when it ended, not when it started.
    assert requests.idle_seconds() == 5.0


def test_a_socket_or_run_resets_the_idle_clock():
    assert server._idle_for(now=100.0, last_busy=40.0, busy=True, request_idle=900.0) == 0.0
    # A socket that closed at 90 after an hour open: idle since 90, not since the last request.
    assert server._idle_for(now=100.0, last_busy=90.0, busy=False, request_idle=3600.0) == 10.0
    # A request since then counts too.
    assert server._idle_for(now=100.0, last_busy=40.0, busy=False, request_idle=2.0) == 2.0


def test_an_open_socket_or_a_run_in_flight_is_a_client(monkeypatch):
    monkeypatch.setattr(ws, "_notebook_connections", {"nb": []})
    monkeypatch.setattr(ws, "_notebook_execution_state", {"nb": ws.NotebookExecutionState()})
    assert not ws.clients_or_runs()

    monkeypatch.setattr(ws, "_notebook_connections", {"nb": [object()]})
    assert ws.clients_or_runs()

    async def running() -> None:
        state = ws.NotebookExecutionState()
        state.execution_task = asyncio.create_task(asyncio.sleep(10))
        monkeypatch.setattr(ws, "_notebook_connections", {})
        monkeypatch.setattr(ws, "_notebook_execution_state", {"nb": state})
        assert ws.clients_or_runs()
        state.execution_task.cancel()

    asyncio.run(running())


@pytest.mark.asyncio
async def test_the_loop_stops_an_unused_server(monkeypatch):
    monkeypatch.setattr(ws, "_notebook_connections", {})
    monkeypatch.setattr(ws, "_notebook_execution_state", {})
    stopped = asyncio.Event()

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
    # uvicorn ends on the signal it caught, as after Ctrl-C.
    assert proc.returncode == -signal.SIGTERM, log
    assert "idle_exit" in log, log
    # The lifespan's shutdown ran, as on Ctrl-C.
    assert "Application shutdown complete" in log, log

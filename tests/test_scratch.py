"""A project's scratchpad daemon: started once, found again, shared, stopped.

Each test starts real servers, so each stops what it started.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys

import httpx
import pytest

from strata import scratch
from strata.cli import main


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project; the daemons it starts keep their stores under *tmp_path*."""
    for key in [k for k in os.environ if k.startswith("STRATA_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("STRATA_NOTEBOOK_WARM_POOL_SIZE", "0")
    # A backstop: a daemon a failed test leaves behind stops itself.
    monkeypatch.setenv("STRATA_IDLE_EXIT_MINUTES", "1")
    started: list[int] = []
    spawn = scratch._spawn

    def recorded(anchor, port):
        pid = spawn(anchor, port)
        started.append(pid)
        return pid

    monkeypatch.setattr(scratch, "_spawn", recorded)
    root = tmp_path / "project"
    root.mkdir()
    yield root
    # Every one this test started, not only the one server.json names.
    for pid in started:
        # Gone already: ProcessLookupError, or on Windows a plain OSError.
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)


def test_the_first_call_starts_it_and_the_next_finds_it(project):
    first = scratch.ensure_daemon(project)
    assert list(first.setup_ms) == [
        "creating the scratchpad",
        "starting the scratchpad",
        "opening the scratchpad",
        "syncing the scratchpad's environment",
    ]

    again = scratch.ensure_daemon(project)
    assert (again.url, again.session_id, again.pid) == (first.url, first.session_id, first.pid)
    assert again.setup_ms == {}

    record = json.loads((project / ".strata-scratch" / ".strata" / "server.json").read_text())
    assert record["session_id"] == first.session_id
    assert record["anchor"] == str(project.resolve())
    # Ignored by the project's git without touching its .gitignore.
    assert (project / ".strata-scratch" / ".gitignore").read_text() == "*\n"
    assert not (project / ".gitignore").exists()
    # The scratchpad profile (only it sends instructions), on loopback.
    assert first.url.startswith("http://127.0.0.1:")
    initialized = httpx.post(
        f"{first.url}/mcp/",
        headers={"accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
    )
    assert "`run`" in _jsonrpc_result(initialized)["instructions"]


def _jsonrpc_result(response: httpx.Response) -> dict:
    """The result of a streamable-HTTP MCP response, JSON or one SSE event."""
    if response.headers["content-type"].startswith("application/json"):
        return response.json()["result"]
    (data,) = [line[5:] for line in response.text.splitlines() if line.startswith("data:")]
    return json.loads(data)["result"]


def test_two_clients_at_once_share_one_daemon(project):
    probe = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from strata.scratch import ensure_daemon\n"
        "d = ensure_daemon(Path(sys.argv[1]))\n"
        "print(json.dumps([d.pid, d.session_id]))\n"
    )
    clients = [
        subprocess.Popen(
            [sys.executable, "-c", probe, str(project)], stdout=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    seen = []
    for client in clients:
        out, _ = client.communicate(timeout=300)
        assert client.returncode == 0
        seen.append(tuple(json.loads(out)))
    for pid in {pid for pid, _ in seen}:
        # Started in the clients, so the fixture does not know them.
        os.kill(pid, signal.SIGTERM)
    assert seen[0] == seen[1]


def test_a_daemon_that_died_is_replaced(project):
    first = scratch.ensure_daemon(project)
    os.kill(first.pid, 9)

    second = scratch.ensure_daemon(project)
    assert second.pid != first.pid
    # The scratchpad was there already; only the server started again.
    assert "creating the scratchpad" not in second.setup_ms


def test_a_server_that_will_not_start_leaves_no_record(project, monkeypatch):
    monkeypatch.setattr(
        scratch, "_server_command", lambda: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    with pytest.raises(scratch.ScratchError, match="exited while starting"):
        scratch.ensure_daemon(project)
    assert scratch.find_daemon(project) is None
    assert not (project / ".strata-scratch" / ".strata" / "server.json").exists()


def test_the_cli_starts_shows_and_stops_it(project, capsys):
    assert main(["scratch", "status", "--root", str(project)]) == 1
    assert "no scratchpad daemon" in capsys.readouterr().out

    assert main(["scratch", "start", "--root", str(project)]) == 0
    started = json.loads(capsys.readouterr().out)
    assert main(["scratch", "status", "--root", str(project)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert (shown["pid"], shown["session_id"]) == (started["pid"], started["session_id"])

    assert main(["scratch", "stop", "--root", str(project)]) == 0
    assert capsys.readouterr().out.strip() == "stopped"
    assert scratch.find_daemon(project.resolve()) is None

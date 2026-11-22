"""A project's scratchpad daemon: started once, found again, shared, stopped.

Each test starts real servers, so each stops what it started.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from strata import scratch
from strata.cli import main


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
    # The error names the log that says why.
    with pytest.raises(scratch.ScratchError, match=r"exited while starting; see .*server\.log"):
        scratch.ensure_daemon(project)
    assert scratch.find_daemon(project) is None
    assert not (project / ".strata-scratch" / ".strata" / "server.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="a process group is POSIX")
def test_a_failed_start_kills_what_the_server_started(project, monkeypatch):
    child_pid = project / "child.pid"
    server = (
        "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"pathlib.Path({str(child_pid)!r}).write_text(str(child.pid))\n"
        "raise SystemExit(3)\n"
    )
    monkeypatch.setattr(scratch, "_server_command", lambda: [sys.executable, "-c", server])
    with pytest.raises(scratch.ScratchError, match="exited while starting"):
        scratch.ensure_daemon(project)
    pid = int(child_pid.read_text())
    try:
        # Orphaned, then reaped by init once killed: gone within a moment.
        deadline = time.monotonic() + 10
        while scratch._alive(pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not scratch._alive(pid)
    finally:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_open_has_the_setup_budget_and_its_failure_is_a_scratch_error(project, monkeypatch):
    # Open awaits the first sync, which on a cold uv cache outlasts any one request.
    timeouts = []

    def post(url, **kwargs):
        timeouts.append(kwargs["timeout"])
        raise httpx.ReadTimeout("slow sync")

    monkeypatch.setattr(scratch.httpx, "post", post)
    with pytest.raises(scratch.ScratchError, match="did not open it .*slow sync"):
        scratch.ensure_daemon(project)
    assert timeouts and timeouts[0] > 500
    assert scratch.find_daemon(project) is None


def test_a_daemon_another_version_started_is_replaced(project):
    first = scratch.ensure_daemon(project)
    record_path = project / ".strata-scratch" / ".strata" / "server.json"
    record = json.loads(record_path.read_text())
    record["version"] = "0.0.0"
    record_path.write_text(json.dumps(record))

    second = scratch.ensure_daemon(project)
    assert second.pid != first.pid
    # Stopped, not left beside the new one: one session writes the directory.
    assert not scratch._healthy(first.url)
    assert json.loads(record_path.read_text())["version"] == second.version != "0.0.0"


def test_a_busy_daemon_another_version_started_is_kept_until_idle(project):
    first = scratch.ensure_daemon(project)
    record_path = project / ".strata-scratch" / ".strata" / "server.json"
    record = json.loads(record_path.read_text())
    record["version"] = "0.0.0"
    record_path.write_text(json.dumps(record))

    # A request still open on it: its body never comes.
    port = int(first.url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port)) as held:
        held.sendall(
            b"POST /v1/notebooks/open HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\nContent-Length: 100\r\n\r\n"
        )
        deadline = time.monotonic() + 30
        while not scratch._busy(first.url):
            assert time.monotonic() < deadline, "a request still open is not busy"
            time.sleep(0.05)
        kept = scratch.ensure_daemon(project)
        assert (kept.pid, kept.version) == (first.pid, "0.0.0")
        assert scratch._healthy(first.url)

    deadline = time.monotonic() + 30
    while scratch._busy(first.url):
        assert time.monotonic() < deadline, "a request that ended is still busy"
        time.sleep(0.05)
    replaced = scratch.ensure_daemon(project)
    assert replaced.pid != first.pid
    assert not scratch._healthy(first.url)


@pytest.mark.parametrize(
    "answer",
    [httpx.ReadTimeout("slow"), httpx.Response(404, request=httpx.Request("GET", "http://x"))],
)
def test_a_daemon_that_does_not_say_whether_it_is_busy_is_busy(monkeypatch, answer):
    def get(url, **kwargs):
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(scratch.httpx, "get", get)
    assert scratch._busy("http://127.0.0.1:9")


def test_the_daemon_keeps_the_personal_caches_and_none_of_the_callers_config(tmp_path, monkeypatch):
    for key in [k for k in os.environ if k.startswith("STRATA_") or k == "UV_LINK_MODE"]:
        monkeypatch.delenv(key)
    anchor = tmp_path / "project"
    (anchor / ".strata-scratch" / ".strata").mkdir(parents=True)
    seen = {}

    def run(command, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="1\n")

    monkeypatch.setattr(scratch.subprocess, "run", run)
    scratch._spawn(anchor, 1)
    home = scratch.Path.home() / ".strata"
    # Not beside the storage dir, which is the project here.
    assert seen["env"]["STRATA_NOTEBOOK_MOUNT_CACHE_DIR"] == str(home / "mount_cache")
    assert seen["env"]["STRATA_NOTEBOOK_SHARED_ENV_DIR"] == str(home / "envs")
    # Its environment's files are the uv cache's own, not clones to be checked again.
    assert seen["env"]["UV_LINK_MODE"] == "hardlink"
    # Its pyproject.toml is the scratchpad's, wherever the client runs.
    assert seen["cwd"] == anchor / ".strata-scratch"
    if sys.platform != "win32":
        log = anchor / ".strata-scratch" / ".strata" / "server.log"
        assert log.stat().st_mode & 0o077 == 0

    monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_DIR", str(tmp_path / "mine"))
    monkeypatch.setenv("UV_LINK_MODE", "copy")
    scratch._spawn(anchor, 1)
    assert seen["env"]["STRATA_NOTEBOOK_MOUNT_CACHE_DIR"] == str(tmp_path / "mine")
    assert seen["env"]["UV_LINK_MODE"] == "copy"


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(pid, 0) is POSIX's probe")
def test_another_users_process_is_alive(monkeypatch):
    def kill(pid, sig):
        raise PermissionError

    monkeypatch.setattr(scratch.os, "kill", kill)
    assert scratch._alive(1)


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


def test_the_cli_says_when_another_versions_daemon_was_kept(project, monkeypatch, capsys):
    kept = scratch.Daemon(url="http://127.0.0.1:9", session_id="s", pid=1, version="0.0.0")
    monkeypatch.setattr(scratch, "ensure_daemon", lambda root, progress: kept)
    assert main(["scratch", "start", "--root", str(project)]) == 0
    assert "runs Strata 0.0.0" in capsys.readouterr().err


def _served(project, **over):
    state = project / ".strata-scratch" / ".strata"
    state.mkdir(parents=True)
    record = {
        "url": "http://127.0.0.1:9",
        "pid": os.getpid(),
        "session_id": "s",
        "version": "0",
        "anchor": str(project),
        **over,
    }
    (state / "server.json").write_text(json.dumps(record))


def test_a_daemon_too_busy_to_answer_is_not_replaced(project, monkeypatch):
    _served(project, pid=os.getpid())

    def slow(url, **kwargs):
        raise httpx.ReadTimeout("busy")

    monkeypatch.setattr(scratch.httpx, "get", slow)
    started = []
    monkeypatch.setattr(
        scratch, "_spawn", lambda anchor, port, python=None: started.append(port) or 1
    )
    # Alive and slow: a second daemon would write the directory beside it.
    with pytest.raises(scratch.NotAnswering, match=rf"pid {os.getpid()}\) is not answering"):
        scratch.ensure_daemon(project)
    assert started == []


def test_a_daemon_dying_as_it_is_asked_is_replaced(project, monkeypatch):
    """Killed a moment ago: its connections reset while its process still exists."""
    _served(project, pid=12345)

    def reset(url, **kwargs):
        raise httpx.ReadError("connection reset by peer")

    checks = iter([True, True, False])
    monkeypatch.setattr(scratch.httpx, "get", reset)
    monkeypatch.setattr(scratch, "_alive", lambda pid: next(checks))
    assert scratch.find_daemon(project) is None


def _stopping(monkeypatch, alive, timeout: float = 30.0) -> None:
    monkeypatch.setattr(scratch, "_healthy", lambda url: False)
    monkeypatch.setattr(scratch, "_alive", alive)
    monkeypatch.setattr(scratch.os, "kill", lambda pid, sig: None)
    daemon = scratch.Daemon(url="http://127.0.0.1:1", session_id="s", pid=12345)
    scratch._stop(daemon, timeout)


def test_a_stop_waits_for_the_process_not_only_its_port(monkeypatch):
    """Its port closes before it has closed its sessions and let go of their stores."""
    checks = iter([True, True, False])
    seen: list[bool] = []

    def alive(pid):
        seen.append(next(checks))
        return seen[-1]

    _stopping(monkeypatch, alive)

    assert seen == [True, True, False]


def test_a_process_that_outlasts_the_stop_is_an_error(monkeypatch):
    with pytest.raises(scratch.ScratchError, match="did not stop"):
        _stopping(monkeypatch, lambda pid: True, timeout=0.1)


@pytest.mark.skipif(
    sys.platform != "linux" or not Path("/proc/self/stat").exists(),
    reason="a zombie is read from /proc",
)
def test_a_zombie_is_not_alive():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    stat = Path(f"/proc/{child.pid}/stat")
    try:
        # Exited, not reaped: os.kill(pid, 0) still finds it.
        deadline = time.monotonic() + 60
        while stat.read_text().rpartition(")")[2].split()[0] != "Z":
            assert time.monotonic() < deadline, "the child did not exit"
            time.sleep(0.01)
        os.kill(child.pid, 0)
        assert not scratch._alive(child.pid)
    finally:
        child.wait()


class _Proc:
    """``Path`` as ``_alive`` sees it: ``/proc`` there or not, and what a stat read gives."""

    mounted = True
    stat: str | Exception = ProcessLookupError("reaped")

    def __init__(self, path: str) -> None:
        self.path = path

    def exists(self) -> bool:
        return self.mounted

    def read_text(self) -> str:
        if isinstance(self.stat, Exception):
            raise self.stat
        return self.stat


@pytest.mark.parametrize(
    ("stat", "alive"),
    [
        (ProcessLookupError("reaped"), False),
        (PermissionError("hidepid"), True),
        ("", False),
        ("1 (a) b) S 0", True),
        ("1 (py) Z 0", False),
        ("1 (py) X 0", False),
        ("1 (py) x 0", False),
    ],
)
def test_a_linux_process_is_alive_as_its_stat_says(monkeypatch, stat, alive):
    monkeypatch.setattr(scratch.sys, "platform", "linux")
    monkeypatch.setattr(scratch.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(_Proc, "stat", stat)
    monkeypatch.setattr(scratch, "Path", _Proc)

    assert scratch._alive(12345) is alive


def test_without_proc_a_process_os_kill_finds_is_alive(monkeypatch):
    monkeypatch.setattr(scratch.sys, "platform", "linux")
    monkeypatch.setattr(scratch.os, "kill", lambda pid, sig: None)
    monkeypatch.setattr(_Proc, "mounted", False)
    monkeypatch.setattr(scratch, "Path", _Proc)

    assert scratch._alive(12345)


def test_a_copy_of_a_served_directory_is_not_served(project, tmp_path, monkeypatch):
    """Its record came along, but that daemon holds the original, not the copy."""
    _served(project, anchor=str(tmp_path / "original"))

    def asked(url, **kwargs):
        raise AssertionError("asked the original's daemon")

    monkeypatch.setattr(scratch.httpx, "get", asked)
    assert scratch.find_daemon(project) is None


def test_a_daemon_answering_with_an_error_may_still_hold_the_session(project, monkeypatch):
    _served(project)
    monkeypatch.setattr(
        scratch.httpx,
        "get",
        lambda url, **kwargs: httpx.Response(500, request=httpx.Request("GET", url)),
    )
    monkeypatch.setattr(scratch, "_DYING_GRACE", 0.1)
    # Not taken for gone: the files would be edited behind its session.
    with pytest.raises(scratch.ScratchError, match="is not answering"):
        scratch.find_daemon(project)


def test_a_launch_that_fails_is_a_scratch_error(project, monkeypatch):
    def run(command, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(scratch.subprocess, "run", run)
    with pytest.raises(scratch.ScratchError, match="could not be launched: no such interpreter"):
        scratch.ensure_daemon(project)


def test_status_of_a_daemon_not_answering_is_an_error(project, monkeypatch, capsys):
    def busy(anchor):
        raise scratch.ScratchError("the scratchpad's server (pid 1) is not answering")

    monkeypatch.setattr(scratch, "find_daemon", busy)
    assert main(["scratch", "status", "--root", str(project)]) == 2
    assert "not answering" in capsys.readouterr().err


def test_the_scratchpad_keeps_every_value(project):
    from strata.notebook.parser import parse_notebook

    scratch._scaffold(project)
    assert parse_notebook(scratch.scratch_dir(project)).scratchpad

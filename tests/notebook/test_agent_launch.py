"""Tests for the offline pieces of the ``strata agent`` launcher.

Server spawn and TUI attach need a live process and are not covered here.
"""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from strata.notebook import agent_launch


def test_server_endpoints_defaults_and_explicit() -> None:
    assert agent_launch._server_endpoints("http://localhost:8765") == ("localhost", 8765)
    assert agent_launch._server_endpoints("http://127.0.0.1:9000") == ("127.0.0.1", 9000)
    # No explicit port → the notebook server default, not HTTP's 80.
    assert agent_launch._server_endpoints("http://localhost") == ("localhost", 8765)


def test_resolve_notebook_dir_creates_then_reuses(tmp_path) -> None:
    target = tmp_path / "demo"
    # First call scaffolds it (no venv build, to keep the test fast/offline).
    created = agent_launch._resolve_notebook_dir(str(target), None, initialize_environment=False)
    assert (created / "notebook.toml").is_file()

    # Second call sees notebook.toml and returns the same dir untouched.
    reused = agent_launch._resolve_notebook_dir(str(created), None, initialize_environment=False)
    assert reused == created


def test_write_agent_config_writes_mcp_json(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-abc")

    cfg = json.loads((nb / ".mcp.json").read_text())
    server = cfg["mcpServers"]["strata-notebook"]
    # Trailing slash matters: without it Starlette redirects /mcp -> /mcp/ and
    # the MCP client's handshake POST loses its body, so the tools never load.
    assert server == {"type": "http", "url": "http://localhost:8765/mcp/"}


def test_write_agent_config_claude_md_has_session_and_markers(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-xyz")

    text = (nb / "CLAUDE.md").read_text()
    assert agent_launch._BLOCK_START in text
    assert agent_launch._BLOCK_END in text
    assert "sess-xyz" in text
    # Points the agent at the MCP tools, not scratch scripts.
    assert "list_notebooks" in text
    assert "run_cell" in text


def test_write_agent_config_is_idempotent(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-aaa111")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-bbb222")

    text = (nb / "CLAUDE.md").read_text()
    # The managed block is rewritten in place, not duplicated.
    assert text.count(agent_launch._BLOCK_START) == 1
    assert text.count(agent_launch._BLOCK_END) == 1
    assert "sess-bbb222" in text
    assert "sess-aaa111" not in text


def test_write_agent_config_preserves_user_claude_md(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "CLAUDE.md").write_text("# My own notes\n\nKeep me.\n")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-1")

    text = (nb / "CLAUDE.md").read_text()
    assert "# My own notes" in text
    assert "Keep me." in text
    assert agent_launch._BLOCK_START in text


def test_write_agent_config_rewrites_block_keeps_user_text(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "CLAUDE.md").write_text("# Mine\n\nprose\n")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-first99")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-second99")

    text = (nb / "CLAUDE.md").read_text()
    assert "# Mine" in text
    assert "prose" in text
    assert text.count(agent_launch._BLOCK_START) == 1
    assert "sess-second99" in text
    assert "sess-first99" not in text


class _Resp:
    def __init__(self, status_code: int, content_type: str) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}


@pytest.mark.parametrize("status", [400, 406])
def test_mcp_mounted_true_for_json_4xx(monkeypatch, status) -> None:
    # A mounted MCP server endpoint answers a bare GET with a JSON 4xx.
    resp = _Resp(status, "application/json")
    monkeypatch.setattr(agent_launch.httpx, "get", lambda *a, **k: resp)
    assert agent_launch._mcp_mounted("http://localhost:8765") is True


def test_mcp_mounted_false_for_spa_html(monkeypatch) -> None:
    # MCP not mounted: /mcp/ falls through to the SPA and returns 200 text/html.
    resp = _Resp(200, "text/html; charset=utf-8")
    monkeypatch.setattr(agent_launch.httpx, "get", lambda *a, **k: resp)
    assert agent_launch._mcp_mounted("http://localhost:8765") is False


def test_mcp_mounted_false_for_404(monkeypatch) -> None:
    resp = _Resp(404, "application/json")
    monkeypatch.setattr(agent_launch.httpx, "get", lambda *a, **k: resp)
    assert agent_launch._mcp_mounted("http://localhost:8765") is False


def test_mcp_mounted_false_when_unreachable(monkeypatch) -> None:
    def _boom(*a, **k):
        raise agent_launch.httpx.ConnectError("refused")

    monkeypatch.setattr(agent_launch.httpx, "get", _boom)
    assert agent_launch._mcp_mounted("http://localhost:8765") is False


def test_guidance_mentions_remote_ssh_worker() -> None:
    text = agent_launch._agent_guidance("sess-abc")
    # The agent must know to reach for a remote worker when handed an SSH target.
    assert "connect_ssh_worker" in text
    assert "# @worker local" in text


def test_establish_ssh_worker_reports_success(monkeypatch, capsys) -> None:
    class _Resp:
        is_error = False

        def json(self):
            return {"worker": {"name": "gpu-box"}}

    monkeypatch.setattr(agent_launch.httpx, "post", lambda *a, **k: _Resp())
    agent_launch._establish_ssh_worker("http://localhost:8765", "sess", "user@gpu-box")
    assert "gpu-box" in capsys.readouterr().out


def test_establish_ssh_worker_warns_on_error_status(monkeypatch, capsys) -> None:
    class _Resp:
        is_error = True
        status_code = 400

        def json(self):
            return {"detail": "key auth failed"}

    monkeypatch.setattr(agent_launch.httpx, "post", lambda *a, **k: _Resp())
    agent_launch._establish_ssh_worker("http://localhost:8765", "sess", "user@box")
    err = capsys.readouterr().err
    assert "not connected" in err and "key auth failed" in err


def test_establish_ssh_worker_warns_on_network_error(monkeypatch, capsys) -> None:
    def _boom(*a, **k):
        raise agent_launch.httpx.ConnectError("refused")

    monkeypatch.setattr(agent_launch.httpx, "post", _boom)
    # A network failure is a warning, never an exception that aborts the launch.
    agent_launch._establish_ssh_worker("http://localhost:8765", "sess", "user@box")
    assert "could not connect" in capsys.readouterr().err


def test_ready_message_matches_whether_a_viewer_is_attached(tmp_path, capsys):
    """`--no-tui` attaches no viewer, so it must not say to watch or quit one."""
    agent_launch._print_ready(tmp_path, "http://127.0.0.1:8765", "sid", tui=True)
    with_tui = capsys.readouterr().out
    assert "TUI below" in with_tui

    agent_launch._print_ready(tmp_path, "http://127.0.0.1:8765", "sid", tui=False)
    without = capsys.readouterr().out
    assert "TUI" not in without
    assert "web UI at http://127.0.0.1:8765" in without


# Runs `strata agent --no-tui` with the server, its probes and the session open
# replaced; the stand-in server is a child process that waits to be stopped.
_SCRIPTED_LAUNCHER = """
import subprocess
import sys

from strata.cli import main
from strata.notebook import agent_launch

pid_file, notebook = sys.argv[1], sys.argv[2]


def _spawn(host, port, notebook_dir):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    return proc


agent_launch._server_alive = lambda url: False
agent_launch._spawn_server = _spawn
agent_launch._await_health = lambda url, proc: True
agent_launch._mcp_mounted = lambda url: True
agent_launch._open_session = lambda url, notebook_dir: "sid-from-the-pipe"
agent_launch._write_agent_config = lambda *args: None
sys.exit(main(["agent", notebook, "--no-tui", "--server", "http://127.0.0.1:9"]))
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_a_scripted_launcher_shows_the_session_and_sigterm_stops_its_server(tmp_path):
    """Piped stdout is block-buffered, and SIGTERM used to skip the cleanup."""
    from strata.notebook.writer import create_notebook

    notebook = create_notebook(tmp_path, "nb", initialize_environment=False)
    script = tmp_path / "launcher.py"
    script.write_text(_SCRIPTED_LAUNCHER)
    pid_file = tmp_path / "server.pid"
    src = Path(agent_launch.__file__).resolve().parents[2]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")]),
    }
    launcher = subprocess.Popen(
        [sys.executable, str(script), str(pid_file), str(notebook)],
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    )
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(
        target=lambda: [lines.put(line) for line in launcher.stdout], daemon=True
    ).start()
    server_pid: int | None = None
    try:
        line = ""
        while "sid-from-the-pipe" not in line:
            line = lines.get(timeout=60)
        server_pid = int(pid_file.read_text())
        assert _alive(server_pid)

        launcher.send_signal(signal.SIGTERM)
        launcher.wait(timeout=60)

        assert not _alive(server_pid)
    finally:
        if launcher.poll() is None:
            launcher.kill()
        if server_pid is not None and _alive(server_pid):
            os.kill(server_pid, signal.SIGKILL)

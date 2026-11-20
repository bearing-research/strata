"""``strata scratch run`` / ``show``: the MCP tools for an agent that has only a shell."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

import httpx
import pytest

from strata import scratch
from strata.cli import main


def test_scratch_run_starts_the_scratchpad_and_show_finds_what_it_holds(project, capsys):
    root = str(project)

    assert main(["scratch", "run", "-c", "x = 41\nx + 1", "--root", root]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "ok"
    assert first["value"] == 42
    assert "setup_ms" in first

    assert main(["scratch", "show", "x", "--root", root]) == 0
    held = json.loads(capsys.readouterr().out)
    assert "x = 41" in held["source"]

    assert main(["scratch", "run", "-c", "x + 1", "--root", root]) == 0
    again = json.loads(capsys.readouterr().out)
    assert again["value"] == 42
    assert "setup_ms" not in again
    assert scratch.find_daemon(project) is not None


def test_a_cell_that_fails_exits_1_with_its_error(project, capsys):
    assert main(["scratch", "run", "-c", "1 / 0", "--root", str(project)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "error"
    assert "ZeroDivisionError" in json.dumps(failed)


def test_a_source_file_and_a_cell_to_replace(project, tmp_path, capsys):
    root = str(project)
    source = tmp_path / "cell.py"
    source.write_text("y = 2\ny * 2")
    assert main(["scratch", "run", "--file", str(source), "--root", root]) == 0
    cell = json.loads(capsys.readouterr().out)["cell"]

    assert main(["scratch", "run", "-c", "y = 3\ny * 2", "--cell", cell, "--root", root]) == 0
    replaced = json.loads(capsys.readouterr().out)
    assert (replaced["cell"], replaced["value"]) == (cell, 6)


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX")
def test_a_terminated_run_cancels_its_cell_on_the_daemon(project):
    """An agent's shell tool ends a command past its timeout so; the cell must not run on."""
    daemon = scratch.ensure_daemon(project)
    cells_url = f"{daemon.url}/v1/notebooks/{daemon.session_id}/cells"
    command = [
        sys.executable,
        "-c",
        "import sys; from strata.cli import main; sys.exit(main(sys.argv[1:]))",
        *["scratch", "run", "-c", "import time\ntime.sleep(600)", "--root", str(project)],
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert _eventually(lambda: _statuses(cells_url) == ["running"]), _statuses(cells_url)
        process.send_signal(signal.SIGTERM)
        out, _ = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert process.returncode == 1
    assert json.loads(out)["error"].startswith("terminated")
    assert _eventually(lambda: _statuses(cells_url) not in ([], ["running"])), _statuses(cells_url)


def _statuses(url: str) -> list[str]:
    return [cell["status"] for cell in httpx.get(url).json()["cells"]]


def _eventually(check, attempts: int = 600) -> bool:
    """Polls *check* for up to a minute: a wait for a state, never a bound on a duration."""
    for _ in range(attempts):
        if check():
            return True
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("raised", ["scratch", "http"])
def test_a_call_that_fails_is_a_json_error_not_a_traceback(tmp_path, monkeypatch, capsys, raised):
    def fails(root, progress=None):
        if raised == "scratch":
            raise scratch.ScratchError("no project here")
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(scratch, "ensure_daemon", fails)

    assert main(["scratch", "run", "-c", "x = 1", "--root", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"

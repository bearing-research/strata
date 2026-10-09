"""Cancelling a cell's run from outside the WebSocket: the REST route, ``strata cell
cancel`` and the MCP ``cancel_cell`` tool.

Each drives a real harness subprocess running a sleep loop, and checks what the
browser's stop button guarantees: the cell's process group is gone, the cell is idle,
and the notebook runs again afterwards.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

from strata.cli import main
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX")

_BOUND_SECONDS = 60.0


def _slow_source(marker: Path) -> str:
    # The pgid is written once the loop is about to start, so a test can wait on it.
    return (
        "# @timeout 600\n"
        "import os, time\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(str(os.getpgid(0)))\n"
        "while True:\n"
        "    time.sleep(0.05)\n"
    )


def _notebook(tmp_path: Path) -> tuple[Path, Path]:
    marker = tmp_path / "pgid.txt"
    nb = create_notebook(tmp_path / "nb", "Cancel", initialize_environment=False)
    add_cell_to_notebook(nb, "slow", None)
    write_cell(nb, "slow", _slow_source(marker))
    add_cell_to_notebook(nb, "fast", "slow")
    write_cell(nb, "fast", "y = 1\n")
    return nb, marker


def _wait_for(predicate, what: str) -> None:
    deadline = time.monotonic() + _BOUND_SECONDS
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.05)


def _group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    return False


def _started_pgid(marker: Path) -> int:
    _wait_for(lambda: marker.exists() and marker.read_text().strip(), "the cell to start")
    pgid = int(marker.read_text())
    assert not _group_gone(pgid)
    return pgid


class TestOverTheServer:
    """REST route and CLI against a real server, with a REST run waiting on the cell."""

    @pytest.fixture
    def server(self, notebook_personal_server):
        # The server opens only notebooks under its storage dir.
        storage = Path(notebook_personal_server["config"].notebook_storage_dir)
        storage.mkdir(parents=True, exist_ok=True)
        nb, marker = _notebook(storage)
        base = notebook_personal_server["base_url"]
        opened = httpx.post(f"{base}/v1/notebooks/open", json={"path": str(nb)}, timeout=60)
        assert opened.status_code == 200, opened.text
        return base, opened.json()["session_id"], marker

    def _status(self, base: str, session_id: str, cell_id: str) -> str:
        state = httpx.get(f"{base}/v1/notebooks/sessions/{session_id}", timeout=30).json()
        return next(c["status"] for c in state["cells"] if c["id"] == cell_id)

    def _start_slow_run(self, pool, base, session_id, marker):
        marker.unlink(missing_ok=True)
        run = pool.submit(
            httpx.post,
            f"{base}/v1/notebooks/{session_id}/cells/slow/execute",
            timeout=_BOUND_SECONDS,
        )
        return run, _started_pgid(marker)

    def test_route_and_cli_cancel_a_running_cell(self, server, capsys):
        base, session_id, marker = server
        cancel_url = f"{base}/v1/notebooks/{session_id}/cells/slow/cancel"
        cli = ["cell", "cancel", "--server", base, "--session", session_id]

        with ThreadPoolExecutor(max_workers=1) as pool:
            run, pgid = self._start_slow_run(pool, base, session_id, marker)
            response = httpx.post(cancel_url, timeout=_BOUND_SECONDS)
            assert response.status_code == 200, response.text
            assert response.json() == {"cell_id": "slow", "cancelled": True, "status": "idle"}
            _wait_for(lambda: _group_gone(pgid), "the cell's process group to exit")
            waited = run.result(timeout=_BOUND_SECONDS)
            assert waited.status_code == 200
            assert waited.json()["status"] == "error"
            assert waited.json()["error_code"] == "cancelled"
            assert self._status(base, session_id, "slow") == "idle"

            # The cancelled run released the notebook: a second run starts, and
            # the CLI cancels it the same way.
            run, pgid = self._start_slow_run(pool, base, session_id, marker)
            capsys.readouterr()
            assert main([*cli, "slow", "--format", "json"]) == 0
            assert json.loads(capsys.readouterr().out) == {
                "cell_id": "slow",
                "cancelled": True,
                "status": "idle",
            }
            _wait_for(lambda: _group_gone(pgid), "the cell's process group to exit")
            assert run.result(timeout=_BOUND_SECONDS).json()["error_code"] == "cancelled"
            assert self._status(base, session_id, "slow") == "idle"

        ran = ["cell", "run", "--server", base, "--session", session_id, "fast"]
        assert main([*ran, "--format", "json"]) == 0
        assert json.loads(capsys.readouterr().out)["status"] == "ok"

    def test_a_cell_that_is_not_running_reports_so(self, server, capsys):
        base, session_id, _ = server
        url = f"{base}/v1/notebooks/{session_id}/cells/fast"
        cli = ["cell", "cancel", "--server", base, "--session", session_id, "fast"]

        idle = httpx.post(f"{url}/cancel", timeout=30)
        assert idle.json() == {"cell_id": "fast", "cancelled": False, "status": "idle"}

        assert httpx.post(f"{url}/execute", timeout=_BOUND_SECONDS).json()["status"] == "ready"
        # A late cancel does not clobber the finished cell.
        done = httpx.post(f"{url}/cancel", timeout=30)
        assert done.json() == {"cell_id": "fast", "cancelled": False, "status": "ready"}

        capsys.readouterr()
        assert main([*cli, "--format", "json"]) == 1
        assert json.loads(capsys.readouterr().out)["cancelled"] is False
        assert main([*cli, "--format", "human"]) == 1
        assert "not running" in capsys.readouterr().out

        missing = httpx.post(f"{base}/v1/notebooks/{session_id}/cells/ghost/cancel", timeout=30)
        assert missing.status_code == 404


def test_cli_cancel_of_a_local_notebook_says_it_cannot(tmp_path, capsys):
    nb, _ = _notebook(tmp_path)

    assert main(["cell", "cancel", str(nb), "slow"]) == 2
    err = capsys.readouterr().err
    assert "--server" in err and "Ctrl-C" in err


@pytest.mark.asyncio
async def test_mcp_cancel_cell_stops_a_run_cell(tmp_path):
    from strata.notebook.mcp_server import _cancel_cell, _run_cell
    from strata.notebook.session import SessionManager

    nb, marker = _notebook(tmp_path)
    sm = SessionManager()
    session = sm.open_notebook(nb)
    try:
        idle = await _cancel_cell(sm, session.id, "slow")
        assert idle == {"cell_id": "slow", "cancelled": False, "status": "idle"}

        run = asyncio.create_task(_run_cell(sm, session.id, "slow"))
        pgid = await asyncio.to_thread(_started_pgid, marker)

        cancelled = await _cancel_cell(sm, session.id, "slow")
        assert cancelled == {"cell_id": "slow", "cancelled": True, "status": "idle"}
        result = await asyncio.wait_for(run, timeout=_BOUND_SECONDS)
        assert result["status"] == "error" and result["error_code"] == "cancelled"
        await asyncio.to_thread(_wait_for, lambda: _group_gone(pgid), "the process group to exit")

        assert (await _run_cell(sm, session.id, "fast"))["status"] == "ok"
        done = await _cancel_cell(sm, session.id, "fast")
        assert done == {"cell_id": "fast", "cancelled": False, "status": "ready"}
    finally:
        await asyncio.gather(*sm.close_session(session.id))

"""``strata cell|status`` on a directory a daemon serves go through the daemon."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from strata import scratch
from strata.cli import main
from strata.notebook.writer import create_notebook


def test_a_served_directory_is_driven_through_its_daemon(project, capsys):
    daemon = scratch.ensure_daemon(project)
    notebook = str(project / ".strata-scratch")

    assert main(["cell", "add", notebook, "-c", "y = 7\ny * 6", "--run"]) == 0
    added = json.loads(capsys.readouterr().out)
    # The daemon's own session holds the cell and ran it: no offline writer raced it.
    cells = httpx.get(f"{daemon.url}/v1/notebooks/{daemon.session_id}/cells").json()["cells"]
    (cell,) = [c for c in cells if c["source"] == "y = 7\ny * 6"]
    assert cell["status"] == "ready"
    assert added["id"] == cell["id"]

    # Read through the daemon too, closing what they opened.
    assert main(["status", notebook]) == 0
    assert cell["id"] in capsys.readouterr().out
    assert main(["dag", notebook]) == 0
    capsys.readouterr()
    # A command that only edits the files directly refuses instead of racing the session.
    assert main(["worker", "add", notebook, "gpu", "--url", "http://127.0.0.1:1"]) == 2
    assert "holds" in capsys.readouterr().err
    # So does running it all in a second session; one that only reads goes ahead.
    assert main(["run", notebook]) == 2
    assert "runs it in a session of its own" in capsys.readouterr().err
    assert main(["worker", "ls", notebook]) == 0
    capsys.readouterr()

    # Nothing runs, and that is the daemon's answer, not "needs --server".
    assert main(["cell", "cancel", notebook, cell["id"]]) == 1
    capsys.readouterr()

    assert scratch.stop_daemon(project)
    # With no daemon, the same command reads the files the daemon wrote.
    assert main(["cell", "list", notebook]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert cell["id"] in [c["id"] for c in listed]


@pytest.mark.parametrize(
    "command",
    [
        ["status"],
        ["cell", "list"],
        ["worker", "add", "{nb}", "gpu", "--url", "http://127.0.0.1:1"],
        ["cell", "cancel", "{nb}", "c1"],
    ],
)
def test_a_daemon_not_answering_is_an_error_not_a_traceback(command, tmp_path, monkeypatch, capsys):
    notebook = _scratchpad(tmp_path)
    _record(notebook)

    def slow(url, **kwargs):
        raise httpx.ReadTimeout("busy")

    monkeypatch.setattr(scratch.httpx, "get", slow)
    args = [a.replace("{nb}", str(notebook)) for a in command]
    if "{nb}" not in command:
        args.append(str(notebook))
    # Alive but slow: neither driven through nor edited behind.
    assert main(args) == 2
    assert "is not answering" in capsys.readouterr().err


def _scratchpad(anchor):
    scratch._scaffold(anchor)
    return scratch.scratch_dir(anchor)


def _record(notebook, **over):
    record = {
        "url": "http://127.0.0.1:9",
        "pid": os.getpid(),
        "session_id": "s",
        "version": "0",
        "anchor": str(notebook.parent),
        **over,
    }
    (notebook / ".strata").mkdir(exist_ok=True)
    (notebook / ".strata" / "server.json").write_text(json.dumps(record))


def test_a_path_that_is_a_file_is_not_a_notebook(tmp_path, capsys):
    path = tmp_path / "notes.txt"
    path.write_text("")
    assert main(["status", str(path)]) == 2
    assert "is not a Strata notebook" in capsys.readouterr().err


def test_a_read_asks_for_the_daemon_once(tmp_path, monkeypatch, capsys):
    from strata.notebook import cli as notebook_cli

    notebook = create_notebook(tmp_path, "nb", initialize_environment=False)
    asked = []
    monkeypatch.setattr(notebook_cli, "_daemon_serving", lambda d: asked.append(d))
    assert main(["cell", "list", str(notebook)]) == 0
    assert len(asked) == 1


def test_cancel_with_a_session_but_no_server_does_not_ask_the_daemon(tmp_path, monkeypatch, capsys):
    notebook = _scratchpad(tmp_path)
    _record(notebook)

    def asked(url, **kwargs):
        raise AssertionError("asked the daemon")

    monkeypatch.setattr(scratch.httpx, "get", asked)
    assert main(["cell", "cancel", str(notebook), "c1", "--session", "mine"]) == 2
    assert "needs --server" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_a_remote_run_resolve_or_test_has_no_read_timeout():
    from strata.notebook.ops import RemoteNotebookOps

    reads = {}

    def daemon(request):
        reads[request.url.path.rsplit("/", 1)[-1]] = request.extensions["timeout"]["read"]
        if request.url.path.endswith("/dependencies"):
            return httpx.Response(200, json={"package": "six", "action": "add"})
        return httpx.Response(200, json={})

    ops = RemoteNotebookOps(
        "http://daemon", "s", client=httpx.Client(transport=httpx.MockTransport(daemon))
    )
    for call in (ops.run_cell("c1"), ops.run_tests("c1"), ops.add_dependency("six")):
        try:
            await call
        except Exception:  # noqa: BLE001 (only the request's timeout is under test)
            pass
    # A cell's own timeout ends a run; the client's 30s would cut a long one off.
    assert reads == {"execute": None, "tests": None, "dependencies": None}

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
    notebook = create_notebook(tmp_path, "nb", initialize_environment=False)
    record = {"url": "http://127.0.0.1:9", "pid": os.getpid(), "session_id": "s", "version": "0"}
    (notebook / ".strata").mkdir(exist_ok=True)
    (notebook / ".strata" / "server.json").write_text(json.dumps(record))

    def slow(url, **kwargs):
        raise httpx.ReadTimeout("busy")

    monkeypatch.setattr(scratch.httpx, "get", slow)
    args = [a.replace("{nb}", str(notebook)) for a in command]
    if "{nb}" not in command:
        args.append(str(notebook))
    # Alive but slow: neither driven through nor edited behind.
    assert main(args) == 2
    assert "is not answering" in capsys.readouterr().err

"""``strata cell|status`` on a directory a daemon serves go through the daemon."""

from __future__ import annotations

import json

import httpx

from strata import scratch
from strata.cli import main


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

    # Nothing runs, and that is the daemon's answer, not "needs --server".
    assert main(["cell", "cancel", notebook, cell["id"]]) == 1
    capsys.readouterr()

    assert scratch.stop_daemon(project)
    # With no daemon, the same command reads the files the daemon wrote.
    assert main(["cell", "list", notebook]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert cell["id"] in [c["id"] for c in listed]

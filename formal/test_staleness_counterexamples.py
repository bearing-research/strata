"""Replay Staleness.tla's counterexamples through the real notebook WebSocket.

Each test passes while its bug exists (asserts the violating outcome).
Cells really execute, in a harness subprocess, as in tests/notebook.

    uv run pytest formal/ -v
"""

# Fixtures are imported from tests/ and then requested by name (or autouse).
# ruff: noqa: F811

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _reset_process_globals  # noqa: F401  (autouse)
from tests.notebook.conftest import fast_notebook_env  # noqa: F401  (autouse)
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    _reset_ws_globals,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)


@pytest.fixture
def setup():
    _reset_ws_globals()
    client = TestClient(create_test_app())
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


def _statuses(ws):
    return {cell["id"]: cell["status"] for cell in ws.sync()["payload"]["cells"]}


def test_upstream_edited_during_a_run_leaves_the_downstream_ready(setup):
    """Staleness_EditDuringRun / ReadyMeansCurrent and RunningShown.

    b reads a. While b runs, a is edited, which is allowed because only the
    running cell is locked. The flush's walk marks a stale and overwrites
    b's "running" status with a walk verdict. When b finishes, the walk
    rightly finds b stale (its upstream is), but preserve_ready_cell_id
    marks it READY anyway, and that is what the UI, the cell list and
    agents are told.

    The damage stops at the reported status: the next run re-checks
    provenance itself, so running c silently rebuilds a and b and computes
    from the new source.
    """
    client, tmp = setup
    nb = (
        NotebookBuilder(tmp)
        .add_cell("a", "x = 1")
        .add_cell("b", "import time\ntime.sleep(2)\ny = x + 1", "a")
        .add_cell("c", "z = y * 10\nz", "b")
    )
    with open_notebook_session(client, nb.path) as (sid, session):
        with ws_connect(client, sid) as ws:
            execute_cell_and_wait(ws, "a")

            ws.execute_cell("b")
            ws.receive_until("cell_status", cell_id="b", status="running")
            ws.clear()
            ws.update_source("a", "x = 100")
            ws.receive_until("dag_update")

            # b is still executing, but the session no longer says so.
            assert session.notebook_state.get_cell("b").status.value != "running"

            ws.receive_until("cell_status", cell_id="b", status="ready", max_messages=200)
            statuses = _statuses(ws)
            assert statuses["a"] in ("idle", "stale")  # edited, not re-run
            assert statuses["b"] == "ready"  # built from the old a

            ws.clear()
            execute_cell_and_wait(ws, "c")
            # No cascade was offered, but the executor rebuilt a and b on
            # its own, so c's value comes from x = 100.
            assert not ws.messages_of_type("cascade_prompt")
            output = next(
                m
                for m in ws.messages
                if m["type"] == "cell_output" and m["payload"]["cell_id"] == "c"
            )
            assert "1010" in str(output["payload"])

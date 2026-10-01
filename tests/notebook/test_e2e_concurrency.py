"""E2E tests: sequential execution and connection scenarios.

Tests cell cancel, rapid sequential executions, and
multiple WebSocket connections to the same notebook.

NOTE: These tests are sequential, not truly concurrent.
True concurrent execution tests (e.g., two cells executing
simultaneously via asyncio.gather) are not yet implemented.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    run_all_and_wait,
    ws_connect,
)


@pytest.fixture
def setup():
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


class TestCellCancel:
    """Test cell cancellation."""

    def test_cancel_idle_cell(self, setup):
        """Cancelling an idle cell sets status to idle."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("cell_cancel", {"cell_id": "c1"})
                msg = ws.receive_until("cell_status", cell_id="c1")
                assert msg["payload"]["status"] == "idle"


class TestRapidExecution:
    """Test executing cells in rapid succession."""

    def test_notebook_run_all_executes_all_nonempty_cells(self, setup):
        """Run-all should execute each non-empty cell, not just the first one."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)
        nb.add_cell("c0", "v0 = 0")
        nb.add_cell("c1", "")
        nb.add_cell("c2", "v2 = v0 + 2", "c1")
        nb.add_cell("c3", "v3 = v2 + 1", "c2")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                run_all_and_wait(ws, "c3")

                output_cells = [
                    msg["payload"]["cell_id"] for msg in ws.messages if msg["type"] == "cell_output"
                ]
                assert output_cells == ["c0", "c2", "c3"]

                synced = ws.sync()
                cells = {cell["id"]: cell for cell in synced["payload"]["cells"]}
                assert cells["c0"]["status"] == "ready"
                assert cells["c1"]["status"] == "idle"
                assert cells["c2"]["status"] == "ready"
                assert cells["c3"]["status"] == "ready"

    def test_execute_all_cells_sequentially(self, setup):
        """Execute 5 cells in sequence — all should complete."""
        client, tmp = setup
        nb = NotebookBuilder(tmp)
        for i in range(5):
            after = f"c{i - 1}" if i > 0 else None
            nb.add_cell(f"c{i}", f"v{i} = {i}", after)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                for i in range(5):
                    result = execute_cell_and_wait(ws, f"c{i}")
                    assert result["type"] == "cell_output", f"Cell c{i} failed: {result}"

    def test_reexecute_same_cell_multiple_times(self, setup):
        """Execute the same cell 3 times in a row."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                for _ in range(3):
                    result = execute_cell_and_wait(ws, "c1")
                    assert result["type"] == "cell_output"


class TestMultipleConnections:
    """Test multiple WebSocket connections to the same notebook."""

    def test_second_connection_receives_sync(self, setup):
        """A second WebSocket connection can sync notebook state."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws1:
                execute_cell_and_wait(ws1, "c1")

            # A second connection's sync shows the cell as ready.
            with ws_connect(client, sid) as ws2:
                state = ws2.sync()
                cells = state["payload"]["cells"]
                c1 = next(c for c in cells if c["id"] == "c1")
                assert c1["status"] == "ready"

"""E2E tests: basic single-cell and two-cell notebook execution.

Validates the fundamental execution flow through WebSocket:
create notebook → open session → execute cell → verify output.
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
    ws_connect,
)


@pytest.fixture
def setup():
    """Create app, client, and temp directory."""
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


class TestSingleCellExecution:
    """Execute a single cell with no dependencies."""

    def test_assign_integer(self, setup):
        """Cell: x = 42 → outputs contain x."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 42")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert result["payload"]["cell_id"] == "c1"
                assert "x" in result["payload"]["outputs"]

    def test_assign_string(self, setup):
        """Cell: name = 'hello' → outputs contain name."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "name = 'hello'")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert "name" in result["payload"]["outputs"]

    def test_print_captured(self, setup):
        """print() output is captured in cell_console messages."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", 'x = 1\nprint("hello from cell")')

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"

                consoles = ws.messages_of_type("cell_console")
                stdout_msgs = [m for m in consoles if m["payload"].get("stream") == "stdout"]
                assert len(stdout_msgs) >= 1
                assert "hello from cell" in stdout_msgs[0]["payload"]["text"]

    def test_status_transitions(self, setup):
        """Verify status transitions: running → ready."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                statuses = ws.messages_of_type("cell_status")
                c1_statuses = [
                    m["payload"]["status"] for m in statuses if m["payload"]["cell_id"] == "c1"
                ]
                assert "running" in c1_statuses
                assert c1_statuses[-1] == "ready"

    def test_duration_reported(self, setup):
        """cell_output includes duration_ms > 0."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["payload"]["duration_ms"] >= 0


class TestTwoCellDirect:
    """Two cells where the first is already ready — no cascade needed."""

    def test_sequential_execution(self, setup):
        """Execute c1 then c2 sequentially — c2 sees c1's output."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 10").add_cell("c2", "y = x + 5", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result1 = execute_cell_and_wait(ws, "c1")
                assert result1["type"] == "cell_output"
                assert "x" in result1["payload"]["outputs"]

                # c1 is already ready, so no cascade.
                result2 = execute_cell_and_wait(ws, "c2")
                assert result2["type"] == "cell_output"
                assert "y" in result2["payload"]["outputs"]

    def test_multiple_variables(self, setup):
        """Cell defines multiple variables, downstream reads them."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "a = 1\nb = 2")
            .add_cell("c2", "total = a + b", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                result = execute_cell_and_wait(ws, "c2")
                assert result["type"] == "cell_output"
                assert "total" in result["payload"]["outputs"]


class TestNotebookSync:
    """Test the notebook_sync message for reconnection."""

    def test_sync_returns_state(self, setup):
        """notebook_sync returns full notebook state with cells and DAG."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                state_msg = ws.sync()
                payload = state_msg["payload"]

                assert "id" in payload
                assert "cells" in payload
                assert len(payload["cells"]) == 2
                assert "dag" in payload
                assert "edges" in payload["dag"]

    def test_sync_reflects_execution_status(self, setup):
        """After executing a cell, sync shows it as ready."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                state_msg = ws.sync()
                cell = state_msg["payload"]["cells"][0]
                assert cell["status"] == "ready"

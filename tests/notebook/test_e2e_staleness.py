"""E2E tests: staleness detection after source edits.

When an upstream cell's source changes, downstream cells should be
marked as stale, and re-execution should reflect the new values.
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
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


class TestStalenessDetection:
    """Editing an upstream cell marks downstream cells as stale."""

    def test_source_edit_sends_dag_update(self, setup):
        """Edit c1 source → dag_update message is sent."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Edit c1 source
                ws.update_source("c1", "x = 100")

                # Should receive dag_update
                dag_msg = ws.receive_until("dag_update")
                assert "edges" in dag_msg["payload"]
                assert "topological_order" in dag_msg["payload"]

    def test_re_execution_after_edit(self, setup):
        """After editing c1, re-executing c2 should use new c1 value."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Run pipeline
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")

                # Edit c1 via WebSocket
                ws.update_source("c1", "x = 100")
                ws.receive_until("dag_update")
                ws.clear()

                # Re-execute c1 with new source
                execute_cell_and_wait(ws, "c1")

                # Re-execute c2 — should use x=100
                r2 = execute_cell_and_wait(ws, "c2")
                assert r2["type"] == "cell_output"
                assert "y" in r2["payload"]["outputs"]
                # Cache miss expected since c1 changed
                assert r2["payload"].get("cache_hit") is not True

    def test_edit_and_cascade(self, setup):
        """After editing c1, executing c2 triggers cascade to re-run c1."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Run both cells
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")

                # Edit c1 source (but don't re-execute c1)
                ws.update_source("c1", "x = 999")
                ws.receive_until("dag_update")
                # Note: cell statuses may follow but we don't need to drain them
                # The cell_status messages will just accumulate in the message buffer
                ws.clear()

                # Now execute c2 — should trigger cascade since c1's source changed
                # and status was reset
                result = execute_cell_and_wait(ws, "c2")
                assert result["type"] == "cell_output"
                assert "y" in result["payload"]["outputs"]

    def test_edit_exported_function_invalidates_downstream(self, setup):
        """Editing a module-exported function should invalidate downstream reuse."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "def add(a, b):\n    return a + b")
            .add_cell("c2", "result = add(2, 3)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                first = execute_cell_and_wait(ws, "c2")
                assert first["type"] == "cell_output"
                assert first["payload"]["outputs"]["result"]["preview"] == 5

                resp = client.put(
                    f"/v1/notebooks/{sid}/cells/c1",
                    json={"source": "def add(a, b):\n    return a * b"},
                )
                assert resp.status_code == 200

                data = resp.json()
                c2 = next(cell for cell in data["cells"] if cell["id"] == "c2")
                # c2 ran, so it holds a result that the c1 edit invalidated →
                # STALE with an UPSTREAM reason (#361), not a bare idle.
                assert c2["status"] == "stale"
                assert "upstream" in c2.get("staleness_reasons", [])

                ws.clear()
                updated = execute_cell_and_wait(ws, "c2")
                assert updated["type"] == "cell_output"
                assert updated["payload"].get("cache_hit") is not True
                assert updated["payload"]["outputs"]["result"]["preview"] == 6


class TestDAGRestructuring:
    """Edits that change the DAG structure."""

    def test_add_new_dependency(self, setup):
        """Edit c2 to reference a new variable from c1."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 1\nz = 99")
            .add_cell("c2", "y = x + 1", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Change c2 to also reference z
                ws.update_source("c2", "y = x + z")
                dag_msg = ws.receive_until("dag_update")

                # DAG should show edges from c1 to c2 for both x and z
                edges = dag_msg["payload"]["edges"]
                c1_to_c2_vars = [
                    e["variable"]
                    for e in edges
                    if e["from_cell_id"] == "c1" and e["to_cell_id"] == "c2"
                ]
                assert "x" in c1_to_c2_vars
                assert "z" in c1_to_c2_vars

    def test_remove_dependency(self, setup):
        """Edit c2 to no longer reference c1's variable."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Change c2 to be independent
                ws.update_source("c2", "y = 999")
                dag_msg = ws.receive_until("dag_update")

                # No edge from c1 to c2 anymore
                edges = dag_msg["payload"]["edges"]
                c1_to_c2 = [
                    e for e in edges if e["from_cell_id"] == "c1" and e["to_cell_id"] == "c2"
                ]
                assert len(c1_to_c2) == 0

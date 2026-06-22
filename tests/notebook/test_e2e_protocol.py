"""E2E tests: WebSocket protocol correctness.

Validates message ordering, error handling for malformed messages,
and unknown notebook connections.
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


class TestMessageOrdering:
    """Verify correct ordering of WebSocket messages."""

    def test_running_before_output(self, setup):
        """cell_status(running) must come before cell_output."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                # Find indices of running status and output
                running_idx = None
                output_idx = None
                for i, m in enumerate(ws.messages):
                    if (
                        m["type"] == "cell_status"
                        and m["payload"].get("cell_id") == "c1"
                        and m["payload"].get("status") == "running"
                    ):
                        running_idx = i
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c1":
                        output_idx = i

                assert running_idx is not None, "No running status found"
                assert output_idx is not None, "No cell_output found"
                assert running_idx < output_idx, (
                    f"running (idx {running_idx}) should come before output (idx {output_idx})"
                )

    def test_output_before_ready(self, setup):
        """cell_output must come before cell_status(ready)."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                output_idx = None
                ready_idx = None
                for i, m in enumerate(ws.messages):
                    if m["type"] == "cell_output" and m["payload"].get("cell_id") == "c1":
                        output_idx = i
                    if (
                        m["type"] == "cell_status"
                        and m["payload"].get("cell_id") == "c1"
                        and m["payload"].get("status") == "ready"
                    ):
                        ready_idx = i

                assert output_idx is not None, "No cell_output found"
                assert ready_idx is not None, "No ready status found"
                assert output_idx < ready_idx

    def test_cascade_order_is_topological(self, setup):
        """During cascade, cells execute in topological order."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 1")
            .add_cell("c2", "y = x + 1", after="c1")
            .add_cell("c3", "z = y + 1", after="c2")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c3")

                # Extract the order cells went to "running"
                running_order = []
                for m in ws.messages:
                    if m["type"] == "cell_status" and m["payload"].get("status") == "running":
                        running_order.append(m["payload"]["cell_id"])

                # c1 should run before c2, c2 before c3
                if "c1" in running_order and "c2" in running_order:
                    assert running_order.index("c1") < running_order.index("c2")
                if "c2" in running_order and "c3" in running_order:
                    assert running_order.index("c2") < running_order.index("c3")


class TestProtocolErrors:
    """Malformed or invalid messages."""

    def test_missing_cell_id(self, setup):
        """cell_execute without cell_id returns error."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("cell_execute", {})  # No cell_id
                msg = ws.receive()
                assert msg["type"] == "error"
                assert (
                    "cell_id" in msg["payload"]["error"].lower()
                    or "missing" in msg["payload"]["error"].lower()
                )

    def test_unknown_message_type(self, setup):
        """Unknown message type returns error."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("totally_made_up", {"foo": "bar"})
                msg = ws.receive()
                assert msg["type"] == "error"

    def test_nonexistent_cell(self, setup):
        """Executing a non-existent cell returns error."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.execute_cell("nonexistent_cell_id")
                msg = ws.receive()
                assert msg["type"] == "error"
                assert "not found" in msg["payload"]["error"].lower()

    def test_unknown_notebook_ws(self):
        """Connecting to WebSocket for unknown session should fail."""
        app = create_test_app()
        client = TestClient(app)

        with pytest.raises(Exception):
            with client.websocket_connect("/v1/notebooks/ws/nonexistent") as _:
                pass


class TestSourceUpdate:
    """Source update protocol tests."""

    def test_source_update_returns_dag(self, setup):
        """cell_source_update → dag_update with edges."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.update_source("c1", "x = 999")
                msg = ws.receive_until("dag_update")

                assert "edges" in msg["payload"]
                assert "topological_order" in msg["payload"]
                assert "roots" in msg["payload"]
                assert "leaves" in msg["payload"]

    def test_source_update_missing_source(self, setup):
        """cell_source_update without source field → error."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                ws.send("cell_source_update", {"cell_id": "c1"})
                msg = ws.receive()
                assert msg["type"] == "error"


class TestDataTypes:
    """Test different Python data types survive the artifact round-trip."""

    @pytest.mark.parametrize(
        ("source", "var_name"),
        [
            ('d = {"a": 1, "b": [2, 3]}', "d"),
            ("items = [1, 2, 3, 4, 5]", "items"),
            ("nothing = None", "nothing"),
            ("flag = True", "flag"),
        ],
    )
    def test_primitive_output_is_serialized(self, setup, source, var_name):
        """Dict / list / None / bool variables are serialized into cell_output."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", source)

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert var_name in result["payload"]["outputs"]

    def test_variable_round_trip(self, setup):
        """Variable from c1 survives artifact store and is usable in c2."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", 'data = {"key": "value", "num": 42}')
            .add_cell("c2", 'result = data["key"] + str(data["num"])', after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                result = execute_cell_and_wait(ws, "c2")
                assert result["type"] == "cell_output"
                assert "result" in result["payload"]["outputs"]

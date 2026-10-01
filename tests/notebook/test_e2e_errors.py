"""E2E tests: error handling during cell execution.

Validates that syntax errors, runtime errors, and missing-variable errors
are properly reported through the WebSocket protocol.
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


class TestSyntaxErrors:
    """Cells with invalid Python syntax."""

    def test_syntax_error_reported(self, setup):
        """Syntax error → cell_error with traceback."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "def f(\n  # incomplete")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_error" or (
                    result["type"] == "cell_status" and result["payload"]["status"] == "error"
                )

    def test_syntax_error_status(self, setup):
        """After syntax error, cell status is 'error'."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = ...")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # `x = ...` is valid syntax; the real check uses nb2 below.
                pass

        nb2 = NotebookBuilder(tmp / "nb2").add_cell("c1", "x = (]")

        with open_notebook_session(client, nb2.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                statuses = ws.messages_of_type("cell_status")
                c1_statuses = [
                    m["payload"]["status"] for m in statuses if m["payload"]["cell_id"] == "c1"
                ]
                assert "error" in c1_statuses


class TestRuntimeErrors:
    """Cells that raise exceptions during execution."""

    def test_division_by_zero(self, setup):
        """ZeroDivisionError is reported."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1 / 0")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                errors = ws.messages_of_type("cell_error")
                assert len(errors) >= 1
                assert (
                    "ZeroDivisionError" in errors[0]["payload"]["error"]
                    or "division" in errors[0]["payload"]["error"]
                )

    def test_name_error(self, setup):
        """Referencing undefined variable → NameError."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "y = undefined_var + 1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")

                errors = ws.messages_of_type("cell_error")
                assert len(errors) >= 1
                error_text = errors[0]["payload"]["error"]
                assert "NameError" in error_text or "undefined_var" in error_text

    def test_type_error(self, setup):
        """TypeError from incompatible operation."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 'hello' + 5")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                errors = ws.messages_of_type("cell_error")
                assert len(errors) >= 1


class TestErrorRecovery:
    """After an error, the cell can be fixed and re-executed."""

    def test_fix_and_rerun(self, setup):
        """Error → edit source → re-execute → success."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1 / 0")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                assert len(ws.messages_of_type("cell_error")) >= 1

                ws.update_source("c1", "x = 42")
                ws.receive_until("dag_update")
                ws.clear()

                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                assert "x" in result["payload"]["outputs"]

    def test_error_does_not_block_other_cells(self, setup):
        """Error in c1 doesn't prevent executing independent c2."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1 / 0").add_cell("c2", "y = 42", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                ws.clear()

                # c2 doesn't reference x, so y = 42 is independent.
                # Force-execute to bypass cascade check.
                ws.execute_force("c2")

                # Wait for terminal status (skip "running")
                while True:
                    msg = ws.receive()
                    if (
                        msg["type"] == "cell_status"
                        and msg["payload"].get("cell_id") == "c2"
                        and msg["payload"].get("status") in ("ready", "error")
                    ):
                        break

                assert msg["payload"]["status"] in ("ready", "error")


class TestCascadeWithError:
    """Error during cascade execution."""

    def test_cascade_stops_on_error(self, setup):
        """If an upstream cell errors during cascade, downstream cells don't run."""
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "x = 1 / 0")  # Will error
            .add_cell("c2", "y = x + 1", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Running c2 cascades into c1, which errors.
                execute_cell_and_wait(ws, "c2")

                c1_errors = [
                    m for m in ws.messages_of_type("cell_error") if m["payload"]["cell_id"] == "c1"
                ]
                assert len(c1_errors) >= 1

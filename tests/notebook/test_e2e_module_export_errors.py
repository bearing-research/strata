"""E2E tests: unsupported cross-cell module export errors."""

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


class TestUnsupportedModuleExports:
    """Unsupported reusable-code cells should fail clearly over WS."""

    def test_def_with_unresolved_runtime_dep_error_surfaces_over_websocket(self, setup):
        # ``add`` references ``x``, which is bound nowhere in the notebook —
        # a truly-unknown name. The synthetic module would NameError when
        # ``add`` is called, so we block at execution time and surface a
        # precise message. (A same-cell/upstream runtime ``x`` would now be
        # hydrated instead.)
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                """
def add(y):
    return x + y
""".strip(),
            )
            .add_cell("c2", "result = add(2)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")

                assert result["type"] == "cell_error"
                assert "cannot be shared across cells yet" in result["payload"]["error"]
                assert "function `add`" in result["payload"]["error"]
                assert "x" in result["payload"]["error"]

                state = ws.sync()
                c1 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c1")
                c2 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c2")
                assert c1["status"] == "error"
                assert c2["status"] == "idle"

    def test_top_level_lambda_error_surfaces_over_websocket(self, setup):
        client, tmp = setup
        nb = (
            NotebookBuilder(tmp)
            .add_cell("c1", "add = lambda y: y + 1")
            .add_cell("c2", "result = add(2)", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")

                assert result["type"] == "cell_error"
                assert "cannot be shared across cells yet" in result["payload"]["error"]
                assert (
                    "top-level lambdas are not shareable across cells" in result["payload"]["error"]
                )

                state = ws.sync()
                c1 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c1")
                c2 = next(cell for cell in state["payload"]["cells"] if cell["id"] == "c2")
                assert c1["status"] == "error"
                assert c2["status"] == "idle"

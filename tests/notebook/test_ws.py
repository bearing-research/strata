"""Tests for WebSocket notebook execution."""

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from strata.notebook.parser import parse_notebook
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

_MINIMAL_PNG_LITERAL = (
    'b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01'
    "\\x08\\x04\\x00\\x00\\x00\\xb5\\x1c\\x0c\\x02\\x00\\x00\\x00\\x0bIDATx\\xdac\\xfc\\xff"
    '\\x1f\\x00\\x03\\x03\\x02\\x00\\xef\\x9b\\xe0M\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"'
)
_MARKDOWN_LITERAL = '"# Title\\n\\nRendered over websocket."'


@pytest.fixture
def temp_notebook():
    """Create a temporary notebook with simple cells."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create notebook structure
        notebook_dir = create_notebook(tmpdir, "test_notebook")

        # Create simple cells
        cells_data = [
            ("root", "x = 1"),
            ("middle", "y = x + 1"),
            ("leaf", "z = y + 1"),
        ]

        for cell_id, source in cells_data:
            add_cell_to_notebook(notebook_dir, cell_id)
            write_cell(notebook_dir, cell_id, source)

        # Parse the notebook
        notebook_state = parse_notebook(notebook_dir)

        yield notebook_dir, notebook_state


@pytest.fixture
def app():
    """Create FastAPI test app with notebook routes."""
    from fastapi import FastAPI

    from strata.notebook.routes import router as notebook_router
    from strata.notebook.ws import router as notebook_ws_router

    app = FastAPI()
    app.include_router(notebook_router)
    app.include_router(notebook_ws_router)

    return app


@pytest.fixture
def client(app):
    """Create TestClient for the app."""
    return TestClient(app)


def test_notebook_sync(client, temp_notebook, app):
    """Test notebook_sync message returns full notebook state."""
    notebook_dir, notebook_state = temp_notebook

    # Register notebook with session manager
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    # Connect WebSocket
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        # Send notebook_sync
        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )

        # Receive notebook_state
        response = websocket.receive_json()
        assert response["type"] == "notebook_state"
        assert "payload" in response
        state = response["payload"]
        assert "id" in state
        assert "cells" in state
        assert "dag" in state


def test_serialize_dag_edges_uses_frontend_field_names(temp_notebook):
    """Edges sent to the frontend must use ``from_cell_id`` / ``to_cell_id``.

    Regression for a field-name drift in ``broadcast_notebook_sync`` (used by
    the agent's ``edit_cell`` / ``run_cell`` flows) where edges were emitted
    as ``{"from": ..., "to": ...}``. The frontend's ``applyBackendDag`` keys
    off ``from_cell_id`` / ``to_cell_id``, so every edge silently failed the
    cell lookup and the DAG view rendered as disconnected nodes after every
    agent edit until the user hard-refreshed.
    """
    from strata.notebook.routes import get_session_manager

    notebook_dir, _ = temp_notebook
    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    edges = session.dag.serialize_edges() if session.dag else []
    assert edges, "fixture defines x->y->z, expected at least one DAG edge"
    for edge in edges:
        assert set(edge) == {"from_cell_id", "to_cell_id", "variable"}, (
            f"DAG edge field names must match the frontend contract; got {edge}"
        )


def test_broadcast_notebook_sync_emits_correct_edge_field_names(temp_notebook):
    """End-to-end: ``broadcast_notebook_sync`` must put correctly-keyed edges on the wire.

    Same regression as above but exercises the full agent-broadcast path: a
    fake WS captures the message and we inspect the payload directly. If
    this test ever flips back to ``from`` / ``to``, the agent edit flow has
    drifted again and the DAG view will silently break.
    """
    import asyncio
    import json

    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_connections, broadcast_notebook_sync

    notebook_dir, _ = temp_notebook
    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    captured: list[dict] = []

    class FakeWebSocket:
        async def send_text(self, text: str) -> None:
            captured.append(json.loads(text))

    fake = cast(WebSocket, FakeWebSocket())
    _notebook_connections.setdefault(session.id, []).append(fake)
    try:
        asyncio.run(broadcast_notebook_sync(session.id, session))
    finally:
        _notebook_connections.get(session.id, []).remove(fake)

    assert captured, "broadcast_notebook_sync did not deliver a message"
    msg = captured[-1]
    assert msg["type"] == "notebook_state"
    edges = msg["payload"]["dag"]["edges"]
    assert edges, "fixture defines x->y->z, expected at least one DAG edge"
    for edge in edges:
        assert set(edge) == {"from_cell_id", "to_cell_id", "variable"}, (
            f"broadcast_notebook_sync emitted bad edge shape: {edge}"
        )


def test_notebook_sync_includes_causality_and_staleness(client, temp_notebook, app):
    """notebook_sync should return enriched cell state, not just bare DAG fields."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("root", "x = 1")).success
        root = next(c for c in session.notebook_state.cells if c.id == "root")
        root.source = "x = 2"
        write_cell(notebook_dir, "root", "x = 2")
        session.re_analyze_cell("root")
        session.compute_staleness()

    asyncio.run(_prime())

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "notebook_state"
        state = response["payload"]
        root = next(cell for cell in state["cells"] if cell["id"] == "root")

        assert root["status"] == "idle"
        assert "staleness_reasons" in root
        assert root["causality"]["reason"] == "self"


def test_notebook_sync_includes_remote_execution_metadata(
    client,
    temp_notebook,
    app,
    notebook_executor_server,
    notebook_build_server,
):
    """Notebook sync should retain remote execution metadata from the live session."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": {
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        }
    ]

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config={
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        )
    ]
    root = next(c for c in session.notebook_state.cells if c.id == "root")
    root.worker = "gpu-http-signed"

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("root", "x = 1")).success

    asyncio.run(_prime())

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "notebook_state"
        state = response["payload"]
        root = next(cell for cell in state["cells"] if cell["id"] == "root")

        assert root["execution_method"] == "executor"
        assert root["remote_worker"] == "gpu-http-signed"
        assert root["remote_transport"] == "signed"
        assert isinstance(root["remote_build_id"], str)
        assert root["remote_build_state"] == "ready"
        assert root["remote_error_code"] is None


def test_cell_execute_no_cascade(client, temp_notebook, app):
    """Test cell_execute on a root cell (no cascade needed)."""
    notebook_dir, notebook_state = temp_notebook

    # Register notebook
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    # Connect WebSocket
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        # Find root cell
        root_cell = next((c for c in session.notebook_state.cells if c.upstream_ids == []), None)
        if not root_cell:
            # If no root (no DAG), just pick first cell
            root_cell = session.notebook_state.cells[0]

        # Send cell_execute
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        # Should receive cell_status(running)
        response = websocket.receive_json()
        assert response["type"] == "cell_status"
        assert response["payload"]["status"] == "running"

        # Should receive cell_output or cell_error
        response = websocket.receive_json()
        assert response["type"] in ["cell_output", "cell_error"]


def test_cell_execute_emits_explicit_display_payload(client, temp_notebook, app):
    """Image-like last-expression results should be sent in the dedicated display payload."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "root",
        f"""
class Display:
    def _repr_png_(self):
        return {_MINIMAL_PNG_LITERAL}

Display()
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "root")

        assert output_message["type"] == "cell_output"
        assert output_message["payload"]["display"]["content_type"] == "image/png"
        assert output_message["payload"]["display"]["inline_data_url"].startswith(
            "data:image/png;base64,"
        )
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_explicit_markdown_display_payload(client, temp_notebook, app):
    """Markdown last-expression results should be sent in the dedicated display payload."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "root",
        f"""
class Display:
    def _repr_markdown_(self):
        return {_MARKDOWN_LITERAL}

Display()
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "root")

        assert output_message["type"] == "cell_output"
        assert output_message["payload"]["display"]["content_type"] == "text/markdown"
        assert (
            output_message["payload"]["display"]["markdown_text"]
            == "# Title\n\nRendered over websocket."
        )
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_display_side_effect_payload(client, temp_notebook, app):
    """display(...) side effects should be surfaced through the websocket display payload."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# Side effect\\n\\nVia websocket."))
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "root")

        assert output_message["type"] == "cell_output"
        assert output_message["payload"]["display"]["content_type"] == "text/markdown"
        assert (
            output_message["payload"]["display"]["markdown_text"]
            == "# Side effect\n\nVia websocket."
        )
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_multiple_display_payloads_in_order(client, temp_notebook, app):
    """Ordered visible outputs should be sent together, with the last one preserved as display."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# First"))
42
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "root")

        assert output_message["type"] == "cell_output"
        assert len(output_message["payload"]["displays"]) == 2
        assert output_message["payload"]["displays"][0]["content_type"] == "text/markdown"
        assert output_message["payload"]["displays"][0]["markdown_text"] == "# First"
        assert output_message["payload"]["displays"][1]["content_type"] == "json/object"
        assert output_message["payload"]["displays"][1]["preview"] == 42
        assert output_message["payload"]["display"]["content_type"] == "json/object"
        assert output_message["payload"]["display"]["preview"] == 42
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_refreshes_downstream_staleness(client, temp_notebook, app):
    """Successful execution should immediately invalidate downstream cell state."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("root", "x = 1")).success
        assert (await executor.execute_cell("middle", "y = x + 1")).success
        assert (await executor.execute_cell("leaf", "z = y + 1")).success
        session.compute_staleness()

    asyncio.run(_prime())

    root = next(c for c in session.notebook_state.cells if c.id == "root")
    root.source = "x = 2"
    write_cell(notebook_dir, "root", "x = 2")
    session.re_analyze_cell("root")

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        messages = [websocket.receive_json() for _ in range(5)]
        status_updates = [msg["payload"] for msg in messages if msg["type"] == "cell_status"]

        assert any(p["cell_id"] == "root" and p["status"] == "ready" for p in status_updates)
        assert any(p["cell_id"] == "middle" and p["status"] == "idle" for p in status_updates)
        assert any(p["cell_id"] == "leaf" and p["status"] == "idle" for p in status_updates)


def test_cell_execute_surfaces_module_export_error(client, temp_notebook, app):
    """Unsupported cross-cell code export should surface as a direct cell error."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    # ``x = len([])`` is a non-literal runtime assignment; plain literal
    # constants (``x = 1``) would now export fine alongside the def.
    write_cell(
        notebook_dir,
        "root",
        "x = len([])\n\ndef add(y):\n    return x + y\n",
    )
    write_cell(notebook_dir, "middle", "result = add(2)")

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-31T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["status"] == "running"

        error_msg = websocket.receive_json()
        assert error_msg["type"] == "cell_error"
        assert "cannot be shared across cells yet" in error_msg["payload"]["error"]
        # The slicer pinpoints the unresolved free var and the symbol
        # that depends on it.
        assert "function `add`" in error_msg["payload"]["error"]
        assert "x" in error_msg["payload"]["error"]

        terminal = websocket.receive_json()
        assert terminal["type"] == "cell_status"
        assert terminal["payload"]["status"] == "error"


def test_cell_execute_surfaces_module_export_lambda_error(client, temp_notebook, app):
    """The WS path should surface top-level lambda export errors clearly."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(notebook_dir, "root", "add = lambda y: y + 1\n")
    write_cell(notebook_dir, "middle", "result = add(2)")

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-31T00:00:00Z",
                "payload": {"cell_id": "root"},
            }
        )

        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["status"] == "running"

        error_msg = websocket.receive_json()
        assert error_msg["type"] == "cell_error"
        assert "cannot be shared across cells yet" in error_msg["payload"]["error"]
        assert "top-level lambdas are not shareable across cells" in error_msg["payload"]["error"]

        terminal = websocket.receive_json()
        assert terminal["type"] == "cell_status"
        assert terminal["payload"]["status"] == "error"


def test_cell_execute_uses_warm_pool_when_available(client, temp_notebook, app, monkeypatch):
    """Test the WebSocket path is wired to use the session warm pool."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.pool import PooledCellExecutor, WarmProcessPool
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.warm_pool = cast(WarmProcessPool, object())

    root_cell = next((c for c in session.notebook_state.cells if c.upstream_ids == []), None)
    if not root_cell:
        root_cell = session.notebook_state.cells[0]

    warm_calls = 0

    async def fake_execute_with_pool(pool, manifest_path, notebook_dir, timeout_seconds=30):
        nonlocal warm_calls
        warm_calls += 1
        return {
            "success": True,
            "variables": {
                "x": {
                    "content_type": "json/object",
                    "preview": 1,
                    "bytes": 1,
                    "file": "x.json",
                }
            },
            "stdout": "",
            "stderr": "",
            "mutation_warnings": [],
        }

    def fake_store_outputs(
        self,
        cell_id,
        output_dir,
        provenance_hash,
        input_hashes,
        *,
        source_hash="",
        env_hash="",
    ):
        return True

    monkeypatch.setattr(
        PooledCellExecutor,
        "execute_with_pool",
        staticmethod(fake_execute_with_pool),
    )
    monkeypatch.setattr(CellExecutor, "_store_outputs", fake_store_outputs)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        output_message = None
        final_status = None
        for _ in range(10):
            response = websocket.receive_json()
            if response["type"] in ["cell_output", "cell_error"]:
                output_message = response
            if response["type"] == "cell_status" and response["payload"]["status"] in [
                "ready",
                "error",
            ]:
                final_status = response
                break

        assert output_message is not None
        assert final_status is not None
        assert output_message["type"] == "cell_output"
        assert output_message["payload"]["execution_method"] == "warm"
        assert warm_calls == 1


def _receive_execution_terminal_messages(websocket, cell_id: str) -> tuple[dict, dict]:
    """Collect the output/error message and terminal status for one execution."""
    output_message = None
    terminal_status = None

    for _ in range(20):
        response = websocket.receive_json()
        if (
            response["type"] in ["cell_output", "cell_error"]
            and response.get("payload", {}).get("cell_id") == cell_id
        ):
            output_message = response
        if (
            response["type"] == "cell_status"
            and response["payload"].get("cell_id") == cell_id
            and response["payload"]["status"] in ["ready", "error"]
        ):
            terminal_status = response
            break

    assert output_message is not None
    assert terminal_status is not None
    return output_message, terminal_status


def test_notebook_run_all_emits_multiple_display_payloads_in_order(client, temp_notebook, app):
    """Run-all should preserve ordered display payloads on the websocket path."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# First"))
42
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "notebook_run_all",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "root")

        assert output_message["type"] == "cell_output"
        assert len(output_message["payload"]["displays"]) == 2
        assert output_message["payload"]["displays"][0]["content_type"] == "text/markdown"
        assert output_message["payload"]["displays"][0]["markdown_text"] == "# First"
        assert output_message["payload"]["displays"][1]["content_type"] == "json/object"
        assert output_message["payload"]["displays"][1]["preview"] == 42
        assert output_message["payload"]["display"]["content_type"] == "json/object"
        assert output_message["payload"]["display"]["preview"] == 42
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_cascade_emits_multiple_display_payloads_in_order(client, temp_notebook, app):
    """Cascade execution should preserve ordered display payloads for the target cell."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    write_cell(
        notebook_dir,
        "leaf",
        """
display(Markdown("# First"))
y + 1
""",
    )

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": "leaf"},
            }
        )

        prompt = websocket.receive_json()
        assert prompt["type"] == "cascade_prompt"

        websocket.send_json(
            {
                "type": "cell_execute_cascade",
                "seq": 2,
                "ts": "2026-03-23T00:00:01Z",
                "payload": {"cell_id": "leaf", "plan_id": prompt["payload"]["plan_id"]},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(websocket, "leaf")

        assert output_message["type"] == "cell_output"
        assert len(output_message["payload"]["displays"]) == 2
        assert output_message["payload"]["displays"][0]["content_type"] == "text/markdown"
        assert output_message["payload"]["displays"][0]["markdown_text"] == "# First"
        assert output_message["payload"]["displays"][1]["content_type"] == "json/object"
        assert output_message["payload"]["displays"][1]["preview"] == 3
        assert output_message["payload"]["display"]["content_type"] == "json/object"
        assert output_message["payload"]["display"]["preview"] == 3
        assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_blocked_when_environment_runtime_is_unavailable(client, temp_notebook, app):
    """Execution should be blocked when no notebook runtime is available after bootstrap failure."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    cell_id = session.notebook_state.cells[0].id
    session.environment_job = None
    session.venv_python = None
    session.environment_interpreter_source = "unknown"
    session.environment_sync_state = "failed"
    session.environment_sync_error = "Failed to start notebook environment initialization: boom"
    session.environment_sync_notice = None

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "error"
        assert response["payload"]["code"] == "ENVIRONMENT_BUSY"
        assert "environment" in response["payload"]["error"].lower()


def test_environment_job_submission_rejects_execution_already_accepted(monkeypatch, temp_notebook):
    """Execution acceptance should block env jobs before the task starts."""
    notebook_dir, _ = temp_notebook

    from strata.notebook import ws as notebook_ws
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    execution_state = notebook_ws._ensure_execution_state(session.id)
    entered_schedule = asyncio.Event()
    release_schedule = asyncio.Event()

    async def _gated_schedule(
        websocket,
        execution_state_arg,
        notebook_id,
        requested_cell,
        seq,
        operation_factory,
    ):
        del websocket, notebook_id, requested_cell, seq, operation_factory
        assert execution_state_arg is execution_state
        entered_schedule.set()
        await release_schedule.wait()
        return True

    class _FakeWebSocket:
        async def send_text(self, _text: str) -> None:
            return None

    monkeypatch.setattr(notebook_ws, "_schedule_execution", _gated_schedule)

    async def _noop_environment_job(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(session, "_run_environment_job", _noop_environment_job)

    async def _exercise() -> None:
        execute_task = asyncio.create_task(
            notebook_ws._handle_cell_execute(
                cast(WebSocket, _FakeWebSocket()),
                session,
                {"cell_id": "root"},
                execution_state,
                session.id,
            )
        )
        await asyncio.wait_for(entered_schedule.wait(), timeout=1)
        try:
            with pytest.raises(RuntimeError):
                await session.submit_environment_job(action="sync")
        finally:
            release_schedule.set()
            await execute_task

    asyncio.run(_exercise())


def test_ws_execute_supports_http_executor_worker(
    client, temp_notebook, app, notebook_executor_server
):
    """The live WebSocket execution path should support HTTP notebook workers."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-a100",
            config={"url": notebook_executor_server["execute_url"]},
        )
    ]
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http"

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(
            websocket, root_cell.id
        )

        assert output_message["type"] == "cell_output"
        assert output_message["payload"]["execution_method"] == "executor"
        assert output_message["payload"]["remote_worker"] == "gpu-http"
        assert output_message["payload"]["remote_transport"] == "direct"
        assert output_message["payload"]["outputs"]["x"]["preview"] == 1
        assert terminal_status["payload"]["status"] == "ready"


def test_ws_execute_supports_signed_http_executor_worker(
    client,
    temp_notebook,
    app,
    notebook_executor_server,
    notebook_build_server,
):
    """The live WebSocket path should support signed remote notebook workers."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": {
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        }
    ]

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config={
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        )
    ]
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        first_output, first_terminal = _receive_execution_terminal_messages(websocket, root_cell.id)

        assert first_output["type"] == "cell_output"
        assert first_output["payload"]["execution_method"] == "executor"
        assert first_output["payload"]["remote_worker"] == "gpu-http-signed"
        assert first_output["payload"]["remote_transport"] == "signed"
        assert isinstance(first_output["payload"]["remote_build_id"], str)
        assert first_output["payload"]["outputs"]["x"]["preview"] == 1
        assert first_terminal["payload"]["status"] == "ready"

        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 2,
                "ts": "2026-03-30T00:00:01Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        second_output, second_terminal = _receive_execution_terminal_messages(
            websocket, root_cell.id
        )

        assert second_output["type"] == "cell_output"
        assert second_output["payload"]["execution_method"] == "cached"
        assert second_output["payload"]["remote_worker"] == "gpu-http-signed"
        assert second_output["payload"]["remote_transport"] == "signed"
        assert "remote_build_id" not in second_output["payload"]
        assert second_terminal["payload"]["status"] == "ready"


def test_ws_execute_supports_signed_http_executor_worker_with_class_instances(
    client,
    app,
    notebook_executor_server,
    notebook_build_server,
):
    """The live WS path should preserve exported class instances over signed transport."""
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "signed_class_instances")
        add_cell_to_notebook(notebook_dir, "cell1")
        add_cell_to_notebook(notebook_dir, "cell2", "cell1")
        add_cell_to_notebook(notebook_dir, "cell3", "cell2")
        write_cell(
            notebook_dir,
            "cell1",
            """
class Person:
    name = "John"
    age = 20

    def __str__(self):
        return f"{self.name}:{self.age}"
""".strip(),
        )
        write_cell(notebook_dir, "cell2", "p = Person()")
        write_cell(notebook_dir, "cell3", "rendered = str(p)")

        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": {
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            }
        ]

        session_manager = get_session_manager()
        session = session_manager.open_notebook(notebook_dir)
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu-http-signed",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-http-signed-a100",
                config={
                    "url": notebook_executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": notebook_build_server["base_url"],
                },
            )
        ]
        for cell in session.notebook_state.cells:
            cell.worker = "gpu-http-signed"

        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
            for cell_id in ("cell1", "cell2", "cell3"):
                websocket.send_json(
                    {
                        "type": "cell_execute",
                        "seq": 1,
                        "ts": "2026-03-31T00:00:00Z",
                        "payload": {"cell_id": cell_id},
                    }
                )
                output_message, terminal_status = _receive_execution_terminal_messages(
                    websocket, cell_id
                )
                assert output_message["type"] == "cell_output"
                assert output_message["payload"]["execution_method"] == "executor"
                assert output_message["payload"]["remote_worker"] == "gpu-http-signed"
                assert output_message["payload"]["remote_transport"] == "signed"
                assert output_message["payload"]["remote_build_state"] == "ready"
                assert terminal_status["payload"]["status"] == "ready"

            websocket.send_json(
                {
                    "type": "notebook_sync",
                    "seq": 2,
                    "ts": "2026-03-31T00:00:01Z",
                    "payload": {},
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "notebook_state"
            state = response["payload"]
            cell2 = next(cell for cell in state["cells"] if cell["id"] == "cell2")
            cell3 = next(cell for cell in state["cells"] if cell["id"] == "cell3")

            assert "p" in cell2["artifact_uris"]
            assert cell2["remote_transport"] == "signed"
            assert cell2["remote_build_state"] == "ready"
            assert cell2["status"] == "ready"
            assert cell3["remote_transport"] == "signed"
            assert cell3["remote_build_state"] == "ready"
            assert cell3["status"] == "ready"


def test_ws_execute_reports_unavailable_http_executor_worker(client, temp_notebook, app):
    """The live WS path should surface unreachable HTTP executor workers."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-dead",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-dead-a100",
            config={"url": "http://127.0.0.1:9/v1/execute"},
        )
    ]
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-dead"

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(
            websocket, root_cell.id
        )

        assert output_message["type"] == "cell_error"
        assert "Remote executor request failed" in output_message["payload"]["error"]
        assert terminal_status["payload"]["status"] == "error"


def test_ws_execute_reports_signed_finalize_failure(
    client,
    temp_notebook,
    app,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """The live WS path should surface signed transport finalize failures."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager
    from strata.transforms.signed_urls import (
        generate_build_manifest as real_generate_build_manifest,
    )

    class _BadFinalizeManifest:
        def __init__(self, manifest):
            self._manifest = manifest

        def to_dict(self):
            data = self._manifest.to_dict()
            data["finalize_url"] = f"{data['finalize_url']}/missing-finalize"
            return data

    def fake_generate_build_manifest(*args, **kwargs):
        return _BadFinalizeManifest(real_generate_build_manifest(*args, **kwargs))

    monkeypatch.setattr(
        "strata.notebook.executor.generate_build_manifest",
        fake_generate_build_manifest,
    )

    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": {
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        }
    ]

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config={
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        )
    ]
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        output_message, terminal_status = _receive_execution_terminal_messages(
            websocket, root_cell.id
        )

        assert output_message["type"] == "cell_error"
        assert "Failed to finalize notebook bundle build" in output_message["payload"]["error"]
        assert output_message["payload"]["remote_worker"] == "gpu-http-signed"
        assert output_message["payload"]["remote_transport"] == "signed"
        assert isinstance(output_message["payload"]["remote_build_id"], str)
        assert output_message["payload"]["remote_build_state"] == "failed"
        assert output_message["payload"]["remote_error_code"] == "FINALIZE_FAILED"
        assert terminal_status["payload"]["status"] == "error"


def test_ws_cancelled_signed_http_executor_marks_build_failed(
    client,
    temp_notebook,
    app,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """Cancelling signed remote execution over WS should fail the build cleanly."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.routes import get_session_manager

    started = threading.Event()

    async def _slow_run_harness(
        harness_path: Path,
        manifest_path: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        del harness_path, manifest_path, timeout_seconds
        started.set()
        await asyncio.sleep(60)
        return {
            "success": True,
            "variables": {
                "x": {
                    "content_type": "json/object",
                    "file": "x.json",
                    "preview": 1,
                }
            },
            "stdout": "",
            "stderr": "",
            "mutation_warnings": [],
        }

    monkeypatch.setattr(
        "strata.notebook.remote_executor._run_harness",
        _slow_run_harness,
    )

    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": {
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        }
    ]

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config={
                "url": notebook_executor_server["execute_url"],
                "transport": "signed",
                "strata_url": notebook_build_server["base_url"],
            },
        )
    ]
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-30T00:00:00Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["cell_id"] == root_cell.id
        assert running["payload"]["status"] == "running"
        assert started.wait(timeout=2.0)

        websocket.send_json(
            {
                "type": "cell_cancel",
                "seq": 2,
                "ts": "2026-03-30T00:00:01Z",
                "payload": {"cell_id": root_cell.id},
            }
        )

        idle_message = None
        for _ in range(10):
            response = websocket.receive_json()
            if (
                response["type"] == "cell_status"
                and response["payload"].get("cell_id") == root_cell.id
                and response["payload"]["status"] == "idle"
            ):
                idle_message = response
                break

        assert idle_message is not None

    for _ in range(20):
        stats = notebook_build_server["build_store"].get_stats()
        if stats["pending"] == 0 and stats["building"] == 0:
            break
        time.sleep(0.05)

    stats = notebook_build_server["build_store"].get_stats()
    assert stats["failed"] == 1
    assert stats["pending"] == 0
    assert stats["building"] == 0


def test_cascade_prompt_is_sent_only_to_requesting_websocket(client, temp_notebook, app):
    """A cascade prompt should not fan out to other clients on the notebook."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws1:
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws2:
            ws1.send_json(
                {
                    "type": "cell_execute",
                    "seq": 1,
                    "ts": "2026-03-23T00:00:00Z",
                    "payload": {"cell_id": "middle"},
                }
            )

            response = ws1.receive_json()
            assert response["type"] == "cascade_prompt"
            assert response["payload"]["cell_id"] == "middle"

            with pytest.raises(Exception):
                ws2.receive_json(timeout=0.1)


def test_impact_preview_is_sent_only_to_requesting_websocket(client, temp_notebook, app):
    """Impact preview responses should stay scoped to the requesting client."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws1:
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws2:
            ws1.send_json(
                {
                    "type": "impact_preview_request",
                    "seq": 1,
                    "ts": "2026-03-23T00:00:00Z",
                    "payload": {"cell_id": "middle"},
                }
            )

            response = ws1.receive_json()
            assert response["type"] == "impact_preview"
            assert response["payload"]["target_cell_id"] == "middle"

            with pytest.raises(Exception):
                ws2.receive_json(timeout=0.1)


def test_inspect_repl_round_trip(client, temp_notebook, app):
    """Test the live inspect REPL path over WebSocket."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    middle_cell = next((c for c in session.notebook_state.cells if "x" in c.references), None)
    assert middle_cell is not None

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "inspect_open",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": middle_cell.id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "open"
        assert response["payload"]["ok"] is True

        websocket.send_json(
            {
                "type": "inspect_eval",
                "seq": 2,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": middle_cell.id, "expr": "x + 1"},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "eval"
        assert response["payload"]["ok"] is True
        assert response["payload"]["result"] == "2"
        assert response["payload"]["type"] == "int"

        websocket.send_json(
            {
                "type": "inspect_close",
                "seq": 3,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": middle_cell.id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "close"
        assert response["payload"]["ok"] is True


def test_active_websocket_session_is_not_evicted(client, temp_notebook, app):
    """TTL eviction should skip sessions that still have connected sockets."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    session.last_accessed = time.time() - session_manager.SESSION_TTL_SECONDS - 60

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        session_manager._evict_stale()
        assert session.id in session_manager.list_sessions()

        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )
        websocket.receive_json()
        assert session.last_accessed > time.time() - 5


def test_inspect_sessions_closed_when_last_websocket_disconnects(
    client, temp_notebook, app, monkeypatch
):
    """Disconnecting the last socket should close notebook inspect sessions."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.inspect_repl import InspectManager
    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_inspect_managers

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    close_all_calls = 0

    async def fake_open_session(self, cell_id, notebook_session):
        return SimpleNamespace(ready=True), "ready"

    async def fake_close_all(self):
        nonlocal close_all_calls
        close_all_calls += 1

    monkeypatch.setattr(InspectManager, "open_session", fake_open_session)
    monkeypatch.setattr(InspectManager, "close_all", fake_close_all)

    middle_cell = next((c for c in session.notebook_state.cells if "x" in c.references), None)
    assert middle_cell is not None

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "inspect_open",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": middle_cell.id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["ok"] is True

    assert close_all_calls == 1
    assert session.id not in _notebook_inspect_managers


def test_grace_window_preserves_inspect_state_on_reconnect(client, temp_notebook, app, monkeypatch):
    """A disconnect-then-reconnect within the grace window keeps state.

    Without the grace window, the inspect manager would be dropped the
    moment the WS context exits. The TUI's target audience (tmux / SSH
    sessions) would lose any open inspect REPL on a network blip — bad
    UX for exactly the users the reconnect window exists for.
    """
    notebook_dir, _ = temp_notebook

    from strata.notebook.inspect_repl import InspectManager
    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_inspect_managers

    # Long enough window that the test thread can't accidentally cross
    # it; the reconnect cancels the task explicitly.
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    close_all_calls = 0

    async def fake_open_session(self, cell_id, notebook_session):
        return SimpleNamespace(ready=True), "ready"

    async def fake_close_all(self):
        nonlocal close_all_calls
        close_all_calls += 1

    monkeypatch.setattr(InspectManager, "open_session", fake_open_session)
    monkeypatch.setattr(InspectManager, "close_all", fake_close_all)

    cell = next((c for c in session.notebook_state.cells if "x" in c.references), None)
    assert cell is not None

    # First connection opens an inspect REPL.
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "inspect_open",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell.id},
            }
        )
        response = websocket.receive_json()
        assert response["type"] == "inspect_result"

    # Inside the grace window: state must be preserved.
    assert session.id in _notebook_inspect_managers
    assert not close_all_calls

    # Reconnect cancels the pending teardown and resumes.
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )
        websocket.receive_json()
        # Still alive.
        assert session.id in _notebook_inspect_managers
        assert not close_all_calls


def test_grace_window_expires_drops_state(client, temp_notebook, app, monkeypatch):
    """When the grace window elapses with no reconnect, state is dropped."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.inspect_repl import InspectManager
    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_grace_tasks, _notebook_inspect_managers

    # Short window: long enough that the test thread observes the
    # "pending teardown" state, short enough that polling for completion
    # doesn't drag the suite.
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 0.1)

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    close_all_calls = 0

    async def fake_open_session(self, cell_id, notebook_session):
        return SimpleNamespace(ready=True), "ready"

    async def fake_close_all(self):
        nonlocal close_all_calls
        close_all_calls += 1

    monkeypatch.setattr(InspectManager, "open_session", fake_open_session)
    monkeypatch.setattr(InspectManager, "close_all", fake_close_all)

    cell = next((c for c in session.notebook_state.cells if "x" in c.references), None)
    assert cell is not None

    # Hold the TestClient open across requests so the server's event loop
    # outlives the WS context; without this, the grace task scheduled at
    # disconnect would die with the portal.
    with client:
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
            websocket.send_json(
                {
                    "type": "inspect_open",
                    "seq": 1,
                    "ts": "2026-03-23T00:00:00Z",
                    "payload": {"cell_id": cell.id},
                }
            )
            websocket.receive_json()

        # A grace task was scheduled — state is still alive.
        assert session.id in _notebook_grace_tasks
        assert session.id in _notebook_inspect_managers

        # Poll until the task runs in the server's background event loop.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if close_all_calls and session.id not in _notebook_inspect_managers:
                break
            time.sleep(0.02)

    assert close_all_calls == 1
    assert session.id not in _notebook_inspect_managers
    assert session.id not in _notebook_grace_tasks


def test_grace_window_preserves_active_execution_on_reconnect(
    client, temp_notebook, app, monkeypatch
):
    """A running execution task survives a disconnect-reconnect cycle within the window.

    This is the load-bearing #42 contract: tmux detach during a long
    cell, reconnect within the window, find the cell still running.
    The inspect-state companion test covers a different cleanup hook;
    this one targets ``NotebookExecutionState.execution_task`` which is
    the actual mechanism behind cancel-on-disconnect.
    """
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _ensure_execution_state, _notebook_execution_state

    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    captured: dict[str, asyncio.Task[None]] = {}

    async def long_sleep() -> None:
        await asyncio.sleep(60)

    async def inject_task() -> asyncio.Task[None]:
        state = _ensure_execution_state(session.id)
        task = asyncio.create_task(long_sleep())
        state.execution_task = task
        return task

    with client:
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws:
            ws.send_json(
                {
                    "type": "notebook_sync",
                    "seq": 1,
                    "ts": "2026-05-25T00:00:00Z",
                    "payload": {},
                }
            )
            ws.receive_json()
            captured["task"] = client.portal.call(inject_task)

        # Inside grace window: task is still alive.
        assert not captured["task"].done(), (
            "execution task was cancelled before the grace window expired"
        )
        assert _notebook_execution_state.get(session.id) is not None

        # Reconnect cancels the pending teardown — task stays alive.
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws:
            ws.send_json(
                {
                    "type": "notebook_sync",
                    "seq": 1,
                    "ts": "2026-05-25T00:00:00Z",
                    "payload": {},
                }
            )
            ws.receive_json()
            assert not captured["task"].done()
            assert _notebook_execution_state.get(session.id) is not None

        # Test cleanup — cancel the injected long-running task so the
        # next test's loop doesn't inherit it.
        async def _cancel_injected(task: asyncio.Task[None]) -> None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Cancellation is the expected path — the test injected a
                # bare ``asyncio.sleep(60)`` whose only normal exit is
                # ``CancelledError``. Any other exception should surface so
                # we notice if cleanup behavior regresses.
                pass

        client.portal.call(_cancel_injected, captured["task"])


def test_grace_window_expiry_cancels_active_execution(client, temp_notebook, app, monkeypatch):
    """Past the grace window with no reconnect, the running execution task is cancelled."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import (
        _ensure_execution_state,
        _notebook_execution_state,
        _notebook_grace_tasks,
    )

    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 0.1)

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    captured: dict[str, asyncio.Task[None]] = {}

    async def long_sleep() -> None:
        await asyncio.sleep(60)

    async def inject_task() -> asyncio.Task[None]:
        state = _ensure_execution_state(session.id)
        task = asyncio.create_task(long_sleep())
        state.execution_task = task
        return task

    with client:
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws:
            ws.send_json(
                {
                    "type": "notebook_sync",
                    "seq": 1,
                    "ts": "2026-05-25T00:00:00Z",
                    "payload": {},
                }
            )
            ws.receive_json()
            captured["task"] = client.portal.call(inject_task)

        assert not captured["task"].done()
        assert session.id in _notebook_grace_tasks

        # Poll for the grace task to fire and cancel the execution.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if captured["task"].done() and session.id not in _notebook_execution_state:
                break
            time.sleep(0.02)

        assert captured["task"].done()
        assert captured["task"].cancelled()
        assert session.id not in _notebook_execution_state
        assert session.id not in _notebook_grace_tasks


def test_cell_source_update(client, temp_notebook, app):
    """Test cell_source_update triggers DAG recomputation."""
    notebook_dir, notebook_state = temp_notebook

    # Register notebook
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    # Connect WebSocket
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        # Send cell_source_update
        cell_id = session.notebook_state.cells[0].id
        websocket.send_json(
            {
                "type": "cell_source_update",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id, "source": "x = 2\ny = 3"},
            }
        )

        # Should receive dag_update
        response = websocket.receive_json()
        assert response["type"] == "dag_update"
        assert "edges" in response["payload"]
        assert "topological_order" in response["payload"]

        # May receive cell_status updates
        while True:
            try:
                response = websocket.receive_json(timeout=0.1)
            except Exception:
                break
            if response["type"] == "cell_status":
                # Cell status updates are expected
                assert "status" in response["payload"]


def test_cell_cancel(client, temp_notebook, app):
    """Test cell_cancel stops execution."""
    notebook_dir, notebook_state = temp_notebook

    # Register notebook
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    # Connect WebSocket
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        cell_id = session.notebook_state.cells[0].id

        # Send cell_cancel
        websocket.send_json(
            {
                "type": "cell_cancel",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        # Should receive cell_status(idle)
        response = websocket.receive_json()
        assert response["type"] == "cell_status"
        assert response["payload"]["status"] == "idle"


def test_cell_cancel_interrupts_running_execution_on_same_websocket(
    client, temp_notebook, app, monkeypatch
):
    """A single WebSocket can cancel its own in-flight execution."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    cell_id = session.notebook_state.cells[0].id

    cancelled = threading.Event()

    async def fake_execute_cell(self, cell_id: str, source: str, timeout_seconds: float = 30):
        del self
        del cell_id
        del source
        del timeout_seconds
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(CellExecutor, "execute_cell", fake_execute_cell)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "cell_status"
        assert response["payload"]["cell_id"] == cell_id
        assert response["payload"]["status"] == "running"

        websocket.send_json(
            {
                "type": "cell_cancel",
                "seq": 2,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "cell_status"
        assert response["payload"]["cell_id"] == cell_id
        assert response["payload"]["status"] == "idle"

    assert cancelled.is_set()


def test_stale_cell_cancel_does_not_clobber_ready_state(client, temp_notebook, app):
    """A late cancel should not rewrite a completed cell back to idle."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    cell_id = session.notebook_state.cells[0].id

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        while True:
            msg = websocket.receive_json()
            if (
                msg["type"] == "cell_status"
                and msg["payload"]["cell_id"] == cell_id
                and msg["payload"]["status"] in ["ready", "error"]
            ):
                assert msg["payload"]["status"] == "ready"
                break

        websocket.send_json(
            {
                "type": "cell_cancel",
                "seq": 2,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )
        websocket.send_json(
            {
                "type": "notebook_sync",
                "seq": 3,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},
            }
        )

        messages = []
        while True:
            response = websocket.receive_json()
            messages.append(response)
            if response["type"] == "notebook_state":
                break

        idle_messages = [
            msg
            for msg in messages
            if msg["type"] == "cell_status"
            and msg["payload"].get("cell_id") == cell_id
            and msg["payload"].get("status") == "idle"
        ]
        assert idle_messages == []

        state = messages[-1]["payload"]["cells"]
        cell = next(c for c in state if c["id"] == cell_id)
        assert cell["status"] == "ready"


def test_last_websocket_disconnect_cancels_running_execution(
    client, temp_notebook, app, monkeypatch
):
    """Closing the final socket should cancel the active notebook execution."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.executor import CellExecutor
    from strata.notebook.routes import get_session_manager
    from strata.notebook.ws import _notebook_execution_state

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    cell_id = session.notebook_state.cells[0].id

    cancelled = threading.Event()

    async def fake_execute_cell(self, cell_id: str, source: str, timeout_seconds: float = 30):
        del self
        del cell_id
        del source
        del timeout_seconds
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(CellExecutor, "execute_cell", fake_execute_cell)

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "cell_status"
        assert response["payload"]["status"] == "running"

    assert cancelled.wait(timeout=1)
    for _ in range(20):
        state = _notebook_execution_state.get(session.id)
        if state is None or state.execution_task is None:
            break
        time.sleep(0.05)

    state = _notebook_execution_state.get(session.id)
    if state is not None:
        assert state.execution_task is None
        assert state.running_cell is None
        assert state.requested_cell is None


def test_malformed_message(client, temp_notebook, app):
    """Test handling of malformed messages."""
    notebook_dir, notebook_state = temp_notebook

    # Register notebook
    from strata.notebook.routes import get_session_manager

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)

    # Connect WebSocket
    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        # Send message without cell_id
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {},  # Missing cell_id
            }
        )

        # Should receive error
        response = websocket.receive_json()
        assert response["type"] == "error"
        assert "error" in response["payload"]


def test_cell_execute_blocked_while_environment_job_running(client, temp_notebook, app):
    """Cell execution should be rejected while an environment job is active."""
    notebook_dir, _ = temp_notebook

    from strata.notebook.routes import get_session_manager
    from strata.notebook.session import EnvironmentJobSnapshot

    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    cell_id = session.notebook_state.cells[0].id
    session.environment_job = EnvironmentJobSnapshot(
        id="job-1",
        action="sync",
        command="uv sync",
        status="running",
        phase="uv_running",
        started_at=1,
    )

    with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as websocket:
        websocket.send_json(
            {
                "type": "cell_execute",
                "seq": 1,
                "ts": "2026-03-23T00:00:00Z",
                "payload": {"cell_id": cell_id},
            }
        )

        response = websocket.receive_json()
        assert response["type"] == "error"
        assert response["payload"]["code"] == "ENVIRONMENT_BUSY"


def test_unknown_notebook(client, app):
    """Test connecting to non-existent notebook."""
    # Try to connect to non-existent notebook
    with pytest.raises(Exception):  # Should raise connection error
        with client.websocket_connect("/v1/notebooks/ws/nonexistent") as _websocket:
            pass


class TestRunningPayloadHelper:
    """Tests for the ``_running_payload`` helper that decorates the
    ``cell_status: running`` broadcast with remote worker metadata.

    Local cells must keep the existing, minimal payload so existing
    clients don't regress. Remote cells must include ``remote_worker``
    and ``remote_transport`` so the UI can render a live dispatch badge
    while the cell executes on the remote worker.
    """

    @staticmethod
    def _build_session(tmp_path, cells):
        """Build a NotebookSession with the given (cell_id, source) pairs.

        The notebook is created with two pre-registered workers: a
        DataFusion cluster at port 9000 and a GPU worker at 9001, both
        configured as HTTP executors.
        """
        from strata.notebook.models import WorkerBackendType, WorkerSpec
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        notebook_dir = create_notebook(tmp_path, "RunningPayloadTest", initialize_environment=False)
        prev_id = None
        for cell_id, source in cells:
            add_cell_to_notebook(notebook_dir, cell_id, prev_id)
            write_cell(notebook_dir, cell_id, source)
            prev_id = cell_id

        state = parse_notebook(notebook_dir)
        state.workers = [
            WorkerSpec(
                name="df-cluster",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="df-cluster",
                config={"url": "http://127.0.0.1:9000/v1/execute"},
            ),
            WorkerSpec(
                name="gpu-fly",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="gpu-fly",
                config={"url": "http://127.0.0.1:9001/v1/execute"},
            ),
        ]
        return NotebookSession(state, notebook_dir)

    def test_local_cell_returns_minimal_payload(self, tmp_path):
        from strata.notebook.ws import _running_payload

        session = self._build_session(tmp_path, [("c1", "x = 1")])
        payload = _running_payload(session, "c1", "x = 1")
        assert payload == {"cell_id": "c1", "status": "running"}

    def test_remote_cell_annotation_adds_worker_metadata(self, tmp_path):
        from strata.notebook.ws import _running_payload

        source = "# @worker gpu-fly\ny = 2"
        session = self._build_session(tmp_path, [("c1", source)])
        payload = _running_payload(session, "c1", source)
        assert payload["cell_id"] == "c1"
        assert payload["status"] == "running"
        assert payload["remote_worker"] == "gpu-fly"
        assert payload["remote_transport"] == "direct"

    def test_df_cluster_annotation_routes_to_df_cluster(self, tmp_path):
        from strata.notebook.ws import _running_payload

        source = "# @worker df-cluster\nz = 3"
        session = self._build_session(tmp_path, [("c1", source)])
        payload = _running_payload(session, "c1", source)
        assert payload["remote_worker"] == "df-cluster"

    def test_unknown_worker_falls_back_to_minimal_payload(self, tmp_path):
        from strata.notebook.ws import _running_payload

        source = "# @worker nonexistent-worker\nw = 4"
        session = self._build_session(tmp_path, [("c1", source)])
        payload = _running_payload(session, "c1", source)
        # Unknown worker name resolves to None → we drop the remote fields
        # rather than broadcasting a lie.
        assert "remote_worker" not in payload
        assert payload == {"cell_id": "c1", "status": "running"}

    def test_cell_level_worker_override_is_respected(self, tmp_path):
        from strata.notebook.ws import _running_payload

        session = self._build_session(tmp_path, [("c1", "q = 5")])
        # No annotation, but the cell has a persisted worker override
        cell = next(c for c in session.notebook_state.cells if c.id == "c1")
        cell.worker = "df-cluster"

        payload = _running_payload(session, "c1", "q = 5")
        assert payload["remote_worker"] == "df-cluster"


def test_variant_add_broadcasts_new_cell(client, app):
    """variant_add must broadcast notebook_state (with the new cell's full
    payload), not just dag_update — otherwise the frontend skips the new
    cell and the user sees the active variant 'disappear'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "variant_ws", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "model_a")
        write_cell(notebook_dir, "model_a", "# @variant g a\npreds = 1\n")

        from strata.notebook.routes import get_session_manager

        session = get_session_manager().open_notebook(notebook_dir)
        with client.websocket_connect(f"/v1/notebooks/ws/{session.id}") as ws:
            ws.send_json(
                {
                    "type": "variant_add",
                    "seq": 1,
                    "ts": "2026-05-10T00:00:00Z",
                    "payload": {"group": "g"},
                }
            )
            # Look for the notebook_state broadcast (other messages like
            # cell_status may interleave).
            for _ in range(10):
                response = ws.receive_json()
                if response["type"] == "notebook_state":
                    break
            else:
                raise AssertionError("Never received notebook_state for variant_add")

        state = response["payload"]
        cells = state["cells"]
        # Both the original and the new variant must be in the payload
        # with full source attached.
        new_cells = [c for c in cells if c.get("variant_name") == "a_copy"]
        assert len(new_cells) == 1
        assert "# @variant g a_copy" in new_cells[0]["source"]
        assert "preds = 1" in new_cells[0]["source"]
        # New variant is active; old becomes inactive.
        assert new_cells[0]["variant_active"] is True
        old = next(c for c in cells if c.get("variant_name") == "a")
        assert old["variant_active"] is False

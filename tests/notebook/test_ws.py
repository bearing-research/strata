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

# Sentinel timestamp for protocol envelopes. No test asserts on the value;
# the date is arbitrary and uniform so we don't reintroduce drift like the
# 2026-03-23 / 2026-03-30 / 2026-05-25 spread the file used to carry.
_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_session(notebook_dir):
    """Open a notebook session via the route layer."""
    from strata.notebook.routes import get_session_manager

    return get_session_manager().open_notebook(notebook_dir)


def ws_send(websocket, msg_type, payload=None, *, seq=1):
    """Send a notebook WS protocol message with the standard envelope.

    Centralizes the ``{type, seq, ts, payload}`` shape every emit site used to
    build inline. The timestamp is a sentinel — tests don't assert on it.
    """
    websocket.send_json(
        {
            "type": msg_type,
            "seq": seq,
            "ts": _TS,
            "payload": payload if payload is not None else {},
        }
    )


def receive_message_type(websocket, msg_type, *, max_messages=20):
    """Drain frames until one matches ``msg_type`` (str or iterable of str)."""
    accepted = {msg_type} if isinstance(msg_type, str) else set(msg_type)
    for _ in range(max_messages):
        response = websocket.receive_json()
        if response["type"] in accepted:
            return response
    raise AssertionError(f"Did not receive {msg_type} within {max_messages} messages")


def receive_execution_terminal(websocket, cell_id: str) -> tuple[dict, dict]:
    """Collect the output/error message and terminal status for one execution."""
    output_message = None
    terminal_status = None

    for _ in range(20):
        response = websocket.receive_json()
        if (
            response["type"] in ("cell_output", "cell_error")
            and response.get("payload", {}).get("cell_id") == cell_id
        ):
            output_message = response
        if (
            response["type"] == "cell_status"
            and response["payload"].get("cell_id") == cell_id
            and response["payload"]["status"] in ("ready", "error")
        ):
            terminal_status = response
            break

    assert output_message is not None
    assert terminal_status is not None
    return output_message, terminal_status


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_notebook():
    """Create a temporary notebook with three linear cells (x → y → z)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        notebook_dir = create_notebook(tmpdir, "test_notebook")

        for cell_id, source in (("root", "x = 1"), ("middle", "y = x + 1"), ("leaf", "z = y + 1")):
            add_cell_to_notebook(notebook_dir, cell_id)
            write_cell(notebook_dir, cell_id, source)

        notebook_state = parse_notebook(notebook_dir)
        yield notebook_dir, notebook_state


@pytest.fixture
def notebook_session(temp_notebook):
    """Open a session over ``temp_notebook`` and yield (notebook_dir, session).

    The common-case fixture for WS tests. Use ``temp_notebook`` + ``open_session``
    explicitly when the test must mutate cells (e.g. overwrite ``root``) before
    the session resolves.
    """
    notebook_dir, _ = temp_notebook
    yield notebook_dir, open_session(notebook_dir)


@pytest.fixture(scope="module")
def app():
    """FastAPI app with notebook routes. Module-scoped — the router is stateless."""
    from fastapi import FastAPI

    from strata.notebook.routes import router as notebook_router
    from strata.notebook.ws import router as notebook_ws_router

    fastapi_app = FastAPI()
    fastapi_app.include_router(notebook_router)
    fastapi_app.include_router(notebook_ws_router)
    return fastapi_app


@pytest.fixture
def client(app):
    """TestClient bound to the module-scoped app."""
    return TestClient(app)


def _ws(client, session):
    """Shorthand for ``client.websocket_connect`` against a notebook session."""
    return client.websocket_connect(f"/v1/notebooks/ws/{session.id}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_notebook_sync(client, notebook_session):
    """notebook_sync returns a full notebook state snapshot."""
    _, session = notebook_session

    with _ws(client, session) as websocket:
        ws_send(websocket, "notebook_sync")
        response = websocket.receive_json()

    assert response["type"] == "notebook_state"
    state = response["payload"]
    assert {"id", "cells", "dag"} <= state.keys()


def test_serialize_dag_edges_uses_frontend_field_names(notebook_session):
    """Edges sent to the frontend must use ``from_cell_id`` / ``to_cell_id``.

    Regression for a field-name drift in ``broadcast_notebook_sync`` (used by
    the agent's ``edit_cell`` / ``run_cell`` flows) where edges were emitted
    as ``{"from": ..., "to": ...}``. The frontend's ``applyBackendDag`` keys
    off ``from_cell_id`` / ``to_cell_id``, so every edge silently failed the
    cell lookup and the DAG view rendered as disconnected nodes after every
    agent edit until the user hard-refreshed.
    """
    _, session = notebook_session
    edges = session.dag.serialize_edges() if session.dag else []
    assert edges, "fixture defines x->y->z, expected at least one DAG edge"
    for edge in edges:
        assert set(edge) == {"from_cell_id", "to_cell_id", "variable"}, (
            f"DAG edge field names must match the frontend contract; got {edge}"
        )


def test_broadcast_notebook_sync_emits_correct_edge_field_names(notebook_session):
    """End-to-end: ``broadcast_notebook_sync`` must put correctly-keyed edges on the wire.

    Same regression as above but exercises the full agent-broadcast path: a
    fake WS captures the message and we inspect the payload directly. If
    this test ever flips back to ``from`` / ``to``, the agent edit flow has
    drifted again and the DAG view will silently break.
    """
    import json

    from strata.notebook.ws import _notebook_connections, broadcast_notebook_sync

    _, session = notebook_session
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


def test_notebook_sync_includes_causality_and_staleness(client, temp_notebook):
    """notebook_sync should return enriched cell state, not just bare DAG fields."""
    from strata.notebook.executor import CellExecutor

    notebook_dir, _ = temp_notebook
    session = open_session(notebook_dir)

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("root", "x = 1")).success
        root = next(c for c in session.notebook_state.cells if c.id == "root")
        root.source = "x = 2"
        write_cell(notebook_dir, "root", "x = 2")
        session.re_analyze_cell("root")
        session.compute_staleness()

    asyncio.run(_prime())

    with _ws(client, session) as websocket:
        ws_send(websocket, "notebook_sync")
        response = websocket.receive_json()

    assert response["type"] == "notebook_state"
    root = next(cell for cell in response["payload"]["cells"] if cell["id"] == "root")
    assert root["status"] == "idle"
    assert "staleness_reasons" in root
    assert root["causality"]["reason"] == "self"


def test_notebook_sync_includes_remote_execution_metadata(
    client,
    temp_notebook,
    notebook_executor_server,
    notebook_build_server,
):
    """Notebook sync should retain remote execution metadata from the live session."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec

    notebook_dir, _ = temp_notebook
    worker_config = {
        "url": notebook_executor_server["execute_url"],
        "transport": "signed",
        "strata_url": notebook_build_server["base_url"],
    }
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": worker_config,
        }
    ]

    session = open_session(notebook_dir)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu-http-signed",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-http-signed-a100",
            config=worker_config,
        )
    ]
    root = next(c for c in session.notebook_state.cells if c.id == "root")
    root.worker = "gpu-http-signed"

    async def _prime() -> None:
        executor = CellExecutor(session)
        assert (await executor.execute_cell("root", "x = 1")).success

    asyncio.run(_prime())

    with _ws(client, session) as websocket:
        ws_send(websocket, "notebook_sync")
        response = websocket.receive_json()

    assert response["type"] == "notebook_state"
    root = next(cell for cell in response["payload"]["cells"] if cell["id"] == "root")
    assert root["execution_method"] == "executor"
    assert root["remote_worker"] == "gpu-http-signed"
    assert root["remote_transport"] == "signed"
    assert isinstance(root["remote_build_id"], str)
    assert root["remote_build_state"] == "ready"
    assert root["remote_error_code"] is None


def test_cell_execute_no_cascade(client, notebook_session):
    """cell_execute on a root cell does not trigger cascade."""
    _, session = notebook_session
    root_cell = next(
        (c for c in session.notebook_state.cells if not c.upstream_ids),
        session.notebook_state.cells[0],
    )

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})

        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["status"] == "running"

        result = websocket.receive_json()
        assert result["type"] in ("cell_output", "cell_error")


def test_cell_execute_emits_explicit_display_payload(client, temp_notebook):
    """Image-like last-expression results should be sent in the dedicated display payload."""
    notebook_dir, _ = temp_notebook
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
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})
        output_message, terminal_status = receive_execution_terminal(websocket, "root")

    assert output_message["type"] == "cell_output"
    assert output_message["payload"]["display"]["content_type"] == "image/png"
    assert output_message["payload"]["display"]["inline_data_url"].startswith(
        "data:image/png;base64,"
    )
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_explicit_markdown_display_payload(client, temp_notebook):
    """Markdown last-expression results should be sent in the dedicated display payload."""
    notebook_dir, _ = temp_notebook
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
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})
        output_message, terminal_status = receive_execution_terminal(websocket, "root")

    assert output_message["type"] == "cell_output"
    assert output_message["payload"]["display"]["content_type"] == "text/markdown"
    assert (
        output_message["payload"]["display"]["markdown_text"]
        == "# Title\n\nRendered over websocket."
    )
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_display_side_effect_payload(client, temp_notebook):
    """display(...) side effects should be surfaced through the websocket display payload."""
    notebook_dir, _ = temp_notebook
    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# Side effect\\n\\nVia websocket."))
""",
    )
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})
        output_message, terminal_status = receive_execution_terminal(websocket, "root")

    assert output_message["type"] == "cell_output"
    assert output_message["payload"]["display"]["content_type"] == "text/markdown"
    assert (
        output_message["payload"]["display"]["markdown_text"] == "# Side effect\n\nVia websocket."
    )
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_emits_multiple_display_payloads_in_order(client, temp_notebook):
    """Ordered visible outputs should be sent together, with the last one preserved as display."""
    notebook_dir, _ = temp_notebook
    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# First"))
42
""",
    )
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})
        output_message, terminal_status = receive_execution_terminal(websocket, "root")

    payload = output_message["payload"]
    assert output_message["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 42
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 42
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_refreshes_downstream_staleness(client, temp_notebook):
    """Successful execution should immediately invalidate downstream cell state."""
    from strata.notebook.executor import CellExecutor

    notebook_dir, _ = temp_notebook
    session = open_session(notebook_dir)

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

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})
        messages = [websocket.receive_json() for _ in range(5)]

    status_updates = [msg["payload"] for msg in messages if msg["type"] == "cell_status"]
    assert any(p["cell_id"] == "root" and p["status"] == "ready" for p in status_updates)
    assert any(p["cell_id"] == "middle" and p["status"] == "idle" for p in status_updates)
    assert any(p["cell_id"] == "leaf" and p["status"] == "idle" for p in status_updates)


def test_cell_execute_surfaces_module_export_error(client, temp_notebook):
    """Unsupported cross-cell code export should surface as a direct cell error."""
    notebook_dir, _ = temp_notebook
    # ``x = len([])`` is a non-literal runtime assignment; plain literal
    # constants (``x = 1``) would now export fine alongside the def.
    write_cell(notebook_dir, "root", "x = len([])\n\ndef add(y):\n    return x + y\n")
    write_cell(notebook_dir, "middle", "result = add(2)")
    session = open_session(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})

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


def test_cell_execute_surfaces_module_export_lambda_error(client, temp_notebook):
    """The WS path should surface top-level lambda export errors clearly."""
    notebook_dir, _ = temp_notebook
    write_cell(notebook_dir, "root", "add = lambda y: y + 1\n")
    write_cell(notebook_dir, "middle", "result = add(2)")
    session = open_session(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "root"})

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


def test_cell_execute_uses_warm_pool_when_available(client, notebook_session, monkeypatch):
    """The WebSocket path is wired to use the session warm pool."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.pool import PooledCellExecutor, WarmProcessPool

    _, session = notebook_session
    session.warm_pool = cast(WarmProcessPool, object())

    root_cell = next(
        (c for c in session.notebook_state.cells if not c.upstream_ids),
        session.notebook_state.cells[0],
    )

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
        PooledCellExecutor, "execute_with_pool", staticmethod(fake_execute_with_pool)
    )
    monkeypatch.setattr(CellExecutor, "_store_outputs", fake_store_outputs)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})
        output_message, final_status = receive_execution_terminal(websocket, root_cell.id)

    assert output_message["type"] == "cell_output"
    assert output_message["payload"]["execution_method"] == "warm"
    assert final_status["payload"]["status"] == "ready"
    assert warm_calls == 1


def test_notebook_run_all_emits_multiple_display_payloads_in_order(client, temp_notebook):
    """Run-all should preserve ordered display payloads on the websocket path."""
    notebook_dir, _ = temp_notebook
    write_cell(
        notebook_dir,
        "root",
        """
display(Markdown("# First"))
42
""",
    )
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "notebook_run_all")
        output_message, terminal_status = receive_execution_terminal(websocket, "root")

    payload = output_message["payload"]
    assert output_message["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 42
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 42
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_cascade_emits_multiple_display_payloads_in_order(client, temp_notebook):
    """Cascade execution should preserve ordered display payloads for the target cell."""
    notebook_dir, _ = temp_notebook
    write_cell(
        notebook_dir,
        "leaf",
        """
display(Markdown("# First"))
y + 1
""",
    )
    session = open_session(notebook_dir)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": "leaf"})
        prompt = websocket.receive_json()
        assert prompt["type"] == "cascade_prompt"

        ws_send(
            websocket,
            "cell_execute_cascade",
            {"cell_id": "leaf", "plan_id": prompt["payload"]["plan_id"]},
            seq=2,
        )
        output_message, terminal_status = receive_execution_terminal(websocket, "leaf")

    payload = output_message["payload"]
    assert output_message["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 3
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 3
    assert terminal_status["payload"]["status"] == "ready"


def test_cell_execute_blocked_when_environment_runtime_is_unavailable(client, notebook_session):
    """Execution should be blocked when no notebook runtime is available after bootstrap failure."""
    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    session.environment_job = None
    session.venv_python = None
    session.environment_interpreter_source = "unknown"
    session.environment_sync_state = "failed"
    session.environment_sync_error = "Failed to start notebook environment initialization: boom"
    session.environment_sync_notice = None

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": cell_id})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["payload"]["code"] == "ENVIRONMENT_BUSY"
    assert "environment" in response["payload"]["error"].lower()


def test_environment_job_submission_rejects_execution_already_accepted(monkeypatch, temp_notebook):
    """Execution acceptance should block env jobs before the task starts."""
    from strata.notebook import ws as notebook_ws

    notebook_dir, _ = temp_notebook
    session = open_session(notebook_dir)
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


def _http_worker_config(executor_url, *, build_url=None, transport=None):
    """Build a worker config dict for HTTP executor tests."""
    config = {"url": executor_url}
    if transport:
        config["transport"] = transport
    if build_url:
        config["strata_url"] = build_url
    return config


def _attach_worker(session, name, runtime_id, config):
    """Attach a single HTTP-executor worker to the session and return it."""
    from strata.notebook.models import WorkerBackendType, WorkerSpec

    session.notebook_state.workers = [
        WorkerSpec(
            name=name,
            backend=WorkerBackendType.EXECUTOR,
            runtime_id=runtime_id,
            config=config,
        )
    ]
    return session.notebook_state.workers[0]


def test_ws_execute_supports_http_executor_worker(
    client, notebook_session, notebook_executor_server
):
    """The live WebSocket execution path should support HTTP notebook workers."""
    _, session = notebook_session
    _attach_worker(
        session,
        "gpu-http",
        "gpu-http-a100",
        _http_worker_config(notebook_executor_server["execute_url"]),
    )
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http"

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})
        output_message, terminal_status = receive_execution_terminal(websocket, root_cell.id)

    payload = output_message["payload"]
    assert output_message["type"] == "cell_output"
    assert payload["execution_method"] == "executor"
    assert payload["remote_worker"] == "gpu-http"
    assert payload["remote_transport"] == "direct"
    assert payload["outputs"]["x"]["preview"] == 1
    assert terminal_status["payload"]["status"] == "ready"


def test_ws_execute_supports_signed_http_executor_worker(
    client,
    notebook_session,
    notebook_executor_server,
    notebook_build_server,
):
    """The live WebSocket path should support signed remote notebook workers."""
    _, session = notebook_session
    config = _http_worker_config(
        notebook_executor_server["execute_url"],
        build_url=notebook_build_server["base_url"],
        transport="signed",
    )
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": config,
        }
    ]
    _attach_worker(session, "gpu-http-signed", "gpu-http-signed-a100", config)
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})
        first_output, first_terminal = receive_execution_terminal(websocket, root_cell.id)

        first_payload = first_output["payload"]
        assert first_output["type"] == "cell_output"
        assert first_payload["execution_method"] == "executor"
        assert first_payload["remote_worker"] == "gpu-http-signed"
        assert first_payload["remote_transport"] == "signed"
        assert isinstance(first_payload["remote_build_id"], str)
        assert first_payload["outputs"]["x"]["preview"] == 1
        assert first_terminal["payload"]["status"] == "ready"

        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id}, seq=2)
        second_output, second_terminal = receive_execution_terminal(websocket, root_cell.id)

    second_payload = second_output["payload"]
    assert second_output["type"] == "cell_output"
    assert second_payload["execution_method"] == "cached"
    assert second_payload["remote_worker"] == "gpu-http-signed"
    assert second_payload["remote_transport"] == "signed"
    assert "remote_build_id" not in second_payload
    assert second_terminal["payload"]["status"] == "ready"


def test_ws_execute_supports_signed_http_executor_worker_with_class_instances(
    client,
    notebook_executor_server,
    notebook_build_server,
):
    """The live WS path should preserve exported class instances over signed transport."""
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

        config = _http_worker_config(
            notebook_executor_server["execute_url"],
            build_url=notebook_build_server["base_url"],
            transport="signed",
        )
        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {
                "name": "gpu-http-signed",
                "backend": "executor",
                "runtime_id": "gpu-http-signed-a100",
                "config": config,
            }
        ]

        session = open_session(notebook_dir)
        _attach_worker(session, "gpu-http-signed", "gpu-http-signed-a100", config)
        for cell in session.notebook_state.cells:
            cell.worker = "gpu-http-signed"

        with _ws(client, session) as websocket:
            for cell_id in ("cell1", "cell2", "cell3"):
                ws_send(websocket, "cell_execute", {"cell_id": cell_id})
                output_message, terminal_status = receive_execution_terminal(websocket, cell_id)
                payload = output_message["payload"]
                assert output_message["type"] == "cell_output"
                assert payload["execution_method"] == "executor"
                assert payload["remote_worker"] == "gpu-http-signed"
                assert payload["remote_transport"] == "signed"
                assert payload["remote_build_state"] == "ready"
                assert terminal_status["payload"]["status"] == "ready"

            ws_send(websocket, "notebook_sync", seq=2)
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


def test_ws_execute_reports_unavailable_http_executor_worker(client, notebook_session):
    """The live WS path should surface unreachable HTTP executor workers."""
    _, session = notebook_session
    _attach_worker(
        session,
        "gpu-http-dead",
        "gpu-http-dead-a100",
        _http_worker_config("http://127.0.0.1:9/v1/execute"),
    )
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-dead"

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})
        output_message, terminal_status = receive_execution_terminal(websocket, root_cell.id)

    assert output_message["type"] == "cell_error"
    assert "Remote executor request failed" in output_message["payload"]["error"]
    assert terminal_status["payload"]["status"] == "error"


def test_ws_execute_reports_signed_finalize_failure(
    client,
    notebook_session,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """The live WS path should surface signed transport finalize failures."""
    from strata.transforms.signed_urls import (
        generate_build_manifest as real_generate_build_manifest,
    )

    _, session = notebook_session

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
        "strata.notebook.executor.generate_build_manifest", fake_generate_build_manifest
    )

    config = _http_worker_config(
        notebook_executor_server["execute_url"],
        build_url=notebook_build_server["base_url"],
        transport="signed",
    )
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": config,
        }
    ]
    _attach_worker(session, "gpu-http-signed", "gpu-http-signed-a100", config)
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})
        output_message, terminal_status = receive_execution_terminal(websocket, root_cell.id)

    payload = output_message["payload"]
    assert output_message["type"] == "cell_error"
    assert "Failed to finalize notebook bundle build" in payload["error"]
    assert payload["remote_worker"] == "gpu-http-signed"
    assert payload["remote_transport"] == "signed"
    assert isinstance(payload["remote_build_id"], str)
    assert payload["remote_build_state"] == "failed"
    assert payload["remote_error_code"] == "FINALIZE_FAILED"
    assert terminal_status["payload"]["status"] == "error"


def test_ws_cancelled_signed_http_executor_marks_build_failed(
    client,
    notebook_session,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """Cancelling signed remote execution over WS should fail the build cleanly."""
    _, session = notebook_session
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
            "variables": {"x": {"content_type": "json/object", "file": "x.json", "preview": 1}},
            "stdout": "",
            "stderr": "",
            "mutation_warnings": [],
        }

    monkeypatch.setattr("strata.notebook.remote_executor._run_harness", _slow_run_harness)

    config = _http_worker_config(
        notebook_executor_server["execute_url"],
        build_url=notebook_build_server["base_url"],
        transport="signed",
    )
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {
            "name": "gpu-http-signed",
            "backend": "executor",
            "runtime_id": "gpu-http-signed-a100",
            "config": config,
        }
    ]
    _attach_worker(session, "gpu-http-signed", "gpu-http-signed-a100", config)
    root_cell = next(c for c in session.notebook_state.cells if c.id == "root")
    root_cell.worker = "gpu-http-signed"

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": root_cell.id})

        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["cell_id"] == root_cell.id
        assert running["payload"]["status"] == "running"
        assert started.wait(timeout=2.0)

        ws_send(websocket, "cell_cancel", {"cell_id": root_cell.id}, seq=2)

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


def test_cascade_prompt_is_sent_only_to_requesting_websocket(client, notebook_session):
    """A cascade prompt should not fan out to other clients on the notebook."""
    _, session = notebook_session

    with _ws(client, session) as ws1, _ws(client, session) as ws2:
        ws_send(ws1, "cell_execute", {"cell_id": "middle"})
        response = ws1.receive_json()
        assert response["type"] == "cascade_prompt"
        assert response["payload"]["cell_id"] == "middle"

        with pytest.raises(Exception):
            ws2.receive_json(timeout=0.1)


def test_impact_preview_is_sent_only_to_requesting_websocket(client, notebook_session):
    """Impact preview responses should stay scoped to the requesting client."""
    _, session = notebook_session

    with _ws(client, session) as ws1, _ws(client, session) as ws2:
        ws_send(ws1, "impact_preview_request", {"cell_id": "middle"})
        response = ws1.receive_json()
        assert response["type"] == "impact_preview"
        assert response["payload"]["target_cell_id"] == "middle"

        with pytest.raises(Exception):
            ws2.receive_json(timeout=0.1)


def test_inspect_repl_round_trip(client, notebook_session):
    """The inspect REPL round-trips over the websocket."""
    _, session = notebook_session
    middle_cell = next(c for c in session.notebook_state.cells if "x" in c.references)

    with _ws(client, session) as websocket:
        ws_send(websocket, "inspect_open", {"cell_id": middle_cell.id})
        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "open"
        assert response["payload"]["ok"] is True

        ws_send(websocket, "inspect_eval", {"cell_id": middle_cell.id, "expr": "x + 1"}, seq=2)
        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "eval"
        assert response["payload"]["ok"] is True
        assert response["payload"]["result"] == "2"
        assert response["payload"]["type"] == "int"

        ws_send(websocket, "inspect_close", {"cell_id": middle_cell.id}, seq=3)
        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["action"] == "close"
        assert response["payload"]["ok"] is True


def test_active_websocket_session_is_not_evicted(client, notebook_session):
    """TTL eviction should skip sessions that still have connected sockets."""
    from strata.notebook.routes import get_session_manager

    _, session = notebook_session
    session_manager = get_session_manager()
    session.last_accessed = time.time() - session_manager.SESSION_TTL_SECONDS - 60

    with _ws(client, session) as websocket:
        session_manager._evict_stale()
        assert session.id in session_manager.list_sessions()

        ws_send(websocket, "notebook_sync")
        websocket.receive_json()
        assert session.last_accessed > time.time() - 5


def _patch_inspect_manager(monkeypatch, *, close_counter):
    """Patch InspectManager.open_session / close_all and count close calls."""
    from strata.notebook.inspect_repl import InspectManager

    async def fake_open_session(self, cell_id, notebook_session):
        return SimpleNamespace(ready=True), "ready"

    async def fake_close_all(self):
        close_counter["count"] += 1

    monkeypatch.setattr(InspectManager, "open_session", fake_open_session)
    monkeypatch.setattr(InspectManager, "close_all", fake_close_all)


def test_inspect_sessions_closed_when_last_websocket_disconnects(
    client, notebook_session, monkeypatch
):
    """Disconnecting the last socket should close notebook inspect sessions."""
    from strata.notebook.ws import _notebook_inspect_managers

    _, session = notebook_session
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    middle_cell = next(c for c in session.notebook_state.cells if "x" in c.references)

    with _ws(client, session) as websocket:
        ws_send(websocket, "inspect_open", {"cell_id": middle_cell.id})
        response = websocket.receive_json()
        assert response["type"] == "inspect_result"
        assert response["payload"]["ok"] is True

    assert close_counter["count"] == 1
    assert session.id not in _notebook_inspect_managers


def _open_inspect_and_disconnect(client, session, cell_id):
    """Open an inspect REPL on ``cell_id``, then close the WS cleanly."""
    with _ws(client, session) as websocket:
        ws_send(websocket, "inspect_open", {"cell_id": cell_id})
        websocket.receive_json()


def _inject_long_running_execution(client, session):
    """Open a WS, sync, then inject a long-sleep task as the active execution.

    Returns the injected task. Caller is responsible for cancelling it
    when the test ends so the next test's loop doesn't inherit it.
    """
    from strata.notebook.ws import _ensure_execution_state

    async def long_sleep() -> None:
        await asyncio.sleep(60)

    async def inject_task() -> asyncio.Task[None]:
        state = _ensure_execution_state(session.id)
        task = asyncio.create_task(long_sleep())
        state.execution_task = task
        return task

    with _ws(client, session) as ws:
        ws_send(ws, "notebook_sync")
        ws.receive_json()
        return client.portal.call(inject_task)


def _cancel_async_task(client, task):
    """Cancel an injected task from the server's event loop."""

    async def _cancel(task_to_cancel: asyncio.Task[None]) -> None:
        task_to_cancel.cancel()
        try:
            await task_to_cancel
        except asyncio.CancelledError:
            pass

    client.portal.call(_cancel, task)


def test_grace_window_preserves_inspect_state_on_reconnect(client, notebook_session, monkeypatch):
    """Disconnect-then-reconnect within the grace window keeps inspect state.

    Without the grace window, the inspect manager would be dropped the moment
    the WS context exits. The TUI's tmux / SSH audience would lose any open
    inspect REPL on a network blip — bad UX for exactly the users the window
    exists for.
    """
    from strata.notebook.ws import _notebook_inspect_managers

    _, session = notebook_session
    # Long enough that the test thread can't accidentally cross it; the
    # reconnect cancels the task explicitly.
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    cell = next(c for c in session.notebook_state.cells if "x" in c.references)
    _open_inspect_and_disconnect(client, session, cell.id)

    # Inside the grace window: state must be preserved.
    assert session.id in _notebook_inspect_managers
    assert close_counter["count"] == 0

    # Reconnect cancels the pending teardown and resumes.
    with _ws(client, session) as websocket:
        ws_send(websocket, "notebook_sync")
        websocket.receive_json()
        assert session.id in _notebook_inspect_managers
        assert close_counter["count"] == 0


def test_grace_window_expires_drops_state(client, notebook_session, monkeypatch):
    """When the grace window elapses with no reconnect, state is dropped."""
    from strata.notebook.ws import _notebook_grace_tasks, _notebook_inspect_managers

    _, session = notebook_session
    # Short window: long enough that the test thread observes the "pending
    # teardown" state, short enough that polling for completion doesn't drag
    # the suite.
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 0.1)
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    cell = next(c for c in session.notebook_state.cells if "x" in c.references)

    # Hold the TestClient open across requests so the server's event loop
    # outlives the WS context; without this, the grace task scheduled at
    # disconnect would die with the portal.
    with client:
        _open_inspect_and_disconnect(client, session, cell.id)

        assert session.id in _notebook_grace_tasks
        assert session.id in _notebook_inspect_managers

        # Poll until the task runs in the server's background event loop.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if close_counter["count"] and session.id not in _notebook_inspect_managers:
                break
            time.sleep(0.02)

    assert close_counter["count"] == 1
    assert session.id not in _notebook_inspect_managers
    assert session.id not in _notebook_grace_tasks


def test_grace_window_preserves_active_execution_on_reconnect(
    client, notebook_session, monkeypatch
):
    """A running execution task survives a disconnect-reconnect cycle within the window.

    Load-bearing #42 contract: tmux detach during a long cell, reconnect
    within the window, find the cell still running. This targets
    ``NotebookExecutionState.execution_task`` — the actual mechanism behind
    cancel-on-disconnect — rather than the inspect cleanup hook.
    """
    from strata.notebook.ws import _notebook_execution_state

    _, session = notebook_session
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)

    with client:
        task = _inject_long_running_execution(client, session)

        # Inside grace window: task is still alive.
        assert not task.done(), "execution task was cancelled before the grace window expired"
        assert _notebook_execution_state.get(session.id) is not None

        # Reconnect cancels the pending teardown — task stays alive.
        with _ws(client, session) as ws:
            ws_send(ws, "notebook_sync")
            ws.receive_json()
            assert not task.done()
            assert _notebook_execution_state.get(session.id) is not None

        _cancel_async_task(client, task)


def test_grace_window_expiry_cancels_active_execution(client, notebook_session, monkeypatch):
    """Past the grace window with no reconnect, the running execution task is cancelled."""
    from strata.notebook.ws import _notebook_execution_state, _notebook_grace_tasks

    _, session = notebook_session
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 0.1)

    with client:
        task = _inject_long_running_execution(client, session)

        assert not task.done()
        assert session.id in _notebook_grace_tasks

        # Poll for the grace task to fire and cancel the execution.
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if task.done() and session.id not in _notebook_execution_state:
                break
            time.sleep(0.02)

        assert task.done()
        assert task.cancelled()
        assert session.id not in _notebook_execution_state
        assert session.id not in _notebook_grace_tasks


def test_cell_source_update(client, notebook_session):
    """cell_source_update triggers DAG recomputation."""
    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_source_update", {"cell_id": cell_id, "source": "x = 2\ny = 3"})

        response = websocket.receive_json()
        assert response["type"] == "dag_update"
        assert "edges" in response["payload"]
        assert "topological_order" in response["payload"]

        # Drain any trailing cell_status updates.
        while True:
            try:
                response = websocket.receive_json(timeout=0.1)
            except Exception:
                break
            if response["type"] == "cell_status":
                assert "status" in response["payload"]


def test_cell_cancel(client, notebook_session):
    """cell_cancel stops execution."""
    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_cancel", {"cell_id": cell_id})
        response = websocket.receive_json()

    assert response["type"] == "cell_status"
    assert response["payload"]["status"] == "idle"


def test_cell_cancel_interrupts_running_execution_on_same_websocket(
    client, notebook_session, monkeypatch
):
    """A single WebSocket can cancel its own in-flight execution."""
    from strata.notebook.executor import CellExecutor

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    cancelled = threading.Event()

    async def fake_execute_cell(self, cell_id: str, source: str, timeout_seconds: float = 30):
        del self, cell_id, source, timeout_seconds
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(CellExecutor, "execute_cell", fake_execute_cell)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": cell_id})
        running = websocket.receive_json()
        assert running["type"] == "cell_status"
        assert running["payload"]["cell_id"] == cell_id
        assert running["payload"]["status"] == "running"

        ws_send(websocket, "cell_cancel", {"cell_id": cell_id}, seq=2)
        idle = websocket.receive_json()
        assert idle["type"] == "cell_status"
        assert idle["payload"]["cell_id"] == cell_id
        assert idle["payload"]["status"] == "idle"

    assert cancelled.is_set()


def test_stale_cell_cancel_does_not_clobber_ready_state(client, notebook_session):
    """A late cancel should not rewrite a completed cell back to idle."""
    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": cell_id})

        while True:
            msg = websocket.receive_json()
            if (
                msg["type"] == "cell_status"
                and msg["payload"]["cell_id"] == cell_id
                and msg["payload"]["status"] in ("ready", "error")
            ):
                assert msg["payload"]["status"] == "ready"
                break

        ws_send(websocket, "cell_cancel", {"cell_id": cell_id}, seq=2)
        ws_send(websocket, "notebook_sync", seq=3)

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

    cells = messages[-1]["payload"]["cells"]
    cell = next(c for c in cells if c["id"] == cell_id)
    assert cell["status"] == "ready"


def test_last_websocket_disconnect_cancels_running_execution(client, notebook_session, monkeypatch):
    """Closing the final socket should cancel the active notebook execution."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.ws import _notebook_execution_state

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    cancelled = threading.Event()

    async def fake_execute_cell(self, cell_id: str, source: str, timeout_seconds: float = 30):
        del self, cell_id, source, timeout_seconds
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(CellExecutor, "execute_cell", fake_execute_cell)

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": cell_id})
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


def test_malformed_message(client, notebook_session):
    """Malformed messages produce a generic error frame."""
    _, session = notebook_session

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute")  # missing cell_id
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert "error" in response["payload"]


def test_cell_execute_blocked_while_environment_job_running(client, notebook_session):
    """Cell execution should be rejected while an environment job is active."""
    from strata.notebook.session import EnvironmentJobSnapshot

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    session.environment_job = EnvironmentJobSnapshot(
        id="job-1",
        action="sync",
        command="uv sync",
        status="running",
        phase="uv_running",
        started_at=1,
    )

    with _ws(client, session) as websocket:
        ws_send(websocket, "cell_execute", {"cell_id": cell_id})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["payload"]["code"] == "ENVIRONMENT_BUSY"


def test_unknown_notebook(client):
    """Connecting to a non-existent notebook should fail the upgrade."""
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/notebooks/ws/nonexistent"):
            pass


class TestRunningPayloadHelper:
    """Tests for the ``_running_payload`` helper that decorates the
    ``cell_status: running`` broadcast with remote worker metadata.

    Local cells must keep the existing, minimal payload so existing clients
    don't regress. Remote cells must include ``remote_worker`` and
    ``remote_transport`` so the UI can render a live dispatch badge while
    the cell executes on the remote worker.
    """

    @staticmethod
    def _build_session(tmp_path, cells):
        """Build a NotebookSession with the given (cell_id, source) pairs.

        The notebook is created with two pre-registered workers: a DataFusion
        cluster at port 9000 and a GPU worker at 9001, both configured as
        HTTP executors.
        """
        from strata.notebook.models import WorkerBackendType, WorkerSpec
        from strata.notebook.session import NotebookSession

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


def test_variant_add_broadcasts_new_cell(client):
    """variant_add must broadcast notebook_state (with the new cell's full
    payload), not just dag_update — otherwise the frontend skips the new
    cell and the user sees the active variant 'disappear'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "variant_ws", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "model_a")
        write_cell(notebook_dir, "model_a", "# @variant g a\npreds = 1\n")

        session = open_session(notebook_dir)
        with _ws(client, session) as ws:
            ws_send(ws, "variant_add", {"group": "g"})
            # Look for the notebook_state broadcast (other messages like
            # cell_status may interleave).
            response = receive_message_type(ws, "notebook_state", max_messages=10)

    cells = response["payload"]["cells"]
    new_cells = [c for c in cells if c.get("variant_name") == "a_copy"]
    assert len(new_cells) == 1
    assert "# @variant g a_copy" in new_cells[0]["source"]
    assert "preds = 1" in new_cells[0]["source"]
    # New variant is active; old becomes inactive.
    assert new_cells[0]["variant_active"] is True
    old = next(c for c in cells if c.get("variant_name") == "a")
    assert old["variant_active"] is False


# WS upgrade owner gating: the parallel gate in ``ws.py`` mirrors the REST
# dependency (``get_notebook_session`` → ``_require_owner``) and shares the
# same ``_require_owner`` / ``_user_scoping_enabled`` helpers as the REST
# surface — the REST-side coverage in
# ``test_routes::TestPersonalModeUserScoping`` (owner allowed, wrong header
# refused, missing header refused, legacy unowned passthrough) is the
# load-bearing test for the gate logic. A direct TestClient-based WS
# upgrade test was attempted but pulled because the per-test anyio portal
# teardown reproducibly hangs on GitHub Actions Python 3.12 runners —
# ``thread.join()`` on the portal thread blocks indefinitely while waiting
# on lingering asyncio tasks from session-manager teardown. Tracked as a
# follow-up to land WS-specific coverage via fake websocket objects.


# ---------------------------------------------------------------------------
# Prompt-cell streaming broadcast wiring (issue #110)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_executor_with_progress_wires_prompt_delta_broadcast(
    notebook_session, monkeypatch
):
    """``_make_executor_with_progress`` must wire ``on_prompt_delta`` to a
    ``cell_output_delta`` broadcast, mirroring the loop-progress wiring.
    Tested through the callback directly (fake broadcast, no WS upgrade —
    see the TestClient WS portal hang note in the test-suite memory)."""
    from strata.notebook import ws as ws_module
    from strata.notebook.ws import _make_executor_with_progress

    _, session = notebook_session
    broadcasts: list[tuple[str, dict]] = []

    async def fake_broadcast(notebook_id, message):
        broadcasts.append((notebook_id, message))

    monkeypatch.setattr(ws_module, "_broadcast_message", fake_broadcast)

    executor = _make_executor_with_progress(session, "nb-stream-test")
    assert executor.on_prompt_delta is not None
    assert executor.on_iteration_complete is not None

    payload = {"cell_id": "p1", "attempt": 1, "kind": "delta", "text": "chunk"}
    await executor.on_prompt_delta(payload)

    assert len(broadcasts) == 1
    notebook_id, message = broadcasts[0]
    assert notebook_id == "nb-stream-test"
    assert message["type"] == "cell_output_delta"
    assert message["payload"] == payload
    assert isinstance(message["seq"], int)


@pytest.mark.asyncio
async def test_final_output_seq_is_newer_than_streamed_deltas(notebook_session, monkeypatch):
    """Streaming frames draw from the same per-notebook counter as the
    execution envelope; the canonical ``cell_output`` must carry a seq
    LATER than every delta it supersedes, or seq-ordering clients treat
    the final result as stale (PR #111 review finding)."""
    from strata.notebook import ws as ws_module
    from strata.notebook.executor import CellExecutionResult
    from strata.notebook.ws import _ensure_execution_state, _execute_cell_directly

    _, session = notebook_session
    broadcasts: list[dict] = []

    async def fake_broadcast(notebook_id, message):
        broadcasts.append(message)

    monkeypatch.setattr(ws_module, "_broadcast_message", fake_broadcast)

    class _StreamingStubExecutor:
        """Stands in for CellExecutor inside _make_executor_with_progress:
        emits two prompt deltas mid-"execution", then returns success."""

        def __init__(self, session, warm_pool=None):
            self.on_iteration_complete = None
            self.on_prompt_delta = None

        async def execute_cell(self, cell_id, source):
            assert self.on_prompt_delta is not None
            for chunk in ("hel", "lo"):
                await self.on_prompt_delta(
                    {"cell_id": cell_id, "attempt": 1, "kind": "delta", "text": chunk}
                )
            return CellExecutionResult(cell_id=cell_id, success=True)

    monkeypatch.setattr(ws_module, "CellExecutor", _StreamingStubExecutor)

    # Use the registry-backed state — that's what the WS handler passes in
    # production, and it's the same counter ``next_notebook_sequence``
    # draws delta seqs from.
    execution_state = _ensure_execution_state("nb-seq-test")
    try:
        await _execute_cell_directly(
            cast(WebSocket, None),
            session,
            "root",
            execution_state,
            "nb-seq-test",
        )
    finally:
        ws_module._notebook_execution_state.pop("nb-seq-test", None)

    running = next(
        m for m in broadcasts if m["type"] == "cell_status" and m["payload"]["status"] == "running"
    )
    deltas = [m for m in broadcasts if m["type"] == "cell_output_delta"]
    output = next(m for m in broadcasts if m["type"] == "cell_output")

    assert len(deltas) == 2
    assert all(d["seq"] > running["seq"] for d in deltas)
    assert output["seq"] > max(d["seq"] for d in deltas)

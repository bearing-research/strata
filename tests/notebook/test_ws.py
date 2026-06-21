"""Tests for WebSocket notebook execution.

These tests drive the notebook WS handlers (and the ``notebook_websocket``
endpoint) directly in the event loop against a ``FakeNotebookWebSocket``,
rather than through ``TestClient.websocket_connect``. The TestClient runs
the ASGI app on an anyio blocking portal; on macOS + Python 3.12 the portal
teardown's ``thread.join()`` deadlocks on lingering session tasks created
during a WS upgrade when a module-scope ``app`` fixture meets
session-creating fixtures (#206). Calling the handlers in-loop sidesteps the
portal entirely and gives #52 direct owner-gate coverage. See #205 / #52.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import WebSocket

from strata.notebook.parser import parse_notebook
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket

_MINIMAL_PNG_LITERAL = (
    'b"\\x89PNG\\r\\n\\x1a\\n\\x00\\x00\\x00\\rIHDR\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x01'
    "\\x08\\x04\\x00\\x00\\x00\\xb5\\x1c\\x0c\\x02\\x00\\x00\\x00\\x0bIDATx\\xdac\\xfc\\xff"
    '\\x1f\\x00\\x03\\x03\\x02\\x00\\xef\\x9b\\xe0M\\x00\\x00\\x00\\x00IEND\\xaeB`\\x82"'
)
_MARKDOWN_LITERAL = '"# Title\\n\\nRendered over websocket."'

# Sentinel timestamp for protocol envelopes. No test asserts on the value;
# the date is arbitrary and uniform so we don't reintroduce drift.
_TS = "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def open_session(notebook_dir):
    """Open a notebook session via the route layer."""
    from strata.notebook.routes import get_session_manager

    return get_session_manager().open_notebook(notebook_dir)


def _envelope(msg_type, payload=None, *, seq=1):
    """Build a raw JSON WS protocol envelope to script ``inbound`` queues."""
    return json.dumps(
        {
            "type": msg_type,
            "seq": seq,
            "ts": _TS,
            "payload": payload if payload is not None else {},
        }
    )


def _make_fake_ws(session, *, inbound=None, headers=None):
    """Create a fake WS already registered for ``session`` broadcasts.

    Handlers fan results out through ``_broadcast_message`` (which reads
    ``_notebook_connections``) in addition to direct ``send_text`` replies,
    so the fake must be registered as a connection to capture every frame.
    Returns ``(fake, execution_state)``.
    """
    from strata.notebook.ws import _ensure_execution_state, _notebook_connections

    fake = FakeNotebookWebSocket(inbound=inbound, headers=headers)
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, fake))
    return fake, _ensure_execution_state(session.id)


async def _wait_until(predicate, *, yields=2000):
    """Yield control to the loop until ``predicate()`` is true.

    Used to let a handler-scheduled background task reach its next ``await``
    (e.g. broadcast the ``running`` frame, then block on a long sleep) before
    the test cancels it. Bounded by a yield count rather than wall-clock time
    so it carries no timing assertion; raises if the predicate never holds.
    """
    for _ in range(yields):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true while draining the event loop")


def _running_frames(fake, cell_id):
    """Return the ``cell_status: running`` frames emitted for ``cell_id``."""
    return [
        f
        for f in fake.frames_of("cell_status")
        if f["payload"].get("cell_id") == cell_id and f["payload"]["status"] == "running"
    ]


def _terminal_frames(fake, cell_id):
    """Return (output_or_error_frame, terminal_status_frame) for ``cell_id``."""
    output = next(
        (
            f
            for f in fake.sent
            if f["type"] in ("cell_output", "cell_error") and f["payload"].get("cell_id") == cell_id
        ),
        None,
    )
    terminal = next(
        (
            f
            for f in fake.sent
            if f["type"] == "cell_status"
            and f["payload"].get("cell_id") == cell_id
            and f["payload"].get("status") in ("ready", "error")
        ),
        None,
    )
    return output, terminal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_ws_globals():
    """Reset per-notebook WS global state before and after every test."""
    from strata.notebook.ws import (
        _notebook_grace_tasks,
        _notebook_inspect_managers,
    )
    from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset

    def _clear():
        _reset()
        _notebook_inspect_managers.clear()
        _notebook_grace_tasks.clear()

    _clear()
    yield
    _clear()


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


# ---------------------------------------------------------------------------
# Sync / DAG serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notebook_sync(notebook_session):
    """notebook_sync returns a full notebook state snapshot."""
    from strata.notebook.ws import _handle_notebook_sync

    _, session = notebook_session
    fake, _ = _make_fake_ws(session)

    await _handle_notebook_sync(cast(WebSocket, fake), session, session.id)

    states = fake.frames_of("notebook_state")
    assert states
    state = states[-1]["payload"]
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
    from strata.notebook.ws import _notebook_connections, broadcast_notebook_sync

    _, session = notebook_session
    fake = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, fake))
    asyncio.run(broadcast_notebook_sync(session.id, session))

    msgs = fake.frames_of("notebook_state")
    assert msgs, "broadcast_notebook_sync did not deliver a notebook_state message"
    edges = msgs[-1]["payload"]["dag"]["edges"]
    assert edges, "fixture defines x->y->z, expected at least one DAG edge"
    for edge in edges:
        assert set(edge) == {"from_cell_id", "to_cell_id", "variable"}, (
            f"broadcast_notebook_sync emitted bad edge shape: {edge}"
        )


@pytest.mark.asyncio
async def test_notebook_sync_includes_causality_and_staleness(temp_notebook):
    """notebook_sync should return enriched cell state, not just bare DAG fields."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.ws import _handle_notebook_sync

    notebook_dir, _ = temp_notebook
    session = open_session(notebook_dir)

    executor = CellExecutor(session)
    assert (await executor.execute_cell("root", "x = 1")).success
    root = next(c for c in session.notebook_state.cells if c.id == "root")
    root.source = "x = 2"
    write_cell(notebook_dir, "root", "x = 2")
    session.re_analyze_cell("root")
    session.compute_staleness()

    fake, _ = _make_fake_ws(session)
    await _handle_notebook_sync(cast(WebSocket, fake), session, session.id)

    state = fake.frames_of("notebook_state")[-1]["payload"]
    root = next(cell for cell in state["cells"] if cell["id"] == "root")
    assert root["status"] == "idle"
    assert "staleness_reasons" in root
    assert root["causality"]["reason"] == "self"


@pytest.mark.asyncio
async def test_notebook_sync_includes_remote_execution_metadata(
    temp_notebook,
    notebook_executor_server,
    notebook_build_server,
):
    """Notebook sync should retain remote execution metadata from the live session."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.ws import _handle_notebook_sync

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

    executor = CellExecutor(session)
    assert (await executor.execute_cell("root", "x = 1")).success

    fake, _ = _make_fake_ws(session)
    await _handle_notebook_sync(cast(WebSocket, fake), session, session.id)

    state = fake.frames_of("notebook_state")[-1]["payload"]
    root = next(cell for cell in state["cells"] if cell["id"] == "root")
    assert root["execution_method"] == "executor"
    assert root["remote_worker"] == "gpu-http-signed"
    assert root["remote_transport"] == "signed"
    assert isinstance(root["remote_build_id"], str)
    assert root["remote_build_state"] == "ready"
    assert root["remote_error_code"] is None


# ---------------------------------------------------------------------------
# cell_execute / display payloads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cell_execute_no_cascade(notebook_session):
    """cell_execute on a root cell does not trigger cascade."""
    from strata.notebook.ws import _handle_cell_execute

    _, session = notebook_session
    root_cell = next(
        (c for c in session.notebook_state.cells if not c.upstream_ids),
        session.notebook_state.cells[0],
    )

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": root_cell.id}, execution_state, session.id
    )
    await _drain_execution(execution_state)

    running = [
        f
        for f in fake.frames_of("cell_status")
        if f["payload"].get("status") == "running" and f["payload"]["cell_id"] == root_cell.id
    ]
    assert running
    assert fake.frames_of("cell_output") or fake.frames_of("cell_error")
    # A root cell has no upstreams, so no cascade prompt.
    assert not fake.frames_of("cascade_prompt")


async def _drain_execution(execution_state):
    """Await the background execution task scheduled by a handler, if any."""
    task = execution_state.execution_task
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)


async def _run_cell_to_terminal(session, cell_id, *, msg_type="cell_execute", payload=None):
    """Drive ``cell_id`` through a handler and await its execution task.

    Returns the registered fake WS so the caller can inspect emitted frames.
    Handles the cascade prompt → auto-accept loop for non-root cells.
    """
    from strata.notebook.ws import (
        _handle_cell_execute,
        _handle_cell_execute_cascade,
        _handle_notebook_run_all,
    )

    fake, execution_state = _make_fake_ws(session)
    if msg_type == "notebook_run_all":
        await _handle_notebook_run_all(
            cast(WebSocket, fake), session, execution_state, session.id, payload or {}
        )
        await _drain_execution(execution_state)
        return fake

    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": cell_id}, execution_state, session.id
    )
    await _drain_execution(execution_state)

    prompts = fake.frames_of("cascade_prompt")
    if prompts:
        plan_id = prompts[-1]["payload"]["plan_id"]
        await _handle_cell_execute_cascade(
            cast(WebSocket, fake),
            session,
            {"cell_id": cell_id, "plan_id": plan_id},
            execution_state,
            session.id,
        )
        await _drain_execution(execution_state)
    return fake


@pytest.mark.asyncio
async def test_cell_execute_emits_explicit_display_payload(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    assert output["type"] == "cell_output"
    assert output["payload"]["display"]["content_type"] == "image/png"
    assert output["payload"]["display"]["inline_data_url"].startswith("data:image/png;base64,")
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_cell_execute_emits_explicit_markdown_display_payload(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    assert output["type"] == "cell_output"
    assert output["payload"]["display"]["content_type"] == "text/markdown"
    assert output["payload"]["display"]["markdown_text"] == "# Title\n\nRendered over websocket."
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_cell_execute_emits_display_side_effect_payload(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    assert output["type"] == "cell_output"
    assert output["payload"]["display"]["content_type"] == "text/markdown"
    assert output["payload"]["display"]["markdown_text"] == "# Side effect\n\nVia websocket."
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_cell_execute_emits_multiple_display_payloads_in_order(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    payload = output["payload"]
    assert output["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 42
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 42
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_cell_execute_refreshes_downstream_staleness(temp_notebook):
    """Successful execution should immediately invalidate downstream cell state."""
    from strata.notebook.executor import CellExecutor

    notebook_dir, _ = temp_notebook
    session = open_session(notebook_dir)

    executor = CellExecutor(session)
    assert (await executor.execute_cell("root", "x = 1")).success
    assert (await executor.execute_cell("middle", "y = x + 1")).success
    assert (await executor.execute_cell("leaf", "z = y + 1")).success
    session.compute_staleness()

    root = next(c for c in session.notebook_state.cells if c.id == "root")
    root.source = "x = 2"
    write_cell(notebook_dir, "root", "x = 2")
    session.re_analyze_cell("root")

    fake = await _run_cell_to_terminal(session, "root")

    status_updates = [f["payload"] for f in fake.frames_of("cell_status")]
    assert any(p["cell_id"] == "root" and p["status"] == "ready" for p in status_updates)
    assert any(p["cell_id"] == "middle" and p["status"] == "idle" for p in status_updates)
    assert any(p["cell_id"] == "leaf" and p["status"] == "idle" for p in status_updates)


@pytest.mark.asyncio
async def test_cell_execute_surfaces_module_export_error(temp_notebook):
    """Unsupported cross-cell code export should surface as a direct cell error."""
    notebook_dir, _ = temp_notebook
    # ``x = len([])`` is a non-literal runtime assignment; plain literal
    # constants (``x = 1``) would now export fine alongside the def.
    write_cell(notebook_dir, "root", "x = len([])\n\ndef add(y):\n    return x + y\n")
    write_cell(notebook_dir, "middle", "result = add(2)")
    session = open_session(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    assert any(
        f["payload"].get("status") == "running" and f["payload"]["cell_id"] == "root"
        for f in fake.frames_of("cell_status")
    )
    assert output["type"] == "cell_error"
    assert "cannot be shared across cells yet" in output["payload"]["error"]
    # The slicer pinpoints the unresolved free var and the symbol that depends on it.
    assert "function `add`" in output["payload"]["error"]
    assert "x" in output["payload"]["error"]
    assert terminal["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_cell_execute_surfaces_module_export_lambda_error(temp_notebook):
    """The WS path should surface top-level lambda export errors clearly."""
    notebook_dir, _ = temp_notebook
    write_cell(notebook_dir, "root", "add = lambda y: y + 1\n")
    write_cell(notebook_dir, "middle", "result = add(2)")
    session = open_session(notebook_dir)
    session.re_analyze_cell("root")
    session.re_analyze_cell("middle")

    fake = await _run_cell_to_terminal(session, "root")
    output, terminal = _terminal_frames(fake, "root")

    assert output["type"] == "cell_error"
    assert "cannot be shared across cells yet" in output["payload"]["error"]
    assert "top-level lambdas are not shareable across cells" in output["payload"]["error"]
    assert terminal["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_cell_execute_uses_warm_pool_when_available(notebook_session, monkeypatch):
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

    fake = await _run_cell_to_terminal(session, root_cell.id)
    output, terminal = _terminal_frames(fake, root_cell.id)

    assert output["type"] == "cell_output"
    assert output["payload"]["execution_method"] == "warm"
    assert terminal["payload"]["status"] == "ready"
    assert warm_calls == 1


@pytest.mark.asyncio
async def test_notebook_run_all_emits_multiple_display_payloads_in_order(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "root", msg_type="notebook_run_all")
    output, terminal = _terminal_frames(fake, "root")

    payload = output["payload"]
    assert output["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 42
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 42
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_cell_execute_cascade_emits_multiple_display_payloads_in_order(temp_notebook):
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

    fake = await _run_cell_to_terminal(session, "leaf")
    assert fake.frames_of("cascade_prompt"), "leaf has stale upstreams; expected a cascade prompt"
    output, terminal = _terminal_frames(fake, "leaf")

    payload = output["payload"]
    assert output["type"] == "cell_output"
    assert len(payload["displays"]) == 2
    assert payload["displays"][0]["content_type"] == "text/markdown"
    assert payload["displays"][0]["markdown_text"] == "# First"
    assert payload["displays"][1]["content_type"] == "json/object"
    assert payload["displays"][1]["preview"] == 3
    assert payload["display"]["content_type"] == "json/object"
    assert payload["display"]["preview"] == 3
    assert terminal["payload"]["status"] == "ready"


# ---------------------------------------------------------------------------
# Environment-busy / blocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cell_execute_blocked_when_environment_runtime_is_unavailable(notebook_session):
    """Execution should be blocked when no notebook runtime is available after bootstrap failure."""
    from strata.notebook.ws import _handle_cell_execute

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    session.environment_job = None
    session.venv_python = None
    session.environment_interpreter_source = "unknown"
    session.environment_sync_state = "failed"
    session.environment_sync_error = "Failed to start notebook environment initialization: boom"
    session.environment_sync_notice = None

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": cell_id}, execution_state, session.id
    )

    errors = fake.frames_of("error")
    assert errors
    assert errors[-1]["payload"]["code"] == "ENVIRONMENT_BUSY"
    assert "environment" in errors[-1]["payload"]["error"].lower()


@pytest.mark.asyncio
async def test_cell_execute_blocked_while_environment_job_running(notebook_session):
    """Cell execution should be rejected while an environment job is active."""
    from strata.notebook.session import EnvironmentJobSnapshot
    from strata.notebook.ws import _handle_cell_execute

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

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": cell_id}, execution_state, session.id
    )

    errors = fake.frames_of("error")
    assert errors
    assert errors[-1]["payload"]["code"] == "ENVIRONMENT_BUSY"


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

    monkeypatch.setattr(notebook_ws, "_schedule_execution", _gated_schedule)

    async def _noop_environment_job(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(session, "_run_environment_job", _noop_environment_job)

    async def _exercise() -> None:
        fake = FakeNotebookWebSocket()
        execute_task = asyncio.create_task(
            notebook_ws._handle_cell_execute(
                cast(WebSocket, fake),
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


# ---------------------------------------------------------------------------
# Remote-executor consumers
# ---------------------------------------------------------------------------


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


@pytest.mark.asyncio
async def test_ws_execute_supports_http_executor_worker(notebook_session, notebook_executor_server):
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

    fake = await _run_cell_to_terminal(session, root_cell.id)
    output, terminal = _terminal_frames(fake, root_cell.id)

    payload = output["payload"]
    assert output["type"] == "cell_output"
    assert payload["execution_method"] == "executor"
    assert payload["remote_worker"] == "gpu-http"
    assert payload["remote_transport"] == "direct"
    assert payload["outputs"]["x"]["preview"] == 1
    assert terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_ws_execute_supports_signed_http_executor_worker(
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

    fake = await _run_cell_to_terminal(session, root_cell.id)
    first_output, first_terminal = _terminal_frames(fake, root_cell.id)

    first_payload = first_output["payload"]
    assert first_output["type"] == "cell_output"
    assert first_payload["execution_method"] == "executor"
    assert first_payload["remote_worker"] == "gpu-http-signed"
    assert first_payload["remote_transport"] == "signed"
    assert isinstance(first_payload["remote_build_id"], str)
    assert first_payload["outputs"]["x"]["preview"] == 1
    assert first_terminal["payload"]["status"] == "ready"

    second = await _run_cell_to_terminal(session, root_cell.id)
    second_output, second_terminal = _terminal_frames(second, root_cell.id)

    second_payload = second_output["payload"]
    assert second_output["type"] == "cell_output"
    assert second_payload["execution_method"] == "cached"
    assert second_payload["remote_worker"] == "gpu-http-signed"
    assert second_payload["remote_transport"] == "signed"
    assert "remote_build_id" not in second_payload
    assert second_terminal["payload"]["status"] == "ready"


@pytest.mark.asyncio
async def test_ws_execute_supports_signed_http_executor_worker_with_class_instances(
    notebook_executor_server,
    notebook_build_server,
):
    """The live WS path should preserve exported class instances over signed transport."""
    from strata.notebook.ws import _handle_notebook_sync

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

        for cell_id in ("cell1", "cell2", "cell3"):
            fake = await _run_cell_to_terminal(session, cell_id)
            output, terminal = _terminal_frames(fake, cell_id)
            payload = output["payload"]
            assert output["type"] == "cell_output"
            assert payload["execution_method"] == "executor"
            assert payload["remote_worker"] == "gpu-http-signed"
            assert payload["remote_transport"] == "signed"
            assert payload["remote_build_state"] == "ready"
            assert terminal["payload"]["status"] == "ready"

        sync_fake, _ = _make_fake_ws(session)
        await _handle_notebook_sync(cast(WebSocket, sync_fake), session, session.id)

        state = sync_fake.frames_of("notebook_state")[-1]["payload"]
        cell2 = next(cell for cell in state["cells"] if cell["id"] == "cell2")
        cell3 = next(cell for cell in state["cells"] if cell["id"] == "cell3")

        assert "p" in cell2["artifact_uris"]
        assert cell2["remote_transport"] == "signed"
        assert cell2["remote_build_state"] == "ready"
        assert cell2["status"] == "ready"
        assert cell3["remote_transport"] == "signed"
        assert cell3["remote_build_state"] == "ready"
        assert cell3["status"] == "ready"


@pytest.mark.asyncio
async def test_ws_execute_reports_unavailable_http_executor_worker(notebook_session):
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

    fake = await _run_cell_to_terminal(session, root_cell.id)
    output, terminal = _terminal_frames(fake, root_cell.id)

    assert output["type"] == "cell_error"
    assert "Remote executor request failed" in output["payload"]["error"]
    assert terminal["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_ws_execute_reports_signed_finalize_failure(
    notebook_session,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """The live WS path should surface signed transport finalize failures."""
    from strata.transforms.signed_urls import URLSigner

    _real_generate = URLSigner.generate_build_manifest

    _, session = notebook_session

    class _BadFinalizeManifest:
        def __init__(self, manifest):
            self._manifest = manifest

        def to_dict(self):
            data = self._manifest.to_dict()
            data["finalize_url"] = f"{data['finalize_url']}/missing-finalize"
            return data

    def fake_generate_build_manifest(self, *args, **kwargs):
        return _BadFinalizeManifest(_real_generate(self, *args, **kwargs))

    monkeypatch.setattr(URLSigner, "generate_build_manifest", fake_generate_build_manifest)

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

    fake = await _run_cell_to_terminal(session, root_cell.id)
    output, terminal = _terminal_frames(fake, root_cell.id)

    payload = output["payload"]
    assert output["type"] == "cell_error"
    assert "Failed to finalize notebook bundle build" in payload["error"]
    assert payload["remote_worker"] == "gpu-http-signed"
    assert payload["remote_transport"] == "signed"
    assert isinstance(payload["remote_build_id"], str)
    assert payload["remote_build_state"] == "failed"
    assert payload["remote_error_code"] == "FINALIZE_FAILED"
    assert terminal["payload"]["status"] == "error"


@pytest.mark.asyncio
async def test_ws_cancelled_signed_http_executor_marks_build_failed(
    notebook_session,
    notebook_executor_server,
    notebook_build_server,
    monkeypatch,
):
    """Cancelling signed remote execution over WS should fail the build cleanly."""
    from strata.notebook.ws import _handle_cell_cancel, _handle_cell_execute

    _, session = notebook_session
    started = asyncio.Event()

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

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": root_cell.id}, execution_state, session.id
    )

    # Let the scheduled task broadcast ``running`` and reach the slow harness.
    await _wait_until(lambda: bool(_running_frames(fake, root_cell.id)))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    await _handle_cell_cancel(session, {"cell_id": root_cell.id}, execution_state, session.id)

    idle = [
        f
        for f in fake.frames_of("cell_status")
        if f["payload"].get("cell_id") == root_cell.id and f["payload"]["status"] == "idle"
    ]
    assert idle

    # The cancelled signed build must be marked failed (not left pending/building).
    # The finalize-failed marking happens on a background server thread; wait
    # for it to settle via the build store rather than asserting on timing.
    await _wait_until(
        lambda: notebook_build_server["build_store"].get_stats()["building"] == 0
        and notebook_build_server["build_store"].get_stats()["pending"] == 0
    )
    stats = notebook_build_server["build_store"].get_stats()
    assert stats["failed"] == 1
    assert stats["pending"] == 0
    assert stats["building"] == 0


# ---------------------------------------------------------------------------
# Per-socket scoping of cascade / impact responses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_prompt_is_sent_only_to_requesting_websocket(notebook_session):
    """A cascade prompt should be a direct reply, not a broadcast to other clients."""
    from strata.notebook.ws import _handle_cell_execute, _notebook_connections

    _, session = notebook_session
    requester, execution_state = _make_fake_ws(session)
    observer = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, observer))

    await _handle_cell_execute(
        cast(WebSocket, requester), session, {"cell_id": "middle"}, execution_state, session.id
    )

    prompts = requester.frames_of("cascade_prompt")
    assert prompts
    assert prompts[-1]["payload"]["cell_id"] == "middle"
    # The cascade prompt goes out via ``_send_message`` (direct), so the
    # second client must never see it.
    assert observer.frames_of("cascade_prompt") == []


@pytest.mark.asyncio
async def test_impact_preview_is_sent_only_to_requesting_websocket(notebook_session):
    """Impact preview responses should stay scoped to the requesting client."""
    from strata.notebook.ws import _handle_impact_preview_request, _notebook_connections

    _, session = notebook_session
    requester, execution_state = _make_fake_ws(session)
    observer = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, observer))

    await _handle_impact_preview_request(
        cast(WebSocket, requester), session, {"cell_id": "middle"}, execution_state, session.id
    )

    previews = requester.frames_of("impact_preview")
    assert previews
    assert previews[-1]["payload"]["target_cell_id"] == "middle"
    assert observer.frames_of("impact_preview") == []


# ---------------------------------------------------------------------------
# Inspect REPL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspect_repl_round_trip(notebook_session):
    """The inspect REPL round-trips over the websocket."""
    from strata.notebook.ws import (
        _handle_inspect_close,
        _handle_inspect_eval,
        _handle_inspect_open,
    )

    _, session = notebook_session
    middle_cell = next(c for c in session.notebook_state.cells if "x" in c.references)

    fake, execution_state = _make_fake_ws(session)

    await _handle_inspect_open(
        cast(WebSocket, fake),
        session,
        {"cell_id": middle_cell.id},
        execution_state,
        session.id,
    )
    opened = fake.frames_of("inspect_result")[-1]["payload"]
    assert opened["action"] == "open"
    assert opened["ok"] is True

    await _handle_inspect_eval(
        cast(WebSocket, fake),
        session,
        {"cell_id": middle_cell.id, "expr": "x + 1"},
        execution_state,
        session.id,
    )
    evaluated = fake.frames_of("inspect_result")[-1]["payload"]
    assert evaluated["action"] == "eval"
    assert evaluated["ok"] is True
    assert evaluated["result"] == "2"
    assert evaluated["type"] == "int"

    await _handle_inspect_close(
        cast(WebSocket, fake),
        session,
        {"cell_id": middle_cell.id},
        execution_state,
        session.id,
    )
    closed = fake.frames_of("inspect_result")[-1]["payload"]
    assert closed["action"] == "close"
    assert closed["ok"] is True


def _patch_inspect_manager(monkeypatch, *, close_counter):
    """Patch InspectManager.open_session / close_all and count close calls."""
    from strata.notebook.inspect_repl import InspectManager

    async def fake_open_session(self, cell_id, notebook_session):
        return SimpleNamespace(ready=True), "ready"

    async def fake_close_all(self):
        close_counter["count"] += 1

    monkeypatch.setattr(InspectManager, "open_session", fake_open_session)
    monkeypatch.setattr(InspectManager, "close_all", fake_close_all)


@pytest.mark.asyncio
async def test_inspect_sessions_closed_when_last_websocket_disconnects(
    notebook_session, monkeypatch
):
    """Tearing down the last socket should close notebook inspect sessions.

    The zero-grace teardown path (``_tear_down_notebook_state``) is what
    ``_cleanup_notebook_websocket`` invokes inline when the grace window is
    disabled or no loop is running; exercise it directly.
    """
    from strata.notebook.ws import (
        _handle_inspect_open,
        _notebook_inspect_managers,
        _tear_down_notebook_state,
    )

    _, session = notebook_session
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    middle_cell = next(c for c in session.notebook_state.cells if "x" in c.references)
    fake, execution_state = _make_fake_ws(session)

    await _handle_inspect_open(
        cast(WebSocket, fake),
        session,
        {"cell_id": middle_cell.id},
        execution_state,
        session.id,
    )
    assert fake.frames_of("inspect_result")[-1]["payload"]["ok"] is True

    # The last disconnect tears down notebook state (zero-grace path).
    await _tear_down_notebook_state(session.id)

    assert close_counter["count"] == 1
    assert session.id not in _notebook_inspect_managers


# ---------------------------------------------------------------------------
# Grace window (disconnect / reconnect)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grace_window_preserves_inspect_state_on_reconnect(notebook_session, monkeypatch):
    """Disconnect-then-reconnect within the grace window keeps inspect state.

    Without the grace window, the inspect manager would be dropped the moment
    the WS disconnects. The TUI's tmux / SSH audience would lose any open
    inspect REPL on a network blip — bad UX for exactly the users the window
    exists for.
    """
    from strata.notebook.ws import (
        _cancel_pending_grace_teardown,
        _cleanup_notebook_websocket,
        _handle_inspect_open,
        _notebook_grace_tasks,
        _notebook_inspect_managers,
    )

    _, session = notebook_session
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    cell = next(c for c in session.notebook_state.cells if "x" in c.references)
    fake, execution_state = _make_fake_ws(session)
    await _handle_inspect_open(
        cast(WebSocket, fake), session, {"cell_id": cell.id}, execution_state, session.id
    )

    # Last socket disconnects → grace task scheduled, state preserved.
    await _cleanup_notebook_websocket(session.id, cast(WebSocket, fake))
    assert session.id in _notebook_grace_tasks
    assert session.id in _notebook_inspect_managers
    assert close_counter["count"] == 0

    # Reconnect cancels the pending teardown — inspect state survives.
    _cancel_pending_grace_teardown(session.id)
    grace_task = _notebook_grace_tasks.pop(session.id, None)
    if grace_task is not None:
        await asyncio.gather(grace_task, return_exceptions=True)

    assert session.id in _notebook_inspect_managers
    assert close_counter["count"] == 0


@pytest.mark.asyncio
async def test_grace_window_expires_drops_state(notebook_session, monkeypatch):
    """When the grace window elapses with no reconnect, state is dropped."""
    from strata.notebook.ws import (
        _grace_cancel_then_tear_down,
        _handle_inspect_open,
        _notebook_inspect_managers,
    )

    _, session = notebook_session
    close_counter = {"count": 0}
    _patch_inspect_manager(monkeypatch, close_counter=close_counter)

    cell = next(c for c in session.notebook_state.cells if "x" in c.references)
    fake, execution_state = _make_fake_ws(session)
    await _handle_inspect_open(
        cast(WebSocket, fake), session, {"cell_id": cell.id}, execution_state, session.id
    )
    assert session.id in _notebook_inspect_managers

    # Simulate the last socket having gone away (no connections), then run
    # the grace-teardown body directly with a zero wait — no polling.
    from strata.notebook.ws import _notebook_connections

    _notebook_connections.pop(session.id, None)
    await _grace_cancel_then_tear_down(session.id, 0.0)

    assert close_counter["count"] == 1
    assert session.id not in _notebook_inspect_managers


@pytest.mark.asyncio
async def test_grace_window_preserves_active_execution_on_reconnect(notebook_session, monkeypatch):
    """A running execution task survives a disconnect-reconnect cycle within the window.

    Load-bearing #42 contract: tmux detach during a long cell, reconnect
    within the window, find the cell still running. This targets
    ``NotebookExecutionState.execution_task`` — the actual mechanism behind
    cancel-on-disconnect — rather than the inspect cleanup hook.
    """
    from strata.notebook.ws import (
        _cancel_pending_grace_teardown,
        _cleanup_notebook_websocket,
        _ensure_execution_state,
        _notebook_execution_state,
        _notebook_grace_tasks,
    )

    _, session = notebook_session
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 30.0)

    async def long_sleep() -> None:
        await asyncio.sleep(60)

    fake, _ = _make_fake_ws(session)
    state = _ensure_execution_state(session.id)
    task = asyncio.create_task(long_sleep())
    state.execution_task = task
    try:
        # Last socket disconnects → grace task scheduled, execution preserved.
        await _cleanup_notebook_websocket(session.id, cast(WebSocket, fake))
        assert not task.done(), "execution task was cancelled before the grace window expired"
        assert _notebook_execution_state.get(session.id) is not None
        assert session.id in _notebook_grace_tasks

        # Reconnect cancels the pending teardown — task stays alive.
        _cancel_pending_grace_teardown(session.id)
        grace_task = _notebook_grace_tasks.pop(session.id, None)
        if grace_task is not None:
            await asyncio.gather(grace_task, return_exceptions=True)
        assert not task.done()
        assert _notebook_execution_state.get(session.id) is not None
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_grace_window_expiry_cancels_active_execution(notebook_session):
    """Past the grace window with no reconnect, the running execution task is cancelled."""
    from strata.notebook.ws import (
        _ensure_execution_state,
        _grace_cancel_then_tear_down,
        _notebook_connections,
        _notebook_execution_state,
    )

    _, session = notebook_session

    async def long_sleep() -> None:
        await asyncio.sleep(60)

    state = _ensure_execution_state(session.id)
    task = asyncio.create_task(long_sleep())
    state.execution_task = task

    # No connections remain; run the grace-teardown body directly with a
    # zero wait so the running execution task is cancelled deterministically.
    _notebook_connections.pop(session.id, None)
    await _grace_cancel_then_tear_down(session.id, 0.0)

    assert task.done()
    assert task.cancelled()
    assert session.id not in _notebook_execution_state


@pytest.mark.asyncio
async def test_last_websocket_disconnect_cancels_running_execution(notebook_session):
    """Closing the final socket should cancel the active notebook execution."""
    from strata.notebook.ws import (
        _ensure_execution_state,
        _notebook_execution_state,
        _tear_down_notebook_state,
    )

    _, session = notebook_session
    cancelled = asyncio.Event()
    entered = asyncio.Event()

    async def long_sleep() -> None:
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    state = _ensure_execution_state(session.id)
    task = asyncio.create_task(long_sleep())
    state.execution_task = task

    # Let the task reach its sleep so the cancel unwinds through its handler.
    await _wait_until(entered.is_set)

    # Zero-grace teardown cancels the in-flight execution and drops state.
    await _tear_down_notebook_state(session.id)

    assert cancelled.is_set()
    assert session.id not in _notebook_execution_state


# ---------------------------------------------------------------------------
# Source update / cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cell_source_update(notebook_session):
    """cell_source_update triggers DAG recomputation."""
    from strata.notebook.ws import _handle_cell_source_update

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_source_update(
        cast(WebSocket, fake),
        session,
        {"cell_id": cell_id, "source": "x = 2\ny = 3"},
        execution_state,
        session.id,
    )

    dag_updates = fake.frames_of("dag_update")
    assert dag_updates
    payload = dag_updates[-1]["payload"]
    assert "edges" in payload
    assert "topological_order" in payload
    for status in fake.frames_of("cell_status"):
        assert "status" in status["payload"]


@pytest.mark.asyncio
async def test_cell_cancel(notebook_session):
    """cell_cancel with no running execution marks the cell idle."""
    from strata.notebook.ws import _handle_cell_cancel

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_cancel(session, {"cell_id": cell_id}, execution_state, session.id)

    statuses = fake.frames_of("cell_status")
    assert statuses
    assert statuses[-1]["payload"]["status"] == "idle"


@pytest.mark.asyncio
async def test_cell_cancel_interrupts_running_execution(notebook_session, monkeypatch):
    """A cell_cancel must cancel its own in-flight execution."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.ws import _handle_cell_cancel, _handle_cell_execute

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id
    cancelled = asyncio.Event()

    async def fake_execute_cell(self, cell_id: str, source: str, timeout_seconds: float = 30):
        del self, cell_id, source, timeout_seconds
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(CellExecutor, "execute_cell", fake_execute_cell)

    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(
        cast(WebSocket, fake), session, {"cell_id": cell_id}, execution_state, session.id
    )

    # Let the scheduled background task broadcast ``running`` and reach its
    # long sleep before we cancel it.
    await _wait_until(lambda: bool(_running_frames(fake, cell_id)))

    await _handle_cell_cancel(session, {"cell_id": cell_id}, execution_state, session.id)

    idle = [
        f
        for f in fake.frames_of("cell_status")
        if f["payload"].get("cell_id") == cell_id and f["payload"]["status"] == "idle"
    ]
    assert idle
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_stale_cell_cancel_does_not_clobber_ready_state(notebook_session):
    """A late cancel should not rewrite a completed cell back to idle."""
    from strata.notebook.ws import _handle_cell_cancel

    _, session = notebook_session
    cell_id = session.notebook_state.cells[0].id

    fake = await _run_cell_to_terminal(session, cell_id)
    _, terminal = _terminal_frames(fake, cell_id)
    assert terminal["payload"]["status"] == "ready"

    cancel_fake, execution_state = _make_fake_ws(session)
    await _handle_cell_cancel(session, {"cell_id": cell_id}, execution_state, session.id)

    idle = [
        f
        for f in cancel_fake.frames_of("cell_status")
        if f["payload"].get("cell_id") == cell_id and f["payload"].get("status") == "idle"
    ]
    assert idle == []
    cell = session.notebook_state.get_cell(cell_id)
    assert cell.status.value == "ready"


# ---------------------------------------------------------------------------
# Malformed dispatch / variant add
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_malformed_cell_execute_missing_cell_id(notebook_session):
    """cell_execute with no cell_id produces a generic error frame."""
    from strata.notebook.ws import _handle_cell_execute

    _, session = notebook_session
    fake, execution_state = _make_fake_ws(session)
    await _handle_cell_execute(cast(WebSocket, fake), session, {}, execution_state, session.id)

    errors = fake.frames_of("error")
    assert errors
    assert "error" in errors[-1]["payload"]


@pytest.mark.asyncio
async def test_variant_add_broadcasts_new_cell():
    """variant_add must broadcast notebook_state (with the new cell's full
    payload), not just dag_update — otherwise the frontend skips the new
    cell and the user sees the active variant 'disappear'."""
    from strata.notebook.ws import _handle_variant_add

    with tempfile.TemporaryDirectory() as tmpdir:
        notebook_dir = create_notebook(Path(tmpdir), "variant_ws", initialize_environment=False)
        add_cell_to_notebook(notebook_dir, "model_a")
        write_cell(notebook_dir, "model_a", "# @variant g a\npreds = 1\n")

        session = open_session(notebook_dir)
        fake, execution_state = _make_fake_ws(session)
        await _handle_variant_add(
            cast(WebSocket, fake), session, {"group": "g"}, execution_state, session.id
        )

    states = fake.frames_of("notebook_state")
    assert states
    cells = states[-1]["payload"]["cells"]
    new_cells = [c for c in cells if c.get("variant_name") == "a_copy"]
    assert len(new_cells) == 1
    assert "# @variant g a_copy" in new_cells[0]["source"]
    assert "preds = 1" in new_cells[0]["source"]
    # New variant is active; old becomes inactive.
    assert new_cells[0]["variant_active"] is True
    old = next(c for c in cells if c.get("variant_name") == "a")
    assert old["variant_active"] is False


# ---------------------------------------------------------------------------
# Endpoint-level: notebook_websocket upgrade + owner gate (#52)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_notebook_closes_upgrade(monkeypatch):
    """Connecting to a non-existent notebook should close the upgrade, not accept."""
    from strata.notebook import ws as notebook_ws

    monkeypatch.setattr(
        notebook_ws, "_get_session_manager", lambda: SimpleNamespace(get_session=lambda _id: None)
    )

    fake = FakeNotebookWebSocket()
    await notebook_ws.notebook_websocket(cast(WebSocket, fake), "nonexistent")

    assert fake.accepted is False
    assert fake.closed == (1008, "Notebook not found")


@pytest.mark.asyncio
async def test_endpoint_dispatch_handles_malformed_message(notebook_session):
    """Driving the endpoint with a scripted malformed message returns an error frame."""
    from strata.notebook.ws import notebook_websocket

    _, session = notebook_session
    fake = FakeNotebookWebSocket(inbound=[_envelope("cell_execute")])  # missing cell_id

    await notebook_websocket(cast(WebSocket, fake), session.id)

    assert fake.accepted is True
    assert fake.frames_of("error")
    assert "error" in fake.frames_of("error")[-1]["payload"]


@pytest.mark.asyncio
async def test_endpoint_dispatch_unknown_message_type(notebook_session):
    """An unknown message type yields an 'Unknown message type' error frame."""
    from strata.notebook.ws import notebook_websocket

    _, session = notebook_session
    fake = FakeNotebookWebSocket(inbound=[_envelope("not_a_real_type")])

    await notebook_websocket(cast(WebSocket, fake), session.id)

    errors = fake.frames_of("error")
    assert errors
    assert "Unknown message type" in errors[-1]["payload"]["error"]


class TestWebSocketOwnerGate:
    """Direct WS-upgrade owner-gate coverage (#52).

    Mirrors ``test_routes::TestPersonalModeUserScoping`` for the WS surface:
    owner allowed, wrong-header refused, missing-header refused, legacy
    unowned passthrough. Driven through ``notebook_websocket`` against a fake
    WS so there's no anyio portal — the gate closes with 1008 (the WS twin of
    the REST 404) and never accepts on refusal.
    """

    HEADER = "X-Strata-Test-User"

    def _configure_scoping(self, monkeypatch):
        """Set server state so ``personal_mode_user_header`` is active."""
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(personal_mode_user_header=self.HEADER, transforms_config={})
            ),
        )

    @pytest.mark.asyncio
    async def test_owner_allowed(self, notebook_session, monkeypatch):
        """The owner connecting with a matching header is accepted."""
        from strata.notebook.ws import notebook_websocket

        _, session = notebook_session
        session.notebook_state.owner = "alice@example.com"
        self._configure_scoping(monkeypatch)

        fake = FakeNotebookWebSocket(headers={self.HEADER: "alice@example.com"})
        await notebook_websocket(cast(WebSocket, fake), session.id)

        assert fake.accepted is True
        assert fake.closed is None

    @pytest.mark.asyncio
    async def test_wrong_header_refused(self, notebook_session, monkeypatch):
        """A non-owner identity is refused with a generic not-found close."""
        from strata.notebook.ws import notebook_websocket

        _, session = notebook_session
        session.notebook_state.owner = "alice@example.com"
        self._configure_scoping(monkeypatch)

        fake = FakeNotebookWebSocket(headers={self.HEADER: "bob@example.com"})
        await notebook_websocket(cast(WebSocket, fake), session.id)

        assert fake.accepted is False
        assert fake.closed == (1008, "Notebook not found")

    @pytest.mark.asyncio
    async def test_missing_header_refused(self, notebook_session, monkeypatch):
        """When scoping is on, omitting the identity header must not bypass the gate."""
        from strata.notebook.ws import notebook_websocket

        _, session = notebook_session
        session.notebook_state.owner = "alice@example.com"
        self._configure_scoping(monkeypatch)

        fake = FakeNotebookWebSocket()  # no identity header
        await notebook_websocket(cast(WebSocket, fake), session.id)

        assert fake.accepted is False
        assert fake.closed == (1008, "Notebook not found")

    @pytest.mark.asyncio
    async def test_legacy_unowned_notebook_passthrough(self, notebook_session, monkeypatch):
        """An ``owner = None`` notebook stays accessible to any caller, even with scoping on."""
        from strata.notebook.ws import notebook_websocket

        _, session = notebook_session
        session.notebook_state.owner = None
        self._configure_scoping(monkeypatch)

        fake = FakeNotebookWebSocket()  # no header at all
        await notebook_websocket(cast(WebSocket, fake), session.id)

        assert fake.accepted is True
        assert fake.closed is None


class TestWsOwnerAllowedHelper:
    """Unit coverage for the extracted ``_ws_owner_allowed`` decision helper."""

    def _scoping(self, monkeypatch, *, enabled):
        if enabled:
            monkeypatch.setattr(
                "strata.server._state",
                SimpleNamespace(
                    config=SimpleNamespace(personal_mode_user_header="X-User", transforms_config={})
                ),
            )
        else:
            monkeypatch.setattr(
                "strata.server._state",
                SimpleNamespace(
                    config=SimpleNamespace(personal_mode_user_header=None, transforms_config={})
                ),
            )

    def test_unowned_always_allowed(self, monkeypatch):
        from strata.notebook.ws import _ws_owner_allowed

        self._scoping(monkeypatch, enabled=True)
        assert _ws_owner_allowed(None, None) is True
        assert _ws_owner_allowed(None, "anyone") is True

    def test_owned_missing_caller_denied_when_scoping_on(self, monkeypatch):
        from strata.notebook.ws import _ws_owner_allowed

        self._scoping(monkeypatch, enabled=True)
        assert _ws_owner_allowed("alice", None) is False

    def test_owned_missing_caller_allowed_when_scoping_off(self, monkeypatch):
        from strata.notebook.ws import _ws_owner_allowed

        self._scoping(monkeypatch, enabled=False)
        assert _ws_owner_allowed("alice", None) is True

    def test_owned_mismatch_denied(self, monkeypatch):
        from strata.notebook.ws import _ws_owner_allowed

        self._scoping(monkeypatch, enabled=True)
        assert _ws_owner_allowed("alice", "bob") is False

    def test_owned_match_allowed(self, monkeypatch):
        from strata.notebook.ws import _ws_owner_allowed

        self._scoping(monkeypatch, enabled=True)
        assert _ws_owner_allowed("alice", "alice") is True


# ---------------------------------------------------------------------------
# _running_payload helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Prompt-cell streaming broadcast wiring (issue #110)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_make_executor_with_progress_wires_prompt_delta_broadcast(
    notebook_session, monkeypatch
):
    """``_make_executor_with_progress`` must wire ``on_prompt_delta`` to a
    ``cell_output_delta`` broadcast, mirroring the loop-progress wiring.
    Tested through the callback directly (fake broadcast, no WS upgrade)."""
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

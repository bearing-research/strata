"""End-to-end test fixtures for notebook WebSocket integration tests.

Provides helpers for creating test notebooks, managing WebSocket connections,
and collecting/asserting on WebSocket message sequences.
"""

from __future__ import annotations

import json
import tempfile
from collections import deque
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from strata.notebook.routes import get_session_manager
from strata.notebook.routes import router as notebook_router
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.notebook.ws import (
    _notebook_connections,
    _notebook_execution_state,
)
from strata.notebook.ws import (
    router as notebook_ws_router,
)

# ============================================================================
# WebSocket Test Helper
# ============================================================================


class FakeNotebookWebSocket:
    """In-process stand-in for a Starlette ``WebSocket`` for notebook tests.

    The notebook WS handlers only ever call ``send_text`` (the handlers
    broadcast through ``_broadcast_message`` / ``_send_message``, which both
    bottom out at ``send_text``); ``notebook_websocket`` additionally calls
    ``accept``, ``close``, ``receive_text``, and reads ``headers``. This fake
    implements exactly that surface so tests can drive handlers (and the
    endpoint) directly in the event loop — no anyio portal, no real upgrade.

    Drive an inbound message script by pushing raw JSON strings onto
    ``inbound``; ``receive_text`` pops them in order and raises
    ``WebSocketDisconnect`` once the queue drains (mirroring a client that
    closed the socket), which is how ``notebook_websocket`` exits its loop.
    """

    def __init__(
        self,
        *,
        inbound: list[str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.inbound: deque[str] = deque(inbound or [])
        self.headers: dict[str, str] = headers or {}
        self.raw_sent: list[str] = []
        self.accepted = False
        self.closed: tuple[int, str] | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)

    async def send_text(self, text: str) -> None:
        self.raw_sent.append(text)

    async def receive_text(self) -> str:
        if not self.inbound:
            raise WebSocketDisconnect(code=1000)
        return self.inbound.popleft()

    @property
    def sent(self) -> list[dict[str, Any]]:
        """Every sent frame, parsed from JSON, in emission order."""
        return [json.loads(text) for text in self.raw_sent]

    def frames_of(self, msg_type: str) -> list[dict[str, Any]]:
        """Return all sent frames whose ``type`` matches ``msg_type``."""
        return [frame for frame in self.sent if frame.get("type") == msg_type]


class WebSocketTestHelper:
    """Helper for sending/receiving WebSocket messages in tests.

    Wraps FastAPI TestClient WebSocket with convenience methods for
    the Strata Notebook protocol.
    """

    def __init__(self, ws):
        self.ws = ws
        self._seq = 0
        self.messages: list[dict[str, Any]] = []

    def send(self, msg_type: str, payload: dict[str, Any] | None = None) -> None:
        """Send a typed message with auto-incrementing seq."""
        self._seq += 1
        self.ws.send_json(
            {
                "type": msg_type,
                "seq": self._seq,
                "ts": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
                "payload": payload or {},
            }
        )

    def receive(self, timeout: float = 5.0) -> dict[str, Any]:
        """Receive one message and store it."""
        msg = self.ws.receive_json()
        self.messages.append(msg)
        return msg

    def receive_until(
        self,
        msg_type: str,
        *,
        cell_id: str | None = None,
        status: str | None = None,
        timeout: float = 10.0,
        max_messages: int = 50,
    ) -> dict[str, Any]:
        """Receive messages until one matches the given type (and optional filters).

        Returns the matching message. All received messages are stored in self.messages.
        """
        for _ in range(max_messages):
            msg = self.receive(timeout=timeout)
            if msg["type"] == msg_type:
                if cell_id and msg.get("payload", {}).get("cell_id") != cell_id:
                    continue
                if status and msg.get("payload", {}).get("status") != status:
                    continue
                return msg
        raise TimeoutError(
            f"Did not receive {msg_type}"
            f"{f' for cell {cell_id}' if cell_id else ''}"
            f"{f' with status={status}' if status else ''}"
            f" within {max_messages} messages. "
            f"Got: {[m['type'] for m in self.messages[-10:]]}"
        )

    def receive_all_of_type(self, msg_type: str, max_messages: int = 20) -> list[dict[str, Any]]:
        """Receive messages, collecting all of a given type.

        WARNING: This blocks until max_messages are consumed or an
        unrelated message arrives. Prefer receive_until() for most use cases.
        """
        collected = []
        for _ in range(max_messages):
            msg = self.receive()
            if msg["type"] == msg_type:
                collected.append(msg)
        return collected

    def messages_of_type(self, msg_type: str) -> list[dict[str, Any]]:
        """Return all stored messages of a given type."""
        return [m for m in self.messages if m["type"] == msg_type]

    def clear(self) -> None:
        """Clear stored messages."""
        self.messages.clear()

    def execute_cell(self, cell_id: str) -> None:
        """Send cell_execute message."""
        self.send("cell_execute", {"cell_id": cell_id})

    def execute_cascade(self, cell_id: str, plan_id: str) -> None:
        """Send cell_execute_cascade message."""
        self.send("cell_execute_cascade", {"cell_id": cell_id, "plan_id": plan_id})

    def execute_force(self, cell_id: str) -> None:
        """Send cell_execute_force message."""
        self.send("cell_execute_force", {"cell_id": cell_id})

    def execute_rerun(self, cell_id: str) -> None:
        """Send cell_execute_rerun message."""
        self.send("cell_execute_rerun", {"cell_id": cell_id})

    def update_source(self, cell_id: str, source: str) -> None:
        """Send cell_source_update message."""
        self.send("cell_source_update", {"cell_id": cell_id, "source": source})

    def sync(self) -> dict[str, Any]:
        """Send notebook_sync and return the notebook_state response."""
        self.send("notebook_sync")
        return self.receive_until("notebook_state")

    def run_all(self) -> None:
        """Send notebook_run_all message."""
        self.send("notebook_run_all")

    def rerun_all(self) -> None:
        """Send notebook_rerun_all message."""
        self.send("notebook_rerun_all")


# ============================================================================
# Notebook Builder
# ============================================================================


class NotebookBuilder:
    """Fluent builder for creating test notebooks with cells."""

    def __init__(self, parent_path: Path, name: str = "test_notebook"):
        self.notebook_dir = create_notebook(parent_path, name)
        self.cell_ids: list[str] = []

    def add_cell(self, cell_id: str, source: str, after: str | None = None) -> NotebookBuilder:
        """Add a cell with given source."""
        add_cell_to_notebook(self.notebook_dir, cell_id, after)
        write_cell(self.notebook_dir, cell_id, source)
        self.cell_ids.append(cell_id)
        return self

    @property
    def path(self) -> Path:
        return self.notebook_dir


# ============================================================================
# Fixtures
# ============================================================================


def _reset_ws_globals():
    """Reset global WebSocket state between tests."""
    _notebook_connections.clear()
    _notebook_execution_state.clear()


def create_test_app() -> FastAPI:
    """Create a fresh FastAPI app with notebook routes."""
    app = FastAPI()
    app.include_router(notebook_router)
    app.include_router(notebook_ws_router)
    return app


@pytest.fixture
def app():
    """Create FastAPI test app with notebook routes."""
    _reset_ws_globals()
    return create_test_app()


@pytest.fixture
def client(app):
    """Create TestClient for the app."""
    return TestClient(app)


@pytest.fixture
def tmp_notebook_dir():
    """Provide a temporary directory for notebook creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@contextmanager
def open_notebook_session(client: TestClient, notebook_dir: Path):
    """Context manager: open a notebook and yield (session_id, session).

    Uses the REST API to open the notebook, which registers it with the
    session manager.
    """
    session_manager = get_session_manager()
    session = session_manager.open_notebook(notebook_dir)
    yield session.id, session


@contextmanager
def ws_connect(client: TestClient, session_id: str):
    """Context manager: connect a WebSocket and yield a WebSocketTestHelper."""
    with client.websocket_connect(f"/v1/notebooks/ws/{session_id}") as ws:
        yield WebSocketTestHelper(ws)


def execute_cell_and_wait(
    helper: WebSocketTestHelper,
    cell_id: str,
) -> dict[str, Any]:
    """Execute a cell via WebSocket and wait for it to finish.

    Handles cascade auto-accept if needed.
    Returns the final cell_output or cell_error message.
    """
    helper.execute_cell(cell_id)
    saw_running = False
    terminal_message: dict[str, Any] | None = None

    # Collect messages until we see cell_status(ready) or cell_status(error)
    # for the target cell. Handle cascade_prompt by auto-accepting.
    while True:
        msg = helper.receive()

        if msg["type"] == "cascade_prompt":
            plan_id = msg["payload"]["plan_id"]
            helper.execute_cascade(cell_id, plan_id)
            continue

        if msg["type"] == "cell_status":
            p = msg["payload"]
            if p.get("cell_id") == cell_id and p.get("status") == "running":
                saw_running = True
            # "stale" means this cell was skipped because an upstream cell
            # errored mid-cascade. Treat it as a terminal state.
            if p.get("cell_id") == cell_id and p.get("status") in ("ready", "error", "stale"):
                terminal_message = msg
                if saw_running or p.get("status") in ("error", "stale"):
                    break

    # Find the output/error message for this cell
    for m in reversed(helper.messages):
        if m["type"] in ("cell_output", "cell_error") and m["payload"].get("cell_id") == cell_id:
            return m

    # If no output found, return the last status message
    return terminal_message or msg


def run_all_and_wait(
    helper: WebSocketTestHelper,
    terminal_cell_id: str,
) -> list[dict[str, Any]]:
    """Run all notebook cells and wait until the terminal cell finishes."""
    helper.run_all()

    saw_terminal_running = False
    while True:
        msg = helper.receive()
        if msg["type"] == "cell_status":
            payload = msg["payload"]
            if payload.get("cell_id") == terminal_cell_id and payload.get("status") == "running":
                saw_terminal_running = True
            if (
                payload.get("cell_id") == terminal_cell_id
                and payload.get("status") in ("ready", "error")
                and saw_terminal_running
            ):
                return helper.messages

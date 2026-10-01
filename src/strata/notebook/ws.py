"""WebSocket handler for real-time notebook execution updates.

Manages connections per notebook, dispatches client frames and streams server
updates (cell status, console output, execution results).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from strata.notebook import console_relay
from strata.notebook.annotations import parse_annotations
from strata.notebook.authorship import resolve_author
from strata.notebook.cascade import CascadePlanner
from strata.notebook.causality import skip_none
from strata.notebook.executor import (
    BatchCellResult,
    CellExecutionResult,
    CellExecutor,
    partition_batchable_runs,
)
from strata.notebook.harness_user import LocalExecutionRefused, resolve_harness_user
from strata.notebook.impact import ImpactAnalyzer
from strata.notebook.inspect_repl import InspectManager
from strata.notebook.models import (
    CellLanguage,
    CellStaleness,
    CellStatus,
    StalenessReason,
    WorkerBackendType,
)
from strata.notebook.presence import lock_window_seconds
from strata.notebook.protocol import MessageType
from strata.notebook.scopes import (
    required_scope_for_frame,
)
from strata.notebook.session import CellStateSnapshot, SessionManager
from strata.notebook.workers import resolve_worker_spec, worker_transport
from strata.notebook.writer import write_cell, write_cell_tests
from strata.notebook.ws_payloads import (
    CascadeProgressPayload,
    CascadePromptPayload,
    CellConsolePayload,
    CellIterationProgressPayload,
    CellOutputDeltaPayload,
    CellTestResultsPayload,
    CellTestStatusPayload,
    CellVariantProgressPayload,
    PresencePayload,
    cell_status_payload,
    dag_update_payload,
    error_payload,
    impact_preview_payload,
    profiling_summary_payload,
)

if TYPE_CHECKING:
    from strata.notebook.cascade import CascadePlan
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/notebooks", tags=["notebooks_ws"])

_notebook_connections: dict[str, list[WebSocket]] = {}


@dataclass
class NotebookExecutionState:
    """Per-notebook WebSocket execution bookkeeping.

    ``requested_cell`` is reserved before execution starts and ``running_cell`` once
    it runs; ``control_lock`` serializes start / stop / requeue transitions.
    """

    sequence: int = 0
    running_cell: str | None = None
    requested_cell: str | None = None
    cascade_plan: CascadePlan | None = None
    # WS runs resolve to None; REST/MCP exclusive runs to a CellExecutionResult.
    execution_task: asyncio.Task[Any] | None = None
    control_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def next_sequence(self) -> int:
        """Increment and return the outbound message sequence number."""
        self.sequence += 1
        return self.sequence

    def active_task(self) -> asyncio.Task[Any] | None:
        """Return the live execution task, clearing fields if it's already done."""
        task = self.execution_task
        if task is not None and task.done():
            self.execution_task = None
            self.requested_cell = None
            self.running_cell = None
            return None
        return task

    def reset_execution(self) -> None:
        """Clear all fields tracking the in-flight execution and cascade."""
        self.execution_task = None
        self.requested_cell = None
        self.running_cell = None
        self.cascade_plan = None


_notebook_execution_state: dict[str, NotebookExecutionState] = {}

_notebook_inspect_managers: dict[str, InspectManager] = {}

# Deferred teardowns after the last WS disconnects, so a refresh or VPN blip
# doesn't cancel a long-running cell. A reconnect cancels the pending task.
_notebook_grace_tasks: dict[str, asyncio.Task[None]] = {}
_GRACE_CANCEL_SECONDS = 60.0


def _get_session_manager() -> SessionManager:
    """Get the session manager from routes module."""
    from strata.notebook.routes import get_session_manager

    return get_session_manager()


# --- Message envelope ---


def _utc_timestamp() -> str:
    """Return an ISO-8601 ``...Z`` timestamp for the current UTC moment."""
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _make_message(
    msg_type: MessageType | str,
    seq: int,
    payload: Any,
    *,
    ts: str | None = None,
) -> dict[str, Any]:
    """Build a ``{type, seq, ts, payload}`` protocol message envelope.

    ``ts`` defaults to now; pass one to make several messages share a timestamp.
    """
    return {
        "type": msg_type,
        "seq": seq,
        "ts": ts if ts is not None else _utc_timestamp(),
        "payload": payload,
    }


# --- Message Serialization ---


def _serialize_datetime(obj: Any) -> str:
    """Serialize datetime to ISO 8601 string."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_encode(obj: Any) -> str:
    """Encode object to JSON, handling datetime and Path objects."""
    return json.dumps(
        obj,
        default=_serialize_datetime,
        ensure_ascii=False,
    )


def _json_decode(text: str) -> Any:
    """Decode JSON from string."""
    return json.loads(text)


def _ensure_execution_state(notebook_id: str) -> NotebookExecutionState:
    """Get or create per-notebook execution bookkeeping."""
    return _notebook_execution_state.setdefault(notebook_id, NotebookExecutionState())


def forget_notebook_execution_state(notebook_id: str) -> None:
    """Drop a closed session's bookkeeping, including its sequence counter.

    The counter outlives a disconnect on purpose; this is the one place it may go,
    since the session id is never reused. Connections are left alone: an open socket
    has to reach its own cleanup, which closes the inspect sessions behind it.
    """
    _notebook_execution_state.pop(notebook_id, None)


def next_notebook_sequence(notebook_id: str) -> int:
    """Increment and return the next outbound sequence for a notebook."""
    return _ensure_execution_state(notebook_id).next_sequence()


def notebook_has_active_execution(notebook_id: str) -> bool:
    """Return whether a notebook currently has an active execution task."""
    execution_state = _notebook_execution_state.get(notebook_id)
    if execution_state is None:
        return False
    return (
        execution_state.active_task() is not None
        or execution_state.running_cell is not None
        or execution_state.requested_cell is not None
    )


async def cancel_notebook_execution(notebook_id: str) -> list[str]:
    """Cancel whatever a notebook is running; return the cells that were.

    For a quiesce whose timeout ran out: the hold cannot start while a cell is still
    writing.
    """
    execution_state = _notebook_execution_state.get(notebook_id)
    if execution_state is None:
        return []
    async with execution_state.control_lock:
        task = execution_state.active_task()
        cells = {c for c in (execution_state.running_cell, execution_state.requested_cell) if c}
        if task is not None:
            task.cancel()
    if task is not None:
        await asyncio.gather(task, return_exceptions=True)
    return sorted(cells)


async def broadcast_notebook_message(notebook_id: str, message: dict[str, Any]) -> None:
    """Public wrapper for broadcasting notebook protocol messages."""
    await _broadcast_message(notebook_id, message)


async def _send_message(websocket: WebSocket, message: dict[str, Any]) -> None:
    """Send one protocol message to a single WebSocket client."""
    await websocket.send_text(_json_encode(message))


async def _send_error_message(
    websocket: WebSocket,
    seq: int,
    error: str,
) -> None:
    """Send a protocol error to one WebSocket client."""
    await websocket.send_text(
        _json_encode(_make_message(MessageType.ERROR, seq, error_payload(error)))
    )


async def _set_cell_idle(
    session: NotebookSession,
    notebook_id: str,
    seq: int,
    cell_id: str,
) -> None:
    """Mark a cell idle in backend state and broadcast the update."""
    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None:
        cell.status = CellStatus.IDLE

    await _broadcast_message(
        notebook_id,
        _make_message(MessageType.CELL_STATUS, seq, cell_status_payload(cell_id, "idle")),
    )


async def _broadcast_downstream_stale(notebook_id: str, affected_cell_ids: list[str]) -> None:
    """Broadcast STALE status for the cells ``mark_cell_error`` flipped from READY.

    Each frame takes its own sequence: a client deduping on ``seq``, as the protocol
    reference says to, would drop all but the first of a shared one.
    """
    for cell_id in affected_cell_ids:
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(notebook_id),
                cell_status_payload(cell_id, CellStatus.STALE),
            ),
        )


async def _broadcast_staleness_updates(
    session: NotebookSession,
    notebook_id: str,
    staleness_map: dict[str, CellStaleness],
) -> None:
    """Broadcast backend staleness state to all notebook clients.

    Each cell's frame takes its own sequence, since a client deduping on ``seq``
    would otherwise drop all but the first. Status only: the failure details belong
    to whoever saw the cell fail, which holds the result.
    """
    for cell_id, staleness in staleness_map.items():
        causality = session.causality_map.get(cell_id)
        payload = cell_status_payload(
            cell_id,
            staleness.status,
            staleness_reasons=(
                [reason.value for reason in staleness.reasons] if staleness.reasons else []
            ),
            causality=asdict(causality, dict_factory=skip_none) if causality else None,
        )
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_STATUS, next_notebook_sequence(notebook_id), payload),
        )


async def _refresh_and_broadcast_changed_staleness(
    session: NotebookSession,
    notebook_id: str,
    previous_snapshot: dict[str, CellStateSnapshot],
    *,
    preserve_ready_cell_id: str | None = None,
    mark_error_cell_id: str | None = None,
) -> dict[str, CellStaleness]:
    """Recompute notebook staleness and broadcast only changed cells.

    *mark_error_cell_id* is the failure counterpart of *preserve_ready_cell_id*:
    after the recompute, that cell is error and its readers stop claiming ready.
    Recomputing first matters because a failed attempt may have re-run upstreams.
    """
    # Deliberately on-loop: this runs between a cell's result and its frames,
    # and an await here would let other frames interleave out of order.
    staleness_map = session.compute_staleness()
    if preserve_ready_cell_id is not None:
        # Ready even if the walk says idle (an uncached leaf still ran), unless
        # an upstream changed during the run: then it read the old value.
        ran = session.notebook_state.get_cell(preserve_ready_cell_id)
        if ran is not None and any(
            staleness_map[upstream_id].status != CellStatus.READY
            for upstream_id in ran.upstream_ids
            if upstream_id in staleness_map
        ):
            ran.status = CellStatus.STALE
            ran.staleness = CellStaleness(
                status=CellStatus.STALE, reasons=[StalenessReason.UPSTREAM]
            )
            staleness_map[preserve_ready_cell_id] = ran.staleness
        else:
            session.mark_executed_ready(preserve_ready_cell_id)
            staleness_map[preserve_ready_cell_id] = CellStaleness(
                status=CellStatus.READY,
                reasons=[],
            )
    if mark_error_cell_id is not None:
        for stale_id in session.mark_cell_error(mark_error_cell_id):
            staleness_map[stale_id] = CellStaleness(status=CellStatus.STALE, reasons=[])
        staleness_map[mark_error_cell_id] = CellStaleness(status=CellStatus.ERROR, reasons=[])
    changed: dict[str, CellStaleness] = {}

    for cell in session.notebook_state.cells:
        staleness = staleness_map.get(cell.id)
        if staleness is None:
            continue

        causality = session.causality_map.get(cell.id)
        current = CellStateSnapshot(
            status=staleness.status.value,
            reasons=tuple(reason.value for reason in staleness.reasons),
            causality=asdict(causality, dict_factory=skip_none) if causality else None,
        )
        if previous_snapshot.get(cell.id) != current:
            changed[cell.id] = staleness

    if changed:
        await _broadcast_staleness_updates(session, notebook_id, changed)

    return staleness_map


async def _run_execution_task(
    execution_state: NotebookExecutionState,
    requested_cell: str,
    notebook_id: str,
    operation: Any,
) -> None:
    """Run one notebook execution in the background and clean up state."""
    try:
        await operation
    except asyncio.CancelledError:
        logger.info(
            "Notebook execution cancelled for notebook %s requested_cell=%s",
            notebook_id,
            requested_cell,
        )
        raise
    except Exception:
        logger.exception(
            "Unhandled notebook execution error for notebook %s requested_cell=%s",
            notebook_id,
            requested_cell,
        )
    finally:
        if execution_state.execution_task is asyncio.current_task():
            execution_state.reset_execution()


async def _schedule_execution(
    websocket: WebSocket,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    requested_cell: str,
    operation_factory: Any,
) -> bool:
    """Schedule notebook execution so the WebSocket can keep receiving messages.

    Draws a sequence only when it sends the busy refusal; an unused number would be
    a gap that tells the client to resync.
    """
    busy_cell: str | None = None
    operation: Any | None = None

    async with execution_state.control_lock:
        task = execution_state.active_task()
        active_request = execution_state.running_cell or execution_state.requested_cell
        if task is not None:
            busy_cell = execution_state.running_cell or execution_state.requested_cell
        elif active_request not in {None, requested_cell}:
            busy_cell = active_request
        else:
            execution_state.requested_cell = requested_cell
            try:
                operation = operation_factory()
                execution_state.execution_task = asyncio.create_task(
                    _run_execution_task(
                        execution_state,
                        requested_cell,
                        notebook_id,
                        operation,
                    ),
                    name=f"notebook-exec-{notebook_id}-{requested_cell}",
                )
            except Exception:
                execution_state.requested_cell = None
                raise

    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return False

    return True


async def _reserve_execution_request(
    execution_state: NotebookExecutionState,
    requested_cell: str,
) -> str | None:
    """Reserve execution for a cell before validation/scheduling."""
    async with execution_state.control_lock:
        task = execution_state.active_task()
        busy_cell = execution_state.running_cell or execution_state.requested_cell
        if task is not None or busy_cell is not None:
            return busy_cell
        execution_state.requested_cell = requested_cell
        return None


async def _release_execution_request(
    execution_state: NotebookExecutionState,
    requested_cell: str,
) -> None:
    """Release a pre-scheduling execution reservation when execution did not start."""
    async with execution_state.control_lock:
        task = execution_state.active_task()
        if task is None and execution_state.requested_cell == requested_cell:
            execution_state.requested_cell = None


async def _tear_down_notebook_state(notebook_id: str) -> None:
    """Cancel the active execution task and drop inspect/exec state. Idempotent."""
    execution_state = _notebook_execution_state.get(notebook_id)
    if execution_state is not None:
        task = execution_state.active_task()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        # Keep the object: its outbound sequence belongs to the still-open
        # session, and restarting at 1 reads as already-seen frames to a client.
        execution_state.reset_execution()

    inspect_manager = _notebook_inspect_managers.pop(notebook_id, None)
    if inspect_manager is not None:
        try:
            await inspect_manager.close_all()
        except Exception:
            logger.exception(
                "Failed to close inspect sessions during cleanup for notebook %s",
                notebook_id,
            )


async def _grace_cancel_then_tear_down(notebook_id: str, grace_seconds: float) -> None:
    """Wait the grace window, then drop notebook state if nobody reconnected.

    A reconnect during the window cancels this task; otherwise the running cell is
    cancelled and execution and inspect state are dropped.
    """
    try:
        await asyncio.sleep(grace_seconds)
    except asyncio.CancelledError:
        # A client reconnected during the grace window.
        raise
    if _notebook_connections.get(notebook_id):
        # A reconnect whose cancel lost the race; connections are the truth.
        return
    try:
        await _tear_down_notebook_state(notebook_id)
    finally:
        _notebook_grace_tasks.pop(notebook_id, None)


async def _cleanup_notebook_websocket(
    notebook_id: str,
    websocket: WebSocket,
) -> None:
    """Remove a WebSocket and schedule notebook teardown if it was the last one."""
    connections = _notebook_connections.get(notebook_id)
    if connections is None:
        return

    try:
        connections.remove(websocket)
    except ValueError:
        # Already removed by a concurrent cleanup path.
        pass

    session = _get_session_manager().get_session(notebook_id)
    if session is not None and session.presence.leave(websocket):
        await broadcast_presence(notebook_id, session)

    if connections:
        return

    del _notebook_connections[notebook_id]

    existing = _notebook_grace_tasks.pop(notebook_id, None)
    if existing is not None and not existing.done():
        existing.cancel()

    if _GRACE_CANCEL_SECONDS <= 0:
        # Tests zero the window; shutdown paths would leak if deferred.
        await _tear_down_notebook_state(notebook_id)
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. at shutdown): tear down inline.
        await _tear_down_notebook_state(notebook_id)
        return
    _notebook_grace_tasks[notebook_id] = loop.create_task(
        _grace_cancel_then_tear_down(notebook_id, _GRACE_CANCEL_SECONDS)
    )


def _cancel_pending_grace_teardown(notebook_id: str) -> None:
    """Abort a pending teardown for *notebook_id* when a client reconnects in grace."""
    task = _notebook_grace_tasks.pop(notebook_id, None)
    if task is not None and not task.done():
        task.cancel()


# --- WebSocket Handler ---


def _ws_caller_identity(websocket: WebSocket) -> str | None:
    """Return the caller identity from ``personal_mode_user_header``, or None.

    The WS counterpart of ``routes._caller_identity``. None when per-user scoping is
    unconfigured or the header is absent or blank, so the gate passes through.
    """
    try:
        from strata.server import get_state

        header_name = get_state().config.personal_mode_user_header
    except RuntimeError:
        return None
    if not header_name:
        return None
    return (websocket.headers.get(header_name) or "").strip() or None


def _ws_owner_allowed(owner: str | None, caller: str | None) -> bool:
    """Return whether *caller* may open a WS for an *owner*-scoped notebook.

    The boolean twin of ``routes._require_owner``: unowned notebooks are allowed; a
    missing caller identity under per-user scoping is denied (so omitting the header
    is no bypass); a mismatched owner is denied.
    """
    from strata.notebook.routes import _user_scoping_enabled

    if owner is None:
        return True
    if caller is None:
        return not _user_scoping_enabled()
    return owner == caller


# A ``?role=viewer`` (app mode) connection may only drive widgets and request
# state; every other C->S frame is rejected.
_VIEWER_ALLOWED_FRAMES = frozenset(
    {
        MessageType.WIDGET_UPDATE,
        MessageType.NOTEBOOK_SYNC,
    }
)


def _configured_auth_mode() -> str:
    """The server's configured auth mode, or ``"none"`` with no ``ServerState``.

    A process without server state is not a running server and has nothing to
    enforce, as ``routes._get_notebook_storage_root`` also assumes.
    """
    from strata.server import get_state

    try:
        return get_state().config.auth_mode
    except RuntimeError:
        return "none"


def _frame_scope_error(msg_type: str) -> str | None:
    """Return an error message when the caller lacks the frame's scope.

    ``None`` means allowed, including every no-auth deployment.
    """
    from strata.auth import get_principal

    if _configured_auth_mode() not in ("trusted_proxy", "api_key"):
        return None
    required = required_scope_for_frame(msg_type)
    principal = get_principal()
    if principal is not None and principal.has_scope(required):
        return None
    return f"'{msg_type}' requires the {required} scope"


async def _authenticate_websocket(websocket: WebSocket) -> bool:
    """Authenticate a WS upgrade; False means the socket is already closed.

    No HTTP middleware runs for a WebSocket upgrade (``BaseHTTPMiddleware`` passes
    non-http scopes through), so this establishes the principal itself; otherwise
    anything that could open a socket could run code in service mode. Handles both
    ``trusted_proxy`` and ``api_key``, like the HTTP middleware. With
    ``auth_mode="none"`` there is no principal and the socket stays open.
    """
    from strata.auth import (
        AuthError,
        parse_api_key_principal,
        parse_principal,
        set_principal,
        verify_proxy_token,
    )
    from strata.server import get_state

    mode = _configured_auth_mode()
    if mode not in ("trusted_proxy", "api_key"):
        return True
    config = get_state().config

    headers = dict(websocket.headers)
    if mode == "trusted_proxy":
        header_name = config.proxy_token_header
        token = headers.get(header_name) or headers.get(header_name.lower())
        if not verify_proxy_token(token, config.proxy_token):
            logger.warning("ws_auth_failed reason=invalid_proxy_token")
            await websocket.close(code=1008, reason="Unauthorized")
            return False

    try:
        if mode == "api_key":
            set_principal(parse_api_key_principal(headers, config))
        else:
            set_principal(parse_principal(headers, config))
    except AuthError:
        logger.warning("ws_auth_failed reason=missing_principal")
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


@router.websocket("/ws/{notebook_id}")
async def notebook_websocket(websocket: WebSocket, notebook_id: str):
    """WebSocket endpoint for real-time notebook updates.

    Frame types are the ``MessageType`` enum in ``protocol.py``;
    ``docs/reference/notebook-protocol.md`` is the client reference.
    """
    # No HTTP middleware runs for a WS upgrade. First, so an unauthenticated
    # caller can't learn whether a notebook_id exists.
    if not await _authenticate_websocket(websocket):
        return

    session_manager = _get_session_manager()
    session = session_manager.get_session(notebook_id)
    if not session:
        await websocket.close(code=1008, reason="Notebook not found")
        return

    # Per-user scoping, else a leaked notebook_id exposes live state. As in
    # ``_require_owner``, a missing identity header is denied when scoping is on.
    owner = session.notebook_state.owner
    if not _ws_owner_allowed(owner, _ws_caller_identity(websocket)):
        await websocket.close(code=1008, reason="Notebook not found")
        return

    # Before the accept await: a grace task expiring during it would see no
    # connections (not registered yet) and cancel the running cell.
    _cancel_pending_grace_teardown(notebook_id)

    await websocket.accept()

    # Enforced per frame in the dispatch loop below.
    read_only = websocket.query_params.get("role") == "viewer"

    if notebook_id not in _notebook_connections:
        _notebook_connections[notebook_id] = []
    _notebook_connections[notebook_id].append(websocket)
    session.presence.join(websocket, resolve_author())
    await broadcast_presence(notebook_id, session)

    execution_state = _ensure_execution_state(notebook_id)

    try:
        while True:
            data = await websocket.receive_text()
            msg = _json_decode(data)
            session.touch()

            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            handler = _C2S_HANDLERS.get(msg_type) if isinstance(msg_type, str) else None
            if handler is None:
                await websocket.send_text(
                    _json_encode(
                        _make_message(
                            MessageType.ERROR,
                            execution_state.next_sequence(),
                            error_payload(f"Unknown message type: {msg_type}"),
                        )
                    )
                )
                continue
            # Under trusted-proxy auth each frame needs its notebook scope.
            scope_error = _frame_scope_error(msg_type)
            if scope_error is not None:
                await websocket.send_text(
                    _json_encode(
                        _make_message(
                            MessageType.ERROR,
                            execution_state.next_sequence(),
                            error_payload(scope_error, code="insufficient_scope"),
                        )
                    )
                )
                continue
            if read_only and msg_type not in _VIEWER_ALLOWED_FRAMES:
                await websocket.send_text(
                    _json_encode(
                        _make_message(
                            MessageType.ERROR,
                            execution_state.next_sequence(),
                            error_payload(
                                f"'{msg_type}' is not allowed in read-only app view",
                                code="read_only",
                            ),
                        )
                    )
                )
                continue
            dispatch_ctx = {
                "websocket": websocket,
                "session": session,
                "payload": payload,
                "execution_state": execution_state,
                "notebook_id": notebook_id,
            }
            await handler(**{name: dispatch_ctx[name] for name in _handler_args(handler)})

    except WebSocketDisconnect:
        await _cleanup_notebook_websocket(notebook_id, websocket)
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        await _cleanup_notebook_websocket(notebook_id, websocket)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except Exception:
            pass


# --- Message Handlers ---


async def _handle_cell_execute(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_execute: send a cascade_prompt if upstreams need running, else run."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing cell_id"),
                )
            )
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    # Any raise before scheduling must release the reservation: the dispatch
    # loop tears state down only for the last connection, so with another tab
    # open every later run would be refused as busy.
    try:
        await _handle_cell_execute_reserved(
            websocket, session, execution_state, notebook_id, cell_id
        )
    except BaseException:
        await _release_execution_request(execution_state, cell_id)
        raise


async def _handle_cell_execute_reserved(
    websocket: WebSocket,
    session: NotebookSession,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    cell_id: str,
) -> None:
    """The post-reservation body of ``_handle_cell_execute``."""
    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(f"Cell {cell_id} not found"),
                )
            )
        )
        return

    planner = CascadePlanner(session)
    plan = planner.plan(cell_id)

    if plan:
        # Downstream staleness goes out via cell_status, not impact_preview.
        logger.info(
            "Cascade needed for cell %s — upstream statuses: %s",
            cell_id,
            {
                uid: next(
                    (c.status for c in session.notebook_state.cells if c.id == uid),
                    "?",
                )
                for uid in (session.dag.cell_upstream.get(cell_id, []) if session.dag else [])
            },
        )
        execution_state.cascade_plan = plan
        await _send_message(
            websocket,
            _make_message(
                MessageType.CASCADE_PROMPT,
                next_notebook_sequence(notebook_id),
                CascadePromptPayload(
                    cell_id=cell_id,
                    plan_id=plan.plan_id,
                    cells_to_run=[s.cell_id for s in plan.steps],
                    estimated_duration_ms=plan.estimated_duration_ms,
                ).model_dump(mode="json"),
            ),
        )
        await _release_execution_request(execution_state, cell_id)
    else:
        scheduled = await _schedule_execution(
            websocket,
            execution_state,
            notebook_id,
            cell_id,
            lambda: execute_cell_and_broadcast(session, cell_id, execution_state, notebook_id),
        )
        if not scheduled:
            await _release_execution_request(execution_state, cell_id)


async def _handle_notebook_run_all(
    websocket: WebSocket,
    session: NotebookSession,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    payload: dict[str, Any],
) -> None:
    """Handle notebook_run_all: run every non-empty cell in notebook order.

    ``continue_on_error`` (default True) keeps the run going past a failed cell.
    """
    continue_on_error = bool(payload.get("continue_on_error", True))

    # Inactive variants aren't in the DAG, so their references don't resolve.
    runnable_cells = [
        cell.id
        for cell in session.notebook_state.cells
        if cell.source.strip() and cell.variant_active
    ]
    if not runnable_cells:
        return

    requested_cell = runnable_cells[0]
    busy_cell = await _reserve_execution_request(execution_state, requested_cell)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, requested_cell)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        requested_cell,
        lambda: _execute_run_all(
            websocket,
            session,
            runnable_cells,
            execution_state,
            notebook_id,
            continue_on_error=continue_on_error,
        ),
    )
    if not scheduled:
        await _release_execution_request(execution_state, requested_cell)


async def _handle_notebook_rerun_all(
    websocket: WebSocket,
    session: NotebookSession,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    payload: dict[str, Any],
) -> None:
    """Handle notebook_rerun_all: like run_all, but every cell bypasses its own cache."""
    continue_on_error = bool(payload.get("continue_on_error", True))

    runnable_cells = [
        cell.id
        for cell in session.notebook_state.cells
        if cell.source.strip() and cell.variant_active
    ]
    if not runnable_cells:
        return

    requested_cell = runnable_cells[0]
    busy_cell = await _reserve_execution_request(execution_state, requested_cell)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, requested_cell)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        requested_cell,
        lambda: _execute_run_all(
            websocket,
            session,
            runnable_cells,
            execution_state,
            notebook_id,
            force=True,
            continue_on_error=continue_on_error,
        ),
    )
    if not scheduled:
        await _release_execution_request(execution_state, requested_cell)


async def _handle_cell_execute_cascade(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_execute_cascade: the user confirmed, so run every cell in the plan."""
    cell_id = payload.get("cell_id")
    plan_id = payload.get("plan_id")

    if not cell_id or not plan_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing cell_id or plan_id"),
                )
            )
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    plan = execution_state.cascade_plan
    if not plan or plan.plan_id != plan_id:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload("Cascade plan not found or expired"),
                )
            )
        )
        return

    # In the background so this socket can still receive cancel.
    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        cell_id,
        lambda: _execute_cascade(websocket, session, plan, execution_state, notebook_id),
    )
    if not scheduled:
        await _release_execution_request(execution_state, cell_id)


async def _handle_cell_execute_force(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_execute_force: run the cell with stale inputs ("Run this only")."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing cell_id"),
                )
            )
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        cell_id,
        lambda: execute_cell_and_broadcast(
            session, cell_id, execution_state, notebook_id, mode="force"
        ),
    )
    if not scheduled:
        await _release_execution_request(execution_state, cell_id)


async def _handle_cell_execute_rerun(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_execute_rerun: bypass the target's cache, materialize upstreams."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing cell_id"),
                )
            )
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    # Release on exception, as in _handle_cell_execute.
    try:
        await _handle_cell_execute_rerun_reserved(
            websocket, session, execution_state, notebook_id, cell_id
        )
    except BaseException:
        await _release_execution_request(execution_state, cell_id)
        raise


async def _handle_cell_execute_rerun_reserved(
    websocket: WebSocket,
    session: NotebookSession,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    cell_id: str,
) -> None:
    """The post-reservation body of ``_handle_cell_execute_rerun``."""
    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    next_notebook_sequence(notebook_id),
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    # Stale upstreams go through the cascade so every step broadcasts frames.
    planner = CascadePlanner(session)
    plan = planner.plan(cell_id)

    if plan is not None:
        scheduled = await _schedule_execution(
            websocket,
            execution_state,
            notebook_id,
            cell_id,
            lambda: _execute_cascade(
                websocket,
                session,
                plan,
                execution_state,
                notebook_id,
                target_force=True,
            ),
        )
    else:
        scheduled = await _schedule_execution(
            websocket,
            execution_state,
            notebook_id,
            cell_id,
            lambda: execute_cell_and_broadcast(
                session, cell_id, execution_state, notebook_id, mode="rerun"
            ),
        )
    if not scheduled:
        await _release_execution_request(execution_state, cell_id)


async def _handle_cell_run_tests(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_run_tests: persist the test source, run it, broadcast results.

    Takes the cell-execution reservation, since a test run materializes upstreams.
    Emits CELL_TEST_STATUS around a CELL_TEST_RESULTS frame. Python cells only.
    """
    cell_id = payload.get("cell_id")
    test_source = payload.get("test_source", "")
    seq = execution_state.next_sequence()

    if not cell_id:
        await _send_error_message(websocket, seq, "Missing cell_id")
        return

    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        await _send_error_message(websocket, seq, f"Cell {cell_id} not found")
        return
    if cell.language != CellLanguage.PYTHON:
        await _send_error_message(websocket, seq, "Cell tests are only supported for Python cells")
        return
    # Tests run the cell's code, so they wait for the environment too.
    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    seq,
                    error_payload(environment_block_reason, code="ENVIRONMENT_BUSY"),
                )
            )
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            seq,
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    try:
        write_cell_tests(session.path, cell_id, test_source)
        cell.test_source = test_source

        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_TEST_STATUS,
                seq,
                CellTestStatusPayload(cell_id=cell_id, status="running").model_dump(mode="json"),
            ),
        )

        executor = _make_executor_with_progress(session, notebook_id)
        result = await executor.run_cell_tests(cell_id, test_source)

        seq = execution_state.next_sequence()
        results_payload = CellTestResultsPayload(
            cell_id=cell_id,
            passed=result.passed,
            failed=result.failed,
            errored=result.errored,
            skipped=result.skipped,
            tests=result.tests,
            stale=False,
            pytest_unavailable=result.pytest_unavailable,
            ran_at=result.ran_at,
            auto_installed=result.auto_installed,
        ).model_dump(mode="json")
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_TEST_RESULTS, seq, results_payload),
        )
        status = "error" if (result.failed or result.errored) else "ready"
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_TEST_STATUS,
                seq,
                CellTestStatusPayload(cell_id=cell_id, status=status).model_dump(mode="json"),
            ),
        )
    except Exception as e:
        logger.exception("Cell test run failed for %s: %s", cell_id, e)
        seq = execution_state.next_sequence()
        await _send_error_message(websocket, seq, str(e))
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_TEST_STATUS,
                seq,
                CellTestStatusPayload(cell_id=cell_id, status="error").model_dump(mode="json"),
            ),
        )
    finally:
        await _release_execution_request(execution_state, cell_id)


async def _handle_cell_cancel(
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_cancel without clobbering completed cell state."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        return

    async with execution_state.control_lock:
        running_cell = execution_state.running_cell
        requested_cell = execution_state.requested_cell
        task = execution_state.active_task()

        should_cancel = task is not None and cell_id in {running_cell, requested_cell}
        if should_cancel and task is not None:
            task.cancel()

    if should_cancel and task is not None:
        await asyncio.gather(task, return_exceptions=True)
        if requested_cell and requested_cell != running_cell and requested_cell == cell_id:
            await _set_cell_idle(
                session, notebook_id, next_notebook_sequence(notebook_id), requested_cell
            )
        return

    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None and cell.status in {CellStatus.IDLE, CellStatus.RUNNING}:
        await _set_cell_idle(session, notebook_id, next_notebook_sequence(notebook_id), cell_id)


async def _handle_cell_source_update(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_source_update: re-analyze the cell and update the DAG."""
    cell_id = payload.get("cell_id")
    source = payload.get("source")
    # Agents send their own name, keeping their edits distinguishable on a
    # server that authenticates nobody; the browser sends none.
    author = resolve_author(payload.get("author"))

    if not cell_id or source is None:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing cell_id or source"),
                )
            )
        )
        return

    if len(source) > 1_000_000:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Cell source exceeds 1MB limit"),
                )
            )
        )
        return

    # Reject edits to the running cell, or its artifact is stored under a
    # source hash the saved source no longer matches (stale forever).
    # control_lock covers scheduling only, so this never waits on the run;
    # the frontend retries on the next cell_status.
    async with execution_state.control_lock:
        running = execution_state.running_cell
        requested = execution_state.requested_cell
    if cell_id in {running, requested}:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload(
                        f"Cannot update cell {cell_id} while it is executing; "
                        "retry after cell finishes",
                        code="cell_busy",
                        cell_id=cell_id,
                    ),
                )
            )
        )
        return

    # Someone else just edited this cell; don't overwrite unless forced.
    held_by = session.presence.holder(cell_id, author, lock_window_seconds())
    if held_by is not None and not payload.get("force"):
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload(
                        f"{held_by} changed cell {cell_id} moments ago; "
                        "resend with force to take it over",
                        code="cell_locked",
                        cell_id=cell_id,
                        held_by=held_by,
                    ),
                )
            )
        )
        return

    try:
        write_cell(session.path, cell_id, source, author=author)
        session.presence.record_edit(cell_id, author)
        if session.presence.focus(websocket, author, cell_id):
            await broadcast_presence(notebook_id, session)

        # Must happen before re-analysis
        cell_in_session = session.notebook_state.get_cell(cell_id)
        if cell_in_session:
            cell_in_session.source = source
            # Else broadcasts name the previous author until the next reload.
            cell_in_session.updated_by = author

        session.re_analyze_cell(cell_id)
        session._run_annotation_validation()

        # Leave a running cell as running
        staleness_map = await session.compute_staleness_async(
            executing=execution_state.running_cell
        )

        dag_edges = session.dag.serialize_edges() if session.dag else []

        # Per-cell analysis, so the frontend needs no REST round-trip.
        from strata.notebook.module_export import build_module_export_plan

        cells_analysis = []
        for cell in session.notebook_state.cells:
            entry: dict[str, Any] = {
                "id": cell.id,
                "defines": cell.defines,
                "references": cell.references,
                "upstream_ids": cell.upstream_ids,
                "downstream_ids": cell.downstream_ids,
                "is_leaf": cell.is_leaf,
                "annotation_diagnostics": [d.model_dump() for d in cell.annotation_diagnostics],
                "variant_group": cell.variant_group,
                "variant_name": cell.variant_name,
                "variant_active": cell.variant_active,
                "created_by": cell.created_by,
                "updated_by": cell.updated_by,
            }
            if cell.language == CellLanguage.PYTHON:
                plan = build_module_export_plan(cell.source)
                has_code_export = any(
                    s.kind in ("function", "async function", "class")
                    for s in plan.exported_symbols.values()
                )
                entry["is_module_cell"] = plan.is_exportable and has_code_export
                if entry["is_module_cell"]:
                    entry["module_exports"] = [
                        {"name": name, "kind": sym.kind}
                        for name, sym in sorted(plan.exported_symbols.items())
                    ]
            cells_analysis.append(entry)

        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.DAG_UPDATE,
                next_notebook_sequence(notebook_id),
                dag_update_payload(
                    {
                        "edges": dag_edges,
                        "roots": list(session.dag.roots) if session.dag else [],
                        "leaves": list(session.dag.leaves) if session.dag else [],
                        "topological_order": (session.dag.topological_order if session.dag else []),
                        "cells": cells_analysis,
                        "variant_groups": [
                            vg.model_dump() for vg in session.notebook_state.variant_groups
                        ],
                    }
                ),
            ),
        )

        await _broadcast_staleness_updates(session, notebook_id, staleness_map)

    except Exception as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, next_notebook_sequence(notebook_id), error_payload(str(e))
                )
            )
        )


async def _handle_variant_set_active(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Switch the active variant for a group, then broadcast a dag_update.

    The DAG effect matches a source update: a different cell produces the group's
    defines, and downstream cells go stale.
    """
    group = payload.get("group")
    variant_name = payload.get("name")

    if not isinstance(group, str) or not isinstance(variant_name, str):
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing group or name"),
                )
            )
        )
        return

    # Sweep mode ignores the active pointer: no-op rather than churn
    # notebook.toml or restale downstream.
    if session.notebook_state.variant_modes.get(group) == "sweep":
        return

    try:
        session.set_variant_active(group, variant_name)
        staleness_map = await session.compute_staleness_async(
            executing=execution_state.running_cell
        )

        dag_edges = session.dag.serialize_edges() if session.dag else []
        from strata.notebook.module_export import build_module_export_plan

        cells_analysis = []
        for cell in session.notebook_state.cells:
            entry: dict[str, Any] = {
                "id": cell.id,
                "defines": cell.defines,
                "references": cell.references,
                "upstream_ids": cell.upstream_ids,
                "downstream_ids": cell.downstream_ids,
                "is_leaf": cell.is_leaf,
                "annotation_diagnostics": [d.model_dump() for d in cell.annotation_diagnostics],
                "variant_group": cell.variant_group,
                "variant_name": cell.variant_name,
                "variant_active": cell.variant_active,
                "created_by": cell.created_by,
                "updated_by": cell.updated_by,
            }
            if cell.language == CellLanguage.PYTHON:
                plan = build_module_export_plan(cell.source)
                has_code_export = any(
                    s.kind in ("function", "async function", "class")
                    for s in plan.exported_symbols.values()
                )
                entry["is_module_cell"] = plan.is_exportable and has_code_export
                if entry["is_module_cell"]:
                    entry["module_exports"] = [
                        {"name": name, "kind": sym.kind}
                        for name, sym in sorted(plan.exported_symbols.items())
                    ]
            cells_analysis.append(entry)

        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.DAG_UPDATE,
                next_notebook_sequence(notebook_id),
                dag_update_payload(
                    {
                        "edges": dag_edges,
                        "roots": list(session.dag.roots) if session.dag else [],
                        "leaves": list(session.dag.leaves) if session.dag else [],
                        "topological_order": (session.dag.topological_order if session.dag else []),
                        "cells": cells_analysis,
                        "variant_groups": [
                            vg.model_dump() for vg in session.notebook_state.variant_groups
                        ],
                    }
                ),
            ),
        )

        await _broadcast_staleness_updates(session, notebook_id, staleness_map)

    except Exception as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, next_notebook_sequence(notebook_id), error_payload(str(e))
                )
            )
        )


async def _handle_variant_add(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Add a sibling variant to a group, then broadcast a dag_update.

    The new variant becomes active, so the producer of the group's defines moves.
    """
    group = payload.get("group")
    if not isinstance(group, str):
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing group"),
                )
            )
        )
        return

    try:
        session.add_variant(group, author=resolve_author(payload.get("author")))
        staleness_map = await session.compute_staleness_async(
            executing=execution_state.running_cell
        )

        # A new cell: dag_update only updates existing cells and would drop
        # it, so send the full notebook_state.
        state_payload = session.serialize_notebook_state()
        state_payload["dag"] = {
            "edges": session.dag.serialize_edges() if session.dag else [],
            "roots": list(session.dag.roots) if session.dag else [],
            "leaves": list(session.dag.leaves) if session.dag else [],
            "topological_order": session.dag.topological_order if session.dag else [],
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
        }

        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.NOTEBOOK_STATE, next_notebook_sequence(notebook_id), state_payload
            ),
        )

        await _broadcast_staleness_updates(session, notebook_id, staleness_map)

    except ValueError as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, next_notebook_sequence(notebook_id), error_payload(str(e))
                )
            )
        )
    except Exception as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, next_notebook_sequence(notebook_id), error_payload(str(e))
                )
            )
        )


async def _handle_notebook_sync(
    websocket: WebSocket,
    session: NotebookSession,
    notebook_id: str,
) -> None:
    """Handle notebook_sync: return the full notebook state (for reconnection)."""
    dag_edges = session.dag.serialize_edges() if session.dag else []

    state = session.serialize_notebook_state()
    state["dag"] = {
        "edges": dag_edges,
        "roots": list(session.dag.roots) if session.dag else [],
        "leaves": list(session.dag.leaves) if session.dag else [],
        "topological_order": (session.dag.topological_order if session.dag else []),
    }

    # A real sequence: 0 reads as a gap to the client.
    await websocket.send_text(
        _json_encode(
            _make_message(MessageType.NOTEBOOK_STATE, next_notebook_sequence(notebook_id), state)
        )
    )


# --- Execution Helpers ---


def _make_executor_with_progress(
    session: NotebookSession,
    notebook_id: str,
) -> CellExecutor:
    """Build a CellExecutor that broadcasts ``cell_iteration_progress`` per loop iteration."""
    executor = CellExecutor(session, session.warm_pool)

    async def _broadcast_iteration_progress(progress: dict[str, Any]) -> None:
        seq = next_notebook_sequence(notebook_id)
        payload = CellIterationProgressPayload(**progress).model_dump(mode="json")
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_ITERATION_PROGRESS, seq, payload),
        )

    async def _broadcast_prompt_delta(payload: dict[str, Any]) -> None:
        seq = next_notebook_sequence(notebook_id)
        typed = CellOutputDeltaPayload(**payload).model_dump(mode="json")
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_OUTPUT_DELTA, seq, typed),
        )

    async def _broadcast_variant_progress(progress: dict[str, Any]) -> None:
        seq = next_notebook_sequence(notebook_id)
        payload = CellVariantProgressPayload(**progress).model_dump(mode="json")
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_VARIANT_PROGRESS, seq, payload),
        )

    executor.on_iteration_complete = _broadcast_iteration_progress
    executor.on_prompt_delta = _broadcast_prompt_delta
    executor.on_variant_complete = _broadcast_variant_progress
    return executor


class NotebookBusyError(RuntimeError):
    """The notebook is already executing a cell (raised on the REST/MCP drive)."""

    def __init__(self, busy_cell: str | None) -> None:
        self.busy_cell = busy_cell
        super().__init__(
            f"Notebook is already executing cell {busy_cell}"
            if busy_cell
            else "Notebook is already executing another cell"
        )


async def execute_cell_exclusive(
    session: NotebookSession,
    cell_id: str,
    notebook_id: str,
    mode: Literal["normal", "force", "rerun"] = "normal",
    operation: Callable[[NotebookExecutionState], Coroutine[Any, Any, CellExecutionResult | None]]
    | None = None,
) -> CellExecutionResult | None:
    """Reserve, execute and release for the non-WS drivers (REST, MCP).

    Takes the same reservation as the WS handlers and registers the run as the
    execution task, so an agent run and a browser run cannot execute concurrently
    and ``cell_cancel`` and grace teardown can reach it. Raises
    :class:`NotebookBusyError` when a run is already active.
    """
    execution_state = _ensure_execution_state(notebook_id)
    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        raise NotebookBusyError(busy_cell)

    # ``operation`` runs inside the reservation (e.g. a widget writing its
    # values): written outside, a busy refusal would still change the next run.
    task = asyncio.create_task(
        operation(execution_state)
        if operation is not None
        else execute_cell_and_broadcast(session, cell_id, execution_state, notebook_id, mode=mode),
        name=f"notebook-exec-{notebook_id}-{cell_id}",
    )
    async with execution_state.control_lock:
        execution_state.execution_task = task
    try:
        return await task
    except asyncio.CancelledError:
        if task.cancelled():
            # The run was cancelled (cell_cancel, grace teardown): no result.
            return None
        # The caller was cancelled: the run continues for spectators, like a
        # WS run outliving its socket.
        raise
    finally:
        async with execution_state.control_lock:
            if execution_state.execution_task is task and task.done():
                execution_state.reset_execution()


async def execute_cell_and_broadcast(
    session: NotebookSession,
    cell_id: str,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    mode: Literal["normal", "force", "rerun"] = "normal",
) -> CellExecutionResult | None:
    """Execute a cell and broadcast the live frames to every spectator.

    Shared by the WS and REST drives, so WS spectators see the same frame sequence
    whichever transport triggered the run. Returns the execution result, or ``None``
    when the cell is missing or the executor raised unexpectedly. *mode* is
    ``normal`` (cache on), ``force`` (cache off, no upstream materialization) or
    ``rerun`` (cache off, upstreams materialized).
    """
    seq = execution_state.next_sequence()

    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        return None

    execution_state.running_cell = cell_id
    session.mark_cell_running(cell_id)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_STATUS, seq, _running_payload(session, cell_id, cell.source)
        ),
    )

    executor = _make_executor_with_progress(session, notebook_id)
    try:
        if mode == "force":
            result = await executor.execute_cell_force(cell_id, cell.source)
        elif mode == "rerun":
            result = await executor.execute_cell_rerun(cell_id, cell.source)
        else:
            result = await executor.execute_cell(cell_id, cell.source)

        # Before broadcasting, so the payload carries this run's metadata.
        session.record_execution(
            cell_id,
            result.duration_ms,
            result.cache_hit,
            from_team=result.from_team_cache,
            team_principal=result.team_cache_principal,
            team_promotion=result.team_cache_promotion,
            team_saved_ms=result.team_cache_saved_ms,
        )
        session.apply_execution_result_metadata(cell_id, result)

        await _broadcast_execution_result(notebook_id, cell_id, result)

        # After the cell's own frame so sequence matches send order; innermost
        # first so the root cause is announced first.
        await _broadcast_upstream_results(notebook_id, executor)

        if result.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                previous_snapshot,
                preserve_ready_cell_id=cell_id,
            )
        else:
            # Re-classify first: the attempt may have re-run upstreams, so
            # their readers' old "ready" labels may no longer hold.
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                previous_snapshot,
                mark_error_cell_id=cell_id,
            )

        return result

    except asyncio.CancelledError:
        await _set_cell_idle(session, notebook_id, execution_state.next_sequence(), cell_id)
        raise
    except Exception as e:
        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_ERROR,
                next_notebook_sequence(notebook_id),
                {"cell_id": cell_id, "error": str(e)},
            ),
        )
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(notebook_id),
                cell_status_payload(cell_id, "error"),
            ),
        )
        await _broadcast_downstream_stale(notebook_id, downstream_stale)
        return None
    finally:
        execution_state.running_cell = None


async def _execute_cascade(
    websocket: WebSocket,
    session: NotebookSession,
    plan: CascadePlan,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    target_force: bool = False,
) -> None:
    """Execute all cells in a cascade plan.

    With *target_force*, the final (requested) cell runs ``execute_cell_rerun`` and
    bypasses its own cache; upstream steps use normal cached execution.
    """
    del websocket

    executor = _make_executor_with_progress(session, notebook_id)

    logger.info(
        "Cascade %s: executing %d steps: %s",
        plan.plan_id,
        len(plan.steps),
        [(s.cell_id, s.reason, s.skip) for s in plan.steps],
    )

    cascade_failed = False

    # So a @nocache step isn't re-executed by each later step's upstreams.
    with executor.one_run():
        try:
            for i, step in enumerate(plan.steps):
                # Under target_force the cached-ready target must still rerun.
                if step.skip and not (target_force and step.cell_id == plan.target_cell_id):
                    continue

                cell_id = step.cell_id
                cell = session.notebook_state.get_cell(cell_id)
                if not cell:
                    continue

                if cascade_failed:
                    logger.warning(
                        "Cascade %s: skipping cell %s (earlier step failed)",
                        plan.plan_id,
                        cell_id,
                    )
                    # "stale", not "idle", marks a cascade abort.
                    cell_to_skip = session.notebook_state.get_cell(cell_id)
                    if cell_to_skip:
                        cell_to_skip.status = CellStatus.STALE
                    await _broadcast_message(
                        notebook_id,
                        _make_message(
                            MessageType.CELL_STATUS,
                            next_notebook_sequence(notebook_id),
                            cell_status_payload(cell_id, "stale"),
                        ),
                    )
                    continue

                execution_state.running_cell = cell_id

                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CASCADE_PROGRESS,
                        next_notebook_sequence(notebook_id),
                        CascadeProgressPayload(
                            plan_id=plan.plan_id,
                            current_cell_id=cell_id,
                            completed=i,
                            total=len([s for s in plan.steps if not s.skip]),
                        ).model_dump(mode="json"),
                    ),
                )

                session.mark_cell_running(cell_id)
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_STATUS,
                        next_notebook_sequence(notebook_id),
                        _running_payload(session, cell_id, cell.source),
                    ),
                )

                try:
                    if target_force and cell_id == plan.target_cell_id:
                        result = await executor.execute_cell_rerun(cell_id, cell.source)
                    else:
                        result = await executor.execute_cell(cell_id, cell.source)
                    session.record_execution(
                        cell_id,
                        result.duration_ms,
                        result.cache_hit,
                        from_team=result.from_team_cache,
                        team_principal=result.team_cache_principal,
                        team_promotion=result.team_cache_promotion,
                        team_saved_ms=result.team_cache_saved_ms,
                    )
                    session.apply_execution_result_metadata(cell_id, result)

                    # Shared with the direct-execute path so frames match.
                    await _broadcast_execution_result(notebook_id, cell_id, result)

                    status = CellStatus.READY if result.success else CellStatus.ERROR
                    cascade_cell = session.notebook_state.get_cell(cell_id)
                    if cascade_cell:
                        cascade_cell.status = status
                    await _broadcast_message(
                        notebook_id,
                        _make_message(
                            MessageType.CELL_STATUS,
                            next_notebook_sequence(notebook_id),
                            cell_status_payload(cell_id, status),
                        ),
                    )

                    logger.info(
                        "Cascade %s: cell %s finished status=%s artifact_uri=%s cache_hit=%s",
                        plan.plan_id,
                        cell_id,
                        status,
                        getattr(cascade_cell, "artifact_uri", None) if cascade_cell else None,
                        result.cache_hit,
                    )

                    if not result.success:
                        cascade_failed = True

                except asyncio.CancelledError:
                    await _set_cell_idle(
                        session, notebook_id, execution_state.next_sequence(), cell_id
                    )
                    raise
                except Exception as e:
                    downstream_stale = session.mark_cell_error(cell_id)
                    await _broadcast_message(
                        notebook_id,
                        _make_message(
                            MessageType.CELL_ERROR,
                            next_notebook_sequence(notebook_id),
                            {"cell_id": cell_id, "error": str(e)},
                        ),
                    )
                    await _broadcast_message(
                        notebook_id,
                        _make_message(
                            MessageType.CELL_STATUS,
                            next_notebook_sequence(notebook_id),
                            cell_status_payload(cell_id, "error"),
                        ),
                    )
                    await _broadcast_downstream_stale(notebook_id, downstream_stale)
                    cascade_failed = True
            if not cascade_failed:
                previous_snapshot = session.capture_cell_state_snapshot()
                await _refresh_and_broadcast_changed_staleness(
                    session,
                    notebook_id,
                    previous_snapshot,
                    preserve_ready_cell_id=plan.target_cell_id,
                )
        finally:
            execution_state.running_cell = None


async def _execute_run_all(
    websocket: WebSocket,
    session: NotebookSession,
    cell_ids: list[str],
    execution_state: NotebookExecutionState,
    notebook_id: str,
    force: bool = False,
    continue_on_error: bool = True,
) -> None:
    """Execute all requested notebook cells in notebook order.

    *force* means rerun-all: every cell bypasses its own cache. Runs of consecutive
    batchable cells go through ``CellExecutor.execute_batch``; the rest (workers,
    loops, explicit timeouts, RW mounts) run one at a time. After a failure, cells
    that read from the failed one are blocked, or the run stops when
    *continue_on_error* is False.
    """
    del websocket

    executor = _make_executor_with_progress(session, notebook_id)

    requested_ids = set(cell_ids)
    runnable = [
        cell
        for cell in session.notebook_state.cells
        if cell.id in requested_ids and cell.source.strip()
    ]
    partition = partition_batchable_runs(executor, runnable)
    batching_allowed = True
    try:
        resolve_harness_user()
    except LocalExecutionRefused:
        batching_allowed = False

    logger.info(
        "Run all for notebook %s: %d cells, %d partitioned runs (force=%s, continue_on_error=%s)",
        notebook_id,
        len(runnable),
        len(partition),
        force,
        continue_on_error,
    )

    had_failure = False
    # Failed cells and their dependents; a dependent must not run on the
    # pre-failure artifacts and report success.
    failed: set[str] = set()
    # Each cell executes at most once: in display order a consumer above its
    # producer would materialise it, then the producer's row would run again.
    with executor.one_run():
        try:
            for kind, cells_in_run in partition:
                if had_failure and not continue_on_error:
                    break

                # Single-cell for size-1 batches (nothing to amortize) and for a
                # host that refuses cell code (each cell then shows the refusal;
                # a refused batch would leave the rest silently idle).
                if kind == "batch" and len(cells_in_run) >= 2 and batching_allowed:
                    batch_result = await _run_partition_batch(
                        session=session,
                        executor=executor,
                        cells_in_run=cells_in_run,
                        notebook_id=notebook_id,
                        force=force,
                        execution_state=execution_state,
                    )
                    if not batch_result.completed:
                        had_failure = True
                        # Cells after the failure are not_run. With continue_on_error, run
                        # them single-cell; any with a failed upstream is marked blocked.
                        not_run_ids = {
                            r.cell_id for r in batch_result.cell_results if r.status == "not_run"
                        }
                        # Any status that is neither clean nor not_run.
                        failed.update(
                            r.cell_id
                            for r in batch_result.cell_results
                            if r.status not in ("ok", "cache_hit", "not_run")
                        )
                        for cell in cells_in_run:
                            if cell.id not in not_run_ids:
                                continue
                            if not continue_on_error:
                                break
                            if _upstream_that_failed(session, cell.id, failed) is not None:
                                await _mark_blocked_by_failure(session, notebook_id, cell.id)
                                failed.add(cell.id)
                                continue
                            ok = await _run_partition_single_cell(
                                session=session,
                                executor=executor,
                                cell=cell,
                                notebook_id=notebook_id,
                                force=force,
                                execution_state=execution_state,
                            )
                            if not ok:
                                failed.add(cell.id)
                    continue

                for cell in cells_in_run:
                    if had_failure and not continue_on_error:
                        break
                    if _upstream_that_failed(session, cell.id, failed) is not None:
                        await _mark_blocked_by_failure(session, notebook_id, cell.id)
                        failed.add(cell.id)
                        continue
                    ok = await _run_partition_single_cell(
                        session=session,
                        executor=executor,
                        cell=cell,
                        notebook_id=notebook_id,
                        force=force,
                        execution_state=execution_state,
                    )
                    if not ok:
                        had_failure = True
                        failed.add(cell.id)
        finally:
            execution_state.running_cell = None


async def _run_partition_batch(
    *,
    session: NotebookSession,
    executor: CellExecutor,
    cells_in_run: list,
    notebook_id: str,
    force: bool,
    execution_state: NotebookExecutionState,
):
    """Execute a partition run via ``execute_batch``, streaming per-cell broadcasts.

    Returns the raw ``BatchExecutionResult`` so the caller can send ``not_run``
    cells through single-cell continuation.
    """
    cell_specs: list[dict[str, Any]] = []
    # A failed mount becomes a per-cell error (as in single-cell), not an
    # abort of the whole batch.
    mount_failed_cells: list[tuple[str, Exception]] = []
    for cell in cells_in_run:
        annotations = parse_annotations(cell.source)

        effective_env = executor._resolve_effective_runtime_env(cell.id, annotations.env)

        # Batched cells may have RO mounts (never RW); materialize them.
        mount_specs = executor._resolve_cell_mount_specs(cell.id, cell.source)
        mount_manifest: dict[str, dict[str, str]] = {}
        if mount_specs:
            try:
                # Not the resolver directly: ``_prepare_mounts`` loads
                # credentials first.
                resolved_mounts = await executor._prepare_mounts(mount_specs)
            except Exception as exc:
                mount_failed_cells.append((cell.id, exc))
                continue
            mount_manifest = {
                name: {
                    "local_path": str(rm.local_path),
                    "mode": rm.spec.mode.value,
                }
                for name, rm in resolved_mounts.items()
            }

        # Pin declared lake tables to snapshots, as single-cell does.
        table_manifest: dict[str, dict[str, Any]] = {}
        if annotations.tables:
            _, table_snapshots = await executor._fingerprint_tables(annotations.tables)
            try:
                table_manifest = executor._manifest_tables(annotations.tables, table_snapshots)
            except RuntimeError as exc:
                mount_failed_cells.append((cell.id, exc))
                continue

        cell_specs.append(
            {
                "cell_id": cell.id,
                "source": cell.source,
                "consumed_vars": sorted(
                    session.dag.consumed_variables.get(cell.id, set())
                    if session.dag is not None
                    else set()
                ),
                # Read-set for runtime mutation detection (forms the static
                # analyzer can't see).
                "references": sorted(cell.references or []),
                "env": effective_env,
                "mount_manifest": mount_manifest,
                "table_manifest": table_manifest,
                # Same url as single-cell; not ``_ambient_strata_url()``,
                # which may be the team store.
                "strata_url": executor._cell_strata_url(),
                # Needed for ``strata.promote(...)`` inside Run All.
                "strata_promote_url": executor._ambient_promote_url(),
                "source_hash": "",
                "env_hash": "",
            }
        )

    cells_by_id = {c.id: c for c in cells_in_run}

    async def _emit(result: BatchCellResult) -> None:
        cell = cells_by_id.get(result.cell_id)
        if cell is None:
            return

        execution_state.running_cell = result.cell_id

        # So the frontend sees idle, running, ready in order.
        session.mark_cell_running(result.cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(notebook_id),
                _running_payload(session, result.cell_id, cell.source),
            ),
        )

        synthetic = CellExecutionResult(
            cell_id=result.cell_id,
            success=result.status in ("ok", "cache_hit"),
            stdout=result.stdout,
            stderr=result.stderr,
            outputs=dict(result.outputs),
            display_outputs=list(result.display_outputs),
            duration_ms=0.0,  # not tracked per cell inside batches
            cache_hit=result.cache_hit,
            error=result.error,
            execution_method="batch" if not result.cache_hit else "cached",
            mutation_warnings=result.mutation_warnings,
        )

        session.record_execution(result.cell_id, 0.0, result.cache_hit)
        session.apply_execution_result_metadata(result.cell_id, synthetic)
        await _broadcast_execution_result(notebook_id, result.cell_id, synthetic)

        if synthetic.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                previous_snapshot,
                preserve_ready_cell_id=result.cell_id,
            )
        else:
            downstream_stale = session.mark_cell_error(result.cell_id)
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_STATUS,
                    next_notebook_sequence(notebook_id),
                    cell_status_payload(result.cell_id, CellStatus.ERROR),
                ),
            )
            await _broadcast_downstream_stale(notebook_id, downstream_stale)

    from strata.notebook.executor import BatchExecutionResult

    if cell_specs:
        batch_result = await executor.execute_batch(
            cell_specs,
            use_cache=not force,
            on_cell_event=_emit,
        )
    else:
        # Every cell had a mount failure: skip the subprocess.
        batch_result = BatchExecutionResult(
            cell_results=[],
            completed=True,
            end_reason="complete",
        )

    # Emitted after the batch's cells, and added to cell_results so the
    # dispatcher's continue_on_error tracking sees them.
    for failed_cell_id, exc in mount_failed_cells:
        synthetic = BatchCellResult(
            cell_id=failed_cell_id,
            status="cell_error",
            error=f"Mount preparation failed: {exc}",
            traceback=None,
        )
        await _emit(synthetic)
        batch_result.cell_results.append(synthetic)

    # So the dispatcher treats mount failures as a batch failure.
    if mount_failed_cells:
        batch_result.completed = False
        if batch_result.end_reason == "complete":
            batch_result.end_reason = "cell_error"
        if batch_result.failed_cell_id is None:
            batch_result.failed_cell_id = mount_failed_cells[0][0]

    return batch_result


def _upstream_that_failed(session: NotebookSession, cell_id: str, failed: set[str]) -> str | None:
    """The failed cell this one reads from, directly or through others.

    Run All walks display order, not topological order, so a consumer can precede
    its producer.
    """
    dag = session.dag
    if dag is None or not failed:
        return None
    seen: set[str] = set()
    queue = list(dag.cell_upstream.get(cell_id, []))
    while queue:
        upstream_id = queue.pop()
        if upstream_id in seen:
            continue
        seen.add(upstream_id)
        if upstream_id in failed:
            return upstream_id
        queue.extend(dag.cell_upstream.get(upstream_id, []))
    return None


async def _mark_blocked_by_failure(
    session: NotebookSession, notebook_id: str, cell_id: str
) -> None:
    """Say a cell did not run because what it reads from failed.

    Stale rather than error: nothing went wrong in this cell, and its last result is
    still the last thing it computed.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None:
        cell.status = CellStatus.STALE
        cell.staleness = CellStaleness(status=CellStatus.STALE, reasons=[StalenessReason.UPSTREAM])
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_STATUS,
            next_notebook_sequence(notebook_id),
            # With the reason, so a client can say why it did not run.
            cell_status_payload(
                cell_id, CellStatus.STALE, staleness_reasons=[StalenessReason.UPSTREAM.value]
            ),
        ),
    )


async def _run_partition_single_cell(
    *,
    session: NotebookSession,
    executor: CellExecutor,
    cell,
    notebook_id: str,
    force: bool,
    execution_state: NotebookExecutionState,
) -> bool:
    """Run one cell through the per-cell broadcast flow; True on success.

    The caller blocks any cell with a failed producer behind it, so upstream
    materialization here is always safe; a consumer cannot publish a success built
    on artifacts from before the failure.
    """
    cell_id = cell.id
    execution_state.running_cell = cell_id
    session.mark_cell_running(cell_id)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_STATUS,
            next_notebook_sequence(notebook_id),
            _running_payload(session, cell_id, cell.source),
        ),
    )

    try:
        if force:
            result = await executor.execute_cell_rerun(cell_id, cell.source)
        else:
            result = await executor.execute_cell(cell_id, cell.source)

        session.record_execution(
            cell_id,
            result.duration_ms,
            result.cache_hit,
            from_team=result.from_team_cache,
            team_principal=result.team_cache_principal,
            team_promotion=result.team_cache_promotion,
            team_saved_ms=result.team_cache_saved_ms,
        )
        session.apply_execution_result_metadata(cell_id, result)
        await _broadcast_execution_result(notebook_id, cell_id, result)

        if result.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                previous_snapshot,
                preserve_ready_cell_id=cell_id,
            )
            return True

        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(notebook_id),
                cell_status_payload(cell_id, CellStatus.ERROR),
            ),
        )
        await _broadcast_downstream_stale(notebook_id, downstream_stale)
        return False

    except asyncio.CancelledError:
        await _set_cell_idle(session, notebook_id, execution_state.next_sequence(), cell_id)
        raise
    except Exception as exc:
        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_ERROR,
                next_notebook_sequence(notebook_id),
                {"cell_id": cell_id, "error": str(exc)},
            ),
        )
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                next_notebook_sequence(notebook_id),
                cell_status_payload(cell_id, CellStatus.ERROR),
            ),
        )
        await _broadcast_downstream_stale(notebook_id, downstream_stale)
        return False


def _get_inspect_manager(notebook_id: str) -> InspectManager:
    """Get or create an InspectManager for a notebook."""
    if notebook_id not in _notebook_inspect_managers:
        _notebook_inspect_managers[notebook_id] = InspectManager()
    return _notebook_inspect_managers[notebook_id]


async def _handle_inspect_open(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle inspect_open — spawn REPL with cell's inputs loaded."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        return

    seq = execution_state.next_sequence()

    mgr = _get_inspect_manager(notebook_id)
    inspect_session, status = await mgr.open_session(cell_id, session)

    await websocket.send_text(
        _json_encode(
            _make_message(
                MessageType.INSPECT_RESULT,
                seq,
                {
                    "cell_id": cell_id,
                    "action": "open",
                    "ok": inspect_session.ready,
                    "result": status,
                    "type": "str",
                },
            )
        )
    )


async def _handle_inspect_eval(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle inspect_eval — evaluate expression in REPL."""
    cell_id = payload.get("cell_id")
    expr = payload.get("expr", "")
    if not cell_id or not expr:
        return

    seq = execution_state.next_sequence()

    mgr = _get_inspect_manager(notebook_id)
    inspect_session = await mgr.get_session(cell_id)

    if inspect_session is None:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.INSPECT_RESULT,
                    seq,
                    {
                        "cell_id": cell_id,
                        "action": "eval",
                        "ok": False,
                        "error": "No inspect session open for this cell",
                    },
                )
            )
        )
        return

    result = await inspect_session.evaluate(expr)

    await websocket.send_text(
        _json_encode(
            _make_message(
                MessageType.INSPECT_RESULT,
                seq,
                {
                    "cell_id": cell_id,
                    "action": "eval",
                    "expr": expr,
                    **result,
                },
            )
        )
    )


async def _handle_inspect_close(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle inspect_close — shut down REPL."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        return

    seq = execution_state.next_sequence()

    mgr = _get_inspect_manager(notebook_id)
    await mgr.close_session(cell_id)

    await websocket.send_text(
        _json_encode(
            _make_message(
                MessageType.INSPECT_RESULT,
                seq,
                {
                    "cell_id": cell_id,
                    "action": "close",
                    "ok": True,
                    "result": "closed",
                },
            )
        )
    )


async def _handle_impact_preview_request(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle impact_preview_request — user wants to see impact before running."""
    cell_id = payload.get("cell_id")
    if not cell_id:
        return

    seq = execution_state.next_sequence()

    analyzer = ImpactAnalyzer(session)
    impact = analyzer.preview(cell_id)

    await _send_message(
        websocket,
        _make_message(MessageType.IMPACT_PREVIEW, seq, impact_preview_payload(asdict(impact))),
    )


async def _handle_profiling_request(
    websocket: WebSocket,
    session: NotebookSession,
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle profiling_request — return notebook profiling summary."""
    seq = execution_state.next_sequence()

    summary = session.get_profiling_summary()

    await websocket.send_text(
        _json_encode(
            _make_message(MessageType.PROFILING_SUMMARY, seq, profiling_summary_payload(summary))
        )
    )


async def _handle_dependency_add(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle dependency_add — submit an async env job for ``uv add``."""
    from strata.notebook.routes import validate_package_name

    package = payload.get("package", "")
    if not package:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing 'package' in payload"),
                )
            )
        )
        return

    try:
        package = validate_package_name(package)
    except ValueError as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.next_sequence(), error_payload(str(e))
                )
            )
        )
        return

    try:
        await session.submit_environment_job(action="add", package=package)
    except RuntimeError as exc:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload(str(exc), code="ENVIRONMENT_BUSY"),
                )
            )
        )


async def _handle_dependency_remove(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle dependency_remove — submit an async env job for ``uv remove``."""
    from strata.notebook.routes import validate_package_name

    package = payload.get("package", "")
    if not package:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload("Missing 'package' in payload"),
                )
            )
        )
        return

    try:
        package = validate_package_name(package)
    except ValueError as e:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.next_sequence(), error_payload(str(e))
                )
            )
        )
        return

    try:
        await session.submit_environment_job(action="remove", package=package)
    except RuntimeError as exc:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.next_sequence(),
                    error_payload(str(exc), code="ENVIRONMENT_BUSY"),
                )
            )
        )


async def broadcast_notebook_sync(notebook_id: str, session: Any) -> None:
    """Broadcast full notebook state to all WS clients.

    For REST and MCP changes made outside a WebSocket.
    """
    dag_edges = session.dag.serialize_edges() if session.dag else []

    state = session.serialize_notebook_state()
    state["dag"] = {
        "edges": dag_edges,
        "roots": list(session.dag.roots) if session.dag else [],
        "leaves": list(session.dag.leaves) if session.dag else [],
        "topological_order": (session.dag.topological_order if session.dag else []),
    }

    await _broadcast_message(
        notebook_id,
        _make_message(MessageType.NOTEBOOK_STATE, next_notebook_sequence(notebook_id), state),
    )


def _running_payload(session, cell_id: str, source: str) -> dict[str, Any]:
    """Build the payload for a ``cell_status: running`` broadcast.

    A cell bound for a remote worker adds ``remote_worker`` and ``remote_transport``
    for the UI's dispatching badge. Uses the executor's own resolver
    (:meth:`CellExecutor._resolve_effective_worker`) so the two cannot disagree.
    """
    try:
        annotations = parse_annotations(source)
    except Exception:
        return cell_status_payload(cell_id, "running")

    cell = session.notebook_state.get_cell(cell_id)
    effective_name = (
        annotations.worker
        or (cell.worker if cell else None)
        or session.notebook_state.worker
        or "local"
    )

    try:
        worker_spec = resolve_worker_spec(session.notebook_state, effective_name)
    except Exception:
        return cell_status_payload(cell_id, "running")

    if worker_spec is None or worker_spec.backend == WorkerBackendType.LOCAL:
        return cell_status_payload(cell_id, "running")

    return cell_status_payload(
        cell_id,
        "running",
        remote_worker=worker_spec.name,
        remote_transport=worker_transport(worker_spec),
    )


def _execution_result_payload(cell_id: str, result: CellExecutionResult) -> dict[str, Any]:
    """Build the payload for ``cell_output`` (success) or ``cell_error`` (failure).

    The one place this shape is built. Remote fields (worker, transport, build_id,
    build_state, error_code) appear on both, so the UI shows where a cell ran either
    way.
    """
    payload: dict[str, Any] = {"cell_id": cell_id}
    if result.success:
        payload.update(
            {
                "outputs": result.outputs,
                "cache_hit": result.cache_hit,
                "duration_ms": int(result.duration_ms),
                "artifact_uri": result.artifact_uri,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_method": result.execution_method,
                "mutation_warnings": result.mutation_warnings,
            }
        )
        if result.display_outputs:
            payload["displays"] = result.display_outputs
        if result.display_output:
            payload["display"] = result.display_output
    else:
        payload["error"] = result.error
        if result.suggest_install:
            payload["suggest_install"] = result.suggest_install
            # Picks the install endpoint (uv vs install.packages).
            payload["suggest_install_language"] = result.suggest_install_language or "python"

    for field_name in (
        "remote_worker",
        "remote_transport",
        "remote_build_id",
        "remote_build_state",
        "remote_error_code",
    ):
        value = getattr(result, field_name, None)
        if value:
            payload[field_name] = value

    return payload


async def _broadcast_upstream_results(notebook_id: str, executor: Any) -> None:
    """Announce every cell this run settled on the way to the one asked for.

    A client needs each result, not just a status: a broken cell needs its error,
    and a cell that ran clean again needs the output that clears it. Every cell kind
    turns a broken upstream into a failed result, so every driving path reaches here.
    """
    for upstream_id, upstream_result in getattr(executor, "upstream_results", {}).items():
        await _broadcast_execution_result(notebook_id, upstream_id, upstream_result)


async def _broadcast_execution_result(
    notebook_id: str,
    cell_id: str,
    result: CellExecutionResult,
) -> None:
    """Broadcast stdout and stderr ``cell_console`` frames, then the result.

    The result is ``cell_output`` on success or ``cell_error`` on failure. Each
    frame draws its own sequence as it is sent; a shared one would make a client
    deduping on ``seq`` keep the console and drop the result.
    """
    ts = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")

    # The frontend appends, so send only what streaming has not shown
    # (including any chunk that was dropped).
    delivered = console_relay.streamed(notebook_id, cell_id)
    if delivered is not None:
        console_relay.clear_streamed(notebook_id, cell_id)
        for stream, text in (("stdout", result.stdout), ("stderr", result.stderr)):
            tail = (text or "")[delivered.get(stream, 0) :]
            if tail:
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_CONSOLE,
                        next_notebook_sequence(notebook_id),
                        CellConsolePayload(cell_id=cell_id, stream=stream, text=tail).model_dump(
                            mode="json"
                        ),
                        ts=ts,
                    ),
                )
        await _broadcast_output_or_error(notebook_id, cell_id, result, ts)
        return

    if result.stdout:
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_CONSOLE,
                next_notebook_sequence(notebook_id),
                CellConsolePayload(cell_id=cell_id, stream="stdout", text=result.stdout).model_dump(
                    mode="json"
                ),
                ts=ts,
            ),
        )

    if result.stderr:
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_CONSOLE,
                next_notebook_sequence(notebook_id),
                CellConsolePayload(cell_id=cell_id, stream="stderr", text=result.stderr).model_dump(
                    mode="json"
                ),
                ts=ts,
            ),
        )

    await _broadcast_output_or_error(notebook_id, cell_id, result, ts)


async def _broadcast_output_or_error(
    notebook_id: str,
    cell_id: str,
    result: CellExecutionResult,
    ts: str,
) -> None:
    """Emit the terminal ``cell_output`` / ``cell_error`` for one execution."""
    payload = _execution_result_payload(cell_id, result)
    if result.success:
        # Every stored variable, not just ``artifact_uri``: each may be promoted.
        session = _get_session_manager().get_session(notebook_id)
        cell = session.notebook_state.get_cell(cell_id) if session else None
        if cell is not None:
            payload["artifact_uris"] = dict(cell.artifact_uris)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_OUTPUT if result.success else MessageType.CELL_ERROR,
            next_notebook_sequence(notebook_id),
            payload,
            ts=ts,
        ),
    )


async def broadcast_presence(notebook_id: str, session: NotebookSession) -> None:
    """Send every connection on the session who is on it.

    Per connection, because each frame names the receiver's own identity in ``you``.
    """
    connections = _notebook_connections.get(notebook_id, [])
    if not connections:
        return
    principals = session.presence.snapshot()
    # A fresh sequence (clients dedupe on `seq`), allocated once: per-recipient
    # numbers would leave each client a gap the size of the audience.
    seq = next_notebook_sequence(notebook_id)
    for ws in list(connections):
        you = session.presence.principal_of(ws) or resolve_author()
        message = _make_message(
            MessageType.PRESENCE,
            seq,
            PresencePayload.model_validate({"principals": principals, "you": you}).model_dump(
                mode="json"
            ),
        )
        try:
            await ws.send_text(_json_encode(message))
        except Exception:
            logger.debug("Presence frame not delivered to a closing connection")


async def _handle_cell_focus(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    notebook_id: str,
) -> None:
    """Handle cell_focus: the cell this connection is on, or null."""
    cell_id = payload.get("cell_id")
    if cell_id is not None and session.notebook_state.get_cell(str(cell_id)) is None:
        return
    author = resolve_author(payload.get("author"))
    if session.presence.focus(websocket, author, cell_id):
        await broadcast_presence(notebook_id, session)


async def _broadcast_message(notebook_id: str, message: dict[str, Any]) -> None:
    """Broadcast a message to all connected clients for a notebook."""
    connections = _notebook_connections.get(notebook_id, [])
    if not connections:
        return

    message_text = _json_encode(message)
    disconnected = []

    # Copy: another coroutine can remove from the list during a send await,
    # which would skip the next client.
    for ws in list(connections):
        try:
            await ws.send_text(message_text)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in connections:
            connections.remove(ws)


# Live-mode cost gate: a downstream cell whose last run took longer stays
# STALE instead of auto-running on every control change. High enough that only
# batch-sized cells (long queries, training) are gated.
_LIVE_COST_THRESHOLD_MS = 30_000.0


async def _run_live_cascade(
    session: NotebookSession,
    widget_cell_id: str,
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Auto-run the cheap downstream cells after a @live widget change.

    Runs each stale transitive downstream cell in topological order in force mode
    (no upstream re-materialization). A cell over the cost threshold, or downstream
    of a skipped or failed one, stays STALE so an expensive tail does not re-run on
    every drag.
    """
    dag = session.dag
    if dag is None:
        return

    reachable: set[str] = set()
    queue = list(dag.cell_downstream.get(widget_cell_id, []))
    while queue:
        cid = queue.pop()
        if cid in reachable:
            continue
        reachable.add(cid)
        queue.extend(dag.cell_downstream.get(cid, []))

    # Decide targets before running any: each run's staleness recompute can
    # demote a not-yet-run sibling from STALE to IDLE, and it would be skipped.
    stale_targets = {
        cid
        for cid in reachable
        if (c := session.notebook_state.get_cell(cid)) is not None and c.status == CellStatus.STALE
    }

    blocked: set[str] = set()
    for cid in dag.topological_order:
        if cid not in stale_targets:
            continue
        if any(up in blocked for up in dag.cell_upstream.get(cid, [])):
            blocked.add(cid)  # a stale input can't be produced
            continue
        samples = session.execution_history.get(cid) or []
        if samples and samples[-1].duration_ms > _LIVE_COST_THRESHOLD_MS:
            blocked.add(cid)  # too expensive to auto-run; leave it stale
            continue
        result = await execute_cell_and_broadcast(
            session, cid, execution_state, notebook_id, mode="force"
        )
        if result is None or not result.success:
            blocked.add(cid)


async def apply_widget_values(
    session: NotebookSession,
    cell_id: str,
    coerced: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> CellExecutionResult | None:
    """Set a widget's controls and re-materialize it, chaining the live cascade.

    Shared by the WebSocket handler and the MCP tool so both get the ``# @live``
    cascade. The caller must hold the execution reservation; values are written
    inside it, so an update refused as busy leaves the stored values unchanged.
    """
    from strata.notebook.annotations import parse_annotations
    from strata.notebook.runtime_state import persist_cell_widget_values

    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        return None
    cell.widget_values = persist_cell_widget_values(session.path, cell_id, coerced)
    # force: cache-off, no upstream materialization (widgets have none).
    result = await execute_cell_and_broadcast(
        session, cell_id, execution_state, notebook_id, mode="force"
    )
    if parse_annotations(cell.source).live:
        await _run_live_cascade(session, cell_id, execution_state, notebook_id)
    return result


async def _handle_widget_update(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle widget_update: persist values, re-run the widget cell, flag downstream.

    A value change is an upstream artifact change: the widget cell re-runs in force
    mode and downstream cells go stale for the user to run.
    """
    cell_id = payload.get("cell_id")
    values = payload.get("values")
    if not cell_id or not isinstance(values, dict):
        await _send_error_message(
            websocket, execution_state.next_sequence(), "Missing cell_id or values"
        )
        return

    cell = session.notebook_state.get_cell(cell_id)
    if cell is None or cell.language != CellLanguage.WIDGET:
        await _send_error_message(
            websocket, execution_state.next_sequence(), f"Cell {cell_id} is not a widget cell"
        )
        return

    from strata.notebook.widget_analyzer import analyze_widget_cell, coerce_widget_values

    coerced = coerce_widget_values(analyze_widget_cell(cell.source).descriptors, values)
    if not coerced:
        await _send_error_message(
            websocket, execution_state.next_sequence(), "No valid widget values in update"
        )
        return

    busy_cell = await _reserve_execution_request(execution_state, cell_id)
    if busy_cell is not None:
        await _send_error_message(
            websocket,
            next_notebook_sequence(notebook_id),
            (
                f"Notebook is already executing cell {busy_cell}"
                if busy_cell
                else "Notebook is already executing another cell"
            ),
        )
        return

    async def _operation() -> None:
        await apply_widget_values(session, cell_id, coerced, execution_state, notebook_id)

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        cell_id,
        _operation,
    )
    if not scheduled:
        await _release_execution_request(execution_state, cell_id)


# --- C->S dispatch registry ---
# Handlers declare only the dispatch args they consume; the loop passes those
# as kwargs (as FastAPI does). At module bottom so every handler exists.

_DISPATCH_FIELDS = frozenset({"websocket", "session", "payload", "execution_state", "notebook_id"})

_C2SHandler = Callable[..., Awaitable[None]]

_C2S_HANDLERS: dict[str, _C2SHandler] = {
    MessageType.CELL_EXECUTE: _handle_cell_execute,
    MessageType.CELL_EXECUTE_CASCADE: _handle_cell_execute_cascade,
    MessageType.CELL_EXECUTE_FORCE: _handle_cell_execute_force,
    MessageType.CELL_EXECUTE_RERUN: _handle_cell_execute_rerun,
    MessageType.CELL_CANCEL: _handle_cell_cancel,
    MessageType.NOTEBOOK_RUN_ALL: _handle_notebook_run_all,
    MessageType.NOTEBOOK_RERUN_ALL: _handle_notebook_rerun_all,
    MessageType.CELL_SOURCE_UPDATE: _handle_cell_source_update,
    MessageType.CELL_FOCUS: _handle_cell_focus,
    MessageType.CELL_RUN_TESTS: _handle_cell_run_tests,
    MessageType.NOTEBOOK_SYNC: _handle_notebook_sync,
    MessageType.IMPACT_PREVIEW_REQUEST: _handle_impact_preview_request,
    MessageType.PROFILING_REQUEST: _handle_profiling_request,
    MessageType.INSPECT_OPEN: _handle_inspect_open,
    MessageType.INSPECT_EVAL: _handle_inspect_eval,
    MessageType.INSPECT_CLOSE: _handle_inspect_close,
    MessageType.DEPENDENCY_ADD: _handle_dependency_add,
    MessageType.DEPENDENCY_REMOVE: _handle_dependency_remove,
    MessageType.VARIANT_SET_ACTIVE: _handle_variant_set_active,
    MessageType.VARIANT_ADD: _handle_variant_add,
    MessageType.WIDGET_UPDATE: _handle_widget_update,
}


@cache
def _handler_args(handler: _C2SHandler) -> tuple[str, ...]:
    """Return the dispatch arg names this handler declares (cached per handler)."""
    return tuple(inspect.signature(handler).parameters)


# An unknown param name is a typo: fail at import, not at request time.
for _msg_type, _handler in _C2S_HANDLERS.items():
    _unknown = set(_handler_args(_handler)) - _DISPATCH_FIELDS
    if _unknown:
        _name = getattr(_handler, "__name__", repr(_handler))
        raise RuntimeError(
            f"Handler {_name} for {_msg_type!r} declares unknown "
            f"dispatch arg(s): {sorted(_unknown)}. "
            f"Available: {sorted(_DISPATCH_FIELDS)}"
        )
del _msg_type, _handler, _unknown

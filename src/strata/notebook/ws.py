"""WebSocket handler for real-time notebook execution updates.

Manages WebSocket connections per notebook, dispatches client messages,
and streams server updates (cell status, console output, execution results).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from strata.notebook.annotations import parse_annotations
from strata.notebook.cascade import CascadePlanner
from strata.notebook.causality import skip_none
from strata.notebook.executor import (
    BatchCellResult,
    CellExecutionResult,
    CellExecutor,
    partition_batchable_runs,
)
from strata.notebook.impact import ImpactAnalyzer
from strata.notebook.inspect_repl import InspectManager
from strata.notebook.models import CellLanguage, CellStaleness, CellStatus, WorkerBackendType
from strata.notebook.protocol import MessageType
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
)

if TYPE_CHECKING:
    from strata.notebook.cascade import CascadePlan
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/notebooks", tags=["notebooks_ws"])

# Per-notebook WebSocket connections (for broadcast)
_notebook_connections: dict[str, list[WebSocket]] = {}


@dataclass
class NotebookExecutionState:
    """Per-notebook WebSocket execution bookkeeping.

    Attributes
    ----------
    sequence : int
        Monotonic outbound message counter for ordering WS broadcasts.
    running_cell : str or None
        Cell ID currently executing on the worker.
    requested_cell : str or None
        Cell ID the user has asked to run, queued before execution starts.
    cascade_plan : CascadePlan or None
        Active cascade plan when a multi-cell cascade is in flight.
    execution_task : asyncio.Task[None] or None
        Background task running the cell; cleared once it completes.
    control_lock : asyncio.Lock
        Serializes execution-control transitions (start / stop / requeue).
    """

    sequence: int = 0
    running_cell: str | None = None
    requested_cell: str | None = None
    cascade_plan: CascadePlan | None = None
    execution_task: asyncio.Task[None] | None = None
    control_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def next_sequence(self) -> int:
        """Increment and return the outbound message sequence number."""
        self.sequence += 1
        return self.sequence

    def active_task(self) -> asyncio.Task[None] | None:
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


# Per-notebook execution state
_notebook_execution_state: dict[str, NotebookExecutionState] = {}

# Per-notebook inspect managers
_notebook_inspect_managers: dict[str, InspectManager] = {}

# Pending "cancel after grace period" tasks. When the last WS disconnects
# we don't tear down execution + inspect state immediately — the user may
# be reconnecting after a tmux detach / VPN blip / browser refresh, and
# cancelling a long-running cell because of that is bad UX. We schedule a
# grace-period task instead, and any incoming WS upgrade for the same
# notebook cancels the pending teardown.
_notebook_grace_tasks: dict[str, asyncio.Task[None]] = {}
_GRACE_CANCEL_SECONDS = 60.0


def _get_session_manager() -> SessionManager:
    """Get the session manager from routes module."""
    from strata.notebook.routes import get_session_manager

    return get_session_manager()


# ============================================================================
# Protocol message types
# ============================================================================


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
    """Build a notebook WebSocket protocol message envelope.

    Centralizes the ``{type, seq, ts, payload}`` shape every send site
    used to build inline. ``ts`` defaults to "now"; callers that need
    multiple messages to share a single timestamp (e.g. the stdout /
    stderr / output trio emitted from one execution result) pass an
    explicit value.
    """
    return {
        "type": msg_type,
        "seq": seq,
        "ts": ts if ts is not None else _utc_timestamp(),
        "payload": payload,
    }


# ============================================================================
# Message Serialization
# ============================================================================


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
    await websocket.send_text(_json_encode(_make_message(MessageType.ERROR, seq, {"error": error})))


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
        _make_message(MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": "idle"}),
    )


async def _broadcast_downstream_stale(
    notebook_id: str, seq: int, affected_cell_ids: list[str]
) -> None:
    """Broadcast STALE status for cells downstream of a just-errored cell.

    ``Session.mark_cell_error`` returns the list of downstream cells
    whose status it flipped from READY → STALE; this helper pushes
    a cell_status frame for each so the frontend stops showing them
    green when an upstream cell is red.
    """
    for cell_id in affected_cell_ids:
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": CellStatus.STALE}
            ),
        )


async def _broadcast_staleness_updates(
    session: NotebookSession,
    notebook_id: str,
    seq: int,
    staleness_map: dict[str, CellStaleness],
) -> None:
    """Broadcast backend staleness state to all notebook clients."""
    for cell_id, staleness in staleness_map.items():
        payload: dict[str, Any] = {
            "cell_id": cell_id,
            "status": staleness.status,
            "staleness_reasons": (
                [reason.value for reason in staleness.reasons] if staleness.reasons else []
            ),
        }
        causality = session.causality_map.get(cell_id)
        if causality:
            payload["causality"] = asdict(causality, dict_factory=skip_none)

        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_STATUS, seq, payload),
        )


async def _refresh_and_broadcast_changed_staleness(
    session: NotebookSession,
    notebook_id: str,
    seq: int,
    previous_snapshot: dict[str, CellStateSnapshot],
    *,
    preserve_ready_cell_id: str | None = None,
) -> dict[str, CellStaleness]:
    """Recompute notebook staleness and broadcast only changed cells."""
    staleness_map = session.compute_staleness()
    if preserve_ready_cell_id is not None:
        session.mark_executed_ready(preserve_ready_cell_id)
        staleness_map[preserve_ready_cell_id] = CellStaleness(
            status=CellStatus.READY,
            reasons=[],
        )
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
        await _broadcast_staleness_updates(session, notebook_id, seq, changed)

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
    seq: int,
    operation_factory: Any,
) -> bool:
    """Schedule notebook execution so the WebSocket can keep receiving messages."""
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
            seq,
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
    """Cancel the active execution task and drop inspect/exec state.

    Split out of ``_cleanup_notebook_websocket`` so the grace-period task
    can call it after the reconnect window expires. Idempotent — safe to
    call when state has already been drained.
    """
    execution_state = _notebook_execution_state.get(notebook_id)
    if execution_state is not None:
        task = execution_state.active_task()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        _notebook_execution_state.pop(notebook_id, None)

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

    Spawned as a background task by ``_cleanup_notebook_websocket`` when
    the last WS for a notebook disconnects. A reconnect during the
    window cancels this task before it fires; otherwise the running cell
    is cancelled and execution / inspect state is dropped.
    """
    try:
        await asyncio.sleep(grace_seconds)
    except asyncio.CancelledError:
        # A new client reconnected during the grace window; preserve
        # state by exiting before the teardown runs.
        raise
    if _notebook_connections.get(notebook_id):
        # Defense-in-depth: a reconnect happened but the cancellation
        # raced; the task scheduler dropped the cancel. Honor the
        # connections list as the source of truth.
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
        # Already removed — concurrent cleanup paths (server shutdown,
        # repeated disconnect) can race here. The removal is idempotent.
        pass

    if connections:
        return

    del _notebook_connections[notebook_id]

    # Don't tear down immediately — give the client a chance to reconnect
    # within the grace window (tmux detach, VPN blip, browser refresh).
    # Replace any existing grace task; the scheduler will discard the
    # old one when garbage-collected.
    existing = _notebook_grace_tasks.pop(notebook_id, None)
    if existing is not None and not existing.done():
        existing.cancel()

    if _GRACE_CANCEL_SECONDS <= 0:
        # Tests opt into immediate teardown by zeroing the grace window;
        # also covers server-shutdown paths where deferring would leak.
        await _tear_down_notebook_state(notebook_id)
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop (e.g. shutdown without a live loop) — tear down inline.
        await _tear_down_notebook_state(notebook_id)
        return
    _notebook_grace_tasks[notebook_id] = loop.create_task(
        _grace_cancel_then_tear_down(notebook_id, _GRACE_CANCEL_SECONDS)
    )


def _cancel_pending_grace_teardown(notebook_id: str) -> None:
    """Abort any pending teardown task for ``notebook_id``.

    Called from the WS upgrade handler when a new client reconnects to a
    notebook that's mid-grace-period. Preserves the running execution
    task + inspect sessions across the disconnect.
    """
    task = _notebook_grace_tasks.pop(notebook_id, None)
    if task is not None and not task.done():
        task.cancel()


# ============================================================================
# WebSocket Handler
# ============================================================================


def _ws_caller_identity(websocket: WebSocket) -> str | None:
    """Resolve the calling user's identity from the WS upgrade headers.

    Mirrors ``routes._caller_identity`` for the WS surface: reads the
    configured ``personal_mode_user_header`` from the request headers and
    returns the stripped value (or ``None`` when scoping is unconfigured
    or the header is absent/blank). The single-user case (header unset)
    returns ``None`` so the gate passes through unchanged.
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
    """Return whether ``caller`` may open a WS for an ``owner``-scoped notebook.

    The WS twin of ``routes._require_owner`` (which raises) — same rules,
    expressed as a boolean so the handler closes with 1008 on ``False``:

    - Unowned notebooks (``owner is None``) → allow (legacy / serviced).
    - Per-user scoping configured but no caller identity → deny, so the
      bypass can't be "just don't send the header."
    - Owned notebook with a mismatched caller → deny.
    """
    from strata.notebook.routes import _user_scoping_enabled

    if owner is None:
        return True
    if caller is None:
        return not _user_scoping_enabled()
    return owner == caller


@router.websocket("/ws/{notebook_id}")
async def notebook_websocket(websocket: WebSocket, notebook_id: str):
    """WebSocket endpoint for real-time notebook updates.

    Accepts messages (C→S):
    - cell_execute              Run a cell (check if cascade needed)
    - cell_execute_cascade      Execute cascade plan
    - cell_execute_force        Run with stale inputs
    - cell_execute_rerun        Force re-execute target cell (cache upstreams)
    - notebook_run_all          Run all non-empty cells in notebook order
    - notebook_rerun_all        Force re-execute every cell (cache off)
    - cell_cancel               Cancel execution
    - cell_source_update        Source code changed (debounced flush)
    - notebook_sync             Request full state
    - impact_preview_request    Compute upstream + downstream impact
    - profiling_request         Compute per-cell duration summary
    - inspect_open              Open an inspect REPL on a cell
    - inspect_eval              Eval an expression in an open REPL
    - inspect_close             Close the REPL
    - dependency_add            Add a Python dep via writer
    - dependency_remove         Remove a Python dep via writer
    - variant_set_active        Switch active variant in a group
    - variant_add               Add a new variant cell
    - agent_cancel              Cancel a running agent task
    - agent_confirm_response    Reply to an agent's pending confirmation

    Sends messages (S→C):
    - cell_status               Status changed (idle/running/ready/error/stale)
    - cell_output               Execution result
    - cell_console              Incremental stdout/stderr
    - cell_error                Execution failed
    - cell_iteration_progress   Loop-cell iteration update
    - dag_update                DAG changed
    - cascade_prompt            Cascade needed
    - cascade_progress          Progress during cascade
    - notebook_state            Full state (response to sync)
    - impact_preview            Upstream + downstream impact
    - inspect_result            Result of an inspect_eval
    - profiling_summary         Per-cell duration summary
    - error                     Generic error frame
    """
    # Get or create session
    session_manager = _get_session_manager()
    session = session_manager.get_session(notebook_id)
    if not session:
        await websocket.close(code=1008, reason="Notebook not found")
        return

    # Per-user scoping: refuse the upgrade if a non-owner tries to
    # connect to someone else's notebook session. Without this gate, a
    # leaked notebook_id would let user B observe user A's live state
    # (status, console, DAG) over WS even after the REST surface has
    # been locked down.
    #
    # Same close-on-missing-header rule as ``_require_owner`` in routes.py:
    # if per-user scoping is configured but the request omitted the
    # identity header, deny — otherwise the bypass is "just don't send
    # the header." Single-user deployments (header unset) keep their
    # existing pass-through behavior.
    owner = session.notebook_state.owner
    if not _ws_owner_allowed(owner, _ws_caller_identity(websocket)):
        await websocket.close(code=1008, reason="Notebook not found")
        return

    # Accept connection
    await websocket.accept()

    # If a previous client disconnected within the grace window and the
    # cell is still running, abort the pending teardown so we keep the
    # execution alive for this reconnect.
    _cancel_pending_grace_teardown(notebook_id)

    # Add to connections list
    if notebook_id not in _notebook_connections:
        _notebook_connections[notebook_id] = []
    _notebook_connections[notebook_id].append(websocket)

    execution_state = _ensure_execution_state(notebook_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            msg = _json_decode(data)
            session.touch()

            # Extract message type and payload
            msg_type = msg.get("type")
            payload = msg.get("payload", {})

            handler = _C2S_HANDLERS.get(msg_type) if isinstance(msg_type, str) else None
            if handler is None:
                await websocket.send_text(
                    _json_encode(
                        _make_message(
                            MessageType.ERROR,
                            execution_state.sequence,
                            {"error": f"Unknown message type: {msg_type}"},
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


# ============================================================================
# Message Handlers
# ============================================================================


async def _handle_cell_execute(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_execute message.

    Check if cascade needed. If yes, send cascade_prompt.
    If no, execute cell directly.
    """
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.sequence, {"error": "Missing cell_id"}
                )
            )
        )
        return

    seq = execution_state.next_sequence()

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

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    # Find cell
    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(MessageType.ERROR, seq, {"error": f"Cell {cell_id} not found"})
            )
        )
        return

    # Check if cascade is needed
    planner = CascadePlanner(session)
    plan = planner.plan(cell_id)

    if plan:
        # Cascade needed — send cascade_prompt so the frontend can
        # auto-accept or prompt the user.  No impact_preview here;
        # downstream staleness is communicated via cell_status updates.
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
                seq,
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
        # No cascade needed — execute directly.
        scheduled = await _schedule_execution(
            websocket,
            execution_state,
            notebook_id,
            cell_id,
            seq,
            lambda: _execute_cell_directly(
                websocket, session, cell_id, execution_state, notebook_id
            ),
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
    """Handle notebook_run_all message.

    Execute all non-empty cells in notebook order. ``continue_on_error``
    (default ``True``) controls whether the run continues past a failed
    cell — historically the loop stopped on first failure; with batching
    (issue #26) the per-cell granularity makes continuing useful.
    """
    seq = execution_state.next_sequence()
    continue_on_error = bool(payload.get("continue_on_error", True))

    # Skip inactive variants — they aren't in the DAG, so their references
    # don't resolve (e.g. `X_train` would NameError because the upstream
    # split cell wasn't wired to them).
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
            seq,
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
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        requested_cell,
        seq,
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
    """Handle notebook_rerun_all message.

    Like ``notebook_run_all`` but every cell bypasses its own cache so the
    entire notebook is re-executed end-to-end. Accepts the same
    ``continue_on_error`` (default ``True``) field as ``run_all``.
    """
    seq = execution_state.next_sequence()
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
            seq,
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
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        requested_cell,
        seq,
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
    """Handle cell_execute_cascade message.

    User confirmed cascade — execute all cells in the plan.
    """
    cell_id = payload.get("cell_id")
    plan_id = payload.get("plan_id")

    if not cell_id or not plan_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.sequence,
                    {"error": "Missing cell_id or plan_id"},
                )
            )
        )
        return

    seq = execution_state.next_sequence()

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

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    # Get the cascade plan
    plan = execution_state.cascade_plan
    if not plan or plan.plan_id != plan_id:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, seq, {"error": "Cascade plan not found or expired"}
                )
            )
        )
        return

    # Execute cascade in the background so this socket can still receive cancel.
    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        cell_id,
        seq,
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
    """Handle cell_execute_force message.

    Execute cell with stale inputs ("Run this only").
    """
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.sequence, {"error": "Missing cell_id"}
                )
            )
        )
        return

    seq = execution_state.next_sequence()

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

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    # Execute cell directly, ignoring staleness.
    scheduled = await _schedule_execution(
        websocket,
        execution_state,
        notebook_id,
        cell_id,
        seq,
        lambda: _execute_cell_directly(
            websocket, session, cell_id, execution_state, notebook_id, mode="force"
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
    """Handle cell_execute_rerun message.

    Force re-execute the target cell, bypassing its cache but still
    materializing upstreams through the normal cache path.
    """
    cell_id = payload.get("cell_id")
    if not cell_id:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.sequence, {"error": "Missing cell_id"}
                )
            )
        )
        return

    seq = execution_state.next_sequence()

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

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        await _release_execution_request(execution_state, cell_id)
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    seq,
                    {
                        "error": environment_block_reason,
                        "code": "ENVIRONMENT_BUSY",
                    },
                )
            )
        )
        return

    # If any upstream is stale/missing, run them through cascade so the
    # client sees per-cell status and output frames for every step.
    # The target itself still runs cache-off via target_force.
    planner = CascadePlanner(session)
    plan = planner.plan(cell_id)

    if plan is not None:
        scheduled = await _schedule_execution(
            websocket,
            execution_state,
            notebook_id,
            cell_id,
            seq,
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
            seq,
            lambda: _execute_cell_directly(
                websocket, session, cell_id, execution_state, notebook_id, mode="rerun"
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

    Reuses the cell-execution reservation so a test run (which materializes
    upstreams) can't race a concurrent cell execution. Emits CELL_TEST_STATUS
    (running → ready/error) around a CELL_TEST_RESULTS frame. v1 supports
    Python cells only.
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
    """Handle cell_cancel message.

    Cancel a running cell without clobbering completed cell state.
    """
    cell_id = payload.get("cell_id")
    if not cell_id:
        return

    seq = execution_state.next_sequence()

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
            await _set_cell_idle(session, notebook_id, seq, requested_cell)
        return

    cell = session.notebook_state.get_cell(cell_id)
    if cell is not None and cell.status in {CellStatus.IDLE, CellStatus.RUNNING}:
        await _set_cell_idle(session, notebook_id, seq, cell_id)


async def _handle_agent_cancel(notebook_id: str) -> None:
    """Handle agent_cancel message — abort the active agent run for this notebook."""
    from strata.notebook.routes import cancel_agent

    cancel_agent(notebook_id)


async def _handle_agent_confirm_response(payload: dict[str, Any]) -> None:
    """Handle agent_confirm_response message — relay an approval decision to the LLM gate."""
    from strata.notebook.llm import resolve_approval

    request_id = payload.get("request_id")
    approved = bool(payload.get("approved", False))
    if isinstance(request_id, str):
        resolve_approval(request_id, approved)


async def _handle_cell_source_update(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Handle cell_source_update message.

    Cell source changed — re-analyze and update DAG.
    """
    cell_id = payload.get("cell_id")
    source = payload.get("source")

    if not cell_id or source is None:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.sequence,
                    {"error": "Missing cell_id or source"},
                )
            )
        )
        return

    if len(source) > 1_000_000:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.sequence,
                    {"error": "Cell source exceeds 1MB limit"},
                )
            )
        )
        return

    # Reject updates to the cell currently being executed. Without this
    # guard the executor can hash one source version, write the artifact
    # under that hash, and then have disk + in-memory source overwritten
    # by this update before the run completes — leaving compute_staleness
    # to see a different source on next open and mark the cell stale
    # forever despite having a fresh artifact. control_lock is held only
    # during execution *scheduling* (not the run itself), so we read the
    # running-cell snapshot under it and reject without blocking on long
    # cells. Frontend retries on the next cell_status: idle/ready/error.
    async with execution_state.control_lock:
        running = execution_state.running_cell
        requested = execution_state.requested_cell
    if cell_id in {running, requested}:
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR,
                    execution_state.sequence,
                    {
                        "error": (
                            f"Cannot update cell {cell_id} while it is executing; "
                            "retry after cell finishes"
                        ),
                        "code": "cell_busy",
                        "cell_id": cell_id,
                    },
                )
            )
        )
        return

    seq = execution_state.next_sequence()

    try:
        # Write to disk
        write_cell(session.path, cell_id, source)

        # Update source in session (must happen before re-analysis)
        cell_in_session = session.notebook_state.get_cell(cell_id)
        if cell_in_session:
            cell_in_session.source = source

        # Re-analyze cell and rebuild DAG
        session.re_analyze_cell(cell_id)
        session._run_annotation_validation()

        # Recompute staleness
        staleness_map = session.compute_staleness()

        # Build DAG update message
        dag_edges = session.dag.serialize_edges() if session.dag else []

        # Include per-cell analysis so the frontend can merge
        # authoritative defines/references without a REST round-trip.
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

        # Send DAG update
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.DAG_UPDATE,
                seq,
                {
                    "edges": dag_edges,
                    "roots": list(session.dag.roots) if session.dag else [],
                    "leaves": list(session.dag.leaves) if session.dag else [],
                    "topological_order": (session.dag.topological_order if session.dag else []),
                    "cells": cells_analysis,
                    "variant_groups": [
                        vg.model_dump() for vg in session.notebook_state.variant_groups
                    ],
                },
            ),
        )

        await _broadcast_staleness_updates(session, notebook_id, seq, staleness_map)

    except Exception as e:
        await websocket.send_text(
            _json_encode(_make_message(MessageType.ERROR, seq, {"error": str(e)}))
        )


async def _handle_variant_set_active(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Switch the active variant for a group, then broadcast a dag_update.

    Reuses the same broadcast shape as ``cell_source_update`` since the
    effect on the DAG is the same: a different cell becomes the producer
    for the group's defines, downstream cells go stale.
    """
    group = payload.get("group")
    variant_name = payload.get("name")

    if not isinstance(group, str) or not isinstance(variant_name, str):
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.sequence, {"error": "Missing group or name"}
                )
            )
        )
        return

    seq = execution_state.next_sequence()

    try:
        session.set_variant_active(group, variant_name)
        staleness_map = session.compute_staleness()

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
                seq,
                {
                    "edges": dag_edges,
                    "roots": list(session.dag.roots) if session.dag else [],
                    "leaves": list(session.dag.leaves) if session.dag else [],
                    "topological_order": (session.dag.topological_order if session.dag else []),
                    "cells": cells_analysis,
                    "variant_groups": [
                        vg.model_dump() for vg in session.notebook_state.variant_groups
                    ],
                },
            ),
        )

        await _broadcast_staleness_updates(session, notebook_id, seq, staleness_map)

    except Exception as e:
        await websocket.send_text(
            _json_encode(_make_message(MessageType.ERROR, seq, {"error": str(e)}))
        )


async def _handle_variant_add(
    websocket: WebSocket,
    session: NotebookSession,
    payload: dict[str, Any],
    execution_state: NotebookExecutionState,
    notebook_id: str,
) -> None:
    """Add a sibling variant to a group, then broadcast a dag_update.

    Same broadcast shape as ``variant_set_active`` since the effect on
    the DAG is identical: a new cell appears, and (because the new
    variant becomes active) the producer for the group's defines moves.
    """
    group = payload.get("group")
    if not isinstance(group, str):
        await websocket.send_text(
            _json_encode(
                _make_message(
                    MessageType.ERROR, execution_state.sequence, {"error": "Missing group"}
                )
            )
        )
        return

    seq = execution_state.next_sequence()

    try:
        session.add_variant(group)
        staleness_map = session.compute_staleness()

        # variant_add creates a new cell, so the frontend store needs
        # the full cell payload (source, language, order, ...). The
        # dag_update broadcast only updates *existing* cells — it would
        # silently drop the new variant. Send notebook_state instead,
        # which the frontend handler treats as authoritative when cells
        # are added or removed.
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
            _make_message(MessageType.NOTEBOOK_STATE, seq, state_payload),
        )

        await _broadcast_staleness_updates(session, notebook_id, seq, staleness_map)

    except ValueError as e:
        await websocket.send_text(
            _json_encode(_make_message(MessageType.ERROR, seq, {"error": str(e)}))
        )
    except Exception as e:
        await websocket.send_text(
            _json_encode(_make_message(MessageType.ERROR, seq, {"error": str(e)}))
        )


async def _handle_notebook_sync(
    websocket: WebSocket,
    session: NotebookSession,
    notebook_id: str,
) -> None:
    """Handle notebook_sync message.

    Return full notebook state (for reconnection).
    """
    # Build DAG
    dag_edges = session.dag.serialize_edges() if session.dag else []

    state = session.serialize_notebook_state()
    state["dag"] = {
        "edges": dag_edges,
        "roots": list(session.dag.roots) if session.dag else [],
        "leaves": list(session.dag.leaves) if session.dag else [],
        "topological_order": (session.dag.topological_order if session.dag else []),
    }

    await websocket.send_text(_json_encode(_make_message(MessageType.NOTEBOOK_STATE, 0, state)))


# ============================================================================
# Execution Helpers
# ============================================================================


def _make_executor_with_progress(
    session: NotebookSession,
    notebook_id: str,
) -> CellExecutor:
    """Build a CellExecutor whose loop iterations broadcast progress.

    Each iteration of a ``@loop`` cell fires an ``on_iteration_complete``
    callback; we forward it as a ``cell_iteration_progress`` WS message so
    the frontend can keep its per-cell iteration badge in sync with the
    live execution.
    """
    executor = CellExecutor(session, session.warm_pool)

    async def _broadcast_iteration_progress(progress: dict[str, Any]) -> None:
        seq = next_notebook_sequence(notebook_id)
        # Validate the executor's dict as it crosses into the protocol layer.
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

    executor.on_iteration_complete = _broadcast_iteration_progress
    executor.on_prompt_delta = _broadcast_prompt_delta
    return executor


async def _execute_cell_directly(
    websocket: WebSocket,
    session: NotebookSession,
    cell_id: str,
    execution_state: NotebookExecutionState,
    notebook_id: str,
    mode: Literal["normal", "force", "rerun"] = "normal",
) -> None:
    """Execute a cell directly (not part of cascade).

    ``mode`` selects the executor entry point:

    - ``normal`` — cache-on, materialize upstreams (default).
    - ``force``  — cache-off, *do not* materialize upstreams ("Run this only").
    - ``rerun``  — cache-off, but still materialize upstreams (force the
      target cell only against the current valid upstream graph).
    """
    del websocket
    seq = execution_state.next_sequence()

    # Find cell
    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        return

    # Mark as running — update backend state AND broadcast
    execution_state.running_cell = cell_id
    session.mark_cell_running(cell_id)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_STATUS, seq, _running_payload(session, cell_id, cell.source)
        ),
    )

    # Execute
    executor = _make_executor_with_progress(session, notebook_id)
    try:
        if mode == "force":
            result = await executor.execute_cell_force(cell_id, cell.source)
        elif mode == "rerun":
            result = await executor.execute_cell_rerun(cell_id, cell.source)
        else:
            result = await executor.execute_cell(cell_id, cell.source)

        # Post-execution frames draw a fresh sequence: streaming frames
        # emitted during execution (cell_output_delta,
        # cell_iteration_progress) pull from the same per-notebook
        # counter, so reusing the pre-execution seq would make the
        # canonical result look older than the deltas it supersedes.
        seq = execution_state.next_sequence()

        # Record execution for profiling before broadcasting so the
        # output payload reflects the just-recorded metadata.
        session.record_execution(cell_id, result.duration_ms, result.cache_hit)
        session.apply_execution_result_metadata(cell_id, result)

        await _broadcast_execution_result(notebook_id, seq, cell_id, result)

        if result.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                seq,
                previous_snapshot,
                preserve_ready_cell_id=cell_id,
            )
        else:
            downstream_stale = session.mark_cell_error(cell_id)
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": CellStatus.ERROR}
                ),
            )
            await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)

    except asyncio.CancelledError:
        await _set_cell_idle(session, notebook_id, execution_state.next_sequence(), cell_id)
        raise
    except Exception as e:
        # Fresh seq for the same reason as the success path: deltas may
        # have streamed before the failure.
        seq = execution_state.next_sequence()
        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_ERROR, seq, {"cell_id": cell_id, "error": str(e)}),
        )
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": "error"}),
        )
        await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)
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

    When ``target_force`` is true, the final step (the user-requested target
    cell) runs ``execute_cell_rerun`` so its own cache is bypassed; upstream
    steps still go through normal cached execution.
    """
    del websocket
    seq = execution_state.next_sequence()

    executor = _make_executor_with_progress(session, notebook_id)

    logger.info(
        "Cascade %s: executing %d steps: %s",
        plan.plan_id,
        len(plan.steps),
        [(s.cell_id, s.reason, s.skip) for s in plan.steps],
    )

    cascade_failed = False

    try:
        for i, step in enumerate(plan.steps):
            # Under target_force the cached-ready target must still rerun.
            if step.skip and not (target_force and step.cell_id == plan.target_cell_id):
                continue

            cell_id = step.cell_id
            cell = session.notebook_state.get_cell(cell_id)
            if not cell:
                continue

            # If an earlier cascade step failed, abort remaining steps
            if cascade_failed:
                logger.warning(
                    "Cascade %s: skipping cell %s (earlier step failed)",
                    plan.plan_id,
                    cell_id,
                )
                # Use "stale" (not "idle") so the client can distinguish a
                # cascade-abort from a normal staleness notification.
                cell_to_skip = session.notebook_state.get_cell(cell_id)
                if cell_to_skip:
                    cell_to_skip.status = CellStatus.STALE
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": "stale"}
                    ),
                )
                continue

            execution_state.running_cell = cell_id

            # Send cascade progress
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CASCADE_PROGRESS,
                    seq,
                    CascadeProgressPayload(
                        plan_id=plan.plan_id,
                        current_cell_id=cell_id,
                        completed=i,
                        total=len([s for s in plan.steps if not s.skip]),
                    ).model_dump(mode="json"),
                ),
            )

            # Execute cell — update backend state AND broadcast
            session.mark_cell_running(cell_id)
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_STATUS, seq, _running_payload(session, cell_id, cell.source)
                ),
            )

            try:
                if target_force and cell_id == plan.target_cell_id:
                    result = await executor.execute_cell_rerun(cell_id, cell.source)
                else:
                    result = await executor.execute_cell(cell_id, cell.source)

                # Fresh seq after execution — streaming frames drew from
                # the same counter; the result must not look older.
                seq = execution_state.next_sequence()

                # v1.1: Record execution for profiling
                session.record_execution(cell_id, result.duration_ms, result.cache_hit)
                session.apply_execution_result_metadata(cell_id, result)

                # Broadcast stdout/stderr console + output/error in the
                # same shape as the direct-execute path. Note: cascade
                # previously skipped the stderr console broadcast — that
                # drift is fixed by going through the shared helper.
                await _broadcast_execution_result(notebook_id, seq, cell_id, result)

                # Mark as ready — update backend state AND broadcast
                status = CellStatus.READY if result.success else CellStatus.ERROR
                cascade_cell = session.notebook_state.get_cell(cell_id)
                if cascade_cell:
                    cascade_cell.status = status
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": status}
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

                # If a step fails, abort the rest of the cascade
                if not result.success:
                    cascade_failed = True

            except asyncio.CancelledError:
                await _set_cell_idle(session, notebook_id, execution_state.next_sequence(), cell_id)
                raise
            except Exception as e:
                seq = execution_state.next_sequence()
                downstream_stale = session.mark_cell_error(cell_id)
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_ERROR, seq, {"cell_id": cell_id, "error": str(e)}
                    ),
                )
                await _broadcast_message(
                    notebook_id,
                    _make_message(
                        MessageType.CELL_STATUS, seq, {"cell_id": cell_id, "status": "error"}
                    ),
                )
                await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)
                cascade_failed = True
        if not cascade_failed:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                seq,
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

    ``force`` is rerun-all semantics: every cell bypasses its own cache.

    The cells are partitioned into runs of consecutive batchable cells (per
    ``executor.partition_batchable_runs``) — batches go through
    ``CellExecutor.execute_batch``; non-batchable cells (workers, loops,
    explicit timeouts, RW mounts) use the existing single-cell
    ``execute_cell``. Once a cell fails, the rest of the run-all proceeds
    with ``skip_upstream_materialization=True`` (or aborts entirely when
    ``continue_on_error`` is False) so the failed cell isn't recursively
    re-executed via ``_materialize_upstreams``.
    """
    del websocket
    seq = execution_state.next_sequence()

    executor = _make_executor_with_progress(session, notebook_id)

    requested_ids = set(cell_ids)
    runnable = [
        cell
        for cell in session.notebook_state.cells
        if cell.id in requested_ids and cell.source.strip()
    ]
    partition = partition_batchable_runs(executor, runnable)

    logger.info(
        "Run all for notebook %s: %d cells, %d partitioned runs (force=%s, continue_on_error=%s)",
        notebook_id,
        len(runnable),
        len(partition),
        force,
        continue_on_error,
    )

    had_failure = False
    try:
        for kind, cells_in_run in partition:
            if had_failure and not continue_on_error:
                break

            # Size-1 batches gain nothing from subprocess amortization;
            # route them through single-cell.
            if kind == "batch" and len(cells_in_run) >= 2:
                batch_result = await _run_partition_batch(
                    session=session,
                    executor=executor,
                    cells_in_run=cells_in_run,
                    seq=seq,
                    notebook_id=notebook_id,
                    force=force,
                    execution_state=execution_state,
                )
                if not batch_result.completed:
                    had_failure = True
                    # Batch ended early — any cells after the failed one
                    # are status=not_run. Per issue #26 round-5 design,
                    # they continue via single-cell with
                    # skip_upstream_materialization=True so the failed
                    # upstream isn't recursively re-executed.
                    not_run_ids = {
                        r.cell_id for r in batch_result.cell_results if r.status == "not_run"
                    }
                    for cell in cells_in_run:
                        if cell.id not in not_run_ids:
                            continue
                        if not continue_on_error:
                            break
                        await _run_partition_single_cell(
                            session=session,
                            executor=executor,
                            cell=cell,
                            seq=seq,
                            notebook_id=notebook_id,
                            force=force,
                            skip_upstream=True,
                            execution_state=execution_state,
                        )
                continue

            for cell in cells_in_run:
                if had_failure and not continue_on_error:
                    break
                ok = await _run_partition_single_cell(
                    session=session,
                    executor=executor,
                    cell=cell,
                    seq=seq,
                    notebook_id=notebook_id,
                    force=force,
                    skip_upstream=had_failure,
                    execution_state=execution_state,
                )
                if not ok:
                    had_failure = True
    finally:
        execution_state.running_cell = None


async def _run_partition_batch(
    *,
    session: NotebookSession,
    executor: CellExecutor,
    cells_in_run: list,
    seq: int,
    notebook_id: str,
    force: bool,
    execution_state: NotebookExecutionState,
):
    """Execute a partition run via ``execute_batch`` + stream per-cell broadcasts.

    Returns the raw ``BatchExecutionResult`` so the caller can identify
    the cells that ended up ``status=not_run`` and route them through
    single-cell continuation.
    """
    cell_specs: list[dict[str, Any]] = []
    # Cells whose mount-prep failed get a synthetic cell_error broadcast
    # and don't go into the batch. Without this, a single failed mount
    # would abort the whole batch before any cell ran — single-cell at
    # executor.py L1140 converts the same failure into a per-cell error,
    # not a kill-everything exception.
    mount_failed_cells: list[tuple[str, Exception]] = []
    for cell in cells_in_run:
        annotations = parse_annotations(cell.source)

        # Resolve effective env (notebook + cell + annotation layers) —
        # matches single-cell's _resolve_effective_runtime_env path so
        # # @env annotations and notebook-level overrides reach the
        # batched cell.
        effective_env = executor._resolve_effective_runtime_env(cell.id, annotations.env)

        # Resolve and materialize RO mounts. The partitioner only batches
        # cells without RW mounts, but RO mounts are allowed — we must
        # download/cache their contents and pass local paths into the
        # harness manifest, exactly like single-cell does. Empty list →
        # no mounts → no work.
        mount_specs = executor._resolve_cell_mount_specs(cell.id, cell.source)
        mount_manifest: dict[str, dict[str, str]] = {}
        if mount_specs:
            try:
                resolved_mounts = await executor._mount_resolver.prepare_mounts(mount_specs)
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

        # Resolve declared lake tables to concrete snapshots — the harness
        # injects <name> / <name>_snapshot, exactly like single-cell does.
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
                "env": effective_env,
                "mount_manifest": mount_manifest,
                "table_manifest": table_manifest,
                "strata_url": executor._ambient_strata_url(),
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

        # Brief running broadcast so the frontend transitions idle → running →
        # ready in order. Batched cells complete fast enough that the gap
        # between this and the output frame is hardly visible.
        session.mark_cell_running(result.cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                seq,
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
            duration_ms=0.0,  # Per-cell timing isn't tracked inside batches.
            cache_hit=result.cache_hit,
            error=result.error,
            execution_method="batch" if not result.cache_hit else "cached",
        )

        session.record_execution(result.cell_id, 0.0, result.cache_hit)
        session.apply_execution_result_metadata(result.cell_id, synthetic)
        await _broadcast_execution_result(notebook_id, seq, result.cell_id, synthetic)

        if synthetic.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                seq,
                previous_snapshot,
                preserve_ready_cell_id=result.cell_id,
            )
        else:
            downstream_stale = session.mark_cell_error(result.cell_id)
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_STATUS,
                    seq,
                    {"cell_id": result.cell_id, "status": CellStatus.ERROR},
                ),
            )
            await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)

    from strata.notebook.executor import BatchExecutionResult

    # Run the surviving cells through the batch (if any).
    if cell_specs:
        batch_result = await executor.execute_batch(
            cell_specs,
            use_cache=not force,
            on_cell_event=_emit,
        )
    else:
        # Nothing left to batch — all cells had mount failures. Skip the
        # subprocess spawn and synthesize a completed result.
        batch_result = BatchExecutionResult(
            cell_results=[],
            completed=True,
            end_reason="complete",
        )

    # Now broadcast mount-failure cells via the same _emit path so they
    # land in the WS stream AFTER the successful cells (not before — the
    # prior shape emitted them before the batch ran, swapping notebook
    # order). The results also get included in batch_result.cell_results
    # so the dispatcher's had_failure / continue_on_error tracking sees
    # them — otherwise mount failures were silently ignored by
    # continue_on_error=false (#35 review finding #1).
    for failed_cell_id, exc in mount_failed_cells:
        synthetic = BatchCellResult(
            cell_id=failed_cell_id,
            status="cell_error",
            error=f"Mount preparation failed: {exc}",
            traceback=None,
        )
        await _emit(synthetic)
        batch_result.cell_results.append(synthetic)

    # Mount failures must flip completed → False so the dispatcher
    # picks them up as a real batch-level failure (had_failure tracking,
    # continue_on_error gate).
    if mount_failed_cells:
        batch_result.completed = False
        if batch_result.end_reason == "complete":
            batch_result.end_reason = "cell_error"
        if batch_result.failed_cell_id is None:
            batch_result.failed_cell_id = mount_failed_cells[0][0]

    return batch_result


async def _run_partition_single_cell(
    *,
    session: NotebookSession,
    executor: CellExecutor,
    cell,
    seq: int,
    notebook_id: str,
    force: bool,
    skip_upstream: bool,
    execution_state: NotebookExecutionState,
) -> bool:
    """Existing per-cell broadcast flow. Returns True on success.

    ``skip_upstream`` is set when a prior cell in this run-all has failed —
    avoids ``_materialize_upstreams`` recursively re-executing the failed
    cell. Equivalent to single-cell continuation after batch failure per
    issue #26 round-6 finding #3.
    """
    cell_id = cell.id
    execution_state.running_cell = cell_id
    session.mark_cell_running(cell_id)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_STATUS, seq, _running_payload(session, cell_id, cell.source)
        ),
    )

    try:
        if force and skip_upstream:
            # Continuation after a batch failure during rerun-all needs
            # BOTH "bypass target cache" AND "don't recursively
            # materialize the failed upstream." execute_cell_force is
            # exactly this combination (materialize_upstreams=False +
            # use_cache=False).
            result = await executor.execute_cell_force(cell_id, cell.source)
        elif force:
            result = await executor.execute_cell_rerun(cell_id, cell.source)
        elif skip_upstream:
            result = await executor.execute_cell(
                cell_id, cell.source, skip_upstream_materialization=True
            )
        else:
            result = await executor.execute_cell(cell_id, cell.source)

        # Fresh seq after execution — streaming frames drew from the
        # same counter; the result must not look older.
        seq = execution_state.next_sequence()

        session.record_execution(cell_id, result.duration_ms, result.cache_hit)
        session.apply_execution_result_metadata(cell_id, result)
        await _broadcast_execution_result(notebook_id, seq, cell_id, result)

        if result.success:
            previous_snapshot = session.capture_cell_state_snapshot()
            await _refresh_and_broadcast_changed_staleness(
                session,
                notebook_id,
                seq,
                previous_snapshot,
                preserve_ready_cell_id=cell_id,
            )
            return True

        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                seq,
                {"cell_id": cell_id, "status": CellStatus.ERROR},
            ),
        )
        await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)
        return False

    except asyncio.CancelledError:
        await _set_cell_idle(session, notebook_id, execution_state.next_sequence(), cell_id)
        raise
    except Exception as exc:
        seq = execution_state.next_sequence()
        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_ERROR, seq, {"cell_id": cell_id, "error": str(exc)}),
        )
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_STATUS,
                seq,
                {"cell_id": cell_id, "status": CellStatus.ERROR},
            ),
        )
        await _broadcast_downstream_stale(notebook_id, seq, downstream_stale)
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
        _make_message(MessageType.IMPACT_PREVIEW, seq, asdict(impact)),
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
        _json_encode(_make_message(MessageType.PROFILING_SUMMARY, seq, summary))
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
                    {"error": "Missing 'package' in payload"},
                )
            )
        )
        return

    try:
        package = validate_package_name(package)
    except ValueError as e:
        await websocket.send_text(
            _json_encode(
                _make_message(MessageType.ERROR, execution_state.next_sequence(), {"error": str(e)})
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
                    {"error": str(exc), "code": "ENVIRONMENT_BUSY"},
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
                    {"error": "Missing 'package' in payload"},
                )
            )
        )
        return

    try:
        package = validate_package_name(package)
    except ValueError as e:
        await websocket.send_text(
            _json_encode(
                _make_message(MessageType.ERROR, execution_state.next_sequence(), {"error": str(e)})
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
                    {"error": str(exc), "code": "ENVIRONMENT_BUSY"},
                )
            )
        )


async def execute_cell_for_agent(
    notebook_id: str,
    session: Any,
    cell_id: str,
    source: str,
) -> Any:
    """Execute a cell on behalf of the agent, respecting WS execution state.

    Acquires the control lock, sets running_cell, broadcasts status,
    executes, broadcasts result, and cleans up — same as user-initiated
    execution but without a WebSocket sender.
    """

    execution_state = _notebook_execution_state.get(notebook_id)
    if execution_state is None:
        # No WS clients — execute directly without state tracking
        executor = _make_executor_with_progress(session, notebook_id)
        return await executor.execute_cell(cell_id, source)

    async with execution_state.control_lock:
        task = execution_state.active_task()
        if task is not None:
            raise RuntimeError("Another cell is currently executing. Wait and retry.")

    # Broadcast running status
    await _broadcast_message(
        notebook_id,
        _make_message(MessageType.CELL_STATUS, 0, _running_payload(session, cell_id, source)),
    )

    cell = session.notebook_state.get_cell(cell_id)
    session.mark_cell_running(cell_id)

    try:
        executor = _make_executor_with_progress(session, notebook_id)
        result = await executor.execute_cell(cell_id, source)

        # Update cell status
        status = CellStatus.READY if result.success else CellStatus.ERROR
        if cell:
            cell.status = status

        # Broadcast result
        if result.success and result.outputs:
            await _broadcast_message(
                notebook_id,
                _make_message(
                    MessageType.CELL_OUTPUT,
                    0,
                    {
                        "cell_id": cell_id,
                        "outputs": result.outputs,
                        "cache_hit": result.cache_hit,
                        "duration_ms": int(result.duration_ms),
                        "execution_method": result.execution_method,
                    },
                ),
            )

        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_STATUS, 0, {"cell_id": cell_id, "status": status}),
        )

        return result
    except Exception:
        downstream_stale = session.mark_cell_error(cell_id)
        await _broadcast_message(
            notebook_id,
            _make_message(MessageType.CELL_STATUS, 0, {"cell_id": cell_id, "status": "error"}),
        )
        await _broadcast_downstream_stale(notebook_id, 0, downstream_stale)
        raise


async def broadcast_notebook_sync(notebook_id: str, session: Any) -> None:
    """Broadcast full notebook state to all WS clients.

    Used by the agent loop to push intermediate state changes so
    frontends stay in sync during multi-tool operations.
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
        _make_message(MessageType.NOTEBOOK_STATE, 0, state),
    )


def _running_payload(session, cell_id: str, source: str) -> dict[str, Any]:
    """Build the payload for a ``cell_status: running`` broadcast.

    If the cell will dispatch to a remote worker, include ``remote_worker``
    and ``remote_transport`` so the UI can render a live "dispatching → X"
    badge while the cell executes. Local cells get an unchanged payload.

    Mirrors the precedence chain in
    :meth:`CellExecutor._resolve_effective_worker`: annotation → cell
    override → notebook default → implicit local. We consult the same
    resolver to avoid drifting from the executor's decision.
    """
    payload: dict[str, Any] = {"cell_id": cell_id, "status": "running"}

    try:
        annotations = parse_annotations(source)
    except Exception:
        return payload

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
        return payload

    if worker_spec is None or worker_spec.backend == WorkerBackendType.LOCAL:
        return payload

    payload["remote_worker"] = worker_spec.name
    payload["remote_transport"] = worker_transport(worker_spec)
    return payload


def _execution_result_payload(cell_id: str, result: CellExecutionResult) -> dict[str, Any]:
    """Build the payload for ``cell_output`` (success) or ``cell_error`` (failure).

    Single source of truth for the post-execution payload shape —
    previously inlined four times (``_execute_cell_directly``,
    ``_execute_cascade``, ``_execute_run_all``, ``execute_cell_for_agent``)
    with ~30 lines of ``**({"key": value} if value else {})`` spreads.
    Adding a field to ``CellExecutionResult`` used to require touching
    every site; now it's one place.

    Remote-* fields (worker / transport / build_id / build_state /
    error_code) appear on both success and failure responses so the
    frontend can render the same "ran on X via Y" badge regardless of
    outcome.
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
            # Carry the language tag so the frontend can dispatch to
            # the right install endpoint (uv vs install.packages).
            # Default to ``"python"`` for back-compat with cell results
            # produced before ``suggest_install_language`` was added.
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


async def _broadcast_execution_result(
    notebook_id: str,
    seq: int,
    cell_id: str,
    result: CellExecutionResult,
) -> None:
    """Broadcast the standard execution-finished message sequence.

    Emits, in order:

    1. ``cell_console`` for stdout (if any)
    2. ``cell_console`` for stderr (if any)
    3. ``cell_output`` (success) or ``cell_error`` (failure)

    All four execution-driving handlers (``_execute_cell_directly``,
    ``_execute_cascade``, ``_execute_run_all``,
    ``execute_cell_for_agent``) used to inline this block. The cascade
    path was already drifting — it skipped the stderr broadcast.
    """
    ts = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")

    if result.stdout:
        await _broadcast_message(
            notebook_id,
            _make_message(
                MessageType.CELL_CONSOLE,
                seq,
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
                seq,
                CellConsolePayload(cell_id=cell_id, stream="stderr", text=result.stderr).model_dump(
                    mode="json"
                ),
                ts=ts,
            ),
        )

    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_OUTPUT if result.success else MessageType.CELL_ERROR,
            seq,
            _execution_result_payload(cell_id, result),
            ts=ts,
        ),
    )


async def _broadcast_message(notebook_id: str, message: dict[str, Any]) -> None:
    """Broadcast a message to all connected clients for a notebook."""
    connections = _notebook_connections.get(notebook_id, [])
    if not connections:
        return

    message_text = _json_encode(message)
    disconnected = []

    for ws in connections:
        try:
            await ws.send_text(message_text)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        if ws in connections:
            connections.remove(ws)


# ============================================================================
# C→S dispatch registry
# ============================================================================
#
# Maps every client-to-server message type to its handler. Each handler
# declares only the dispatch args it actually consumes -- e.g.
# ``_handle_agent_cancel`` takes ``(notebook_id)``, ``_handle_notebook_sync``
# takes ``(websocket, session, notebook_id)``. The dispatch loop introspects
# the handler signature at registration time (cached) and passes a kwargs
# dict containing only the requested fields. This is the same technique
# FastAPI's HTTP routes and Slack Bolt's listeners use; the alternative
# (uniform signature with ``del`` for unused args) made handler signatures
# lie about what they consume.
#
# Defined at module bottom so every handler exists at registry-build time.

# The fixed vocabulary of dispatch-context fields. A handler that declares
# any param outside this set is a typo and is caught at registration time
# below.
_DISPATCH_FIELDS = frozenset({"websocket", "session", "payload", "execution_state", "notebook_id"})

# Handlers vary in signature so the precise type is ``Callable[..., ...]``;
# the registration-time check below catches the actual mistakes (unknown
# param names) that type-checking alone wouldn't.
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
    MessageType.AGENT_CANCEL: _handle_agent_cancel,
    MessageType.AGENT_CONFIRM_RESPONSE: _handle_agent_confirm_response,
}


@cache
def _handler_args(handler: _C2SHandler) -> tuple[str, ...]:
    """Return the dispatch arg names this handler declares.

    Cached on first lookup so the per-message dispatch overhead is one
    dict access. Validated against ``_DISPATCH_FIELDS`` at registration
    time below so a typo doesn't make it past module import.
    """
    return tuple(inspect.signature(handler).parameters)


# Validate every handler's signature at import: any param name outside
# the dispatch vocabulary is a typo we want to catch loudly, not silently
# drop on the floor at request time.
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

"""Typed payload models for notebook WebSocket frames.

Built and validated at the emit site in ``ws.py`` and serialized with
``.model_dump(mode="json")``; executor-originated dicts are validated as they
cross into the protocol layer. ``extra="forbid"`` turns an unmodeled field into
a construction error instead of silent protocol drift.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata.notebook.models import CellTestCase
from strata.notebook.protocol import MessageType


class WsPayload(BaseModel):
    """Base for typed WS frame payloads; ``extra="forbid"`` rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class CellStatusPayload(WsPayload):
    """``cell_status``: a cell's execution status changed.

    One model for three shapes: a bare status change; a ``running`` broadcast that
    adds ``remote_worker`` / ``remote_transport`` for a remote cell; and a staleness
    update with ``staleness_reasons`` (and ``causality`` when known). Unset optional
    fields are dropped on the wire (see :func:`cell_status_payload`).
    """

    cell_id: str
    status: str
    remote_worker: str | None = None
    remote_transport: str | None = None
    # ``starting`` while a remote job is provisioned, ``running`` once it is, for a
    # worker that runs cells asynchronously.
    remote_build_state: str | None = None
    staleness_reasons: list[str] | None = None
    causality: dict[str, Any] | None = None


def cell_status_payload(
    cell_id: str,
    status: object,
    *,
    remote_worker: str | None = None,
    remote_transport: str | None = None,
    remote_build_state: str | None = None,
    staleness_reasons: list[str] | None = None,
    causality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated ``cell_status`` wire payload.

    *status* may be a ``CellStatus`` or a plain string. Absent optional fields are
    omitted (``exclude_none``).
    """
    return CellStatusPayload(
        cell_id=cell_id,
        status=str(status),
        remote_worker=remote_worker,
        remote_transport=remote_transport,
        remote_build_state=remote_build_state,
        staleness_reasons=staleness_reasons,
        causality=causality,
    ).model_dump(mode="json", exclude_none=True)


class CellConsolePayload(WsPayload):
    """``cell_console``: incremental stdout/stderr from a running cell.

    ``chunk_seq`` numbers a remote run's streamed chunks per stream from 0, so 0
    starts that stream's console for the run; None for console sent when the cell
    finishes.
    """

    cell_id: str
    stream: Literal["stdout", "stderr"]
    text: str
    chunk_seq: int | None = None


class CellOutputDeltaPayload(WsPayload):
    """``cell_output_delta``: streamed partial output (prompt cells).

    ``kind`` is ``"delta"`` (append ``text``), ``"retry"`` (validation failed: clear
    the buffer; ``attempt`` is the new attempt and ``text`` the first error) or
    ``"notice"`` (a provider-degradation note kept out of the accumulated content).
    """

    cell_id: str
    attempt: int
    kind: Literal["delta", "retry", "notice"]
    text: str


class CellIterationProgressPayload(WsPayload):
    """``cell_iteration_progress``: one completed iteration of a ``@loop`` cell."""

    cell_id: str
    iteration: int
    max_iter: int
    artifact_uri: str | None = None
    content_type: str | None = None
    until_reached: bool = False
    duration_ms: int


class CellVariantProgressPayload(WsPayload):
    """``cell_variant_progress``: one completed variant of a ``# @per_variant`` cell."""

    cell_id: str
    variant: str
    index: int
    total: int
    success: bool
    duration_ms: int
    error: str | None = None


class CascadePromptPayload(WsPayload):
    """``cascade_prompt``: upstream cells must run before the requested cell.

    The client confirms by sending ``cell_execute_cascade`` with the ``plan_id``.
    """

    cell_id: str
    plan_id: str
    cells_to_run: list[str]
    estimated_duration_ms: int


class CascadeProgressPayload(WsPayload):
    """``cascade_progress``: which cell of a confirmed cascade is now running."""

    plan_id: str
    current_cell_id: str
    completed: int
    total: int


class CellTestStatusPayload(WsPayload):
    """``cell_test_status``: cell unit-test run lifecycle (mirrors cell_status)."""

    cell_id: str
    status: Literal["running", "ready", "error"]


class CellTestResultsPayload(WsPayload):
    """``cell_test_results``: per-test outcomes and totals from a test run.

    The client-facing fields of ``CellTestResult`` plus the cell id and an emit-time
    ``stale`` flag. The internal staleness hashes are deliberately not sent.
    """

    cell_id: str
    passed: int
    failed: int
    errored: int
    skipped: int
    tests: list[CellTestCase]
    stale: bool
    pytest_unavailable: bool
    ran_at: int
    auto_installed: list[str] = Field(default_factory=list)


class EnvironmentJobModel(WsPayload):
    """One background environment operation, mirroring ``session.EnvironmentJobSnapshot``.

    Fields match the snapshot one-for-one; ``extra="forbid"`` makes a field added to
    the snapshot but not here fail loudly.
    """

    id: str
    action: str
    command: str
    status: str
    started_at: int
    package: str | None = None
    phase: str | None = None
    duration_ms: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    finished_at: int | None = None
    lockfile_changed: bool = False
    stale_cell_count: int = 0
    stale_cell_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class EnvironmentJobEventPayload(WsPayload):
    """``environment_job_started`` / ``environment_job_progress``: one job snapshot.

    ``environment_job_finished`` and ``dependency_changed`` carry heavier aggregate
    payloads and are not typed here.
    """

    environment_job: EnvironmentJobModel


def environment_job_event_payload(job: dict[str, Any]) -> dict[str, Any]:
    """Validate ``dataclasses.asdict(EnvironmentJobSnapshot)`` into the wire dict."""
    return EnvironmentJobEventPayload.model_validate({"environment_job": job}).model_dump(
        mode="json"
    )


class CascadeStepModel(WsPayload):
    """One cell in an upstream cascade (mirrors ``cascade.CascadeStep``)."""

    cell_id: str
    cell_name: str
    reason: str  # a CascadeReason value: stale | missing | target
    skip: bool = False
    estimated_ms: int = 0


class DownstreamImpactModel(WsPayload):
    """A downstream cell a run would invalidate (mirrors ``impact.DownstreamImpact``)."""

    cell_id: str
    cell_name: str
    current_status: str
    new_status: str = "stale:upstream"


class ImpactPreviewPayload(WsPayload):
    """``impact_preview``: upstream/downstream effects of running a cell.

    Mirrors ``impact.ImpactPreview``; ``downstream`` lists the cells that go stale.
    """

    target_cell_id: str
    upstream: list[CascadeStepModel] = Field(default_factory=list)
    downstream: list[DownstreamImpactModel] = Field(default_factory=list)
    estimated_ms: int = 0


def impact_preview_payload(impact: dict[str, Any]) -> dict[str, Any]:
    """Validate ``dataclasses.asdict(ImpactPreview)`` → the wire dict."""
    return ImpactPreviewPayload.model_validate(impact).model_dump(mode="json")


class CellProfileModel(WsPayload):
    """Per-cell profiling row in a ``profiling_summary``."""

    cell_id: str
    cell_name: str
    status: str
    duration_ms: int
    cache_hit: bool
    artifact_uri: str | None = None
    execution_count: int


class ProfilingSummaryPayload(WsPayload):
    """``profiling_summary``: notebook-level execution metrics."""

    total_execution_ms: int
    cache_hits: int
    cache_misses: int
    cache_savings_ms: int
    # The share of savings from someone else's machine, plus who contributed: the
    # total alone cannot say whether the shared store earns its keep.
    team_cache_savings_ms: int = 0
    team_cache_hits: int = 0
    team_contributors: list[str] = Field(default_factory=list)
    team_promotions: list[str] = Field(default_factory=list)
    total_artifact_bytes: int
    cell_profiles: list[CellProfileModel] = Field(default_factory=list)


def profiling_summary_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Validate ``session.get_profiling_summary()`` → the wire dict."""
    return ProfilingSummaryPayload.model_validate(summary).model_dump(mode="json")


class DagEdgeModel(WsPayload):
    """One DAG edge (mirrors ``NotebookDag.serialize_edges`` entries)."""

    from_cell_id: str
    to_cell_id: str
    variable: str


class ModuleExportModel(WsPayload):
    """A symbol a module cell exports (name + kind)."""

    name: str
    kind: str


class CellAnalysisModel(WsPayload):
    """Per-cell DAG analysis carried on ``dag_update``.

    ``is_module_cell`` / ``module_exports`` are present only for Python cells that
    export code symbols.
    """

    id: str
    defines: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    upstream_ids: list[str] = Field(default_factory=list)
    downstream_ids: list[str] = Field(default_factory=list)
    is_leaf: bool
    # Already ``model_dump``-ed AnnotationDiagnostic rows, passed through.
    annotation_diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    variant_group: str | None = None
    variant_name: str | None = None
    variant_active: bool | None = None
    is_module_cell: bool = False
    module_exports: list[ModuleExportModel] | None = None
    # Carried on the edit frame, else the browser shows the previous author until reload.
    created_by: str | None = None
    updated_by: str | None = None
    # The edited cell's new text, sent only to the session's other connections: a
    # second tab follows the edit, and the sender is never echoed text it has typed past.
    source: str | None = None


class DagUpdatePayload(WsPayload):
    """``dag_update``: the DAG changed.

    Carries edges, roots/leaves/topological order, per-cell analysis and active
    variant groups, so the client re-renders without a REST round-trip.
    """

    edges: list[DagEdgeModel] = Field(default_factory=list)
    roots: list[str] = Field(default_factory=list)
    leaves: list[str] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)
    cells: list[CellAnalysisModel] = Field(default_factory=list)
    # Already ``model_dump``-ed VariantGroup rows, passed through.
    variant_groups: list[dict[str, Any]] = Field(default_factory=list)


def dag_update_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate an assembled ``dag_update`` dict → the wire dict."""
    return DagUpdatePayload.model_validate(raw).model_dump(mode="json")


class PresenceEntryModel(WsPayload):
    """One identity on a session and the cell it is on."""

    principal: str
    focused_cell_id: str | None = None
    # Wall-clock seconds since the epoch when this entry last changed.
    since: float


class PresencePayload(WsPayload):
    """``presence``: who is on the session, sent on join, leave and focus change.

    One entry per identity (a principal, or a declared author where nobody is
    authenticated), not per socket. ``you`` is the receiving connection's identity.
    """

    principals: list[PresenceEntryModel]
    you: str


SessionClosedReason = Literal["idle", "session_limit", "memory", "closed", "deleted", "shutdown"]

_SESSION_CLOSED_MESSAGES: dict[SessionClosedReason, str] = {
    "idle": "This notebook was closed after a period without activity.",
    "session_limit": (
        "This notebook was closed because the server reached its limit of open notebooks."
    ),
    "memory": "This notebook was closed because the server was low on memory.",
    "closed": "This notebook session was closed.",
    "deleted": "This notebook was deleted.",
    "shutdown": "The server is shutting down.",
}


class SessionClosedPayload(WsPayload):
    """``session_closed``: the server closed this session; the socket closes next.

    ``reason`` is ``idle`` (nobody edited, ran or focused for the session timeout),
    ``session_limit``, ``memory`` (available memory fell below the configured floor),
    ``closed`` (the close route), ``deleted`` or ``shutdown`` (the server is stopping,
    and sessions do not survive a restart). Nothing computed is lost.
    """

    reason: SessionClosedReason
    message: str


def session_closed_payload(reason: SessionClosedReason) -> dict[str, Any]:
    """Build the wire dict for a ``session_closed`` frame."""
    return SessionClosedPayload(reason=reason, message=_SESSION_CLOSED_MESSAGES[reason]).model_dump(
        mode="json"
    )


# Typed so a mistyped code at an emit site is a type error, not a silent frontend miss.
ErrorCode = Literal[
    "ENVIRONMENT_BUSY",
    "notebook_busy",
    "cell_busy",
    "cell_locked",
    "read_only",
    "insufficient_scope",
]


class ErrorPayload(WsPayload):
    """``error``: a request could not be served.

    ``code`` is part of the contract: the frontend branches on ``ENVIRONMENT_BUSY``.
    Known codes: ``ENVIRONMENT_BUSY`` (an environment job holds the notebook),
    ``notebook_busy`` (run refused while another run holds the notebook),
    ``cell_busy`` (edit refused while the cell runs), ``cell_locked`` (someone else
    just changed the cell), ``read_only`` (not allowed in app view),
    ``insufficient_scope`` (auth).
    """

    error: str
    code: ErrorCode | None = None
    # On ``cell_busy`` and ``cell_locked``, the cell that refused the edit; on
    # ``notebook_busy``, the cell whose run was refused (absent for run-all).
    cell_id: str | None = None
    # Only on ``cell_locked``: who changed the cell.
    held_by: str | None = None


def error_payload(
    error: str,
    code: ErrorCode | None = None,
    cell_id: str | None = None,
    held_by: str | None = None,
) -> dict[str, Any]:
    """Build the wire dict for an ``error`` frame, omitting unset fields.

    A plain error stays ``{"error": ...}`` rather than gaining null keys.
    """
    return ErrorPayload(error=error, code=code, cell_id=cell_id, held_by=held_by).model_dump(
        mode="json", exclude_none=True
    )


# Which model describes which frame's payload. Frames absent here still send
# hand-built dicts and a client gets ``unknown`` for them. Generated TypeScript
# derives from this map; a drift test asserts every payload model is registered.
FRAME_PAYLOADS: dict[MessageType, type[WsPayload]] = {
    MessageType.CASCADE_PROGRESS: CascadeProgressPayload,
    MessageType.CASCADE_PROMPT: CascadePromptPayload,
    MessageType.CELL_CONSOLE: CellConsolePayload,
    MessageType.CELL_ITERATION_PROGRESS: CellIterationProgressPayload,
    MessageType.CELL_OUTPUT_DELTA: CellOutputDeltaPayload,
    MessageType.CELL_STATUS: CellStatusPayload,
    MessageType.CELL_TEST_RESULTS: CellTestResultsPayload,
    MessageType.CELL_TEST_STATUS: CellTestStatusPayload,
    MessageType.CELL_VARIANT_PROGRESS: CellVariantProgressPayload,
    MessageType.DAG_UPDATE: DagUpdatePayload,
    MessageType.ENVIRONMENT_JOB_PROGRESS: EnvironmentJobEventPayload,
    MessageType.ENVIRONMENT_JOB_STARTED: EnvironmentJobEventPayload,
    MessageType.ERROR: ErrorPayload,
    MessageType.IMPACT_PREVIEW: ImpactPreviewPayload,
    MessageType.PRESENCE: PresencePayload,
    MessageType.PROFILING_SUMMARY: ProfilingSummaryPayload,
    MessageType.SESSION_CLOSED: SessionClosedPayload,
}

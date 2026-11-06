"""Typed payload models for notebook WebSocket frames.

Built and validated at the emit site in ``ws.py`` and serialized with
``.model_dump(mode="json")``; executor-originated dicts are validated as they
cross into the protocol layer. ``extra="forbid"`` turns an unmodeled field into
a construction error instead of silent protocol drift.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strata.notebook.annotations import AnnotationsWirePayload
from strata.notebook.immutability import MutationWarning
from strata.notebook.models import (
    CellState,
    CellTestCase,
    CellTestResult,
    InputWeight,
    NotebookState,
)
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
    # What a dispatcher says about a job: while ``starting``, its place in line and a
    # sentence for the badge ("Behind 2 cells for a slot on team-a100."); while ``running``,
    # only the sentence ("Runs on team-cpu, slot 1.").
    remote_queue_position: int | None = None
    remote_queue_detail: str | None = None
    staleness_reasons: list[str] | None = None
    causality: dict[str, Any] | None = None


def cell_status_payload(
    cell_id: str,
    status: object,
    *,
    remote_worker: str | None = None,
    remote_transport: str | None = None,
    remote_build_state: str | None = None,
    remote_queue_position: int | None = None,
    remote_queue_detail: str | None = None,
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
        remote_queue_position=remote_queue_position,
        remote_queue_detail=remote_queue_detail,
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


class CellInputsPayload(WsPayload):
    """``cell_inputs``: what a cell's declared inputs weigh, for the badge.

    Sent once a run has resolved its inputs, before it starts. ``machine`` is the worker
    the cell runs on (``local`` for this server).
    """

    cell_id: str
    machine: str
    inputs: list[InputWeight]


class ReadConfirmCell(WsPayload):
    """One cell of a ``read_confirm_prompt``: its inputs on *machine* and what they'd move.

    ``bytes_to_move`` is what the machine does not hold yet, of the sizes that are known;
    ``cost`` is that at the host's rates, in dollars.
    """

    cell_id: str
    machine: str
    inputs: list[InputWeight]
    bytes_to_move: int
    cost: float


class ReadConfirmPromptPayload(WsPayload):
    """``read_confirm_prompt``: cells of a run would read more than the host allows unasked.

    Nothing started. The client runs it by sending ``read_confirm`` with the ``prompt_id``;
    each yes is kept for the cell's machine and inputs. ``message`` is the host's sentence.
    """

    prompt_id: str
    cells: list[ReadConfirmCell]
    message: str | None = None


def cell_inputs_payload(cell_id: str, machine: str, weights: list[InputWeight]) -> dict[str, Any]:
    """Build a validated ``cell_inputs`` wire payload."""
    return CellInputsPayload(cell_id=cell_id, machine=machine, inputs=weights).model_dump(
        exclude_none=True
    )


class CascadePromptPayload(WsPayload):
    """``cascade_prompt``: upstream cells must run before the requested cell.

    ``cells_to_run`` lists, in run order, the cells that will run (ready upstreams are
    left out). The client confirms by sending ``cell_execute_cascade`` with the ``plan_id``.
    """

    cell_id: str
    plan_id: str
    cells_to_run: list[str]
    estimated_duration_ms: int


class CascadeProgressPayload(WsPayload):
    """``cascade_progress``: which cell of a confirmed cascade is now running.

    ``completed`` counts the cells finished of the ``total`` that run; a cascade that
    succeeds ends with a frame where they are equal.
    """

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

    ``environment_job_finished`` carries more: :class:`EnvironmentJobFinishedPayload`.
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
    duration_ms: int | None = None
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


class InspectResultPayload(WsPayload):
    """``inspect_result``: the answer to ``inspect_open``, ``inspect_eval`` or ``inspect_close``.

    ``open`` and ``close`` carry ``ok`` and a ``result`` message. ``eval`` echoes
    ``expr`` and adds the REPL's answer: ``result`` and ``type`` (and ``stdout`` when
    the expression printed) on success, ``error`` on failure. Unset fields are dropped.
    """

    cell_id: str
    action: Literal["open", "eval", "close"]
    ok: bool
    result: str | None = None
    type: str | None = None
    stdout: str | None = None
    error: str | None = None
    expr: str | None = None


def inspect_result_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate an ``inspect_result`` dict, including the REPL's own answer, for the wire."""
    return InspectResultPayload.model_validate(raw).model_dump(mode="json", exclude_none=True)


class AgentNotePayload(WsPayload):
    """``agent_note``: one line for a terminal viewer's Agent tab.

    ``source`` is ``"mcp"`` for a narrated tool action and ``"agent"`` for a note the
    agent wrote itself.
    """

    source: Literal["mcp", "agent"]
    text: str


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
class OutputMetadata(WsPayload):
    """One variable a run produced, as its harness described it.

    The harness may be a worker on another Strata version, so a field this one does not
    know is kept, not refused.
    """

    model_config = ConfigDict(extra="allow")

    content_type: str
    file: str | None = None
    bytes: int | None = None
    rows: int | None = None
    columns: list[str] | None = None
    preview: Any = None
    artifact_uri: str | None = None
    error: str | None = None
    markdown_text: str | None = None
    inline_data_url: str | None = None
    width: int | None = None
    height: int | None = None
    # Pickled and cell-defined objects: the codec and the value's type name.
    codec: str | None = None
    type: str | None = None


class DisplayOutput(WsPayload):
    """One thing a cell showed: a stored display, or one its harness just reported.

    Unknown fields are kept, as on :class:`OutputMetadata`.
    """

    model_config = ConfigDict(extra="allow")

    content_type: str
    file: str | None = None
    bytes: int | None = None
    rows: int | None = None
    columns: list[str] | None = None
    preview: Any = None
    # A prompt cell's display text.
    scalar: Any = None
    artifact_uri: str | None = None
    inline_data_url: str | None = None
    markdown_text: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None
    computed_at: float | None = None
    variant: str | None = None


class CellOutputPayload(WsPayload):
    """``cell_output``: a cell ran, or was a cache hit.

    Serialized with ``exclude_unset``, so a field the server did not send stays off
    the wire (see :func:`cell_result_payload`).
    """

    cell_id: str
    # The run's whole console, which replaces the client's.
    stdout: str
    stderr: str
    outputs: dict[str, OutputMetadata]
    cache_hit: bool
    duration_ms: int
    artifact_uri: str | None
    execution_method: str | None
    mutation_warnings: list[MutationWarning]
    displays: list[DisplayOutput] | None = None
    display: DisplayOutput | None = None
    # Every stored variable's artifact, each of which may be promoted.
    artifact_uris: dict[str, str] | None = None
    # What a hit saved, priced against the run it stands for.
    saved_ms: int | None = None
    remote_worker: str | None = None
    remote_transport: str | None = None
    remote_build_id: str | None = None
    remote_build_state: str | None = None
    remote_error_code: str | None = None


class CellErrorPayload(WsPayload):
    """``cell_error``: a cell failed, or could not be run.

    A failure the server raised itself carries only ``cell_id`` and ``error``.
    """

    cell_id: str
    error: str | None
    stdout: str | None = None
    stderr: str | None = None
    traceback: str | None = None
    # With the harness frames that lead to the cell, which ``traceback`` leaves out.
    traceback_full: str | None = None
    error_code: str | None = None
    suggest_install: str | None = None
    # ``python`` or ``r``: which install endpoint the suggestion is for.
    suggest_install_language: str | None = None
    remote_worker: str | None = None
    remote_transport: str | None = None
    remote_build_id: str | None = None
    remote_build_state: str | None = None
    remote_error_code: str | None = None


def cell_result_payload(success: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``cell_output`` (*success*) or ``cell_error`` payload for the wire.

    Only the keys *payload* holds are sent, as they were before the frame was typed.
    """
    model = CellOutputPayload if success else CellErrorPayload
    return model.model_validate(payload).model_dump(mode="json", exclude_unset=True)


def cell_error_payload(cell_id: str, error: str) -> dict[str, Any]:
    """Build the ``cell_error`` payload for a failure the server raised itself."""
    return cell_result_payload(False, {"cell_id": cell_id, "error": error})


class LastGotModel(WsPayload):
    """How a cell's last result came: a hit and what it saved, or a run and what it took."""

    cache_hit: bool
    ms: int


class WidgetDescriptorModel(WsPayload):
    """One control a widget cell declares."""

    name: str
    kind: str
    params: dict[str, Any]
    default: Any = None


class WidgetModel(WsPayload):
    """A widget cell's controls and their current values."""

    descriptors: list[WidgetDescriptorModel]
    values: dict[str, Any]


class CellTestResultView(CellTestResult, WsPayload):
    """A cell's last test run, and whether the cell, its tests or inputs changed since."""

    stale: bool | None = None


class CellView(CellState, WsPayload):
    """A cell as clients see it: ``NotebookSession.serialize_cell``.

    The stored cell, plus what is derived from its source (annotations, module exports,
    a widget's controls) and what the session knows (displays with their data, why it
    is stale, how its last result came).
    """

    # Narrowed to what the session sends; pydantic validates a subclass's fields as declared.
    test_result: CellTestResultView | None = None  # ty: ignore[invalid-mutable-override]
    display_outputs: list[DisplayOutput] = Field(default_factory=list)  # ty: ignore[invalid-attribute-override]
    display_output: DisplayOutput | None = None  # ty: ignore[invalid-attribute-override]
    staleness_reasons: list[str]
    annotations: AnnotationsWirePayload
    widget: WidgetModel | None = None
    is_module_cell: bool | None = None
    module_exports: list[ModuleExportModel] | None = None
    causality: dict[str, Any] | None = None
    # Names this cell defines that an earlier cell defined too.
    shadow_warnings: list[str] | None = None
    last_got: LastGotModel | None = None
    # What the last run took, or the last hit loaded in.
    last_duration_ms: int | None = None


class EnvironmentStateModel(WsPayload):
    """The notebook's Python environment: ``NotebookSession.serialize_environment_state``."""

    requested_python_version: str
    runtime_python_version: str
    python_version: str
    lockfile_hash: str
    package_count: int
    declared_package_count: int
    resolved_package_count: int
    sync_state: str
    sync_error: str | None
    sync_notice: str | None
    last_synced_at: int | None
    last_sync_duration_ms: int | None
    has_lockfile: bool
    venv_python: str | None
    interpreter_source: str


class RPackageModel(WsPayload):
    """One package in the notebook's renv library."""

    name: str
    version: str


class REnvironmentStateModel(WsPayload):
    """The notebook's R environment: ``NotebookSession.serialize_r_environment_state``."""

    has_lockfile: bool
    current_lock_hash: str
    lock_hash: str
    r_version: str
    system_r_version: str | None
    last_synced_at: int
    sync_state: str
    sync_error: str | None
    packages: list[RPackageModel]
    packages_status: str
    packages_error: str | None


class NotebookDagModel(WsPayload):
    """The DAG a ``notebook_state`` carries."""

    edges: list[DagEdgeModel]
    roots: list[str]
    leaves: list[str]
    topological_order: list[str]
    # Already ``model_dump``-ed VariantGroup rows; sent only after a variant is added.
    variant_groups: list[dict[str, Any]] | None = None


class NotebookStatePayload(NotebookState, WsPayload):
    """``notebook_state``: the whole notebook, on reconnect or a change made elsewhere.

    ``NotebookSession.serialize_notebook_state`` plus the DAG. Serialized with
    ``exclude_unset`` in Python mode, so the wire is what it was before it was typed.
    """

    cells: list[CellView] = Field(default_factory=list)  # ty: ignore[invalid-attribute-override]
    environment: EnvironmentStateModel
    environment_job: EnvironmentJobModel | None
    environment_job_history: list[EnvironmentJobModel]
    r_environment: REnvironmentStateModel
    dag: NotebookDagModel | None = None


class DependencyModel(WsPayload):
    """A Python dependency, declared or resolved."""

    name: str
    version: str | None
    specifier: str | None


class EnvironmentJobFinishedPayload(WsPayload):
    """``environment_job_finished``: an environment job ended.

    The cells come with it, since a job can make them stale. A completed job adds the
    environment it left; an import adds its warnings and count.
    """

    environment_job: EnvironmentJobModel
    environment_job_history: list[EnvironmentJobModel]
    cells: list[CellView]
    lockfile_changed: bool
    stale_cell_count: int
    stale_cell_ids: list[str]
    warnings: list[str] | None = None
    imported_count: int | None = None
    environment: EnvironmentStateModel | None = None
    r_environment: REnvironmentStateModel | None = None
    dependencies: list[DependencyModel] | None = None
    resolved_dependencies: list[DependencyModel] | None = None


def notebook_state_payload(state: dict[str, Any]) -> dict[str, Any]:
    """Validate a ``notebook_state`` payload for the wire, keeping its keys and values."""
    return NotebookStatePayload.model_validate(state).model_dump(exclude_unset=True)


def environment_job_finished_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an ``environment_job_finished`` payload for the wire."""
    return EnvironmentJobFinishedPayload.model_validate(payload).model_dump(exclude_unset=True)


FRAME_PAYLOADS: dict[MessageType, type[WsPayload]] = {
    MessageType.AGENT_NOTE: AgentNotePayload,
    MessageType.CASCADE_PROGRESS: CascadeProgressPayload,
    MessageType.CASCADE_PROMPT: CascadePromptPayload,
    MessageType.CELL_CONSOLE: CellConsolePayload,
    MessageType.CELL_ERROR: CellErrorPayload,
    MessageType.CELL_INPUTS: CellInputsPayload,
    MessageType.CELL_ITERATION_PROGRESS: CellIterationProgressPayload,
    MessageType.CELL_OUTPUT: CellOutputPayload,
    MessageType.CELL_OUTPUT_DELTA: CellOutputDeltaPayload,
    MessageType.CELL_STATUS: CellStatusPayload,
    MessageType.CELL_TEST_RESULTS: CellTestResultsPayload,
    MessageType.CELL_TEST_STATUS: CellTestStatusPayload,
    MessageType.CELL_VARIANT_PROGRESS: CellVariantProgressPayload,
    MessageType.DAG_UPDATE: DagUpdatePayload,
    MessageType.ENVIRONMENT_JOB_FINISHED: EnvironmentJobFinishedPayload,
    MessageType.ENVIRONMENT_JOB_PROGRESS: EnvironmentJobEventPayload,
    MessageType.ENVIRONMENT_JOB_STARTED: EnvironmentJobEventPayload,
    MessageType.ERROR: ErrorPayload,
    MessageType.IMPACT_PREVIEW: ImpactPreviewPayload,
    MessageType.INSPECT_RESULT: InspectResultPayload,
    MessageType.NOTEBOOK_STATE: NotebookStatePayload,
    MessageType.PRESENCE: PresencePayload,
    MessageType.PROFILING_SUMMARY: ProfilingSummaryPayload,
    MessageType.READ_CONFIRM_PROMPT: ReadConfirmPromptPayload,
    MessageType.SESSION_CLOSED: SessionClosedPayload,
}

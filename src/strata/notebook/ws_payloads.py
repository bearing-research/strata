"""Typed payload models for notebook WebSocket frames (#44).

WS frame payloads were inline dicts assembled at the emit site, with the shape
documented only by the Vue client's TypeScript types and whatever test happened
to assert a field. This module promotes them to ``pydantic`` models so the
protocol is self-describing and a second client (the TUI, #37) can share one
source of truth instead of re-deriving each shape.

Models are constructed and validated at the **emit site in ``ws.py``** — the
protocol boundary — and serialized with ``.model_dump(mode="json")``. Payloads
that originate in the executor (loop-iteration progress, prompt-cell deltas) are
validated here as they cross into the protocol layer, so the executor keeps
emitting plain dicts and stays decoupled from the wire contract.

``extra="forbid"`` makes an unmodeled field a loud construction error rather than
a silently-shipped one — the point of typing the protocol is to catch drift.

This is the **incremental first phase** (#44 is explicitly phase-by-phase): the
during/after-execution streaming + test frames, which have small, stable shapes.
Later phases cover ``cell_status`` / ``cell_output`` / ``cell_error`` (the big
trio with many optional fields), the cascade / dag / ``notebook_state`` frames,
the agent and environment-job frames, and the client→server frames.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from strata.notebook.models import CellTestCase


class WsPayload(BaseModel):
    """Base for typed WS frame payloads.

    ``extra="forbid"`` rejects any field the emit site adds but the model
    doesn't declare, turning protocol drift into an immediate error at the
    boundary instead of an undocumented field on the wire.
    """

    model_config = ConfigDict(extra="forbid")


class CellStatusPayload(WsPayload):
    """``cell_status`` — a cell's execution status changed.

    The most-emitted notebook frame, with three shapes that share one model:

    - a bare status change (``cell_id`` + ``status``);
    - a *running* broadcast that, for a remote cell, adds ``remote_worker`` +
      ``remote_transport`` so the UI can show a "dispatching → X" badge;
    - a staleness update that adds ``staleness_reasons`` (and ``causality`` when
      the backend tracked why).

    The optional fields default to ``None`` and are dropped on the wire via
    ``exclude_none=True`` (see :func:`cell_status_payload`), so each emit site
    keeps its exact historical shape. ``status`` is a ``CellStatus`` value
    (``idle`` / ``running`` / ``ready`` / ``error`` / ``stale``).
    """

    cell_id: str
    status: str
    remote_worker: str | None = None
    remote_transport: str | None = None
    staleness_reasons: list[str] | None = None
    causality: dict[str, Any] | None = None


def cell_status_payload(
    cell_id: str,
    status: object,
    *,
    remote_worker: str | None = None,
    remote_transport: str | None = None,
    staleness_reasons: list[str] | None = None,
    causality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated ``cell_status`` wire payload.

    Shared by every emit site (in ``ws.py`` and ``session.py``) so the frame
    has one construction point. ``status`` accepts a ``CellStatus`` enum or a
    plain string and is coerced to the enum's string value. Absent optional
    fields are omitted (``exclude_none``), preserving each site's exact shape.
    """
    return CellStatusPayload(
        cell_id=cell_id,
        status=str(status),
        remote_worker=remote_worker,
        remote_transport=remote_transport,
        staleness_reasons=staleness_reasons,
        causality=causality,
    ).model_dump(mode="json", exclude_none=True)


class CellConsolePayload(WsPayload):
    """``cell_console`` — incremental stdout/stderr from a running cell."""

    cell_id: str
    stream: Literal["stdout", "stderr"]
    text: str


class CellOutputDeltaPayload(WsPayload):
    """``cell_output_delta`` — streamed partial output (prompt cells today).

    ``kind`` is ``"delta"`` (append ``text`` to the per-cell buffer), ``"retry"``
    (schema validation failed — clear the buffer, ``attempt`` is the new attempt,
    ``text`` is the first validator error), or ``"notice"`` (provider-degradation
    announcement shown on the stream without polluting accumulated content).
    """

    cell_id: str
    attempt: int
    kind: Literal["delta", "retry", "notice"]
    text: str


class CellIterationProgressPayload(WsPayload):
    """``cell_iteration_progress`` — one completed iteration of a ``@loop`` cell."""

    cell_id: str
    iteration: int
    max_iter: int
    artifact_uri: str | None = None
    content_type: str | None = None
    until_reached: bool = False
    duration_ms: int


class CascadePromptPayload(WsPayload):
    """``cascade_prompt`` — upstream cells must run before the requested cell.

    Sent when a cell's upstreams are stale/idle; the client confirms by sending
    ``cell_execute_cascade`` with the ``plan_id``.
    """

    cell_id: str
    plan_id: str
    cells_to_run: list[str]
    estimated_duration_ms: int


class CascadeProgressPayload(WsPayload):
    """``cascade_progress`` — which cell of a confirmed cascade is now running."""

    plan_id: str
    current_cell_id: str
    completed: int
    total: int


class CellTestStatusPayload(WsPayload):
    """``cell_test_status`` — cell unit-test run lifecycle (mirrors cell_status)."""

    cell_id: str
    status: Literal["running", "ready", "error"]


class CellTestResultsPayload(WsPayload):
    """``cell_test_results`` — per-test outcomes + totals from a test run.

    A flat mirror of the client-facing fields of ``CellTestResult`` plus the
    cell id and the ``stale`` flag computed at emit time. The internal staleness
    hashes (``cell_source_hash`` / ``test_source_hash`` / ``input_fingerprint``)
    are deliberately *not* on the wire — they were only ever an incidental
    ``**model_dump()`` leak; the client never read them.
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

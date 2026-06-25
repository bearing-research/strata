"""View model for the notebook TUI — the pure, UI-free state core.

The TUI is a read-only spectator: it seeds state from a ``notebook_state``
snapshot and then folds live WS frames into a per-cell view. Keeping this
logic free of Textual / sockets makes it directly unit-testable with a
fake-frame feed (the WS portal can't be driven from ``TestClient`` on
py3.12/macOS — see project memory), and keeps the app a thin renderer.

Frame shapes are parsed as plain dicts per ``docs/reference/notebook-protocol.md``
(typed payloads, #44, are optional and not required here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from strata.notebook.annotations import parse_annotations


@dataclass
class CellView:
    """The spectator's view of a single cell."""

    id: str
    name: str = ""
    language: str = "python"
    source: str = ""
    status: str = "idle"
    # Hydrated display outputs from the snapshot (markdown_text / inline_data_url
    # / preview), rendered in the output panel.
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    # Variable outputs from a live ``cell_output`` frame (name + preview).
    outputs: list[dict[str, Any]] = field(default_factory=list)
    # Streamed prompt-cell text (``cell_output_delta``).
    stream_text: str = ""
    # Accumulated stdout/stderr (``cell_console``).
    console: str = ""
    error: str | None = None
    # Loop-cell progress, e.g. "iter 3/10" (``cell_iteration_progress``).
    iteration: str = ""
    # Last run timing from a ``cell_output`` frame.
    duration_ms: int | None = None
    cache_hit: bool = False
    # Cell-unit-test outcome badge from ``cell_test_*`` frames, e.g. "✓ 4/4",
    # "✗ 2/4", or "tests…" while running. The other 0.4.0 headline — surfaced so
    # the spectator sees the driver run a cell's tests.
    test_summary: str = ""
    # Per-test outcomes (name / outcome / message) from the last cell_test_results
    # frame, rendered in the Tests tab (the "pytest window").
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    test_unavailable: bool = False


class NotebookViewModel:
    """Folds a ``notebook_state`` snapshot + live frames into per-cell views."""

    def __init__(self) -> None:
        self.notebook_name: str = ""
        self.cell_order: list[str] = []
        self.cells: dict[str, CellView] = {}
        # DAG edges as (from_cell_id, to_cell_id), from the notebook_state `dag`
        # block (notebook_sync includes it) and refreshed by dag_update frames.
        self.edges: list[tuple[str, str]] = []
        # Notebook-level activity line (cascade / environment job / agent), shown
        # in the header. Updated by cascade_* / environment_job_* / agent_* frames.
        self.banner: str = ""
        # Agent activity feed (chronological): streamed reasoning interleaved with
        # tool/step events, confirm prompts, and completion. The headline of the
        # spectator — watch an agent drive the notebook.
        self.agent_feed: list[str] = []
        self.agent_status: str = ""
        self._agent_streaming = False  # last feed entry is a growing text block

    # -- snapshot ------------------------------------------------------------

    def apply_notebook_state(self, payload: dict[str, Any]) -> None:
        """Seed (or re-seed) all cells from a ``notebook_state`` snapshot.

        Live-only fields a snapshot doesn't carry (console buffer, streamed
        deltas, the last live ``cell_output``) are preserved across a manual
        resync for cells that still exist, so re-syncing doesn't wipe what the
        spectator already saw.
        """
        self.notebook_name = str(payload.get("name") or "")
        raw_cells = payload.get("cells") or []

        order: list[str] = []
        new_cells: dict[str, CellView] = {}
        for raw in raw_cells:
            if not isinstance(raw, dict):
                continue
            cid = raw.get("id")
            if not isinstance(cid, str):
                continue
            order.append(cid)
            prior = self.cells.get(cid)
            source = str(raw.get("source") or "")
            new_cells[cid] = CellView(
                id=cid,
                # The display name is the ``# @name`` annotation (which always
                # wins), falling back to the persisted name — same precedence the
                # web UI and export use. The serialized ``name`` field is the
                # persisted notebook.toml name, which the annotation overrides.
                name=parse_annotations(source).name or str(raw.get("name") or ""),
                language=str(raw.get("language") or "python"),
                source=source,
                status=str(raw.get("status") or "idle"),
                display_outputs=_snapshot_display_outputs(raw),
                outputs=prior.outputs if prior else [],
                stream_text=prior.stream_text if prior else "",
                console=prior.console if prior else "",
                error=prior.error if prior else None,
                duration_ms=prior.duration_ms if prior else None,
                cache_hit=prior.cache_hit if prior else False,
            )

        self.cell_order = order
        self.cells = new_cells
        self.edges = _parse_edges(payload.get("dag"))

    # -- incremental frames --------------------------------------------------

    def apply_frame(self, msg_type: str, payload: dict[str, Any]) -> set[str]:
        """Fold one live frame in; return the set of affected cell ids.

        Unknown frame types are no-ops here and return an empty set.
        ``notebook_state`` is handled by :meth:`apply_notebook_state`, not here.

        ``impact_preview`` / ``profiling_summary`` / ``inspect_result`` are
        intentionally *not* handled: the server sends them point-to-point to the
        client that requested them (not via ``_broadcast_message``), so a
        read-only spectator never receives them — there is nothing to surface.
        """
        if msg_type == "dag_update":
            self.edges = _parse_edges(payload)
            return set(self.cell_order)  # whole-graph change

        # Notebook-level activity (not tied to one cell) → header banner.
        if msg_type in ("cascade_prompt", "cascade_progress"):
            self.banner = _cascade_banner(msg_type, payload)
            return set()
        if msg_type in (
            "environment_job_started",
            "environment_job_progress",
            "environment_job_finished",
        ):
            self.banner = _env_banner(payload)
            return set()
        if msg_type in (
            "agent_text_delta",
            "agent_progress",
            "agent_confirm_request",
            "agent_done",
        ):
            self._apply_agent_frame(msg_type, payload)
            return set()

        cid = payload.get("cell_id")
        if not isinstance(cid, str):
            return set()
        cell = self.cells.get(cid)
        if cell is None:
            return set()

        if msg_type == "cell_status":
            cell.status = str(payload.get("status") or cell.status)
        elif msg_type == "cell_console":
            cell.console += str(payload.get("text") or "")
        elif msg_type == "cell_output":
            outputs = payload.get("outputs")
            cell.outputs = outputs if isinstance(outputs, list) else []
            cell.error = None
            duration = payload.get("duration_ms")
            cell.duration_ms = int(duration) if isinstance(duration, (int, float)) else None
            cell.cache_hit = bool(payload.get("cache_hit"))
        elif msg_type == "cell_output_delta":
            if payload.get("kind") == "retry":
                cell.stream_text = ""
            cell.stream_text += str(payload.get("text") or "")
        elif msg_type == "cell_error":
            cell.error = str(payload.get("error") or "error")
        elif msg_type == "cell_iteration_progress":
            iteration = payload.get("iteration")
            max_iter = payload.get("max_iter")
            cell.iteration = f"iter {iteration}/{max_iter}" if iteration and max_iter else ""
        elif msg_type == "cell_test_status":
            # "running" sets a pending badge; the final ready/error state's real
            # counts arrive in the cell_test_results frame, so keep that badge.
            if str(payload.get("status") or "") == "running":
                cell.test_summary = "tests…"
                self.banner = f"🧪 {cell.name or cid}: running tests"
        elif msg_type == "cell_test_results":
            cell.test_summary = _test_badge(payload)
            cases = payload.get("tests")
            cell.test_cases = (
                [c for c in cases if isinstance(c, dict)] if isinstance(cases, list) else []
            )
            cell.test_unavailable = bool(payload.get("pytest_unavailable"))
            self.banner = f"🧪 {cell.name or cid} tests: {cell.test_summary}"
        else:
            return set()
        return {cid}

    def _apply_agent_frame(self, msg_type: str, payload: dict[str, Any]) -> None:
        """Fold an ``agent_*`` frame into the chronological agent feed + status.

        Read-only: ``agent_confirm_request`` is shown as "awaiting driver" (the
        driver/Vue answers; the spectator never sends ``agent_confirm_response``).
        """
        if msg_type == "agent_text_delta":
            text = str(payload.get("text") or "")
            if self._agent_streaming and self.agent_feed:
                self.agent_feed[-1] += text  # extend the streaming reasoning block
            else:
                self.agent_feed.append(text)
                self._agent_streaming = True
            self.agent_status = "thinking"
            self.banner = "🤖 agent: thinking"
            return

        self._agent_streaming = False
        if msg_type == "agent_progress":
            event = str(payload.get("event") or "step")
            detail = str(payload.get("detail") or "")
            self.agent_feed.append(f"• {event}: {detail}".rstrip(": "))
            self.agent_status = "running"
            self.banner = f"🤖 agent: {event}"
        elif msg_type == "agent_confirm_request":
            self.agent_feed.append(f"⚠ awaiting driver confirmation: {_confirm_desc(payload)}")
            self.agent_status = "awaiting confirm"
            self.banner = "🤖 agent: awaiting driver confirmation"
        elif msg_type == "agent_done":
            self.agent_feed.append(_agent_done_line(payload))
            self.agent_status = "done"
            self.banner = "🤖 agent: done"


def _test_badge(payload: dict[str, Any]) -> str:
    """Compact cell-test outcome: "✓ 4/4", "✗ 2/4", "⚠ pytest n/a" (+ " ·stale")."""
    if payload.get("pytest_unavailable"):
        return "⚠ pytest n/a"

    def _count(key: str) -> int:
        value = payload.get(key)
        return value if isinstance(value, int) else 0

    passed = _count("passed")
    total = passed + _count("failed") + _count("errored") + _count("skipped")
    glyph = "✓" if (_count("failed") == 0 and _count("errored") == 0) else "✗"
    badge = f"{glyph} {passed}/{total}"
    return f"{badge} ·stale" if payload.get("stale") else badge


def _confirm_desc(payload: dict[str, Any]) -> str:
    for key in ("description", "message", "summary", "tool", "tool_name", "action"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return "(action)"


def _agent_done_line(payload: dict[str, Any]) -> str:
    model = str(payload.get("model") or "")
    tokens = payload.get("tokens")
    tok = ""
    if isinstance(tokens, dict):
        tok = f", {tokens.get('input', 0)}+{tokens.get('output', 0)} tok"
    return f"✓ agent done ({model}{tok})".replace("()", "").rstrip()


def _cascade_banner(msg_type: str, payload: dict[str, Any]) -> str:
    """One-line cascade status for the header (running upstreams before a cell)."""
    if msg_type == "cascade_prompt":
        n = len(payload.get("cells_to_run") or [])
        return f"⟳ cascade: {n} upstream cell(s) to run"
    completed = payload.get("completed")
    total = payload.get("total")
    current = payload.get("current_cell_id") or ""
    head = f"⟳ cascade {completed}/{total}" if total else "⟳ cascade"
    return f"{head} · {current}".rstrip(" ·")


def _env_banner(payload: dict[str, Any]) -> str:
    """One-line environment-job status (uv add / sync / import …)."""
    job = payload.get("environment_job")
    if not isinstance(job, dict):
        return ""
    action = str(job.get("action") or "env")
    package = str(job.get("package") or "")
    status = str(job.get("status") or "")
    phase = str(job.get("phase") or "")
    label = f"⚙ {action} {package}".rstrip()
    tail = " · ".join(p for p in (status, phase) if p)
    return f"{label}: {tail}" if tail else label


def _parse_edges(dag: Any) -> list[tuple[str, str]]:
    """Extract (from_cell_id, to_cell_id) pairs from a dag/dag_update payload."""
    if not isinstance(dag, dict):
        return []
    raw_edges = dag.get("edges")
    if not isinstance(raw_edges, list):
        return []
    edges: list[tuple[str, str]] = []
    for edge in raw_edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get("from_cell_id")
        dst = edge.get("to_cell_id")
        if isinstance(src, str) and isinstance(dst, str):
            edges.append((src, dst))
    return edges


def _snapshot_display_outputs(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a serialized cell's display outputs to a list of dicts."""
    outputs = raw.get("display_outputs")
    if isinstance(outputs, list):
        return [o for o in outputs if isinstance(o, dict)]
    single = raw.get("display_output")
    if isinstance(single, dict):
        return [single]
    return []

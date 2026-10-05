"""View model for the notebook TUI: the pure, UI-free state core.

The TUI is a read-only spectator: it seeds from a ``notebook_state`` snapshot
and folds live WS frames (plain dicts, per ``docs/reference/notebook-protocol.md``)
into per-cell views. No Textual or sockets here, so it is testable with fake frames.
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
    # Hydrated snapshot outputs (markdown_text / inline_data_url / preview).
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    # From a live ``cell_output`` frame (name + preview).
    outputs: list[dict[str, Any]] = field(default_factory=list)
    # ``cell_output_delta``
    stream_text: str = ""
    # ``cell_console``, per stream: chunk 0 of a remote run starts that stream afresh.
    console_stdout: str = ""
    console_stderr: str = ""
    error: str | None = None
    # e.g. "iter 3/10" (``cell_iteration_progress``)
    iteration: str = ""
    # From a ``cell_output`` frame.
    duration_ms: int | None = None
    cache_hit: bool = False
    # From ``cell_test_*`` frames: "✓ 4/4", "✗ 2/4", or "tests…" while running.
    test_summary: str = ""
    # Per-test outcomes from the last cell_test_results frame (the Results tab).
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    test_unavailable: bool = False
    # ``cells/{id}.test.py``. Carried in every notebook_state snapshot, so read fresh.
    test_source: str = ""

    @property
    def console(self) -> str:
        return self.console_stdout + self.console_stderr


class NotebookViewModel:
    """Folds a ``notebook_state`` snapshot + live frames into per-cell views."""

    def __init__(self) -> None:
        self.notebook_name: str = ""
        self.cell_order: list[str] = []
        self.cells: dict[str, CellView] = {}
        # (from_cell_id, to_cell_id), from the snapshot ``dag`` block and dag_update frames.
        self.edges: list[tuple[str, str]] = []
        # Notebook-level activity line in the header (cascade / environment job / agent).
        self.banner: str = ""
        # External agent's narrated tool actions and explicit notes, chronological.
        self.agent_feed: list[str] = []
        self.agent_status: str = ""

    # --- snapshot ---

    def apply_notebook_state(self, payload: dict[str, Any]) -> None:
        """Seed (or re-seed) all cells from a ``notebook_state`` snapshot.

        Live-only fields the snapshot lacks (console, streamed deltas, last live
        ``cell_output``) are kept for cells that still exist.
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
                # ``# @name`` wins over the persisted notebook.toml name, as in the web UI.
                name=parse_annotations(source).name or str(raw.get("name") or ""),
                language=str(raw.get("language") or "python"),
                source=source,
                test_source=str(raw.get("test_source") or ""),
                status=str(raw.get("status") or "idle"),
                display_outputs=_snapshot_display_outputs(raw),
                outputs=prior.outputs if prior else [],
                stream_text=prior.stream_text if prior else "",
                console_stdout=prior.console_stdout if prior else "",
                console_stderr=prior.console_stderr if prior else "",
                error=prior.error if prior else None,
                duration_ms=prior.duration_ms if prior else None,
                cache_hit=prior.cache_hit if prior else False,
                # Test results arrive only via cell_test_* frames, so keep them across a
                # resync or the 2.5s auto-resync blanks the badge and Tests tab.
                test_summary=prior.test_summary if prior else "",
                test_cases=prior.test_cases if prior else [],
                test_unavailable=prior.test_unavailable if prior else False,
            )

        self.cell_order = order
        self.cells = new_cells
        self.edges = _parse_edges(payload.get("dag"))

    # --- incremental frames ---

    def apply_frame(self, msg_type: str, payload: dict[str, Any]) -> set[str]:
        """Fold one live frame in; return the affected cell ids (empty for unknown types).

        ``notebook_state`` goes through :meth:`apply_notebook_state`.
        ``impact_preview``, ``profiling_summary`` and ``inspect_result`` are sent
        only to the requesting client, so a spectator never receives them.
        """
        if msg_type == "dag_update":
            self.edges = _parse_edges(payload)
            return set(self.cell_order)  # whole-graph change

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
        if msg_type == "agent_note":
            self._apply_agent_note(payload)
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
            text = str(payload.get("text") or "")
            # Chunk 0 starts the stream for this run, so the last run's text goes.
            fresh = payload.get("chunk_seq") == 0
            if payload.get("stream") == "stderr":
                cell.console_stderr = ("" if fresh else cell.console_stderr) + text
            else:
                cell.console_stdout = ("" if fresh else cell.console_stdout) + text
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
            # The real counts for ready/error arrive in cell_test_results, so keep that badge.
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

    def _apply_agent_note(self, payload: dict[str, Any]) -> None:
        """Fold an ``agent_note`` frame into the chronological agent feed + status."""
        # "↹" marks an auto-narrated tool action (source="mcp"); "✎" an explicit note.
        source = str(payload.get("source") or "agent")
        text = str(payload.get("text") or "")
        glyph = "✎" if source == "agent" else "↹"
        self.agent_feed.append(f"{glyph} {text}")
        self.agent_status = source
        self.banner = f"🤖 {source}: {text}"


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

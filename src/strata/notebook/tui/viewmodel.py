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


class NotebookViewModel:
    """Folds a ``notebook_state`` snapshot + live frames into per-cell views."""

    def __init__(self) -> None:
        self.notebook_name: str = ""
        self.cell_order: list[str] = []
        self.cells: dict[str, CellView] = {}
        # DAG edges as (from_cell_id, to_cell_id), from the notebook_state `dag`
        # block (notebook_sync includes it) and refreshed by dag_update frames.
        self.edges: list[tuple[str, str]] = []

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
            new_cells[cid] = CellView(
                id=cid,
                name=str(raw.get("name") or ""),
                language=str(raw.get("language") or "python"),
                source=str(raw.get("source") or ""),
                status=str(raw.get("status") or "idle"),
                display_outputs=_snapshot_display_outputs(raw),
                outputs=prior.outputs if prior else [],
                stream_text=prior.stream_text if prior else "",
                console=prior.console if prior else "",
                error=prior.error if prior else None,
            )

        self.cell_order = order
        self.cells = new_cells
        self.edges = _parse_edges(payload.get("dag"))

    # -- incremental frames --------------------------------------------------

    def apply_frame(self, msg_type: str, payload: dict[str, Any]) -> set[str]:
        """Fold one live frame in; return the set of affected cell ids.

        Unknown / not-yet-handled frame types (cascade, dag, agent, env — M2+)
        are no-ops here and return an empty set. ``notebook_state`` is handled
        by :meth:`apply_notebook_state`, not here.
        """
        if msg_type == "dag_update":
            self.edges = _parse_edges(payload)
            return set(self.cell_order)  # whole-graph change

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
        elif msg_type == "cell_output_delta":
            if payload.get("kind") == "retry":
                cell.stream_text = ""
            cell.stream_text += str(payload.get("text") or "")
        elif msg_type == "cell_error":
            cell.error = str(payload.get("error") or "error")
        else:
            return set()
        return {cid}


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

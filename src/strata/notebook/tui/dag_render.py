"""Render a notebook DAG as a layered, boxed ASCII graph.

Pure (no Textual, no sockets) so it unit-tests with fixed inputs: given the cell
ids/labels/statuses and the edges, return a multi-line string with one box per
cell, laid out top-to-bottom in dependency layers and connected by lines.

Layout uses ``grandalf`` (pure-Python Sugiyama layered layout): it assigns each
cell a layer + a crossing-minimized horizontal position, and routes long edges
through *dummy* waypoints that it keeps clear of the boxes (``sug.ctrls``). This
module rasterizes that onto a character grid — boxes at each cell's x, edges as
orthogonal polylines through their waypoints. When grandalf is unavailable or the
layout fails (e.g. a cycle), it falls back to a simple longest-path layering with
even spacing so the view degrades rather than crashes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Status glyphs (kept in sync with app._STATUS_GLYPHS).
_GLYPHS: dict[str, str] = {
    "idle": "○",
    "running": "▶",
    "ready": "✓",
    "error": "✗",
    "stale": "⊘",
    "queued": "…",
}

_BOX_H = 3  # rendered box height (top border, label, bottom border)
_LAYER_GAP = 3  # rows between box rows, for edge routing
_ROW_PER_LAYER = _BOX_H + _LAYER_GAP


@dataclass
class _Placed:
    cid: str
    text: str
    row: int  # top row of the box
    center_col: int
    width: int

    @property
    def left(self) -> int:
        return self.center_col - self.width // 2

    @property
    def bottom(self) -> int:
        return self.row + _BOX_H - 1


def render_dag(
    order: list[str],
    labels: dict[str, str],
    statuses: dict[str, str],
    edges: list[tuple[str, str]],
    *,
    selected: str | None = None,
) -> str:
    """Render the DAG. ``order`` is cell ids in topological order; ``edges`` are
    ``(from_cell_id, to_cell_id)`` pairs; ``selected`` highlights one box.
    """
    if not order:
        return "(no cells)"

    adjacency = [(a, b) for a, b in edges if a in labels and b in labels and a != b]
    layout = _layout(order, labels, statuses, adjacency)
    return _rasterize(layout, selected).to_string()


@dataclass
class _Layout:
    boxes: dict[str, _Placed]
    # Per edge: orthogonal waypoints (row, col) from source bottom to target top.
    edge_paths: list[list[tuple[int, int]]]
    rows: int
    cols: int


def _box_text(cid: str, labels: dict[str, str], statuses: dict[str, str]) -> str:
    glyph = _GLYPHS.get(statuses.get(cid, ""), "?")
    return f"{glyph} {labels.get(cid, cid)}"


# -- layout -----------------------------------------------------------------


def _layout(
    order: list[str],
    labels: dict[str, str],
    statuses: dict[str, str],
    edges: list[tuple[str, str]],
) -> _Layout:
    try:
        return _grandalf_layout(order, labels, statuses, edges)
    except Exception:  # noqa: BLE001 — grandalf missing / layout failure → fallback
        return _fallback_layout(order, labels, statuses, edges)


def _grandalf_layout(
    order: list[str],
    labels: dict[str, str],
    statuses: dict[str, str],
    edges: list[tuple[str, str]],
) -> _Layout:
    from grandalf.graphs import Edge, Graph, Vertex
    from grandalf.layouts import SugiyamaLayout

    texts = {cid: _box_text(cid, labels, statuses) for cid in order}
    # grandalf is untyped and sets `.view` dynamically — keep these Any.
    vertices: dict[str, Any] = {cid: Vertex(cid) for cid in order}
    for cid, vertex in vertices.items():
        vertex.view = _VertexView(len(texts[cid]) + 2, _BOX_H)
    graph_edges = [Edge(vertices[a], vertices[b]) for a, b in edges]
    graph = Graph(list(vertices.values()), graph_edges)

    # rank (layer) + raw x for every real vertex; per-edge waypoint chain.
    rank: dict[str, int] = {}
    rawx: dict[str, float] = {}
    edge_chains: list[list[tuple[int, float]]] = []  # [(rank, x), …] per edge

    # grandalf lays out each connected component at the origin; offset each one
    # to the right of the previous so separate subgraphs / isolated cells don't
    # overlap.
    x_offset = 0.0
    for component in graph.C:
        sug = SugiyamaLayout(component)
        sug.init_all()
        sug.xspace = 2  # tighten horizontal spread (default leaves wide gaps)
        sug.draw()

        comp_xs = [v.view.xy[0] for layer in sug.layers for v in layer]
        comp_min = min(comp_xs, default=0.0)
        comp_max = max(comp_xs, default=0.0)
        shift = x_offset - comp_min

        for layer_idx, layer in enumerate(sug.layers):
            for vertex in layer:
                cid = getattr(vertex, "data", None)
                if isinstance(cid, str):
                    rank[cid] = layer_idx
                    rawx[cid] = float(vertex.view.xy[0]) + shift
        for edge in component.sE:
            src = getattr(edge.v[0], "data", None)
            dst = getattr(edge.v[1], "data", None)
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            ctrl = sug.ctrls.get(edge)
            if ctrl:  # long edge: rank → (real or dummy) vertex
                chain = [(r, float(ctrl[r].view.xy[0]) + shift) for r in sorted(ctrl) if r in ctrl]
            else:
                chain = [(rank[src], rawx[src]), (rank[dst], rawx[dst])]
            edge_chains.append(chain)

        x_offset += (comp_max - comp_min) + 12.0  # gap between components

    # Cells grandalf didn't place (isolated) → top layer, appended in order.
    for cid in order:
        rank.setdefault(cid, 0)
        rawx.setdefault(cid, 0.0)

    return _assemble(order, texts, rank, rawx, edge_chains)


def _fallback_layout(
    order: list[str],
    labels: dict[str, str],
    statuses: dict[str, str],
    edges: list[tuple[str, str]],
) -> _Layout:
    texts = {cid: _box_text(cid, labels, statuses) for cid in order}
    preds: dict[str, list[str]] = {c: [] for c in order}
    for a, b in edges:
        if b in preds and a in preds:
            preds[b].append(a)

    rank: dict[str, int] = {}

    def _depth(cid: str, seen: frozenset[str]) -> int:
        if cid in rank:
            return rank[cid]
        if cid in seen:
            return 0
        d = 0 if not preds[cid] else 1 + max(_depth(p, seen | {cid}) for p in preds[cid])
        rank[cid] = d
        return d

    for cid in order:
        _depth(cid, frozenset())

    # Even horizontal spacing within each layer, in topological order.
    by_layer: dict[int, list[str]] = {}
    for cid in order:
        by_layer.setdefault(rank[cid], []).append(cid)
    rawx: dict[str, float] = {}
    for layer_cells in by_layer.values():
        for i, cid in enumerate(layer_cells):
            rawx[cid] = float(i) * 1000.0  # wide units; _assemble normalizes
    edge_chains = [[(rank[a], rawx[a]), (rank[b], rawx[b])] for a, b in edges]
    return _assemble(order, texts, rank, rawx, edge_chains)


def _assemble(
    order: list[str],
    texts: dict[str, str],
    rank: dict[str, int],
    rawx: dict[str, float],
    edge_chains: list[list[tuple[int, float]]],
) -> _Layout:
    """Map ranks→rows and raw x→columns, then build boxes + edge polylines."""
    # Normalize x to non-negative integer columns. grandalf x is roughly in the
    # character units we gave each view, so a 1:1 map keeps boxes apart; a global
    # shift then ensures the left-most box edge (not just its center) clears the
    # margin so nothing clips off the left.
    margin = 2
    centers = {cid: int(round(x)) for cid, x in rawx.items()}
    min_left = min(
        (centers[cid] - (len(texts[cid]) + 2) // 2 for cid in order),
        default=0,
    )
    min_wp = min((int(round(x)) for chain in edge_chains for _, x in chain), default=0)
    shift = margin - min(min_left, min_wp)

    def to_col(x: float) -> int:
        return int(round(x)) + shift

    boxes: dict[str, _Placed] = {}
    for cid in order:
        boxes[cid] = _Placed(
            cid=cid,
            text=texts[cid],
            row=rank[cid] * _ROW_PER_LAYER,
            center_col=to_col(rawx[cid]),
            width=len(texts[cid]) + 2,
        )

    # Edge polylines: source bottom-center → waypoints → target top-center.
    paths: list[list[tuple[int, int]]] = []
    for chain in edge_chains:
        pts: list[tuple[int, int]] = []
        for i, (rnk, x) in enumerate(chain):
            col = to_col(x)
            if i == 0:  # source: leave from the box bottom
                pts.append((rnk * _ROW_PER_LAYER + _BOX_H - 1, col))
            elif i == len(chain) - 1:  # target: arrive at the box top
                pts.append((rnk * _ROW_PER_LAYER, col))
            else:  # dummy waypoint: mid of that layer band
                pts.append((rnk * _ROW_PER_LAYER + _BOX_H // 2, col))
        paths.append(pts)

    rows = (max(rank.values(), default=0) + 1) * _ROW_PER_LAYER
    cols = max((box.left + box.width for box in boxes.values()), default=1) + margin
    cols = max(cols, max((c for path in paths for _, c in path), default=0) + margin)
    return _Layout(boxes=boxes, edge_paths=paths, rows=rows, cols=cols)


class _VertexView:
    def __init__(self, w: int, h: int) -> None:
        self.w = w
        self.h = h
        self.xy = (0.0, 0.0)


# -- rasterization ----------------------------------------------------------


@dataclass
class _Grid:
    rows: int
    cols: int
    _cells: list[list[str]] = field(init=False)

    def __post_init__(self) -> None:
        self._cells = [[" "] * self.cols for _ in range(self.rows)]

    def put(self, r: int, c: int, ch: str) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self._cells[r][c] = ch

    def put_text(self, r: int, c: int, text: str) -> None:
        for i, ch in enumerate(text):
            self.put(r, c + i, ch)

    def line(self, r: int, c: int, ch: str) -> None:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        self._cells[r][c] = _merge(self._cells[r][c], ch)

    def to_string(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self._cells).rstrip("\n")


_SEG = {
    "─": frozenset("WE"),
    "│": frozenset("NS"),
    "┌": frozenset("SE"),
    "┐": frozenset("SW"),
    "└": frozenset("NE"),
    "┘": frozenset("NW"),
    "├": frozenset("NSE"),
    "┤": frozenset("NSW"),
    "┬": frozenset("SEW"),
    "┴": frozenset("NEW"),
    "┼": frozenset("NSEW"),
}
_SEG_INV = {v: k for k, v in _SEG.items()}


def _merge(existing: str, incoming: str) -> str:
    a = _SEG.get(existing)
    b = _SEG.get(incoming)
    if a is None or b is None:
        return incoming
    return _SEG_INV.get(a | b, incoming)


def _rasterize(layout: _Layout, selected: str | None) -> _Grid:
    grid = _Grid(rows=max(layout.rows, 1), cols=max(layout.cols, 1))
    for index, path in enumerate(layout.edge_paths):  # edges first; boxes on top
        _draw_polyline(grid, path, lane=index % _LAYER_GAP)
    for box in layout.boxes.values():
        _draw_box(grid, box, selected=box.cid == selected)
    return grid


def _draw_box(grid: _Grid, box: _Placed, *, selected: bool) -> None:
    left, top, w = box.left, box.row, box.width
    tl, tr, bl, br, hz, vt = (
        ("╔", "╗", "╚", "╝", "═", "║") if selected else ("┌", "┐", "└", "┘", "─", "│")
    )
    grid.put(top, left, tl)
    grid.put(top, left + w - 1, tr)
    grid.put(top + 2, left, bl)
    grid.put(top + 2, left + w - 1, br)
    for i in range(left + 1, left + w - 1):
        # ``line`` so an edge junction (┬/┴) under/over the border shows through.
        grid.line(top, i, hz)
        grid.line(top + 2, i, hz)
    grid.put(top + 1, left, vt)
    grid.put(top + 1, left + w - 1, vt)
    grid.put_text(top + 1, left + 1, box.text)


def _draw_polyline(grid: _Grid, pts: list[tuple[int, int]], *, lane: int) -> None:
    """Draw an orthogonal edge through ``pts`` (top→down); horizontals jog in the
    inter-layer gap on a per-edge ``lane`` row so parallel edges don't overlap.
    """
    if len(pts) < 2:
        return
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        if c0 == c1:
            for r in range(min(r0, r1), max(r0, r1) + 1):
                grid.line(r, c0, "│")
            continue
        # Jog row inside the gap below the source row (never on a box row).
        jog = min(r0 + 1 + lane, r1 - 1) if r1 > r0 else r0
        for r in range(min(r0, jog), max(r0, jog) + 1):
            grid.line(r, c0, "│")
        lo, hi = sorted((c0, c1))
        for c in range(lo, hi + 1):
            grid.line(jog, c, "─")
        grid.line(jog, c0, "┐" if c1 < c0 else "┌")
        grid.line(jog, c1, "└" if c1 < c0 else "┘")
        for r in range(min(jog, r1), max(jog, r1) + 1):
            grid.line(r, c1, "│")

    # Stamp the box junctions last (overwrite) so they're clean tees, not the
    # ``┼`` a passing vertical would merge to: tee DOWN out of the source bottom,
    # tee UP into the target top.
    grid.put(pts[0][0], pts[0][1], "┬")
    grid.put(pts[-1][0], pts[-1][1], "┴")

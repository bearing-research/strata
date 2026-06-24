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
_OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}


def _merge(existing: str, incoming: str) -> str:
    a = _SEG.get(existing)
    b = _SEG.get(incoming)
    if a is None or b is None:
        return incoming
    return _SEG_INV.get(a | b, incoming)


@dataclass
class _Grid:
    rows: int
    cols: int
    _cells: list[list[str]] = field(init=False)
    # Accumulated connection directions per line cell. Rendering the glyph from
    # the *union* of directions makes every junction correct by construction:
    # an elbow → corner, a branch off a shared trunk → tee, a true crossing → ┼.
    _dirs: dict[tuple[int, int], set[str]] = field(init=False)

    def __post_init__(self) -> None:
        self._cells = [[" "] * self.cols for _ in range(self.rows)]
        self._dirs = {}

    def _in(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def put(self, r: int, c: int, ch: str) -> None:
        if self._in(r, c):
            self._cells[r][c] = ch

    def put_text(self, r: int, c: int, text: str) -> None:
        for i, ch in enumerate(text):
            self.put(r, c + i, ch)

    def merge(self, r: int, c: int, ch: str) -> None:
        if self._in(r, c):
            self._cells[r][c] = _merge(self._cells[r][c], ch)

    def add_dir(self, r: int, c: int, direction: str) -> None:
        if self._in(r, c):
            self._dirs.setdefault((r, c), set()).add(direction)

    def render_dirs(self) -> None:
        for (r, c), dirs in self._dirs.items():
            glyph = _SEG_INV.get(frozenset(dirs))
            if glyph is not None:
                self._cells[r][c] = glyph

    def to_string(self) -> str:
        return "\n".join("".join(row).rstrip() for row in self._cells).rstrip("\n")


def _rasterize(layout: _Layout, selected: str | None) -> _Grid:
    grid = _Grid(rows=max(layout.rows, 1), cols=max(layout.cols, 1))
    # 1. Accumulate every edge's connection directions, then render junctions.
    for index, path in enumerate(layout.edge_paths):
        _accumulate_edge(grid, path, lane=index % _LAYER_GAP)
    grid.render_dirs()
    # 2. Boxes draw over the lines.
    for box in layout.boxes.values():
        _draw_box(grid, box, selected=box.cid == selected)
    # 3. Box-attach junctions, merged onto the (now-drawn) box borders: a tee
    #    DOWN out of each source bottom, a tee UP into each target top.
    for path in layout.edge_paths:
        if path:
            grid.merge(path[0][0], path[0][1], "┬")
            grid.merge(path[-1][0], path[-1][1], "┴")
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
        grid.put(top, i, hz)
        grid.put(top + 2, i, hz)
    grid.put(top + 1, left, vt)
    grid.put(top + 1, left + w - 1, vt)
    grid.put_text(top + 1, left + 1, box.text)


def _accumulate_edge(grid: _Grid, pts: list[tuple[int, int]], *, lane: int) -> None:
    """Walk an edge's full orthogonal cell path and record each step's directions.

    Horizontal jogs sit on a per-edge ``lane`` row inside the inter-layer gap so
    parallel edges don't pile onto the same row.
    """
    cells = _edge_cells(pts, lane)
    for prev, cur in zip(cells, cells[1:]):
        step = _step_dir(prev, cur)
        grid.add_dir(prev[0], prev[1], step)
        grid.add_dir(cur[0], cur[1], _OPP[step])


def _edge_cells(pts: list[tuple[int, int]], lane: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(pts, pts[1:]):
        seg = _elbow_cells(r0, c0, r1, c1, lane)
        if out and seg and out[-1] == seg[0]:
            seg = seg[1:]  # don't repeat the shared join cell
        out.extend(seg)
    return out


def _elbow_cells(r0: int, c0: int, r1: int, c1: int, lane: int) -> list[tuple[int, int]]:
    if c0 == c1:
        return [(r, c0) for r in _between(r0, r1)]
    # Down the source column to a jog row in the gap, across, down to the target.
    jog = min(r0 + 1 + lane, r1 - 1) if r1 > r0 else r0
    cells = [(r, c0) for r in _between(r0, jog)]
    cells += [(jog, c) for c in _between(c0, c1)][1:]
    cells += [(r, c1) for r in _between(jog, r1)][1:]
    return cells


def _between(a: int, b: int) -> list[int]:
    step = 1 if b >= a else -1
    return list(range(a, b + step, step))


def _step_dir(a: tuple[int, int], b: tuple[int, int]) -> str:
    if b[0] < a[0]:
        return "N"
    if b[0] > a[0]:
        return "S"
    return "W" if b[1] < a[1] else "E"

"""Unit tests for the ASCII DAG renderer.

Pure-function tests: assert structural properties (every cell boxed, glyphs +
selection shown, edges drawn, graceful fallback/edge cases) rather than exact
art, which would be brittle to layout tweaks.
"""

from __future__ import annotations

from strata.notebook.tui import dag_render
from strata.notebook.tui.dag_render import render_dag

_BOX_CHARS = set("┌┐└┘─│┬┴├┤┼")


def _diamond():
    order = ["a", "b", "c", "d"]
    labels = {c: c for c in order}
    statuses = {"a": "ready", "b": "running", "c": "stale", "d": "error"}
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]
    return order, labels, statuses, edges


def test_empty_notebook():
    assert render_dag([], {}, {}, []) == "(no cells)"


def test_every_cell_is_boxed_with_its_glyph():
    order, labels, statuses, edges = _diamond()
    art = render_dag(order, labels, statuses, edges)
    for cid in order:
        assert cid in art  # the label
    # Status glyphs are rendered.
    assert "✓" in art and "▶" in art and "⊘" in art and "✗" in art
    # Box-drawing characters are present (boxes + edges were drawn).
    assert _BOX_CHARS & set(art)


def test_selected_cell_uses_double_border():
    order, labels, statuses, edges = _diamond()
    art = render_dag(order, labels, statuses, edges, selected="b")
    assert "╔" in art and "╝" in art  # the heavy box is the selection


def test_isolated_cells_no_edges():
    order = ["x", "y"]
    art = render_dag(order, {"x": "x", "y": "y"}, {"x": "idle", "y": "idle"}, [])
    assert "x" in art and "y" in art  # both render, no crash without edges


def test_cycle_does_not_crash():
    order = ["a", "b"]
    art = render_dag(
        order, {"a": "a", "b": "b"}, {"a": "ready", "b": "ready"}, [("a", "b"), ("b", "a")]
    )
    assert "a" in art and "b" in art


def test_falls_back_when_grandalf_unavailable(monkeypatch):
    """If grandalf layout raises, the fallback longest-path layout still renders."""

    def _boom(*args, **kwargs):
        raise RuntimeError("no grandalf")

    monkeypatch.setattr(dag_render, "_grandalf_layout", _boom)
    order, labels, statuses, edges = _diamond()
    art = render_dag(order, labels, statuses, edges)
    for cid in order:
        assert cid in art
    assert _BOX_CHARS & set(art)


def test_layers_are_top_down_in_dependency_order():
    """A linear chain a→b→c puts a above b above c (more rows)."""
    order = ["a", "b", "c"]
    art = render_dag(
        order, {c: c for c in order}, {c: "ready" for c in order}, [("a", "b"), ("b", "c")]
    )
    lines = art.splitlines()
    row_of = {cid: next(i for i, ln in enumerate(lines) if cid in ln) for cid in order}
    assert row_of["a"] < row_of["b"] < row_of["c"]

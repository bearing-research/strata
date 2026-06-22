"""Tests for the run-impact preview (upstream cascade + downstream staleness)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strata.notebook.impact import ImpactAnalyzer, ImpactPreview
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _session(tmp_path, cells):
    notebook_dir = create_notebook(tmp_path, "impact_test")
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id)
        write_cell(notebook_dir, cell_id, source)
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


@pytest.fixture
def linear(tmp_path):
    # root -> m1 -> m2 -> leaf
    return _session(
        tmp_path,
        [
            ("root", "x = 1"),
            ("m1", "y = x + 1"),
            ("m2", "z = y + 1"),
            ("leaf", "w = z + 1"),
        ],
    )


@pytest.fixture
def diamond(tmp_path):
    # root -> a, root -> b, (a, b) -> join
    return _session(
        tmp_path,
        [
            ("root", "x = 1"),
            ("a", "p = x + 1"),
            ("b", "q = x + 1"),
            ("join", "r = p + q"),
        ],
    )


class TestDownstream:
    def test_ready_downstream_cells_are_reported_stale(self, linear):
        for cell in linear.notebook_state.cells:
            cell.status = "ready"

        preview = ImpactAnalyzer(linear).preview("root")

        ds_ids = {d.cell_id for d in preview.downstream}
        assert ds_ids == {"m1", "m2", "leaf"}
        assert all(d.new_status == "stale:upstream" for d in preview.downstream)
        assert all(d.current_status == "ready" for d in preview.downstream)
        assert preview.has_impact

    def test_idle_downstream_cells_are_not_reported(self, linear):
        # Cells default to idle; only currently-ready cells can go stale.
        preview = ImpactAnalyzer(linear).preview("root")
        assert preview.downstream == []

    def test_leaf_has_no_downstream(self, linear):
        for cell in linear.notebook_state.cells:
            cell.status = "ready"
        preview = ImpactAnalyzer(linear).preview("leaf")
        assert preview.downstream == []

    def test_downstream_cell_name_uses_first_define(self, linear):
        for cell in linear.notebook_state.cells:
            cell.status = "ready"
        preview = ImpactAnalyzer(linear).preview("root")
        by_id = {d.cell_id: d for d in preview.downstream}
        assert by_id["m1"].cell_name == "y"  # m1 defines `y`

    def test_diamond_visits_each_cell_once(self, diamond):
        # `join` is reachable from both `a` and `b`; the BFS must dedupe it.
        for cell in diamond.notebook_state.cells:
            cell.status = "ready"
        preview = ImpactAnalyzer(diamond).preview("root")
        ds_ids = [d.cell_id for d in preview.downstream]
        assert sorted(ds_ids) == ["a", "b", "join"]
        assert ds_ids.count("join") == 1

    def test_compute_downstream_without_dag_is_empty(self):
        analyzer = ImpactAnalyzer.__new__(ImpactAnalyzer)
        analyzer.session = SimpleNamespace(dag=None)
        assert analyzer._compute_downstream("anything") == []


class TestUpstream:
    def test_upstream_empty_when_all_ready(self, linear):
        for cell in linear.notebook_state.cells:
            cell.status = "ready"
        preview = ImpactAnalyzer(linear).preview("leaf")
        # Everything upstream is ready → no cascade → no upstream steps.
        assert preview.upstream == []
        assert preview.estimated_ms == 0

    def test_upstream_includes_stale_ancestors(self, linear):
        # All idle: running leaf needs root, m1, m2 first.
        preview = ImpactAnalyzer(linear).preview("leaf")
        upstream_ids = {s.cell_id for s in preview.upstream}
        assert {"root", "m1", "m2"} <= upstream_ids
        assert preview.has_impact


class TestHasImpact:
    def test_no_impact_for_isolated_cell(self, tmp_path):
        session = _session(tmp_path, [("solo", "a = 1")])
        preview = ImpactAnalyzer(session).preview("solo")
        assert preview.has_impact is False

    def test_target_cell_alone_in_upstream_is_not_impact(self):
        from strata.notebook.cascade import CascadeStep

        preview = ImpactPreview(
            target_cell_id="t",
            upstream=[CascadeStep(cell_id="t", cell_name="t", reason="stale", estimated_ms=0)],
        )
        # The only upstream step is the target itself → no real impact.
        assert preview.has_impact is False

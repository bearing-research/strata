"""Tests for the widget cell analyzer (P1 — declaration + DAG participation).

A widget cell is declarative: each ``name = control(...)`` line defines a DAG
variable and carries a descriptor. The cell is never executed; these tests
exercise the static ``ast``-based extraction, its error reporting, the language
registry wiring, and that a widget cell participates in the DAG as a producer.
"""

from __future__ import annotations

from strata.notebook.dag import CellAnalysisWithId, NotebookDag
from strata.notebook.languages.analyzer import (
    analyze_cell_by_language,
    get_language_analyzer,
)
from strata.notebook.models import CellLanguage, CellState
from strata.notebook.widget_analyzer import analyze_widget_cell


class TestAnalyzeWidgetCell:
    def test_extracts_defines_and_descriptors(self):
        src = (
            "alpha = slider(0, 1, step=0.01, default=0.5)\n"
            'optimizer = dropdown(["adam", "sgd"], default="adam")\n'
            "epochs = number(default=10, min=1, max=200)\n"
            "use_gpu = checkbox(default=True)\n"
            'label = text("baseline")\n'
        )
        result = analyze_widget_cell(src)

        assert result.defines == ["alpha", "optimizer", "epochs", "use_gpu", "label"]
        assert result.references == []
        assert result.errors == []
        by_name = {d.name: d for d in result.descriptors}
        assert by_name["alpha"].kind == "slider"
        assert by_name["alpha"].params == {"min": 0, "max": 1, "step": 0.01, "default": 0.5}
        assert by_name["optimizer"].default == "adam"
        assert by_name["optimizer"].params["options"] == ["adam", "sgd"]

    def test_default_resolution_when_omitted(self):
        result = analyze_widget_cell(
            "a = slider(2, 8)\n"  # default → min
            'b = dropdown(["x", "y"])\n'  # default → first option
            "c = checkbox()\n"  # default → False
            "d = text()\n"  # default → ""
        )
        defaults = {d.name: d.default for d in result.descriptors}
        assert defaults == {"a": 2, "b": "x", "c": False, "d": ""}

    def test_ignores_non_control_lines(self):
        # Blank lines and non-assignment statements are skipped, not errors.
        result = analyze_widget_cell("\nalpha = slider(0, 1)\n")
        assert result.defines == ["alpha"]
        assert result.errors == []

    def test_unknown_control_is_an_error(self):
        result = analyze_widget_cell("x = frobnicate(1)")
        assert result.defines == []
        assert any("Unknown widget control" in e for e in result.errors)

    def test_non_literal_argument_is_an_error(self):
        result = analyze_widget_cell("x = slider(0, some_var)")
        assert result.defines == []
        assert any("literal" in e for e in result.errors)

    def test_duplicate_variable_is_an_error(self):
        result = analyze_widget_cell("x = number(1)\nx = number(2)")
        assert result.defines == ["x"]  # first wins
        assert any("Duplicate" in e for e in result.errors)

    def test_too_many_positional_args(self):
        result = analyze_widget_cell("x = text('a', 'b')")
        assert result.defines == []
        assert any("positional" in e for e in result.errors)

    def test_syntax_error_is_reported_not_raised(self):
        result = analyze_widget_cell("x = slider(")
        assert result.defines == []
        assert any("invalid syntax" in e for e in result.errors)


class TestRegistryDispatch:
    def test_widget_language_dispatches_to_widget_analyzer(self):
        assert get_language_analyzer(CellLanguage.WIDGET).__class__.__name__ == "_WidgetAnalyzer"

    def test_analyze_cell_by_language_returns_defines_no_references(self):
        cell = CellState(
            id="w1",
            source="alpha = slider(0, 1, default=0.5)\nbeta = number(default=3)",
            language=CellLanguage.WIDGET,
        )
        analyzed = analyze_cell_by_language(cell, session=None)
        assert analyzed.defines == ["alpha", "beta"]
        assert analyzed.references == []
        assert analyzed.mutation_defines == []


class TestWidgetDagParticipation:
    def test_widget_cell_wires_edges_to_downstream_consumers(self):
        # The DAG is language-agnostic — a widget cell is just a producer whose
        # defines come from analyze_widget_cell. A downstream Python cell that
        # references the widget variable gets an edge.
        widget = analyze_widget_cell("alpha = slider(0, 1, default=0.5)")
        cells = [
            CellAnalysisWithId(id="controls", defines=widget.defines, references=[]),
            CellAnalysisWithId(id="use", defines=["result"], references=["alpha"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert "use" in dag.cell_downstream["controls"]
        assert "controls" in dag.cell_upstream["use"]
        assert "alpha" in dag.consumed_variables["controls"]

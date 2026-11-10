"""``NotebookDag.from_cells``: wiring names already analyzed, as the Jupyter import does.

How cells wire (edges, order, cycles, variants, builtins) is tested in cellgraph.
"""

import pytest
from cellgraph.python import analyze_cell

from strata.notebook.dag import (
    CellAnalysisWithId,
    DagEdge,
    NotebookDag,
    VariantNameCollisionError,
)


def _analyzed_cell(cell_id: str, source: str) -> CellAnalysisWithId:
    """Build a DAG cell from real source via the analyzer (defines/references)."""
    analysis = analyze_cell(source)
    return CellAnalysisWithId(
        id=cell_id,
        defines=analysis.defines,
        references=analysis.references,
        builtin_references=analysis.builtin_references,
    )


class TestRealWorldDAGs:
    def test_data_analysis_pipeline(self):
        """Typical data analysis: load → clean → aggregate → plot."""
        cells = [
            CellAnalysisWithId(
                id="load",
                defines=["raw_data"],
                references=[],
            ),
            CellAnalysisWithId(
                id="clean",
                defines=["cleaned_data"],
                references=["raw_data"],
            ),
            CellAnalysisWithId(
                id="aggregate",
                defines=["summary"],
                references=["cleaned_data"],
            ),
            CellAnalysisWithId(
                id="plot",
                defines=[],
                references=["summary"],
            ),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.topological_order == ["load", "clean", "aggregate", "plot"]
        assert dag.roots == {"load"}
        assert dag.leaves == {"plot"}

    def test_ml_pipeline(self):
        """ML pipeline: load → split → train, evaluate."""
        cells = [
            CellAnalysisWithId(id="load", defines=["X", "y"], references=[]),
            CellAnalysisWithId(
                id="split",
                defines=["X_train", "X_test", "y_train", "y_test"],
                references=["X", "y"],
            ),
            CellAnalysisWithId(
                id="train",
                defines=["model"],
                references=["X_train", "y_train"],
            ),
            CellAnalysisWithId(
                id="evaluate",
                defines=["accuracy"],
                references=["model", "X_test", "y_test"],
            ),
        ]
        dag = NotebookDag.from_cells(cells)

        # Diamond-like: load → split → train, and split → evaluate (but train also → evaluate)
        assert dag.roots == {"load"}
        assert dag.leaves == {"evaluate"}
        assert dag.cell_upstream["evaluate"] == ["train", "split"]

    def test_exploratory_cells(self):
        """Exploratory notebook: load + independent explorations."""
        cells = [
            CellAnalysisWithId(id="load", defines=["df"], references=[]),
            CellAnalysisWithId(id="explore1", defines=[], references=["df"]),
            CellAnalysisWithId(id="explore2", defines=[], references=["df"]),
            CellAnalysisWithId(id="explore3", defines=[], references=["df"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # Fan-out from load to multiple explorations
        assert dag.roots == {"load"}
        assert dag.leaves == {"explore1", "explore2", "explore3"}
        assert dag.cell_downstream["load"] == ["explore1", "explore2", "explore3"]

    def test_shared_computation(self):
        """Shared computation: load → features → model, load → features → evaluate."""
        cells = [
            CellAnalysisWithId(id="load", defines=["raw"], references=[]),
            CellAnalysisWithId(id="features", defines=["X", "y"], references=["raw"]),
            CellAnalysisWithId(id="train", defines=["model"], references=["X", "y"]),
            CellAnalysisWithId(
                id="evaluate",
                defines=["score"],
                references=["model", "X", "y"],
            ),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.cell_downstream["features"] == ["train", "evaluate"]


class TestAfterEdges:
    """``@after`` ordering-only edges.

    A cell that depends on an upstream side effect (a setup cell seeding a SQLite file)
    declares ``# @after <cell-id>``. The edge joins the wiring and topological order but adds
    nothing to ``consumed_variables``.
    """

    def test_after_cycle_still_detected(self):
        """Ordering edges feed cycle detection same as variable edges."""
        cells = [
            CellAnalysisWithId(id="a", defines=[], references=[], after=["b"]),
            CellAnalysisWithId(id="b", defines=[], references=[], after=["a"]),
        ]
        with pytest.raises(ValueError, match="Cycle"):
            NotebookDag.from_cells(cells)


class TestVariantGroups:
    """Variant cells share a defines contract; only the active variant is in the DAG."""

    def test_name_collision_raises(self):
        """Two cells with same (group, variant_name) is irrecoverable."""
        cells = [
            CellAnalysisWithId(
                id="c1",
                defines=["preds"],
                references=[],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="c2",
                defines=["preds"],
                references=[],
                variant_group="model",
                variant_name="a",
            ),
        ]
        with pytest.raises(VariantNameCollisionError):
            NotebookDag.from_cells(cells)


class TestBuiltinShadowingEdges:
    """``from_cells`` hands ``builtin_references`` on: a variable named after a builtin
    (``input``, ``type``, ``id``) still wires its reader.
    """

    def test_shadowed_builtin_wires_edge_and_consumption(self):
        cells = [
            _analyzed_cell("load", "input = load_data()"),
            _analyzed_cell("fit", "model = fit(input)"),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["fit"] == ["load"]
        assert "input" in dag.consumed_variables["load"]
        assert DagEdge(from_cell_id="load", to_cell_id="fit", variable="input") in dag.edges

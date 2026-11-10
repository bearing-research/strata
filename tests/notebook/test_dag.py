"""Tests for DAG construction and analysis."""

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


class TestInplaceMutationRouting:
    """An ``inplace=True`` method call makes the mutating cell a (re)producer of the receiver.

    Downstream reads then resolve to the post-mutation value in both single-cell and batch
    execution.
    """

    def test_inplace_mutation_routes_downstream_through_mutating_cell(self):
        cells = [
            _analyzed_cell("load", "df = load_data()"),
            _analyzed_cell("mutate", "df.drop(columns=['x'], inplace=True)"),
            _analyzed_cell("use", "total = df.sum()"),
        ]
        dag = NotebookDag.from_cells(cells)

        # df is read by the mutating cell (edge load → mutate) and the mutating
        # cell becomes df's producer for the downstream reader (mutate → use).
        assert dag.cell_upstream["mutate"] == ["load"]
        assert dag.cell_upstream["use"] == ["mutate"]

        # So the mutating cell must serialize df (recapture), and load's df is
        # consumed by the mutating cell, not directly by the reader.
        assert dag.consumed_variables["mutate"] == {"df"}
        assert dag.consumed_variables["load"] == {"df"}

    def test_non_inplace_method_does_not_reroute(self):
        """Without inplace=True the call is a pure read; df still flows load → use."""
        cells = [
            _analyzed_cell("load", "df = load_data()"),
            _analyzed_cell("peek", "head = df.head()"),
            _analyzed_cell("use", "total = df.sum()"),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.cell_upstream["use"] == ["load"]
        assert dag.consumed_variables["peek"] == set()


class TestDagBuildingBasics:
    def test_diamond_dependency(self):
        """Diamond dependency: A → B, A → C, B+C → D."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert len(dag.edges) == 4
        edges_str = {(e.from_cell_id, e.to_cell_id, e.variable) for e in dag.edges}
        assert edges_str == {
            ("a", "b", "x"),
            ("a", "c", "x"),
            ("b", "d", "y"),
            ("c", "d", "z"),
        }

        assert dag.cell_upstream == {
            "a": [],
            "b": ["a"],
            "c": ["a"],
            "d": ["b", "c"],
        }
        assert dag.cell_downstream == {
            "a": ["b", "c"],
            "b": ["d"],
            "c": ["d"],
            "d": [],
        }

        assert dag.roots == {"a"}
        assert dag.leaves == {"d"}

    def test_multiple_roots(self):
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=[]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x", "y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.roots == {"a", "b"}
        assert dag.leaves == {"c"}


class TestDagMultipleVariables:
    def test_multiple_vars_one_edge(self):
        """Two cells sharing multiple variables create one edge per variable."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x", "y"], references=[]),
            CellAnalysisWithId(id="b", defines=["z"], references=["x", "y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # Two edges: one for x, one for y
        assert len(dag.edges) == 2
        edges_vars = {e.variable for e in dag.edges}
        assert edges_vars == {"x", "y"}

        # But only one upstream/downstream relationship
        assert dag.cell_upstream == {"a": [], "b": ["a"]}
        assert dag.cell_downstream == {"a": ["b"], "b": []}


class TestTopologicalSort:
    def test_topo_sort_diamond(self):
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)

        order = dag.topological_order
        assert order[0] == "a"
        assert order[-1] == "d"
        # b and c can be in any order as long as they're between a and d
        assert set(order[1:3]) == {"b", "c"}


class TestCycleDetection:
    def test_forward_reference_no_cycle(self):
        """A references y, defined later by B: no cycle and no edge.

        References resolve against producers before each cell, so the runtime reports a NameError
        rather than the build raising.
        """
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=["y"]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)
        # b links to a via x; a has no predecessor for its y reference.
        assert dag.cell_upstream["a"] == []
        assert dag.cell_upstream["b"] == ["a"]

    def test_self_reference_is_not_a_cycle(self):
        """``x = x + 1`` with no upstream is an intra-cell rebind, not a cycle.

        analyze_cell usually strips ``x`` from references already; this pins the DAG side.
        """
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["a"] == []

    def test_forward_chain_no_cycle(self):
        """A→z, B→x, C→y where producers appear after consumers.

        Only B links to A (x is available); A and C's forward refs
        resolve to nothing and leave them as roots in source order.
        """
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=["z"]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["a"] == []
        assert dag.cell_upstream["b"] == ["a"]
        assert dag.cell_upstream["c"] == ["b"]


class TestCascadePlan:
    def test_cascade_linear(self):
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # To run c, need a and b, in execution order.
        reach = dag.upstream_reachable("c")
        assert [cid for cid in dag.topological_order if cid in reach] == ["a", "b", "c"]

    def test_cascade_diamond(self):
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # To run d, need a, b, c
        assert dag.upstream_reachable("d") == {"a", "b", "c", "d"}

    def test_cascade_root_cell(self):
        """Cascade for root cell needs only itself."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.upstream_reachable("a") == {"a"}

    def test_cascade_no_deps(self):
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=[]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.upstream_reachable("b") == {"b"}


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

    def test_after_an_inactive_variant_adds_no_edge(self):
        cells = [
            CellAnalysisWithId(
                id="a", defines=["m"], references=[], variant_group="g", variant_name="a"
            ),
            CellAnalysisWithId(
                id="b", defines=["m"], references=[], variant_group="g", variant_name="b"
            ),
            CellAnalysisWithId(id="query", defines=[], references=[], after=["b"]),
        ]
        dag = NotebookDag.from_cells(cells, variant_active_selections={"g": "a"})
        assert dag.cell_upstream["query"] == []
        assert dag.cell_downstream["b"] == []

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

    def test_unknown_active_falls_back_to_first(self):
        """toml names a nonexistent variant → fall back to source-order first."""
        cells = [
            CellAnalysisWithId(
                id="model_a",
                defines=["preds"],
                references=[],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="model_b",
                defines=["preds"],
                references=[],
                variant_group="model",
                variant_name="b",
            ),
        ]
        dag = NotebookDag.from_cells(cells, variant_active_selections={"model": "ghost"})

        assert dag.variable_producer["preds"] == "model_a"

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
    """A variable named after a builtin (``input``, ``type``, ``id``) still wires edges.

    ``references`` drops builtin names for display, so the DAG resolves
    ``builtin_references`` against the producer map; an unshadowed builtin wires nothing.
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

    def test_unshadowed_builtin_wires_nothing(self):
        cells = [
            _analyzed_cell("a", "x = 1"),
            _analyzed_cell("b", "print(x)"),  # print: builtin, no producer
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["b"] == ["a"]  # only the x edge
        assert dag.consumed_variables["a"] == {"x"}

    def test_common_ml_names(self):
        # id / type / filter are all builtins users routinely shadow.
        cells = [
            _analyzed_cell("a", "id = next_run_id()\ntype = pick_model()"),
            _analyzed_cell("b", "run = launch(id, type)"),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["b"] == ["a"]
        assert dag.consumed_variables["a"] == {"id", "type"}

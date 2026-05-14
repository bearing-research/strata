"""Tests for DAG construction and analysis."""

import pytest

from strata.notebook.dag import (
    CellAnalysisWithId,
    DagEdge,
    NotebookDag,
    VariantNameCollisionError,
)


class TestDagBuildingBasics:
    """Test basic DAG construction."""

    def test_single_cell_no_deps(self):
        """Single cell with no dependencies."""
        cells = [CellAnalysisWithId(id="a", defines=["x"], references=[])]
        dag = NotebookDag.from_cells(cells)

        assert len(dag.edges) == 0
        assert dag.cell_upstream == {"a": []}
        assert dag.cell_downstream == {"a": []}
        assert dag.roots == {"a"}
        assert dag.leaves == {"a"}

    def test_linear_chain(self):
        """Linear dependency chain: A → B → C."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # Check edges
        assert len(dag.edges) == 2
        assert dag.edges[0] == DagEdge("a", "b", "x")
        assert dag.edges[1] == DagEdge("b", "c", "y")

        # Check upstream/downstream
        assert dag.cell_upstream == {"a": [], "b": ["a"], "c": ["b"]}
        assert dag.cell_downstream == {"a": ["b"], "b": ["c"], "c": []}

        # Check roots and leaves
        assert dag.roots == {"a"}
        assert dag.leaves == {"c"}

    def test_diamond_dependency(self):
        """Diamond dependency: A → B, A → C, B+C → D."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # Check edges
        assert len(dag.edges) == 4
        edges_str = {(e.from_cell_id, e.to_cell_id, e.variable) for e in dag.edges}
        assert edges_str == {
            ("a", "b", "x"),
            ("a", "c", "x"),
            ("b", "d", "y"),
            ("c", "d", "z"),
        }

        # Check upstream/downstream
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

        # Check roots and leaves
        assert dag.roots == {"a"}
        assert dag.leaves == {"d"}

    def test_multiple_roots(self):
        """DAG with multiple root cells."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=[]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x", "y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.roots == {"a", "b"}
        assert dag.leaves == {"c"}

    def test_multiple_leaves(self):
        """DAG with multiple leaf cells."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.roots == {"a"}
        assert dag.leaves == {"b", "c"}


class TestDagMultipleVariables:
    """Test DAG with multiple variables between cells."""

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

    def test_variable_shadowing(self):
        """When multiple cells define same variable, last one wins."""
        cells = [
            CellAnalysisWithId(id="a", defines=["df"], references=[]),
            CellAnalysisWithId(id="b", defines=["df"], references=[]),
            CellAnalysisWithId(id="c", defines=["result"], references=["df"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # df should flow from b (last define) to c, not from a
        edges_dict = {(e.from_cell_id, e.to_cell_id): e.variable for e in dag.edges}
        assert edges_dict == {("b", "c"): "df"}

        # a has no downstream (its output is shadowed)
        assert dag.cell_downstream["a"] == []
        assert dag.cell_upstream["c"] == ["b"]

        # c consumes df from b
        assert dag.consumed_variables["b"] == {"df"}


class TestDagConsumedVariables:
    """Test consumed_variables tracking."""

    def test_consumed_variables_basic(self):
        """Track which variables are consumed by downstream cells."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x", "y"], references=[]),
            CellAnalysisWithId(id="b", defines=[], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # x is consumed by b, y is not
        assert dag.consumed_variables["a"] == {"x"}

    def test_unconsumed_variables_not_cached(self):
        """Variables not consumed by downstream are not cached."""
        cells = [
            CellAnalysisWithId(id="a", defines=["df", "temp"], references=[]),
            CellAnalysisWithId(id="b", defines=[], references=["df"]),
        ]
        dag = NotebookDag.from_cells(cells)

        # Only df is consumed; temp is not
        assert dag.consumed_variables["a"] == {"df"}


class TestTopologicalSort:
    """Test topological sorting."""

    def test_topo_sort_simple(self):
        """Simple topological sort."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        assert dag.topological_order == ["a", "b", "c"]

    def test_topo_sort_diamond(self):
        """Topological sort of diamond DAG."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)

        order = dag.topological_order
        # a must come first
        assert order[0] == "a"
        # d must come last
        assert order[-1] == "d"
        # b and c can be in any order as long as they're between a and d
        assert set(order[1:3]) == {"b", "c"}

    def test_topo_sort_multiple_roots(self):
        """Topological sort with multiple roots."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=[]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x", "y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        order = dag.topological_order
        # c must come last
        assert order[-1] == "c"
        # a and b can be in any order
        assert set(order[:2]) == {"a", "b"}


class TestCycleDetection:
    """Test cycle detection."""

    def test_no_cycle_linear(self):
        """Linear chain has no cycle."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)

        cycles = dag.detect_cycles([c.id for c in cells])
        assert cycles == []

    def test_forward_reference_no_cycle(self):
        """A references y (defined later by B): no cycle, no edge.

        The single-pass DAG builder resolves each cell's references
        against producers available at its own position. A forward
        reference leaves the consuming cell without an upstream edge
        and the runtime surfaces a NameError, which is the honest
        signal. No error-style ValueError is raised at build time.
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
        """``x = x + 1`` with no upstream is valid intra-cell rebind.

        Old behavior raised "cycle detected"; the new model treats a
        reference-to-own-define as a pure rebind with no external
        producer. (The filter in analyze_cell strips ``x`` from
        references when it's a pure define, so this case rarely
        reaches NotebookDag.from_cells in practice.)
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
    """Test cascade planning."""

    def test_cascade_linear(self):
        """Cascade for target cell in linear chain."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["y"]),
        ]
        dag = NotebookDag.from_cells(cells)
        cell_ids = [c.id for c in cells]

        # To run c, need a and b
        plan = dag.cascade_plan("c", cell_ids)
        assert set(plan) == {"a", "b"}
        assert plan == ["a", "b"]  # In execution order

    def test_cascade_diamond(self):
        """Cascade for target cell in diamond DAG."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
            CellAnalysisWithId(id="c", defines=["z"], references=["x"]),
            CellAnalysisWithId(id="d", defines=["w"], references=["y", "z"]),
        ]
        dag = NotebookDag.from_cells(cells)
        cell_ids = [c.id for c in cells]

        # To run d, need a, b, c
        plan = dag.cascade_plan("d", cell_ids)
        assert set(plan) == {"a", "b", "c"}

    def test_cascade_root_cell(self):
        """Cascade for root cell needs only itself."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=["x"]),
        ]
        dag = NotebookDag.from_cells(cells)
        cell_ids = [c.id for c in cells]

        plan = dag.cascade_plan("a", cell_ids)
        assert plan == ["a"]

    def test_cascade_no_deps(self):
        """Cascade for cell with no upstream deps."""
        cells = [
            CellAnalysisWithId(id="a", defines=["x"], references=[]),
            CellAnalysisWithId(id="b", defines=["y"], references=[]),
        ]
        dag = NotebookDag.from_cells(cells)
        cell_ids = [c.id for c in cells]

        plan = dag.cascade_plan("b", cell_ids)
        assert plan == ["b"]


class TestRealWorldDAGs:
    """Test DAGs from real-world notebook patterns."""

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

        # Linear chain
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

        # features is shared by both train and evaluate
        assert dag.cell_downstream["features"] == ["train", "evaluate"]


class TestAfterEdges:
    """Tests for the ``@after`` ordering-only edge wiring.

    SQL cells whose dependency is on an upstream side effect (e.g.
    a setup cell that seeds a SQLite file) declare it via
    ``# @after <cell-id>``. The DAG edge participates in
    upstream/downstream wiring and the topological order without
    contributing a variable to ``consumed_variables``.
    """

    def test_after_creates_upstream_edge(self):
        cells = [
            CellAnalysisWithId(id="setup", defines=[], references=[]),
            CellAnalysisWithId(id="query", defines=[], references=[], after=["setup"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["query"] == ["setup"]
        assert dag.cell_downstream["setup"] == ["query"]
        assert dag.topological_order == ["setup", "query"]

    def test_after_edge_carries_no_variable(self):
        """Ordering-only — no variable, so consumed_variables stays empty."""
        cells = [
            CellAnalysisWithId(id="setup", defines=[], references=[]),
            CellAnalysisWithId(id="query", defines=[], references=[], after=["setup"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.consumed_variables["setup"] == set()
        assert all(e.variable == "" for e in dag.edges if e.from_cell_id == "setup")

    def test_after_dangling_id_silently_dropped(self):
        """Reference to a cell that doesn't exist — no edge, no
        crash. ``annotation_validation`` is the surface that flags
        this for the user; the DAG builder stays robust."""
        cells = [
            CellAnalysisWithId(id="query", defines=[], references=[], after=["does-not-exist"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["query"] == []
        assert dag.edges == []

    def test_after_self_reference_silently_dropped(self):
        cells = [
            CellAnalysisWithId(id="c", defines=[], references=[], after=["c"]),
        ]
        dag = NotebookDag.from_cells(cells)
        assert dag.cell_upstream["c"] == []

    def test_after_combines_with_variable_edges(self):
        """A cell can have both ``@after`` and variable references; the
        edges accumulate, no double-edge for the same upstream."""
        cells = [
            CellAnalysisWithId(id="setup", defines=[], references=[]),
            CellAnalysisWithId(id="config", defines=["min_value"], references=[]),
            CellAnalysisWithId(
                id="query",
                defines=[],
                references=["min_value"],
                after=["setup"],
            ),
        ]
        dag = NotebookDag.from_cells(cells)
        assert sorted(dag.cell_upstream["query"]) == ["config", "setup"]

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

    def test_inactive_variant_excluded_from_producer_map(self):
        """Downstream references resolve to the active variant only."""
        cells = [
            CellAnalysisWithId(id="load", defines=["X"], references=[]),
            CellAnalysisWithId(
                id="model_a",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="model_b",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="b",
            ),
            CellAnalysisWithId(id="post", defines=["score"], references=["preds"]),
        ]
        dag = NotebookDag.from_cells(cells, variant_active_selections={"model": "b"})

        assert dag.variable_producer["preds"] == "model_b"
        assert dag.cell_upstream["post"] == ["model_b"]
        assert "model_a" in dag.inactive_cells
        assert "model_b" not in dag.inactive_cells
        # Inactive variant has no edges and isn't a leaf/root
        assert dag.cell_upstream["model_a"] == []
        assert dag.cell_downstream["model_a"] == []
        assert "model_a" not in dag.leaves
        assert "model_a" not in dag.roots
        assert "model_a" not in dag.topological_order

    def test_first_in_source_order_when_toml_silent(self):
        """No selection in toml → first variant in source order is active."""
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
        dag = NotebookDag.from_cells(cells)

        assert dag.variable_producer["preds"] == "model_a"
        assert "model_b" in dag.inactive_cells

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

    def test_consumed_variables_only_for_active(self):
        """Inactive variants never appear as producers, so consumed_variables is empty for them."""
        cells = [
            CellAnalysisWithId(id="load", defines=["X"], references=[]),
            CellAnalysisWithId(
                id="model_a",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="model_b",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="b",
            ),
            CellAnalysisWithId(id="post", defines=[], references=["preds"]),
        ]
        dag = NotebookDag.from_cells(cells, variant_active_selections={"model": "a"})

        assert dag.consumed_variables["model_a"] == {"preds"}
        assert dag.consumed_variables["model_b"] == set()
        # The inactive variant's reference to X also doesn't drag load into consumed
        assert dag.consumed_variables["load"] == {"X"}

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

    def test_variant_groups_resolution_surface(self):
        """Resolved groups expose members in source order with the active flag."""
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
        dag = NotebookDag.from_cells(cells, variant_active_selections={"model": "b"})

        assert len(dag.variant_groups) == 1
        group = dag.variant_groups[0]
        assert group.group == "model"
        assert group.active_name == "b"
        assert group.active_cell_id == "model_b"
        assert group.members == [("model_a", "a"), ("model_b", "b")]

    def test_switching_variant_redirects_downstream(self):
        """Same cells, different active selection → producer map changes."""
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
            CellAnalysisWithId(id="post", defines=[], references=["preds"]),
        ]

        dag_a = NotebookDag.from_cells(cells, variant_active_selections={"model": "a"})
        dag_b = NotebookDag.from_cells(cells, variant_active_selections={"model": "b"})

        assert dag_a.cell_upstream["post"] == ["model_a"]
        assert dag_b.cell_upstream["post"] == ["model_b"]

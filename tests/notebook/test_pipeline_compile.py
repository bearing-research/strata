"""P0 tests for the notebook -> pipeline IR compiler.

Structural coverage on real example notebooks (Python, SQL, widget, prompt)
plus provenance-propagation properties. No cells are executed; the builder is
offline and read-only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from strata.notebook.compile import build_pipeline_ir_from_dir
from strata.notebook.compile.builder import PipelineCompileError

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _prov(ir) -> dict[str, str]:
    return {n.id: n.static_provenance for n in ir.nodes}


def _edges(ir) -> set[tuple[str, str, str]]:
    return {(e.from_node, e.to_node, e.variable) for e in ir.edges}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_iris_ir_structure():
    ir = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification")

    node_ids = [n.id for n in ir.nodes]
    # All seven Python cells are nodes; the markdown cell is dropped, not listed.
    assert node_ids == [
        "load-data",
        "explore-stats",
        "scatter-plot",
        "train-test",
        "train-model",
        "evaluate",
        "confusion",
    ]
    assert "train-intro" not in node_ids
    assert all(u.id != "train-intro" for u in ir.unsupported)

    # Default runtime is container -> every Python cell is a container node.
    assert ir.runtime == "container"
    assert {n.compute_target for n in ir.nodes} == {"container"}
    assert ir.topological_order == node_ids
    assert ir.unsupported == []
    assert ir.parameters == []
    assert ir.warnings == []

    # Name comes from the @name annotation.
    load = next(n for n in ir.nodes if n.id == "load-data")
    assert load.name == "Load iris dataset"
    assert {o.variable for o in load.outputs} == {"df", "feature_names"}
    assert {o.artifact_path for o in load.outputs} == {
        "load-data/df.artifact",
        "load-data/feature_names.artifact",
    }

    # A downstream node resolves its inputs to the producing nodes' outputs.
    evaluate = next(n for n in ir.nodes if n.id == "evaluate")
    ev_inputs = {(i.variable, i.from_node, i.artifact_path) for i in evaluate.inputs}
    assert ("model", "train-model", "train-model/model.artifact") in ev_inputs
    assert ("X_test", "train-test", "train-test/X_test.artifact") in ev_inputs
    assert ("y_test", "train-test", "train-test/y_test.artifact") in ev_inputs

    edges = _edges(ir)
    assert ("load-data", "train-test", "df") in edges
    assert ("train-model", "evaluate", "model") in edges
    assert ("evaluate", "confusion", "y_pred") in edges


def test_sql_read_cells_run_connection_faithfully():
    # SQL nodes only compile under the glue runtime today.
    ir = build_pipeline_ir_from_dir(EXAMPLES / "sql_orders_report", runtime="glue")
    by_id = {n.id: n for n in ir.nodes}

    # Read SQL cells run against their own engine (sqlite) in a Glue job, not
    # Athena — that is the parity-faithful target.
    for sql_id in ("top-orders", "category-summary"):
        node = by_id[sql_id]
        assert node.kind == "sql"
        assert node.compute_target == "glue_sql"
        assert node.connection == "warehouse"
        assert node.connection_config and node.connection_config["driver"] == "sqlite"

    # The write SQL cell (seed) mutates the warehouse -> not compiled in v1.
    assert "seed" not in by_id
    assert any(u.id == "seed" and u.language == "sql" for u in ir.unsupported)

    for py_id in ("threshold", "report"):
        node = by_id[py_id]
        assert node.kind == "python"
        assert node.compute_target == "glue_python_shell"
        assert node.connection is None


def test_widget_controls_become_parameters():
    ir = build_pipeline_ir_from_dir(EXAMPLES / "widget_playground")

    param_names = {p.name for p in ir.parameters}
    assert {"alpha", "n", "curve"} <= param_names
    # Widget cells are parameters, not nodes.
    node_ids = {n.id for n in ir.nodes}
    assert "controls" not in node_ids
    # A node that reads a control records it as a parameter input.
    consumers = [n for n in ir.nodes if any(i.from_param for i in n.inputs)]
    assert consumers, "expected at least one node consuming a widget parameter"


def test_prompt_cell_is_unsupported_not_dropped():
    ir = build_pipeline_ir_from_dir(EXAMPLES / "review_triage")

    unsupported = {u.id: u for u in ir.unsupported}
    assert "triage" in unsupported
    assert unsupported["triage"].language == "prompt"
    # The prompt cell is not silently a node.
    assert all(n.id != "triage" for n in ir.nodes)
    # A downstream node that consumed it is warned about, not left dangling.
    assert any("triage" in w for w in ir.warnings)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_is_deterministic():
    a = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification")
    b = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification")
    assert a.model_dump(mode="json") == b.model_dump(mode="json")


def _copy_example(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_editing_a_leaf_changes_only_that_node(tmp_path):
    nb = _copy_example(EXAMPLES / "iris_classification", tmp_path / "nb")
    before = _prov(build_pipeline_ir_from_dir(nb))

    # confusion is a leaf; a semantic edit must not touch any other node.
    leaf = nb / "cells" / "confusion.py"
    leaf.write_text(leaf.read_text() + "\npipeline_edit_marker = 123\n", encoding="utf-8")
    after = _prov(build_pipeline_ir_from_dir(nb))

    assert after["confusion"] != before["confusion"]
    for node_id, digest in before.items():
        if node_id != "confusion":
            assert after[node_id] == digest, f"{node_id} provenance changed unexpectedly"


def test_editing_a_root_propagates_downstream(tmp_path):
    nb = _copy_example(EXAMPLES / "iris_classification", tmp_path / "nb")
    before = _prov(build_pipeline_ir_from_dir(nb))

    root = nb / "cells" / "load_data.py"
    root.write_text(root.read_text() + "\npipeline_edit_marker = 456\n", encoding="utf-8")
    after = _prov(build_pipeline_ir_from_dir(nb))

    # The edited root and a transitively-downstream node both shift.
    assert after["load-data"] != before["load-data"]
    assert after["evaluate"] != before["evaluate"]
    assert after["confusion"] != before["confusion"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_missing_dag_raises(tmp_path):
    # A cycle or variant collision leaves the session with no DAG (dag=None).
    # The compiler must refuse rather than emit a partial pipeline.
    from strata.notebook.compile.builder import build_pipeline_ir
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    nb = tmp_path / "nb"
    (nb / "cells").mkdir(parents=True)
    (nb / "notebook.toml").write_text(
        'notebook_id = "nb-001"\n'
        'name = "NB"\n'
        "cells = [\n"
        '    { id = "a", file = "a.py", language = "python", order = 0 },\n'
        "]\n",
        encoding="utf-8",
    )
    (nb / "cells" / "a.py").write_text("x = 1\n", encoding="utf-8")

    state = parse_notebook(nb)
    session = NotebookSession(state, nb)
    session.dag = None

    with pytest.raises(PipelineCompileError):
        build_pipeline_ir(session)

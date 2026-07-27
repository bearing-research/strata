"""The @node frontend: a non-notebook producer of the SAME pipeline IR.

Proves the engine (``strata.pipeline``) is frontend-agnostic. A plain Python
module of ``@node`` functions lowers to a :class:`PipelineIR` that the identical
renderers consume, and whose lowered node sources actually run end-to-end
through the same runtime the notebook pipeline uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import pytest

from strata.notebook.serializer import deserialize_value
from strata.pipeline import (
    build_pipeline_ir_from_functions,
    node,
    render_state_machine,
    write_bundle,
)
from strata.pipeline.authoring import PipelineCompileError
from strata.pipeline.runtime import run_python_node

_LAMBDA = "arn:aws:states:::lambda:invoke"


# A three-node pipeline authored as plain functions — no notebook involved.
@node
def load():
    values = [1, 2, 3, 4]
    return {"values": values}


@node
def scale(values):
    scaled = [v * 10 for v in values]
    return {"scaled": scaled}


@node(name="reduce-sum")
def total(scaled):
    return {"result": sum(scaled)}


PIPE = [load, scale, total]


def _read_artifact(uri: str, scratch: Path):
    with fsspec.open(uri + ".meta.json", "rb") as fh:
        content_type = json.loads(fh.read())["content_type"]
    local = scratch / "read"
    with fsspec.open(uri, "rb") as fh:
        local.write_bytes(fh.read())
    return deserialize_value(content_type, local)


def test_node_module_builds_wired_ir():
    ir = build_pipeline_ir_from_functions(PIPE, name="Scale Demo")

    assert ir.runtime == "container"
    assert ir.pipeline_name == "Scale Demo"
    assert ir.topological_order == ["load", "scale", "total"]

    by_id = {n.id: n for n in ir.nodes}
    # Inputs wired by parameter name -> the upstream node's output of that name.
    assert [i.variable for i in by_id["scale"].inputs] == ["values"]
    assert by_id["scale"].inputs[0].from_node == "load"
    assert by_id["total"].inputs[0].from_node == "scale"
    # Only consumed outputs persist; the terminal result is consumed by nobody.
    assert [o.variable for o in by_id["load"].outputs] == ["values"]
    assert [o.variable for o in by_id["scale"].outputs] == ["scaled"]
    assert by_id["total"].outputs == []
    # Display name comes from @node(name=...).
    assert by_id["total"].name == "reduce-sum"
    # Every node carries a compile-time provenance preview.
    assert all(n.static_provenance for n in ir.nodes)


def test_node_ir_renders_through_the_same_engine(tmp_path):
    ir = build_pipeline_ir_from_functions(PIPE, name="Scale Demo")

    # The unchanged ASL renderer produces a well-formed Lambda state machine.
    sm = render_state_machine(ir)
    assert sm["StartAt"] == "load"
    assert sm["States"]["load"]["Resource"] == _LAMBDA
    assert sm["States"]["load"]["Next"] == "scale"
    assert sm["States"]["scale"]["Next"] == "total"
    assert sm["States"]["total"]["End"] is True

    # The unchanged bundle writer emits a full container bundle.
    written = write_bundle(ir, tmp_path / "bundle")
    names = {Path(f).name for f in written}
    assert {"Dockerfile", "handler.py", "nodes.json", "statemachine.asl.json"} <= names


def test_node_sources_run_end_to_end_through_the_runtime(tmp_path):
    """The lowered node sources execute faithfully through the same runtime."""
    ir = build_pipeline_ir_from_functions(PIPE, name="Scale Demo")
    src = {n.id: n.source for n in ir.nodes}

    values_uri = str(tmp_path / "load_values")
    scaled_uri = str(tmp_path / "scale_scaled")
    result_uri = str(tmp_path / "total_result")

    run_python_node(
        source=src["load"], inputs={}, outputs={"values": values_uri}, workdir=tmp_path / "wl"
    )
    run_python_node(
        source=src["scale"],
        inputs={"values": values_uri},
        outputs={"scaled": scaled_uri},
        workdir=tmp_path / "ws",
    )
    run_python_node(
        source=src["total"],
        inputs={"scaled": scaled_uri},
        outputs={"result": result_uri},
        workdir=tmp_path / "wt",
    )

    assert _read_artifact(values_uri, tmp_path) == [1, 2, 3, 4]
    assert _read_artifact(result_uri, tmp_path) == 100  # sum(v*10 for v in 1..4)


def test_functions_stay_callable_locally():
    """@node leaves the function callable, so the module also runs as plain code."""
    assert load() == {"values": [1, 2, 3, 4]}
    assert scale([1, 2]) == {"scaled": [10, 20]}


def test_undecorated_function_rejected():
    def bare():
        return {"x": 1}

    with pytest.raises(PipelineCompileError, match="not decorated"):
        build_pipeline_ir_from_functions([bare], name="x")


def test_missing_return_dict_rejected():
    @node
    def no_return(values):
        print(values)

    with pytest.raises(PipelineCompileError, match="must end with"):
        build_pipeline_ir_from_functions([no_return], name="x")


def test_forward_reference_rejected():
    @node
    def consumer(produced):
        return {"out": produced}

    @node
    def producer():
        return {"produced": 1}

    # Listed out of dependency order.
    with pytest.raises(PipelineCompileError, match="dependency order"):
        build_pipeline_ir_from_functions([consumer, producer], name="x")


def test_duplicate_output_rejected():
    @node
    def a():
        return {"dup": 1}

    @node
    def b():
        return {"dup": 2}

    with pytest.raises(PipelineCompileError, match="unique across nodes"):
        build_pipeline_ir_from_functions([a, b], name="x")

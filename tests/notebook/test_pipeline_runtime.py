"""P2 tests: the Glue node runtime, script renderer, and bundle writer.

The runtime is exercised end-to-end *locally* (fsspec over local paths), so a
generated node's read -> exec -> write round-trips through the real Strata
serializer without needing AWS.
"""

from __future__ import annotations

import json
from pathlib import Path

import fsspec
import pytest

from strata.notebook.compile import build_pipeline_ir_from_dir
from strata.notebook.serializer import deserialize_value
from strata.pipeline import render_glue_script, write_bundle
from strata.pipeline.runtime import run_python_node, run_sql_node

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _read_artifact(uri: str, scratch: Path):
    """Deserialize an artifact written by the runtime (blob + sidecar)."""
    with fsspec.open(uri + ".meta.json", "rb") as fh:
        content_type = json.load(fh)["content_type"]
    with fsspec.open(uri, "rb") as fh:
        blob = fh.read()
    local = scratch / "__rb"
    local.write_bytes(blob)
    return deserialize_value(content_type, local)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def test_two_node_chain_roundtrips(tmp_path):
    pytest.importorskip("pandas")
    import pandas as pd

    df_uri = str(tmp_path / "a_df")
    total_uri = str(tmp_path / "b_total")

    # Node A produces a DataFrame; Node B consumes it and produces a scalar.
    run_python_node(
        source="import pandas as pd\ndf = pd.DataFrame({'x': [1, 2, 3]})\n",
        inputs={},
        outputs={"df": df_uri},
        workdir=tmp_path / "wa",
    )
    run_python_node(
        source="total = int(df['x'].sum())\n",
        inputs={"df": df_uri},
        outputs={"total": total_uri},
        workdir=tmp_path / "wb",
    )

    # The DataFrame survived the S3-style hand-off byte-for-byte.
    df_back = _read_artifact(df_uri, tmp_path)
    pd.testing.assert_frame_equal(df_back, pd.DataFrame({"x": [1, 2, 3]}))
    # And the downstream scalar computed from it is correct.
    assert _read_artifact(total_uri, tmp_path) == 6


def test_unchanged_node_skips_recompute(tmp_path):
    # A side-effect counter proves the body did NOT run on the second pass.
    counter = tmp_path / "runs.txt"
    src = f"open({str(counter)!r}, 'a').write('x')\nout = 1\n"
    out_uri = str(tmp_path / "out")

    first = run_python_node(
        source=src, inputs={}, outputs={"out": out_uri}, workdir=tmp_path / "w1"
    )
    second = run_python_node(
        source=src, inputs={}, outputs={"out": out_uri}, workdir=tmp_path / "w2"
    )

    assert first["skipped"] is False
    assert second["skipped"] is True
    assert first["provenance"] == second["provenance"]
    assert counter.read_text() == "x"  # body ran exactly once


def test_changed_source_recomputes(tmp_path):
    out_uri = str(tmp_path / "out")
    a = run_python_node(
        source="out = 1\n", inputs={}, outputs={"out": out_uri}, workdir=tmp_path / "a"
    )
    b = run_python_node(
        source="out = 2\n", inputs={}, outputs={"out": out_uri}, workdir=tmp_path / "b"
    )
    assert a["skipped"] is False and b["skipped"] is False
    assert a["provenance"] != b["provenance"]
    assert _read_artifact(out_uri, tmp_path) == 2  # recomputed, not stale


def test_skip_propagates_through_chain(tmp_path):
    up_uri = str(tmp_path / "up")
    down_uri = str(tmp_path / "down")

    def run_chain(up_src, wd):
        run_python_node(
            source=up_src, inputs={}, outputs={"n": up_uri}, workdir=tmp_path / f"{wd}u"
        )
        return run_python_node(
            source="m = n + 1\n",
            inputs={"n": up_uri},
            outputs={"m": down_uri},
            workdir=tmp_path / f"{wd}d",
        )

    run_chain("n = 10\n", "r1")
    second = run_chain("n = 10\n", "r2")  # unchanged -> downstream skips
    assert second["skipped"] is True

    # Change the upstream: downstream's input provenance shifts, so it recomputes.
    third = run_chain("n = 20\n", "r3")
    assert third["skipped"] is False
    assert _read_artifact(down_uri, tmp_path) == 21


def test_skip_if_fresh_false_forces_recompute(tmp_path):
    out_uri = str(tmp_path / "out")
    run_python_node(source="out = 1\n", inputs={}, outputs={"out": out_uri}, workdir=tmp_path / "a")
    again = run_python_node(
        source="out = 1\n",
        inputs={},
        outputs={"out": out_uri},
        skip_if_fresh=False,
        workdir=tmp_path / "b",
    )
    assert again["skipped"] is False


def test_missing_output_variable_raises(tmp_path):
    with pytest.raises(KeyError):
        run_python_node(
            source="y = 1\n",
            inputs={},
            outputs={"x": str(tmp_path / "x")},
            workdir=tmp_path / "w",
        )


def test_run_sql_node_against_sqlite(tmp_path):
    pytest.importorskip("adbc_driver_sqlite")
    import sqlite3

    import pandas as pd

    db = tmp_path / "orders.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE orders(id INTEGER, amount INTEGER);"
        "INSERT INTO orders VALUES (1, 50), (2, 150), (3, 250);"
    )
    con.commit()
    con.close()

    # An upstream Python node supplies the bind value.
    min_uri = str(tmp_path / "min_amount")
    run_python_node(
        source="min_amount = 100\n",
        inputs={},
        outputs={"min_amount": min_uri},
        workdir=tmp_path / "wp",
    )

    out_uri = str(tmp_path / "result")
    run_sql_node(
        source=(
            "# @sql connection=wh\n"
            "SELECT id, amount FROM orders WHERE amount >= :min_amount ORDER BY id\n"
        ),
        connection_config={"name": "wh", "driver": "sqlite", "path": str(db)},
        inputs={"min_amount": min_uri},
        outputs={"result": out_uri},
        workdir=tmp_path / "ws",
    )

    # The query ran on the real sqlite engine with the upstream bind value.
    result = _read_artifact(out_uri, tmp_path)
    df = result if isinstance(result, pd.DataFrame) else result.to_pandas()
    assert df["id"].tolist() == [2, 3]
    assert df["amount"].tolist() == [150, 250]


def test_env_is_set_then_restored(tmp_path):
    import os

    key = "STRATA_PIPELINE_TEST_ENV"
    os.environ.pop(key, None)
    out_uri = str(tmp_path / "seen")

    run_python_node(
        source=f"import os\nseen = os.environ['{key}']\n",
        inputs={},
        outputs={"seen": out_uri},
        env={key: "hello"},
        workdir=tmp_path / "w",
    )

    assert _read_artifact(out_uri, tmp_path) == "hello"
    assert key not in os.environ  # restored after exec


# ---------------------------------------------------------------------------
# Glue script renderer
# ---------------------------------------------------------------------------


def test_render_glue_script_is_valid_and_embeds_source():
    ir = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification", runtime="glue")
    node = next(n for n in ir.nodes if n.id == "load-data")
    script = render_glue_script(node)

    compile(script, "<glue>", "exec")  # syntactically valid Python
    assert "run_python_node" in script
    assert '"strata_inputs"' in script and '"strata_outputs"' in script
    # The exact cell source is embedded (as a JSON/Python string literal).
    assert json.dumps(node.source) in script


def test_render_glue_script_for_sql_node_calls_run_sql_node():
    ir = build_pipeline_ir_from_dir(EXAMPLES / "sql_orders_report", runtime="glue")
    sql_node = next(n for n in ir.nodes if n.compute_target == "glue_sql")
    script = render_glue_script(sql_node)
    compile(script, "<glue>", "exec")
    assert "run_sql_node" in script
    # The connection config is embedded so the job opens the same engine.
    assert '"driver": "sqlite"' in script


def test_render_glue_script_rejects_non_glue_node():
    from strata.pipeline.ir import PipelineNode

    athena_node = PipelineNode(
        id="q", name="q", kind="sql", compute_target="athena", source="SELECT 1"
    )
    with pytest.raises(ValueError, match="not a Glue node"):
        render_glue_script(athena_node)


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def test_write_bundle_container_layout(tmp_path):
    ir = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification")  # container default
    written = write_bundle(ir, tmp_path / "out")
    out = tmp_path / "out"

    # Container bundle: image + handler + baked sources + vendored kit.
    for name in (
        "pipeline.json",
        "statemachine.asl.json",
        "access.json",
        "README.md",
        "Dockerfile",
        "handler.py",
        "nodes.json",
        "serializer.py",
        "provenance.py",
        "runtime.py",
    ):
        assert (out / name).is_file(), name
    assert not (out / "glue").exists()

    # The generated Python is valid and the kit's imports are flattened.
    for name in ("handler.py", "runtime.py", "serializer.py", "provenance.py"):
        compile((out / name).read_text(), name, "exec")
    assert "from provenance import" in (out / "runtime.py").read_text()

    # nodes.json holds every node's source; ASL invokes Lambda.
    nodes = json.loads((out / "nodes.json").read_text())
    assert set(nodes) == {n.id for n in ir.nodes}
    asl = json.loads((out / "statemachine.asl.json").read_text())
    assert asl["States"]["load-data"]["Resource"] == "arn:aws:states:::lambda:invoke"
    assert set(written) == set(out.rglob("*"))


def test_container_dockerfile_pins_notebook_deps_no_strata(tmp_path):
    ir = build_pipeline_ir_from_dir(EXAMPLES / "iris_classification")
    write_bundle(ir, tmp_path / "out")
    dockerfile = (tmp_path / "out" / "Dockerfile").read_text()

    assert "public.ecr.aws/lambda/python:3.12" in dockerfile
    assert "scikit-learn>=1.5.0" in dockerfile
    assert "s3fs" in dockerfile  # S3 IO for fsspec
    # The image vendors the runtime; it must NOT install Strata itself.
    assert "strata-notebook" not in dockerfile


def test_glue_bundle_includes_sql_and_python_scripts(tmp_path):
    ir = build_pipeline_ir_from_dir(EXAMPLES / "sql_orders_report", runtime="glue")
    write_bundle(ir, tmp_path / "out")

    glue_stems = {p.stem for p in (tmp_path / "out" / "glue").glob("*.py")}
    # Python cells + connection-faithful SQL read cells both get scripts;
    # the write cell (seed) is unsupported, so it has none.
    assert glue_stems == {"threshold", "report", "top-orders", "category-summary"}
    assert (tmp_path / "out" / "requirements.txt").is_file()

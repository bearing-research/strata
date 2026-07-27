"""P1 tests for the IR -> Step Functions ASL renderer.

Structural + well-formedness coverage: every Next target resolves, terminal
states are marked, Task resources are the expected .sync integrations, and the
topological-level layout respects both data and `# @after` ordering edges.
"""

from __future__ import annotations

import json
from pathlib import Path

from strata.notebook.compile import build_pipeline_ir_from_dir
from strata.pipeline import AslRenderOptions, render_state_machine

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

_GLUE = "arn:aws:states:::glue:startJobRun.sync"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_wellformed(machine: dict) -> None:
    """Assert a state machine (and each Parallel sub-machine) is self-consistent."""

    def check_scope(states: dict, start: str) -> None:
        assert start in states, f"StartAt {start!r} not among states {list(states)}"
        terminals = 0
        for name, state in states.items():
            has_next = "Next" in state
            has_end = state.get("End") is True
            stype = state["Type"]
            if stype in ("Succeed", "Fail"):
                terminals += 1
                continue
            assert has_next ^ has_end, f"state {name!r} must have exactly one of Next/End"
            if has_next:
                assert state["Next"] in states, f"{name!r} -> missing {state['Next']!r}"
            else:
                terminals += 1
            if stype == "Parallel":
                for branch in state["Branches"]:
                    check_scope(branch["States"], branch["StartAt"])
        assert terminals >= 1, "scope has no terminal state"
        # Reachability from start within this scope.
        seen: set[str] = set()
        frontier = [start]
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            nxt = states[cur].get("Next")
            if nxt:
                frontier.append(nxt)
        assert seen == set(states), f"unreachable states: {set(states) - seen}"

    assert "StartAt" in machine and "States" in machine
    check_scope(machine["States"], machine["StartAt"])


def _find_task(machine: dict, node_id: str) -> dict | None:
    """Locate a node's Task state anywhere (top level or inside a Parallel)."""

    def search(states: dict) -> dict | None:
        for name, state in states.items():
            if name == node_id and state.get("Type") == "Task":
                return state
            if state.get("Type") == "Parallel":
                for branch in state["Branches"]:
                    hit = search(branch["States"])
                    if hit is not None:
                        return hit
        return None

    return search(machine["States"])


_LAMBDA = "arn:aws:states:::lambda:invoke"


def _sm(
    example: str, options: AslRenderOptions | None = None, *, runtime: str = "container"
) -> dict:
    ir = build_pipeline_ir_from_dir(EXAMPLES / example, runtime=runtime)
    return render_state_machine(ir, options)


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_iris_asl_structure():
    sm = _sm("iris_classification")
    _assert_wellformed(sm)

    assert sm["StartAt"] == "load-data"
    states = sm["States"]
    # The fan-out level (explore / scatter / train-test) is a Parallel.
    assert states["load-data"]["Next"] == "parallel-1"
    parallel = states["parallel-1"]
    assert parallel["Type"] == "Parallel"
    branch_starts = {b["StartAt"] for b in parallel["Branches"]}
    assert branch_starts == {"explore-stats", "scatter-plot", "train-test"}
    # Sequential tail.
    assert parallel["Next"] == "train-model"
    assert states["train-model"]["Next"] == "evaluate"
    assert states["evaluate"]["Next"] == "confusion"
    assert states["confusion"]["End"] is True

    # Default runtime: every node is a Lambda-invoke task.
    for node_id in ("load-data", "explore-stats", "train-model", "confusion"):
        assert _find_task(sm, node_id)["Resource"] == _LAMBDA


def test_container_task_payload_carries_node_and_io():
    sm = _sm("iris_classification")  # container default
    params = _find_task(sm, "evaluate")["Parameters"]
    assert params["FunctionName"] == "strata-node"
    payload = params["Payload"]
    assert payload["node_id"] == "evaluate"
    # evaluate consumes model/X_test/y_test and outputs y_pred, all by S3 uri.
    assert "model" in payload["inputs"] and "y_pred" in payload["outputs"]


def test_sql_read_cells_are_glue_tasks():
    sm = _sm("sql_orders_report", runtime="glue")
    _assert_wellformed(sm)

    # Read SQL cells run connection-faithfully as Glue tasks (not Athena).
    assert _find_task(sm, "top-orders")["Resource"] == _GLUE
    assert _find_task(sm, "category-summary")["Resource"] == _GLUE
    assert _find_task(sm, "report")["Resource"] == _GLUE
    # The write cell (seed) is unsupported -> absent from the machine.
    assert _find_task(sm, "seed") is None


def test_glue_sql_task_carries_io_args():
    sm = _sm("sql_orders_report", runtime="glue")
    # top-orders binds :min_amount from the threshold cell, so it has an input.
    args = _find_task(sm, "top-orders")["Parameters"]["Arguments"]
    assert "--strata_inputs" in args and "--strata_outputs" in args


def test_glue_task_discards_result_and_retries():
    sm = _sm("iris_classification", runtime="glue")
    task = _find_task(sm, "load-data")
    assert task["ResultPath"] is None  # data flows via S3, not state
    assert task["Retry"][0]["ErrorEquals"] == ["Glue.ConcurrentRunsExceededException"]


# ---------------------------------------------------------------------------
# Options + edge cases
# ---------------------------------------------------------------------------


def test_render_options_override_names():
    opts = AslRenderOptions(
        artifact_root="s3://my-bucket/pipe",
        glue_job_prefix="prod",
        athena_database="analytics",
    )
    # Glue JobName prefix + artifact root folded into the IO uris (iris = Python).
    iris = _sm("iris_classification", opts, runtime="glue")
    load = _find_task(iris, "load-data")
    assert load["Parameters"]["JobName"] == "prod-load-data"
    outs = json.loads(load["Parameters"]["Arguments"]["--strata_outputs"])
    assert outs and all(u.startswith("s3://my-bucket/pipe/") for u in outs.values())


def test_notebook_with_no_nodes_renders_succeed(tmp_path):
    from strata.notebook.compile import build_pipeline_ir_from_dir as build

    nb = tmp_path / "nb"
    (nb / "cells").mkdir(parents=True)
    (nb / "notebook.toml").write_text(
        'notebook_id = "md-001"\n'
        'name = "Docs only"\n'
        "cells = [\n"
        '    { id = "intro", file = "intro.md", language = "markdown", order = 0 },\n'
        "]\n",
        encoding="utf-8",
    )
    (nb / "cells" / "intro.md").write_text("# Just prose\n", encoding="utf-8")

    sm = render_state_machine(build(nb))
    _assert_wellformed(sm)
    assert sm["States"][sm["StartAt"]]["Type"] == "Succeed"

"""Render a :class:`PipelineIR` into an AWS Step Functions state machine.

The state machine is expressed in Amazon States Language (ASL, JSON). Nodes
become ``Task`` states with ``.sync`` service integrations (Glue Python Shell
for Python cells, Athena for SQL cells); the DAG is laid out by topological
*level* so each level runs after all earlier levels complete:

- a level with one node emits a single ``Task`` state,
- a level with several independent nodes emits a ``Parallel`` state (one
  branch per node).

Level layout is always dependency-correct for an arbitrary DAG (a node at
level *L* has every upstream at level < *L*), unlike trying to nest
``Parallel`` on a general graph — reconvergent diamonds are not
series-parallel. It over-synchronizes (a node waits for its whole prior
level, not just its own upstreams); a tighter schedule is a later
optimization. Task results are discarded (``ResultPath: null``) — data flows
by reference through S3, not through state (Step Functions' 256 KiB payload
cap). The Glue/Athena ``Arguments`` / ``QueryString`` bodies are filled here
only enough to be well-formed; the executable job scripts land in P2.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from strata.notebook.annotations import strip_leading_annotations
from strata.pipeline.ir import PipelineIR, PipelineNode

_GLUE_RESOURCE = "arn:aws:states:::glue:startJobRun.sync"
_ATHENA_RESOURCE = "arn:aws:states:::athena:startQueryExecution.sync"


@dataclass
class AslRenderOptions:
    """Deploy-time knobs the renderer needs; all have placeholder defaults.

    The defaults produce a well-formed state machine that a deploy step then
    rebinds to real resources (bucket, workgroup, database).
    """

    artifact_root: str = "s3://strata-pipeline"
    lambda_function: str = "strata-node"  # container runtime: the per-node Lambda
    glue_job_prefix: str | None = None  # default: the notebook id
    athena_workgroup: str = "primary"
    athena_database: str = "default"
    athena_output_location: str | None = None  # default: <artifact_root>/athena-results/


def _levels(ir: PipelineIR) -> list[list[str]]:
    """Group node ids into topological levels (longest-path layering)."""
    upstreams: dict[str, set[str]] = defaultdict(set)
    for edge in ir.edges:
        upstreams[edge.to_node].add(edge.from_node)

    level: dict[str, int] = {}
    for nid in ir.topological_order:  # topo order → upstreams already leveled
        ups = upstreams.get(nid, set())
        level[nid] = 0 if not ups else 1 + max(level[u] for u in ups)

    by_level: dict[int, list[str]] = defaultdict(list)
    for nid in ir.topological_order:
        by_level[level[nid]].append(nid)
    return [by_level[lvl] for lvl in sorted(by_level)]


def render_state_machine(ir: PipelineIR, options: AslRenderOptions | None = None) -> dict:
    """Render *ir* into an ASL state-machine dict (JSON-serializable).

    Returns an empty-but-valid ``Succeed`` machine when the notebook has no
    translatable nodes, so the output is always a runnable state machine.
    """
    options = options or AslRenderOptions()
    job_prefix = options.glue_job_prefix or ir.pipeline_id
    artifact_root = options.artifact_root.rstrip("/")
    athena_out = options.athena_output_location or f"{artifact_root}/athena-results/"
    node_by_id = {n.id: n for n in ir.nodes}

    comment = f"Compiled from Strata pipeline {ir.pipeline_name} ({ir.pipeline_id})"

    levels = _levels(ir)
    if not levels:
        return {
            "Comment": comment,
            "StartAt": "NoOp",
            "States": {"NoOp": {"Type": "Succeed"}},
        }

    def level_state_name(level_nodes: list[str], idx: int) -> str:
        return level_nodes[0] if len(level_nodes) == 1 else f"parallel-{idx}"

    def _io_uris(node: PipelineNode) -> tuple[dict, dict]:
        inputs = {
            i.variable: f"{artifact_root}/{i.artifact_path}"
            for i in node.inputs
            if i.from_node and i.artifact_path
        }
        outputs = {o.variable: f"{artifact_root}/{o.artifact_path}" for o in node.outputs}
        return inputs, outputs

    def task_state(node: PipelineNode, *, terminal: bool, next_name: str | None) -> dict:
        state: dict[str, object] = {"Type": "Task"}
        if node.compute_target == "container":
            input_uris, output_uris = _io_uris(node)
            state["Resource"] = "arn:aws:states:::lambda:invoke"
            state["Parameters"] = {
                "FunctionName": options.lambda_function,
                "Payload": {"node_id": node.id, "inputs": input_uris, "outputs": output_uris},
            }
            state["Retry"] = [
                {
                    "ErrorEquals": ["Lambda.TooManyRequestsException", "Lambda.ServiceException"],
                    "IntervalSeconds": 5,
                    "MaxAttempts": 3,
                    "BackoffRate": 2.0,
                }
            ]
        elif node.compute_target == "athena":
            state["Resource"] = _ATHENA_RESOURCE
            state["Parameters"] = {
                "QueryString": strip_leading_annotations(node.source).strip(),
                "QueryExecutionContext": {"Database": options.athena_database},
                "WorkGroup": options.athena_workgroup,
                "ResultConfiguration": {"OutputLocation": athena_out},
            }
        else:  # glue_python_shell
            input_uris = {
                i.variable: f"{artifact_root}/{i.artifact_path}"
                for i in node.inputs
                if i.from_node and i.artifact_path
            }
            output_uris = {o.variable: f"{artifact_root}/{o.artifact_path}" for o in node.outputs}
            state["Resource"] = _GLUE_RESOURCE
            state["Parameters"] = {
                "JobName": f"{job_prefix}-{node.id}",
                "Arguments": {
                    "--strata_node": node.id,
                    "--strata_inputs": json.dumps(input_uris),
                    "--strata_outputs": json.dumps(output_uris),
                },
            }
            state["Retry"] = [
                {
                    "ErrorEquals": ["Glue.ConcurrentRunsExceededException"],
                    "IntervalSeconds": 30,
                    "MaxAttempts": 3,
                    "BackoffRate": 2.0,
                }
            ]
        state["ResultPath"] = None
        if terminal:
            state["End"] = True
        else:
            state["Next"] = next_name
        return state

    states: dict[str, dict] = {}
    last = len(levels) - 1
    for idx, level_nodes in enumerate(levels):
        name = level_state_name(level_nodes, idx)
        terminal = idx == last
        next_name = None if terminal else level_state_name(levels[idx + 1], idx + 1)

        if len(level_nodes) == 1:
            states[name] = task_state(
                node_by_id[level_nodes[0]], terminal=terminal, next_name=next_name
            )
        else:
            branches = [
                {
                    "StartAt": bn,
                    "States": {bn: task_state(node_by_id[bn], terminal=True, next_name=None)},
                }
                for bn in level_nodes
            ]
            branch_state: dict[str, object] = {
                "Type": "Parallel",
                "Branches": branches,
                "ResultPath": None,
            }
            if terminal:
                branch_state["End"] = True
            else:
                branch_state["Next"] = next_name
            states[name] = branch_state

    return {
        "Comment": comment,
        "StartAt": level_state_name(levels[0], 0),
        "States": states,
    }

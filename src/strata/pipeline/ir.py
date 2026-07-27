"""The Pipeline IR — a serializable description of a compiled notebook DAG.

One :class:`PipelineNode` per translatable cell; edges and per-node
inputs/outputs carry the upstream variables as artifacts passed by
reference (an S3 Parquet path in the AWS target). Widget cells become
:class:`PipelineParameter` entries; prompt/R cells (deferred past v1) and
sweep producers are recorded as :class:`UnsupportedCell` / ``warnings`` so
the IR is honest about what did not translate rather than silently dropping
it.

The models are plain data (Pydantic for stable ``model_dump(mode="json")``);
all construction logic lives in ``builder.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class NodeInput(BaseModel):
    """One upstream value a node consumes, resolved to its producer.

    ``from_node`` is the producer node id (empty when the value comes from a
    pipeline parameter, in which case ``from_param`` names it).
    ``artifact_path`` is the producer's output path for ``variable`` — the
    by-reference handle the running step reads (bucket prefixed at deploy).
    """

    variable: str
    from_node: str = ""
    from_param: str | None = None
    artifact_path: str = ""


class NodeOutput(BaseModel):
    """One variable a node produces that a downstream node consumes.

    ``artifact_path`` is where the step writes ``variable`` (relative;
    the deploy layer prefixes the S3 bucket/root).
    """

    variable: str
    artifact_path: str


class NodeResources(BaseModel):
    """Per-node compute hints, carried from ``@worker`` / ``@timeout``."""

    timeout: float | None = None
    worker: str | None = None


class MountRef(BaseModel):
    """An external data/model mount a node reads or writes (``@mount``).

    ``option_keys`` lists the fsspec storage-option *keys* only — values are
    omitted because they can carry credentials (endpoint tokens, keys).
    """

    name: str
    uri: str
    scheme: str  # file | s3 | gs | az
    mode: str  # ro | rw
    option_keys: list[str] = Field(default_factory=list)


class TableRef(BaseModel):
    """An Iceberg table a node scans (``@table``).

    ``snapshot_pin`` is the crux of experiment↔production data parity: when set,
    the notebook and the pipeline read the byte-identical snapshot forever.
    """

    name: str
    uri: str
    snapshot_pin: int | None = None


class ConnectionRef(BaseModel):
    """A SQL connection the pipeline uses (``[connections.<name>]``).

    ``auth_env_vars`` are the ``${VAR}`` indirection names the connection
    resolves at run time — the pipeline must be given the same variables
    (via Secrets Manager / job env); values are never captured.
    """

    name: str
    driver: str
    auth_env_vars: list[str] = Field(default_factory=list)


class PipelineNode(BaseModel):
    """One translatable cell as a pipeline step.

    ``static_provenance`` is a compile-time preview of the node's provenance
    hash (recursive over source + env + upstream static ids). It is NOT the
    runtime cache key — the running step recomputes provenance from actual
    upstream artifact content hashes (see design doc, open question 1). It is
    emitted so the IR is deterministic and so a change to a cell visibly
    propagates to its downstream nodes.
    """

    id: str
    name: str
    kind: str  # "python" | "sql"
    compute_target: str  # "glue_python_shell" | "athena"
    source: str
    inputs: list[NodeInput] = Field(default_factory=list)
    outputs: list[NodeOutput] = Field(default_factory=list)
    static_provenance: str = ""
    env: dict[str, str] = Field(default_factory=dict)
    connection: str | None = None  # SQL connection name (SQL nodes only)
    # Full connection spec (driver + host/path + ${VAR} auth indirections, never
    # real secrets) so a glue_sql node can open the same engine the notebook used.
    connection_config: dict[str, Any] | None = None
    mounts: list[MountRef] = Field(default_factory=list)
    tables: list[TableRef] = Field(default_factory=list)
    resources: NodeResources = Field(default_factory=NodeResources)


class PipelineEdge(BaseModel):
    """A variable dependency between two nodes (both translatable)."""

    from_node: str
    to_node: str
    variable: str


class PipelineParameter(BaseModel):
    """A widget-derived value threaded into the pipeline as a parameter."""

    name: str
    from_cell: str


class UnsupportedCell(BaseModel):
    """A cell that has no v1 compile target (prompt, R, …), recorded not dropped."""

    id: str
    language: str
    reason: str


class PipelineIR(BaseModel):
    """The compiled pipeline: nodes + edges + params, target-agnostic.

    Consumed by per-target renderers. ``topological_order`` lists node ids in
    a valid execution order (the notebook DAG's order, filtered to nodes).
    """

    ir_version: int = 1
    target: str = "aws"
    runtime: str = "container"  # node execution: "container" (Lambda) or "glue"
    notebook_id: str
    notebook_name: str
    env_hash: str
    requirements: list[str] = Field(default_factory=list)  # notebook [project.dependencies]
    requires_python: str = ""  # notebook requires-python (Glue Python Shell is 3.9)
    nodes: list[PipelineNode] = Field(default_factory=list)
    edges: list[PipelineEdge] = Field(default_factory=list)
    topological_order: list[str] = Field(default_factory=list)
    parameters: list[PipelineParameter] = Field(default_factory=list)
    connections: list[ConnectionRef] = Field(default_factory=list)
    unsupported: list[UnsupportedCell] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

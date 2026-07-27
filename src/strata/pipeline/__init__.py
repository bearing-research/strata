"""The pipeline engine: map a content-addressed computation DAG to production.

This package is **frontend-agnostic**. Its input is a :class:`PipelineIR` —
an internal, serializable description of nodes (units of ``materialize``),
their inputs/outputs, compute targets, environment, access requirements, and a
static provenance preview. Whatever authored the graph (a notebook, a
``@node``-decorated Python module, an imported DAG) is out of scope here: the
IR is the contract, and everything below it optimizes, renders, and runs.

The IR is modeled on dbt's ``manifest.json`` — a documented serialized DAG
artifact, not a standard. See ``docs/internal/design-pipeline-compile.md`` and
``docs/internal/design-ir-compute-fabric.md``.

Renderers (AWS container + Step Functions first, Glue second) consume the IR;
:mod:`strata.pipeline.runtime` is the node runtime that runs in production and
carries content-addressed execution-skip (``.prov`` markers) across the
boundary. :mod:`strata.pipeline.access` derives the experiment<->production
parity contract.

Notebook -> IR lives in :mod:`strata.notebook.compile` (one frontend adapter).
"""

from __future__ import annotations

from strata.pipeline.access import AccessManifest, build_access_manifest
from strata.pipeline.asl import AslRenderOptions, render_state_machine
from strata.pipeline.authoring import (
    build_pipeline_ir_from_functions,
    build_pipeline_ir_from_module,
    node,
)
from strata.pipeline.bundle import write_bundle
from strata.pipeline.container import (
    build_nodes_manifest,
    render_dockerfile,
    render_handler,
    vendor_runtime_kit,
)
from strata.pipeline.glue import render_glue_script
from strata.pipeline.ir import (
    ConnectionRef,
    MountRef,
    NodeInput,
    NodeOutput,
    NodeResources,
    PipelineEdge,
    PipelineIR,
    PipelineNode,
    PipelineParameter,
    TableRef,
    UnsupportedCell,
)

__all__ = [
    "AccessManifest",
    "AslRenderOptions",
    "ConnectionRef",
    "MountRef",
    "NodeInput",
    "NodeOutput",
    "NodeResources",
    "PipelineEdge",
    "PipelineIR",
    "PipelineNode",
    "PipelineParameter",
    "TableRef",
    "UnsupportedCell",
    "build_access_manifest",
    "build_nodes_manifest",
    "build_pipeline_ir_from_functions",
    "build_pipeline_ir_from_module",
    "node",
    "render_dockerfile",
    "render_glue_script",
    "render_handler",
    "render_state_machine",
    "vendor_runtime_kit",
    "write_bundle",
]

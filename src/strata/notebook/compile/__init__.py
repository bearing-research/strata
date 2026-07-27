"""Notebook -> pipeline IR: one frontend over the general pipeline engine.

The notebook is already a content-addressed compute DAG. This package is the
**frontend adapter** that lowers it into a :class:`~strata.pipeline.PipelineIR`
— cells become nodes, upstream variables become artifact inputs/outputs, source
annotations become access refs. Everything downstream (optimize, render,
bundle, run with skip + parity) is frontend-agnostic and lives in
:mod:`strata.pipeline`; the IR is the seam between the two.

Any other producer of the same IR (a ``@node`` Python module, an imported DAG)
compiles through the identical engine. See
``docs/internal/design-pipeline-compile.md``.
"""

from __future__ import annotations

from strata.notebook.compile.builder import (
    PipelineCompileError,
    build_pipeline_ir,
    build_pipeline_ir_from_dir,
)

__all__ = [
    "PipelineCompileError",
    "build_pipeline_ir",
    "build_pipeline_ir_from_dir",
]

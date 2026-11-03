"""Widget cells: one value artifact per control, no upstreams and no subprocess."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from strata.notebook.execution_result import CellExecutionResult

if TYPE_CHECKING:
    from strata.notebook.cell_paths.context import PathContext
    from strata.notebook.executor_host import ExecutorHost


async def run_widget_cell(
    host: ExecutorHost,
    context: PathContext,
    cell_id: str,
    source: str,
    start_time: float,
    *,
    use_cache: bool,
) -> CellExecutionResult:
    """Materialize each control's value artifact.

    Like prompt/SQL it persists the generic provenance triplet so ``compute_staleness`` keeps
    it READY when values are unchanged.
    """
    from strata.notebook.widget_executor import execute_widget_cell

    result_dict = execute_widget_cell(host, cell_id, source, use_cache=use_cache)

    if result_dict.get("success"):
        # Publish each control's value artifact onto ``artifact_uris``, as a Python cell does
        # for its outputs. Otherwise a downstream key sees no upstream
        # artifact and cache-hits the old output: dragging a slider never updates downstream.
        cell = host.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.artifact_uris = {
                name: out["artifact_uri"]
                for name, out in (result_dict.get("outputs") or {}).items()
                if isinstance(out, dict) and out.get("artifact_uri")
            }

        prov = await context.compute_provenance(cell_id, source)
        host.record_successful_execution_provenance(
            cell_id,
            prov.provenance_hash,
            prov.source_hash,
            prov.env_hash,
            table_reads=prov.table_reads,
            reads=prov.reads(*context.moved(cell_id)),
            key_parts=prov.key_parts,
        )

    return CellExecutionResult(
        cell_id=cell_id,
        success=result_dict["success"],
        outputs=result_dict.get("outputs", {}),
        display_outputs=result_dict.get("display_outputs") or [],
        error=result_dict.get("error"),
        cache_hit=result_dict.get("cache_hit", False),
        duration_ms=int((time.time() - start_time) * 1000),
        execution_method=result_dict.get("execution_method", "widget"),
        artifact_uri=result_dict.get("artifact_uri"),
    )

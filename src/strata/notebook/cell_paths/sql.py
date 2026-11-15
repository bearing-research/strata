"""SQL cells: run through ``strata.notebook.sql.cell_executor``, with no subprocess."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from strata.notebook.execution_result import CellExecutionResult

if TYPE_CHECKING:
    from strata.notebook.cell_paths.context import PathContext
    from strata.notebook.executor_host import ExecutorHost


async def run_sql_cell(
    host: ExecutorHost,
    context: PathContext,
    cell_id: str,
    source: str,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
) -> CellExecutionResult:
    """Execute a SQL cell via ``strata.notebook.sql.cell_executor``."""
    from strata.notebook.sql.cell_executor import execute_sql_cell

    if materialize_upstreams:
        failure = await context.materialize_upstreams_or_failure(cell_id, start_time, "sql")
        if failure is not None:
            return failure

    result_dict = await execute_sql_cell(
        host,
        cell_id,
        source,
        use_cache=use_cache,
    )

    if result_dict.get("success"):
        cell = host.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.cache_hit = bool(result_dict.get("cache_hit"))

        # compute_staleness compares the generic triplet (inputs + source + env) to
        # ``last_provenance_hash`` and can't see the SQL-specific hash. Persist the generic
        # one, as the prompt path does, so reopen marks the cell READY.
        prov = await context.compute_provenance(cell_id, source)
        host.record_successful_execution_provenance(
            cell_id,
            prov.provenance_hash,
            prov.source_hash,
            prov.env_hash,
            table_reads=prov.table_reads + result_dict.get("table_reads", []),
            reads=prov.reads(*context.moved(cell_id)),
            key_parts=prov.key_parts,
            reopen_identity=result_dict.get("reopen_identity"),
        )
        # Back the result-table preview with an artifact. Otherwise `save_cell_output` refuses
        # it, export drops it (``markdown_text`` is stripped at persist time and re-fetched
        # via the uri), and staleness can't resolve a cached display, so an upstream SQL cell
        # shows nothing or a stale table.
        stored_displays = context.store_inline_display_outputs(
            cell_id,
            prov.provenance_hash,
            result_dict.get("display_outputs") or [],
            source_hash=prov.source_hash,
            source=source,
            env_hash=prov.env_hash,
        )
        if stored_displays:
            result_dict["display_outputs"] = stored_displays
            result_dict["display_output"] = stored_displays[-1]
        # Record them where reopen and export read from, as the Python and R paths do.
        host.persist_display_outputs(cell_id, result_dict.get("display_outputs") or None)
    else:
        # A failed run clears what the cell showed, as in Python and R; otherwise the last
        # good table reads as a current result.
        host.persist_display_outputs(cell_id, None)

    # ``execute_sql_cell`` times only its own work; ``start_time`` also covers upstream
    # materialization and dispatch.
    duration_ms = (time.time() - start_time) * 1000

    result = CellExecutionResult(
        cell_id=cell_id,
        success=result_dict["success"],
        outputs=result_dict["outputs"],
        display_outputs=result_dict.get("display_outputs") or [],
        display_output=result_dict.get("display_output"),
        stdout=result_dict.get("stdout", ""),
        stderr=result_dict.get("stderr", ""),
        error=result_dict.get("error"),
        cache_hit=result_dict.get("cache_hit", False),
        duration_ms=int(duration_ms),
        execution_method=result_dict.get("execution_method", "sql"),
        artifact_uri=result_dict.get("artifact_uri"),
        mutation_warnings=result_dict.get("mutation_warnings", []),
    )
    # Record the run's error against its source and clear it on success, as Python and R
    # do, so a fixed query doesn't carry the previous error.
    host.apply_execution_result_metadata(cell_id, result)
    return result

"""Prompt cells: rendered and sent to the LLM provider, with no subprocess."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from strata.notebook.execution_result import CellExecutionResult

if TYPE_CHECKING:
    from strata.notebook.cell_paths.context import PathContext
    from strata.notebook.executor_host import ExecutorHost


async def run_prompt_cell(
    host: ExecutorHost,
    context: PathContext,
    cell_id: str,
    source: str,
    start_time: float,
    *,
    materialize_upstreams: bool,
    use_cache: bool,
    on_delta: Callable[[dict[str, Any]], Awaitable[None]] | None,
) -> CellExecutionResult:
    """Execute a prompt cell via the LLM provider."""
    from strata.notebook.llm.config import llm_config_for_session
    from strata.notebook.prompt_executor import execute_prompt_cell

    if materialize_upstreams:
        failure = await context.materialize_upstreams_or_failure(cell_id, start_time, "prompt")
        if failure is not None:
            return failure

    llm_config = llm_config_for_session(host)
    if llm_config is None:
        return CellExecutionResult(
            cell_id=cell_id,
            success=False,
            outputs={},
            stdout="",
            stderr="",
            error=(
                "LLM not configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
                "STRATA_AI_API_KEY in the notebook's environment (the Runtime panel, "
                "or [env] in notebook.toml), or STRATA_AI_API_KEY where the server starts. "
                "The server's key is not sent to an [ai] base_url of the notebook's own."
            ),
            cache_hit=False,
            duration_ms=int((time.time() - start_time) * 1000),
            execution_method="llm",
        )

    # The prompt executor caches by its own provenance (rendered text + model), which
    # compute_staleness can't see. Record the standard hash so "can_preserve_ready" matches.
    prov = await context.compute_provenance(cell_id, source)
    standard_provenance = prov.provenance_hash

    result_dict = await execute_prompt_cell(
        host,
        cell_id,
        source,
        llm_config,
        use_cache=use_cache,
        on_delta=on_delta,
    )

    if result_dict.get("success"):
        # Persisted, like every other cell kind's, or a prompt cell is idle after a restart
        # and re-issues a paid call.
        host.record_successful_execution_provenance(
            cell_id,
            standard_provenance,
            prov.source_hash,
            prov.env_hash,
            table_reads=prov.table_reads,
            reads=prov.reads(*context.moved(cell_id)),
            key_parts=prov.key_parts,
        )

    return CellExecutionResult(
        cell_id=cell_id,
        success=result_dict["success"],
        outputs=result_dict["outputs"],
        display_outputs=result_dict.get("display_outputs") or [],
        display_output=result_dict.get("display_output"),
        stdout=result_dict.get("stdout", ""),
        stderr=result_dict.get("stderr", ""),
        error=result_dict.get("error"),
        cache_hit=result_dict.get("cache_hit", False),
        duration_ms=result_dict.get("duration_ms", 0),
        execution_method=result_dict.get("execution_method", "llm"),
        artifact_uri=result_dict.get("artifact_uri"),
        mutation_warnings=result_dict.get("mutation_warnings", []),
        validation_retries=int(result_dict.get("validation_retries", 0) or 0),
    )

"""R cell executor adapter.

Counterpart of ``analyzer.py`` for the execution side. Delegates to
``CellExecutor._execute_r_cell``, which runs the cell via the R harness
(``harness.R`` in this same package) under Rscript with renv-activated
``.Rprofile``.

Phase 1 (#57) is local-only — no warm pool, no HTTP executor, no remote
workers, no batching. Those follow once the basic single-cell path is
proven; see #53 for the broader R roadmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata.notebook.languages.executor import register_language_executor
from strata.notebook.models import CellLanguage

if TYPE_CHECKING:
    from strata.notebook.executor import CellExecutionResult, CellExecutor
    from strata.notebook.models import CellState


class _RExecutor:
    """Adapter that delegates to ``CellExecutor._execute_r_cell``.

    Behaviour flags mirror ``_PythonExecutor`` — R cells go through the
    standard provenance/cache pipeline, no per-language alternate cache
    scheme, and execution provenance is always computed.

    ``is_batchable`` returns ``False`` for Phase 1 — R cells run one
    Rscript invocation per execute and don't share a process with the
    Python warm pool. Batching across R cells is a future optimization
    once the single-cell path is proven; see #26 for the Python-side
    batching protocol that any R batching would need to mirror.
    """

    skips_execution_provenance = False
    has_alternate_cache_scheme = False

    async def execute(
        self,
        executor: CellExecutor,
        cell_id: str,
        source: str,
        start_time: float,
        *,
        timeout_seconds: float,
        materialize_upstreams: bool,
        use_cache: bool,
    ) -> CellExecutionResult:
        return await executor._execute_r_cell(
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


register_language_executor(CellLanguage.R, _RExecutor())

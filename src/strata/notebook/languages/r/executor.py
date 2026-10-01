"""R cell executor adapter.

Delegates to ``CellExecutor._execute_r_cell``, which runs the cell through
``harness.R`` under Rscript with the renv-activated ``.Rprofile``.
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

    Uses the standard provenance/cache pipeline. Not batchable: each R cell runs in
    its own Rscript invocation.
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

    def reopen_identity(self, cell: CellState, session: object) -> str | None:
        del cell, session  # The generic triplet covers this language.
        return ""

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


register_language_executor(CellLanguage.R, _RExecutor())

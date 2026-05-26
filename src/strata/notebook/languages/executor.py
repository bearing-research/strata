"""``LanguageExecutor`` protocol + registry + built-in adapters.

The companion to ``analyzer.py``. The notebook's per-cell *execution*
dispatch used to branch via ``if cell.language == CellLanguage.X`` in
three places:

- ``executor.py:_materialize_cell`` (PROMPT / SQL / MARKDOWN short-circuit
  + PYTHON fall-through).
- ``executor.py:is_cell_batchable`` (PYTHON-only batching gate).
- ``session.py`` staleness gates (MARKDOWN skips provenance; PROMPT and
  SQL use an alternate per-variable cache scheme so the generic miss
  check needs to preserve READY status).

This module collapses all three behaviours onto per-language adapters:

- ``execute(...)`` — runs the cell. Each language's adapter delegates to
  its existing ``CellExecutor`` method, which is what stops this PR
  from rewriting ~400 lines of subprocess plumbing.
- ``is_batchable(cell, executor)`` — only PYTHON returns True (after the
  worker/loop/timeout/rw_mount checks); others return False.
- ``skips_execution_provenance`` / ``has_alternate_cache_scheme`` —
  flags consumed by the staleness gates in ``session.py``.

Adding R (or Lean) becomes a new adapter module + one
``register_language_executor`` call. No edits scattered across the
executor or session.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

from strata.notebook.models import CellLanguage, MountMode

if TYPE_CHECKING:
    from strata.notebook.executor import CellExecutionResult, CellExecutor
    from strata.notebook.models import CellState


class LanguageExecutor(Protocol):
    """Run a cell of a particular language + answer its behaviour flags.

    Each adapter is registered once at import time; ``execute_cell`` and
    related dispatch sites look up the adapter for the cell's language
    and call ``execute(...)``. Languages that bypass the subprocess
    pipeline (prompt = LLM call, sql = ADBC query, markdown = no-op)
    return their result directly; Python delegates to the existing
    subprocess pipeline.
    """

    # Behaviour flags consumed by session.py's staleness gates.

    skips_execution_provenance: bool
    """``True`` when the language has no inputs, no subprocess, and no
    provenance chain — i.e. cells of this language are always READY.
    Markdown is the only ``True`` today."""

    has_alternate_cache_scheme: bool
    """``True`` when the language stores artifacts under a per-language
    cache scheme that the generic per-variable lookup in
    ``compute_staleness`` won't match. PROMPT and SQL both qualify;
    they persist the generic hash via
    ``record_successful_execution_provenance`` so the staleness gate
    can preserve READY status when ``last_provenance_hash`` matches
    despite a cache miss."""

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
        """Run the cell and return its execution result."""
        ...

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        """Return whether ``cell`` is eligible for run-all batching."""
        ...


class UnknownLanguageError(LookupError):
    """Raised when a cell's language has no registered executor.

    Distinct from ``KeyError`` so callers can ``except`` it specifically.
    Mirrors the analyzer registry's error shape.
    """


_REGISTRY: dict[CellLanguage, LanguageExecutor] = {}


def register_language_executor(language: CellLanguage, executor_adapter: LanguageExecutor) -> None:
    """Bind ``executor_adapter`` to ``language`` in the global registry.

    Later registrations silently overwrite earlier ones; matches the
    analyzer registry + the SQL ``DriverAdapter`` registry.
    """
    _REGISTRY[language] = executor_adapter


def get_language_executor(language: CellLanguage) -> LanguageExecutor:
    """Look up the executor adapter for ``language``.

    Raises ``UnknownLanguageError`` rather than falling back to PYTHON
    so a missing registration surfaces immediately instead of silently
    routing R or Lean cells through the Python subprocess pipeline.
    """
    try:
        return _REGISTRY[language]
    except KeyError as exc:
        raise UnknownLanguageError(f"No language executor registered for {language!r}") from exc


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


class _PythonExecutor:
    """Adapter that delegates to the existing subprocess pipeline.

    The pipeline (provenance compute → upstream materialize → cache
    check → harness dispatch → persist) lives on
    ``CellExecutor._execute_python_cell`` — kept on ``CellExecutor``
    rather than moved here because it accesses ~30 private helper
    methods on the same object. Moving the body in this PR would mean
    promoting all of those to module-level or this adapter would need
    a private attribute on CellExecutor for each.
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
        return await executor._execute_python_cell(
            cell_id,
            source,
            timeout_seconds,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        # The full PYTHON-only batching gate from executor.py:is_cell_batchable.
        # Per issue #26: PYTHON cell, resolved worker is "local", no
        # ``# @loop`` annotation, no explicit timeout at any level, no
        # rw mount at any level.
        from strata.notebook.annotations import parse_annotations

        annotations = parse_annotations(cell.source)

        if executor._resolve_effective_worker(cell.id, annotations.worker) != "local":
            return False

        if annotations.loop is not None:
            return False

        notebook_state = executor.session.notebook_state
        if (
            annotations.timeout is not None
            or cell.timeout is not None
            or notebook_state.timeout is not None
        ):
            return False

        all_mounts = list(annotations.mounts) + list(cell.mounts) + list(notebook_state.mounts)
        if any(m.mode == MountMode.READ_WRITE for m in all_mounts):
            return False

        return True


class _PromptExecutor:
    """Adapter that delegates to ``CellExecutor._execute_prompt_cell``."""

    skips_execution_provenance = False
    has_alternate_cache_scheme = True  # per-variable hash via compute_prompt_provenance_hash

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
        # PROMPT path doesn't need ``timeout_seconds`` — the LLM provider
        # call has its own timeout. Drop the unused arg rather than
        # threading it through.
        del timeout_seconds
        return await executor._execute_prompt_cell(
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


class _SqlExecutor:
    """Adapter that delegates to ``CellExecutor._execute_sql_cell``."""

    skips_execution_provenance = False
    has_alternate_cache_scheme = True  # per-variable hash via compute_sql_provenance_hash

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
        del timeout_seconds  # SQL path manages its own deadline at the connection level.
        return await executor._execute_sql_cell(
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


class _MarkdownExecutor:
    """No-op adapter.

    Markdown cells are pure prose — no execution, no subprocess, no
    provenance chain. The frontend renders the source in-place via the
    cell's preview view, so emitting a display output would duplicate
    the same content in the output panel.
    """

    skips_execution_provenance = True
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
        # Unused — markdown returns success immediately without inspecting
        # source / upstreams / cache.
        del source, timeout_seconds, materialize_upstreams, use_cache
        # ``start_time`` is wall-clock (``time.time()``), so subtract in the
        # same clock — mixing ``monotonic()`` here produces a ~1.7e12 ms
        # negative because the two clocks have different epochs.
        duration_ms = (time.time() - start_time) * 1000
        # Import here to avoid a circular at module-import time
        # (executor imports languages, languages imports executor).
        from strata.notebook.executor import CellExecutionResult

        return CellExecutionResult(
            cell_id=cell_id,
            success=True,
            duration_ms=duration_ms,
            execution_method="cached",
            cache_hit=True,
        )

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


# Built-in registrations — performed at import time so the registry is
# populated by the time any dispatch site runs.
register_language_executor(CellLanguage.PYTHON, _PythonExecutor())
register_language_executor(CellLanguage.PROMPT, _PromptExecutor())
register_language_executor(CellLanguage.SQL, _SqlExecutor())
register_language_executor(CellLanguage.MARKDOWN, _MarkdownExecutor())


# Re-export ``Any`` so the package-level ``__init__`` doesn't need a
# separate annotation for the optional ``CellExecutor`` reference.
_: Any = None

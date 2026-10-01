"""``LanguageExecutor`` protocol, registry and built-in adapters.

Each adapter runs its language's cells (``execute``), answers whether they can
join run-all batching (``is_batchable``, Python only), and sets the flags the
staleness gates in ``session.py`` read (``skips_execution_provenance``,
``has_alternate_cache_scheme``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol

from strata.notebook.models import CellLanguage, MountMode

if TYPE_CHECKING:
    from strata.notebook.executor import CellExecutionResult, CellExecutor
    from strata.notebook.models import CellState


class LanguageExecutor(Protocol):
    """Run a cell of a particular language and answer its behaviour flags.

    Adapters register at import time. Prompt, SQL and markdown return their
    result directly; Python delegates to the subprocess pipeline.
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

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        """What this cell's cache identity rests on beyond the generic triplet.

        A language with its own cache scheme (a SQL cell's connection and cache
        policy, a prompt cell's model) folds more than ``compute_staleness``
        does, so READY on the generic hash alone would be unearned. ``None``
        means it cannot be settled without probing the outside world (the gate
        then runs the cell); ``""`` means the triplet already covers it.
        """
        ...


class UnknownLanguageError(LookupError):
    """Raised when a cell's language has no registered executor (distinct from ``KeyError``)."""


_REGISTRY: dict[CellLanguage, LanguageExecutor] = {}


def register_language_executor(language: CellLanguage, executor_adapter: LanguageExecutor) -> None:
    """Bind ``executor_adapter`` to ``language``; a later registration overwrites."""
    _REGISTRY[language] = executor_adapter


def get_language_executor(language: CellLanguage) -> LanguageExecutor:
    """Look up the executor adapter for ``language``.

    Raises ``UnknownLanguageError`` rather than falling back to Python, which
    would silently route other languages through the Python pipeline.
    """
    try:
        return _REGISTRY[language]
    except KeyError as exc:
        raise UnknownLanguageError(f"No language executor registered for {language!r}") from exc


# --- Built-in adapters ---


class _PythonExecutor:
    """Adapter that delegates to ``CellExecutor._execute_python_cell``.

    The pipeline stays on ``CellExecutor`` because it uses many of its private helpers.
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

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        del cell, session  # The generic triplet covers this language.
        return ""

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        # Mirrors executor.py:is_cell_batchable: local worker, no ``# @loop``,
        # no explicit timeout and no rw mount at any level.
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
        # The LLM provider call has its own timeout.
        del timeout_seconds
        return await executor._execute_prompt_cell(
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        """The model the answer came from, and the shape it was asked for.

        Settled by the annotations and the notebook's ``[ai]`` block, so reopen
        can check it without calling a provider.
        """
        from strata.notebook.prompt_executor import prompt_reopen_identity

        return prompt_reopen_identity(cell, session)

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

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        """The connection read and the cache policy, when they settle the identity.

        ``None`` for a policy that needs a freshness probe, which an open cannot make.
        """
        from strata.notebook.sql.cell_executor import sql_reopen_identity

        return sql_reopen_identity(cell, session)

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


class _MarkdownExecutor:
    """No-op adapter: markdown is not executed.

    No display output is emitted, since the frontend already renders the source
    in place and an output would duplicate it.
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
        # Markdown returns success without inspecting anything.
        del source, timeout_seconds, materialize_upstreams, use_cache
        # ``start_time`` is ``time.time()``; ``monotonic()`` has a different epoch.
        duration_ms = (time.time() - start_time) * 1000
        # Avoids a circular import (executor imports languages).
        from strata.notebook.executor import CellExecutionResult

        return CellExecutionResult(
            cell_id=cell_id,
            success=True,
            duration_ms=duration_ms,
            execution_method="cached",
            cache_hit=True,
        )

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        del cell, session  # The generic triplet covers this language.
        return ""

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


class _WidgetExecutor:
    """Adapter that delegates to ``CellExecutor._execute_widget_cell``."""

    skips_execution_provenance = False
    has_alternate_cache_scheme = True  # per-value hash via descriptor_provenance

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
        del timeout_seconds  # widget cells do no timed work.
        return await executor._execute_widget_cell(
            cell_id,
            source,
            start_time,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    def reopen_identity(self, cell: CellState, session: Any) -> str | None:
        """The control values the cell last rendered, which persist with it."""
        import hashlib
        import json

        del session
        values = json.dumps(cell.widget_values, sort_keys=True, default=str)
        return hashlib.sha256(values.encode()).hexdigest()

    def is_batchable(self, cell: CellState, executor: CellExecutor) -> bool:
        return False


# Registered at import time so the registry is populated before any dispatch.
register_language_executor(CellLanguage.PYTHON, _PythonExecutor())
register_language_executor(CellLanguage.PROMPT, _PromptExecutor())
register_language_executor(CellLanguage.SQL, _SqlExecutor())
register_language_executor(CellLanguage.MARKDOWN, _MarkdownExecutor())
register_language_executor(CellLanguage.WIDGET, _WidgetExecutor())


# Re-exports ``Any`` for the package-level ``__init__``.
_: Any = None

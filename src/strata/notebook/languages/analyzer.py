"""``LanguageAnalyzer`` protocol + registry + built-in adapters.

The four built-in languages (Python, prompt, SQL, markdown) used to
branch via ``if cell.language == CellLanguage.X`` in two near-identical
copies inside ``session.py`` — once in ``_analyze_and_build_dag`` and
once in ``re_analyze_cell``. Both copies extracted ``defines`` and
``references`` from a per-language analyzer, plus ``mutation_defines``
for Python.

This module collapses both call sites onto a registry.
``register_language_analyzer(language, analyzer)`` is the extension
point; adding R (or Lean) means a new adapter module + one
``register_language_analyzer`` call — no edits to ``session.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from strata.notebook.models import CellLanguage

if TYPE_CHECKING:
    from strata.notebook.models import CellState
    from strata.notebook.session import NotebookSession


@dataclass(frozen=True)
class AnalyzedCell:
    """Uniform analyzer result across languages.

    Each language's native analyzer returns its own dataclass
    (``CellAnalysis`` for Python, ``PromptAnalysis``, ``SqlAnalysis``);
    the dispatch sites only ever needed ``defines`` + ``references``
    plus Python's ``mutation_defines``. Coalescing here lets the
    dispatch site treat all languages uniformly.
    """

    defines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    mutation_defines: list[str] = field(default_factory=list)


class LanguageAnalyzer(Protocol):
    """Extract DAG inputs/outputs from a cell's source.

    The session passed in is the read-only context the analyzer needs —
    currently only the SQL adapter consumes it (to resolve the connection
    dialect via ``session._resolve_sql_dialect``); other languages
    ignore the arg. A flat parameter rather than a side-channel keeps
    the protocol surface small and explicit.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell: ...


class UnknownLanguageError(LookupError):
    """Raised when a cell's language has no registered analyzer.

    Distinct from ``KeyError`` so callers can ``except`` it specifically
    without catching unrelated dict misses inside the analyzer chain.
    """


_REGISTRY: dict[CellLanguage, LanguageAnalyzer] = {}


def register_language_analyzer(language: CellLanguage, analyzer: LanguageAnalyzer) -> None:
    """Bind ``analyzer`` to ``language`` in the global registry.

    Idempotent on a same-instance re-register; later registrations
    silently overwrite earlier ones (matches the SQL ``DriverAdapter``
    registry's behaviour at ``src/strata/notebook/sql/registry.py``).
    """
    _REGISTRY[language] = analyzer


def get_language_analyzer(language: CellLanguage) -> LanguageAnalyzer:
    """Look up the analyzer for ``language``.

    Raises ``UnknownLanguageError`` rather than returning a default so a
    missing registration surfaces immediately at the dispatch site
    instead of producing an empty ``AnalyzedCell`` that would silently
    drop every reference in the cell and break DAG construction.
    """
    try:
        return _REGISTRY[language]
    except KeyError as exc:
        raise UnknownLanguageError(f"No language analyzer registered for {language!r}") from exc


def analyze_cell_by_language(cell: CellState, session: NotebookSession) -> AnalyzedCell:
    """Dispatch helper: look up the analyzer and run it.

    Convenience over ``get_language_analyzer(...).analyze(cell, session)``
    for the common dispatch path.
    """
    return get_language_analyzer(cell.language).analyze(cell, session)


# ---------------------------------------------------------------------------
# Built-in adapters
# ---------------------------------------------------------------------------


class _PythonAnalyzer:
    """Adapter over ``strata.notebook.analyzer.analyze_cell``.

    Returns the only ``mutation_defines`` payload of the four languages —
    Python tracks subscript-assign style mutations so downstream cells
    that consume the mutated value know to invalidate.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.analyzer import analyze_cell

        result = analyze_cell(cell.source)
        return AnalyzedCell(
            defines=list(result.defines),
            references=list(result.references),
            mutation_defines=list(result.mutation_defines),
        )


class _PromptAnalyzer:
    """Adapter over ``strata.notebook.prompt_analyzer.analyze_prompt_cell``."""

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.prompt_analyzer import analyze_prompt_cell

        result = analyze_prompt_cell(cell.source)
        return AnalyzedCell(
            defines=list(result.defines),
            references=list(result.references),
        )


class _SqlAnalyzer:
    """Adapter over ``strata.notebook.sql.analyzer.analyze_sql_cell``.

    SQL needs the connection's dialect to extract table references via
    sqlglot; the resolver lives on the session. When the dialect can't
    be resolved (no connection declared yet) the analyzer falls back to
    a dialect-independent regex path that still gets bind-placeholder
    references right.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.sql.analyzer import analyze_sql_cell

        dialect = session._resolve_sql_dialect(cell)
        result = analyze_sql_cell(cell.source, dialect=dialect)
        return AnalyzedCell(
            defines=list(result.defines),
            references=list(result.references),
        )


class _MarkdownAnalyzer:
    """No-op analyzer.

    Markdown cells are pure prose — no identifiers in or out of the DAG,
    so they sit isolated with no edges. Empty ``AnalyzedCell`` is the
    correct answer; not raising ``UnknownLanguageError`` is the
    correct shape (the language IS known, it just has no analysis).
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        return AnalyzedCell()


# Built-in registrations. Performed at import time so the registry is
# populated by the time any ``session.py`` dispatch runs. New languages
# (R, Lean) register similarly from their own modules.
register_language_analyzer(CellLanguage.PYTHON, _PythonAnalyzer())
register_language_analyzer(CellLanguage.PROMPT, _PromptAnalyzer())
register_language_analyzer(CellLanguage.SQL, _SqlAnalyzer())
register_language_analyzer(CellLanguage.MARKDOWN, _MarkdownAnalyzer())

"""``LanguageAnalyzer`` protocol, registry and built-in adapters.

``register_language_analyzer(language, analyzer)`` is the extension point; the
session dispatches every cell's defines/references extraction through it.
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
    """Uniform analyzer result across languages; only Python sets ``mutation_defines``."""

    defines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    mutation_defines: list[str] = field(default_factory=list)
    # Python only: free names that shadow a builtin (``input``, ``type``). Hidden
    # from ``references`` but resolved by the DAG so the edge still wires.
    builtin_references: list[str] = field(default_factory=list)


class LanguageAnalyzer(Protocol):
    """Extract DAG inputs/outputs from a cell's source.

    ``session`` is read-only context; the built-in languages ignore it.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        """Return defines / references / mutation_defines for ``cell``."""
        ...


class UnknownLanguageError(LookupError):
    """Raised when a cell's language has no registered analyzer.

    Distinct from ``KeyError`` so callers can catch it without catching
    unrelated dict misses.
    """


_REGISTRY: dict[CellLanguage, LanguageAnalyzer] = {}


def register_language_analyzer(language: CellLanguage, analyzer: LanguageAnalyzer) -> None:
    """Bind ``analyzer`` to ``language`` in the global registry; later registrations overwrite."""
    _REGISTRY[language] = analyzer


def get_language_analyzer(language: CellLanguage) -> LanguageAnalyzer:
    """Look up the analyzer for ``language``.

    Raises ``UnknownLanguageError`` rather than returning an empty result,
    which would silently drop every reference and break the DAG.
    """
    try:
        return _REGISTRY[language]
    except KeyError as exc:
        raise UnknownLanguageError(f"No language analyzer registered for {language!r}") from exc


def analyze_cell_by_language(cell: CellState, session: NotebookSession) -> AnalyzedCell:
    """Look up the analyzer for ``cell``'s language and run it."""
    return get_language_analyzer(cell.language).analyze(cell, session)


# --- Built-in adapters ---


class _PythonAnalyzer:
    """Adapter over ``strata.notebook.analyzer.analyze_cell``; sets ``mutation_defines``."""

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.analyzer import analyze_cell

        result = analyze_cell(cell.source)
        return AnalyzedCell(
            defines=list(result.defines),
            references=list(result.references),
            mutation_defines=list(result.mutation_defines),
            builtin_references=list(result.builtin_references),
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

    The DAG needs only the output name and ``:name`` placeholders, so no
    dialect is passed and sqlglot never parses the body: a query sqlglot
    cannot read must not keep the notebook from opening.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.sql.analyzer import analyze_sql_cell

        result = analyze_sql_cell(cell.source)
        return AnalyzedCell(
            defines=list(result.defines),
            references=list(result.references),
        )


class _MarkdownAnalyzer:
    """No-op analyzer: markdown cells have no DAG edges (the language is known, so no error)."""

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        return AnalyzedCell()


class _WidgetAnalyzer:
    """Adapter over ``strata.notebook.widget_analyzer.analyze_widget_cell``.

    A widget cell is a pure producer: each control is a define, and
    ``references`` is always empty.
    """

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        from strata.notebook.widget_analyzer import analyze_widget_cell

        result = analyze_widget_cell(cell.source)
        return AnalyzedCell(defines=list(result.defines), references=[])


# Registered at import time so the registry is populated before any dispatch.
register_language_analyzer(CellLanguage.PYTHON, _PythonAnalyzer())
register_language_analyzer(CellLanguage.PROMPT, _PromptAnalyzer())
register_language_analyzer(CellLanguage.SQL, _SqlAnalyzer())
register_language_analyzer(CellLanguage.MARKDOWN, _MarkdownAnalyzer())
register_language_analyzer(CellLanguage.WIDGET, _WidgetAnalyzer())

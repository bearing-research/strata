"""Per-language adapters for the notebook subsystem.

Notebook cells can be Python, prompt-LLM, SQL, or markdown today; R is
planned via #53. Each language used to be a hard-coded ``if/elif`` branch
in ``executor.py`` and ``session.py``. This package extracts those
branches into Protocol-based adapters keyed by ``CellLanguage``, so
adding a new language is a new module + one registry entry instead of
edits scattered across the executor + session.

Phase 0 of #54 lands the **analyzer** side (defines/references
extraction). The executor side follows in a separate PR; see #54 for
the split rationale.
"""

from __future__ import annotations

from strata.notebook.languages.analyzer import (
    AnalyzedCell,
    LanguageAnalyzer,
    UnknownLanguageError,
    analyze_cell_by_language,
    get_language_analyzer,
    register_language_analyzer,
)

__all__ = [
    "AnalyzedCell",
    "LanguageAnalyzer",
    "UnknownLanguageError",
    "analyze_cell_by_language",
    "get_language_analyzer",
    "register_language_analyzer",
]

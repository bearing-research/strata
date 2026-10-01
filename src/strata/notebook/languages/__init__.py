"""Per-language adapters for the notebook subsystem.

Python, prompt, SQL, R, markdown and widget cells are Protocol-based adapters
keyed by ``CellLanguage``, on both the analyzer and executor side. A new
language is a new module plus one registry entry.
"""

from __future__ import annotations

# Non-core languages (R, ...) register their adapters at import time from their own
# sub-packages, so the core registry doesn't pull in every language's helpers.
from strata.notebook.languages import r as _r  # noqa: F401, E402
from strata.notebook.languages.analyzer import (
    AnalyzedCell,
    LanguageAnalyzer,
    UnknownLanguageError,
    analyze_cell_by_language,
    get_language_analyzer,
    register_language_analyzer,
)
from strata.notebook.languages.executor import (
    LanguageExecutor,
    get_language_executor,
    register_language_executor,
)

__all__ = [
    "AnalyzedCell",
    "LanguageAnalyzer",
    "LanguageExecutor",
    "UnknownLanguageError",
    "analyze_cell_by_language",
    "get_language_analyzer",
    "get_language_executor",
    "register_language_analyzer",
    "register_language_executor",
]

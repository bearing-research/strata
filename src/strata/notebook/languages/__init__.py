"""Per-language adapters for the notebook subsystem.

Notebook cells can be Python, prompt-LLM, SQL, R, or markdown. Each
language used to be a hard-coded ``if/elif`` branch in ``executor.py``
and ``session.py``. This package extracts those branches into
Protocol-based adapters keyed by ``CellLanguage``, so adding a new
language is a new module + one registry entry instead of edits scattered
across the executor + session.

Both sides are registry-driven: the **analyzer** side (defines/references
extraction, #54) and the **executor** side (#54 follow-up) each expose
``get_*`` / ``register_*`` hooks below.
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

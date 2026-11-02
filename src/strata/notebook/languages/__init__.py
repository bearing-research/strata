"""Per-language adapters for the notebook subsystem.

Python, prompt, SQL, R, markdown and widget cells run through Protocol-based
executors keyed by ``CellLanguage``; their analyzers are cellgraph's, registered
in ``strata.notebook.vocabulary``. A new language is a new module plus one entry
in each registry.
"""

from __future__ import annotations

# Non-core languages (R, ...) register their adapters at import time from their own
# sub-packages, so the core registry doesn't pull in every language's helpers.
from strata.notebook.languages import r as _r  # noqa: F401, E402
from strata.notebook.languages.analyzer import AnalyzedCell
from strata.notebook.languages.executor import (
    LanguageExecutor,
    get_language_executor,
    register_language_executor,
)

__all__ = [
    "AnalyzedCell",
    "LanguageExecutor",
    "get_language_executor",
    "register_language_executor",
]

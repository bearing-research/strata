"""R cell support for the Strata notebook.

Implementation lives in ``strata.notebook.languages.r``; the package
imports its submodules at module-load time so the analyzer registers
itself against the ``LanguageAnalyzer`` registry from #54.

R support is being landed incrementally per #53:

- **#56 (this PR)** — ``RLanguageAnalyzer`` (DAG defines/references via
  shelled-out ``Rscript`` with codetools). Lets R cells participate in
  the DAG before they can execute.
- **#55** — renv per-notebook environment.
- **#57** — ``harness.R`` + ``LanguageExecutor`` adapter so R cells
  actually run.
- **#58** — Arrow/RDS serialization tiers for cross-language exchange.
- **#59** — integration tests.

Attempting to execute an R cell before #57 lands raises
``UnknownLanguageError`` from the executor registry — the right shape
(loud failure, not silent fallthrough to Python).
"""

from __future__ import annotations

# Importing the analyzer module runs its ``register_language_analyzer``
# call at module load time. Anyone importing ``strata.notebook.languages``
# transitively gets the R analyzer wired in.
from strata.notebook.languages.r import analyzer as _analyzer  # noqa: F401

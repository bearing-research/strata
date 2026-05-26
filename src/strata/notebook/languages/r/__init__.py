"""R cell support for the Strata notebook.

Implementation lives in ``strata.notebook.languages.r``; the package
imports its submodules at module-load time so the analyzer registers
itself against the ``LanguageAnalyzer`` registry from #54.

R support is being landed incrementally per #53:

- **#56** — ``RLanguageAnalyzer`` (DAG defines/references via shelled-out
  ``Rscript``). Lets R cells participate in the DAG before they can execute.
- **#55** — renv per-notebook environment.
- **#57 (this PR)** — ``harness.R`` + ``LanguageExecutor`` adapter so R cells
  actually run.
- **#58** — Arrow/RDS serialization tiers for cross-language exchange.
- **#59** — integration tests.
"""

from __future__ import annotations

# Importing the analyzer + executor modules runs their respective
# ``register_language_analyzer`` / ``register_language_executor`` calls
# at module load time. Anyone importing ``strata.notebook.languages``
# transitively gets the R analyzer + executor wired in.
from strata.notebook.languages.r import analyzer as _analyzer  # noqa: F401
from strata.notebook.languages.r import executor as _executor  # noqa: F401

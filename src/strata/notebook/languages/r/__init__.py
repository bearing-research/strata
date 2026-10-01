"""R cell support for the Strata notebook.

Implementation lives in ``strata.notebook.languages.r``; the package
imports its submodules at module-load time so the analyzer registers
itself against the ``LanguageAnalyzer`` registry from #54.

R support landed incrementally across #53:

- **#56** — ``RLanguageAnalyzer`` (DAG defines/references via shelled-out
  ``Rscript``), so R cells participate in the DAG.
- **#55** — renv per-notebook environment.
- **#57** — ``harness.R`` + ``LanguageExecutor`` adapter so R cells run.
- **#58** — Arrow/RDS serialization tiers for cross-language exchange.
- **#59** — integration tests.
"""

from __future__ import annotations

# Imported for their side effect: registering the R analyzer and executor.
from strata.notebook.languages.r import analyzer as _analyzer  # noqa: F401
from strata.notebook.languages.r import executor as _executor  # noqa: F401

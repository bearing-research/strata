"""R cell support for the Strata notebook.

Importing the package registers the R analyzer and executor.
"""

from __future__ import annotations

# Imported for their side effect: registering the R analyzer and executor.
from strata.notebook.languages.r import analyzer as _analyzer  # noqa: F401
from strata.notebook.languages.r import executor as _executor  # noqa: F401

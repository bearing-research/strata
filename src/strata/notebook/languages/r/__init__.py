"""R cell support for the Strata notebook.

Importing the package registers the R executor.
"""

from __future__ import annotations

# Imported for its side effect: registering the R executor.
from strata.notebook.languages.r import executor as _executor  # noqa: F401

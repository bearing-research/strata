"""Public helpers for explicit notebook display values, e.g. ``Markdown``.

Implementation lives in :mod:`strata.notebook.display.runtime`, which harness
subprocesses also load by file path (see the constraint at the top of it).
"""

from strata.notebook.display.runtime import Markdown

__all__ = ["Markdown"]

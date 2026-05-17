"""Public helpers for explicit notebook display values.

Notebook cells use these to emit named-type outputs that the frontend
renders specially (markdown rendering, image embedding, JSON pretty-
print, etc.) instead of relying on the cell's last-expression
``__repr__``. New display helpers land in this ``__init__`` so users
can keep a single import path::

    from strata.notebook.display import Markdown

Implementation lives in :mod:`strata.notebook.display.runtime`, which
is *also* loaded directly by subprocess harnesses via file path — see
the constraints documented at the top of ``runtime.py``.
"""

from strata.notebook.display.runtime import Markdown

__all__ = ["Markdown"]

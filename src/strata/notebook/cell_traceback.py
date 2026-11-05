"""How a cell's failure reads, the same from every harness.

Loaded by path into the harnesses, so stdlib only. They compile a cell's source as
``<cell>``, which is how a traceback tells the cell's frames from their own.
"""

from __future__ import annotations

import traceback
from typing import Any

CELL_FILENAME = "<cell>"


def error_text(exc: BaseException) -> str:
    """A cell's failure as one reads it: ``str(KeyError("MAX"))`` alone says only ``'MAX'``."""
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _from_the_cell(exc: BaseException) -> str:
    """The traceback without the harness frames that led to the cell.

    Library frames below the cell stay. An exception with no cell frame keeps them all: it
    is the harness failing, or a library's own chained error.
    """
    top = traceback.TracebackException.from_exception(exc)
    pending = [top]
    while pending:
        entry = pending.pop()
        frames = list(entry.stack)
        first = next((i for i, f in enumerate(frames) if f.filename == CELL_FILENAME), None)
        if first is not None:
            entry.stack = traceback.StackSummary.from_list(frames[first:])
        elif getattr(entry, "filename", None) == CELL_FILENAME:
            # Source that does not parse has no frame of its own to start from.
            entry.stack = traceback.StackSummary.from_list([])
        pending.extend(chained for chained in (entry.__cause__, entry.__context__) if chained)
        pending.extend(entry.exceptions or [])
    return "".join(top.format())


def failure_fields(exc: BaseException, hint: str | None = None) -> dict[str, Any]:
    """A failed run's ``error``, ``traceback`` and ``traceback_full``."""
    text = error_text(exc)
    return {
        "error": f"{text}\n{hint}" if hint else text,
        "traceback": _from_the_cell(exc),
        "traceback_full": "".join(traceback.format_exception(exc)),
    }

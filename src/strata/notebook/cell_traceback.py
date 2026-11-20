"""What every harness does the same around a cell's code, and how its failure reads.

Loaded by path into the harnesses, so stdlib only. They compile a cell's source under
``cell_filename``, which is how a traceback tells the cell's frames from their own.
"""

from __future__ import annotations

import linecache
import os
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

CELL_ID_VAR = "STRATA_CELL_ID"


@contextmanager
def cell_id_set(cell_id: str | None) -> Generator[None]:
    """Set ``STRATA_CELL_ID`` for the cell's code and what it starts, then put it back.

    Put back so a reused worker's next cell, or a caller in this process, keeps its own.
    Without an id it is unset: an inherited one names some other cell.
    """
    previous = os.environ.get(CELL_ID_VAR)
    if cell_id:
        os.environ[CELL_ID_VAR] = cell_id
    else:
        os.environ.pop(CELL_ID_VAR, None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(CELL_ID_VAR, None)
        else:
            os.environ[CELL_ID_VAR] = previous


def safely[T](read: Callable[[], T], default: T) -> T:
    """*read* a failing cell's own objects while its failure is reported: whatever that
    raises (an exit, a cancellation) gives *default*, so a report is still made; an
    interrupt stays one, for the harness's own handler."""
    try:
        return read()
    except KeyboardInterrupt:
        raise
    except BaseException:
        return default


@contextmanager
def exit_fails_the_cell(what: str = "the cell") -> Generator[None]:
    """A cell's ``SystemExit`` (``sys.exit()``, ``exit()``, a library's, as argparse's)
    fails the cell as an exception does, which is what each harness reports; it does not
    end the process before that process returns its result."""
    try:
        yield
    except SystemExit as exc:
        exited = exc
        code = safely(lambda: exited.code, None)
        message = safely(lambda: _exited(what, code), f"{what} exited, {_STOPS}")
        cause = safely(lambda: _cause(exited), None)
        # The cell's own frames: a traceback starts at the first of them.
        raise RuntimeError(message).with_traceback(exc.__traceback__) from cause


_EXIT_TEXT_MAX = 10_000
_STOPS = "which stops it before its end"


def _exited(what: str, code: object) -> str:
    """As Python reads an exit: no code, a status, or text it prints (``str()`` of
    anything else, lines kept, capped)."""
    if code is None:
        return f"{what} exited, {_STOPS}"
    if isinstance(code, int):
        # A bool too, as its status (True exits 1).
        number = safely(lambda: int(code), None)
        status = (
            "<int() failed>"
            if number is None
            else safely(lambda: str(number), "<too large to print>")
        )
        return f"{what} exited with status {status}, {_STOPS}"
    # A plain str, whatever str() returned (a subclass's own methods could raise).
    text = safely(lambda: str.__str__(str(code)), "<str() failed>")
    if len(text) > _EXIT_TEXT_MAX:
        text = f"{text[:_EXIT_TEXT_MAX]}..."
    return f"{what} exited, {_STOPS}: {text}" if text else f"{what} exited, {_STOPS}"


def _cause(exc: BaseException) -> BaseException | None:
    """What Python shows before *exc*: its ``from`` cause, else what it was raised while
    handling (``except ValueError: sys.exit(1)``) unless ``from None`` hid that."""
    if exc.__cause__ is not None:
        return exc.__cause__
    return None if exc.__suppress_context__ else exc.__context__


_CELL_PREFIX = "<cell"


def cell_filename(cell_id: str | None) -> str:
    """The name a cell's code compiles under.

    One per cell: a function defined in one cell can fail in another's run, and its line
    numbers are its own cell's.
    """
    return f"{_CELL_PREFIX} {cell_id}>" if cell_id else f"{_CELL_PREFIX}>"


def remember_source(filename: str, source: str) -> None:
    """Let a traceback quote *filename*'s lines, which no file on disk holds."""
    # A None mtime is what linecache keeps through checkcache: source with no file.
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)  # type: ignore[assignment]


def _is_cell(filename: str | None) -> bool:
    return bool(filename) and filename.startswith(_CELL_PREFIX)


def error_text(exc: BaseException) -> str:
    """A cell's failure as one reads it: ``str(KeyError("MAX"))`` alone says only ``'MAX'``."""
    # As the traceback module shows it: the failure is still reported.
    text = safely(lambda: str.__str__(str(exc)), "<exception str() failed>")
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
        first = next((i for i, f in enumerate(frames) if _is_cell(f.filename)), None)
        if first is not None:
            entry.stack = traceback.StackSummary.from_list(frames[first:])
        elif _is_cell(getattr(entry, "filename", None)):
            # Source that does not parse has no frame of its own to start from.
            entry.stack = traceback.StackSummary.from_list([])
        pending.extend(chained for chained in (entry.__cause__, entry.__context__) if chained)
        pending.extend(entry.exceptions or [])
    return "".join(top.format())


def failure_fields(exc: BaseException, hint: str | None = None) -> dict[str, Any]:
    """A failed run's ``error``, ``traceback`` and ``traceback_full``."""
    text = error_text(exc)
    # The exception's own attributes can raise (a NameError's ``name``): every path
    # still reports the failure, with its frames, which need none of them.
    note = f"{text}\n(its traceback could not be formatted)\n"
    trimmed = safely(lambda: _from_the_cell(exc), None)
    if trimmed is None:
        trimmed = _frames_alone(exc, cell_only=True) + note
    full = safely(lambda: "".join(traceback.format_exception(exc)), None)
    if full is None:
        full = _frames_alone(exc, cell_only=False) + note
    return {
        "error": f"{text}\n{hint}" if hint else text,
        "traceback": trimmed,
        "traceback_full": full,
    }


def _frames_alone(exc: BaseException, *, cell_only: bool) -> str:
    def read() -> str:
        frames = traceback.extract_tb(exc.__traceback__)
        first = next((i for i, f in enumerate(frames) if _is_cell(f.filename)), None)
        if cell_only and first is not None:
            frames = traceback.StackSummary.from_list(frames[first:])
        # As Python prints it: no header without a frame.
        return "Traceback (most recent call last):\n" + "".join(frames.format()) if frames else ""

    return safely(read, "")

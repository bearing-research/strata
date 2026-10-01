"""Holding a notebook still for the seconds a copy takes.

A copy taken while a cell completes can pair a new ``runtime.json`` with old
artifacts. A hold covers one notebook directory or a project directory and
every notebook under it, in two phases: **draining** (new runs refused, running
cells may finish and write) and **held** (runs, edits and other writers
refused). It ends on release or after ``max_hold_seconds``. State is
in-process: a hold is a promise about what this server writes.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Literal


class NotebookQuiesced(RuntimeError):
    """A run or a write reached a notebook that is being held still."""

    code = "NOTEBOOK_QUIESCED"


@dataclass
class Hold:
    root: Path
    state: Literal["draining", "held"]
    expires_at: float  # time.monotonic()

    def seconds_left(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


_lock = threading.Lock()
_holds: dict[Path, Hold] = {}


def _resolve(path: Path) -> Path:
    return Path(path).resolve()


def _covering(path: Path) -> Hold | None:
    return _covering_resolved(_resolve(path))


def _covering_resolved(target: Path) -> Hold | None:
    now = time.monotonic()
    with _lock:
        for root in [r for r, hold in _holds.items() if hold.expires_at <= now]:
            del _holds[root]
        for root, hold in _holds.items():
            if target == root or target.is_relative_to(root):
                return hold
    return None


def _message(hold: Hold) -> str:
    return (
        f"The notebook is held still for a copy ({hold.root}); runs and edits are "
        f"refused until it is released, or for at most {int(hold.seconds_left())} "
        "more seconds."
    )


def execution_block(path: Path) -> str | None:
    """Why a run in *path* may not start, or ``None``."""
    hold = _covering(path)
    return _message(hold) if hold is not None else None


def assert_writable(path: Path) -> None:
    """Refuse a write into a held notebook.

    Only in the ``held`` phase: while draining, a running cell must still be
    able to write the result it was allowed to finish.
    """
    hold = _covering(path)
    if hold is not None and hold.state == "held":
        raise NotebookQuiesced(_message(hold))


def refuses_while_held[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Guard a writer whose first argument is the notebook directory.

    Checks at entry, so a writer that touches two files cannot land one and be
    refused on the other.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        notebook_dir = args[0] if args else kwargs.get("notebook_dir")
        if isinstance(notebook_dir, str | Path):
            assert_writable(Path(notebook_dir))
        return func(*args, **kwargs)

    return wrapper


def begin(root: Path, max_hold_seconds: float) -> Hold:
    """Start draining *root*; refuse one that overlaps an existing hold.

    *root* must already be resolved (and confined, for request-supplied paths);
    this does not touch the filesystem.
    """
    if _covering_resolved(root) is not None:
        raise NotebookQuiesced(f"{root} is already held")
    with _lock:
        for other in _holds:
            if other.is_relative_to(root):
                raise NotebookQuiesced(f"{other}, inside {root}, is already held")
        hold = Hold(root=root, state="draining", expires_at=time.monotonic() + max_hold_seconds)
        _holds[root] = hold
    return hold


def settle(hold: Hold) -> None:
    """Running work is done: from here nothing writes."""
    with _lock:
        hold.state = "held"


def release(root: Path) -> bool:
    """End the hold on exactly *root* (already resolved, as for :func:`begin`)."""
    with _lock:
        return _holds.pop(root, None) is not None


def reset() -> None:
    """Forget every hold. For tests."""
    with _lock:
        _holds.clear()

"""Holding a notebook still for the seconds a copy takes.

A notebook is a directory Strata writes as cells finish: sources, ``runtime.json``,
console snapshots, artifacts. A copy taken while a cell completes can carry the
new ``runtime.json`` and the old artifacts, which then disagree about what the
notebook holds. Quiescing is the pause that makes a copy consistent.

A hold covers a directory: one notebook, or a project directory and every
notebook under it. It has two phases.

* **draining** — new runs are refused, and running cells are allowed to finish
  and write their results (or are cancelled at the caller's timeout).
* **held** — nothing writes: runs and edits are refused, and so is any writer
  that reaches the notebook's files some other way.

It ends on release, or when ``max_hold_seconds`` passes, so a caller that dies
mid-copy cannot freeze a notebook for good. State lives in this process: a hold
is a promise about what this server writes, and nothing else writes a notebook.
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
    target = _resolve(path)
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

    Only once the hold is ``held``: while draining, a running cell has to be
    able to write the result it was allowed to finish.
    """
    hold = _covering(path)
    if hold is not None and hold.state == "held":
        raise NotebookQuiesced(_message(hold))


def refuses_while_held[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Guard a writer whose first argument is the notebook directory.

    At entry rather than at the file write, so a writer that touches two files
    cannot land one and be refused on the other.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        notebook_dir = args[0] if args else kwargs.get("notebook_dir")
        if isinstance(notebook_dir, str | Path):
            assert_writable(Path(notebook_dir))
        return func(*args, **kwargs)

    return wrapper


def begin(root: Path, max_hold_seconds: float) -> Hold:
    """Start draining *root*. Refuses one that overlaps an existing hold.

    *root* must already be resolved. The routes hand in either a session's
    path or one ``_validate_notebook_path`` resolved and confined to the
    storage root, and resolving again here would touch the filesystem with a
    request-supplied path for no gain.
    """
    if _covering(root) is not None:
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

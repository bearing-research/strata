"""Route ``# @output_dir`` files a worker uploads while a cell runs to the run that wants them.

The dispatching executor registers a receiver for the build; the upload route asks here
and hands the file over, and the receiver stores it in the notebook's store at once, so a
worker lost afterwards still leaves what it uploaded. Only the node that dispatched the
build has its receiver: an upload that lands on another node is refused (409), and the
worker sends that file in its result instead.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from computation_core.file_modes import private_dir

from strata.notebook import output_dirs


@dataclass
class Receiver:
    """Where one run's uploaded files go, and what has arrived: ``{folder: {path: entry}}``."""

    artifact_mgr: Any
    cell_id: str
    upload_dir: Path  # server-owned, for an upload in progress
    folders: frozenset[str] = frozenset()  # the cell's ``# @output_dir`` names
    files: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _begun: int = 0
    _newest: dict[tuple[str, str], int] = field(default_factory=dict)

    def begin(self) -> int:
        """Number an upload as it starts, for :meth:`accept`."""
        with self._lock:
            self._begun += 1
            return self._begun

    def accept(self, folder: str, rel: str, path: Path, sha256: str, size: int, begun: int) -> None:
        """Store an uploaded file and record it as the folder's *rel*.

        The upload that began later wins, not the one stored last: a worker resends a file
        that changed while an earlier upload of it may still be storing.
        """
        entry = output_dirs.store_received(self.artifact_mgr, self.cell_id, path, sha256, size)
        with self._lock:
            if self._newest.get((folder, rel), 0) > begun:
                return
            self._newest[(folder, rel)] = begun
            self.files.setdefault(folder, {})[rel] = entry

    def received(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            return {folder: dict(files) for folder, files in self.files.items()}


_receivers: dict[str, Receiver] = {}
# A partial upload untouched this long was left by a crash. Another process with the
# notebook open (a server beside the CLI) may be writing one, so its start time is no bound.
_STALE_UPLOAD_SECONDS = 3600


def remove_crashed_uploads(upload_dir: Path) -> None:
    """Remove partial uploads a crash left, untouched for an hour."""
    stale_before = time.time() - _STALE_UPLOAD_SECONDS
    for path in upload_dir.glob(".upload-*"):
        with contextlib.suppress(FileNotFoundError):
            if path.lstat().st_mtime < stale_before:
                path.unlink()


@contextmanager
def receiving(build_id: str, receiver: Receiver) -> Generator[Receiver]:
    """Accept uploads for *build_id* for as long as the dispatch runs."""
    # Owner-only: a cell run as the harness user must not reach an upload in progress.
    private_dir(receiver.upload_dir)
    _receivers[build_id] = receiver
    try:
        yield receiver
    finally:
        _receivers.pop(build_id, None)


def receiver_for(build_id: str) -> Receiver | None:
    return _receivers.get(build_id)

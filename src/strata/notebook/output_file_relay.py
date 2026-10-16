"""Route ``# @output_dir`` files a worker uploads while a cell runs to the run that wants them.

The dispatching executor registers a receiver for the build; the upload route asks here
and hands the file over, and the receiver stores it in the notebook's store at once, so a
worker lost afterwards still leaves what it uploaded. Only the node that dispatched the
build has its receiver: an upload that lands on another node is refused (409), and the
worker sends that file in its result instead.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strata.file_modes import private_dir
from strata.notebook import output_dirs


@dataclass
class Receiver:
    """Where one run's uploaded files go, and what has arrived: ``{folder: {path: entry}}``."""

    artifact_mgr: Any
    cell_id: str
    upload_dir: Path  # server-owned, for an upload in progress
    files: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def accept(self, folder: str, rel: str, path: Path) -> None:
        """Store an uploaded file and record it as the folder's *rel*; a later upload wins."""
        entry = output_dirs.store_received(self.artifact_mgr, self.cell_id, path)
        with self._lock:
            self.files.setdefault(folder, {})[rel] = entry

    def received(self) -> dict[str, dict[str, dict[str, Any]]]:
        with self._lock:
            return {folder: dict(files) for folder, files in self.files.items()}


_receivers: dict[str, Receiver] = {}


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

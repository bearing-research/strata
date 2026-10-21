"""``.strata/writes/index.json``: what each cell's last good run wrote through its mounts.

Beside the reads index, for a host to list the files cells wrote to a source::

    {"<cell_id>": {"ran_at": <unix ms>, "principal": ..., "mounts": [
        {"mount": ..., "uri": ..., "files": [{"path": ..., "size": ..., "sha256": ...}]}]}}

``mounts`` has one entry per read-write mount the run wrote through, as its
``__mount__<name>`` record does, and ``principal`` is the member the run was for (null
in personal mode). A cell's entry is replaced once its writes are synced back, and
dropped when a successful run writes nothing (a cache hit or a batched run too); a
failed run, a failed sync-back included, leaves it as it was. A ``# @per_variant`` cell's
entry merges its variants' writes per mount, recorded once every variant has succeeded.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from strata.notebook.quiesce import refuses_while_held

# Concurrent cell runs both record: serialize the read-modify-write.
_INDEX_LOCK = threading.Lock()


def index_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / "writes" / "index.json"


def load(notebook_dir: Path) -> dict[str, dict[str, Any]]:
    path = index_path(notebook_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Rewritten whole at the next run; a damaged one must not fail this one.
        return {}


@refuses_while_held
def record(
    notebook_dir: Path,
    cell_id: str,
    entry: dict[str, Any] | None,
    cells: Iterable[str],
) -> None:
    """Make *entry* what *cell_id* wrote, dropping entries of cells not in *cells*."""
    live = set(cells)
    with _INDEX_LOCK:
        index = {cid: e for cid, e in load(notebook_dir).items() if cid in live}
        if entry is not None:
            index[cell_id] = entry
        else:
            index.pop(cell_id, None)
        if not index and not index_path(notebook_dir).exists():
            return
        path = index_path(notebook_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as out:
                out.write(json.dumps(index, indent=2, sort_keys=True))
            os.replace(tmp, path)
        finally:
            Path(tmp).unlink(missing_ok=True)

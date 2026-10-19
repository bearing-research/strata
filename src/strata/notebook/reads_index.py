"""``.strata/reads/index.json``: what each cell's last good run moved onto its machine.

Beside the table index (Amber item 89), for a host to count bytes moved per notebook::

    {"<cell_id>": {"machine": ..., "ran_at": <unix ms>, "inputs": [
        {"kind": "mount", "name": ..., "uri": ..., "credential": ..., "bytes_moved": ...}]}}

``machine`` is the worker the cell ran on (``local`` for this server). ``bytes_moved`` is
what the run downloaded for the input, 0 when it was already there, and null when not
known: a table's scan, and a mount read on a remote worker. A cell's entry is replaced
when it runs successfully, a cache hit included, and dropped when it reads nothing.
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

# The executor and a batch's persist both record: serialize the read-modify-write.
_INDEX_LOCK = threading.Lock()


def index_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / "reads" / "index.json"


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
    """Make *entry* what *cell_id* moved, dropping entries of cells not in *cells*."""
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

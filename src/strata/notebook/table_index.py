"""``.strata/tables/index.json``: the tables each cell read on its last good run.

Beside the ``@fetch`` index, for a host to list the cells that read a table and the
snapshot each read::

    {"<cell_id>": [{"uri": ..., "catalog": ..., "table": ..., "snapshot_id": ...}, ...]}

``uri`` is as written (a SQL cell's catalog table as ``<catalog>:<namespace>.<table>``),
``catalog`` the configured catalog it names (null for a warehouse path or the default
catalog), ``table`` the ``<namespace>.<table>`` identifier, and ``snapshot_id`` null for
a table with no snapshots yet. A cell's entry is replaced when it runs successfully, a
cache hit included, so it describes the reads behind the cell's current outputs.
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
    return Path(notebook_dir) / ".strata" / "tables" / "index.json"


def table_read(uri: str, snapshot_id: int | None, config: Any) -> dict[str, Any]:
    """One index entry for a read of *uri* at *snapshot_id*."""
    from strata.iceberg.catalog import PyIcebergCatalog, named_catalog

    catalog, table = named_catalog(uri, config)
    if catalog is None:
        _, table = PyIcebergCatalog.parse_table_uri(uri)
    return {"uri": uri, "catalog": catalog, "table": table, "snapshot_id": snapshot_id}


def load(notebook_dir: Path) -> dict[str, list[dict[str, Any]]]:
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
    reads: list[dict[str, Any]],
    cells: Iterable[str],
) -> None:
    """Make *reads* the tables *cell_id* read, dropping entries of cells not in *cells*."""
    live = set(cells)
    with _INDEX_LOCK:
        index = {cid: entry for cid, entry in load(notebook_dir).items() if cid in live}
        if reads:
            index[cell_id] = sorted(reads, key=lambda read: read["uri"])
        else:
            index.pop(cell_id, None)
        if not index and not index_path(notebook_dir).exists():
            return
        _write(notebook_dir, index)


def _write(notebook_dir: Path, index: dict[str, list[dict[str, Any]]]) -> None:
    path = index_path(notebook_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".index-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(json.dumps(index, indent=2, sort_keys=True))
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)

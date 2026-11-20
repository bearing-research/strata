"""A scratchpad's digest: what its cells held at the last staleness pass (design 4.7).

The daemon writes ``.strata/digest.json`` after each pass; it outlives the daemon. The
agent hook (``strata/agent_hook.py``) reads it at session start and on each prompt, and
checks every entry against the files before it calls a cell cached, so the digest is a
record of what to check, never the answer.
"""

from __future__ import annotations

import json
import logging
import os
import time
import tomllib
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.notebook.code_reads import base_names_to_check, base_paths, observe
from strata.notebook.layout import cell_source_path
from strata.notebook.models import CellLanguage, CellStatus

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

DIGEST = Path(".strata") / "digest.json"
VERSION = 1
# The statuses a cell holds something in; an idle one has nothing to offer.
_HELD = {CellStatus.READY, CellStatus.STALE, CellStatus.ERROR}

logger = logging.getLogger(__name__)


def _files(notebook_dir: Path) -> dict[str, str]:
    """Each cell's source file, by id, as ``notebook.toml`` names it."""
    try:
        cells = tomllib.loads((notebook_dir / "notebook.toml").read_text()).get("cells") or []
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {c["id"]: c["file"] for c in cells if isinstance(c, dict) and "id" in c and "file" in c}


def build(session: NotebookSession) -> dict[str, Any]:
    """The digest of *session*'s scratchpad as it stands."""
    from strata.notebook.project_env import environment_record

    files = _files(session.path)
    cells: list[dict[str, Any]] = []
    for cell in session.notebook_state.cells:
        if cell.language != CellLanguage.PYTHON or cell.status not in _HELD:
            continue
        entry: dict[str, Any] = {
            "cell": cell.id,
            "defines": sorted(cell.defines),
            "status": cell.status.value,
            "upstream": list(cell.upstream_ids),
        }
        if cell.staleness is not None and cell.staleness.reasons:
            entry["stale"] = [reason.value for reason in cell.staleness.reasons]
            if cell.staleness.changed_code:
                entry["changed"] = cell.staleness.changed_code
        cold_ms = session.get_estimated_duration(cell.id)
        if cold_ms:
            entry["cold_ms"] = cold_ms
        if cell.id in files:
            try:
                path = cell_source_path(session.path, cell.id, files[cell.id])
            except ValueError:
                path = None
            if path is not None:
                seen, print_ = observe(str(path))
                entry["source"] = [str(path), seen, print_]
        reads = session.code_reads(cell.id)
        if reads is not None:
            entry["reads"] = reads
        cells.append(entry)
    record = environment_record(session.path) or {}
    anchor = str(session.path.parent)
    base = record.get("base") or {}
    return {
        "version": VERSION,
        "written_at": int(time.time() * 1000),
        "notebook": str(session.path),
        "environment": {
            "observed": record.get("observed") or {},
            # As the daemon checks E's base (``project_env._holds``): with the anchor's
            # listing as it was, only the names something was at; else every place.
            "anchor": anchor,
            "anchor_listing": record.get("anchor_listing"),
            "base": base,
            "base_check": base_paths(
                anchor, base_names_to_check(anchor, record.get("base_names") or [], base)
            ),
        },
        "cells": cells,
    }


def write(session: NotebookSession) -> None:
    """Write the digest; a failure is logged, as nothing but the hook reads it."""
    path = session.path / DIGEST
    partial = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        text = json.dumps(build(session))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(text)
        os.replace(partial, path)
    except OSError as exc:
        logger.warning("Could not write %s: %s", path, exc)
        partial.unlink(missing_ok=True)

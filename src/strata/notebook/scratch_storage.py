"""A scratchpad's storage: a sweep under a soft limit, and the overage it cannot free.

A scratchpad keeps superseded results on purpose (reverting an edit hits again), so it
is swept by idle time and size, not by a count per value: what has not been used for
*max_idle_days* goes, then the least recently used while the store is over its soft
limit. Every live cell's current value is kept whatever the limit, so the limit can be
exceeded; the sweep records by how much, and which cells hold the most, in
``.strata/storage.json``, for ``scratchpad`` and ``strata scratch ls`` to show.

``run`` adds a cell per call, so cells have a way out too: a leaf cell with no
``# @name``, unused for *retire_days*, is retired. Its source moves to ``retired.jsonl``,
it leaves the notebook, and the next sweep releases every value it stored.
"""

from __future__ import annotations

import json
import os
import threading
import time
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import filelock

from strata.notebook.annotations import parse_annotations
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.runtime_state import load_runtime_state
from strata.scratch import register, scratch_dir

if TYPE_CHECKING:
    from strata.notebook.ops import CellView, NotebookOps
    from strata.notebook.session import NotebookSession, SessionManager

_LARGEST = 3
_DAY = 86400.0
# Deleting the scratchpad is the one step that always frees everything.
SAFE_TO_DELETE = "Deleting .strata-scratch/ is always safe: it costs reruns, never a wrong answer."


@dataclass(frozen=True)
class Storage:
    """What a sweep left: the store's size against its soft limit."""

    store_bytes: int
    soft_limit_bytes: int
    over_by_bytes: int
    swept_at: float
    # The live cells holding the most, when over: ``{"cell", "bytes"}``, largest first.
    largest: list[dict[str, int | str]] = field(default_factory=list)


def storage_path(notebook_dir: Path) -> Path:
    return notebook_dir / ".strata" / "storage.json"


def read_storage(notebook_dir: Path) -> Storage | None:
    """The last sweep's record, or None before the first."""
    try:
        return Storage(**json.loads(storage_path(notebook_dir).read_text()))
    except FileNotFoundError:
        return None


def _largest_cells(
    manager: NotebookArtifactManager, live_cells: list[str]
) -> list[dict[str, int | str]]:
    """The live cells whose current values hold the most: what a sweep could not free."""
    live = set(live_cells)
    held: Counter[str] = Counter()
    for latest in manager.artifact_store.list_latest_by_id_prefix(
        f"nb_{manager.notebook_id}_cell_"
    ):
        output = manager.cell_output_of(latest.id)
        if output is not None and output[0] in live:
            held[output[0]] += latest.byte_size or 0
    return [{"cell": cell, "bytes": size} for cell, size in held.most_common(_LARGEST)]


def retired_path(notebook_dir: Path) -> Path:
    return notebook_dir / "retired.jsonl"


def _retired_lock(notebook_dir: Path) -> filelock.FileLock:
    return filelock.FileLock(notebook_dir / ".strata" / "retired.lock")


def retired_cells(notebook_dir: Path) -> list[dict[str, Any]]:
    """Every cell retired from the scratchpad at *notebook_dir*, oldest first."""
    try:
        text = retired_path(notebook_dir).read_text()
    except FileNotFoundError:
        return []
    records = []
    # The last piece has no newline yet while an append is under way.
    for line in text.split("\n")[:-1]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Torn by a crash mid-append: that one record is lost, not every later one.
            continue
    return records


def _write_atomic(path: Path, text: str) -> None:
    # Its own name: the daemon's sweep and `strata scratch gc` may write at once.
    partial = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
    partial.write_text(text)
    os.replace(partial, path)


def _append_retired(notebook_dir: Path, records: list[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(record) + "\n" for record in records)
    with _retired_lock(notebook_dir), open(retired_path(notebook_dir), "a+") as retired:
        retired.seek(0, os.SEEK_END)
        if retired.tell():
            retired.seek(retired.tell() - 1)
            if retired.read(1) != "\n":
                # A crash tore the last append: start on a line of its own.
                lines = "\n" + lines
        retired.write(lines)


def _shown_path(notebook_dir: Path) -> Path:
    return notebook_dir / ".strata" / "shown.json"


def touch_shown(notebook_dir: Path, cell_id: str) -> None:
    """Record that ``scratchpad`` showed *cell_id*'s value: a use, as a run is."""
    path = _shown_path(notebook_dir)
    try:
        shown = json.loads(path.read_text())
    except FileNotFoundError:
        shown = {}
    shown[cell_id] = time.time()
    _write_atomic(path, json.dumps(shown))


def last_uses(notebook_dir: Path, cells: list[CellView]) -> dict[str, float]:
    """When each cell was last run, shown or edited.

    A reader's loads need not count: an upstream is a leaf, so may retire, only once
    every reader retired, each unused past the same cutoff.
    """
    runtime = load_runtime_state(notebook_dir)
    try:
        shown = json.loads(_shown_path(notebook_dir).read_text())
    except FileNotFoundError:
        shown = {}
    edited: dict[str, float] = {}
    sources = notebook_dir / "cells"
    for source in sources.iterdir() if sources.is_dir() else ():
        cell_id = source.name.split(".", 1)[0]
        edited[cell_id] = max(edited.get(cell_id, 0.0), source.stat().st_mtime)
    uses: dict[str, float] = {}
    for cell in cells:
        times = [shown.get(cell.id, 0.0), edited.get(cell.id, 0.0)]
        entry = runtime.cells.get(cell.id)
        if entry is not None:
            times += [sample.get("at", 0.0) for sample in entry.execution_samples]
        uses[cell.id] = max(times)
    return uses


def plan_retirement(
    notebook_dir: Path,
    cells: list[CellView],
    *,
    retire_days: float,
    now: float | None = None,
    uses: dict[str, float] | None = None,
) -> list[CellView]:
    """The leaf cells with no ``# @name`` unused for *retire_days*, in the order they go.

    Retiring a leaf can leave its upstream a leaf, which goes too if it is idle as well.
    A *retire_days* of 0 retires nothing. *uses* is :func:`last_uses`, read if not given.
    """
    if not retire_days:
        return []
    cutoff = (time.time() if now is None else now) - retire_days * _DAY
    if uses is None:
        uses = last_uses(notebook_dir, cells)
    # Named cells stay, and only idle ones can go.
    candidates = [
        cell
        for cell in cells
        if uses[cell.id] < cutoff and parse_annotations(cell.source, cell.language).name is None
    ]
    going: list[CellView] = []
    gone: set[str] = set()
    while True:
        idle = [
            cell
            for cell in candidates
            if cell.id not in gone and not set(cell.downstream_ids) - gone
        ]
        if not idle:
            return going
        going += idle
        gone |= {cell.id for cell in idle}


def retirement_record(
    notebook_dir: Path, manager: NotebookArtifactManager, cell: CellView
) -> dict[str, Any]:
    """What ``retired.jsonl`` keeps of *cell*: enough to run it again, and what to release."""
    tests = notebook_dir / "cells" / f"{cell.id}.test.py"
    return {
        "cell": cell.id,
        "language": cell.language,
        "source": cell.source,
        # Removing the cell deletes its tests file.
        "test_source": tests.read_text() if tests.is_file() else None,
        "defines": cell.defines,
        "retired_at": time.time(),
        # For the sweep to release, current values included.
        "release": [list(value) for value in manager.cell_values(cell.id)],
    }


def retire_cell(
    ops: NotebookOps, notebook_dir: Path, manager: NotebookArtifactManager, cell: CellView
) -> dict[str, Any]:
    """Remove *cell*, then keep it in ``retired.jsonl`` with every value it stored."""
    record = retirement_record(notebook_dir, manager, cell)
    # Recorded only once it is gone: a live cell's values are never released.
    ops.remove_cell(cell.id)
    _append_retired(notebook_dir, [record])
    return record


def retire_idle(
    ops: NotebookOps, notebook_dir: Path, *, retire_days: float, now: float | None = None
) -> list[str]:
    """Retire what :func:`plan_retirement` names, through *ops*; return their ids."""
    going = plan_retirement(notebook_dir, ops.list_cells(), retire_days=retire_days, now=now)
    if not going:
        return []
    manager = scratch_manager(notebook_dir)
    try:
        for cell in going:
            retire_cell(ops, notebook_dir, manager, cell)
    finally:
        manager.artifact_store.close()
    return [cell.id for cell in going]


def _planned(notebook_dir: Path, cells: list[CellView], retire_days: float):
    # The uses the plan judged by: one used since, even while planning, stays.
    uses = last_uses(notebook_dir, cells)
    going = plan_retirement(notebook_dir, cells, retire_days=retire_days, uses=uses)
    if not going:
        return going, [], {}
    manager = scratch_manager(notebook_dir)
    try:
        records = [retirement_record(notebook_dir, manager, cell) for cell in going]
    finally:
        manager.artifact_store.close()
    return going, records, {cell.id: uses[cell.id] for cell in going}


def session_holding(
    session_manager: SessionManager, notebook_dir: Path
) -> tuple[str, NotebookSession] | None:
    """The session open on *notebook_dir*, if any, and its id."""
    for session_id in session_manager.list_sessions():
        session = session_manager.get_session(session_id)
        if session is not None and session.path.resolve() == notebook_dir.resolve():
            return session_id, session
    return None


async def retire_in_session(
    session_manager: SessionManager, notebook_dir: Path, *, retire_days: float
) -> list[str]:
    """Retire idle cells through the session holding *notebook_dir*, which viewers see.

    Planned off the loop; removed on it, only if nothing ran or changed meanwhile.
    Nothing while it has a run reserved or running, or when no session holds it: the
    session is the notebook's writer, so the files are not edited behind it.
    """
    import anyio.to_thread

    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.ws import broadcast_notebook_sync, notebook_has_active_execution

    holding = session_holding(session_manager, notebook_dir)
    if holding is None:
        return []
    session_id, session = holding
    if notebook_has_active_execution(session_id):
        return []
    cells = LocalNotebookOps.from_session(session).list_cells()
    going, records, planned_uses = await anyio.to_thread.run_sync(
        _planned, notebook_dir, cells, retire_days
    )
    if not going:
        return []
    unchanged = {cell.id: cell.source for cell in cells}
    now = {cell.id: cell.source for cell in session.notebook_state.cells}
    if (
        notebook_has_active_execution(session_id)
        or now != unchanged
        # Run or shown while it was planned: used, and its record misses what it stored.
        or await anyio.to_thread.run_sync(last_uses, notebook_dir, going) != planned_uses
    ):
        # The next pass plans again.
        return []
    for cell, record in zip(going, records, strict=True):
        # The session's removal: a variant group's pointer and the read indexes follow.
        session.remove_cell(cell.id)
        # Each as it goes: one that fails to go must not cost the others' records.
        await anyio.to_thread.run_sync(_append_retired, notebook_dir, [record])
    await broadcast_notebook_sync(session_id, session)
    return [cell.id for cell in going]


def scratch_manager(notebook_dir: Path) -> NotebookArtifactManager:
    toml = tomllib.loads((notebook_dir / "notebook.toml").read_text())
    return NotebookArtifactManager(
        toml["notebook_id"], artifact_dir=notebook_dir / ".strata" / "artifacts"
    )


def sweep(
    notebook_dir: Path,
    *,
    soft_limit_bytes: int,
    max_idle_days: float,
    min_idle_seconds: float,
) -> Storage:
    """Sweep the scratchpad at *notebook_dir* and record what is left."""
    toml = tomllib.loads((notebook_dir / "notebook.toml").read_text())
    live_cells = [str(cell["id"]) for cell in toml.get("cells", [])]
    manager = scratch_manager(notebook_dir)
    try:
        return _swept(
            notebook_dir,
            manager,
            live_cells,
            soft_limit_bytes=soft_limit_bytes,
            max_idle_days=max_idle_days,
            min_idle_seconds=min_idle_seconds,
        )
    finally:
        manager.artifact_store.close()


def _swept(
    notebook_dir: Path,
    manager: NotebookArtifactManager,
    live_cells: list[str],
    *,
    soft_limit_bytes: int,
    max_idle_days: float,
    min_idle_seconds: float,
) -> Storage:
    manager.prune(
        None,
        min_idle_seconds,
        live_cells=live_cells,
        # 0 turns either off.
        max_idle_days=max_idle_days or None,
        max_bytes=soft_limit_bytes or None,
        # What a version still held (named, pinned, an input of a live value) keeps it,
        # and a later sweep offers it again.
        released=[
            (artifact_id, version)
            for record in retired_cells(notebook_dir)
            for artifact_id, version in record["release"]
        ],
    )
    _forget_released(notebook_dir, manager)
    store_bytes = manager.artifact_store.stats()["total_bytes"]
    over_by = max(0, store_bytes - soft_limit_bytes) if soft_limit_bytes else 0
    storage = Storage(
        store_bytes=store_bytes,
        soft_limit_bytes=soft_limit_bytes,
        over_by_bytes=over_by,
        swept_at=time.time(),
        largest=_largest_cells(manager, live_cells) if over_by else [],
    )
    _write_atomic(storage_path(notebook_dir), json.dumps(asdict(storage), indent=2) + "\n")
    return storage


def _forget_released(notebook_dir: Path, manager: NotebookArtifactManager) -> None:
    """Keep in each record only the versions the store still has, for the next sweep."""
    with _retired_lock(notebook_dir):
        records = retired_cells(notebook_dir)
        changed = False
        for record in records:
            if not record["release"]:
                continue
            there = {tuple(value) for value in manager.cell_values(record["cell"])}
            left = [value for value in record["release"] if tuple(value) in there]
            if left != record["release"]:
                record["release"] = left
                changed = True
        if changed:
            _write_atomic(
                retired_path(notebook_dir), "".join(json.dumps(r) + "\n" for r in records)
            )


def upkeep(
    anchor: Path,
    *,
    soft_limit_bytes: int,
    max_idle_days: float,
    min_idle_seconds: float,
    used: bool = True,
) -> Storage:
    """Sweep *anchor*'s scratchpad and update its registry entry (*used*: it was in use)."""
    storage = sweep(
        scratch_dir(anchor),
        soft_limit_bytes=soft_limit_bytes,
        max_idle_days=max_idle_days,
        min_idle_seconds=min_idle_seconds,
    )
    register(
        anchor, store_bytes=storage.store_bytes, over_by_bytes=storage.over_by_bytes, used=used
    )
    return storage

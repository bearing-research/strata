"""Retiring a scratchpad's idle cells: their source kept, every value they stored released."""

from __future__ import annotations

import json
import os
import time

import pytest

from strata.cli import main
from strata.notebook.ops import LocalNotebookOps
from strata.notebook.runtime_state import persist_cell_execution_sample
from strata.notebook.scratch_storage import (
    retire_idle,
    retire_in_session,
    retired_cells,
    scratch_manager,
    sweep,
    touch_shown,
)
from strata.notebook.session import SessionManager
from strata.notebook.writer import create_notebook

DAY = 86400.0


@pytest.fixture
def notebook(tmp_path):
    return create_notebook(tmp_path, "scratch", initialize_environment=False)


def _cells(notebook, *sources: str) -> list[str]:
    """Cells with *sources*, last edited a month ago."""
    ops = LocalNotebookOps(notebook)
    ids = [ops.add_cell(source).id for source in sources]
    month_ago = time.time() - 30 * DAY
    for path in (notebook / "cells").iterdir():
        os.utime(path, (month_ago, month_ago))
    return ids


def _stored(notebook, cell: str, variable: str, versions: int = 2) -> str:
    """*versions* values of *cell*'s *variable* in the store."""
    manager = scratch_manager(notebook)
    artifact_id = manager.cell_artifact_id(cell, variable)
    for version in range(1, versions + 1):
        manager.artifact_store.create_artifact(artifact_id, f"prov-{artifact_id}-{version}")
        manager.artifact_store.write_blob(artifact_id, version, b"x")
        manager.artifact_store.finalize_artifact(artifact_id, version, "{}", 1, 1)
    return artifact_id


def _versions(notebook, artifact_id: str) -> list[int]:
    store = scratch_manager(notebook).artifact_store
    return [v.version for v in store.list_versions_by_id_prefix(artifact_id)]


def _sweep(notebook) -> None:
    sweep(notebook, soft_limit_bytes=10**12, max_idle_days=14, min_idle_seconds=0)


def test_idle_leaves_retire_then_what_they_read_and_the_sweep_frees_them(notebook):
    a, b, named, d = _cells(notebook, "x = 1", "y = x + 1", "# @name keep\nz = 3", "w = 4")
    x, y, w = (_stored(notebook, c, v) for c, v in ((a, "x"), (b, "y"), (d, "w")))

    retired = retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14)

    # b and d first; then a, which b's going left a leaf. A named cell never retires by age.
    assert retired == [b, d, a]
    assert [cell.id for cell in LocalNotebookOps(notebook).list_cells()] == [named]
    records = {record["cell"]: record for record in retired_cells(notebook)}
    assert records[b]["source"] == "y = x + 1"
    assert records[b]["defines"] == ["y"]
    assert [x, 1] in records[a]["release"] and [x, 2] in records[a]["release"]

    _sweep(notebook)
    # Every version, current values included.
    assert _versions(notebook, x) == _versions(notebook, y) == _versions(notebook, w) == []


def test_a_cell_in_use_keeps_itself_and_what_it_reads(notebook):
    a, b, looked_at, idle = _cells(notebook, "x = 1", "y = x + 1", "v = 2", "w = 4")
    persist_cell_execution_sample(notebook, b, duration_ms=5, cache_hit=True)
    touch_shown(notebook, looked_at)

    assert retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14) == [idle]


def test_a_released_value_still_held_stays_until_the_hold_ends(notebook):
    (a,) = _cells(notebook, "x = 1")
    x = _stored(notebook, a, "x")
    store = scratch_manager(notebook).artifact_store
    store.set_name("pinned", x, 2)
    retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14)

    _sweep(notebook)
    # Only the named version stays; nothing else of the release set loses its way out.
    assert _versions(notebook, x) == [2]

    store.delete_name("pinned")
    _sweep(notebook)
    assert _versions(notebook, x) == []


@pytest.mark.asyncio
async def test_the_daemon_retires_through_its_session_and_not_mid_run(notebook, monkeypatch):
    from strata.notebook import ws

    (a,) = _cells(notebook, "x = 1")
    sm = SessionManager()
    session = sm.open_notebook(notebook)

    monkeypatch.setattr(ws, "notebook_has_active_execution", lambda notebook_id: True)
    assert await retire_in_session(sm, notebook, retire_days=14) == []

    monkeypatch.undo()
    assert await retire_in_session(sm, notebook, retire_days=14) == [a]
    # The session saw it go: it, not the files behind it, is the writer.
    assert session.notebook_state.get_cell(a) is None


def test_the_cli_retires_a_cell_by_hand_but_not_one_something_reads(project, capsys):
    from strata import scratch

    scratch._scaffold(project)
    a, b = _cells(scratch.scratch_dir(project), "x = 1", "y = x + 1")

    assert main(["scratch", "retire", a, "--root", str(project)]) == 2
    assert b in capsys.readouterr().err

    assert main(["scratch", "retire", b, "--root", str(project)]) == 0
    assert json.loads(capsys.readouterr().out) == {"retired": b, "defines": ["y"]}
    # gc retires what is idle, then sweeps.
    assert main(["scratch", "gc", "--root", str(project)]) == 0
    assert json.loads(capsys.readouterr().out)[str(project.resolve())]["retired"] == [a]

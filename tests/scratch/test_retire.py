"""Retiring a scratchpad's idle cells: their source kept, every value they stored released."""

from __future__ import annotations

import asyncio
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


@pytest.fixture
async def sm():
    """A session manager whose sessions, and the warm processes they start, close after."""
    manager = SessionManager()
    yield manager
    for session_id in list(manager._sessions):
        await asyncio.gather(*manager.close_session(session_id))


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
async def test_the_daemon_retires_through_its_session_and_not_mid_run(notebook, monkeypatch, sm):
    from strata.notebook import ws

    (a,) = _cells(notebook, "x = 1")
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


def test_a_cell_that_fails_to_go_keeps_its_values(notebook):
    from strata.notebook.ops import NotebookOpsError
    from strata.notebook.scratch_storage import retire_cell

    (a,) = _cells(notebook, "x = 1")
    ops = LocalNotebookOps(notebook)
    cell = ops.get_cell(a)

    def refuse(cell_id):
        raise NotebookOpsError("no")

    ops.remove_cell = refuse
    with pytest.raises(NotebookOpsError):
        retire_cell(ops, notebook, scratch_manager(notebook), cell)
    # Not recorded, so the sweep never releases a live cell's values.
    assert retired_cells(notebook) == []


def test_the_record_keeps_the_tests_removal_deletes(notebook):
    (a,) = _cells(notebook, "x = 1")
    (notebook / "cells" / f"{a}.test.py").write_text("def test_x(): assert x == 1\n")
    os.utime(notebook / "cells" / f"{a}.test.py", (0, 0))

    retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14)

    (record,) = retired_cells(notebook)
    assert record["test_source"] == "def test_x(): assert x == 1\n"


def test_a_line_still_being_appended_is_not_read(notebook):
    from strata.notebook.scratch_storage import retired_path

    retired_path(notebook).write_text('{"cell": "a"}\n{"cell": "b", "sou')

    assert retired_cells(notebook) == [{"cell": "a"}]


def test_what_a_sweep_freed_is_not_offered_again(notebook):
    (a,) = _cells(notebook, "x = 1")
    x = _stored(notebook, a, "x")
    retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14)
    assert len(retired_cells(notebook)[0]["release"]) == 2

    _sweep(notebook)

    assert _versions(notebook, x) == []
    assert retired_cells(notebook)[0]["release"] == []


@pytest.mark.asyncio
async def test_a_cell_changed_while_the_daemon_planned_is_left_for_the_next_pass(
    notebook, monkeypatch, sm
):
    from strata.notebook import scratch_storage

    (a,) = _cells(notebook, "x = 1")
    session = sm.open_notebook(notebook)
    planned = scratch_storage._planned

    def then_edited(*args):
        going = planned(*args)
        session.notebook_state.get_cell(a).source = "x = 2"
        return going

    monkeypatch.setattr(scratch_storage, "_planned", then_edited)
    assert await retire_in_session(sm, notebook, retire_days=14) == []
    assert session.notebook_state.get_cell(a) is not None
    assert retired_cells(notebook) == []


def test_a_record_torn_by_a_crash_costs_only_itself(notebook):
    from strata.notebook.scratch_storage import _append_retired, retired_path

    retired_path(notebook).write_text('{"cell": "a"}\n{"cell": "b", "sou')
    _append_retired(notebook, [{"cell": "c"}])
    _append_retired(notebook, [{"cell": "d"}])

    assert retired_cells(notebook) == [{"cell": "a"}, {"cell": "c"}, {"cell": "d"}]


@pytest.mark.asyncio
async def test_the_daemon_removes_a_cell_as_its_session_does(notebook, monkeypatch, sm):
    from strata.notebook.session import NotebookSession

    (a,) = _cells(notebook, "x = 1")
    sm.open_notebook(notebook)
    removed = []
    remove = NotebookSession.remove_cell

    def recorded(self, cell_id):
        removed.append(cell_id)
        remove(self, cell_id)

    # Its removal fixes a variant group's pointer and drops the read indexes' entries.
    monkeypatch.setattr(NotebookSession, "remove_cell", recorded)
    assert await retire_in_session(sm, notebook, retire_days=14) == [a]
    assert removed == [a]


@pytest.mark.asyncio
async def test_a_cell_used_while_the_daemon_planned_is_left_for_the_next_pass(
    notebook, monkeypatch, sm
):
    from strata.notebook import scratch_storage

    (a,) = _cells(notebook, "x = 1")
    sm.open_notebook(notebook)
    planned = scratch_storage._planned

    def then_shown(*args):
        going = planned(*args)
        touch_shown(notebook, a)
        return going

    monkeypatch.setattr(scratch_storage, "_planned", then_shown)
    assert await retire_in_session(sm, notebook, retire_days=14) == []
    assert retired_cells(notebook) == []


def test_zero_retire_days_retires_nothing(notebook):
    _cells(notebook, "x = 1")
    assert retire_idle(LocalNotebookOps(notebook), notebook, retire_days=0) == []


def test_gc_leaves_retirement_to_a_running_daemon(project, monkeypatch, capsys):
    from strata import scratch
    from strata.notebook import scratch_storage

    scratch._scaffold(project)
    _cells(scratch.scratch_dir(project), "x = 1")
    daemon = scratch.Daemon(url="http://127.0.0.1:9", session_id="s", pid=4321, version="0")
    monkeypatch.setattr(scratch, "find_daemon", lambda anchor: daemon)

    def removed(*args, **kwargs):
        raise AssertionError("retired behind the daemon")

    monkeypatch.setattr(scratch_storage, "retire_idle", removed)
    assert main(["scratch", "gc", "--root", str(project)]) == 0
    retired = json.loads(capsys.readouterr().out)[str(project.resolve())]["retired"]
    assert "left to its daemon (pid 4321)" in retired


def test_gc_reads_its_limits_as_the_daemon_does(project, monkeypatch, capsys):
    """From inside the scratchpad, not the project: one policy for both sweeps."""
    from strata import scratch

    scratch._scaffold(project)
    (a,) = _cells(scratch.scratch_dir(project), "x = 1")
    (project / "pyproject.toml").write_text("[tool.strata]\nnotebook_scratch_retire_days = 0\n")
    monkeypatch.chdir(project)
    assert main(["scratch", "gc", "--root", str(project)]) == 0
    assert json.loads(capsys.readouterr().out)[str(project.resolve())]["retired"] == [a]


def test_the_cli_will_not_retire_a_running_cell(project, monkeypatch, capsys):
    from strata import scratch

    scratch._scaffold(project)
    (a,) = _cells(scratch.scratch_dir(project), "x = 1")
    get_cell = LocalNotebookOps.get_cell
    monkeypatch.setattr(
        LocalNotebookOps,
        "get_cell",
        lambda self, cell_id: get_cell(self, cell_id).model_copy(update={"status": "running"}),
    )
    assert main(["scratch", "retire", a, "--root", str(project)]) == 2
    assert "running" in capsys.readouterr().err
    assert retired_cells(scratch.scratch_dir(project)) == []


@pytest.mark.asyncio
async def test_a_cell_used_as_the_daemon_planned_is_left_for_the_next_pass(
    notebook, monkeypatch, sm
):
    """Used after the plan read its uses, before the records were made."""
    from strata.notebook import scratch_storage

    (a,) = _cells(notebook, "x = 1")
    sm.open_notebook(notebook)
    plan = scratch_storage.plan_retirement

    def then_shown(*args, **kwargs):
        going = plan(*args, **kwargs)
        touch_shown(notebook, a)
        return going

    monkeypatch.setattr(scratch_storage, "plan_retirement", then_shown)
    assert await retire_in_session(sm, notebook, retire_days=14) == []
    assert retired_cells(notebook) == []


@pytest.mark.asyncio
async def test_a_cell_that_fails_to_go_costs_only_its_own_record(notebook, monkeypatch, sm):
    from strata.notebook.session import NotebookSession

    a, b = _cells(notebook, "x = 1", "y = 2")
    sm.open_notebook(notebook)
    remove = NotebookSession.remove_cell

    def second_fails(self, cell_id):
        if cell_id == b:
            raise ValueError("held")
        remove(self, cell_id)

    monkeypatch.setattr(NotebookSession, "remove_cell", second_fails)
    with pytest.raises(ValueError, match="held"):
        await retire_in_session(sm, notebook, retire_days=14)
    assert [record["cell"] for record in retired_cells(notebook)] == [a]


def test_an_offline_retirement_removes_a_cell_as_a_session_does(notebook, monkeypatch):
    from strata.notebook.session import NotebookSession

    (a,) = _cells(notebook, "x = 1")
    removed = []
    remove = NotebookSession.remove_cell
    monkeypatch.setattr(
        NotebookSession,
        "remove_cell",
        lambda self, cell_id: removed.append(cell_id) or remove(self, cell_id),
    )
    assert retire_idle(LocalNotebookOps(notebook), notebook, retire_days=14) == [a]
    assert removed == [a]


def _daemon_lock_held(project) -> bool:
    import filelock

    from strata import scratch

    lock = filelock.FileLock(scratch.scratch_dir(project) / ".strata" / "daemon.lock")
    try:
        lock.acquire(timeout=0)
    except filelock.Timeout:
        return True
    lock.release()
    return False


def test_gc_retires_offline_holding_the_daemon_lock(project, monkeypatch, capsys):
    """So no daemon starts and opens a session while the files are edited here."""
    from strata import scratch
    from strata.notebook import scratch_storage

    scratch._scaffold(project)
    _cells(scratch.scratch_dir(project), "x = 1")
    held = []

    def retire(*args, **kwargs):
        held.append(_daemon_lock_held(project))
        return []

    monkeypatch.setattr(scratch_storage, "retire_idle", retire)
    assert main(["scratch", "gc", "--root", str(project)]) == 0
    assert held == [True]


def test_the_cli_retires_holding_the_daemon_lock_and_closes_its_store(project, monkeypatch, capsys):
    from strata import scratch
    from strata.notebook import scratch_storage

    scratch._scaffold(project)
    (a,) = _cells(scratch.scratch_dir(project), "x = 1")
    held, closed = [], []
    retire = scratch_storage.retire_cell
    make = scratch_storage.scratch_manager

    def recorded(*args):
        held.append(_daemon_lock_held(project))
        return retire(*args)

    def manager(directory):
        made = make(directory)
        close = made.artifact_store.close
        monkeypatch.setattr(made.artifact_store, "close", lambda: closed.append(made) or close())
        return made

    monkeypatch.setattr(scratch_storage, "retire_cell", recorded)
    monkeypatch.setattr(scratch_storage, "scratch_manager", manager)
    assert main(["scratch", "retire", a, "--root", str(project)]) == 0
    assert held == [True]
    assert len(closed) == 1


def test_scratch_commands_report_a_daemon_not_answering(project, monkeypatch, capsys):
    from strata import scratch

    scratch._scaffold(project)
    (a,) = _cells(scratch.scratch_dir(project), "x = 1")
    scratch.register(project.resolve(), store_bytes=0, over_by_bytes=0, used=True)

    def not_answering(anchor):
        raise scratch.ScratchError("the scratchpad's server is not answering")

    monkeypatch.setattr(scratch, "find_daemon", not_answering)
    assert main(["scratch", "ls", "--format", "json"]) == 0
    (listed,) = json.loads(capsys.readouterr().out)
    assert listed["running"] is True
    for command in (["retire", a], ["rm"]):
        assert main(["scratch", *command, "--root", str(project)]) == 2
        assert "not answering" in capsys.readouterr().err
    assert scratch.scratch_dir(project).is_dir()


def test_gc_all_sweeps_the_rest_past_a_broken_scratchpad(project, tmp_path, capsys):
    from strata import scratch

    scratch._scaffold(project)
    # Listed first, and half made: no notebook.toml.
    broken = tmp_path / "another"
    (scratch.scratch_dir(broken) / ".strata").mkdir(parents=True)
    for anchor in (broken, project):
        scratch.register(anchor.resolve(), store_bytes=0, over_by_bytes=0, used=True)

    assert main(["scratch", "gc", "--all"]) == 2
    captured = capsys.readouterr()
    assert "has no scratchpad" in captured.err
    assert str(project.resolve()) in json.loads(captured.out)

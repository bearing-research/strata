"""A mount that changes after a run was keyed: the run keys on the listing it read."""

from __future__ import annotations

import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _session(tmp_path: Path, cells: dict[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Drift")
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def _data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("a")
    return data


def _changes_before_prepare(monkeypatch, executor: CellExecutor, data: Path) -> None:
    """The mount gains a file after the key was made, before the run prepares it."""
    prepare = executor._prepare_mounts

    async def moved_first(mount_specs, cell_id=None):
        added = data / "b.txt"
        # Not an upstream's run, which prepares no mounts before the cell is keyed.
        if mount_specs and not added.exists():
            added.write_text("b")
        return await prepare(mount_specs, cell_id)

    monkeypatch.setattr(executor, "_prepare_mounts", moved_first)


def _status(session: NotebookSession, cell_id: str) -> str:
    return session.compute_staleness()[cell_id].status.value


@pytest.mark.asyncio
async def test_a_cell_run_keys_on_the_listing_it_read(tmp_path, monkeypatch):
    data = _data(tmp_path)
    source = f"# @mount data {data.as_uri()} ro\nn = len(list(data.iterdir()))\n"
    session = _session(tmp_path, {"c1": source, "c2": "m = n\n"})
    executor = CellExecutor(session)
    _changes_before_prepare(monkeypatch, executor, data)

    result = await executor.execute_cell("c1", source)

    assert result.success, result.error
    assert _status(session, "c1") == "ready"
    # Back to the listing the key was first made from: the stored result read two files.
    (data / "b.txt").unlink()
    assert _status(session, "c1") != "ready"


@pytest.mark.asyncio
async def test_a_loop_keys_on_the_listing_every_iteration_read(tmp_path, monkeypatch):
    data = _data(tmp_path)
    loop = (
        f"# @loop max_iter=2 carry=state\n# @mount data {data.as_uri()} ro\n"
        "state = state + len(list(data.iterdir()))\n"
    )
    session = _session(tmp_path, {"seed": "state = 0\n", "loop": loop, "use": "s = state\n"})
    executor = CellExecutor(session)
    _changes_before_prepare(monkeypatch, executor, data)

    result = await executor.execute_cell("loop", loop)

    assert result.success, result.error
    assert _status(session, "loop") == "ready"
    (data / "b.txt").unlink()
    assert _status(session, "loop") != "ready"


@pytest.mark.asyncio
async def test_a_run_all_batch_keys_on_the_view_it_read(tmp_path, monkeypatch):
    """A batch prepares before it keys. A remote mount's run reads the view prepared from
    one listing, so a change landing between the two must not rekey it."""
    from strata.notebook import ws

    data = _data(tmp_path)
    source = f"# @mount data {data.as_uri()} ro\nn = len(list(data.iterdir()))\n"
    session = _session(tmp_path, {"c1": source, "c2": "m = n\n"})
    executor = CellExecutor(session)
    prepare = executor._prepare_mounts

    async def view_then_change(mount_specs, cell_id=None):
        prepared = await prepare(mount_specs, cell_id)
        # What a remote mount's run reads: a copy built from the listing it fingerprinted.
        view = tmp_path / "view"
        shutil.copytree(data, view)
        prepared["data"] = replace(prepared["data"], local_path=view)
        (data / "b.txt").write_text("b")
        return prepared

    monkeypatch.setattr(executor, "_prepare_mounts", view_then_change)
    notebook_id = session.notebook_state.id

    batch = await ws._run_partition_batch(
        session=session,
        executor=executor,
        cells_in_run=[session.notebook_state.get_cell("c1")],
        notebook_id=notebook_id,
        force=False,
        execution_state=ws._ensure_execution_state(notebook_id),
    )

    assert [r.status for r in batch.cell_results] == ["ok"]
    # Keyed on the one file its view held, not the two the source now lists.
    assert _status(session, "c1") != "ready"
    (data / "b.txt").unlink()
    assert _status(session, "c1") == "ready"

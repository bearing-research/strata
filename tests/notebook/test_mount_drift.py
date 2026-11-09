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


def _writes_once(data: Path, flag: Path) -> str:
    """Code that rewrites ``a.txt`` in the live folder, once, as another writer would."""
    return (
        f"import os\nif os.path.exists({str(flag)!r}):\n"
        f"    os.remove({str(flag)!r})\n"
        f"    open({str(data / 'a.txt')!r}, 'w').write('changed')\n"
    )


@pytest.mark.asyncio
async def test_a_file_written_while_the_cell_ran_is_never_served_after_it_is_put_back(tmp_path):
    data = _data(tmp_path)
    flag = tmp_path / "flag"
    flag.touch()
    source = f"# @mount data {data.as_uri()} ro\nn = (data / 'a.txt').read_text()\n" + _writes_once(
        data, flag
    )
    session = _session(tmp_path, {"c1": source, "c2": "m = n\n"})
    executor = CellExecutor(session)

    moved = await executor.execute_cell("c1", source)

    assert moved.success, moved.error
    assert moved.moved_input == "data/a.txt"
    assert "data/a.txt changed while the cell ran" in moved.stderr
    assert _status(session, "c1") != "ready"
    # Back to what the run was keyed on: a hit would serve outputs made over other bytes.
    (data / "a.txt").write_text("a")
    assert _status(session, "c1") != "ready"
    again = await executor.execute_cell("c1", source)
    assert again.success and again.cache_hit is False
    assert again.moved_input is None
    assert _status(session, "c1") == "ready"


@pytest.mark.asyncio
async def test_a_run_all_batch_keeps_a_moved_run_for_that_run_only(tmp_path):
    from strata.notebook import ws

    data = _data(tmp_path)
    flag = tmp_path / "flag"
    flag.touch()
    source = f"# @mount data {data.as_uri()} ro\nn = (data / 'a.txt').read_text()\n" + _writes_once(
        data, flag
    )
    session = _session(tmp_path, {"c1": source, "c2": "m = n\n"})
    executor = CellExecutor(session)
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
    assert "data/a.txt changed while the cell ran" in batch.cell_results[0].stderr
    (data / "a.txt").write_text("a")
    assert _status(session, "c1") != "ready"
    again = await executor.execute_cell("c1", source)
    assert again.success and again.cache_hit is False


@pytest.mark.asyncio
async def test_an_each_item_written_while_it_ran_keeps_the_manifest_out_of_reach(tmp_path):
    data = _data(tmp_path)
    (data / "b.txt").write_text("b")
    flag, log = tmp_path / "flag", tmp_path / "ran.log"
    flag.touch()
    source = (
        f"# @mount data {data.as_uri()} ro\n# @each path in data\nout = path.read_text()\n"
        f"open({str(log)!r}, 'a').write(path.name + '\\n')\n" + _writes_once(data, flag)
    )
    session = _session(tmp_path, {"c1": source, "c2": "seen = dict(out)\n"})
    executor = CellExecutor(session)

    moved = await executor.execute_cell("c1", source)

    assert moved.success, moved.error
    # Named in the mount it was listed from, not by the instance's binding.
    assert moved.moved_input == "data/a.txt"
    (data / "a.txt").write_text("a")
    assert _status(session, "c1") != "ready"
    log.write_text("")
    again = await executor.execute_cell("c1", source)
    assert again.success and again.moved_input is None
    # b.txt's instance read a still item: the folder it was listed from is not its key.
    assert log.read_text().split() == ["a.txt"]
    assert _status(session, "c1") == "ready"


@pytest.mark.asyncio
async def test_an_each_instance_checks_the_cells_other_mounts(tmp_path):
    data = _data(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    (other / "c.txt").write_text("c")
    flag = tmp_path / "flag"
    flag.touch()
    source = (
        f"# @mount data {data.as_uri()} ro\n# @mount other {other.as_uri()} ro\n"
        "# @each path in data\nout = path.read_text()\n"
        f"import os\nif os.path.exists({str(flag)!r}):\n"
        f"    os.remove({str(flag)!r})\n"
        f"    open({str(other / 'c.txt')!r}, 'w').write('changed')\n"
    )
    session = _session(tmp_path, {"c1": source, "c2": "seen = dict(out)\n"})

    moved = await CellExecutor(session).execute_cell("c1", source)

    assert moved.success, moved.error
    assert moved.moved_input == "other/c.txt"
    assert _status(session, "c1") != "ready"


@pytest.mark.asyncio
async def test_a_loop_whose_mount_moved_keeps_its_outputs_for_that_run_only(tmp_path):
    data = _data(tmp_path)
    flag = tmp_path / "flag"
    flag.touch()
    source = (
        f"# @mount data {data.as_uri()} ro\n# @loop max_iter=2 carry=state\n"
        "state = state + len((data / 'a.txt').read_text())\n" + _writes_once(data, flag)
    )
    session = _session(tmp_path, {"s": "state = 0\n", "c1": source, "c2": "m = state\n"})
    executor = CellExecutor(session)
    await executor.execute_cell("s", "state = 0\n")

    moved = await executor.execute_cell("c1", source)

    assert moved.success, moved.error
    assert moved.moved_input == "data/a.txt"
    assert "data/a.txt changed while the cell ran" in moved.stderr
    (data / "a.txt").write_text("a")
    assert _status(session, "c1") != "ready"
    again = await executor.execute_cell("c1", source)
    assert again.success and again.cache_hit is False
    assert again.moved_input is None
    assert _status(session, "c1") == "ready"


@pytest.mark.asyncio
async def test_a_moved_run_says_which_mount_moved_once_the_file_is_put_back(tmp_path):
    from strata.notebook.staleness_explanations import CausalityType

    data = _data(tmp_path)
    flag = tmp_path / "flag"
    flag.touch()
    source = f"# @mount data {data.as_uri()} ro\nn = (data / 'a.txt').read_text()\n" + _writes_once(
        data, flag
    )
    session = _session(tmp_path, {"c1": source, "c2": "m = n\n"})
    await CellExecutor(session).execute_cell("c1", source)
    (data / "a.txt").write_text("a")

    assert _status(session, "c1") != "ready"
    details = session.causality_map["c1"].details
    assert [(d.type, d.input_name) for d in details] == [(CausalityType.EXTERNAL_CHANGED, "data")]


@pytest.mark.asyncio
async def test_a_batched_stochastic_cell_keeps_every_run(tmp_path):
    from vestigo.provenance import derive_subkey

    from strata.notebook import ws

    source = "# @stochastic\nimport random\nsample = random.random()\n"
    session = _session(tmp_path, {"c1": source, "c2": "m = sample\n"})
    executor = CellExecutor(session)
    notebook_id = session.notebook_state.id

    for _ in range(2):
        batch = await ws._run_partition_batch(
            session=session,
            executor=executor,
            cells_in_run=[session.notebook_state.get_cell("c1")],
            notebook_id=notebook_id,
            force=False,
            execution_state=ws._ensure_execution_state(notebook_id),
        )
        assert [r.status for r in batch.cell_results] == ["ok"]

    manager = session.get_artifact_manager()
    store = manager.artifact_store
    sample_id = manager.cell_artifact_id("c1", "sample")
    rows = [store.get_artifact(sample_id, v) for v in (1, 2)]
    assert [r.state for r in rows] == ["ready", "ready"], "both runs persist"
    assert rows[0].provenance_hash != rows[1].provenance_hash
    key = session.notebook_state.get_cell("c1").last_provenance_hash
    assert store.find_by_provenance(derive_subkey(key, "sample")) is None


@pytest.mark.asyncio
async def test_a_moved_leafs_displays_are_not_served_after_the_file_is_put_back(tmp_path):
    data = _data(tmp_path)
    flag = tmp_path / "flag"
    flag.touch()
    source = (
        f"# @mount data {data.as_uri()} ro\n"
        "display(Markdown((data / 'a.txt').read_text()))\n" + _writes_once(data, flag)
    )
    session = _session(tmp_path, {"c1": source})
    executor = CellExecutor(session)

    moved = await executor.execute_cell("c1", source)

    assert moved.success, moved.error
    assert moved.moved_input == "data/a.txt"
    (data / "a.txt").write_text("a")
    again = await executor.execute_cell("c1", source)
    assert again.success and again.cache_hit is False

"""A relative local mount path is under the notebook's directory, not the server's cwd."""

from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, MountSpec
from strata.notebook.mounts import local_mount_path
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)

SOURCE = "# @mount shared ../shared ro\ntext = (shared / 'a.txt').read_text()\n"


def test_a_relative_path_is_under_the_notebook():
    assert local_mount_path("../shared", Path("/nb")) == Path("/nb/../shared")
    assert local_mount_path("/data", Path("/nb")) == Path("/data")
    with pytest.raises(ValueError, match="needs the notebook's directory"):
        local_mount_path("shared", None)


def _checkout(root: Path) -> Path:
    """A notebook beside the ``shared`` folder its cell mounts as ``../shared``."""
    (root / "shared").mkdir(parents=True)
    (root / "shared" / "a.txt").write_text("one")
    nb_dir = create_notebook(root, "rel")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", SOURCE)
    add_cell_to_notebook(nb_dir, "c2", after_cell_id="c1")
    write_cell(nb_dir, "c2", "shown = text\n")
    return nb_dir


@pytest.mark.asyncio
async def test_a_cell_reads_its_relative_mount_from_any_server_cwd(tmp_path, monkeypatch):
    nb_dir = _checkout(tmp_path / "repo")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # where a relative path resolved before
    session = SessionManager().open_notebook(nb_dir)
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", SOURCE)
    assert first.success, first.error
    assert first.outputs["text"]["preview"] == "one"

    (tmp_path / "repo" / "shared" / "a.txt").write_text("two, longer")
    assert session.compute_staleness()["c1"].status == CellStatus.STALE
    again = await executor.execute_cell("c1", SOURCE)
    assert not again.cache_hit
    assert again.outputs["text"]["preview"] == "two, longer"


def test_two_checkouts_key_a_relative_mount_alike(tmp_path):
    first = _checkout(tmp_path / "a")
    second = tmp_path / "b" / first.name
    shutil.copytree(tmp_path / "a", tmp_path / "b")

    def mounts(nb_dir: Path) -> list[str]:
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        cell = session.notebook_state.get_cell("c1")
        return session._outside_world_for(cell).mount_fingerprints

    assert mounts(first) == mounts(second)
    (tmp_path / "b" / "shared" / "a.txt").write_text("changed")
    assert mounts(first) != mounts(second)


@pytest.mark.asyncio
async def test_a_sql_cell_reads_a_relative_mount(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    (tmp_path / "raw").mkdir()
    pq.write_table(pa.table({"k": [1, 2]}), tmp_path / "raw" / "a.parquet")
    nb_dir = create_notebook(tmp_path, "rel_sql")
    source = "-- @sql connection=lake\n-- @cache forever\nSELECT sum(k) AS total FROM raw\n"
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    write_cell(nb_dir, "c1", source)
    update_notebook_mounts(nb_dir, [MountSpec(name="raw", uri="../raw")])
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.lake]\ndriver = "duckdb"\npath = ":memory:"\n'
        'mounts = ["raw"]\n'
    )
    (tmp_path / "elsewhere" / "deeper").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "elsewhere" / "deeper")  # where ../raw does not exist
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success, result.error
    art_id, version = result.artifact_uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
    blob = session.get_artifact_manager().load_artifact_data(art_id, int(version))
    assert pa.ipc.open_stream(blob).read_all().to_pylist() == [{"total": 3}]

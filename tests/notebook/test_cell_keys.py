"""A cell's key on cellgraph keeps what the key was before: the inputs a widget folds in,
the cell under a DAG error, and an input an edit adds after staleness fingerprinted."""

from __future__ import annotations

from pathlib import Path

import cellgraph as cg
import pytest

from strata.notebook.env import local_packages
from strata.notebook.executor import CellExecutor
from strata.notebook.keys import cell_key, input_parts
from strata.notebook.parser import parse_notebook
from strata.notebook.provenance import (
    compute_source_hash,
)
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _session(tmp_path: Path, cells: list[tuple[str, str, str]], toml: str = "") -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "Keys")
    previous: str | None = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=previous, language=language)
        write_cell(notebook_dir, cell_id, source)
        previous = cell_id
    if toml:
        path = notebook_dir / "notebook.toml"
        path.write_text(path.read_text() + toml)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _mount(tmp_path: Path) -> str:
    data = tmp_path / "data"
    data.mkdir()
    (data / "f.txt").write_text("hi")
    return f'\n[[mounts]]\nname = "d"\nuri = "file://{data}"\nmode = "ro"\n'


def test_a_widget_key_covers_a_mount_declared_on_it(tmp_path):
    """A widget is keyed over its parts and every fingerprint staleness draws."""
    session = _session(
        tmp_path, [("w", "widget", "a = slider(0, 1, default=0.5)\n")], _mount(tmp_path)
    )
    cell = session.notebook_state.get_cell("w")
    cell.mount_overrides = list(cell.mounts)
    outside = session._outside_world_for(cell)
    assert outside.mount_fingerprints, "the mount reaches the widget"
    env_hash = session.cell_env_hash(cell, local_packages(session.path))
    fingerprints = [
        outside.mount_fingerprints,
        outside.table_fingerprints,
        outside.fetch_fingerprints,
        outside.dataset_fingerprints,
    ]
    key = cell_key(
        session,
        session.analysis,
        "w",
        mounts=outside.mount_fingerprints,
        tables=outside.table_fingerprints,
        fetches=outside.fetch_fingerprints,
        datasets=outside.dataset_fingerprints,
        env_hash=env_hash,
    )
    assert key.parts is not None
    assert key.provenance_hash == cg.parts_digest(key.parts)
    assert key.parts["definition_key"] == compute_source_hash(cell.source, "widget")
    kinds = dict(zip(("mount", "table", "fetch", "dataset"), fingerprints, strict=True))
    assert key.parts["inputs"] == input_parts(kinds)


_CYCLE = "# @after c1\nx = 1\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("flushed", [False, True])
async def test_a_cell_under_a_dag_error_runs(tmp_path, flushed):
    """A source that makes a cycle runs, as before, keyed with nothing upstream."""
    session = _session(tmp_path, [("c0", "python", "x = 1\n"), ("c1", "python", "y = x\n")])
    if flushed:
        session.notebook_state.get_cell("c0").source = _CYCLE
        session.re_analyze_cell("c0")
        assert session.dag_error
    result = await CellExecutor(session).execute_cell("c0", _CYCLE)
    assert result.success, result.error


def test_staleness_reads_an_input_an_edit_added_after_its_prefetch(tmp_path):
    """Fingerprints gathered before an edit miss the mount it adds; the walk reads them."""
    data = tmp_path / "data"
    data.mkdir()
    session = _session(tmp_path, [("c0", "python", "x = 1\n")])
    prefetched = session._outside_world_fingerprints()
    assert prefetched["c0"].mount_fingerprints == []
    session.notebook_state.get_cell("c0").source = f"# @mount d file://{data}\nx = 1\n"
    session.re_analyze_cell("c0")

    staleness = session._compute_staleness_locked(prefetched, local_packages(session.path))
    assert "c0" in staleness

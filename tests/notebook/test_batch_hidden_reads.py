"""A read the analysis cannot see behaves in run-all as it does in a cell run alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# ``mu`` reaches c2 only through a string, so c2 has no edge from c1.
CELLS = [("c1", "mu = 2\n"), ("c2", 'out = eval("mu")\n'), ("c3", "print(out)\n")]


def _session(tmp_path: Path) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "HiddenReads")
    prev: str | None = None
    for cell_id, source in CELLS:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _specs(session: NotebookSession) -> list[dict]:
    consumed = session.dag.consumed_variables if session.dag else {}
    return [
        {
            "cell_id": cell_id,
            "source": source,
            "consumed_vars": sorted(consumed.get(cell_id, set())),
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }
        for cell_id, source in CELLS
    ]


@pytest.mark.asyncio
async def test_alone_a_hidden_read_fails(tmp_path: Path):
    session = _session(tmp_path)
    executor = CellExecutor(session)
    assert (await executor.execute_cell("c1", CELLS[0][1])).success
    alone = await executor.execute_cell("c2", CELLS[1][1])
    assert not alone.success
    assert "mu" in (alone.error or "")


@pytest.mark.asyncio
async def test_run_all_does_not_cache_a_value_read_through_a_hidden_name(tmp_path: Path):
    session = _session(tmp_path)
    executor = CellExecutor(session)
    await executor.execute_batch(_specs(session))

    write_cell(session.path, "c1", "mu = 3\n")
    session.reload()
    assert (await executor.execute_cell("c1", "mu = 3\n")).success

    after = await executor.execute_cell("c3", CELLS[2][1])
    assert "2" not in after.stdout.split(), (
        f"c3 printed the value c2 read before mu changed: {after.stdout!r}"
    )

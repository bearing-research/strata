"""A ``-- @stochastic`` SQL cell keeps every run, as a Python one does.

Each run's result is under its own run, so a sampling query's draws sit side by side in the
ledger and none of them is served as a cache hit.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("adbc_driver_sqlite")

from vestigo.provenance import derive_subkey  # noqa: E402

from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell  # noqa: E402

SAMPLE = (
    "-- @sql connection=db\n-- @name draw\n-- @stochastic\n"
    "SELECT x FROM t ORDER BY random() LIMIT 1\n"
)


def _session(tmp_path: Path) -> NotebookSession:
    db = tmp_path / "t.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.executemany("INSERT INTO t VALUES (?)", [(i,) for i in range(100)])
    notebook_dir = create_notebook(tmp_path / "nb", "Sample")
    add_cell_to_notebook(notebook_dir, "c1", None, language="sql")
    write_cell(notebook_dir, "c1", SAMPLE)
    toml = notebook_dir / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db}"\n')
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


async def test_a_stochastic_sql_cell_keeps_every_run(tmp_path):
    session = _session(tmp_path)
    executor = CellExecutor(session)

    for _ in range(2):
        result = await executor.execute_cell("c1", SAMPLE)
        assert result.success, result.error
        assert not result.cache_hit

    manager = session.get_artifact_manager()
    store = manager.artifact_store
    draw_id = manager.cell_artifact_id("c1", "draw")
    rows = [store.get_artifact(draw_id, v) for v in (1, 2)]
    assert [r.state for r in rows] == ["ready", "ready"], "both runs persist"
    assert rows[0].provenance_hash != rows[1].provenance_hash
    # Neither is under the cell's own key, which a lookup would find.
    key = json.loads(rows[1].transform_spec)["params"]["source_hash"]
    assert store.find_by_provenance(derive_subkey(key, "draw")) is None

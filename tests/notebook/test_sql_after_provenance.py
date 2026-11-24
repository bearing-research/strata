"""A SQL cell's key covers the last run of each cell it runs `-- @after`.

A cell after a write reads what the write left. Under a policy that probes nothing
(`forever`, `session`, `ttl`, or a database whose probe cannot see a change), only the
upstream's run in the key tells a re-seeded database from the last one.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pyarrow as pa
import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell  # noqa: E402


def _seed(n: int) -> str:
    return (
        "-- @sql connection=db write=true\n"
        "DROP TABLE IF EXISTS t;\n"
        "CREATE TABLE t (n INTEGER);\n"
        f"INSERT INTO t VALUES ({n});\n"
    )


READ = "-- @sql connection=db\n-- @name rows\n-- @after seed\n-- @cache {policy}\nSELECT n FROM t\n"


def _notebook(tmp_path: Path, read: str) -> Path:
    nb_dir = create_notebook(tmp_path, "after_key", initialize_environment=False)
    add_cell_to_notebook(nb_dir, "seed", language="sql")
    write_cell(nb_dir, "seed", _seed(1))
    add_cell_to_notebook(nb_dir, "read", after_cell_id="seed", language="sql")
    write_cell(nb_dir, "read", read)
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.db]\ndriver = "sqlite"\npath = "w.db"\n')
    return nb_dir


def _open(nb_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


def _edit(session: NotebookSession, nb_dir: Path, cell_id: str, source: str) -> None:
    """As an editor's source update does."""
    write_cell(nb_dir, cell_id, source)
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    cell.source = source
    session.re_analyze_cell(cell_id)


def _read_rows(session: NotebookSession) -> list[int]:
    manager = session.get_artifact_manager()
    artifact_id = f"nb_{session.notebook_state.id}_cell_read_var_rows"
    latest = manager.artifact_store.get_latest_version(artifact_id)
    assert latest is not None
    blob = manager.load_artifact_data(artifact_id, latest.version)
    return pa.ipc.open_stream(blob).read_all().column("n").to_pylist()


def _value(nb_dir: Path) -> list[int]:
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn:
        return [n for (n,) in conn.execute("SELECT n FROM t")]


@pytest.mark.parametrize("policy", ["forever", "session"])
@pytest.mark.asyncio
async def test_a_reseeded_database_runs_the_cell_after_it_again(tmp_path, policy):
    nb_dir = _notebook(tmp_path, READ.format(policy=policy))
    session = _open(nb_dir)
    await _run(session, "seed")
    assert (await _run(session, "read")).cache_hit is False

    _edit(session, nb_dir, "seed", _seed(5))
    await _run(session, "seed")
    assert _value(nb_dir) == [5]

    assert (await _run(session, "read")).cache_hit is False
    assert _read_rows(session) == [5]


@pytest.mark.asyncio
async def test_a_seed_run_again_unchanged_runs_the_cell_after_it_again(tmp_path):
    """A seed under `session` runs again in a new session and drops what came after it."""
    seed = _seed(1).replace("true\n", "true\n-- @cache session\n")
    nb_dir = _notebook(tmp_path, READ.format(policy="forever"))
    write_cell(nb_dir, "seed", seed)
    first = _open(nb_dir)
    await _run(first, "seed")
    await _run(first, "read")

    second = _open(nb_dir)
    assert (await _run(second, "seed")).cache_hit is False
    assert (await _run(second, "read")).cache_hit is False


@pytest.mark.asyncio
async def test_a_cell_after_itself_keeps_its_cache(tmp_path):
    """No edge, as cellgraph reads it, so its own last run is not in its key."""
    nb_dir = _notebook(
        tmp_path, READ.format(policy="forever").replace("@after seed", "@after read")
    )
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "read")

    assert (await _run(session, "read")).cache_hit is True


def test_a_cell_after_itself_has_no_after_part(tmp_path):
    from strata.notebook.sql.cell_executor import _load_upstream_variables

    nb_dir = _notebook(tmp_path, READ.format(policy="forever"))
    _, hashes, _ = _load_upstream_variables(_open(nb_dir), "read", [], ["read", "seed"])

    assert list(hashes) == ["@after:seed"]


@pytest.mark.asyncio
async def test_an_unchanged_seed_keeps_the_cell_after_it(tmp_path):
    """Across sessions too, now a seed's key is its body (#1672)."""
    nb_dir = _notebook(tmp_path, READ.format(policy="forever"))
    first = _open(nb_dir)
    await _run(first, "seed")
    await _run(first, "read")

    second = _open(nb_dir)
    assert (await _run(second, "seed")).cache_hit is True
    assert (await _run(second, "read")).cache_hit is True


@pytest.mark.asyncio
async def test_a_write_after_a_reseeded_database_runs_again(tmp_path):
    """Its rows were in the old database: a re-seed drops them, so it writes them again."""
    nb_dir = _notebook(
        tmp_path, "-- @sql connection=db write=true\n-- @after seed\nINSERT INTO t VALUES (2);\n"
    )
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "read")

    _edit(session, nb_dir, "seed", _seed(5))
    await _run(session, "seed")

    assert (await _run(session, "read")).cache_hit is False
    assert sorted(_value(nb_dir)) == [2, 5]


@pytest.mark.asyncio
async def test_the_after_of_the_source_run_counts_before_the_session_has_it(tmp_path):
    """A run before the editor's flush carries its own source; its `@after` is the one."""
    with_after = READ.format(policy="forever")
    nb_dir = _notebook(tmp_path, with_after.replace("-- @after seed\n", ""))
    session = _open(nb_dir)
    executor = CellExecutor(session)
    await _run(session, "seed")
    assert (await executor.execute_cell("read", with_after)).success

    _edit(session, nb_dir, "seed", _seed(5))
    await _run(session, "seed")

    again = await executor.execute_cell("read", with_after)
    assert again.success and again.cache_hit is False


def _setup(n: int, *, nocache: bool = False) -> str:
    """A Python cell that writes the database and stores no variable."""
    return (
        ("# @nocache\n" if nocache else "")
        + "import sqlite3\n"
        + "with sqlite3.connect('../w.db') as c:\n"
        + "    c.execute('DROP TABLE IF EXISTS t')\n"
        + "    c.execute('CREATE TABLE t (n INTEGER)')\n"
        + f"    c.execute('INSERT INTO t VALUES ({n})')\n"
        + "print('seeded')\n"
    )


def _after_setup(tmp_path: Path, setup: str) -> Path:
    nb_dir = _notebook(
        tmp_path, READ.format(policy="forever").replace("@after seed", "@after setup")
    )
    add_cell_to_notebook(nb_dir, "setup", language="python")
    write_cell(nb_dir, "setup", setup)
    return nb_dir


@pytest.mark.asyncio
async def test_a_python_setup_run_again_runs_the_cell_after_it_again(tmp_path):
    """Run again, even unchanged, it rebuilt what the cell after it reads, and a write
    after it would be gone."""
    nb_dir = _after_setup(tmp_path, _setup(7))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")
    assert _read_rows(session) == [7]
    assert (await _run(session, "read")).cache_hit is True
    rerun = await CellExecutor(session).execute_cell_rerun("setup", _setup(7))
    assert rerun.success
    assert (await _run(session, "read")).cache_hit is False

    _edit(session, nb_dir, "setup", _setup(8))
    await _run(session, "setup")

    assert (await _run(session, "read")).cache_hit is False
    assert _read_rows(session) == [8]


@pytest.mark.asyncio
async def test_a_nocache_setup_runs_the_cell_after_it_again_each_time(tmp_path):
    """A normal run runs the `@nocache` setup first, and so this one after it."""
    nb_dir = _after_setup(tmp_path, _setup(7, nocache=True))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")

    assert (await _run(session, "read")).cache_hit is False

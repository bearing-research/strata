"""A write cell with no ``-- @cache`` is reused in its session, and after, while its file stands.

In the session that ran it, as before. In a later one, for the only write cell on a
SQLite file, while the file is exactly as its run left it: a reopen finds it ready, and
running it again changes nothing. A database deleted or changed since runs it again, and
so does a second write cell on the database.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.models import CellStatus, ConnectionSpec  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.sql.cell_executor import (  # noqa: E402
    _write_connection_key,
    _write_probed,
)
from strata.notebook.sql.registry import get_adapter  # noqa: E402
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell  # noqa: E402

SEED = (
    "-- @sql connection=db write=true\n"
    "DROP TABLE IF EXISTS t;\n"
    "CREATE TABLE t (n INTEGER);\n"
    "INSERT INTO t VALUES (1);\n"
)
MORE = "-- @sql connection=db write=true\nINSERT INTO t VALUES (2);\n"


def _notebook(tmp_path: Path, *cells: tuple[str, str]) -> Path:
    nb_dir = create_notebook(tmp_path, "write_reopen", initialize_environment=False)
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(nb_dir, cell_id, after_cell_id=after, language="sql")
        write_cell(nb_dir, cell_id, source)
        after = cell_id
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.db]\ndriver = "sqlite"\npath = "w.db"\n')
    return nb_dir


def _open(nb_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    return session


def _source(session: NotebookSession, cell_id: str) -> str:
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    return cell.source


async def _run(session: NotebookSession, cell_id: str):
    result = await CellExecutor(session).execute_cell(cell_id, _source(session, cell_id))
    assert result.success, result.error
    return result


def _rows(nb_dir: Path) -> list[int]:
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn:
        return [n for (n,) in conn.execute("SELECT n FROM t ORDER BY n")]


@pytest.mark.asyncio
async def test_a_reopen_finds_it_ready_and_a_run_changes_nothing(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")

    reopened = _open(nb_dir)
    assert reopened.compute_staleness()["seed"].status == CellStatus.READY

    again = await _run(reopened, "seed")
    assert again.cache_hit is True


@pytest.mark.asyncio
async def test_a_deleted_database_runs_it_again(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")
    (nb_dir / "w.db").unlink()

    reopened = _open(nb_dir)
    assert reopened.compute_staleness()["seed"].status != CellStatus.READY

    again = await _run(reopened, "seed")
    assert again.cache_hit is False and _rows(nb_dir) == [1]


@pytest.mark.asyncio
async def test_a_database_changed_by_hand_runs_it_again(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn, conn:
        conn.execute("DELETE FROM t")

    reopened = _open(nb_dir)
    assert reopened.compute_staleness()["seed"].status != CellStatus.READY

    again = await _run(reopened, "seed")
    assert again.cache_hit is False and _rows(nb_dir) == [1]


@pytest.mark.asyncio
async def test_an_explicit_session_policy_still_runs_once_per_session(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED.replace("true\n", "true\n-- @cache session\n")))
    await _run(_open(nb_dir), "seed")

    reopened = _open(nb_dir)
    assert reopened.compute_staleness()["seed"].status != CellStatus.READY

    again = await _run(reopened, "seed")
    assert again.cache_hit is False


@pytest.mark.asyncio
async def test_in_its_own_session_a_later_write_does_not_undo_the_hit(tmp_path):
    """As before: the session that ran it reuses it, whatever moved the file since."""
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("more", MORE))
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "more")

    assert (await _run(session, "seed")).cache_hit is True
    assert _rows(nb_dir) == [1, 2]


@pytest.mark.asyncio
async def test_after_a_later_write_a_reopen_runs_both_again(tmp_path):
    """Only the last write left the file as it is, and the seed before it is to run again."""
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("more", MORE))
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "more")

    staleness = _open(nb_dir).compute_staleness()

    assert staleness["seed"].status != CellStatus.READY
    assert staleness["more"].status != CellStatus.READY


@pytest.mark.parametrize(
    ("spec", "probed"),
    [
        (ConnectionSpec(name="db", driver="sqlite", path="w.db"), True),
        # A restart empties it under the same token.
        (ConnectionSpec(name="db", driver="sqlite", path=":memory:"), False),
        (ConnectionSpec(name="db", driver="sqlite", uri="file:w?mode=memory&cache=shared"), False),
        # Its token can match two different states.
        (ConnectionSpec(name="db", driver="duckdb", path="w.duckdb"), False),
        (ConnectionSpec(name="db", driver="postgresql", uri="postgresql://h/d"), False),
    ],
)
def test_only_a_sqlite_file_is_probed_for_reuse(spec, probed):
    assert _write_probed(spec) is probed


def _settle(session: NotebookSession, cell_id: str) -> CellStatus:
    """The cell's status once staleness looks again, as after an edit undone."""
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    cell.status = CellStatus.STALE
    return session.compute_staleness()[cell_id].status


@pytest.mark.asyncio
async def test_in_its_own_session_it_settles_ready_after_a_later_write(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("more", MORE))
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "more")

    assert _settle(session, "seed") == CellStatus.READY


@pytest.mark.asyncio
async def test_a_database_no_probe_judges_is_ready_in_its_session_only(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text().replace('path = "w.db"', f'uri = "file:{nb_dir / "w.db"}"'))
    session = _open(nb_dir)
    await _run(session, "seed")

    assert _settle(session, "seed") == CellStatus.READY
    assert _open(nb_dir).compute_staleness()["seed"].status != CellStatus.READY


@pytest.mark.asyncio
async def test_runs_in_two_sessions_share_one_definition(tmp_path):
    """The session and state a run records are not part of what the cell is."""
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    first = _open(nb_dir)
    await _run(first, "seed")
    store = first.get_artifact_manager().artifact_store
    artifact_id = f"nb_{first.notebook_state.id}_cell_seed_var_result"
    before = store.get_latest_version(artifact_id)
    (nb_dir / "w.db").unlink()  # so the next session runs it again

    second = _open(nb_dir)
    assert (await _run(second, "seed")).cache_hit is False
    after = second.get_artifact_manager().artifact_store.get_latest_version(artifact_id)

    assert before is not None and after is not None and after.version != before.version
    assert store.definition_of(artifact_id, after.version) == store.definition_of(
        artifact_id, before.version
    )
    assert store.definition_of(artifact_id, before.version) is not None


@pytest.mark.asyncio
async def test_after_a_reopen_both_run_again(tmp_path):
    """The later write's file matches, but the seed before it runs again: it runs too."""
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("more", MORE))
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "more")
    reopened = _open(nb_dir)

    assert (await _run(reopened, "seed")).cache_hit is False
    assert (await _run(reopened, "more")).cache_hit is False
    assert _rows(nb_dir) == [1, 2]


@pytest.mark.asyncio
async def test_a_connection_pointed_elsewhere_is_not_ready(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    session = _open(nb_dir)
    await _run(session, "seed")
    (spec,) = session.notebook_state.connections
    session.notebook_state.connections = [spec.model_copy(update={"path": "elsewhere.db"})]

    assert _settle(session, "seed") != CellStatus.READY


@pytest.mark.asyncio
async def test_with_a_second_write_cell_a_reopen_runs_it_again(tmp_path):
    """The other write cell may have moved the file, or may undo this one's run."""
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("more", MORE))
    await _run(_open(nb_dir), "seed")

    reopened = _open(nb_dir)
    assert reopened.compute_staleness()["seed"].status != CellStatus.READY
    assert (await _run(reopened, "seed")).cache_hit is False


@pytest.mark.asyncio
async def test_a_write_cell_on_another_database_does_not_count(tmp_path):
    other = MORE.replace("connection=db", "connection=other")
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("elsewhere", other))
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.other]\ndriver = "sqlite"\npath = "o.db"\n')
    await _run(_open(nb_dir), "seed")

    assert _open(nb_dir).compute_staleness()["seed"].status == CellStatus.READY


@pytest.mark.asyncio
async def test_a_change_after_a_reopen_found_it_ready_runs_it_again(tmp_path):
    """A staleness pass holds nothing: the run probes again."""
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")
    session = _open(nb_dir)
    assert session.compute_staleness()["seed"].status == CellStatus.READY
    (nb_dir / "w.db").unlink()

    assert (await _run(session, "seed")).cache_hit is False
    assert _rows(nb_dir) == [1]


@pytest.mark.asyncio
async def test_a_seed_a_run_reused_is_held_for_the_session(tmp_path):
    """As a run made in the session is: a later change to the file does not undo it there."""
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")
    session = _open(nb_dir)
    assert (await _run(session, "seed")).cache_hit is True
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn, conn:
        conn.execute("INSERT INTO t VALUES (2)")

    assert (await _run(session, "seed")).cache_hit is True
    assert _rows(nb_dir) == [1, 2]


@pytest.mark.parametrize(
    "connection",
    [
        lambda nb_dir: f'path = "{nb_dir / "w.db"}"',  # the same file, spelled otherwise
        lambda nb_dir: f'uri = "file:{nb_dir / "other.db"}"',  # a uri may name any file
    ],
)
@pytest.mark.asyncio
async def test_a_writer_through_another_connection_counts(tmp_path, connection):
    other = MORE.replace("connection=db", "connection=other")
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("elsewhere", other))
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + f'\n[connections.other]\ndriver = "sqlite"\n{connection(nb_dir)}\n'
    )
    await _run(_open(nb_dir), "seed")

    assert _open(nb_dir).compute_staleness()["seed"].status != CellStatus.READY


@pytest.mark.asyncio
async def test_a_database_changed_since_is_explained_as_changed(tmp_path):
    nb_dir = _notebook(tmp_path, ("seed", SEED))
    await _run(_open(nb_dir), "seed")
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn, conn:
        conn.execute("DELETE FROM t")
    session = _open(nb_dir)
    session.compute_staleness()
    cell = session.notebook_state.get_cell("seed")
    assert cell is not None

    assert session.probed_identity_changed(cell) is True


def test_a_credentialed_connection_has_a_key_without_its_values(tmp_path):
    """Only the credential's name counts, so no resolver is needed and two names differ."""
    session = _open(_notebook(tmp_path))
    adapter = get_adapter("postgresql")
    spec = ConnectionSpec(name="db", driver="postgresql", uri="postgresql://h/d", credential="a")

    key = _write_connection_key(session, adapter, spec)

    assert key is not None
    assert key != _write_connection_key(
        session, adapter, spec.model_copy(update={"credential": "b"})
    )


@pytest.mark.asyncio
async def test_a_write_cell_on_a_database_in_memory_does_not_count(tmp_path):
    scratch = "-- @sql connection=scratch write=true\nCREATE TABLE s (n INTEGER);\n"
    nb_dir = _notebook(tmp_path, ("seed", SEED), ("scratch", scratch))
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.scratch]\ndriver = "sqlite"\npath = ":memory:"\n'
    )
    await _run(_open(nb_dir), "seed")

    assert _open(nb_dir).compute_staleness()["seed"].status == CellStatus.READY

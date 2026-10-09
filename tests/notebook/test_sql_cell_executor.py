"""End-to-end tests for the SQL cell executor (sqlite, real DB)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

adbc_sqlite = pytest.importorskip("adbc_driver_sqlite")


# --- fixtures -------------------------------------------------------------


def _seed_sqlite(path: Path) -> None:
    """Create a SQLite file with one table the cells can query."""
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)")
        conn.executemany(
            "INSERT INTO events (id, name, value) VALUES (?, ?, ?)",
            [(1, "alpha", 10), (2, "beta", 20), (3, "gamma", 30)],
        )
        conn.commit()


def _with_unresolved_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make analysis report a table named only at run time, like Snowflake ``IDENTIFIER($tbl)``."""
    import dataclasses

    from strata.notebook.sql import cell_executor

    real = cell_executor.analyze_sql_cell

    def analyze(source: str, **kwargs: Any) -> Any:
        return dataclasses.replace(real(source, **kwargs), unresolved_tables=["IDENTIFIER($tbl)"])

    monkeypatch.setattr(cell_executor, "analyze_sql_cell", analyze)


@pytest.mark.asyncio
async def test_a_table_named_at_run_time_is_never_served_from_cache(tmp_path, monkeypatch):
    """Under the default fingerprint policy, a table the analyzer cannot name forces a run.

    It is missing from the freshness token, so a cached result could outlive a
    change to it.
    """
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path, db_path=db_path, cell_source="# @sql connection=db\nSELECT * FROM events\n"
    )
    session = _make_session(nb_dir)
    _with_unresolved_table(monkeypatch)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    second = await execute_sql_cell(session, "c1", src)

    assert first["success"] and second["success"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is False


@pytest.mark.asyncio
async def test_a_declared_cache_policy_still_reuses_a_run_time_table(tmp_path, monkeypatch):
    """``# @cache session`` needs no probe, so a run-time table does not block reuse."""
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source="# @sql connection=db\n# @cache session\nSELECT * FROM events\n",
    )
    session = _make_session(nb_dir)
    _with_unresolved_table(monkeypatch)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    second = await execute_sql_cell(session, "c1", src)

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True


@pytest.mark.asyncio
async def test_an_edit_sqlglot_cannot_tell_apart_still_runs_the_new_query(tmp_path):
    """The cache key is the cell's own text: ``REAL`` and ``NUMERIC`` regenerate alike."""
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    real = "# @sql connection=db\n# @cache forever\nSELECT CAST('12' AS REAL) AS v\n"
    nb_dir = _build_notebook_with_sql_cell(tmp_path, db_path=db_path, cell_source=real)
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    first = await execute_sql_cell(session, "c1", real)
    edited = await execute_sql_cell(session, "c1", real.replace("REAL", "NUMERIC"))

    assert first["success"] and edited["success"], edited["error"]
    assert edited["cache_hit"] is False
    assert first["outputs"]["result"]["preview"] != edited["outputs"]["result"]["preview"]


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("SELECT :x AS v", [{"v": 41}]),
        ("SELECT :x + 1 AS v", [{"v": 42}]),
        ("WITH p AS (SELECT :x AS v) SELECT v FROM p", [{"v": 41}]),
        ("CREATE TABLE t AS SELECT :x AS v; INSERT INTO t SELECT :x + 1", None),
    ],
)
@pytest.mark.asyncio
async def test_a_duckdb_bind_in_the_select_list_runs(tmp_path, monkeypatch, body, expected):
    """sqlglot reads DuckDB ``SELECT :x`` as an alias; the cell is checked as it runs."""
    from strata.notebook.sql import cell_executor
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    monkeypatch.setattr(
        cell_executor,
        "_load_upstream_variables",
        lambda session, cell_id, references: ({"x": 41}, {"x": "upstream"}, {}),
    )
    nb_dir = create_notebook(tmp_path, "duckdb_bind")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    source = f"# @sql connection=db{'' if expected else ' write=true'}\n{body}\n"
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = "nb.duckdb"\n'
    )

    session = _make_session(nb_dir)
    result = await cell_executor.execute_sql_cell(session, "c1", source)

    assert result["success"], result["error"]
    if expected is None:
        import duckdb

        rows = duckdb.connect(str(nb_dir / "nb.duckdb")).execute("SELECT v FROM t").fetchall()
        assert rows == [(41,), (42,)]
    else:
        assert _load_arrow_from_uri(session, result["artifact_uri"]).to_pylist() == expected


def _build_notebook_with_sql_cell(
    tmp_path: Path,
    *,
    db_path: Path,
    cell_id: str = "c1",
    cell_source: str,
) -> Path:
    """Write a notebook directory with one [connections.db] and one SQL cell; returns its path."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "sql_e2e")
    add_cell_to_notebook(nb_dir, cell_id, language="sql")
    write_cell(nb_dir, cell_id, cell_source)

    # Append [connections.db] as text, since the writer rewrites the whole toml on
    # serialize.
    toml_path = nb_dir / "notebook.toml"
    text = toml_path.read_text()
    text += f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    toml_path.write_text(text)
    return nb_dir


def _make_session(nb_dir: Path) -> Any:
    """Parse the notebook and build a session (DAG + analyzer pass)."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    return NotebookSession(parse_notebook(nb_dir), nb_dir)


def _run(coro: Any) -> Any:
    """Run an async coroutine to completion."""
    return asyncio.get_event_loop().run_until_complete(coro)


# --- end-to-end execution -------------------------------------------------


@pytest.mark.asyncio
async def test_sql_cell_executes_and_returns_arrow_table(tmp_path):
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)

    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n"
            "# @cache forever\n"
            "SELECT id, name, value FROM events ORDER BY id\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))

    assert result["success"], result.get("error")
    assert result["cache_hit"] is False
    assert result["execution_method"] == "sql"
    assert result["artifact_uri"]

    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    assert table.num_rows == 3
    assert set(table.schema.names) == {"id", "name", "value"}
    rows = table.to_pylist()
    assert {r["name"] for r in rows} == {"alpha", "beta", "gamma"}


@pytest.mark.asyncio
async def test_sql_cell_cache_hit_on_unchanged_inputs(tmp_path):
    """``# @cache forever`` skips the probe; the second run returns the same artifact as a hit."""
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=("# @sql connection=db\n# @cache forever\nSELECT * FROM events\n"),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    second = await execute_sql_cell(session, "c1", src)

    assert first["success"] and second["success"]
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["execution_method"] == "cached"
    assert first["artifact_uri"] == second["artifact_uri"]


@pytest.mark.parametrize(
    "cell_source",
    [
        "# @sql connection=db\n# @cache forever\n# @nocache\nSELECT * FROM events\n",
        "# @sql connection=db write=true\n# @nocache\n"
        "INSERT INTO events (name, value) VALUES ('d', 40)\n",
    ],
    ids=["read-cache-forever", "write"],
)
@pytest.mark.asyncio
async def test_sql_cell_nocache_reruns_every_time(tmp_path, cell_source):
    """``# @nocache`` outranks every ``# @cache`` policy, including forever and write cells."""
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(tmp_path, db_path=db_path, cell_source=cell_source)
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    second = await execute_sql_cell(session, "c1", src)

    assert first["success"] and second["success"], (first.get("error"), second.get("error"))
    assert first["cache_hit"] is False
    assert second["cache_hit"] is False
    assert second["execution_method"] == "sql"
    if "INSERT" in cell_source:
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM events WHERE name = 'd'").fetchone() == (2,)


@pytest.mark.asyncio
async def test_sql_cell_fingerprint_invalidates_on_schema_change(tmp_path):
    """External DDL changes SQLite's ``schema_version``, so the next run re-executes.

    ``PRAGMA data_version`` also feeds the token but starts at 1 on every fresh
    connection, so DML across processes does not invalidate: a known limitation.
    """
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n"
            "# fingerprint is the default; spelling it out for clarity\n"
            "# @cache fingerprint\n"
            "SELECT * FROM events\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    assert first["success"], first.get("error")
    assert first["cache_hit"] is False

    # Mutate the schema from outside Strata.
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE events ADD COLUMN extra TEXT")
        conn.commit()

    second = await execute_sql_cell(session, "c1", src)
    assert second["success"], second.get("error")
    assert second["cache_hit"] is False, "fingerprint should have invalidated after the ALTER TABLE"

    table = _load_artifact_as_arrow(session, second["artifact_uri"])
    assert "extra" in table.schema.names


@pytest.mark.asyncio
async def test_sql_cell_read_only_enforces_no_writes(tmp_path):
    """An INSERT fails and leaves the DB untouched; the read-only connection is the boundary.

    ADBC reports the read-only write as a generic ``InternalError``, so the
    message text is not pinned.
    """
    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n"
            "# @cache forever\n"
            "INSERT INTO events (id, name, value) VALUES (99, 'hack', 1)\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    assert result["error"], "expected a non-empty error message"

    # Security boundary: the underlying DB row count is unchanged.
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert count == 3


@pytest.mark.asyncio
async def test_sql_cell_bind_param_from_upstream_python_cell(tmp_path):
    """A Python cell's variable binds into a SQL cell via ``:name``."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = create_notebook(tmp_path, "cross_lang")
    add_cell_to_notebook(nb_dir, "py", language="python")
    write_cell(nb_dir, "py", "min_value = 15\n")
    add_cell_to_notebook(nb_dir, "sql", after_cell_id="py", language="sql")
    write_cell(
        nb_dir,
        "sql",
        (
            "# @sql connection=db\n"
            "# @cache forever\n"
            "SELECT id, name, value FROM events WHERE value > :min_value ORDER BY id\n"
        ),
    )
    toml_path = nb_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text()
        + "\n[connections.db]\n"
        + 'driver = "sqlite"\n'
        + f'path = "{db_path}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()

    # Run the Python cell first so the upstream artifact exists.
    from strata.notebook.executor import CellExecutor

    executor = CellExecutor(session)
    py_src = (nb_dir / "cells" / "py.py").read_text()
    py_result = await executor.execute_cell("py", py_src)
    assert py_result.success, py_result.error

    # It resolves :min_value to 15 from the upstream artifact.
    from strata.notebook.sql.cell_executor import execute_sql_cell

    sql_src = (nb_dir / "cells" / "sql.py").read_text()
    result = await execute_sql_cell(session, "sql", sql_src)
    assert result["success"], result.get("error")

    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    rows = table.to_pylist()
    assert len(rows) == 2
    assert {r["name"] for r in rows} == {"beta", "gamma"}


@pytest.mark.asyncio
async def test_sql_cell_missing_connection_yields_clear_error(tmp_path):
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=tmp_path / "anywhere.db",
        cell_source=("# @sql connection=missing_one\nSELECT 1\n"),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    assert "missing_one" in (result["error"] or "")


@pytest.mark.asyncio
async def test_sql_cell_stays_ready_after_staleness_recompute(tmp_path):
    """SQL cells stay READY after a staleness recompute.

    Their artifacts sit under a SQL-specific hash that ``compute_staleness`` does
    not recompute, so the recorded generic provenance triplet is what keeps them READY.
    """
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import CellStatus

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=("# @sql connection=db\n# @cache forever\nSELECT * FROM events\n"),
    )
    session = _make_session(nb_dir)
    executor = CellExecutor(session)

    src = _read_cell(nb_dir, "c1")
    result = await executor.execute_cell("c1", src)
    assert result.success, result.error

    cell = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert cell.last_provenance_hash, (
        "record_successful_execution_provenance must persist last_provenance_hash for SQL cells"
    )

    # Mirror the route handler's post-execute sequence: compute
    # staleness, then mark_executed_ready bumps status to READY.
    session.compute_staleness()
    session.mark_executed_ready("c1")

    # A second staleness recompute (a reopen walks it too) must keep the cell READY.
    # The generic per-variable artifact lookup misses, since the SQL executor stores
    # under SQL-specific provenance, so can_preserve_uncached_ready needs its
    # language=='sql' branch.
    session.compute_staleness()
    cell_after = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert cell_after.status == CellStatus.READY, (
        f"SQL cell should stay READY after staleness recompute; got {cell_after.status!r}"
    )


@pytest.mark.asyncio
async def test_sql_cell_artifact_uri_visible_to_downstream_python(tmp_path):
    """``cell.artifact_uris`` is set after a SQL run so downstream provenance includes it.

    Otherwise the downstream hash ignores the SQL input and serves a stale value
    after the SQL data shifts.
    """
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = create_notebook(tmp_path, "downstream_sql")
    add_cell_to_notebook(nb_dir, "sql", language="sql")
    write_cell(
        nb_dir,
        "sql",
        ("# @sql connection=db\n# @cache forever\nSELECT name FROM events ORDER BY id\n"),
    )
    add_cell_to_notebook(nb_dir, "py", after_cell_id="sql", language="python")
    write_cell(
        nb_dir,
        "py",
        # Python cell consumes the SQL output.
        "names = [r['name'] for r in result.to_pylist()]\n",
    )
    toml_path = nb_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text()
        + "\n[connections.db]\n"
        + 'driver = "sqlite"\n'
        + f'path = "{db_path}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    executor = CellExecutor(session)

    sql_src = (nb_dir / "cells" / "sql.py").read_text()
    sql_result = await executor.execute_cell("sql", sql_src)
    assert sql_result.success, sql_result.error

    # The SQL cell's artifact must be discoverable from the upstream
    # cell-state map; this is what ``_collect_input_hashes`` walks.
    sql_cell = next(c for c in session.notebook_state.cells if c.id == "sql")
    assert sql_cell.artifact_uri, (
        "SQL cell artifact_uri must be set so downstream cells can find the upstream artifact"
    )
    assert "result" in sql_cell.artifact_uris

    # And the downstream Python cell's input hashes pick up the SQL artifact's
    # provenance hash.
    input_hashes = session._collect_input_hashes("py")
    assert len(input_hashes) == 1, (
        f"downstream py cell should see 1 upstream input hash; got {input_hashes!r}"
    )


@pytest.mark.asyncio
async def test_a_downstream_cell_gets_a_pyarrow_table_with_pandas_installed(tmp_path):
    """The reader turns untagged tables into pandas when it can; SQL output stays Arrow."""
    pytest.importorskip("pandas")
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = create_notebook(tmp_path, "sql_type")
    add_cell_to_notebook(nb_dir, "sql", language="sql")
    write_cell(nb_dir, "sql", "# @sql connection=db\nSELECT name FROM events ORDER BY id\n")
    add_cell_to_notebook(nb_dir, "py", after_cell_id="sql", language="python")
    write_cell(nb_dir, "py", "print(type(result).__module__, type(result).__name__)\n")
    toml_path = nb_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    sql_result = await executor.execute_cell("sql", _read_cell(nb_dir, "sql"))
    assert sql_result.success, sql_result.error
    py_result = await executor.execute_cell("py", _read_cell(nb_dir, "py"))
    assert py_result.success, py_result.error
    assert py_result.stdout.strip() == "pyarrow.lib Table"


@pytest.mark.asyncio
async def test_sql_cell_executor_dispatched_via_main_executor(tmp_path):
    """``CellExecutor`` dispatches ``language='sql'`` to the SQL path."""
    from strata.notebook.executor import CellExecutor

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n# @cache forever\nSELECT id, name FROM events ORDER BY id\n"
        ),
    )
    session = _make_session(nb_dir)
    executor = CellExecutor(session)

    src = _read_cell(nb_dir, "c1")
    result = await executor.execute_cell("c1", src)
    assert result.success, result.error
    assert result.execution_method == "sql"
    assert result.artifact_uri is not None


# --- helpers --------------------------------------------------------------


def _read_cell(nb_dir: Path, cell_id: str) -> str:
    return (nb_dir / "cells" / f"{cell_id}.py").read_text()


def _load_artifact_as_arrow(session: Any, uri: str) -> Any:
    """Pull the artifact bytes back and decode to a pyarrow Table."""
    import pyarrow as pa

    # ``strata://artifact/<id>@v=<n>``
    body = uri.removeprefix("strata://artifact/")
    art_id, version = body.rsplit("@v=", 1)
    artifact_mgr = session.get_artifact_manager()
    blob = artifact_mgr.load_artifact_data(art_id, int(version))
    return pa.ipc.open_stream(blob).read_all()


# --- # @sql write=true ----------------------------------------------------


@pytest.mark.asyncio
async def test_sql_write_cell_creates_table_and_inserts_rows(tmp_path):
    """A write cell runs each statement and still stores a status artifact for downstreams."""
    db_path = tmp_path / "fresh.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n"
            "# @cache session\n"
            "CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT);\n"
            "INSERT INTO events VALUES (1, 'alpha');\n"
            "INSERT INTO events VALUES (2, 'beta');\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    result = await execute_sql_cell(session, "c1", src)
    assert result["success"], result.get("error")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, label FROM events ORDER BY id").fetchall()
    assert rows == [(1, "alpha"), (2, "beta")]


@pytest.mark.asyncio
async def test_sql_write_cell_caches_within_session(tmp_path):
    """Write cells default to ``session`` caching, so a re-run in the same session hits."""
    db_path = tmp_path / "cache.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n"
            "CREATE TABLE t (n INTEGER);\n"
            "INSERT INTO t VALUES (1);\n"
        ),
    )
    session = _make_session(nb_dir)
    from strata.notebook.sql.cell_executor import execute_sql_cell

    src = _read_cell(nb_dir, "c1")
    first = await execute_sql_cell(session, "c1", src)
    assert first["success"]
    assert first["cache_hit"] is False

    second = await execute_sql_cell(session, "c1", src)
    assert second["success"]
    assert second["cache_hit"] is True
    # And the second call did NOT actually re-run the inserts.
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM t").fetchone()
        assert count == 1


@pytest.mark.asyncio
async def test_sql_write_cell_rejects_fingerprint_policy(tmp_path):
    """Probe-based policies do not apply to writes; the error is explicit, not a coercion."""
    db_path = tmp_path / "x.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n# @cache fingerprint\nCREATE TABLE t (n INTEGER);\n"
        ),
    )
    session = _make_session(nb_dir)
    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    assert "fingerprint" in (result["error"] or "").lower()


@pytest.mark.asyncio
async def test_sql_write_false_still_blocks_writes(tmp_path):
    """Without ``write=true`` the read-only enforcement still fires."""
    _seed_sqlite(tmp_path / "events.db")
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=tmp_path / "events.db",
        cell_source=(
            "# @sql connection=db\n"  # write flag NOT set
            "INSERT INTO events VALUES (99, 'sneak', 1)\n"
        ),
    )
    session = _make_session(nb_dir)
    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    with sqlite3.connect(tmp_path / "events.db") as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert count == 3


@pytest.mark.asyncio
async def test_sql_write_cell_makes_db_visible_to_read_cell(tmp_path):
    """A write cell creates the DB and a read cell queries it."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    db_path = tmp_path / "shared.db"
    nb_dir = create_notebook(tmp_path, "Write+Read")
    add_cell_to_notebook(nb_dir, "seed", language="sql")
    write_cell(
        nb_dir,
        "seed",
        (
            "# @sql connection=db write=true\n"
            "DROP TABLE IF EXISTS events;\n"
            "CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT);\n"
            "INSERT INTO events VALUES (1, 'alpha'), (2, 'beta');\n"
        ),
    )
    add_cell_to_notebook(nb_dir, "query", after_cell_id="seed", language="sql")
    write_cell(
        nb_dir,
        "query",
        (
            "# @sql connection=db\n"
            "# @cache fingerprint\n"
            "# @after seed\n"
            "SELECT id, label FROM events ORDER BY id\n"
        ),
    )
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    executor = CellExecutor(session)

    seed_src = (nb_dir / "cells" / "seed.py").read_text()
    seed_result = await executor.execute_cell("seed", seed_src)
    assert seed_result.success, seed_result.error

    query_src = (nb_dir / "cells" / "query.py").read_text()
    query_result = await executor.execute_cell("query", query_src)
    assert query_result.success, query_result.error
    table = _load_artifact_as_arrow(session, query_result.artifact_uri)
    assert table.to_pylist() == [
        {"id": 1, "label": "alpha"},
        {"id": 2, "label": "beta"},
    ]


# --- write cells: binds, invalidation, naming, commit, status ---


@pytest.mark.asyncio
async def test_sql_write_cell_resolves_bind_placeholders_from_upstream(tmp_path):
    """Write cells resolve ``:n`` binds against upstreams, as read cells do.

    Without binds the literal ``:n`` token would reach the driver.
    """
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    db_path = tmp_path / "binds.db"
    nb_dir = create_notebook(tmp_path, "Bind Write")
    add_cell_to_notebook(nb_dir, "cfg", language="python")
    write_cell(nb_dir, "cfg", "label = 'alpha'\ncount = 7\n")
    add_cell_to_notebook(nb_dir, "seed", after_cell_id="cfg", language="sql")
    write_cell(
        nb_dir,
        "seed",
        (
            "# @sql connection=db write=true\n"
            "DROP TABLE IF EXISTS t;\n"
            "CREATE TABLE t (label TEXT, n INTEGER);\n"
            "INSERT INTO t VALUES (:label, :count);\n"
        ),
    )
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    cfg_src = (nb_dir / "cells" / "cfg.py").read_text()
    cfg_result = await executor.execute_cell("cfg", cfg_src)
    assert cfg_result.success

    seed_src = (nb_dir / "cells" / "seed.py").read_text()
    seed_result = await executor.execute_cell("seed", seed_src)
    assert seed_result.success, seed_result.error

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT label, n FROM t").fetchall()
    assert rows == [("alpha", 7)]


@pytest.mark.asyncio
async def test_sql_write_cell_invalidates_on_upstream_value_change(tmp_path):
    """Same source with a different upstream value misses the cache."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    db_path = tmp_path / "invalidate.db"
    nb_dir = create_notebook(tmp_path, "Bind Invalidate")
    add_cell_to_notebook(nb_dir, "cfg", language="python")
    write_cell(nb_dir, "cfg", "value = 1\n")
    add_cell_to_notebook(nb_dir, "seed", after_cell_id="cfg", language="sql")
    write_cell(
        nb_dir,
        "seed",
        (
            "# @sql connection=db write=true\n"
            "DROP TABLE IF EXISTS t;\n"
            "CREATE TABLE t (n INTEGER);\n"
            "INSERT INTO t VALUES (:value);\n"
        ),
    )
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)
    cells = {c.id: c for c in session.notebook_state.cells}

    await executor.execute_cell("cfg", cells["cfg"].source)
    seed_src = (nb_dir / "cells" / "seed.py").read_text()
    first = await executor.execute_cell("seed", seed_src)
    assert first.success and not first.cache_hit

    # Change upstream value; rerun cfg + seed.
    cells["cfg"].source = "value = 99\n"
    (nb_dir / "cells" / "cfg.py").write_text(cells["cfg"].source)
    session.re_analyze_cell("cfg")
    await executor.execute_cell("cfg", cells["cfg"].source)

    second = await executor.execute_cell("seed", seed_src)
    assert second.success
    assert second.cache_hit is False, (
        "write cell must re-execute when an upstream bind variable changes"
    )
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT n FROM t").fetchall()
    assert rows == [(99,)]


@pytest.mark.asyncio
async def test_sql_write_cell_honors_at_name_for_artifact_key(tmp_path):
    """The write path keys its artifact by ``# @name``, matching the analyzer's ``defines``."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.sql.cell_executor import execute_sql_cell

    db_path = tmp_path / "named.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n# @name seed_status\nCREATE TABLE t (n INTEGER);\n"
        ),
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    src = _read_cell(nb_dir, "c1")
    result = await execute_sql_cell(session, "c1", src)
    assert result["success"], result.get("error")

    cell = next(c for c in session.notebook_state.cells if c.id == "c1")
    # Analyzer-side defines list reflects the @name override.
    assert cell.defines == ["seed_status"]
    # Executor-side artifact map uses the same key so a downstream
    # ``_collect_input_hashes`` walk lands on the right URI.
    assert "seed_status" in cell.artifact_uris
    assert "result" not in cell.artifact_uris

    # And the canonical artifact id matches the analyzer's output name.
    notebook_id = session.notebook_state.id
    canonical_id = f"nb_{notebook_id}_cell_c1_var_seed_status"
    canonical = session.get_artifact_manager().artifact_store.get_latest_version(canonical_id)
    assert canonical is not None


@pytest.mark.asyncio
async def test_sql_write_cell_propagates_commit_failure(tmp_path, monkeypatch):
    """A failing ``conn.commit()`` fails the cell instead of reporting a false success."""
    from strata.notebook.sql.cell_executor import execute_sql_cell
    from strata.notebook.sql.drivers.sqlite import SqliteAdapter

    db_path = tmp_path / "commit_fail.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=("# @sql connection=db write=true\nCREATE TABLE t (n INTEGER);\n"),
    )
    session = _make_session(nb_dir)

    real_open = SqliteAdapter.open

    def opening(self, spec, *, read_only):
        conn = real_open(self, spec, read_only=read_only)

        original_commit = conn.commit

        def explode(*_args, **_kwargs):
            raise RuntimeError("simulated commit failure")

        # Replace just commit; keep the rest of the interface.
        conn.commit = explode  # type: ignore[method-assign]
        # Keep ``original_commit`` reachable so it isn't GC'd.
        conn._original_commit = original_commit  # type: ignore[attr-defined]
        return conn

    monkeypatch.setattr(SqliteAdapter, "open", opening)

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    err = (result["error"] or "").lower()
    assert "simulated commit failure" in err, f"unexpected error: {result['error']!r}"


@pytest.mark.asyncio
async def test_sql_write_cell_emits_per_statement_status_table(tmp_path):
    """The write artifact has one row per statement: ``stmt`` (1-indexed), ``kind``,
    and ``rows_affected`` (None when the driver does not report, typically DDL).
    """
    db_path = tmp_path / "perstmt.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n"
            "DROP TABLE IF EXISTS t;\n"
            "CREATE TABLE t (n INTEGER);\n"
            "INSERT INTO t VALUES (1), (2), (3);\n"
        ),
    )
    session = _make_session(nb_dir)
    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"], result.get("error")

    table = _load_arrow_from_uri(session, result["artifact_uri"])
    assert table.num_rows == 3
    assert table.schema.names == ["stmt", "kind", "rows_affected"]
    rows = table.to_pylist()
    assert [r["stmt"] for r in rows] == [1, 2, 3]
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "DROP TABLE"
    assert kinds[1] == "CREATE TABLE"
    assert kinds[2] == "INSERT"
    # DDL gets null rows_affected even when SQLite's changes() would still return a
    # prior DML's count.
    assert rows[0]["rows_affected"] is None
    assert rows[1]["rows_affected"] is None
    # INSERT: ADBC SQLite leaves cursor.rowcount at -1, but the
    # SQLite ``SELECT changes()`` fallback recovers the real count.
    assert rows[2]["rows_affected"] == 3


@pytest.mark.asyncio
async def test_sql_write_cell_recovers_rowcount_from_sqlite_changes(tmp_path):
    """ADBC SQLite leaves rowcount at -1, so ``SELECT changes()`` supplies the DML count."""
    db_path = tmp_path / "rowcount.db"
    nb_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db write=true\n"
            "CREATE TABLE t (n INTEGER);\n"
            "INSERT INTO t VALUES (1), (2), (3), (4), (5);\n"
            "UPDATE t SET n = n * 10 WHERE n > 2;\n"
            "DELETE FROM t WHERE n >= 40;\n"
        ),
    )
    session = _make_session(nb_dir)
    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"], result.get("error")

    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    rows = table.to_pylist()
    by_kind = {r["kind"]: r["rows_affected"] for r in rows}
    # CREATE TABLE → DDL → null
    assert by_kind["CREATE TABLE"] is None
    # INSERT VALUES (1)..(5) → 5 rows
    assert by_kind["INSERT"] == 5
    # UPDATE matched rows where n > 2 → 3 rows
    assert by_kind["UPDATE"] == 3
    # DELETE matched rows where n >= 40 (after UPDATE: 30, 40, 50) → 2 rows
    assert by_kind["DELETE"] == 2


def _load_arrow_from_uri(session: Any, uri: str) -> Any:
    """Pull an artifact's bytes back and decode as a pyarrow Table."""
    return _load_artifact_as_arrow(session, uri)


# --- _resolve_runtime_spec: notebook-relative path rebasing ---------------


def test_resolve_runtime_spec_rebases_credentials_paths(tmp_path):
    """Relative credentials paths resolve against the notebook dir, not the server's CWD."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.cell_executor import _resolve_runtime_spec

    spec = ConnectionSpec(
        name="bq",
        driver="bigquery",
        project_id="acme",
        credentials_path="creds/ro.json",
        write_credentials_path="creds/rw.json",
    )
    resolved = _resolve_runtime_spec(spec, tmp_path)

    expected_ro = str((tmp_path / "creds/ro.json").resolve())
    expected_rw = str((tmp_path / "creds/rw.json").resolve())
    assert getattr(resolved, "credentials_path") == expected_ro
    assert getattr(resolved, "write_credentials_path") == expected_rw


def test_resolve_runtime_spec_leaves_absolute_credentials_paths_alone(tmp_path):
    """An absolute ``credentials_path`` is not joined with the notebook dir."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.cell_executor import _resolve_runtime_spec

    abs_path = str(tmp_path / "absolute.json")
    spec = ConnectionSpec(
        name="bq",
        driver="bigquery",
        project_id="acme",
        credentials_path=abs_path,
    )
    resolved = _resolve_runtime_spec(spec, tmp_path / "subdir")
    assert getattr(resolved, "credentials_path") == abs_path


# --- DuckDB end-to-end ----------------------------------------------------

duckdb_lib = pytest.importorskip("duckdb")


def _seed_duckdb(path: Path) -> None:
    """Create a DuckDB file shaped like ``_seed_sqlite``'s ``events`` table."""
    conn = duckdb_lib.connect(str(path))
    try:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name VARCHAR, value INTEGER)")
        conn.execute(
            "INSERT INTO events VALUES (1, 'alpha', 10), (2, 'beta', 20), (3, 'gamma', 30)"
        )
    finally:
        conn.close()


def _build_notebook_with_duckdb_cell(
    tmp_path: Path,
    *,
    db_path: Path,
    cell_id: str = "c1",
    cell_source: str,
) -> Path:
    """Write a notebook with a single DuckDB-bound SQL cell."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "duckdb_e2e")
    add_cell_to_notebook(nb_dir, cell_id, language="sql")
    write_cell(nb_dir, cell_id, cell_source)

    toml_path = nb_dir / "notebook.toml"
    text = toml_path.read_text()
    text += f'\n[connections.db]\ndriver = "duckdb"\npath = "{db_path}"\n'
    toml_path.write_text(text)
    return nb_dir


@pytest.mark.asyncio
async def test_duckdb_cell_executes_and_returns_arrow_table(tmp_path):
    """A SQL cell runs against a real DuckDB file and stores the rows as Arrow."""
    db_path = tmp_path / "events.duckdb"
    _seed_duckdb(db_path)
    nb_dir = _build_notebook_with_duckdb_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n"
            "# @cache forever\n"
            "SELECT id, name, value FROM events ORDER BY id\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"], result.get("error")
    assert result["cache_hit"] is False
    assert result["execution_method"] == "sql"

    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    assert table.num_rows == 3
    assert set(table.schema.names) == {"id", "name", "value"}
    assert {r["name"] for r in table.to_pylist()} == {"alpha", "beta", "gamma"}


@pytest.mark.asyncio
async def test_duckdb_cell_read_only_blocks_writes(tmp_path):
    """An INSERT fails and leaves the DB untouched; DuckDB is opened ``read_only=True``."""
    db_path = tmp_path / "events.duckdb"
    _seed_duckdb(db_path)
    nb_dir = _build_notebook_with_duckdb_cell(
        tmp_path,
        db_path=db_path,
        cell_source=(
            "# @sql connection=db\n# @cache forever\nINSERT INTO events VALUES (99, 'hack', 1)\n"
        ),
    )
    session = _make_session(nb_dir)

    from strata.notebook.sql.cell_executor import execute_sql_cell

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    assert result["error"]

    # Underlying DB row count is unchanged.
    conn = duckdb_lib.connect(str(db_path), read_only=True)
    try:
        (count,) = conn.execute("SELECT count(*) FROM events").fetchone()
        assert count == 3
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("datetime.datetime(2024, 1, 2, 3, 4, 5)", "2024-01-02 03:04:05"),
        ("datetime.date(2024, 1, 2)", "2024-01-02"),
        ("datetime.time(3, 4, 5)", "03:04:05"),
        ("decimal.Decimal('12.50')", "12.50"),
        (
            "uuid.UUID('12345678-1234-5678-1234-567812345678')",
            "12345678-1234-5678-1234-567812345678",
        ),
        ("b'abc'", "abc"),
        ("numpy.int64(5)", "5"),
    ],
)
async def test_duckdb_cell_binds_a_typed_scalar_from_an_upstream_python_cell(
    tmp_path, expr, expected
):
    """These values are stored as Arrow scalar tables; the bind gets the Python value back."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    db_path = tmp_path / "events.duckdb"
    _seed_duckdb(db_path)
    nb_dir = create_notebook(tmp_path, "duckdb_binds")
    add_cell_to_notebook(nb_dir, "py", language="python")
    write_cell(nb_dir, "py", f"import datetime, decimal, uuid, numpy\nv = {expr}\n")
    add_cell_to_notebook(nb_dir, "sql", after_cell_id="py", language="sql")
    write_cell(
        nb_dir, "sql", "# @sql connection=db\n# @cache forever\nSELECT CAST(:v AS VARCHAR) AS v\n"
    )
    toml_path = nb_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text() + f'\n[connections.db]\ndriver = "duckdb"\npath = "{db_path}"\n'
    )
    session = _make_session(nb_dir)
    session.refresh_environment_runtime()

    from strata.notebook.executor import CellExecutor
    from strata.notebook.sql.cell_executor import execute_sql_cell

    py_result = await CellExecutor(session).execute_cell("py", _read_cell(nb_dir, "py"))
    assert py_result.success, py_result.error

    result = await execute_sql_cell(session, "sql", _read_cell(nb_dir, "sql"))
    assert result["success"], result.get("error")
    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    assert table.to_pylist() == [{"v": expected}]


class TestSafelyClose:
    """A handle whose close fails must not be closed again at collection.

    adbc's ``Cursor.close()`` sets ``_closed`` only after ``_stmt.close()``
    returns, and a read-only write error is raised by that close. The finalizer then
    closed it again, underflowing the driver's child count at a random moment.
    """

    def test_a_handle_whose_close_raises_is_marked_closed(self):
        from strata.notebook.sql.cell_executor import _safely_close

        class Handle:
            def __init__(self):
                self._closed = False

            def close(self):
                raise RuntimeError("close failed at the driver")

        handle = Handle()
        _safely_close(handle)

        assert handle._closed is True, "its finalizer will try again"

    def test_a_handle_without_the_flag_is_not_given_one(self):
        """Only a flag the object already keeps is corrected; none is invented on a foreign
        object."""
        from strata.notebook.sql.cell_executor import _safely_close

        class Handle:
            def close(self):
                raise RuntimeError("close failed")

        handle = Handle()
        _safely_close(handle)

        assert not hasattr(handle, "_closed")

    def test_a_handle_that_closed_cleanly_is_left_alone(self):
        from strata.notebook.sql.cell_executor import _safely_close

        class Handle:
            def __init__(self):
                self._closed = False
                self.closed_times = 0

            def close(self):
                self.closed_times += 1
                self._closed = True

        handle = Handle()
        _safely_close(handle)

        assert handle.closed_times == 1
        assert handle._closed is True

    def test_none_is_accepted(self):
        from strata.notebook.sql.cell_executor import _safely_close

        _safely_close(None)


class TestWhatASqlCellRemembersAcrossAReopen:
    """SQL cells across a reopen.

    Their artifacts are keyed under the SQL hash, so the recorded generic triplet
    is what keeps them ready. Status is not persisted, so a cold open starts idle.
    """

    @pytest.mark.asyncio
    async def test_reopening_keeps_it_ready_and_its_output_reachable(self, tmp_path):
        db_path = tmp_path / "events.duckdb"
        _seed_duckdb(db_path)
        nb_dir = _build_notebook_with_duckdb_cell(
            tmp_path,
            db_path=db_path,
            cell_source="# @sql connection=db\n# @cache forever\nSELECT id FROM events\n",
        )
        session = _make_session(nb_dir)
        from strata.notebook.executor import CellExecutor

        result = await CellExecutor(session).execute_cell("c1", _read_cell(nb_dir, "c1"))
        assert result.success, result.error
        session.compute_staleness()
        session.mark_executed_ready("c1")

        reopened = _make_session(nb_dir)
        staleness = reopened.compute_staleness()

        from strata.notebook.models import CellStatus

        assert staleness["c1"].status == CellStatus.READY, (
            "a reopened SQL cell went idle with nothing changed"
        )
        cell = reopened.notebook_state.get_cell("c1")
        assert cell.artifact_uris, "a downstream cell reads these to build its own provenance"

    @staticmethod
    def _repoint(nb_dir: Path, db_path: Path) -> None:
        toml_path = nb_dir / "notebook.toml"
        head, _, _ = toml_path.read_text().partition("[connections.db]")
        toml_path.write_text(head + f'[connections.db]\ndriver = "duckdb"\npath = "{db_path}"\n')

    async def _run_then_reopen(self, tmp_path, cell_source: str, repoint_to: Path | None = None):
        from strata.notebook.executor import CellExecutor

        db_path = tmp_path / "events.duckdb"
        _seed_duckdb(db_path)
        nb_dir = _build_notebook_with_duckdb_cell(
            tmp_path, db_path=db_path, cell_source=cell_source
        )
        session = _make_session(nb_dir)
        result = await CellExecutor(session).execute_cell("c1", _read_cell(nb_dir, "c1"))
        assert result.success, result.error
        session.compute_staleness()
        session.mark_executed_ready("c1")
        if repoint_to is not None:
            _seed_duckdb(repoint_to)
            self._repoint(nb_dir, repoint_to)
        return _make_session(nb_dir).compute_staleness()["c1"].status

    @pytest.mark.asyncio
    async def test_pointing_the_connection_at_another_database_is_not_ready(self, tmp_path):
        """A different database is a different answer, and the generic triplet cannot see it."""
        from strata.notebook.models import CellStatus

        status = await self._run_then_reopen(
            tmp_path,
            "# @sql connection=db\n# @cache forever\nSELECT id FROM events\n",
            repoint_to=tmp_path / "other.duckdb",
        )

        assert status != CellStatus.READY, "a reopen served another database's rows as ready"

    @pytest.mark.asyncio
    async def test_a_session_policy_does_not_survive_the_session(self, tmp_path):
        """``# @cache session`` is invalid after a reopen, which is a new session."""
        from strata.notebook.models import CellStatus

        status = await self._run_then_reopen(
            tmp_path, "# @sql connection=db\n# @cache session\nSELECT id FROM events\n"
        )

        assert status != CellStatus.READY

    @pytest.mark.asyncio
    async def test_the_default_policy_waits_to_be_asked(self, tmp_path):
        """``fingerprint`` promises to check the source, and opening a notebook does not check."""
        from strata.notebook.models import CellStatus

        status = await self._run_then_reopen(
            tmp_path, "# @sql connection=db\nSELECT id FROM events\n"
        )

        assert status != CellStatus.READY

    @pytest.mark.asyncio
    async def test_the_run_that_just_happened_stays_ready(self, tmp_path):
        """A cell that ran in this session has already checked, so it stays ready."""
        from strata.notebook.executor import CellExecutor
        from strata.notebook.models import CellStatus

        db_path = tmp_path / "events.duckdb"
        _seed_duckdb(db_path)
        nb_dir = _build_notebook_with_duckdb_cell(
            tmp_path,
            db_path=db_path,
            cell_source="# @sql connection=db\nSELECT id FROM events\n",
        )
        session = _make_session(nb_dir)

        result = await CellExecutor(session).execute_cell("c1", _read_cell(nb_dir, "c1"))
        assert result.success, result.error
        session.compute_staleness()
        session.mark_executed_ready("c1")

        assert session.compute_staleness()["c1"].status == CellStatus.READY

    @pytest.mark.asyncio
    async def test_an_edit_still_makes_it_stale(self, tmp_path):
        db_path = tmp_path / "events.duckdb"
        _seed_duckdb(db_path)
        nb_dir = _build_notebook_with_duckdb_cell(
            tmp_path,
            db_path=db_path,
            cell_source="# @sql connection=db\n# @cache forever\nSELECT id FROM events\n",
        )
        session = _make_session(nb_dir)
        from strata.notebook.executor import CellExecutor

        await CellExecutor(session).execute_cell("c1", _read_cell(nb_dir, "c1"))
        session.compute_staleness()
        session.mark_executed_ready("c1")

        (nb_dir / "cells" / "c1.py").write_text(
            "# @sql connection=db\n# @cache forever\nSELECT id, name FROM events\n"
        )
        reopened = _make_session(nb_dir)

        from strata.notebook.models import CellStatus

        assert reopened.compute_staleness()["c1"].status != CellStatus.READY

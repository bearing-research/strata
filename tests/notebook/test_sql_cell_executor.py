"""End-to-end tests for the SQL cell executor (sqlite, real DB)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

# Skip the whole suite if optional ADBC packages are missing — the
# tests need adbc-driver-sqlite to actually open a connection.
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


def _build_notebook_with_sql_cell(
    tmp_path: Path,
    *,
    db_path: Path,
    cell_id: str = "c1",
    cell_source: str,
) -> Path:
    """Materialize a notebook directory with one [connections.db] +
    one SQL cell. Returns the notebook directory."""
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "sql_e2e")
    add_cell_to_notebook(nb_dir, cell_id, language="sql")
    write_cell(nb_dir, cell_id, cell_source)

    # Inject [connections.db]. The current writer rewrites the whole
    # toml on serialize, so the safest path is to read the existing
    # toml as text, append the connection block, and write back.
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

    # Verify the stored artifact actually contains the rows.
    table = _load_artifact_as_arrow(session, result["artifact_uri"])
    assert table.num_rows == 3
    assert set(table.schema.names) == {"id", "name", "value"}
    rows = table.to_pylist()
    assert {r["name"] for r in rows} == {"alpha", "beta", "gamma"}


@pytest.mark.asyncio
async def test_sql_cell_cache_hit_on_unchanged_inputs(tmp_path):
    """``# @cache forever`` skips the freshness probe; running twice
    in a row must return the same artifact with cache_hit=True."""
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


@pytest.mark.asyncio
async def test_sql_cell_fingerprint_invalidates_on_schema_change(tmp_path):
    """Fingerprint policy folds SQLite's ``PRAGMA schema_version``
    into the hash. After an external DDL change, the second run
    must re-execute rather than serve the stale artifact.

    Note: ``PRAGMA data_version`` *also* feeds the freshness token,
    but it's only meaningful when the probe connection stays open
    across the write — a fresh-open / read / close cycle always
    starts at 1 cross-process. SQLite's design treats data_version
    as "this connection's view of the file's write counter," not
    "the file's absolute write counter." The schema_version value
    in contrast does refresh on each fresh open, so DDL changes
    invalidate cleanly. DML invalidation in a multi-process SQLite
    setup is a known limitation; users with that workload should
    pin ``# @cache session`` or accept the coarser granularity."""
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

    # The new column should appear in the re-executed result's schema.
    table = _load_artifact_as_arrow(session, second["artifact_uri"])
    assert "extra" in table.schema.names


@pytest.mark.asyncio
async def test_sql_cell_read_only_enforces_no_writes(tmp_path):
    """A SQL cell that tries to INSERT must fail and leave the DB
    untouched. The executor opens the connection in enforced
    read-only mode (URI ``mode=ro`` plus ``PRAGMA query_only=ON``);
    that is the security boundary, not SQL-text keyword filtering.

    Note: ADBC's SQLite driver collapses SQLite's "attempt to write
    a readonly database" message into a generic ``InternalError``
    at the Python surface (the descriptive OperationalError fires
    during statement finalization in ``__del__`` and is only
    logged). The cell-level contract Strata pins is "this fails
    and the DB is untouched"; pinning the exact message text would
    couple us to ADBC internals."""
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
    """Cross-language wiring: a Python cell defines a variable, a
    SQL cell binds against it via ``:name``. This is the integration
    test that exercises every slice 6–9 piece together."""
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

    # Run the Python cell first so the upstream artifact exists.
    from strata.notebook.executor import CellExecutor

    executor = CellExecutor(session)
    py_src = (nb_dir / "cells" / "py.py").read_text()
    py_result = await executor.execute_cell("py", py_src)
    assert py_result.success, py_result.error

    # Now run the SQL cell. It should resolve :min_value to 15 from
    # the upstream artifact and return rows with value > 15.
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
    """Codex review fix: SQL cells must participate in the notebook's
    standard staleness machinery so a recompute or reopen keeps them
    READY. The SQL executor stores artifacts under a SQL-specific
    per-variable hash that ``compute_staleness`` doesn't recompute,
    so we have to persist the generic provenance triplet via
    ``record_successful_execution_provenance`` and let the
    ``can_preserve_uncached_ready`` path mark the cell READY."""
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

    # A *second* staleness recompute (the path a notebook reopen
    # also walks under load) must keep the cell READY. Without the
    # language=='sql' branch in can_preserve_uncached_ready, this
    # drops to IDLE because the generic per-variable artifact
    # lookup misses (the SQL executor stored the artifact under
    # SQL-specific provenance).
    session.compute_staleness()
    cell_after = next(c for c in session.notebook_state.cells if c.id == "c1")
    assert cell_after.status == CellStatus.READY, (
        f"SQL cell should stay READY after staleness recompute; got {cell_after.status!r}"
    )


@pytest.mark.asyncio
async def test_sql_cell_artifact_uri_visible_to_downstream_python(tmp_path):
    """Codex review fix: ``cell.artifact_uris`` must be set after a
    successful SQL execution so a downstream Python cell's
    ``_collect_input_hashes`` finds the upstream artifact. Without
    this, the downstream provenance hash is computed without the
    SQL input hash — silently identical across SQL-content changes,
    so the downstream cell would serve a stale cached value after
    the SQL upstream's data shifts."""
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

    # And the downstream Python cell's input-hashes collection picks
    # up the SQL artifact's provenance hash — exercising the wiring
    # end-to-end.
    input_hashes = session._collect_input_hashes("py")
    assert len(input_hashes) == 1, (
        f"downstream py cell should see 1 upstream input hash; got {input_hashes!r}"
    )


@pytest.mark.asyncio
async def test_sql_cell_executor_dispatched_via_main_executor(tmp_path):
    """The ``CellExecutor`` dispatches ``language='sql'`` to the SQL
    path. This is the wiring point that makes the slice 8 helpers
    no longer dead code in production — see the prior Codex review."""
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
    """A write cell opens the connection writable, splits the body
    into statements via sqlglot, and runs each in sequence. The
    cell still produces an Arrow artifact (a status table) so
    downstream cells can find it via cell.artifact_uris."""
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

    # The DB has the rows.
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id, label FROM events ORDER BY id").fetchall()
    assert rows == [(1, "alpha"), (2, "beta")]


@pytest.mark.asyncio
async def test_sql_write_cell_caches_within_session(tmp_path):
    """Default cache policy for write cells is ``session`` — re-run
    inside the same session cache-hits, dedup'ing minor edits to
    downstream cells. (A new session would re-execute.)"""
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
    """Probe-based policies don't apply to writes — surface a clear
    error instead of silently coercing."""
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
    """Without ``write=true``, the read-only enforcement still fires
    — adding the new flag doesn't broaden the security boundary
    for the default case."""
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
    # Row count unchanged.
    with sqlite3.connect(tmp_path / "events.db") as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert count == 3


@pytest.mark.asyncio
async def test_sql_write_cell_makes_db_visible_to_read_cell(tmp_path):
    """End-to-end: a write cell creates the DB, a read cell queries
    it. This is the core "use a SQL cell instead of a Python seed"
    workflow the example should support."""
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


# --- Codex review fixes for write cells -----------------------------------


@pytest.mark.asyncio
async def test_sql_write_cell_resolves_bind_placeholders_from_upstream(tmp_path):
    """Codex review fix: write cells go through the same analyzer +
    bind layer as read cells. ``INSERT INTO t VALUES (:n)`` resolves
    ``:n`` against the upstream namespace, type-checks via the bind
    allowlist, and rewrites to the dialect's positional form before
    cursor.execute. Without this, the write path bypassed binds
    entirely and a ``:n`` token would appear verbatim in the
    statement (or silently fail)."""
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
    """Codex review fix: upstream_input_hashes feeds the write
    cell's provenance hash. Same source + different upstream value
    must miss the cache (otherwise the seed silently re-uses the
    old value)."""
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
    """Codex review fix: the write path used to hardcode
    ``output_name = "result"``, mismatching what the analyzer's
    ``defines`` advertised when the cell carried ``# @name``.
    Downstream cells looking up the named output by canonical id
    would miss it. Now both the analyzer and the executor agree
    on the name."""
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
    """Codex review fix: the write path used to swallow every
    exception from ``conn.commit()``, so a real deferred-constraint
    or transport failure surfaced as a misleading ``success=True``
    with nothing actually persisted. Now commit errors propagate
    as a normal cell error."""
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
        # Keep ``original_commit`` reachable so it's not GC'd into a
        # dangling reference; not strictly necessary but cleaner.
        conn._original_commit = original_commit  # type: ignore[attr-defined]
        return conn

    monkeypatch.setattr(SqliteAdapter, "open", opening)

    result = await execute_sql_cell(session, "c1", _read_cell(nb_dir, "c1"))
    assert result["success"] is False
    err = (result["error"] or "").lower()
    assert "simulated commit failure" in err, f"unexpected error: {result['error']!r}"


@pytest.mark.asyncio
async def test_sql_write_cell_emits_per_statement_status_table(tmp_path):
    """The write-cell artifact is a per-statement table: one row
    per statement with ``stmt`` (1-indexed), ``kind`` (CREATE
    TABLE / INSERT / ...), and ``rows_affected`` (nullable; None
    when the driver doesn't report — typically DDL). The earlier
    one-row {statements_executed, last_rowcount} shape masked the
    middle of multi-statement bodies and showed -1 for DDL,
    confusing for users."""
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
    # Order preserved.
    assert [r["stmt"] for r in rows] == [1, 2, 3]
    # Kinds reflect the statement type.
    kinds = [r["kind"] for r in rows]
    assert kinds[0] == "DROP TABLE"
    assert kinds[1] == "CREATE TABLE"
    assert kinds[2] == "INSERT"
    # DDL gets null rows_affected — we suppress the count for
    # DDL even when SQLite's changes() would return a value from
    # a prior DML.
    assert rows[0]["rows_affected"] is None
    assert rows[1]["rows_affected"] is None
    # INSERT: ADBC SQLite leaves cursor.rowcount at -1, but the
    # SQLite ``SELECT changes()`` fallback recovers the real count.
    assert rows[2]["rows_affected"] == 3


@pytest.mark.asyncio
async def test_sql_write_cell_recovers_rowcount_from_sqlite_changes(tmp_path):
    """ADBC SQLite never populates cursor.rowcount (always -1).
    The cell executor falls back to ``SELECT changes()`` so the
    user sees the real count for INSERT / UPDATE / DELETE. Pins
    that behavior across DML operations."""
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
    """Pull an artifact's bytes back and decode as a pyarrow Table.

    Older tests in this file hand-rolled this; refactoring the
    helper out keeps the per-statement test self-contained.
    """
    return _load_artifact_as_arrow(session, uri)


# --- _resolve_runtime_spec: notebook-relative path rebasing ---------------


def test_resolve_runtime_spec_rebases_credentials_paths(tmp_path):
    """``credentials_path`` and ``write_credentials_path`` are
    declared in ``notebook.toml`` and should be relative to the
    notebook directory — same convention as SQLite ``path``. The
    server's CWD is unrelated, so without rebasing a relative
    ``creds/ro.json`` would be looked up under the launching
    process's directory and fail to open."""
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
    """An absolute ``credentials_path`` is already meaningful on
    its own — don't munge it by joining with the notebook dir."""
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
    """Create a DuckDB file the cells can query.

    Mirrors ``_seed_sqlite`` shape (``events`` with id/name/value/timestamp).
    Using the native duckdb package directly — no harness, no notebook
    yet — keeps this fixture independent of the rest of the SQL pipeline.
    """
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
    """Materialize a notebook with a single DuckDB-bound SQL cell."""
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
    """End-to-end: open notebook → execute SQL cell against DuckDB →
    verify the artifact carries the rows. Exercises canonicalize →
    open(read_only=True) → query → store-as-Arrow on a real DuckDB
    file."""
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
    """A SQL cell that tries to INSERT against a DuckDB connection
    must fail and leave the DB untouched. The adapter opens the
    file with ``read_only=True``, so the engine refuses the write
    before it touches storage — no SQL-text keyword filtering
    needed."""
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


class TestSafelyClose:
    """A handle whose close fails must not be closed again at collection.

    adbc's ``Cursor.close()`` sets ``_closed = True`` only after
    ``_stmt.close()`` returns, and a write to a read-only connection is
    reported *by* that close — so the cursor stayed marked open and its
    finalizer closed it a second time, underflowing the driver's child count.
    The result was an unraisable exception surfacing at whatever unrelated
    moment the collector happened to run.
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
        """Only a flag the object already keeps is corrected. Inventing one on
        an unrelated library's object would be lying to code we do not own."""
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

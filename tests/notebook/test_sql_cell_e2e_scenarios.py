"""End-to-end SQL cell scenarios that need real DB execution.

Covers injection through binds, cache identity tracking upstream values,
snapshot policy on non-snapshot drivers, NULL binds, and empty results.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

adbc_sqlite = pytest.importorskip("adbc_driver_sqlite")


# --- shared fixtures ------------------------------------------------------


def _seed_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, name TEXT, value INTEGER)")
        conn.executemany(
            "INSERT INTO events (id, name, value) VALUES (?, ?, ?)",
            [(1, "alpha", 10), (2, "beta", 20), (3, "gamma", 30)],
        )
        conn.commit()


def _build_notebook(
    tmp_path: Path,
    *,
    db_path: Path,
    cells: list[tuple[str, str, str]],
) -> Path:
    """Create a notebook with the given cells (id, language, source).

    Connections are appended by hand so the test does not depend on the writer's
    ``[connections.<name>]`` shape.
    """
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    nb_dir = create_notebook(tmp_path, "sql_e2e_scenarios")
    after: str | None = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb_dir, cell_id, after_cell_id=after, language=language)
        write_cell(nb_dir, cell_id, source)
        after = cell_id

    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + "\n[connections.db]\n" + 'driver = "sqlite"\n' + f'path = "{db_path}"\n'
    )
    return nb_dir


def _session(nb_dir: Path) -> Any:
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    return session


def _read(nb_dir: Path, cell_id: str) -> str:
    return (nb_dir / "cells" / f"{cell_id}.py").read_text()


def _load_arrow(session: Any, uri: str) -> Any:
    import pyarrow as pa

    body = uri.removeprefix("strata://artifact/")
    art_id, version = body.rsplit("@v=", 1)
    blob = session.get_artifact_manager().load_artifact_data(art_id, int(version))
    return pa.ipc.open_stream(blob).read_all()


# --- 1. Injection rejection (e2e) ----------------------------------------


@pytest.mark.asyncio
async def test_sql_injection_via_bind_does_not_alter_database(tmp_path):
    """An injection-shaped string bound as a parameter leaves the table intact.

    ADBC parameter binding is the security boundary, not any text filter.
    """
    from strata.notebook.executor import CellExecutor

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook(
        tmp_path,
        db_path=db_path,
        cells=[
            ("py", "python", 'needle = "\'; DROP TABLE events; --"\n'),
            (
                "sql",
                "sql",
                "# @sql connection=db\n"
                "# @cache forever\n"
                "SELECT id FROM events WHERE name = :needle\n",
            ),
        ],
    )
    session = _session(nb_dir)
    executor = CellExecutor(session)

    py_result = await executor.execute_cell("py", _read(nb_dir, "py"))
    assert py_result.success, py_result.error

    sql_result = await executor.execute_cell("sql", _read(nb_dir, "sql"))
    assert sql_result.success, sql_result.error

    # No row matches the literal needle → empty result is fine.
    table = _load_arrow(session, sql_result.artifact_uri)
    assert table.num_rows == 0

    # Critical: the events table is intact.
    with sqlite3.connect(db_path) as conn:
        (count,) = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert count == 3
        names = sorted(r[0] for r in conn.execute("SELECT name FROM events"))
        assert names == ["alpha", "beta", "gamma"]


# --- 2. Cache identity tracks upstream values ----------------------------


@pytest.mark.asyncio
async def test_sql_cache_keyed_on_upstream_bind_value_not_rerun(tmp_path):
    """Cache identity tracks the upstream value, not whether the upstream re-ran.

    Round-trips the value (15, 25, 15): the third run must return the first run's
    artifact URI, which an "always re-execute when upstream re-runs" bug would fail.
    """
    from strata.notebook.executor import CellExecutor

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook(
        tmp_path,
        db_path=db_path,
        cells=[
            ("py", "python", "min_value = 15\n"),
            (
                "sql",
                "sql",
                "# @sql connection=db\n"
                "# @cache forever\n"
                "SELECT id, name FROM events WHERE value > :min_value ORDER BY id\n",
            ),
        ],
    )
    session = _session(nb_dir)
    executor = CellExecutor(session)

    py_cell = next(c for c in session.notebook_state.cells if c.id == "py")
    sql_src = _read(nb_dir, "sql")

    async def set_upstream_and_run_sql(new_src: str):
        (nb_dir / "cells" / "py.py").write_text(new_src)
        py_cell.source = new_src
        session.re_analyze_cell("py")
        await executor.execute_cell("py", new_src)
        return await executor.execute_cell("sql", sql_src)

    def _provenance_hash_for(uri: str) -> str:
        body = uri.removeprefix("strata://artifact/")
        art_id, version = body.rsplit("@v=", 1)
        artifact = session.get_artifact_manager().artifact_store.get_artifact(art_id, int(version))
        assert artifact is not None
        return artifact.provenance_hash

    # Run 1: min_value=15 → 2 rows, fresh artifact.
    await executor.execute_cell("py", _read(nb_dir, "py"))
    first = await executor.execute_cell("sql", sql_src)
    assert first.success
    assert first.cache_hit is False
    assert _load_arrow(session, first.artifact_uri).num_rows == 2
    first_hash = _provenance_hash_for(first.artifact_uri)

    # Run 2: change to min_value=25 → cache miss, different
    # artifact (different bind ⇒ different provenance hash).
    second = await set_upstream_and_run_sql("min_value = 25\n")
    assert second.success
    assert second.cache_hit is False
    assert _load_arrow(session, second.artifact_uri).num_rows == 1
    second_hash = _provenance_hash_for(second.artifact_uri)
    assert second_hash != first_hash, (
        "different bind value must produce a different provenance hash"
    )

    # Run 3: re-run the upstream with the same value. Its source_hash is unchanged,
    # so it cache-hits, and so must the SQL cell (same bind, same hash).
    third = await executor.execute_cell("sql", sql_src)
    assert third.success
    assert third.cache_hit is True
    assert _provenance_hash_for(third.artifact_uri) == second_hash

    # Run 4: back to min_value=15. The SQL hash must equal the first run's, so the
    # cache key is a pure function of the upstream value, not of whether the upstream
    # re-ran. (The store may assign a new version; the hash is the cache-key contract.)
    fourth = await set_upstream_and_run_sql("min_value = 15\n")
    assert fourth.success
    assert _load_arrow(session, fourth.artifact_uri).num_rows == 2
    assert _provenance_hash_for(fourth.artifact_uri) == first_hash, (
        "round-trip to the original upstream value must produce the "
        "original SQL provenance hash — the cache key is keyed on the "
        "bind value, not on whether the upstream re-executed"
    )

    # And as a bytewise sanity check, the data round-trips:
    assert (
        _load_arrow(session, fourth.artifact_uri).to_pylist()
        == _load_arrow(session, first.artifact_uri).to_pylist()
    )


# --- 3. Snapshot policy on non-snapshot driver ---------------------------


@pytest.mark.asyncio
async def test_sql_snapshot_policy_errors_before_opening_connection(tmp_path, monkeypatch):
    """``# @cache snapshot`` on SQLite fails in ``resolve_cache_policy``, before any connection.

    The adapter's ``open`` is patched to crash and counted, so a regression that
    opens a connection or runs probes first fails with the crash, not the diagnostic.
    """
    from strata.notebook.sql.cell_executor import execute_sql_cell
    from strata.notebook.sql.drivers.sqlite import SqliteAdapter

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook(
        tmp_path,
        db_path=db_path,
        cells=[
            (
                "c1",
                "sql",
                "# @sql connection=db\n# @cache snapshot\nSELECT * FROM events\n",
            )
        ],
    )
    session = _session(nb_dir)

    open_calls: list[Any] = []

    def boom(self, spec, *, read_only):
        open_calls.append((spec, read_only))
        raise RuntimeError("SqliteAdapter.open must not be reached for @cache snapshot")

    monkeypatch.setattr(SqliteAdapter, "open", boom)

    result = await execute_sql_cell(session, "c1", _read(nb_dir, "c1"))
    assert result["success"] is False
    err = (result.get("error") or "").lower()
    assert "snapshot" in err, f"unexpected error: {result['error']!r}"
    assert open_calls == [], (
        "executor opened a connection before the snapshot policy "
        f"check fired; open() called {len(open_calls)} time(s)"
    )


# --- 4. NULL bind values --------------------------------------------------


@pytest.mark.asyncio
async def test_sql_null_bind_param_via_none_upstream(tmp_path):
    """A ``None`` upstream value binds as SQL NULL.

    The bound value is selected back as a column, so the bind is the only path to
    the result (an ``IS NULL OR`` filter would pass even with binding broken).
    """
    from strata.notebook.executor import CellExecutor

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook(
        tmp_path,
        db_path=db_path,
        cells=[
            ("py", "python", "sentinel = None\n"),
            # The bind value flows back as the result column, so the
            # only way to get NULL in ``sentinel_back`` is for the
            # binding to actually pass through as NULL.
            (
                "sql",
                "sql",
                "# @sql connection=db\n# @cache forever\nSELECT :sentinel AS sentinel_back\n",
            ),
        ],
    )
    session = _session(nb_dir)
    executor = CellExecutor(session)

    # Run 1: None upstream → NULL bind → result column is NULL.
    await executor.execute_cell("py", _read(nb_dir, "py"))
    none_result = await executor.execute_cell("sql", _read(nb_dir, "sql"))
    assert none_result.success, none_result.error
    table = _load_arrow(session, none_result.artifact_uri)
    assert table.num_rows == 1
    null_mask = table.column("sentinel_back").is_null().to_pylist()
    assert null_mask == [True], (
        "binding None as a SQL parameter must produce SQL NULL — "
        f"got {table.column('sentinel_back').to_pylist()!r}"
    )

    # Run 2: a non-None upstream now comes back as that value, not NULL. Same query,
    # different bind, different result isolates that the bind feeds the column.
    py_cell = next(c for c in session.notebook_state.cells if c.id == "py")
    new_src = "sentinel = 'hello'\n"
    (nb_dir / "cells" / "py.py").write_text(new_src)
    py_cell.source = new_src
    session.re_analyze_cell("py")
    await executor.execute_cell("py", new_src)

    str_result = await executor.execute_cell("sql", _read(nb_dir, "sql"))
    assert str_result.success, str_result.error
    table2 = _load_arrow(session, str_result.artifact_uri)
    assert table2.column("sentinel_back").is_null().to_pylist() == [False]
    assert table2.column("sentinel_back").to_pylist() == ["hello"]


# --- 5. Empty result set --------------------------------------------------


@pytest.mark.asyncio
async def test_sql_empty_result_set_produces_valid_artifact(tmp_path):
    """Zero rows still yields a decodable ``arrow/ipc`` artifact with the right schema.

    A naive IPC writer that writes no batches can produce an undecodable stream.
    """
    from strata.notebook.executor import CellExecutor

    db_path = tmp_path / "events.db"
    _seed_sqlite(db_path)
    nb_dir = _build_notebook(
        tmp_path,
        db_path=db_path,
        cells=[
            (
                "c1",
                "sql",
                "# @sql connection=db\n"
                "# @cache forever\n"
                "SELECT id, name FROM events WHERE value < 0\n",
            )
        ],
    )
    session = _session(nb_dir)
    executor = CellExecutor(session)

    result = await executor.execute_cell("c1", _read(nb_dir, "c1"))
    assert result.success, result.error
    table = _load_arrow(session, result.artifact_uri)
    assert table.num_rows == 0
    # Schema survives with no rows, so downstream cells can still inspect columns.
    assert set(table.schema.names) == {"id", "name"}


class TestAnOutsideWriteIsSeen:
    """The default cache policy sees a write made outside the notebook.

    ``PRAGMA data_version`` only reports changes the probing connection saw since
    it opened, and the probe opens a new connection each time, so it is constant.
    """

    @pytest.mark.asyncio
    async def test_a_write_from_another_connection_makes_the_cell_recompute(self, tmp_path):
        from strata.notebook.executor import CellExecutor

        db_path = tmp_path / "events.db"
        _seed_sqlite(db_path)
        source = "# @sql connection=db\nSELECT sum(value) AS total FROM events\n"
        nb_dir = _build_notebook(tmp_path, db_path=db_path, cells=[("sql", "sql", source)])
        session = _session(nb_dir)
        executor = CellExecutor(session)

        first = await executor.execute_cell("sql", source)
        assert first.success, first.error
        assert _load_arrow(session, first.artifact_uri).column("total").to_pylist() == [60]

        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO events (id, name, value) VALUES (4, 'delta', 40)")
            conn.commit()

        second = await executor.execute_cell("sql", source)

        assert second.success, second.error
        assert second.cache_hit is False, "the cell served a stale answer as a cache hit"
        assert _load_arrow(session, second.artifact_uri).column("total").to_pylist() == [100]

    @pytest.mark.asyncio
    async def test_an_unchanged_database_still_hits_the_cache(self, tmp_path):
        from strata.notebook.executor import CellExecutor

        db_path = tmp_path / "events.db"
        _seed_sqlite(db_path)
        source = "# @sql connection=db\nSELECT sum(value) AS total FROM events\n"
        nb_dir = _build_notebook(tmp_path, db_path=db_path, cells=[("sql", "sql", source)])
        executor = CellExecutor(_session(nb_dir))

        await executor.execute_cell("sql", source)
        again = await executor.execute_cell("sql", source)

        assert again.cache_hit is True

    @pytest.mark.asyncio
    async def test_a_write_in_wal_mode_is_seen_before_any_checkpoint(self, tmp_path):
        """In WAL mode a commit lands beside the database; the header waits for a checkpoint."""
        from strata.notebook.executor import CellExecutor

        db_path = tmp_path / "events.db"
        _seed_sqlite(db_path)
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
        source = "# @sql connection=db\nSELECT sum(value) AS total FROM events\n"
        nb_dir = _build_notebook(tmp_path, db_path=db_path, cells=[("sql", "sql", source)])
        session = _session(nb_dir)
        executor = CellExecutor(session)

        first = await executor.execute_cell("sql", source)
        assert first.success, first.error

        writer = sqlite3.connect(db_path)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("INSERT INTO events (id, name, value) VALUES (5, 'epsilon', 50)")
            writer.commit()
            assert (tmp_path / "events.db-wal").exists(), "the commit should still be in the log"
            second = await executor.execute_cell("sql", source)
        finally:
            writer.close()

        assert second.cache_hit is False
        assert _load_arrow(session, second.artifact_uri).column("total").to_pylist() == [110]


class TestACellThatAlwaysRunsMovesItsConsumers:
    """A SQL cell that always runs gives its consumers a new provenance hash each run.

    A cell whose table no probe can name (Snowflake's ``IDENTIFIER($tbl)``) skips
    its cache, but a freshness token over no tables is constant, so a downstream
    cell keyed on that hash would keep serving results from the old rows.
    """

    @staticmethod
    def _as_if_the_table_were_named_at_run_time(monkeypatch: pytest.MonkeyPatch) -> None:
        import dataclasses

        from strata.notebook.sql import cell_executor
        from strata.notebook.sql.adapter import FreshnessToken

        real = cell_executor.analyze_sql_cell

        def analyze(source: str, **kwargs: Any) -> Any:
            return dataclasses.replace(
                real(source, **kwargs), tables=[], unresolved_tables=["IDENTIFIER($tbl)"]
            )

        monkeypatch.setattr(cell_executor, "analyze_sql_cell", analyze)
        # What Snowflake's probe answers for an empty table list. SQLite's own
        # probe is database-wide and would see the write on its own.
        monkeypatch.setattr(
            cell_executor, "_run_probes", lambda *a, **k: (FreshnessToken(value=b""), None)
        )

    @pytest.mark.asyncio
    async def test_a_consumer_re_runs_exactly_when_the_rows_changed(self, tmp_path, monkeypatch):
        from strata.notebook.executor import CellExecutor

        self._as_if_the_table_were_named_at_run_time(monkeypatch)
        db_path = tmp_path / "events.db"
        _seed_sqlite(db_path)
        nb_dir = _build_notebook(
            tmp_path,
            db_path=db_path,
            cells=[
                ("sql", "sql", "# @sql connection=db\n# @name q\nSELECT id FROM events\n"),
                ("count", "python", "total = len(q)\n"),
                ("show", "python", "print(total)\n"),
            ],
        )
        executor = CellExecutor(_session(nb_dir))

        async def run(cell_id: str) -> Any:
            result = await executor.execute_cell(cell_id, _read(nb_dir, cell_id))
            assert result.success, result.error
            return result

        await run("sql")
        await run("count")

        # The same rows again: the query runs, and its consumer keeps its cache.
        assert (await run("sql")).cache_hit is False
        assert (await run("count")).cache_hit is True

        with sqlite3.connect(db_path) as conn:
            conn.execute("INSERT INTO events (id, name, value) VALUES (4, 'delta', 40)")
            conn.commit()

        await run("sql")
        count = await run("count")
        show = await run("show")

        assert count.cache_hit is False, "the consumer served a count of the old rows"
        assert show.stdout.strip() == "4"

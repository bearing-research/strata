"""DuckDB SQL cells over mounts and a named catalog, without services. Item 26.

A mount is a view the query reads by name, its files are an input (a new file
misses the cache and makes the cell stale), and a catalog table the query reads
is pinned at the snapshot its provenance folds. Reading a real catalog is in
``test_e2e_duckdb_lake.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("duckdb")

from strata.config import StrataConfig  # noqa: E402
from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.models import CellStatus, ConnectionSpec, MountSpec  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.sql.drivers.duckdb import DuckDBAdapter  # noqa: E402
from strata.notebook.sql.lake import lake_tables, pin_snapshots  # noqa: E402
from strata.notebook.writer import (  # noqa: E402
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)


def _notebook(tmp_path: Path, cells: dict[str, str], connection: str) -> Path:
    nb_dir = create_notebook(tmp_path, "duckdb_lake")
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb_dir, cell_id, language="sql")
        write_cell(nb_dir, cell_id, source)
    update_notebook_mounts(nb_dir, [MountSpec(name="raw", uri=(tmp_path / "raw").as_uri())])
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + f"\n[connections.lake]\n{connection}\n")
    return nb_dir


def _write_parquet(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"k": values}), path)


def _lake_name(uri: str) -> str:
    import hashlib

    return "lake_" + hashlib.sha256(uri.encode()).hexdigest()[:16]


def _rows(session: NotebookSession, uri: str) -> list[dict[str, Any]]:
    art_id, version = uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
    blob = session.get_artifact_manager().load_artifact_data(art_id, int(version))
    return pa.ipc.open_stream(blob).read_all().to_pylist()


async def _run(nb_dir: Path, session: NotebookSession, cell_id: str) -> Any:
    source = (nb_dir / "cells" / f"{cell_id}.py").read_text()
    result = await CellExecutor(session).execute_cell(cell_id, source)
    # What the route does after a run.
    session.compute_staleness()
    if result.success:
        session.mark_executed_ready(cell_id)
    return result


@pytest.mark.asyncio
async def test_a_mount_is_a_view_and_a_new_file_makes_the_cell_stale(tmp_path):
    _write_parquet(tmp_path / "raw" / "2026" / "a.parquet", [1, 2])
    nb_dir = _notebook(
        tmp_path,
        # No database probe, so the mount's fingerprint is all that tells the
        # two runs apart.
        {"c1": "# @sql connection=lake\n# @cache forever\nSELECT sum(k) AS total FROM raw\n"},
        'driver = "duckdb"\npath = ":memory:"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    first = await _run(nb_dir, session, "c1")
    assert first.success, first.error
    assert _rows(session, first.artifact_uri) == [{"total": 3}]
    assert session.compute_staleness()["c1"].status == CellStatus.READY

    _write_parquet(tmp_path / "raw" / "2026" / "b.parquet", [10])
    assert session.compute_staleness()["c1"].status != CellStatus.READY
    second = await _run(nb_dir, session, "c1")
    assert second.success, second.error
    assert second.cache_hit is False
    assert _rows(session, second.artifact_uri) == [{"total": 13}]

    third = await _run(nb_dir, session, "c1")
    assert third.cache_hit is True


@pytest.mark.asyncio
async def test_a_database_file_and_its_mounts_are_read_together_and_never_written(tmp_path):
    import duckdb

    db = tmp_path / "base.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE labels AS SELECT * FROM (VALUES (1, 'one'), (2, 'two')) t(k, v)")
    _write_parquet(tmp_path / "raw" / "a.parquet", [1, 2])
    nb_dir = _notebook(
        tmp_path,
        {
            "c1": "# @sql connection=lake\nSELECT v FROM raw JOIN labels USING (k) ORDER BY v\n",
            "c2": "# @sql connection=lake\nINSERT INTO labels VALUES (3, 'three')\n",
        },
        f'driver = "duckdb"\npath = "{db}"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    read = await _run(nb_dir, session, "c1")
    assert read.success, read.error
    assert _rows(session, read.artifact_uri) == [{"v": "one"}, {"v": "two"}]

    write = await _run(nb_dir, session, "c2")
    assert write.success is False
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM labels").fetchone() == (2,)


@pytest.mark.asyncio
async def test_a_mount_the_cell_does_not_declare_fails_naming_it(tmp_path):
    _write_parquet(tmp_path / "raw" / "a.parquet", [1])
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT * FROM cooked\n"},
        'driver = "duckdb"\npath = ":memory:"\nmounts = ["cooked"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")
    assert result.success is False
    assert "mount 'cooked' is not declared" in result.error


@pytest.mark.asyncio
async def test_a_catalog_the_server_does_not_configure_fails_naming_it(tmp_path, monkeypatch):
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT * FROM lake.taxi.trips\n"},
        'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"',
    )
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: StrataConfig())
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: StrataConfig())
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")
    assert result.success is False
    assert "catalog 'lake' is not configured" in result.error


def test_only_the_catalog_tables_a_query_reads_are_pinned():
    sql = (
        "WITH trips AS (SELECT 1 AS x) "
        "SELECT * FROM lake.taxi.trips t JOIN trips USING (x) "
        "JOIN warehouse.taxi.trips w USING (x) JOIN lake.taxi.zones z USING (x) WHERE t.id > ?"
    )
    pinned = pin_snapshots(sql, "lake", {("taxi", "trips"): 11, ("taxi", "zones"): 22})

    assert "lake.taxi.trips AT (VERSION => 11)" in pinned
    assert "lake.taxi.zones AT (VERSION => 22)" in pinned
    assert "warehouse.taxi.trips AS w" in pinned
    assert "JOIN trips USING" in pinned
    assert pinned.endswith("> ?")


def test_lake_tables_are_the_catalog_tables_a_read_cell_reads(tmp_path):
    nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    state = parse_notebook(nb_dir)
    read = "# @sql connection=lake\nSELECT * FROM lake.taxi.trips JOIN lake.taxi.zones USING (id)\n"

    assert sorted(t.uri for t in lake_tables(state, read)) == [
        "lake:taxi.trips",
        "lake:taxi.zones",
    ]
    assert lake_tables(state, "# @sql connection=lake write\nDELETE FROM lake.taxi.trips\n") == []
    state.connections = [ConnectionSpec(name="lake", driver="sqlite", path=":memory:")]
    assert lake_tables(state, read) == []


def test_the_catalog_and_mounts_are_part_of_the_connection_identity():
    adapter = DuckDBAdapter()
    base = {"name": "lake", "driver": "duckdb", "path": ":memory:"}
    identities = {
        adapter.canonicalize_connection_id(ConnectionSpec(**base, **extra))
        for extra in ({}, {"catalog": "lake"}, {"catalog": "other"}, {"mounts": ["raw"]})
    }
    assert len(identities) == 4


@pytest.mark.asyncio
async def test_a_database_file_named_like_a_reserved_database_still_opens(tmp_path):
    import duckdb

    db = tmp_path / "memory.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE labels AS SELECT 1 AS k, 'one' AS v")
    _write_parquet(tmp_path / "raw" / "a.parquet", [1])
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT v FROM raw JOIN labels USING (k)\n"},
        f'driver = "duckdb"\npath = "{db}"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")

    assert result.success, result.error
    assert _rows(session, result.artifact_uri) == [{"v": "one"}]


def test_a_table_the_first_resolution_missed_reads_what_the_retry_found(tmp_path, monkeypatch):
    from strata.notebook import tables
    from strata.notebook.sql.adapter import QualifiedTable
    from strata.notebook.sql.lake import resolve_lake

    nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    config = StrataConfig(catalogs={"lake": {"type": "rest", "uri": "http://catalog"}})
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    monkeypatch.setattr(tables, "fingerprint_tables", lambda specs, cfg: (["x:unresolved"], {}))
    monkeypatch.setattr(tables, "resolve_table_snapshot", lambda spec, cfg: 7)

    lake = resolve_lake(
        session,
        "c1",
        "",
        session.notebook_state.connections[0],
        [QualifiedTable("lake", "taxi", "trips")],
    )

    assert lake.snapshots == {("taxi", "trips"): 7}
    # And the cell's key names that snapshot, not the random stand-in
    # fingerprint_tables invents for what it could not resolve — which no
    # later run would ever reproduce.
    assert lake.fingerprints == [f"{_lake_name('lake:taxi.trips')}:table:lake:taxi.trips:7"]


def test_the_catalogs_s3_secret_reaches_only_its_warehouse():
    from strata.notebook.sql.drivers.duckdb import _attach_catalog

    class Recorder:
        def __init__(self):
            self.statements: list[str] = []

        def execute(self, statement):
            self.statements.append(statement)

    keys = {
        "type": "rest",
        "uri": "http://catalog",
        "s3.access-key-id": "k",
        "s3.secret-access-key": "s",
    }
    scoped, unscoped = Recorder(), Recorder()
    _attach_catalog(scoped, "lake", {**keys, "warehouse": "s3://lake/wh"})
    _attach_catalog(unscoped, "lake", {**keys, "warehouse": "prod"})

    (secret,) = [s for s in scoped.statements if "TYPE s3" in s]
    assert "SCOPE 's3://lake/wh'" in secret
    assert not [s for s in unscoped.statements if "TYPE s3" in s]
    assert all(s.endswith("READ_ONLY)") for s in scoped.statements if s.startswith("ATTACH"))


def test_a_python_notebook_needs_no_sql_extra(tmp_path):
    """Staleness and provenance of non-SQL cells must not import sqlglot."""
    import subprocess
    import sys
    import textwrap

    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "plain")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules["sqlglot"] = None  # an import of it now fails
        import asyncio
        from pathlib import Path
        from strata.notebook.executor import CellExecutor
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        nb = Path({str(nb_dir)!r})
        session = NotebookSession(parse_notebook(nb), nb)
        session.compute_staleness()
        asyncio.run(CellExecutor(session)._compute_cell_provenance("c1", "x = 1\\n"))
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.stdout.strip().endswith("ok"), done.stderr


class TestAReadCellReads:
    """The connection opens read-only, but a body can end that transaction and
    keep going, so what a read cell may run is decided before anything is sent
    to the driver."""

    @staticmethod
    async def _run(tmp_path, body: str):
        nb_dir = _notebook(
            tmp_path,
            {"c1": f"# @sql connection=lake\n{body}\n"},
            'driver = "duckdb"\npath = ":memory:"',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        return await _run(nb_dir, session, "c1")

    @pytest.mark.asyncio
    async def test_a_committed_attach_cannot_write_another_database(self, tmp_path):
        import duckdb

        victim = tmp_path / "victim.duckdb"
        with duckdb.connect(str(victim)) as conn:
            conn.execute("CREATE TABLE t AS SELECT 1 AS x")
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(
            tmp_path,
            f"COMMIT; ATTACH '{victim}' AS w (READ_WRITE); "
            "CREATE TABLE w.main.pwn AS SELECT 42 AS x; SELECT 1 AS ok",
        )

        assert result.success is False
        assert "is not a read" in result.error
        with duckdb.connect(str(victim), read_only=True) as conn:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert tables == ["t"], "a read cell wrote into another database"

    @pytest.mark.asyncio
    async def test_copying_out_of_the_database_is_not_a_read(self, tmp_path):
        target = tmp_path / "leak.csv"
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(tmp_path, f"COPY (SELECT 1 AS x) TO '{target}'")

        assert result.success is False
        assert "COPY is not a read" in result.error
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_reads_still_run(self, tmp_path):
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(
            tmp_path, "WITH a AS (SELECT 1 AS x) SELECT sum(x) AS total FROM a"
        )

        assert result.success, result.error


class TestHowACatalogTableIsWritten:
    """DuckDB resolves a database and a schema case-insensitively, and a
    two-part name takes the catalog's default schema. Every spelling is the
    same table, and one that is missed is read live under a provenance that
    never goes stale."""

    @staticmethod
    def _tables(state, body: str):
        from strata.notebook.sql.lake import lake_tables

        return sorted(t.uri for t in lake_tables(state, f"# @sql connection=lake\n{body}\n"))

    def test_the_spellings_of_one_table_are_that_table(self, tmp_path):
        nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
        state = parse_notebook(nb_dir)

        assert self._tables(state, "SELECT * FROM LAKE.taxi.trips") == ["lake:taxi.trips"]
        assert self._tables(state, 'SELECT * FROM "LAKE".taxi.trips') == ["lake:taxi.trips"]
        assert self._tables(state, "SELECT * FROM lake.trips") == ["lake:main.trips"]
        assert self._tables(state, "SELECT * FROM other.taxi.trips") == []

    def test_every_spelling_is_pinned(self):
        from strata.notebook.sql.lake import pin_snapshots

        snapshots = {("taxi", "trips"): 11, ("main", "zones"): 22}

        pinned = pin_snapshots(
            "SELECT * FROM LAKE.taxi.trips JOIN lake.zones USING (id)", "lake", snapshots
        )

        assert "AT (VERSION => 11)" in pinned
        assert "AT (VERSION => 22)" in pinned


class TestWhatCountsAsAReadStatement:
    """The classifier decides what a read cell may send to the driver. Two
    ways to be wrong: a write it lets through, and a read it refuses."""

    @staticmethod
    def _violation(sql: str, dialect: str = "duckdb"):
        from strata.notebook.sql.analyzer import read_only_violation

        return read_only_violation(sql, dialect)

    @pytest.mark.asyncio
    async def test_explain_analyze_runs_what_it_wraps_so_it_is_not_a_read(self, tmp_path):
        """DuckDB's EXPLAIN ANALYZE executes the statement it describes, which
        made it a way around every refusal below it."""
        target = tmp_path / "leak.csv"
        nb_dir = _notebook(
            tmp_path,
            {"c1": f"# @sql connection=lake\nEXPLAIN ANALYZE COPY (SELECT 1 AS x) TO '{target}'\n"},
            'driver = "duckdb"\npath = ":memory:"',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await _run(nb_dir, session, "c1")

        assert result.success is False
        assert "EXPLAIN ANALYZE" in result.error
        assert not target.exists(), "a read cell wrote a file through EXPLAIN ANALYZE"

    def test_a_plain_explain_is_still_a_read(self):
        assert self._violation("EXPLAIN SELECT 1") is None

    def test_the_parenthesised_form_is_caught_too(self):
        assert "EXPLAIN ANALYZE" in (self._violation("EXPLAIN (ANALYZE) DELETE FROM t") or "")

    def test_reads_that_are_not_selects_are_allowed(self):
        for sql in ("VALUES (1), (2)", "SUMMARIZE t", "TABLE t", "DESCRIBE t", "SHOW TABLES"):
            assert self._violation(sql) is None, sql
        assert self._violation("SHOW search_path", "postgres") is None

    def test_the_message_names_the_annotation_that_works(self):
        """`write` alone is ignored by the parser; the flag is `write=true`."""
        from strata.notebook.annotations import parse_annotations

        message = self._violation("INSERT INTO t VALUES (1)") or ""
        assert "write=true" in message
        annotation = parse_annotations(
            "# @sql connection=db write=true\nINSERT INTO t VALUES (1)\n"
        )
        assert annotation.sql is not None and annotation.sql.write is True

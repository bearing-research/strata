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

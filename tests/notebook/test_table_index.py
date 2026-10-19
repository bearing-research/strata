"""``.strata/tables/index.json``: the tables each cell read on its last good run, and the
snapshot of each, for a host to list the cells that read a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from strata.config import StrataConfig
from strata.notebook import table_index
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

windows = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local warehouse paths fail on Windows"
)


def _warehouse(tmp_path: Path):
    """A one-table warehouse written once; returns (table, uri)."""
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
    )
    catalog.create_namespace("db")
    table = catalog.create_table("db.events", Schema(NestedField(1, "id", LongType())))
    table.append(pa.table({"id": pa.array([1, 2], type=pa.int64())}))
    return table, f"file://{warehouse}#db.events"


def _session(tmp_path: Path, cells: dict[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Reads")
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str, source: str | None = None):
    cell = session.notebook_state.get_cell(cell_id)
    if source is not None and cell.source != source:
        cell.source = source
        session.re_analyze_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


def _index(session: NotebookSession) -> dict:
    return json.loads(table_index.index_path(session.path).read_text())


@windows
class TestATableCell:
    @pytest.mark.asyncio
    async def test_its_run_records_the_table_and_the_snapshot_it_read(self, tmp_path):
        table, uri = _warehouse(tmp_path)
        session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})

        await _run(session, "c1")

        snapshot = table.current_snapshot().snapshot_id
        assert _index(session) == {
            "c1": [{"uri": uri, "catalog": None, "table": "db.events", "snapshot_id": snapshot}]
        }

    @pytest.mark.asyncio
    async def test_a_new_snapshot_read_replaces_the_old_and_a_cache_hit_keeps_it(self, tmp_path):
        table, uri = _warehouse(tmp_path)
        session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
        await _run(session, "c1")

        table.append(pa.table({"id": pa.array([3], type=pa.int64())}))
        await _run(session, "c1")
        second = table.current_snapshot().snapshot_id
        assert _index(session)["c1"][0]["snapshot_id"] == second

        hit = await _run(session, "c1")
        assert hit.cache_hit is True
        assert _index(session)["c1"][0]["snapshot_id"] == second

    @pytest.mark.asyncio
    async def test_a_pinned_table_records_its_pin(self, tmp_path):
        table, uri = _warehouse(tmp_path)
        first = table.current_snapshot().snapshot_id
        table.append(pa.table({"id": pa.array([3], type=pa.int64())}))
        session = _session(
            tmp_path, {"c1": f"# @table events {uri} snapshot={first}\nsnap = events_snapshot\n"}
        )

        await _run(session, "c1")

        assert _index(session)["c1"][0]["snapshot_id"] == first

    @pytest.mark.asyncio
    async def test_a_failed_run_leaves_the_last_good_runs_reads(self, tmp_path):
        table, uri = _warehouse(tmp_path)
        good = f"# @table events {uri}\nsnap = events_snapshot\n"
        session = _session(tmp_path, {"c1": good})
        await _run(session, "c1")
        before = _index(session)

        table.append(pa.table({"id": pa.array([3], type=pa.int64())}))
        cell = session.notebook_state.get_cell("c1")
        cell.source = good + "raise ValueError('no')\n"
        session.re_analyze_cell("c1")
        failed = await CellExecutor(session).execute_cell("c1", cell.source)

        assert not failed.success
        assert _index(session) == before

    @pytest.mark.asyncio
    async def test_a_failed_table_write_leaves_the_last_good_runs_reads(self, tmp_path):
        table, uri = _warehouse(tmp_path)
        target = f"file://{tmp_path / 'out'}#lab.out"
        head = f"# @table events {uri}\n# @write_table rows {target} append\n"
        session = _session(
            tmp_path, {"c1": head + "import pyarrow as pa\nrows = pa.table({'id': [1]})\n"}
        )
        await _run(session, "c1")
        before = _index(session)

        table.append(pa.table({"id": pa.array([3], type=pa.int64())}))
        cell = session.notebook_state.get_cell("c1")
        cell.source = head + "rows = {'not': 'a table'}\n"
        session.re_analyze_cell("c1")
        failed = await CellExecutor(session).execute_cell("c1", cell.source)

        assert not failed.success
        assert _index(session) == before

    @pytest.mark.asyncio
    async def test_a_run_all_cache_hit_records_its_reads(self, tmp_path):
        from tests.notebook.test_executor_batch import _cell_spec

        _, uri = _warehouse(tmp_path)
        source = f"# @table events {uri}\nsnap = events_snapshot\n"
        session = _session(tmp_path, {"c1": source})
        await _run(session, "c1")
        recorded = _index(session)
        table_index.index_path(session.path).unlink()

        batch = await CellExecutor(session).execute_batch([_cell_spec("c1", source)])

        assert [r.status for r in batch.cell_results] == ["cache_hit"]
        assert _index(session) == recorded

    @pytest.mark.asyncio
    async def test_a_loop_cache_hit_records_its_reads(self, tmp_path):
        _, uri = _warehouse(tmp_path)
        loop = f"# @loop max_iter=1 carry=state\n# @table events {uri}\nstate = state + 1\n"
        session = _session(tmp_path, {"seed": "state = 0\n", "loop": loop})
        await _run(session, "loop")
        recorded = _index(session)
        table_index.index_path(session.path).unlink()

        hit = await _run(session, "loop")

        assert hit.cache_hit is True
        assert _index(session) == recorded

    @pytest.mark.asyncio
    async def test_dropping_the_annotation_or_the_cell_drops_its_entry(self, tmp_path):
        _, uri = _warehouse(tmp_path)
        source = f"# @table events {uri}\nsnap = events_snapshot\n"
        session = _session(tmp_path, {"c1": source, "c2": source})
        await _run(session, "c1")
        await _run(session, "c2")
        assert set(_index(session)) == {"c1", "c2"}

        await _run(session, "c1", "snap = 1\n")
        assert set(_index(session)) == {"c2"}

        session.remove_cell("c2")
        assert _index(session) == {}


class TestASqlCellsCatalogTables:
    def test_each_is_a_read_a_table_never_written_with_no_snapshot(self, tmp_path, monkeypatch):
        from strata.notebook import tables
        from strata.notebook.sql.adapter import QualifiedTable
        from strata.notebook.sql.lake import resolve_lake

        nb = create_notebook(tmp_path, "Lake")
        toml = nb / "notebook.toml"
        toml.write_text(
            toml.read_text()
            + '\n[connections.lake]\ndriver = "duckdb"\npath = ":memory:"\ncatalog = "lake"\n'
        )
        session = NotebookSession(parse_notebook(nb), nb)
        config = StrataConfig(catalogs={"lake": {"type": "rest", "uri": "http://catalog"}})
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        monkeypatch.setattr(
            tables,
            "fingerprint_tables",
            lambda specs, cfg, env=None, notebook=None: (
                [],
                {spec.name: 7 if spec.uri.endswith("trips") else None for spec in specs},
            ),
        )

        lake = resolve_lake(
            session,
            "c1",
            "",
            session.notebook_state.connections[0],
            [QualifiedTable("lake", "taxi", "trips"), QualifiedTable("lake", "taxi", "zones")],
        )

        assert sorted(lake.reads, key=lambda read: read["uri"]) == [
            {"uri": "lake:taxi.trips", "catalog": "lake", "table": "taxi.trips", "snapshot_id": 7},
            {
                "uri": "lake:taxi.zones",
                "catalog": "lake",
                "table": "taxi.zones",
                "snapshot_id": None,
            },
        ]


class TestTheIndex:
    def test_a_damaged_one_is_rewritten_by_the_next_run(self, tmp_path):
        path = table_index.index_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        read = {"uri": "u", "catalog": None, "table": "db.t", "snapshot_id": 1}

        table_index.record(tmp_path, "c1", [read], ["c1"])

        assert json.loads(path.read_text()) == {"c1": [read]}

    def test_a_notebook_that_reads_no_tables_has_no_index(self, tmp_path):
        table_index.record(tmp_path, "c1", [], ["c1"])

        assert not table_index.index_path(tmp_path).exists()

    def test_a_cell_gone_from_the_notebook_is_dropped_at_the_next_write(self, tmp_path):
        # As when notebook.toml changes under the session, say by a pull.
        read = {"uri": "u", "catalog": None, "table": "db.t", "snapshot_id": 1}
        table_index.record(tmp_path, "gone", [read], ["gone"])

        table_index.record(tmp_path, "c1", [read], ["c1"])

        assert json.loads(table_index.index_path(tmp_path).read_text()) == {"c1": [read]}

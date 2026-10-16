"""``# @write_table``: a cell's variable written into an Iceberg table when the run succeeds.

The write happens before the run's outputs are kept, so a failed write keeps nothing and
the next run writes again; a cache hit writes nothing. Each snapshot names the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

from strata.iceberg.catalog import SUMMARY_ARTIFACT_ID, SUMMARY_PROVENANCE, SUMMARY_VERSION
from strata.notebook.annotations import parse_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, TableWriteSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.provenance import compute_source_hash
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

windows = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local warehouse paths fail on Windows"
)


def _warehouse(tmp_path: Path) -> tuple[Path, str]:
    warehouse = tmp_path / "warehouse"
    return warehouse, f"file://{warehouse}#lab.readings"


def _load(warehouse: Path):
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=str(warehouse),
    )
    return catalog.load_table("lab.readings")


def _rows(warehouse: Path) -> list[tuple[int, int]]:
    table = _load(warehouse).scan().to_arrow().sort_by("id")
    return list(zip(table["id"].to_pylist(), table["v"].to_pylist(), strict=True))


def _source(uri: str, rows: str, mode: str = "append", extra: str = "") -> str:
    return (
        f"# @write_table rows {uri} {mode}{extra}\nimport pyarrow as pa\nrows = pa.table({rows})\n"
    )


def _session(tmp_path: Path, source: str, reader: str | None = None) -> NotebookSession:
    nb = create_notebook(tmp_path, "Writes")
    add_cell_to_notebook(nb, "w", None)
    write_cell(nb, "w", source)
    if reader is not None:
        add_cell_to_notebook(nb, "r", "w")
        write_cell(nb, "r", reader)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, source: str, cell_id: str = "w"):
    cell = session.notebook_state.get_cell(cell_id)
    if cell.source != source:
        cell.source = source
        session.re_analyze_cell(cell_id)
    return await CellExecutor(session).execute_cell(cell_id, source)


class TestTheAnnotation:
    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            (
                "rows lake:lab.readings append",
                TableWriteSpec(variable="rows", table="lake:lab.readings", mode="append"),
            ),
            (
                "fixes lake:lab.readings merge on=id,day",
                TableWriteSpec(
                    variable="fixes", table="lake:lab.readings", mode="merge", on=["id", "day"]
                ),
            ),
        ],
    )
    def test_it_parses(self, line, expected):
        assert parse_annotations(f"# @write_table {line}\nx = 1").table_writes == [expected]

    @pytest.mark.parametrize(
        "line",
        [
            "rows lake:lab.readings",  # the mode is required
            "rows lake:lab.readings upsert",
            "rows lake:lab.readings merge",  # merge needs keys
            "rows lake:lab.readings append on=id",  # keys mean merge
            "2rows lake:lab.readings append",
        ],
    )
    def test_a_line_it_cannot_read_is_dropped(self, line):
        assert parse_annotations(f"# @write_table {line}\nx = 1").table_writes == []

    def test_the_target_and_mode_are_part_of_the_source_hash(self):
        """Changing them runs the cell again, so the new write is not a cache hit away."""
        base = compute_source_hash("# @write_table rows lake:a.t append\nrows = 1")
        assert base != compute_source_hash("# @write_table rows lake:a.u append\nrows = 1")
        assert base != compute_source_hash("# @write_table rows lake:a.t overwrite\nrows = 1")
        assert base != compute_source_hash("rows = 1")


@windows
class TestAWritingCell:
    async def test_appends_and_merges_each_name_the_run(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)

        result = await _run(session, first)
        assert result.success, result.error
        second = _source(uri, '{"id": [3], "v": [30]}')
        assert (await _run(session, second)).success
        merge = _source(uri, '{"id": [2, 4], "v": [99, 40]}', "merge", " on=id")
        assert (await _run(session, merge)).success

        assert _rows(warehouse) == [(1, 10), (2, 99), (3, 30), (4, 40)]
        manager = session.get_artifact_manager()
        artifact_id = manager.cell_artifact_id("w", "rows")
        latest = manager.artifact_store.get_latest_version(artifact_id)
        summaries = [s.summary for s in _load(warehouse).snapshots()]
        assert {s.get(SUMMARY_ARTIFACT_ID) for s in summaries} == {artifact_id}
        assert summaries[-1].get(SUMMARY_VERSION) == str(latest.version)
        assert summaries[-1].get(SUMMARY_PROVENANCE) == latest.provenance_hash

    async def test_a_cache_hit_writes_nothing(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        source = _source(uri, '{"id": [1], "v": [10]}')
        session = _session(tmp_path, source)

        assert (await _run(session, source)).success
        again = await _run(session, source)

        assert again.cache_hit
        assert _rows(warehouse) == [(1, 10)]
        assert len(_load(warehouse).snapshots()) == 1

    async def test_a_reader_goes_stale_after_each_write(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        source = _source(uri, '{"id": [1], "v": [10]}')
        reader = f"# @table readings {uri}\nseen = readings_snapshot\n"
        session = _session(tmp_path, source, reader)
        assert (await _run(session, source)).success
        assert (await _run(session, reader, "r")).success
        assert session.compute_staleness()["r"].status == CellStatus.READY

        assert (await _run(session, _source(uri, '{"id": [2], "v": [20]}'))).success

        assert session.compute_staleness()["r"].status != CellStatus.READY

    async def test_a_failed_write_keeps_nothing_so_the_next_run_writes(self, tmp_path, monkeypatch):
        """A table another writer created is refused until the server allows it."""
        warehouse, uri = _warehouse(tmp_path)
        warehouse.mkdir()
        from pyiceberg.catalog.sql import SqlCatalog

        catalog = SqlCatalog(
            "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
        )
        catalog.create_namespace("lab")
        catalog.create_table(
            "lab.readings", schema=pa.schema([("id", pa.int64()), ("v", pa.int64())])
        ).append(pa.table({"id": [7], "v": [70]}))
        source = _source(uri, '{"id": [1], "v": [10]}')
        session = _session(tmp_path, source)

        refused = await _run(session, source)

        assert not refused.success
        assert "STRATA_NOTEBOOK_TABLE_WRITES_FOREIGN" in refused.error
        manager = session.get_artifact_manager()
        assert (
            manager.artifact_store.get_latest_version(manager.cell_artifact_id("w", "rows")) is None
        )

        from strata.config import StrataConfig

        allowed = StrataConfig(notebook_table_writes_foreign=True)
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: allowed)
        again = await _run(session, source)

        assert again.success, again.error
        assert not again.cache_hit
        assert _rows(warehouse) == [(1, 10), (7, 70)]

    async def test_overwriting_a_table_strata_did_not_write_stays_refused(
        self, tmp_path, monkeypatch
    ):
        from pyiceberg.catalog.sql import SqlCatalog

        from strata.config import StrataConfig

        warehouse, uri = _warehouse(tmp_path)
        warehouse.mkdir()
        catalog = SqlCatalog(
            "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
        )
        catalog.create_namespace("lab")
        catalog.create_table(
            "lab.readings", schema=pa.schema([("id", pa.int64()), ("v", pa.int64())])
        ).append(pa.table({"id": [7], "v": [70]}))
        monkeypatch.setattr(
            CellExecutor,
            "_lake_config",
            lambda self: StrataConfig(notebook_table_writes_foreign=True),
        )
        source = _source(uri, '{"id": [1], "v": [10]}', "overwrite")

        result = await _run(_session(tmp_path, source), source)

        assert not result.success
        assert "refusing to overwrite" in result.error
        assert _rows(warehouse) == [(7, 70)]

    async def test_a_value_that_is_not_a_table_fails_the_run(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        source = f"# @write_table rows {uri} append\nrows = {{'id': 1}}\n"

        result = await _run(_session(tmp_path, source), source)

        assert not result.success
        assert "not a table" in result.error
        assert not warehouse.exists() or not (warehouse / "catalog.db").exists()

    async def test_a_leaf_variable_is_kept_to_be_written(self, tmp_path):
        """Nothing downstream reads ``rows``; the write still needs it."""
        warehouse, uri = _warehouse(tmp_path)
        source = _source(uri, '{"id": [1], "v": [10]}')
        session = _session(tmp_path, source)

        assert "rows" in session.dag.consumed_variables["w"]
        assert (await _run(session, source)).success
        assert _rows(warehouse) == [(1, 10)]

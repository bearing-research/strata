"""``# @write_table``: a cell's variable written into an Iceberg table when the run succeeds.

The write happens before the run's outputs are kept, so a failed write keeps nothing and
the next run writes again; a cache hit writes nothing. Each snapshot names the run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.exceptions import NoSuchTableError

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


def _load(warehouse: Path, catalog_name: str = "strata"):
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        catalog_name,
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=str(warehouse),
    )
    return catalog.load_table("lab.readings")


def _rows(warehouse: Path, catalog_name: str = "strata") -> list[tuple[int, int]]:
    table = _load(warehouse, catalog_name).scan().to_arrow().sort_by("id")
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
                "now lake:lab.readings sync on=id",
                TableWriteSpec(variable="now", table="lake:lab.readings", mode="sync", on=["id"]),
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
            "rows lake:lab.readings sync",  # sync needs keys
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


def _notebook(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Writes")
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


@windows
class TestALoopCell:
    """The loop's final values are written, once, before they become current."""

    SEED = "import pyarrow as pa\nrows = pa.table({'id': [0], 'v': [0]})\n"

    @staticmethod
    def _loop(uri: str, body: str) -> str:
        return f"# @loop max_iter=2 carry=rows\n# @write_table rows {uri} append\n{body}"

    async def test_writes_its_final_value_once(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        loop = self._loop(
            uri,
            "import pyarrow as pa\n"
            "rows = pa.table({'id': [rows['id'][0].as_py() + 1], 'v': [1]})\n",
        )
        session = _notebook(tmp_path, [("s", self.SEED), ("l", loop)])
        assert (await _run(session, self.SEED, "s")).success

        result = await _run(session, loop, "l")

        assert result.success, result.error
        assert _rows(warehouse) == [(2, 1)]
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("l", "rows"))
        summary = _load(warehouse).current_snapshot().summary
        assert summary.get(SUMMARY_VERSION) == str(latest.version)
        assert summary.get(SUMMARY_PROVENANCE) == latest.provenance_hash

    async def test_a_failed_write_keeps_nothing(self, tmp_path):
        _, uri = _warehouse(tmp_path)
        loop = self._loop(uri, "rows = {'id': 1}\n")
        session = _notebook(tmp_path, [("s", self.SEED), ("l", loop)])
        assert (await _run(session, self.SEED, "s")).success

        result = await _run(session, loop, "l")

        assert not result.success
        assert "not a table" in result.error
        manager = session.get_artifact_manager()
        assert (
            manager.artifact_store.get_latest_version(manager.cell_artifact_id("l", "rows")) is None
        )


@windows
class TestARevert:
    """Going back to source that ran before: a cache hit, unless the write replaced rows."""

    async def test_an_overwrite_writes_its_rows_again(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1], "v": [10]}', "overwrite")
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success
        assert (await _run(session, _source(uri, '{"id": [2], "v": [20]}', "overwrite"))).success

        reverted = await _run(session, first)

        assert reverted.success, reverted.error
        assert _rows(warehouse) == [(1, 10)]
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("w", "rows"))
        summary = _load(warehouse).current_snapshot().summary
        assert summary.get(SUMMARY_VERSION) == str(latest.version)
        assert summary.get(SUMMARY_PROVENANCE) == latest.provenance_hash

    async def test_an_append_is_a_cache_hit_that_writes_nothing(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1], "v": [10]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success
        assert (await _run(session, _source(uri, '{"id": [2], "v": [20]}'))).success

        reverted = await _run(session, first)

        assert reverted.cache_hit
        assert _rows(warehouse) == [(1, 10), (2, 20)]

    @staticmethod
    async def _run_all_reverted(tmp_path: Path, mode: str) -> tuple[Path, NotebookSession]:
        from strata.notebook.ws import (
            _ensure_execution_state,
            _handle_notebook_run_all,
            _notebook_connections,
            forget_notebook_execution_state,
        )

        class _Socket:
            async def send_text(self, text: str) -> None:
                return None

        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1], "v": [10]}', mode)
        session = _session(tmp_path, first, reader="n = rows.num_rows")
        session.environment_sync_state = "ready"
        assert (await _run(session, first)).success
        assert (await _run(session, _source(uri, '{"id": [2], "v": [20]}', mode))).success
        cell = session.notebook_state.get_cell("w")
        cell.source = first
        session.re_analyze_cell("w")

        socket = _Socket()
        _notebook_connections[session.id] = [socket]
        try:
            state = _ensure_execution_state(session.id)
            await _handle_notebook_run_all(socket, session, state, session.id, {})
            assert state.execution_task is not None
            await state.execution_task
        finally:
            _notebook_connections.pop(session.id, None)
            forget_notebook_execution_state(session.id)
        assert [c.status for c in session.notebook_state.cells] == [CellStatus.READY] * 2
        return warehouse, session

    async def test_run_all_keeps_an_append_revert_a_cache_hit(self, tmp_path):
        warehouse, session = await self._run_all_reverted(tmp_path, "append")

        assert _rows(warehouse) == [(1, 10), (2, 20)]
        assert session.notebook_state.get_cell("w").cache_hit

    async def test_run_all_writes_an_overwrite_revert_again(self, tmp_path):
        warehouse, session = await self._run_all_reverted(tmp_path, "overwrite")

        assert not session.notebook_state.get_cell("w").cache_hit
        assert _rows(warehouse) == [(1, 10)]

    async def test_a_loop_append_revert_is_a_cache_hit(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)

        def loop(v: int) -> str:
            return TestALoopCell._loop(
                uri,
                "import pyarrow as pa\n"
                f"rows = pa.table({{'id': [rows['id'][0].as_py() + 1], 'v': [{v}]}})\n",
            )

        session = _notebook(tmp_path, [("s", TestALoopCell.SEED), ("l", loop(1))])
        assert (await _run(session, TestALoopCell.SEED, "s")).success
        assert (await _run(session, loop(1), "l")).success
        assert (await _run(session, loop(2), "l")).success

        reverted = await _run(session, loop(1), "l")

        assert reverted.success, reverted.error
        assert sorted(_rows(warehouse)) == [(2, 1), (2, 2)]
        assert reverted.cache_hit


@windows
class TestASyncedTable:
    """The table ends up holding what the cell produced, by key, changing only what differs."""

    @staticmethod
    def _sync(uri: str, rows: str, on: str = "id") -> str:
        return _source(uri, rows, "sync", f" on={on}")

    async def test_a_row_the_cell_no_longer_produces_is_deleted(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2, 3], "v": [10, 20, 30]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        second = self._sync(uri, '{"id": [2, 3, 4], "v": [20, 33, 40]}')
        result = await _run(session, second)

        assert result.success, result.error
        assert _rows(warehouse) == [(2, 20), (3, 33), (4, 40)]
        manager = session.get_artifact_manager()
        artifact_id = manager.cell_artifact_id("w", "rows")
        latest = manager.artifact_store.get_latest_version(artifact_id)
        snapshots = list(_load(warehouse).snapshots())
        summary = snapshots[-1].summary
        assert summary.get(SUMMARY_VERSION) == str(latest.version)
        assert summary.get(SUMMARY_PROVENANCE) == latest.provenance_hash
        # Every snapshot of the commit names the run, the delete as well as the upsert.
        assert {s.summary.get(SUMMARY_ARTIFACT_ID) for s in snapshots} == {artifact_id}

    async def test_a_cell_producing_no_rows_empties_the_table(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        empty = _source(
            uri,
            '{"id": pa.array([], pa.int64()), "v": pa.array([], pa.int64())}',
            "sync",
            " on=id",
        )
        result = await _run(session, empty)

        assert result.success, result.error
        assert _rows(warehouse) == []
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("w", "rows"))
        summary = _load(warehouse).current_snapshot().summary
        assert summary.get(SUMMARY_VERSION) == str(latest.version)

    async def test_untouched_rows_keep_their_data_file(self, tmp_path):
        """What sets it apart from overwrite: a reader's cached row groups stay valid."""
        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1, 2], "v": [10, 20]}', "append")
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success
        other = _source(uri, '{"id": [3], "v": [30]}', "append")
        assert (await _run(session, other)).success
        files = {task.file.file_path for task in _load(warehouse).scan().plan_files()}

        changed = self._sync(uri, '{"id": [1, 2, 3], "v": [10, 20, 33]}')
        assert (await _run(session, changed)).success

        after = {task.file.file_path for task in _load(warehouse).scan().plan_files()}
        assert _rows(warehouse) == [(1, 10), (2, 20), (3, 33)]
        assert len(files & after) == 1
        assert len(after - files) == 1

    async def test_the_same_rows_commit_nothing(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success
        before = _load(warehouse).current_snapshot().snapshot_id

        # New source, same rows: the cell runs, and the table has nothing to change.
        same = first + "unrelated = 1\n"
        result = await _run(session, same)

        assert result.success, result.error
        assert not result.cache_hit
        assert _load(warehouse).current_snapshot().snapshot_id == before

    async def test_a_composite_key_deletes_only_its_own_rows(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 1, 2], "day": [1, 2, 1], "v": [1, 2, 3]}', "id,day")
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        second = self._sync(uri, '{"id": [1, 2], "day": [1, 1], "v": [1, 3]}', "id,day")
        result = await _run(session, second)

        assert result.success, result.error
        table = _load(warehouse).scan().to_arrow().sort_by("id")
        assert table["id"].to_pylist() == [1, 2]
        assert table["day"].to_pylist() == [1, 1]

    async def test_a_row_its_changed_neighbours_share_key_values_with_stays(self, tmp_path):
        """(1, 2) and (2, 1) are within the ``In`` filter of the two deleted keys, and stay."""
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(
            uri,
            '{"id": [1, 1, 2, 2], "day": [1, 2, 1, 2], "v": [11, 12, 21, 22], '
            '"tags": [[11], [12], [21], [22]]}',
            "id,day",
        )
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        second = self._sync(
            uri, '{"id": [1, 2], "day": [2, 1], "v": [12, 21], "tags": [[12], [21]]}', "id,day"
        )
        result = await _run(session, second)

        assert result.success, result.error
        table = _load(warehouse).scan().to_arrow().sort_by("v")
        assert table.to_pylist() == [
            {"id": 1, "day": 2, "v": 12, "tags": [12]},
            {"id": 2, "day": 1, "v": 21, "tags": [21]},
        ]

    @pytest.mark.parametrize("mode", ["merge", "sync"])
    async def test_thousands_of_composite_keys_do_not_kill_the_process(self, tmp_path, mode):
        """An exact filter over this many pairs is a SIGILL in Arrow; a test fails by dying."""
        warehouse, uri = _warehouse(tmp_path)
        pairs = '"id": list(range(5000)), "day": list(range(5000))'
        first = _source(uri, f'{{{pairs}, "v": [0] * 5000}}', mode, " on=id,day")
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        second = _source(uri, f'{{{pairs}, "v": [1] * 5000}}', mode, " on=id,day")
        result = await _run(session, second)

        assert result.success, result.error
        assert set(_load(warehouse).scan().to_arrow()["v"].to_pylist()) == {1}

    @pytest.mark.parametrize("mode", ["merge", "sync"])
    @pytest.mark.parametrize(
        ("rows", "message"),
        [
            ('{"id": [5, 5], "v": [1, 2]}', "two rows with one key"),
            ('{"id": [5, None], "v": [1, 2]}', "null in its key columns"),
            ('{"other": [5], "v": [1]}', "no key column 'id'"),
        ],
    )
    async def test_rows_without_a_key_of_their_own_keep_the_table(
        self, tmp_path, mode, rows, message
    ):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1], "v": [10]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        result = await _run(session, _source(uri, rows, mode, " on=id"))

        assert not result.success
        assert message in result.error
        assert _rows(warehouse) == [(1, 10)]

    async def test_the_first_write_refuses_a_repeated_key_too(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        source = self._sync(uri, '{"id": [5, 5], "v": [1, 2]}')

        result = await _run(_session(tmp_path, source), source)

        assert not result.success
        assert "two rows with one key" in result.error
        with pytest.raises(NoSuchTableError):
            _load(warehouse)

    async def test_a_null_key_already_in_the_table_is_refused(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = _source(uri, '{"id": [1, None], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        result = await _run(session, self._sync(uri, '{"id": [1], "v": [10]}'))

        assert not result.success
        assert "holds rows with a null in the key columns" in result.error
        assert len(_rows(warehouse)) == 2

    async def test_a_key_of_a_narrower_type_still_matches_its_rows(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        narrower = self._sync(uri, '{"id": pa.array([2], pa.int32()), "v": [22]}')
        result = await _run(session, narrower)

        assert result.success, result.error
        assert _rows(warehouse) == [(2, 22)]

    async def test_a_table_strata_did_not_write_is_refused_even_when_foreign_writes_are_on(
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
        source = self._sync(uri, '{"id": [1], "v": [10]}')

        result = await _run(_session(tmp_path, source), source)

        assert not result.success
        assert "did not write" in result.error
        assert _rows(warehouse) == [(7, 70)]

    async def test_a_merge_keeps_the_rows_it_does_not_name_and_commits_nothing_new_for_none(
        self, tmp_path
    ):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success

        merge = _source(uri, '{"id": [2, 3], "v": [22, 30]}', "merge", " on=id")
        assert (await _run(session, merge)).success
        assert _rows(warehouse) == [(1, 10), (2, 22), (3, 30)]
        before = _load(warehouse).current_snapshot().snapshot_id

        again = _source(uri, '{"id": [2, 3], "v": [22, 30]}', "merge", " on=id") + "other = 1\n"
        result = await _run(session, again)

        assert result.success, result.error
        assert _load(warehouse).current_snapshot().snapshot_id == before

    async def test_a_revert_puts_the_deleted_rows_back(self, tmp_path):
        warehouse, uri = _warehouse(tmp_path)
        first = self._sync(uri, '{"id": [1, 2], "v": [10, 20]}')
        session = _session(tmp_path, first)
        assert (await _run(session, first)).success
        assert (await _run(session, self._sync(uri, '{"id": [2], "v": [20]}'))).success
        assert _rows(warehouse) == [(2, 20)]

        reverted = await _run(session, first)

        assert reverted.success, reverted.error
        assert _rows(warehouse) == [(1, 10), (2, 20)]


@windows
class TestAPerVariantCell:
    """Every variant writes the same table."""

    @staticmethod
    async def _fanout(tmp_path: Path, mode: str):
        from strata.notebook.writer import set_variant_mode

        warehouse, uri = _warehouse(tmp_path)
        cells = [
            ("a", "# @variant model a\nk = 1\n"),
            ("b", "# @variant model b\nk = 2\n"),
            (
                "ev",
                f"# @per_variant\n# @write_table rows {uri} {mode}\nimport pyarrow as pa\n"
                "rows = pa.table({'id': [k], 'v': [k * 10]})\n",
            ),
        ]
        path = _notebook(tmp_path, cells).path
        set_variant_mode(path, "model", "sweep")
        session = NotebookSession(parse_notebook(path), path)
        session.refresh_environment_runtime()
        executor = CellExecutor(session)
        for cell_id, source in cells[:2]:
            assert (await executor.execute_cell(cell_id, source)).success
        return warehouse, await executor.execute_cell("ev", cells[2][1])

    async def test_an_overwrite_is_refused_before_any_variant_runs(self, tmp_path):
        warehouse, result = await self._fanout(tmp_path, "overwrite")

        assert not result.success
        assert result.error_code == "per_variant_write_table_overwrite"
        assert not (warehouse / "catalog.db").exists()

    async def test_a_sync_is_refused_before_any_variant_runs(self, tmp_path):
        warehouse, result = await self._fanout(tmp_path, "sync on=id")

        assert not result.success
        assert result.error_code == "per_variant_write_table_sync"
        assert not (warehouse / "catalog.db").exists()

    async def test_an_append_keeps_every_variants_rows(self, tmp_path):
        warehouse, result = await self._fanout(tmp_path, "append")

        assert result.success, result.error
        assert _rows(warehouse) == [(1, 10), (2, 20)]


class TestWhoMayWrite:
    """The gates an export passes, and the credential-writers setting, before the cell runs."""

    @pytest.fixture
    def service(self, monkeypatch, tmp_path):
        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.server import ServerState

        def _serve(**overrides):
            config = StrataConfig(
                deployment_mode="service",
                auth_mode="trusted_proxy",
                proxy_token="sekrit",
                artifact_dir=tmp_path / "service-artifacts",
                cache_dir=tmp_path / "cache",
                **overrides,
            )
            monkeypatch.setattr(server_module, "_state", ServerState(config))

        return _serve

    @staticmethod
    async def _run_as(session: NotebookSession, source: str, principal):
        from strata.auth.principal import principal_context

        with principal_context(principal):
            return await _run(session, source)

    @pytest.mark.parametrize(
        ("writes", "scopes", "expected"),
        [
            (False, {"artifacts:write"}, "STRATA_SERVICE_WRITES_ENABLED"),
            (True, set(), "'artifacts:write' scope"),
        ],
    )
    async def test_service_mode_refuses_what_an_export_would(
        self, tmp_path, service, writes, scopes, expected
    ):
        from strata.types import Principal

        service(service_writes_enabled=writes)
        _, uri = _warehouse(tmp_path)
        source = _source(uri, '{"id": [1], "v": [10]}')

        result = await self._run_as(
            _session(tmp_path, source), source, Principal(id="ana", scopes=frozenset(scopes))
        )

        assert not result.success
        assert expected in result.error

    async def test_service_mode_with_both_gates_open_goes_on_to_run(self, tmp_path, service):
        """It stops at the harness user this test server lacks, past the write gates."""
        from strata.types import Principal

        service(service_writes_enabled=True)
        _, uri = _warehouse(tmp_path)
        source = _source(uri, '{"id": [1], "v": [10]}')

        result = await self._run_as(
            _session(tmp_path, source),
            source,
            Principal(id="ana", tenant="t", scopes=frozenset({"artifacts:write"})),
        )

        assert not result.success
        assert "STRATA_NOTEBOOK_HARNESS_USER" in result.error

    @staticmethod
    def _catalog(monkeypatch, tmp_path: Path, writers) -> Path:
        from strata.config import StrataConfig

        warehouse = tmp_path / "warehouse"
        warehouse.mkdir()
        config = StrataConfig(
            catalogs={
                "lake": {
                    "type": "sql",
                    "uri": f"sqlite:///{warehouse / 'catalog.db'}",
                    "warehouse": str(warehouse),
                    "strata_credential": "lab",
                }
            },
            notebook_credentials={"lab": {}},
            notebook_credential_writers=writers,
        )
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        return warehouse

    @windows
    async def test_a_catalog_credential_the_setting_calls_read_only_is_refused(
        self, tmp_path, monkeypatch
    ):
        from strata.types import Principal

        warehouse = self._catalog(monkeypatch, tmp_path, {})
        source = _source("lake:lab.readings", '{"id": [1], "v": [10]}')

        result = await self._run_as(_session(tmp_path, source), source, Principal(id="user_ana"))

        assert not result.success
        assert "credential 'lab' is read-only" in result.error
        assert not (warehouse / "catalog.db").exists()

    @windows
    async def test_a_catalog_credential_writes_for_a_listed_principal(self, tmp_path, monkeypatch):
        from strata.types import Principal

        warehouse = self._catalog(monkeypatch, tmp_path, {"lab": ["user_*"]})
        source = _source("lake:lab.readings", '{"id": [1], "v": [10]}')
        session = _session(tmp_path, source)

        refused = await self._run_as(session, source, Principal(id="agent:x"))
        written = await self._run_as(session, source, Principal(id="user_ana"))

        assert "'lab' is not writable by 'agent:x'" in refused.error
        assert written.success, written.error
        assert _rows(warehouse, "lake") == [(1, 10)]

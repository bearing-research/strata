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

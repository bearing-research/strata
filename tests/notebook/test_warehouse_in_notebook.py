"""A warehouse named from the notebook's directory: ``./warehouse#ns.table``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pyiceberg.catalog.sql import SqlCatalog

from strata.iceberg.catalog import in_notebook
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local warehouse paths fail"
)

WRITES = (
    "# @write_table rows ./wh#lab.rows overwrite\n"
    "import pyarrow as pa\n"
    "rows = pa.table({'id': [1, 2, 3]})\n"
)
READS = "# @table rows ./wh#lab.rows\nuri = rows\nsnapshot = rows_snapshot\n"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("./wh#lab.rows", "/nb/wh#lab.rows"),
        ("./a/b#lab.rows", "/nb/a/b#lab.rows"),
        ("../wh#lab.rows", "/wh#lab.rows"),
        ("/abs/wh#lab.rows", "/abs/wh#lab.rows"),
        ("file:///abs/wh#lab.rows", "file:///abs/wh#lab.rows"),
        ("s3://bucket/wh#lab.rows", "s3://bucket/wh#lab.rows"),
        ("lake:lab.rows", "lake:lab.rows"),
        ("lab.rows", "lab.rows"),
        ("./no-table", "./no-table"),
    ],
)
def test_only_a_relative_warehouse_moves(uri, expected):
    assert in_notebook(uri, Path("/nb")) == expected


def test_without_a_notebook_nothing_moves():
    assert in_notebook("./wh#lab.rows", None) == "./wh#lab.rows"


def _session(path: Path, *sources: str) -> NotebookSession:
    nb = create_notebook(path, "Warehouse")
    after = None
    for cell_id, source in zip("abcd", sources, strict=False):
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


def _catalog(warehouse: Path) -> SqlCatalog:
    return SqlCatalog("strata", uri=f"sqlite:///{warehouse}/catalog.db", warehouse=str(warehouse))


async def test_a_cell_writes_and_another_reads_it_beside_the_notebook(tmp_path, monkeypatch):
    """Wherever the server was started."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    session = _session(tmp_path / "nb", WRITES, READS)

    await _run(session, "a")
    with _catalog(session.path / "wh") as catalog:
        table = catalog.load_table("lab.rows")
    assert table.scan().to_arrow().num_rows == 3
    assert not (elsewhere / "wh").exists()

    result = await _run(session, "b")
    assert result.outputs["uri"]["preview"] == f"{session.path.resolve() / 'wh'}#lab.rows"
    assert result.outputs["snapshot"]["preview"] == table.current_snapshot().snapshot_id


async def test_two_notebooks_with_the_same_source_have_a_warehouse_each(tmp_path):
    one = _session(tmp_path / "one", WRITES)
    other = _session(tmp_path / "other", WRITES)

    await _run(one, "a")
    await _run(other, "a")

    assert (one.path / "wh" / "catalog.db").exists()
    assert (other.path / "wh" / "catalog.db").exists()
    with _catalog(one.path / "wh") as catalog:
        assert str(one.path / "wh") in catalog.load_table("lab.rows").location()


async def test_a_new_write_makes_the_reader_run_again(tmp_path):
    session = _session(tmp_path / "nb", WRITES, READS)
    await _run(session, "a")
    first = await _run(session, "b")
    assert (await _run(session, "b")).cache_hit

    with _catalog(session.path / "wh") as catalog:
        table = catalog.load_table("lab.rows")
        table.append(table.scan().to_arrow())

    again = await _run(session, "b")
    assert not again.cache_hit
    assert again.outputs["snapshot"]["preview"] != first.outputs["snapshot"]["preview"]


async def test_reading_a_warehouse_not_written_yet_fails_and_creates_nothing(tmp_path):
    session = _session(tmp_path / "nb", READS)

    cell = session.notebook_state.get_cell("a")
    result = await CellExecutor(session).execute_cell("a", cell.source)

    assert not result.success
    assert "./wh#lab.rows" in result.error
    assert not (session.path / "wh").exists()


@pytest.fixture
def scan_server(tmp_path):
    """A live personal server, whose directory is not the notebook's."""
    from strata.config import StrataConfig
    from tests.conftest import find_free_port, run_server

    config = StrataConfig(
        host="127.0.0.1",
        port=find_free_port(),
        cache_dir=tmp_path / "scan-cache",
        artifact_dir=tmp_path / "scan-artifacts",
        deployment_mode="personal",
        rate_limit_enabled=False,
    )
    with run_server(config) as base:
        yield base


@pytest.mark.integration
async def test_a_later_cell_reads_what_strata_put_returned(scan_server, tmp_path):
    keeps = (
        "kept = strata.put(inputs=[], transform={'ref': 'kept@v1'}, data={'a': [1, 2, 3]})\n"
        "print('TOKEN', strata._headers['X-Strata-Run-Token'])\n"
    )
    reads = "print('ROWS', kept.to_arrow().num_rows, kept.uri)\n"
    session = _session(tmp_path / "nb", keeps, reads)

    kept = await _run(session, "a")
    read = await _run(session, "b")

    assert kept.success, kept.error
    assert read.success, read.error
    assert "ROWS 3 strata://artifact/" in read.stdout
    # The run's token ends with the run, and a kept handle must not carry it into the store.
    token = kept.stdout.split("TOKEN ", 1)[1].split()[0].encode()
    store = session.get_artifact_manager().artifact_dir
    [blob] = store.rglob("*_var_kept@*")
    assert token not in blob.read_bytes()


@pytest.mark.integration
async def test_a_cell_scans_the_table_through_the_server(scan_server, tmp_path):
    scans = (
        "# @table rows ./wh#lab.rows\n"
        "scan = strata.materialize(inputs=[rows], transform={'executor': 'scan@v1'})\n"
        "count = scan.to_arrow().num_rows\n"
        "del scan\n"
    )
    session = _session(tmp_path / "nb", WRITES, scans)

    await _run(session, "a")
    result = await _run(session, "b")

    assert result.outputs["count"]["preview"] == 3

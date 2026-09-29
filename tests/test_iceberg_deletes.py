"""Iceberg merge-on-read: a scan drops the rows a snapshot's positional deletes name. #536.

pyiceberg rewrites copy-on-write when it deletes, so these tests write a real
positional delete file and attach it to the scan the way pyiceberg's planner
does for a table another engine deleted from. tests/test_lake_catalogs_integration.py
runs the same scan over deletes DuckDB wrote, through a REST catalog, where
pyiceberg does the attaching itself.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyiceberg.manifest import DataFile, DataFileContent, FileFormat

from strata.cache import CachedFetcher
from strata.config import StrataConfig
from strata.iceberg_deletes import DeletedRows, in_row_group
from strata.metadata_cache import DeleteFileEntry, ManifestCache
from strata.metadata_store import MetadataStore
from strata.planner import ReadPlanner, UnsupportedTableFormatError
from strata.types import Filter, FilterOp

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)


@pytest.fixture
def table(tmp_path):
    """``db.t`` with ids 0-9 in one data file of three row groups: 0-3, 4-7, 8-9."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    table = catalog.create_table(
        "db.t",
        schema=pa.schema([("id", pa.int64())]),
        properties={"write.parquet.row-group-limit": "4"},
    )
    table.append(pa.table({"id": pa.array(range(10), pa.int64())}))
    (task,) = table.scan().plan_files()
    catalog.engine.dispose()
    return {"uri": f"{warehouse.as_uri()}#db.t", "data_file": task.file.file_path}


def _delete_file(path, rows: dict[str, list[int]], file_format=FileFormat.PARQUET) -> DataFile:
    """A positional delete file deleting *rows* (data file path -> positions)."""
    pq.write_table(
        pa.table(
            {
                "file_path": [p for p, positions in rows.items() for _ in positions],
                "pos": pa.array([n for positions in rows.values() for n in positions], pa.int64()),
            }
        ),
        path,
    )
    return DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path=path.as_uri(),
        file_format=file_format,
        record_count=sum(len(positions) for positions in rows.values()),
        file_size_in_bytes=path.stat().st_size,
    )


@pytest.fixture
def attach(monkeypatch):
    """Attach delete files to every data file, as the index does for a real table."""
    import strata.planner

    real_plan_files = strata.planner.plan_files

    def attach(*delete_files: DataFile) -> None:
        def plan_files(*args, **kwargs):
            planned = real_plan_files(*args, **kwargs)
            for file in planned:
                file.positional_deletes = set(delete_files)
            return planned

        monkeypatch.setattr(strata.planner, "plan_files", plan_files)

    return attach


def _config(tmp_path) -> StrataConfig:
    return StrataConfig(cache_dir=tmp_path / "cache")


def _scan(config, uri, planner=None, filters=None) -> tuple[list[int], list]:
    plan = (planner or ReadPlanner(config)).plan(uri, filters=filters)
    fetcher = CachedFetcher(config)
    ids = [v for task in plan.tasks for v in fetcher.fetch(task).column("id").to_pylist()]
    return ids, plan.tasks


def test_deleted_rows_are_absent_and_stay_absent_from_the_cache(tmp_path, table, attach):
    # One deleted row in each row group, so each needs its own offset.
    attach(_delete_file(tmp_path / "d.parquet", {table["data_file"]: [1, 5, 8]}))
    config = _config(tmp_path)

    ids, tasks = _scan(config, table["uri"])
    assert ids == [0, 2, 3, 4, 6, 7, 9]
    assert [task.num_rows for task in tasks] == [3, 3, 1]
    assert not any(task.cached for task in tasks)

    # Served from the cache, which holds the row groups without the rows,
    # through both the batch path and the stream-bytes path.
    ids, tasks = _scan(config, table["uri"])
    assert ids == [0, 2, 3, 4, 6, 7, 9]
    assert all(task.cached for task in tasks)
    fetcher = CachedFetcher(config)
    plan = ReadPlanner(config).plan(table["uri"])
    streamed = [
        v
        for task in plan.tasks
        for v in pa.ipc.open_stream(fetcher.fetch_as_stream_bytes(task))
        .read_all()
        .column("id")
        .to_pylist()
    ]
    assert streamed == [0, 2, 3, 4, 6, 7, 9]


def test_a_projected_scan_caching_whole_row_groups_drops_the_rows_too(tmp_path, table, attach):
    """At ``row_group`` granularity a projected miss reads the whole row group."""
    attach(_delete_file(tmp_path / "d.parquet", {table["data_file"]: [1, 5, 8]}))
    config = StrataConfig(cache_dir=tmp_path / "cache", cache_granularity="row_group")

    plan = ReadPlanner(config).plan(table["uri"], columns=["id"])
    fetcher = CachedFetcher(config)
    ids = [v for task in plan.tasks for v in fetcher.fetch(task).column("id").to_pylist()]
    assert ids == [0, 2, 3, 4, 6, 7, 9]


def test_a_row_group_with_every_row_deleted_is_not_read(tmp_path, table, attach):
    attach(_delete_file(tmp_path / "d.parquet", {table["data_file"]: [4, 5, 6, 7]}))
    config = _config(tmp_path)

    plan = ReadPlanner(config).plan(table["uri"])
    assert [task.row_group_id for task in plan.tasks] == [0, 2]
    assert plan.pruned_row_groups == 1
    assert _scan(config, table["uri"])[0] == [0, 1, 2, 3, 8, 9]


def test_offsets_hold_when_an_earlier_row_group_is_pruned(tmp_path, table, attach):
    attach(_delete_file(tmp_path / "d.parquet", {table["data_file"]: [2, 5]}))
    config = _config(tmp_path)

    ids, tasks = _scan(config, table["uri"], filters=[Filter("id", FilterOp.GE, 4)])
    assert [task.row_group_id for task in tasks] == [1, 2]
    assert ids == [4, 6, 7, 8, 9]


def test_every_delete_file_for_a_data_file_applies(tmp_path, table, attach):
    """Deletes from two snapshots, overlapping on row 3, and one naming another file."""
    attach(
        _delete_file(tmp_path / "a.parquet", {table["data_file"]: [0, 3]}),
        _delete_file(
            tmp_path / "b.parquet", {table["data_file"]: [3, 9], "file:///elsewhere.parquet": [1]}
        ),
    )
    assert _scan(_config(tmp_path), table["uri"])[0] == [1, 2, 4, 5, 6, 7, 8]


def test_the_persisted_manifest_keeps_the_deletes(tmp_path, table, attach):
    """After a restart the planner resolves the snapshot from SQLite, not pyiceberg."""
    attach(_delete_file(tmp_path / "d.parquet", {table["data_file"]: [1, 5, 8]}))
    config = _config(tmp_path)
    store = MetadataStore(config.cache_dir / "metadata.sqlite")
    ReadPlanner(config, manifest_cache=ManifestCache(store=store)).plan(table["uri"])

    restarted = ReadPlanner(config, manifest_cache=ManifestCache(store=store))
    ids, tasks = _scan(config, table["uri"], planner=restarted)
    assert ids == [0, 2, 3, 4, 6, 7, 9]
    assert store.manifest_hits == 1


def test_a_metadata_store_from_another_version_is_discarded(tmp_path):
    """0.8.0 persisted manifests without their delete files, and on pyiceberg
    before 0.12 could have missed a table's deletes altogether."""
    import sqlite3

    db = tmp_path / "metadata.sqlite"
    old = sqlite3.connect(db)
    old.execute(
        "CREATE TABLE manifest_cache (catalog_name TEXT, table_identity TEXT,"
        " snapshot_id INTEGER, data_files_json TEXT, created_at TIMESTAMP,"
        " PRIMARY KEY (catalog_name, table_identity, snapshot_id))"
    )
    old.execute(
        "INSERT INTO manifest_cache VALUES ('c', 'db.t', 1,"
        ' \'[{"file_path": "a.parquet", "actual_path": "/w/a.parquet"}]\', NULL)'
    )
    old.commit()
    old.close()

    assert MetadataStore(db).get_manifest("c", "db.t", 1) is None


def test_a_delete_file_strata_cannot_read_is_refused(tmp_path, table, attach):
    attach(_delete_file(tmp_path / "d.orc", {table["data_file"]: [1]}, FileFormat.ORC))

    with pytest.raises(UnsupportedTableFormatError, match="ORC delete file"):
        ReadPlanner(_config(tmp_path)).plan(table["uri"])


def test_a_delete_file_is_read_once(tmp_path):
    from pyiceberg.io.pyarrow import PyArrowFileIO

    path = tmp_path / "d.parquet"
    _delete_file(path, {"file:///a.parquet": [7, 2], "file:///b.parquet": [4]})
    opened = []

    class CountingIO(PyArrowFileIO):
        def new_input(self, location):
            opened.append(location)
            return super().new_input(location)

    rows = DeletedRows()
    delete_files = (DeleteFileEntry(file_path=path.as_uri(), file_format="PARQUET"),)
    io = CountingIO()

    assert rows.for_data_file(io, "file:///a.parquet", delete_files).to_pylist() == [2, 7]
    assert rows.for_data_file(io, "file:///b.parquet", delete_files).to_pylist() == [4]
    assert rows.for_data_file(io, "file:///c.parquet", delete_files) is None
    assert opened == [path.as_uri()]


@pytest.mark.parametrize(
    ("start", "num_rows", "expected"),
    [
        (0, 4, [1]),
        (4, 4, [0, 3]),  # positions 4 and 7, both edges
        (8, 2, None),
        (10, 5, [2]),
    ],
)
def test_in_row_group_shifts_positions_to_the_row_group(start, num_rows, expected):
    positions = pa.array([1, 4, 7, 12], pa.int64())
    result = in_row_group(positions, start, num_rows)
    assert (result.to_pylist() if result is not None else None) == expected

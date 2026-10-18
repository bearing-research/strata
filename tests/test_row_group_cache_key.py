"""A cached row group outlives the snapshot that cached it, until a snapshot reads it otherwise.

Data files never change, so the row-group cache key names no snapshot: an append misses only
its new files. What a snapshot can change about a row group's rows is in the key instead: the
rows it deletes, its schema, and how the table reads files (format version, name mapping).
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.cache.disk import CachedFetcher
from strata.config import StrataConfig
from strata.planner import ReadPlanner
from strata.types import CacheKey, TableIdentity

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)


def _catalog(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    return catalog, f"{warehouse.as_uri()}#db.t"


@pytest.fixture
def table(tmp_path):
    """``db.t`` with ids 0-9 in one data file of three row groups: 0-3, 4-7, 8-9."""
    catalog, uri = _catalog(tmp_path)
    table = catalog.create_table(
        "db.t",
        schema=pa.schema([("id", pa.int64())]),
        properties={"write.parquet.row-group-limit": "4"},
    )
    table.append(pa.table({"id": pa.array(range(10), pa.int64())}))
    (task,) = table.scan().plan_files()
    return {"uri": uri, "data_file": task.file.file_path, "table": table, "catalog": catalog}


def _scan(config, uri, planner=None, snapshot_id=None) -> tuple[list, list]:
    plan = (planner or ReadPlanner(config)).plan(uri, snapshot_id=snapshot_id)
    fetcher = CachedFetcher(config)
    rows = [row for task in plan.tasks for row in fetcher.fetch(task).to_pylist()]
    return rows, plan.tasks


def _ids(rows) -> list[int]:
    return [row["id"] for row in rows]


def _config(tmp_path) -> StrataConfig:
    return StrataConfig(cache_dir=tmp_path / "cache")


def test_an_append_keeps_the_earlier_files_cached(tmp_path, table):
    config = _config(tmp_path)
    _scan(config, table["uri"])

    table["table"].append(pa.table({"id": pa.array([10, 11], pa.int64())}))
    rows, tasks = _scan(config, table["uri"])

    assert sorted(_ids(rows)) == list(range(12))
    cached = {
        (task.file_path.endswith(table["data_file"].rsplit("/", 1)[-1]), task.cached)
        for task in tasks
    }
    assert cached == {(True, True), (False, False)}


def test_a_delete_misses_only_the_row_groups_it_deletes_from(tmp_path, table):
    from tests.iceberg_fixtures import commit_files, positional_delete

    config = _config(tmp_path)
    _scan(config, table["uri"])

    iceberg = table["table"]
    commit_files(iceberg, positional_delete(iceberg, {table["data_file"]: [5]}))
    rows, tasks = _scan(config, table["uri"])

    assert _ids(rows) == [0, 1, 2, 3, 4, 6, 7, 8, 9]
    assert [task.cached for task in tasks] == [True, False, True]


def test_an_older_snapshot_keeps_the_rows_a_later_one_deleted(tmp_path, table):
    from tests.iceberg_fixtures import commit_files, positional_delete

    config = _config(tmp_path)
    iceberg = table["table"]
    before = iceberg.current_snapshot().snapshot_id
    commit_files(iceberg, positional_delete(iceberg, {table["data_file"]: [5]}))
    _scan(config, table["uri"])

    rows, _ = _scan(config, table["uri"], snapshot_id=before)

    assert _ids(rows) == list(range(10))


def test_an_equality_delete_misses_the_row_groups_it_may_meet(tmp_path, table):
    from tests.iceberg_fixtures import commit_files, equality_delete

    config = _config(tmp_path)
    _scan(config, table["uri"])

    iceberg = table["table"]
    commit_files(iceberg, equality_delete(iceberg, pa.table({"id": pa.array([9], pa.int64())})))
    rows, tasks = _scan(config, table["uri"])

    assert _ids(rows) == list(range(9))
    assert tasks[-1].cached is False


def test_a_name_mapping_change_is_not_served_from_the_cache(tmp_path):
    """A file without field ids reads by the name mapping, a property that makes no snapshot."""
    from pyiceberg.schema import Schema
    from pyiceberg.table.name_mapping import MappedField, NameMapping
    from pyiceberg.types import LongType, NestedField

    catalog, uri = _catalog(tmp_path)
    table = catalog.create_table(
        "db.t",
        schema=Schema(
            NestedField(1, "a", LongType(), required=False),
            NestedField(2, "b", LongType(), required=False),
        ),
    )
    path = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": pa.array([1], pa.int64()), "b": pa.array([2], pa.int64())}), path)
    table.add_files([path.as_uri()])
    config = _config(tmp_path)
    planner = ReadPlanner(config)
    assert _scan(config, uri, planner)[0] == [{"a": 1, "b": 2}]

    swapped = NameMapping(
        [MappedField(field_id=1, names=["b"]), MappedField(field_id=2, names=["a"])]
    )
    with table.transaction() as transaction:
        transaction.set_properties({"schema.name-mapping.default": swapped.model_dump_json()})

    assert _scan(config, uri, planner)[0] == [{"a": 2, "b": 1}]


def test_a_format_version_upgrade_gives_new_keys(tmp_path):
    catalog, uri = _catalog(tmp_path)
    table = catalog.create_table(
        "db.t", schema=pa.schema([("id", pa.int64())]), properties={"format-version": "1"}
    )
    table.append(pa.table({"id": pa.array(range(3), pa.int64())}))
    config = _config(tmp_path)
    _, before = _scan(config, uri)

    with table.transaction() as transaction:
        transaction.upgrade_table_version(2)
    _, after = _scan(config, uri)

    assert before and after
    assert not {t.cache_key.to_hex() for t in before} & {t.cache_key.to_hex() for t in after}


def test_the_key_carries_the_manifests_file_size(tmp_path, table):
    _, tasks = _scan(_config(tmp_path), table["uri"])

    (size,) = {task.cache_key.file_size for task in tasks}
    assert size == (tmp_path / table["data_file"].split(str(tmp_path) + "/", 1)[1]).stat().st_size


def test_a_path_written_again_with_other_bytes_is_another_key():
    def key(file_size: int) -> CacheKey:
        return CacheKey(
            tenant_id="_default",
            table_identity=TableIdentity.from_table_id("db.t"),
            file_path="/data/f.parquet",
            file_size=file_size,
            row_group_id=0,
            projection_fingerprint="*",
        )

    assert key(1024).to_hex() != key(2048).to_hex()

"""A cached row group outlives the snapshot that cached it, until a snapshot reads it otherwise.

Data files never change, so the row-group cache key names no snapshot: an append misses only
its new files. What a snapshot can change about a row group's rows is in the key instead: the
rows it deletes, its schema, and how the table reads files (format version, name mapping).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext

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


@pytest.fixture
def lake(tmp_path):
    """A local catalog with an empty namespace ``db``, and the uri of its table ``db.t``."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    yield catalog, f"{warehouse.as_uri()}#db.t"
    catalog.close()


@pytest.fixture
def table(lake):
    """``db.t`` with ids 0-9 in one data file of three row groups: 0-3, 4-7, 8-9."""
    catalog, uri = lake
    table = catalog.create_table(
        "db.t",
        schema=pa.schema([("id", pa.int64())]),
        properties={"write.parquet.row-group-limit": "4"},
    )
    table.append(pa.table({"id": pa.array(range(10), pa.int64())}))
    (task,) = table.scan().plan_files()
    return {"uri": uri, "data_file": task.file.file_path, "table": table, "catalog": catalog}


def _scan(config, uri, planner=None, snapshot_id=None) -> tuple[list, list]:
    with nullcontext(planner) if planner else ReadPlanner(config) as planner:
        plan = planner.plan(uri, snapshot_id=snapshot_id)
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


def test_a_name_mapping_change_is_not_served_from_the_cache(tmp_path, lake):
    """A file without field ids reads by the name mapping, a property that makes no snapshot."""
    from pyiceberg.schema import Schema
    from pyiceberg.table.name_mapping import MappedField, NameMapping
    from pyiceberg.types import LongType, NestedField

    catalog, uri = lake
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
    with ReadPlanner(config) as planner:
        assert _scan(config, uri, planner)[0] == [{"a": 1, "b": 2}]

        swapped = NameMapping(
            [MappedField(field_id=1, names=["b"]), MappedField(field_id=2, names=["a"])]
        )
        with table.transaction() as transaction:
            transaction.set_properties({"schema.name-mapping.default": swapped.model_dump_json()})

        assert _scan(config, uri, planner)[0] == [{"a": 2, "b": 1}]


def test_a_format_version_upgrade_gives_new_keys(tmp_path, lake):
    catalog, uri = lake
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


def test_a_recreated_table_reading_a_rewritten_path_is_not_served_the_old_rows(tmp_path, lake):
    """Dropped, recreated, and given the same path with other bytes of the same size."""
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    catalog, uri = lake
    path = tmp_path / "part-00000.parquet"
    config = _config(tmp_path)

    def register(ids: list[int]) -> None:
        pq.write_table(pa.table({"id": pa.array(ids, pa.int64())}), path, compression="none")
        if catalog.table_exists("db.t"):
            catalog.drop_table("db.t")
        schema = Schema(NestedField(1, "id", LongType(), required=False))
        catalog.create_table("db.t", schema=schema).add_files([path.as_uri()])

    register([1, 2, 3])
    size = path.stat().st_size
    assert _ids(_scan(config, uri)[0]) == [1, 2, 3]

    register([7, 8, 9])
    assert path.stat().st_size == size

    assert _ids(_scan(config, uri)[0]) == [7, 8, 9]


class TestAPathRegisteredAgainInTheSameTable:
    """A Hive-layout file rewritten in place at the same size and registered again.

    Same table, same path, same size: only the manifest entry is new, so every cache
    keyed on the file names the snapshot that added it. Finding 16
    (formal/test_cache_soundness_properties.py).
    """

    def _register(self, catalog, path, columns: dict[str, list[int]], old=None):
        from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
        from pyiceberg.typedef import Record

        table = catalog.load_table("db.t")
        fields = [
            pa.field(name, pa.int64(), metadata={b"PARQUET:field_id": str(field_id).encode()})
            for field_id, name in enumerate(columns, start=1)
        ]
        arrays = [pa.array(values, pa.int64()) for values in columns.values()]
        pq.write_table(
            pa.Table.from_arrays(arrays, schema=pa.schema(fields)),
            path,
            compression="none",
            use_dictionary=False,
        )
        new = DataFile.from_args(
            content=DataFileContent.DATA,
            file_path=path.as_uri(),
            file_format=FileFormat.PARQUET,
            partition=Record(),
            record_count=len(arrays[0]),
            file_size_in_bytes=path.stat().st_size,
        )
        new.spec_id = table.metadata.default_spec_id
        with table.transaction() as tx:
            with tx.update_snapshot().overwrite() as overwrite:
                if old is not None:
                    overwrite.delete_data_file(old)
                overwrite.append_data_file(new)
        return new

    def _table(self, catalog, *names):
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType, NestedField

        fields = [
            NestedField(field_id, name, LongType(), required=False)
            for field_id, name in enumerate(names, start=1)
        ]
        catalog.create_table("db.t", schema=Schema(*fields))

    def _scan(self, planner, fetcher, uri, filters=None):
        plan = planner.plan(uri, filters=filters)
        return [row for task in plan.tasks for row in fetcher.fetch(task).to_pylist()]

    def test_its_new_rows_are_read_and_its_new_stats_prune(self, tmp_path, lake):
        from strata.filters import Filter, FilterOp

        catalog, uri = lake
        config = _config(tmp_path)
        path = tmp_path / "part-00000.parquet"
        self._table(catalog, "id")
        with ReadPlanner(config) as planner:
            fetcher = CachedFetcher(config)
            old = self._register(catalog, path, {"id": [1, 2, 3]})
            size = path.stat().st_size
            assert _ids(self._scan(planner, fetcher, uri)) == [1, 2, 3]

            self._register(catalog, path, {"id": [7, 8, 9]}, old=old)
            assert path.stat().st_size == size

            assert _ids(self._scan(planner, fetcher, uri)) == [7, 8, 9]
            # The old footer's stats (1-3) would prune the row group holding 8.
            matching = self._scan(planner, fetcher, uri, [Filter("id", FilterOp.EQ, 8)])
            assert 8 in _ids(matching)

    def test_it_is_read_with_its_own_footer(self, tmp_path, lake):
        """The column was renamed and the file written again under the new name."""
        catalog, uri = lake
        config = _config(tmp_path)
        path = tmp_path / "part-00000.parquet"
        self._table(catalog, "id", "v")
        with ReadPlanner(config) as planner:
            fetcher = CachedFetcher(config)
            old = self._register(catalog, path, {"id": [1], "v": [10]})
            size = path.stat().st_size
            assert self._scan(planner, fetcher, uri) == [{"id": 1, "v": 10}]

            with catalog.load_table("db.t").update_schema() as update:
                update.rename_column("v", "x")
            self._register(catalog, path, {"id": [2], "x": [20]}, old=old)
            assert path.stat().st_size == size

            assert self._scan(planner, fetcher, uri) == [{"id": 2, "x": 20}]

    def test_it_reads_the_partition_value_it_was_registered_under(self, tmp_path, lake):
        """The same bytes, registered again under another value of the column they omit."""
        from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
        from pyiceberg.partitioning import PartitionField, PartitionSpec
        from pyiceberg.schema import Schema
        from pyiceberg.transforms import IdentityTransform
        from pyiceberg.typedef import Record
        from pyiceberg.types import LongType, NestedField, StringType

        catalog, uri = lake
        config = _config(tmp_path)
        schema = Schema(
            NestedField(1, "id", LongType(), required=False),
            NestedField(2, "region", StringType(), required=False),
        )
        spec = PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "region"))
        catalog.create_table("db.t", schema=schema, partition_spec=spec)
        path = tmp_path / "part-00000.parquet"
        field = pa.field("id", pa.int64(), metadata={b"PARQUET:field_id": b"1"})
        pq.write_table(pa.table({"id": pa.array([1], pa.int64())}).cast(pa.schema([field])), path)

        def register(region: str, old=None):
            table = catalog.load_table("db.t")
            new = DataFile.from_args(
                content=DataFileContent.DATA,
                file_path=path.as_uri(),
                file_format=FileFormat.PARQUET,
                partition=Record(region),
                record_count=1,
                file_size_in_bytes=path.stat().st_size,
            )
            new.spec_id = table.metadata.default_spec_id
            with table.transaction() as tx:
                with tx.update_snapshot().overwrite() as overwrite:
                    if old is not None:
                        overwrite.delete_data_file(old)
                    overwrite.append_data_file(new)
            return new

        with ReadPlanner(config) as planner:
            fetcher = CachedFetcher(config)
            old = register("eu")
            assert self._scan(planner, fetcher, uri) == [{"id": 1, "region": "eu"}]
            register("us", old=old)
            assert self._scan(planner, fetcher, uri) == [{"id": 1, "region": "us"}]

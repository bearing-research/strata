"""Iceberg equality deletes: a scan drops every older row whose key a delete names.

pyiceberg refuses to plan these tables, and cannot write them either, so the
snapshots here are built by tests/iceberg_fixtures.py the way Flink's upsert
sink commits them.
"""

from __future__ import annotations

import datetime
import sys
from decimal import Decimal

import pyarrow as pa
import pytest
from pyiceberg.conversions import to_bytes
from pyiceberg.manifest import FileFormat
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import LongType, NestedField, StringType

from strata.cache import CachedFetcher
from strata.config import StrataConfig
from strata.fast_io import IncrementalIpcMerger
from strata.metadata_cache import ManifestCache
from strata.metadata_store import MetadataStore
from strata.planner import ReadPlanner, UnsupportedTableFormatError
from strata.types import Filter, FilterOp
from tests.iceberg_fixtures import commit_files, data_file, equality_delete

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)

PEOPLE = pa.schema([("id", pa.int64()), ("name", pa.string())])


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
def people(tmp_path):
    """``db.t``: ids 1-4 and a null id, three rows to a row group, at sequence 1."""
    catalog, uri = _catalog(tmp_path)
    catalog.create_table(
        "db.t", schema=PEOPLE, properties={"write.parquet.row-group-limit": "3"}
    ).append(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4, None], pa.int64()),
                "name": ["ann", "bo", "cy", "dee", "nul"],
            },
            schema=PEOPLE,
        )
    )
    return catalog, uri, StrataConfig(cache_dir=tmp_path / "cache")


def _table(catalog):
    return catalog.load_table("db.t")


def _delete(catalog, **columns) -> None:
    """Commit one equality delete keyed on *columns* (name -> values)."""
    table = _table(catalog)
    rows = pa.table(columns)
    commit_files(table, equality_delete(table, rows))


def _scan(config, uri, columns=None, planner=None, filters=None):
    plan = (planner or ReadPlanner(config)).plan(uri, columns=columns, filters=filters)
    fetcher = CachedFetcher(config)
    batches = [fetcher.fetch(task) for task in plan.tasks]
    return pa.Table.from_batches(batches, schema=plan.schema), plan


def _names(table: pa.Table) -> list[str]:
    return sorted(table.column("name").to_pylist())


def test_rows_with_a_deleted_key_are_gone_and_stay_gone_from_the_cache(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([2, 4], pa.int64()))

    table, plan = _scan(config, uri)
    assert _names(table) == ["ann", "cy", "nul"]
    assert not any(task.cached for task in plan.tasks)

    table, plan = _scan(config, uri)
    assert _names(table) == ["ann", "cy", "nul"]
    assert all(task.cached for task in plan.tasks)
    fetcher = CachedFetcher(config)
    merger = IncrementalIpcMerger()
    streamed = b"".join(merger.feed(fetcher.fetch_as_stream_bytes(task)) for task in plan.tasks)
    assert _names(pa.ipc.open_stream(streamed + merger.finish()).read_all()) == ["ann", "cy", "nul"]


def test_a_filtered_scan_still_applies_a_delete_whose_keys_miss_the_filter(people):
    """The delete's key bounds (2..2) miss `id >= 3`, so pyiceberg's planner
    prunes it. That is right for a reader that filters rows; Strata returns
    whole row groups, and the first one still holds id 2. The filtered scan
    used to return the deleted row and cache it under a key with no filter,
    where the next unfiltered scan found it."""
    catalog, uri, config = people
    table = _table(catalog)
    commit_files(
        table,
        equality_delete(
            table,
            pa.table({"id": pa.array([2], pa.int64())}),
            lower_bounds={1: to_bytes(LongType(), 2)},
            upper_bounds={1: to_bytes(LongType(), 2)},
            null_value_counts={1: 0},
        ),
    )

    filtered, _ = _scan(config, uri, filters=[Filter("id", FilterOp.GE, 3)])
    assert "bo" not in filtered.column("name").to_pylist()

    unfiltered, plan = _scan(config, uri)
    assert all(task.cached for task in plan.tasks)  # the filtered scan filled the cache
    assert _names(unfiltered) == ["ann", "cy", "dee", "nul"]


@pytest.mark.parametrize(
    ("key_type", "values"),
    [
        (pa.date32(), [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), None]),
        (pa.decimal128(10, 2), [Decimal("1.50"), Decimal("2.50"), None]),
    ],
    ids=["date", "decimal"],
)
def test_a_null_key_of_any_type_deletes_the_null_rows(tmp_path, key_type, values):
    """Matching nulls used to fill them with `pa.array([0]).cast(type)`, which
    date and decimal keys cannot take."""
    catalog, uri = _catalog(tmp_path)
    schema = pa.schema([("k", key_type), ("name", pa.string())])
    catalog.create_table("db.t", schema=schema).append(
        pa.table({"k": pa.array(values, key_type), "name": ["a", "b", "nul"]}, schema=schema)
    )
    _delete(catalog, k=pa.array([None, values[0]], key_type))

    table, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert _names(table) == ["b"]


def test_extension_keys_and_keys_null_on_both_sides_match_null_safely():
    import uuid

    from strata.iceberg_equality import _matching_rows

    a, b = uuid.uuid4().bytes, uuid.uuid4().bytes
    data = pa.table(
        {"1": pa.ExtensionArray.from_storage(pa.uuid(), pa.array([a, b, None], pa.binary(16)))}
    )
    deletes = pa.table(
        {"1": pa.ExtensionArray.from_storage(pa.uuid(), pa.array([b, None], pa.binary(16)))}
    )
    assert _matching_rows(data, deletes).to_pylist() == [False, True, True]

    # A key null on every row of both sides matches on its is-null flag alone.
    all_null = pa.table({"1": pa.nulls(2, pa.date32())})
    assert _matching_rows(all_null, pa.table({"1": pa.nulls(1, pa.date32())})).to_pylist() == [
        True,
        True,
    ]


def test_a_delete_leaves_the_row_its_own_upsert_wrote(people):
    """Delete and new row in one snapshot: the delete removes only older rows."""
    catalog, uri, config = people
    table = _table(catalog)
    commit_files(
        table,
        equality_delete(table, pa.table({"id": pa.array([2], pa.int64())})),
        data_file(table, pa.table({"id": pa.array([2], pa.int64()), "name": ["bo v2"]})),
    )

    assert _names(_scan(config, uri)[0]) == ["ann", "bo v2", "cy", "dee", "nul"]


def test_a_later_row_with_a_deleted_key_is_kept(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([3], pa.int64()))
    _table(catalog).append(
        pa.table({"id": pa.array([3], pa.int64()), "name": ["cy again"]}, schema=PEOPLE)
    )

    assert _names(_scan(config, uri)[0]) == ["ann", "bo", "cy again", "dee", "nul"]


def test_a_null_key_deletes_null_rows_and_only_those(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([None, 1], pa.int64()))

    assert _names(_scan(config, uri)[0]) == ["bo", "cy", "dee"]


def test_a_key_of_several_columns_must_match_on_all_of_them(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([1, 2], pa.int64()), name=["ann", "not bo"])

    assert _names(_scan(config, uri)[0]) == ["bo", "cy", "dee", "nul"]


def test_a_projection_without_the_key_still_drops_the_rows(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([2, 4], pa.int64()))

    table, _ = _scan(config, uri, columns=["name"])
    assert table.column_names == ["name"]
    assert _names(table) == ["ann", "cy", "nul"]


def test_deletes_from_several_snapshots_all_apply(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([1], pa.int64()))
    _delete(catalog, name=["dee"])

    assert _names(_scan(config, uri)[0]) == ["bo", "cy", "nul"]


def test_a_key_column_dropped_since_still_matches(people):
    """The data file still holds the column; the delete names it by field id."""
    catalog, uri, config = people
    table = _table(catalog)
    name_id = str(table.schema().find_field("name").field_id)
    commit_files(table, equality_delete(table, pa.table({name_id: ["bo"]})))
    with _table(catalog).update_schema() as update:
        update.delete_column("name")

    table, _ = _scan(config, uri)
    assert table.column_names == ["id"]
    assert sorted(table.column("id").to_pylist(), key=lambda v: (v is None, v)) == [1, 3, 4, None]


def test_a_key_column_the_file_predates_is_null_in_it(people):
    catalog, uri, config = people
    with _table(catalog).update_schema() as update:
        update.add_column("team", StringType())
    _delete(catalog, team=pa.array(["red"], pa.string()))
    assert len(_names(_scan(config, uri)[0])) == 5  # older rows have no team

    _delete(catalog, team=pa.array([None], pa.string()))
    assert _names(_scan(config, uri)[0]) == []


def test_a_promoted_key_matches_after_widening(tmp_path):
    catalog, uri = _catalog(tmp_path)
    catalog.create_table("db.t", schema=pa.schema([("id", pa.int32()), ("name", pa.string())]))
    _table(catalog).append(pa.table({"id": pa.array([1, 2], pa.int32()), "name": ["ann", "bo"]}))
    with _table(catalog).update_schema() as update:
        update.update_column("id", LongType())
    _delete(catalog, id=pa.array([2], pa.int64()))

    table, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert _names(table) == ["ann"]


NANOS = 1_704_153_600_000_000_123  # 2024-01-02 plus 123 ns


@pytest.mark.parametrize(
    "deleted",
    [
        pa.array([datetime.datetime(2024, 1, 2)], pa.timestamp("us")),
        pa.array([NANOS + 333], pa.timestamp("ns")),
    ],
    ids=["microseconds", "other nanoseconds"],
)
def test_a_nanosecond_key_is_compared_at_the_tables_unit(tmp_path, deleted):
    """A v2 table's timestamps are microseconds, but a file registered from
    elsewhere can hold nanoseconds, which the scan truncates. A JVM writer
    deletes a row by the value it read, microseconds; the key was compared at
    nanoseconds instead, so the delete never matched."""
    catalog, uri = _catalog(tmp_path)
    schema = pa.schema([("ts", pa.timestamp("us")), ("name", pa.string())])
    catalog.create_table("db.t", schema=schema).append(
        pa.table({"ts": [datetime.datetime(2024, 1, 1)], "name": ["a"]}, schema=schema)
    )
    table = _table(catalog)
    commit_files(
        table,
        data_file(table, pa.table({"ts": pa.array([NANOS], pa.timestamp("ns")), "name": ["b"]})),
    )
    _delete(catalog, ts=deleted)

    result, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert _names(result) == ["a"]


def test_keys_of_one_type_in_another_arrow_form_match():
    """A uuid key read as Arrow's uuid type on one side and its 16-byte storage
    on the other, or a string dictionary-encoded on one side, still match."""
    import uuid

    from strata.iceberg_equality import deleted_mask
    from strata.metadata_cache import EqualityDeleteEntry

    a, b = uuid.uuid4().bytes, uuid.uuid4().bytes
    delete = EqualityDeleteEntry(file_path="d", actual_path="d", equality_ids=(1,), record_count=1)
    stored = pa.ExtensionArray.from_storage(pa.uuid(), pa.array([a, b], pa.binary(16)))
    table = pa.table({"k": stored})
    assert deleted_mask(
        table, {1: "k"}, [delete], lambda _: pa.table({"1": pa.array([b], pa.binary(16))})
    ).to_pylist() == [False, True]

    table = pa.table({"k": pa.array(["x", "y"]).dictionary_encode()})
    assert deleted_mask(
        table, {1: "k"}, [delete], lambda _: pa.table({"1": pa.array(["x"])})
    ).to_pylist() == [True, False]


def test_positional_and_equality_deletes_on_one_file(people, tmp_path):
    import pyarrow.parquet as pq
    from pyiceberg.manifest import DataFile, DataFileContent

    import strata.planner

    catalog, uri, config = people
    _delete(catalog, id=pa.array([4], pa.int64()))
    (planned,) = strata.planner.plan_files(
        _table(catalog), _table(catalog).current_snapshot().snapshot_id
    )
    positions = tmp_path / "pos.parquet"
    pq.write_table(
        pa.table({"file_path": [planned.data_file.file_path], "pos": pa.array([0], pa.int64())}),
        positions,
    )
    positional = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path=positions.as_uri(),
        file_format=FileFormat.PARQUET,
        record_count=1,
        file_size_in_bytes=positions.stat().st_size,
    )
    real_plan_files = strata.planner.plan_files

    def plan_files(*args, **kwargs):
        planned = real_plan_files(*args, **kwargs)
        for file in planned:
            file.positional_deletes = {positional}
        return planned

    strata.planner.plan_files = plan_files
    try:
        assert _names(_scan(config, uri)[0]) == ["bo", "cy", "nul"]
    finally:
        strata.planner.plan_files = real_plan_files


def test_a_delete_in_one_partition_leaves_the_others(tmp_path):
    catalog, uri = _catalog(tmp_path)
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "region", StringType(), required=False),
    )
    spec = PartitionSpec(PartitionField(2, 1000, IdentityTransform(), "region"))
    catalog.create_table("db.t", schema=schema, partition_spec=spec)
    _table(catalog).append(
        pa.table(
            {"id": pa.array([1, 1], pa.int64()), "region": ["eu", "us"]},
            schema=pa.schema([("id", pa.int64()), ("region", pa.string())]),
        )
    )
    table = _table(catalog)
    commit_files(
        table,
        equality_delete(table, pa.table({"id": pa.array([1], pa.int64())}), partition=Record("eu")),
    )

    result, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert result.column("region").to_pylist() == ["us"]


def test_a_delete_whose_key_range_misses_a_row_group_is_not_applied_to_it(people):
    """ids 1-3 and 4-null sit in two row groups; a delete of ids 100-200 meets neither."""
    catalog, uri, config = people
    table = _table(catalog)
    commit_files(
        table,
        equality_delete(
            table,
            pa.table({"id": pa.array([100, 200], pa.int64())}),
            lower_bounds={1: to_bytes(LongType(), 100)},
            upper_bounds={1: to_bytes(LongType(), 200)},
            null_value_counts={1: 0},
        ),
        equality_delete(
            table,
            pa.table({"id": pa.array([2], pa.int64())}),
            lower_bounds={1: to_bytes(LongType(), 2)},
            upper_bounds={1: to_bytes(LongType(), 2)},
            null_value_counts={1: 0},
        ),
    )

    table, plan = _scan(config, uri)
    assert _names(table) == ["ann", "cy", "dee", "nul"]
    assert [len(task.equality_deletes) for task in plan.tasks] == [1, 0]


def test_parsed_delete_files_are_held_to_the_row_limit(people, monkeypatch):
    """The fetcher keeps delete keys parsed, but no more rows than the limit."""
    from strata import lake_files
    from strata.iceberg_equality import EqualityDeleteSets

    catalog, uri, config = people
    _delete(catalog, id=pa.array([1, 2], pa.int64()))
    _delete(catalog, id=pa.array([3, 4], pa.int64()))
    (first, second) = sorted(
        ReadPlanner(config).plan(uri).tasks[0].equality_deletes, key=lambda d: d.file_path
    )
    opened = []
    real_open = lake_files.open_parquet
    monkeypatch.setattr(
        lake_files,
        "open_parquet",
        lambda path, *a, **k: opened.append(path) or real_open(path, *a, **k),
    )

    sets = EqualityDeleteSets(max_rows=3)
    sets.keys(first)
    sets.keys(first)
    sets.keys(second)  # 4 rows held: the first file goes
    sets.keys(first)
    assert opened == [first.actual_path, second.actual_path, first.actual_path]


def test_a_row_group_over_the_equality_delete_limit_is_refused(people, tmp_path):
    catalog, uri, _ = people
    _delete(catalog, id=pa.array([2, 4], pa.int64()))
    config = StrataConfig(cache_dir=tmp_path / "limit-cache", max_equality_delete_rows=1)

    with pytest.raises(UnsupportedTableFormatError, match="2 pending equality deletes") as e:
        ReadPlanner(config).plan(uri)
    assert "rewrite_data_files" in str(e.value)
    assert "max_equality_delete_rows" in str(e.value)


def test_a_delete_file_strata_cannot_read_is_refused(people):
    catalog, uri, config = people
    table = _table(catalog)
    commit_files(
        table,
        equality_delete(
            table, pa.table({"id": pa.array([1], pa.int64())}), file_format=FileFormat.AVRO
        ),
    )

    with pytest.raises(UnsupportedTableFormatError, match="AVRO equality delete file"):
        ReadPlanner(config).plan(uri)


def test_the_persisted_manifest_keeps_the_equality_deletes(people):
    catalog, uri, config = people
    _delete(catalog, id=pa.array([2], pa.int64()))
    store = MetadataStore(config.cache_dir / "metadata.sqlite")
    ReadPlanner(config, manifest_cache=ManifestCache(store=store)).plan(uri)

    restarted = ReadPlanner(config, manifest_cache=ManifestCache(store=store))
    table, _ = _scan(config, uri, planner=restarted)
    assert _names(table) == ["ann", "cy", "dee", "nul"]
    assert store.manifest_hits == 1


def test_a_scan_artifact_counts_the_rows_left_and_a_refusal_says_why(people, tmp_path):
    """Through a server: the build's row count is the rows sent (it fails its
    own check otherwise), and a refused table is a 422 with the reason."""
    import httpx
    from strata_client.client import StrataClient

    from tests.conftest import run_server_with_context

    catalog, uri, _ = people
    _delete(catalog, id=pa.array([2, 4], pa.int64()))
    scan = {"executor": "scan@v1", "params": {}}
    with run_server_with_context(tmp_path / "server-cache", tmp_path / "artifacts") as ctx:
        client = StrataClient(base_url=ctx.base_url)
        try:
            artifact = client.materialize(inputs=[uri], transform=scan)
            assert _names(client.fetch(artifact.uri)) == ["ann", "cy", "nul"]
        finally:
            client.close()

    with run_server_with_context(
        tmp_path / "limit-cache", tmp_path / "limit-artifacts", max_equality_delete_rows=1
    ) as ctx:
        response = httpx.post(
            f"{ctx.base_url}/v1/materialize",
            json={"inputs": [uri], "transform": scan},
            timeout=60,
        )
        assert response.status_code == 422
        assert "pending equality deletes" in response.json()["detail"]


def test_a_key_the_file_predates_matches_on_its_v3_initial_default():
    """A data file written before its key column existed holds that column's
    initial-default, not nulls, so a delete of the default removes its rows."""
    from strata.iceberg_equality import deleted_mask
    from strata.metadata_cache import EqualityDeleteEntry

    table = pa.table({"name": ["a", "b"]})
    delete = EqualityDeleteEntry(file_path="d", actual_path="d", equality_ids=(3,), record_count=1)

    def keys(_):
        return pa.table({"3": ["red"]})

    assert deleted_mask(table, {3: None}, [delete], keys).to_pylist() == [False, False]
    assert deleted_mask(table, {3: None}, [delete], keys, defaults={3: "red"}).to_pylist() == [
        True,
        True,
    ]

"""Iceberg schema evolution: a scan reads older data files as the table's schema.

Iceberg names a column by field id and a schema change rewrites no files, so reading by name serves
wrong data after a drop, re-add or rename. Each test checks against pyiceberg's own read.
"""

from __future__ import annotations

import datetime
import sys

import pyarrow as pa
import pytest
from pyiceberg.types import (
    DoubleType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
)

from strata.cache import CachedFetcher
from strata.config import StrataConfig
from strata.fast_io import IncrementalIpcMerger
from strata.planner import ReadPlanner
from strata.services.materialize import materialize_service
from strata.types import Filter, FilterOp

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)


@pytest.fixture
def lake(tmp_path):
    """A local catalog and ``db.t`` with ids 1, 2 and x 10, 20 in one data file."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    catalog.create_table("db.t", schema=pa.schema([("id", pa.int64()), ("x", pa.int64())]))
    catalog.load_table("db.t").append(
        pa.table({"id": pa.array([1, 2], pa.int64()), "x": pa.array([10, 20], pa.int64())})
    )
    return catalog, f"{warehouse.as_uri()}#db.t", StrataConfig(cache_dir=tmp_path / "cache")


def _evolve(catalog, change) -> None:
    with catalog.load_table("db.t").update_schema() as update:
        change(update)


def _scan(config, uri, columns=None, snapshot_id=None, filters=None, planner=None):
    plan = (planner or ReadPlanner(config)).plan(
        uri, columns=columns, snapshot_id=snapshot_id, filters=filters
    )
    fetcher = CachedFetcher(config)
    batches = [fetcher.fetch(task) for task in plan.tasks]
    return pa.Table.from_batches(batches, schema=plan.schema), plan


def _rows(table: pa.Table) -> list[dict]:
    return sorted(table.to_pylist(), key=lambda row: row["id"])


def _pyiceberg(catalog, columns=None, snapshot_id=None) -> list[dict]:
    scan = catalog.load_table("db.t").scan(
        selected_fields=tuple(columns) if columns else ("*",), snapshot_id=snapshot_id
    )
    return _rows(scan.to_arrow())


def test_a_column_dropped_and_added_again_reads_as_nulls(lake):
    """The new ``x`` is a different column; the old file has none of it."""
    catalog, uri, config = lake
    _evolve(catalog, lambda u: u.delete_column("x"))
    _evolve(catalog, lambda u: u.add_column("x", LongType()))

    expected = [{"id": 1, "x": None}, {"id": 2, "x": None}]
    assert _pyiceberg(catalog) == expected
    assert _rows(_scan(config, uri)[0]) == expected
    assert _rows(_scan(config, uri, columns=["id", "x"])[0]) == expected


def test_a_renamed_column_reads_under_its_new_name(lake):
    catalog, uri, config = lake
    _evolve(catalog, lambda u: u.rename_column("x", "y"))

    expected = [{"id": 1, "y": 10}, {"id": 2, "y": 20}]
    assert _pyiceberg(catalog) == expected
    assert _rows(_scan(config, uri)[0]) == expected
    assert _rows(_scan(config, uri, columns=["id", "y"])[0]) == expected


def test_a_new_column_taking_a_renamed_ones_name_is_not_its_data(lake):
    catalog, uri, config = lake
    _evolve(catalog, lambda u: u.rename_column("x", "y"))
    _evolve(catalog, lambda u: u.add_column("x", LongType()))

    expected = [{"id": 1, "y": 10, "x": None}, {"id": 2, "y": 20, "x": None}]
    assert _pyiceberg(catalog) == expected
    assert _rows(_scan(config, uri)[0]) == expected


def test_an_added_column_reads_as_nulls_in_older_files(lake):
    catalog, uri, config = lake
    _evolve(catalog, lambda u: u.add_column("label", StringType()))
    catalog.load_table("db.t").append(
        pa.table(
            {
                "id": pa.array([3], pa.int64()),
                "x": pa.array([30], pa.int64()),
                "label": pa.array(["c"]),
            }
        )
    )

    expected = [
        {"id": 1, "x": 10, "label": None},
        {"id": 2, "x": 20, "label": None},
        {"id": 3, "x": 30, "label": "c"},
    ]
    assert _pyiceberg(catalog) == expected
    table, plan = _scan(config, uri)
    assert _rows(table) == expected
    assert _rows(_scan(config, uri, columns=["id", "label"])[0]) == [
        {k: row[k] for k in ("id", "label")} for row in expected
    ]
    # Every row group's stream carries one schema, or the merge refuses it.
    fetcher = CachedFetcher(config)
    merger = IncrementalIpcMerger()
    streamed = b"".join(merger.feed(fetcher.fetch_as_stream_bytes(task)) for task in plan.tasks)
    streamed += merger.finish()
    assert _rows(pa.ipc.open_stream(streamed).read_all()) == expected


def test_a_projected_scan_caching_whole_row_groups_reads_the_evolved_file(lake, tmp_path):
    """At ``row_group`` granularity a projected miss reads the whole row group."""
    catalog, uri, _ = lake
    _evolve(catalog, lambda u: u.rename_column("x", "y"))
    config = StrataConfig(cache_dir=tmp_path / "rg-cache", cache_granularity="row_group")

    table, _ = _scan(config, uri, columns=["y"])
    assert sorted(table.column("y").to_pylist()) == [10, 20]


def test_a_promoted_type_reads_at_its_new_width(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    catalog.create_table(
        "db.t", schema=pa.schema([("id", pa.int32()), ("v", pa.float32())])
    ).append(pa.table({"id": pa.array([1], pa.int32()), "v": pa.array([1.5], pa.float32())}))
    with catalog.load_table("db.t").update_schema() as update:
        update.update_column("id", LongType())
        update.update_column("v", DoubleType())
    catalog.load_table("db.t").append(
        pa.table({"id": pa.array([2], pa.int64()), "v": pa.array([2.5], pa.float64())})
    )
    config = StrataConfig(cache_dir=tmp_path / "cache")

    table, _ = _scan(config, f"{warehouse.as_uri()}#db.t")
    assert table.schema.field("id").type == pa.int64()
    assert table.schema.field("v").type == pa.float64()
    assert _rows(table) == [{"id": 1, "v": 1.5}, {"id": 2, "v": 2.5}]


def test_a_scan_naming_a_snapshot_reads_that_snapshots_schema(lake):
    """Time travel reads the columns as they were named then, as pyiceberg does."""
    catalog, uri, config = lake
    before = catalog.load_table("db.t").current_snapshot().snapshot_id
    _evolve(catalog, lambda u: u.rename_column("x", "y"))

    assert _pyiceberg(catalog, snapshot_id=before) == [{"id": 1, "x": 10}, {"id": 2, "x": 20}]
    table, plan = _scan(config, uri, snapshot_id=before)
    assert _rows(table) == [{"id": 1, "x": 10}, {"id": 2, "x": 20}]
    assert plan.schema_id == catalog.load_table("db.t").snapshot_by_id(before).schema_id


def test_a_schema_change_is_not_served_from_the_cache_of_the_old_schema(lake):
    """A schema change makes no snapshot, so the snapshot alone cannot key the cache."""
    catalog, uri, config = lake
    planner = ReadPlanner(config)  # one planner, as a server has
    assert _rows(_scan(config, uri, planner=planner)[0]) == [{"id": 1, "x": 10}, {"id": 2, "x": 20}]
    _evolve(catalog, lambda u: u.delete_column("x"))
    _evolve(catalog, lambda u: u.add_column("x", LongType()))

    table, plan = _scan(config, uri, planner=planner)
    assert plan.schema_id != catalog.load_table("db.t").current_snapshot().schema_id
    assert _rows(table) == [{"id": 1, "x": None}, {"id": 2, "x": None}]


def test_a_scan_artifact_is_not_reused_across_a_schema_change(lake, tmp_path):
    """The artifact dedups on the scan's provenance, which must see the schema."""
    import time

    from strata_client.client import StrataClient

    from tests.conftest import run_server_with_context

    catalog, uri, _ = lake
    scan = {"executor": "scan@v1", "params": {}}
    with run_server_with_context(tmp_path / "server-cache", tmp_path / "artifacts") as ctx:
        client = StrataClient(base_url=ctx.base_url)
        try:
            first = client.materialize(inputs=[uri], transform=scan)
            assert _rows(client.fetch(first.uri)) == [{"id": 1, "x": 10}, {"id": 2, "x": 20}]
            time.sleep(0.5)  # let the artifact finalize, as test_unified_materialize does
            _evolve(catalog, lambda u: u.delete_column("x"))
            _evolve(catalog, lambda u: u.add_column("x", LongType()))

            second = client.materialize(inputs=[uri], transform=scan)
            assert second.cache_hit is False
            assert _rows(client.fetch(second.uri)) == [{"id": 1, "x": None}, {"id": 2, "x": None}]
        finally:
            client.close()


def test_the_scan_provenance_names_a_schema_that_is_not_the_snapshots():
    def provenance(schema_id):
        return materialize_service.compute_identity_provenance(
            table_identity="strata.db.t",
            snapshot_id=1,
            columns=None,
            filters=[],
            schema_id=schema_id,
        )

    assert provenance(None) != provenance(2)
    assert provenance(2) != provenance(3)


def test_a_filter_on_a_renamed_column_still_prunes_by_its_statistics(tmp_path):
    """Row-group statistics are found under the file's own name for the column."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    catalog.create_table(
        "db.t",
        schema=pa.schema([("id", pa.int64()), ("x", pa.int64())]),
        properties={"write.parquet.row-group-limit": "2"},
    ).append(
        pa.table(
            {"id": pa.array([1, 2, 3, 4], pa.int64()), "x": pa.array([10, 20, 30, 40], pa.int64())}
        )
    )
    _evolve(catalog, lambda u: u.rename_column("x", "y"))

    config = StrataConfig(cache_dir=tmp_path / "cache")
    _, plan = _scan(config, f"{warehouse.as_uri()}#db.t", filters=[Filter("y", FilterOp.GT, 25)])
    assert [task.row_group_id for task in plan.tasks] == [1]
    assert plan.pruned_row_groups == 1


POINT = pa.struct([("a", pa.int32()), ("b", pa.string())])


@pytest.fixture
def nested(tmp_path):
    """``db.t`` with a struct, a list of structs and a map to structs, one file."""
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    schema = pa.schema(
        [
            ("id", pa.int64()),
            ("p", POINT),
            ("l", pa.list_(POINT)),
            ("m", pa.map_(pa.string(), POINT)),
        ]
    )
    catalog.create_table("db.t", schema=schema).append(
        pa.table(
            {
                "id": pa.array([1, 2], pa.int64()),
                "p": pa.array([{"a": 1, "b": "x"}, None], POINT),
                "l": pa.array([[{"a": 5, "b": "y"}], [{"a": 6, "b": "w"}]], pa.list_(POINT)),
                "m": pa.array([[("k", {"a": 7, "b": "z"})], []], pa.map_(pa.string(), POINT)),
            }
        )
    )
    return catalog, f"{warehouse.as_uri()}#db.t", StrataConfig(cache_dir=tmp_path / "cache")


def test_nested_fields_are_matched_by_field_id(nested):
    """Add, rename and widen in a struct; add in list elements; drop in map values."""
    catalog, uri, config = nested

    def change(u):
        u.add_column(("p", "c"), LongType())
        u.rename_column("p.b", "bee")
        u.update_column("p.a", LongType())
        u.add_column(("l", "element", "c"), StringType())
        u.delete_column("m.value.b")

    _evolve(catalog, change)

    expected = [
        {
            "id": 1,
            "p": {"a": 1, "bee": "x", "c": None},
            "l": [{"a": 5, "b": "y", "c": None}],
            "m": [("k", {"a": 7})],
        },
        {"id": 2, "p": None, "l": [{"a": 6, "b": "w", "c": None}], "m": []},
    ]
    assert _pyiceberg(catalog) == expected
    table, _ = _scan(config, uri)
    assert _rows(table) == expected
    assert table.schema.field("p").type.field("a").type == pa.int64()


def test_a_nested_field_dropped_and_added_again_reads_as_nulls(nested):
    catalog, uri, config = nested
    _evolve(catalog, lambda u: u.delete_column("p.b"))
    _evolve(catalog, lambda u: u.add_column(("p", "b"), StringType()))

    rows = _rows(_scan(config, uri, columns=["id", "p"])[0])
    assert rows == [{"id": 1, "p": {"a": 1, "b": None}}, {"id": 2, "p": None}]


def test_older_and_newer_files_of_a_nested_column_stream_as_one_schema(nested):
    catalog, uri, config = nested
    _evolve(catalog, lambda u: u.add_column(("p", "c"), LongType()))
    point = pa.struct([("a", pa.int32()), ("b", pa.string()), ("c", pa.int64())])
    catalog.load_table("db.t").append(
        pa.table(
            {
                "id": pa.array([3], pa.int64()),
                "p": pa.array([{"a": 3, "b": "n", "c": 30}], point),
                "l": pa.array([[]], pa.list_(POINT)),
                "m": pa.array([[]], pa.map_(pa.string(), POINT)),
            }
        )
    )

    _, plan = _scan(config, uri)
    fetcher = CachedFetcher(config)
    merger = IncrementalIpcMerger()
    streamed = b"".join(merger.feed(fetcher.fetch_as_stream_bytes(task)) for task in plan.tasks)
    streamed += merger.finish()
    rows = _rows(pa.ipc.open_stream(streamed).read_all())
    assert [row["p"] for row in rows] == [
        {"a": 1, "b": "x", "c": None},
        None,
        {"a": 3, "b": "n", "c": 30},
    ]


def test_a_sliced_nested_array_is_rebuilt_from_its_own_rows():
    """Row groups arrive sliced (a delete filter, a chunk boundary): offsets count."""
    from pyiceberg.types import IntegerType, ListType, MapType, NestedField, StructType

    from strata.iceberg_schema import _reshape

    file_type = ListType(3, StructType(NestedField(4, "a", IntegerType(), required=False)))
    table_type = ListType(
        3,
        StructType(
            NestedField(4, "a", LongType(), required=False),
            NestedField(5, "c", StringType(), required=False),
        ),
    )
    target = pa.list_(pa.struct([("a", pa.int64()), ("c", pa.string())]))
    stored = pa.array(
        [[{"a": 1}], None, [{"a": 2}, {"a": 3}], [{"a": 4}]],
        pa.list_(pa.struct([("a", pa.int32())])),
    ).slice(1, 2)

    assert _reshape(stored, file_type, table_type, target).to_pylist() == [
        None,
        [{"a": 2, "c": None}, {"a": 3, "c": None}],
    ]

    map_file = MapType(6, StringType(), 7, StructType(NestedField(4, "a", IntegerType())))
    map_table = MapType(
        6,
        StringType(),
        7,
        StructType(
            NestedField(4, "a", LongType()), NestedField(5, "c", StringType(), required=False)
        ),
    )
    map_target = pa.map_(pa.string(), pa.struct([("a", pa.int64()), ("c", pa.string())]))
    stored_map = pa.array(
        [[("k", {"a": 1})], None, [("m", {"a": 2}), ("n", {"a": 3})]],
        pa.map_(pa.string(), pa.struct([("a", pa.int32())])),
    ).slice(1, 2)
    assert _reshape(stored_map, map_file, map_table, map_target).to_pylist() == [
        None,
        [("m", {"a": 2, "c": None}), ("n", {"a": 3, "c": None})],
    ]


def test_a_nested_type_iceberg_cannot_read_is_refused():
    from pyiceberg.types import ListType, NestedField, StructType

    from strata.iceberg_schema import _unreadable

    struct = StructType(NestedField(4, "a", LongType(), required=False))
    listed = ListType(3, LongType())
    assert _unreadable(struct, listed) is not None
    narrowed = StructType(NestedField(4, "a", StringType(), required=False))
    assert _unreadable(struct, narrowed) is not None
    assert _unreadable(struct, struct) is None


def test_an_unchanged_table_reads_its_files_as_written(lake):
    """No layout when a file already matches: the common case keeps its old path."""
    _, uri, config = lake
    _, plan = _scan(config, uri)
    assert [task.file_columns for task in plan.tasks] == [None]


def _catalog(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    return catalog, f"{warehouse.as_uri()}#db.t"


def test_a_file_holding_nanosecond_timestamps_reads_at_the_tables_unit(tmp_path):
    """A registered file can hold nanoseconds (INT96 reads as nanoseconds too).

    pyiceberg truncates to the table's microseconds, and so does the scan.
    """
    import datetime

    from tests.iceberg_fixtures import commit_files, data_file

    catalog, uri = _catalog(tmp_path)
    schema = pa.schema([("id", pa.int64()), ("ts", pa.timestamp("us"))])
    catalog.create_table("db.t", schema=schema).append(
        pa.table(
            {
                "id": pa.array([1], pa.int64()),
                "ts": pa.array([datetime.datetime(2024, 1, 1)], pa.timestamp("us")),
            },
            schema=schema,
        )
    )
    table = catalog.load_table("db.t")
    nanos = 1_704_153_600_000_000_123  # 2024-01-02 plus 123 ns
    commit_files(
        table,
        data_file(
            table,
            pa.table(
                {"id": pa.array([2], pa.int64()), "ts": pa.array([nanos], pa.timestamp("ns"))}
            ),
        ),
    )

    scanned, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert scanned.schema.field("ts").type == pa.timestamp("us")
    assert _rows(scanned) == _pyiceberg(catalog)
    assert _rows(scanned)[1]["ts"] == datetime.datetime(2024, 1, 2)


def test_a_required_column_made_optional_streams_as_one_schema(tmp_path):
    """The older file says `id` is not null, the newer one that it may be.

    Row groups must take the snapshot's nullability or the merge and plan schema reject the mix.
    """
    from pyiceberg.schema import Schema
    from pyiceberg.types import NestedField

    catalog, uri = _catalog(tmp_path)
    catalog.create_table(
        "db.t",
        schema=Schema(
            NestedField(1, "id", LongType(), required=True),
            NestedField(2, "name", StringType(), required=False),
        ),
    ).append(
        pa.table(
            {"id": pa.array([1], pa.int64()), "name": ["a"]},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False), ("name", pa.string())]),
        )
    )
    with catalog.load_table("db.t").update_schema(allow_incompatible_changes=True) as update:
        update.update_column("id", required=False)
    catalog.load_table("db.t").append(pa.table({"id": pa.array([None], pa.int64()), "name": ["b"]}))
    config = StrataConfig(cache_dir=tmp_path / "cache")

    table, plan = _scan(config, uri)
    assert sorted(table.column("name").to_pylist()) == ["a", "b"]
    fetcher = CachedFetcher(config)
    merger = IncrementalIpcMerger()
    streamed = b"".join(merger.feed(fetcher.fetch_as_stream_bytes(task)) for task in plan.tasks)
    assert pa.ipc.open_stream(streamed + merger.finish()).read_all().num_rows == 2


@pytest.mark.parametrize(
    ("column", "path", "older", "newer"),
    [
        (
            StructType(NestedField(3, "a", LongType(), required=True)),
            ("s", "a"),
            pa.array([{"a": 1}], pa.struct([pa.field("a", pa.int64(), nullable=False)])),
            pa.array([{"a": None}], pa.struct([("a", pa.int64())])),
        ),
        (
            ListType(3, LongType(), element_required=True),
            ("s", "element"),
            pa.array([[1]], pa.list_(pa.field("element", pa.int64(), nullable=False))),
            pa.array([[None]], pa.list_(pa.int64())),
        ),
        (
            MapType(3, StringType(), 4, LongType(), value_required=True),
            ("s", "value"),
            pa.array(
                [[("k", 1)]],
                pa.map_(
                    pa.field("key", pa.string(), nullable=False),
                    pa.field("value", pa.int64(), nullable=False),
                ),
            ),
            pa.array([[("k", None)]], pa.map_(pa.string(), pa.int64())),
        ),
    ],
    ids=["struct child", "list element", "map value"],
)
def test_a_nested_field_made_optional_streams_as_one_schema(tmp_path, column, path, older, newer):
    """The nested-field version of the nullability mix above."""
    from pyiceberg.schema import Schema

    catalog, uri = _catalog(tmp_path)
    table = catalog.create_table(
        "db.t",
        schema=Schema(
            NestedField(1, "id", LongType(), required=False),
            NestedField(2, "s", column, required=False),
        ),
    )
    table.append(pa.table({"id": [1], "s": older}, schema=table.schema().as_arrow()))
    with catalog.load_table("db.t").update_schema() as update:
        update.make_column_optional(path)
    table = catalog.load_table("db.t")
    table.append(pa.table({"id": [2], "s": newer}, schema=table.schema().as_arrow()))
    config = StrataConfig(cache_dir=tmp_path / "cache")

    scanned, plan = _scan(config, uri)
    assert _rows(scanned) == _pyiceberg(catalog)
    fetcher = CachedFetcher(config)
    merger = IncrementalIpcMerger()
    streamed = b"".join(merger.feed(fetcher.fetch_as_stream_bytes(task)) for task in plan.tasks)
    assert _rows(pa.ipc.open_stream(streamed + merger.finish()).read_all()) == _rows(scanned)


NANOS = 1_704_153_600_000_000_123  # 2024-01-02 plus 123 ns


def _nanos_inside(tmp_path, table_type, file_type, first, nanos, evolve=None):
    """``db.t`` with *first* in ``s``, then a file holding *nanos* as *file_type*.

    Field ids are on the file's top-level columns only, so nested fields go by the name mapping.
    """
    from pyiceberg.table.name_mapping import create_mapping_from_schema

    from tests.iceberg_fixtures import commit_files, data_file

    catalog, uri = _catalog(tmp_path)
    schema = pa.schema([("id", pa.int64()), ("s", table_type)])
    table = catalog.create_table("db.t", schema=schema)
    table.append(pa.table({"id": pa.array([1], pa.int64()), "s": pa.array([first], table_type)}))
    mapping = create_mapping_from_schema(table.schema()).model_dump_json()
    with table.transaction() as transaction:
        transaction.set_properties({"schema.name-mapping.default": mapping})
    if evolve is not None:
        _evolve(catalog, evolve)
    table = catalog.load_table("db.t")
    rows = pa.table({"id": pa.array([2], pa.int64()), "s": pa.array([nanos], file_type)})
    commit_files(table, data_file(table, rows))
    return catalog, uri


US, NS = pa.timestamp("us"), pa.timestamp("ns")
JAN_1, JAN_2 = datetime.datetime(2024, 1, 1), datetime.datetime(2024, 1, 2)


@pytest.mark.parametrize(
    ("table_type", "file_type", "first", "nanos", "expected"),
    [
        (pa.struct([("t", US)]), pa.struct([("t", NS)]), {"t": JAN_1}, {"t": NANOS}, {"t": JAN_2}),
        (pa.list_(US), pa.list_(NS), [JAN_1], [NANOS], [JAN_2]),
        (
            pa.map_(pa.string(), US),
            pa.map_(pa.string(), NS),
            [("k", JAN_1)],
            [("k", NANOS)],
            [("k", JAN_2)],
        ),
    ],
    ids=["struct", "list", "map"],
)
def test_nanosecond_timestamps_inside_a_nested_column_read_at_the_tables_unit(
    tmp_path, table_type, file_type, first, nanos, expected
):
    """A nested timestamp is truncated to microseconds inside a struct, list or map.

    pyiceberg refuses the lossy cast inside a list or map; the scan truncates in all three.
    """
    _, uri = _nanos_inside(tmp_path, table_type, file_type, first, nanos)

    scanned, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert scanned.schema.field("s").type == table_type
    assert _rows(scanned) == [{"id": 1, "s": first}, {"id": 2, "s": expected}]


def test_nanosecond_timestamps_inside_a_reshaped_struct_are_truncated(tmp_path):
    """A struct rebuilt field by field still truncates its nanoseconds."""
    catalog, uri = _nanos_inside(
        tmp_path,
        pa.struct([("t", US)]),
        pa.struct([("t", NS)]),
        {"t": JAN_1},
        {"t": NANOS},
        evolve=lambda u: u.add_column(("s", "c"), LongType()),
    )

    scanned, _ = _scan(StrataConfig(cache_dir=tmp_path / "cache"), uri)
    assert _rows(scanned) == _pyiceberg(catalog)
    assert _rows(scanned)[1]["s"] == {"t": JAN_2, "c": None}


def test_a_column_older_files_predate_reads_its_v3_initial_default(tmp_path):
    """Files older than a v3 column read its initial-default, not null.

    pyiceberg cannot write one yet, so the table's metadata gains it in memory.
    """
    from pyiceberg.schema import Schema
    from pyiceberg.table import Table
    from pyiceberg.types import NestedField

    from strata.iceberg import PyIcebergCatalog

    catalog, uri = _catalog(tmp_path)
    schema = pa.schema([("id", pa.int64())])
    catalog.create_table("db.t", schema=schema).append(
        pa.table({"id": pa.array([1], pa.int64())}, schema=schema)
    )
    table = catalog.load_table("db.t")
    evolved = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(
            2, "team", StringType(), required=False, initial_default="red", write_default="red"
        ),
        schema_id=1,
    )
    metadata = table.metadata.model_copy(
        update={
            "schemas": [*table.metadata.schemas, evolved],
            "current_schema_id": 1,
            "last_column_id": 2,
        }
    )
    patched = Table(table._identifier, metadata, table.metadata_location, table.io, table.catalog)

    class WithDefault(PyIcebergCatalog):
        def load_table(self, table_uri):
            return patched

    config = StrataConfig(cache_dir=tmp_path / "cache")
    scanned, _ = _scan(config, uri, planner=ReadPlanner(config, catalog=WithDefault(config)))
    assert scanned.to_pylist() == [{"id": 1, "team": "red"}]

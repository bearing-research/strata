"""Iceberg schema evolution: a scan reads older data files as the table's schema.

Iceberg names a column by field id, and changing a schema rewrites no data
files. Strata read columns by name, so dropping a column and adding one of the
same name served the dropped column's values, a rename served the old name
(and a projection naming the new one crashed mid-stream), and adding a column
refused the table. Each test checks the scan against pyiceberg's own read.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pytest
from pyiceberg.types import DoubleType, LongType, StringType

from strata.cache import CachedFetcher
from strata.config import StrataConfig
from strata.fast_io import IncrementalIpcMerger
from strata.planner import ReadPlanner, UnsupportedTableFormatError
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


def test_a_change_inside_a_nested_column_is_refused(tmp_path):
    from pyiceberg.catalog.sql import SqlCatalog

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    catalog.create_namespace("db")
    point = pa.struct([("a", pa.int64())])
    catalog.create_table("db.t", schema=pa.schema([("id", pa.int64()), ("p", point)])).append(
        pa.table({"id": pa.array([1], pa.int64()), "p": pa.array([{"a": 1}], point)})
    )
    with catalog.load_table("db.t").update_schema() as update:
        update.add_column(("p", "b"), LongType())

    with pytest.raises(UnsupportedTableFormatError, match="nested column 'p'"):
        ReadPlanner(StrataConfig(cache_dir=tmp_path / "cache")).plan(f"{warehouse.as_uri()}#db.t")


def test_an_unchanged_table_reads_its_files_as_written(lake):
    """No layout when a file already matches: the common case keeps its old path."""
    _, uri, config = lake
    _, plan = _scan(config, uri)
    assert [task.file_columns for task in plan.tasks] == [None]

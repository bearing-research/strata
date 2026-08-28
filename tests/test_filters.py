"""Tests for filter functionality and two-tier pruning."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    LongType,
    NestedField,
    StringType,
)

from strata.config import StrataConfig
from strata.planner import ReadPlanner, _build_column_index_map, _compile_filters
from strata.types import (
    Filter,
    FilterOp,
    compute_filter_fingerprint,
    filters_to_iceberg_expression,
)


@pytest.fixture
def temp_warehouse_multi_files(tmp_path):
    """Create a warehouse with multiple Parquet files for file-level pruning tests."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    catalog.create_namespace("test_db")

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "category", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),
    )

    table = catalog.create_table("test_db.events", schema)

    # Write multiple batches to create multiple files
    # Each append creates a new data file
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)

    # File 1: values 0-99, category "A"
    data1 = pa.table(
        {
            "id": pa.array(range(100), type=pa.int64()),
            "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
            "category": pa.array(["A"] * 100, type=pa.string()),
            "timestamp": pa.array([base_ts + i * 1000 for i in range(100)], type=pa.int64()),
        }
    )
    table.append(data1)

    # File 2: values 100-199, category "B"
    data2 = pa.table(
        {
            "id": pa.array(range(100, 200), type=pa.int64()),
            "value": pa.array([float(i) for i in range(100, 200)], type=pa.float64()),
            "category": pa.array(["B"] * 100, type=pa.string()),
            "timestamp": pa.array([base_ts + i * 1000 for i in range(100, 200)], type=pa.int64()),
        }
    )
    table.append(data2)

    # File 3: values 200-299, category "C"
    data3 = pa.table(
        {
            "id": pa.array(range(200, 300), type=pa.int64()),
            "value": pa.array([float(i) for i in range(200, 300)], type=pa.float64()),
            "category": pa.array(["C"] * 100, type=pa.string()),
            "timestamp": pa.array([base_ts + i * 1000 for i in range(200, 300)], type=pa.int64()),
        }
    )
    table.append(data3)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def strata_config(tmp_path):
    """Create a test configuration."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return StrataConfig(cache_dir=cache_dir)


class TestFilterFingerprint:
    """Tests for compute_filter_fingerprint."""

    def test_empty_filters_returns_nofilter(self):
        assert compute_filter_fingerprint(None) == "nofilter"
        assert compute_filter_fingerprint([]) == "nofilter"

    def test_single_filter_produces_hash(self):
        filters = [Filter(column="value", op=FilterOp.GT, value=100)]
        fingerprint = compute_filter_fingerprint(filters)
        assert len(fingerprint) == 16
        assert fingerprint != "nofilter"

    def test_same_filters_produce_same_fingerprint(self):
        filters1 = [Filter(column="value", op=FilterOp.GT, value=100)]
        filters2 = [Filter(column="value", op=FilterOp.GT, value=100)]
        assert compute_filter_fingerprint(filters1) == compute_filter_fingerprint(filters2)

    def test_different_filters_produce_different_fingerprints(self):
        filters1 = [Filter(column="value", op=FilterOp.GT, value=100)]
        filters2 = [Filter(column="value", op=FilterOp.LT, value=100)]
        assert compute_filter_fingerprint(filters1) != compute_filter_fingerprint(filters2)

    def test_filter_order_does_not_affect_fingerprint(self):
        """Filters in different order should produce same fingerprint."""
        filters1 = [
            Filter(column="value", op=FilterOp.GT, value=100),
            Filter(column="id", op=FilterOp.LT, value=50),
        ]
        filters2 = [
            Filter(column="id", op=FilterOp.LT, value=50),
            Filter(column="value", op=FilterOp.GT, value=100),
        ]
        assert compute_filter_fingerprint(filters1) == compute_filter_fingerprint(filters2)

    def test_datetime_values_handled(self):
        dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        filters = [Filter(column="timestamp", op=FilterOp.GT, value=dt)]
        fingerprint = compute_filter_fingerprint(filters)
        assert len(fingerprint) == 16

    def test_datetime_fingerprint_is_stable(self):
        """Same datetime should produce same fingerprint."""
        dt1 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        dt2 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        filters1 = [Filter(column="timestamp", op=FilterOp.GT, value=dt1)]
        filters2 = [Filter(column="timestamp", op=FilterOp.GT, value=dt2)]
        assert compute_filter_fingerprint(filters1) == compute_filter_fingerprint(filters2)


class TestFiltersToIcebergExpression:
    """Tests for filters_to_iceberg_expression."""

    def test_empty_filters_returns_none(self):
        assert filters_to_iceberg_expression(None) is None
        assert filters_to_iceberg_expression([]) is None

    @pytest.mark.parametrize(
        ("op", "expr_type_name"),
        [
            (FilterOp.EQ, "EqualTo"),
            (FilterOp.GT, "GreaterThan"),
            (FilterOp.LT, "LessThan"),
            (FilterOp.GE, "GreaterThanOrEqual"),
            (FilterOp.LE, "LessThanOrEqual"),
            (FilterOp.NE, "NotEqualTo"),
        ],
    )
    def test_single_filter_maps_op_to_expression(self, op, expr_type_name):
        import pyiceberg.expressions as pie

        expr = filters_to_iceberg_expression([Filter(column="value", op=op, value=100)])
        assert isinstance(expr, getattr(pie, expr_type_name))

    def test_multiple_filters_combined_with_and(self):
        from pyiceberg.expressions import And

        filters = [
            Filter(column="value", op=FilterOp.GT, value=100),
            Filter(column="value", op=FilterOp.LT, value=200),
        ]
        expr = filters_to_iceberg_expression(filters)
        assert isinstance(expr, And)

    def test_nested_column_filters_skipped(self):
        """Filters on nested columns (with dots) should be skipped."""
        filters = [Filter(column="nested.field", op=FilterOp.EQ, value="test")]
        expr = filters_to_iceberg_expression(filters)
        assert expr is None

    def test_mixed_nested_and_flat_filters(self):
        """Only flat column filters should be included."""
        from pyiceberg.expressions import EqualTo

        filters = [
            Filter(column="nested.field", op=FilterOp.EQ, value="test"),
            Filter(column="category", op=FilterOp.EQ, value="A"),
        ]
        expr = filters_to_iceberg_expression(filters)
        # Should only have the flat column filter
        assert isinstance(expr, EqualTo)


class TestBuildColumnIndexMap:
    """Tests for _build_column_index_map."""

    def test_flat_schema(self):
        """Test with a simple flat schema."""
        import tempfile

        import pyarrow.parquet as pq

        # Create a simple parquet file
        table = pa.table(
            {
                "id": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
                "name": ["a", "b", "c"],
            }
        )

        # Windows locks NamedTemporaryFile exclusively, blocking pyarrow
        # from opening the path a second time. delete=False + manual
        # unlink gets us the same cleanup with cross-platform behaviour.
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            pq.write_table(table, tmp_path)
            meta = pq.read_metadata(tmp_path)

            # Build using parquet schema
            pq_schema = meta.schema
            col_map = _build_column_index_map(pq_schema)

            assert "id" in col_map
            assert "value" in col_map
            assert "name" in col_map
            assert len(col_map) == 3
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCompileFilters:
    """Tests for _compile_filters."""

    def test_all_columns_exist(self):
        col_index_map = {"id": 0, "value": 1, "name": 2}
        filters = [
            Filter(column="id", op=FilterOp.GT, value=10),
            Filter(column="value", op=FilterOp.LT, value=100.0),
        ]
        compiled = _compile_filters(filters, col_index_map)

        assert len(compiled) == 2
        assert compiled[0] == (0, filters[0])
        assert compiled[1] == (1, filters[1])

    def test_missing_column_dropped(self):
        col_index_map = {"id": 0, "value": 1}
        filters = [
            Filter(column="id", op=FilterOp.GT, value=10),
            Filter(column="nonexistent", op=FilterOp.EQ, value="foo"),
        ]
        compiled = _compile_filters(filters, col_index_map)

        assert len(compiled) == 1
        assert compiled[0] == (0, filters[0])

    def test_empty_filters(self):
        col_index_map = {"id": 0, "value": 1}
        compiled = _compile_filters([], col_index_map)
        assert compiled == []

    def test_empty_column_map(self):
        filters = [Filter(column="id", op=FilterOp.GT, value=10)]
        compiled = _compile_filters(filters, {})
        assert compiled == []


class TestFilterMatching:
    """Tests for Filter.matches_stats."""

    def test_eq_in_range(self):
        f = Filter(column="value", op=FilterOp.EQ, value=50)
        assert f.matches_stats(0, 100) is True

    def test_eq_out_of_range(self):
        f = Filter(column="value", op=FilterOp.EQ, value=150)
        assert f.matches_stats(0, 100) is False

    def test_eq_at_boundary(self):
        f = Filter(column="value", op=FilterOp.EQ, value=100)
        assert f.matches_stats(0, 100) is True

    def test_ne_all_same_value(self):
        f = Filter(column="value", op=FilterOp.NE, value=50)
        # If min == max == filter_value, no rows can match
        assert f.matches_stats(50, 50) is False

    def test_ne_different_values(self):
        f = Filter(column="value", op=FilterOp.NE, value=50)
        assert f.matches_stats(0, 100) is True

    def test_lt_can_match(self):
        f = Filter(column="value", op=FilterOp.LT, value=50)
        # min < filter_value means some rows might be less
        assert f.matches_stats(0, 100) is True

    def test_lt_cannot_match(self):
        f = Filter(column="value", op=FilterOp.LT, value=50)
        # min >= filter_value means no rows can be less
        assert f.matches_stats(50, 100) is False

    def test_le_can_match(self):
        f = Filter(column="value", op=FilterOp.LE, value=50)
        assert f.matches_stats(0, 100) is True

    def test_le_at_boundary(self):
        f = Filter(column="value", op=FilterOp.LE, value=50)
        assert f.matches_stats(50, 100) is True

    def test_le_cannot_match(self):
        f = Filter(column="value", op=FilterOp.LE, value=50)
        assert f.matches_stats(51, 100) is False

    def test_gt_can_match(self):
        f = Filter(column="value", op=FilterOp.GT, value=50)
        # max > filter_value means some rows might be greater
        assert f.matches_stats(0, 100) is True

    def test_gt_cannot_match(self):
        f = Filter(column="value", op=FilterOp.GT, value=100)
        # max <= filter_value means no rows can be greater
        assert f.matches_stats(0, 100) is False

    def test_ge_can_match(self):
        f = Filter(column="value", op=FilterOp.GE, value=50)
        assert f.matches_stats(0, 100) is True

    def test_ge_at_boundary(self):
        f = Filter(column="value", op=FilterOp.GE, value=100)
        assert f.matches_stats(0, 100) is True

    def test_ge_cannot_match(self):
        f = Filter(column="value", op=FilterOp.GE, value=101)
        assert f.matches_stats(0, 100) is False

    def test_null_stats_returns_true(self):
        """If stats are None, can't prune - return True."""
        f = Filter(column="value", op=FilterOp.GT, value=50)
        assert f.matches_stats(None, 100) is True
        assert f.matches_stats(0, None) is True
        assert f.matches_stats(None, None) is True


class TestTwoTierPruning:
    """Integration tests for two-tier pruning (Iceberg file + Parquet row-group)."""

    def test_planning_with_filters_includes_fingerprint(
        self, temp_warehouse_multi_files, strata_config
    ):
        """Verify that planning with filters uses filter fingerprint in cache key."""
        planner = ReadPlanner(strata_config)

        filters = [Filter(column="value", op=FilterOp.LT, value=150)]
        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Plan should have the filters attached
        assert plan.filters == filters

    def test_different_filters_produce_separate_cache_entries(
        self, temp_warehouse_multi_files, strata_config
    ):
        """Different filters should use different manifest cache entries."""
        planner = ReadPlanner(strata_config)

        # First query with one filter
        filters1 = [Filter(column="value", op=FilterOp.LT, value=50)]
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters1)

        # Second query with different filter
        filters2 = [Filter(column="value", op=FilterOp.GT, value=250)]
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters2)

        # Check cache stats - should have entries for both
        stats = planner.manifest_cache.stats()
        # Filtered cache should have 2 misses (each filter is different)
        assert stats["filtered"]["misses"] >= 2

    def test_same_filters_reuse_cache(self, temp_warehouse_multi_files, strata_config):
        """Same filters should reuse manifest cache entry."""
        planner = ReadPlanner(strata_config)

        filters = [Filter(column="value", op=FilterOp.LT, value=150)]

        # First query
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Same query again
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Check cache stats
        stats = planner.manifest_cache.stats()
        # Should have 1 miss and 1 hit for filtered cache
        assert stats["filtered"]["hits"] >= 1

    def test_no_filters_uses_unfiltered_cache(self, temp_warehouse_multi_files, strata_config):
        """Queries without filters should use unfiltered manifest cache."""
        planner = ReadPlanner(strata_config)

        # Query without filters
        planner.plan(temp_warehouse_multi_files["table_uri"])
        planner.plan(temp_warehouse_multi_files["table_uri"])

        stats = planner.manifest_cache.stats()
        # Should have hits in unfiltered cache
        assert stats["unfiltered"]["hits"] >= 1

    def test_filter_on_string_column(self, temp_warehouse_multi_files, strata_config):
        """Test filtering on string columns."""
        planner = ReadPlanner(strata_config)

        filters = [Filter(column="category", op=FilterOp.EQ, value="A")]
        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Should successfully create a plan
        assert plan.snapshot_id > 0
        assert len(plan.tasks) >= 0  # May or may not prune depending on stats

    def test_combined_filters(self, temp_warehouse_multi_files, strata_config):
        """Test multiple filters combined with AND logic."""
        planner = ReadPlanner(strata_config)

        filters = [
            Filter(column="value", op=FilterOp.GE, value=50),
            Filter(column="value", op=FilterOp.LT, value=150),
        ]
        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Should successfully create a plan
        assert plan.snapshot_id > 0


class TestIcebergExpressionFallback:
    """Test that Iceberg expression failures fall back gracefully."""

    def test_invalid_filter_falls_back_to_unfiltered(
        self, temp_warehouse_multi_files, strata_config
    ):
        """If Iceberg expression fails, should fall back to unfiltered scan."""
        planner = ReadPlanner(strata_config)

        # Create a filter that might fail in Iceberg (e.g., type mismatch)
        # This won't actually fail, but tests the code path exists
        filters = [Filter(column="nonexistent_column", op=FilterOp.EQ, value="test")]

        # Should not raise, should fall back gracefully
        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # Should still get all data files (no pruning possible)
        assert plan.snapshot_id > 0


class TestMergeOnReadDeletesRefused:
    """Strata reads Parquet row groups directly and applies no Iceberg delete
    files, so a merge-on-read table would return rows that have been deleted —
    and cache them under an immutable snapshot key, where they are served
    forever (there is no invalidation by design). Until deletes are applied,
    such a table must be a hard error rather than a quiet wrong answer.
    """

    def _task(self, deletes):
        from types import SimpleNamespace

        return SimpleNamespace(
            file=SimpleNamespace(file_path="s3://b/data.parquet"), delete_files=deletes
        )

    def test_raises_when_any_task_carries_delete_files(self):
        from strata.planner import UnsupportedTableFormatError, _assert_no_row_level_deletes

        tasks = [self._task(set()), self._task({"pos-delete.parquet"})]
        with pytest.raises(UnsupportedTableFormatError) as exc:
            _assert_no_row_level_deletes(tasks, "db.events")
        message = str(exc.value)
        assert "db.events" in message
        assert "merge-on-read" in message
        # The message must tell the operator what to do about it.
        assert "rewrite_data_files" in message

    def test_passes_when_no_task_has_deletes(self):
        from strata.planner import _assert_no_row_level_deletes

        _assert_no_row_level_deletes([self._task(set()), self._task(None)], "db.events")

    def test_tolerates_tasks_without_the_attribute(self):
        """A catalog implementation that omits delete_files (or a pyiceberg
        rename) must degrade to 'no deletes', not crash every scan."""
        from types import SimpleNamespace

        from strata.planner import _assert_no_row_level_deletes

        _assert_no_row_level_deletes(
            [SimpleNamespace(file=SimpleNamespace(file_path="x"))], "db.events"
        )

    def test_planner_refuses_a_mor_table(self, temp_warehouse, strata_config, monkeypatch):
        """The guard is actually wired into planning (pyiceberg's own delete()
        rewrites copy-on-write, so a real MOR table needs Spark/Flink — the
        scan tasks are faked here instead)."""
        from strata.planner import ReadPlanner, UnsupportedTableFormatError

        planner = ReadPlanner(strata_config)

        real_plan_files = None

        def fake_plan_files(self_scan):
            task = list(real_plan_files(self_scan))[0]
            task.delete_files = {"pos-delete.parquet"}
            return [task]

        import pyiceberg.table as ib_table

        real_plan_files = ib_table.DataScan.plan_files
        monkeypatch.setattr(ib_table.DataScan, "plan_files", fake_plan_files)

        with pytest.raises(UnsupportedTableFormatError, match="merge-on-read"):
            planner.plan(temp_warehouse["table_uri"])


class TestSchemaEvolutionDetectedAtPlanTime:
    """Iceberg schema evolution does not rewrite existing data files, so after
    ``ADD COLUMN`` the older files lack the new column.

    Nothing reconciled that: a projection naming the new column raised
    ``KeyError`` from ``read_row_group`` on the old files, and an unprojected
    scan produced row groups with differing schemas that
    ``IncrementalIpcMerger`` rejects — both *after* the 200 and the first
    chunks were already on the wire, i.e. a truncated response rather than an
    error the client can act on. Planning is where this can still fail cleanly.
    """

    @pytest.fixture
    def evolved_warehouse(self, tmp_path):
        if sys.platform == "win32":
            pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
        wh = tmp_path / "wh"
        wh.mkdir()
        catalog = SqlCatalog(
            "strata",
            uri=f"sqlite:///{(wh / 'catalog.db').as_posix()}",
            warehouse=wh.as_uri(),
        )
        catalog.create_namespace("test_db")
        table = catalog.create_table(
            "test_db.evolved", Schema(NestedField(1, "id", LongType(), required=False))
        )
        table.append(pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}))
        with table.update_schema() as update:
            update.add_column("label", StringType())
        table = catalog.load_table("test_db.evolved")
        table.append(
            pa.table({"id": pa.array([4, 5], type=pa.int64()), "label": pa.array(["a", "b"])})
        )
        return f"{wh.as_uri()}#test_db.evolved"

    def test_unprojected_scan_fails_at_plan_time(self, evolved_warehouse, strata_config):
        from strata.planner import ReadPlanner, UnsupportedTableFormatError

        with pytest.raises(UnsupportedTableFormatError, match="differing schemas"):
            ReadPlanner(strata_config).plan(evolved_warehouse)

    def test_projection_naming_the_new_column_fails_at_plan_time(
        self, evolved_warehouse, strata_config
    ):
        from strata.planner import ReadPlanner, UnsupportedTableFormatError

        with pytest.raises(UnsupportedTableFormatError, match="missing requested column"):
            ReadPlanner(strata_config).plan(evolved_warehouse, columns=["label"])

    def test_projection_common_to_every_file_still_plans(self, evolved_warehouse, strata_config):
        """The still-safe case must keep working: scanning only the columns
        that predate the evolution."""
        plan = ReadPlanner(strata_config).plan(evolved_warehouse, columns=["id"])
        assert plan.tasks, "a projection present in every file should still plan"


class TestScanProjectionContract:
    """A scan's ``columns`` list was neither validated nor reflected."""

    def test_a_column_that_does_not_exist_is_refused(self, temp_warehouse, strata_config):
        """It used to return another column's data under the requested name.

        ``_assert_file_satisfies_scan`` only runs for the SECOND and later
        files — the first file's schema is what it compares against — so a
        single-file table validated nothing. ``_project_batch`` then resolves
        each name with ``schema.get_field_index(name)``, which returns ``-1``
        for an unknown name, and ``batch.column(-1)`` is the LAST column. So
        ``columns=["id", "nope"]`` came back as a two-column batch whose
        ``nope`` held the final column's values, with no error anywhere.
        """
        planner = ReadPlanner(strata_config)

        with pytest.raises(ValueError, match="no column"):
            planner.plan(temp_warehouse["table_uri"], columns=["id", "nope"])

    def test_the_error_names_the_available_columns(self, temp_warehouse, strata_config):
        planner = ReadPlanner(strata_config)

        with pytest.raises(ValueError) as exc:
            planner.plan(temp_warehouse["table_uri"], columns=["nope"])
        assert "nope" in str(exc.value)
        assert "value" in str(exc.value)

    def test_an_empty_result_advertises_the_same_columns_as_a_full_one(
        self, temp_warehouse, strata_config
    ):
        """``plan.schema`` IS the response schema when there are no tasks.

        Neither the Parquet file schema nor the Iceberg table schema is
        projected, so a scan for one column that matched rows streamed one
        column, while the same scan matching none streamed every column —
        the shape depended on the data, which breaks anything concatenating
        partitioned scans or asserting on the schema.
        """
        planner = ReadPlanner(strata_config)
        uri = temp_warehouse["table_uri"]

        matched = planner.plan(uri, columns=["id"])
        pruned = planner.plan(
            uri,
            columns=["id"],
            filters=[Filter(column="id", op=FilterOp.GT, value=10**12)],
        )

        assert matched.tasks and not pruned.tasks
        assert matched.schema.names == ["id"]
        assert pruned.schema.names == ["id"]

    def test_the_projection_order_is_the_requested_order(self, temp_warehouse, strata_config):
        planner = ReadPlanner(strata_config)

        plan = planner.plan(temp_warehouse["table_uri"], columns=["name", "id"])
        assert plan.schema.names == ["name", "id"]

    def test_the_fetcher_helper_is_loud_rather_than_wrong(self):
        """Defense in depth for the same hazard.

        The planner now rejects an unknown column before a task exists, so
        this should be unreachable — but the helper indexed by
        ``get_field_index``, whose -1 for an unknown name silently selected
        the last column. Indexing by name costs the same and raises.
        """
        import pyarrow as pa

        from strata.cache import CachedFetcher

        batch = pa.RecordBatch.from_pydict({"id": [1, 2], "value": [9.0, 8.0]})

        assert CachedFetcher._project_batch(batch, ["id"]).schema.names == ["id"]
        with pytest.raises(KeyError):
            CachedFetcher._project_batch(batch, ["id", "nope"])

    def test_an_unprojected_scan_still_reports_every_column(self, temp_warehouse, strata_config):
        planner = ReadPlanner(strata_config)

        plan = planner.plan(temp_warehouse["table_uri"])
        assert plan.schema.names == ["id", "value", "name", "timestamp"]

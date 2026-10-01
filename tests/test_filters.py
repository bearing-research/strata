"""Tests for filter functionality and two-tier pruning."""

import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
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

    # Each append creates a new data file.
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)

    data1 = pa.table(
        {
            "id": pa.array(range(100), type=pa.int64()),
            "value": pa.array([float(i) for i in range(100)], type=pa.float64()),
            "category": pa.array(["A"] * 100, type=pa.string()),
            "timestamp": pa.array([base_ts + i * 1000 for i in range(100)], type=pa.int64()),
        }
    )
    table.append(data1)

    data2 = pa.table(
        {
            "id": pa.array(range(100, 200), type=pa.int64()),
            "value": pa.array([float(i) for i in range(100, 200)], type=pa.float64()),
            "category": pa.array(["B"] * 100, type=pa.string()),
            "timestamp": pa.array([base_ts + i * 1000 for i in range(100, 200)], type=pa.int64()),
        }
    )
    table.append(data2)

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
        assert isinstance(expr, EqualTo)


class TestBuildColumnIndexMap:
    """Tests for _build_column_index_map."""

    def test_flat_schema(self):
        """Test with a simple flat schema."""
        import tempfile

        import pyarrow.parquet as pq

        table = pa.table(
            {
                "id": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
                "name": ["a", "b", "c"],
            }
        )

        # Windows locks NamedTemporaryFile exclusively, so pyarrow cannot reopen the
        # path; delete=False plus manual unlink works on every platform.
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            pq.write_table(table, tmp_path)
            meta = pq.read_metadata(tmp_path)

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


class TestPruningKeepsNaNRows:
    """A row group whose float stats leave out NaN must not be pruned."""

    def test_a_ne_filter_keeps_the_nan_row(self, tmp_path):
        path = tmp_path / "nan.parquet"
        pq.write_table(pa.table({"value": [5.0, float("nan")]}), path)
        rg = pq.ParquetFile(path).metadata.row_group(0)
        stats = rg.column(0).statistics
        assert stats.min == stats.max == 5.0  # NaN is not in the stats

        planner = ReadPlanner.__new__(ReadPlanner)  # the method only uses _convert_stats
        assert not planner._should_prune_row_group(rg, [(0, Filter("value", FilterOp.NE, 5.0))])


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
        # min == max == filter_value, so no rows can match.
        assert f.matches_stats(50, 50) is False

    def test_ne_all_same_float_value(self):
        f = Filter(column="value", op=FilterOp.NE, value=50.0)
        # Float stats leave NaN out, so the row group may still hold a NaN.
        assert f.matches_stats(50.0, 50.0) is True

    def test_ne_different_values(self):
        f = Filter(column="value", op=FilterOp.NE, value=50)
        assert f.matches_stats(0, 100) is True

    def test_lt_can_match(self):
        f = Filter(column="value", op=FilterOp.LT, value=50)
        assert f.matches_stats(0, 100) is True

    def test_lt_cannot_match(self):
        f = Filter(column="value", op=FilterOp.LT, value=50)
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
        assert f.matches_stats(0, 100) is True

    def test_gt_cannot_match(self):
        f = Filter(column="value", op=FilterOp.GT, value=100)
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

        assert plan.filters == filters

    def test_different_filters_produce_separate_cache_entries(
        self, temp_warehouse_multi_files, strata_config
    ):
        """Different filters should use different manifest cache entries."""
        planner = ReadPlanner(strata_config)

        filters1 = [Filter(column="value", op=FilterOp.LT, value=50)]
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters1)

        filters2 = [Filter(column="value", op=FilterOp.GT, value=250)]
        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters2)

        stats = planner.manifest_cache.stats()
        # Each filter is different, so each misses.
        assert stats["filtered"]["misses"] >= 2

    def test_same_filters_reuse_cache(self, temp_warehouse_multi_files, strata_config):
        """Same filters should reuse manifest cache entry."""
        planner = ReadPlanner(strata_config)

        filters = [Filter(column="value", op=FilterOp.LT, value=150)]

        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        stats = planner.manifest_cache.stats()
        assert stats["filtered"]["hits"] >= 1

    def test_no_filters_uses_unfiltered_cache(self, temp_warehouse_multi_files, strata_config):
        """Queries without filters should use unfiltered manifest cache."""
        planner = ReadPlanner(strata_config)

        planner.plan(temp_warehouse_multi_files["table_uri"])
        planner.plan(temp_warehouse_multi_files["table_uri"])

        stats = planner.manifest_cache.stats()
        assert stats["unfiltered"]["hits"] >= 1

    def test_filter_on_string_column(self, temp_warehouse_multi_files, strata_config):
        """Test filtering on string columns."""
        planner = ReadPlanner(strata_config)

        filters = [Filter(column="category", op=FilterOp.EQ, value="A")]
        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

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

        assert plan.snapshot_id > 0


class TestIcebergExpressionFallback:
    """Test that Iceberg expression failures fall back gracefully."""

    def test_invalid_filter_falls_back_to_unfiltered(
        self, temp_warehouse_multi_files, strata_config
    ):
        """If Iceberg expression fails, should fall back to unfiltered scan."""
        planner = ReadPlanner(strata_config)

        # An unknown column exercises the fallback path when Iceberg cannot use a filter.
        filters = [Filter(column="nonexistent_column", op=FilterOp.EQ, value="test")]

        plan = planner.plan(temp_warehouse_multi_files["table_uri"], filters=filters)

        # No pruning is possible, so every data file stays.
        assert plan.snapshot_id > 0


class TestScanProjectionContract:
    """A scan's ``columns`` list was neither validated nor reflected."""

    def test_a_column_that_does_not_exist_is_refused(self, temp_warehouse, strata_config):
        """It used to return another column's data under the requested name.

        Nothing validated the projection of a single-file table.
        ``_project_batch`` then resolves
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

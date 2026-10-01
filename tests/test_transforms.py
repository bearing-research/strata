"""Tests for the transform system.

These tests verify:
1. Transform base class functionality
2. Built-in transforms (scan@v1, duckdb_sql@v1)
3. Transform registration and lookup
4. Parameter validation
5. Local execution via _run_local()
"""

import pyarrow as pa
import pytest
from pydantic import ValidationError

from strata.executors import _run_local as run_local
from strata.transforms import (
    DuckDBSQLParams,
    DuckDBSQLTransform,
    ScanParams,
    ScanTransform,
    Transform,
    build_duckdb_sql_transform,
    build_scan_transform,
    get_transform,
    list_transforms,
    register_transform,
)
from strata.transforms.base import _run_transform as run_transform
from strata.transforms.base import _transforms


class TestTransformBase:
    """Tests for Transform base class."""

    def test_custom_transform_registration(self):
        """Test registering a custom transform."""

        @register_transform("test_custom@v1")
        class CustomTransform(Transform):
            class Params:
                pass

            def execute(self, inputs, params):
                return inputs[0]

        assert "test_custom@v1" in list_transforms()
        transform = get_transform("test_custom@v1")
        assert transform is not None
        assert transform.ref == "test_custom@v1"

        del _transforms["test_custom@v1"]

    def test_get_transform_strips_prefix(self):
        """Test that get_transform strips local:// prefix."""
        # Found with or without the prefix.
        t1 = get_transform("duckdb_sql@v1")
        t2 = get_transform("local://duckdb_sql@v1")

        assert t1 is not None
        assert t2 is not None
        assert type(t1) is type(t2)

    def test_get_transform_unknown(self):
        """Test that get_transform returns None for unknown transforms."""
        assert get_transform("unknown@v1") is None

    def test_list_transforms_includes_builtin(self):
        """Test that list_transforms includes built-in transforms."""
        transforms = list_transforms()
        assert "scan@v1" in transforms
        assert "duckdb_sql@v1" in transforms


class TestScanTransform:
    """Tests for scan@v1 transform."""

    def test_scan_params_validation(self):
        """Test ScanParams validation."""
        params = ScanParams(columns=["a", "b"], snapshot_id=123)
        assert params.columns == ["a", "b"]
        assert params.snapshot_id == 123
        assert params.filters is None

        # All fields optional
        params = ScanParams()
        assert params.columns is None
        assert params.filters is None
        assert params.snapshot_id is None

    def test_scan_params_with_filters(self):
        """Test ScanParams with filters."""
        from strata.transforms.scan import FilterSpec

        filters = [
            FilterSpec(column="value", op=">", value=100),
            FilterSpec(column="name", op="=", value="test"),
        ]
        params = ScanParams(filters=filters)
        assert params.filters is not None
        assert len(params.filters) == 2
        assert params.filters[0].op == ">"

    def test_scan_filter_invalid_op(self):
        """Test that invalid filter operators are rejected."""
        from strata.transforms.scan import FilterSpec

        with pytest.raises(ValidationError):
            FilterSpec(column="x", op="LIKE", value="test")

    def test_scan_execute_not_supported(self):
        """Test that scan@v1 cannot be executed locally."""
        transform = ScanTransform()
        table = pa.table({"x": [1, 2, 3]})

        with pytest.raises(NotImplementedError) as exc_info:
            transform.execute([table], ScanParams())

        assert "handled by the Strata server" in str(exc_info.value)

    def test_build_scan_transform(self):
        """Test build_scan_transform helper."""
        spec = build_scan_transform(
            columns=["a", "b"],
            filters=[{"column": "x", "op": ">", "value": 10}],
            snapshot_id=123,
        )

        assert spec["executor"] == "scan@v1"
        assert spec["params"]["columns"] == ["a", "b"]
        assert spec["params"]["filters"] == [{"column": "x", "op": ">", "value": 10}]
        assert spec["params"]["snapshot_id"] == 123

    def test_build_scan_transform_minimal(self):
        """Test build_scan_transform with no arguments."""
        spec = build_scan_transform()

        assert spec["executor"] == "scan@v1"
        assert spec["params"] == {}


class TestDuckDBSQLTransform:
    """Tests for duckdb_sql@v1 transform."""

    def test_duckdb_params_validation(self):
        """Test DuckDBSQLParams validation."""
        params = DuckDBSQLParams(sql="SELECT * FROM input0")
        assert params.sql == "SELECT * FROM input0"

    def test_duckdb_params_empty_sql_rejected(self):
        """Test that empty SQL is rejected."""
        with pytest.raises(ValidationError):
            DuckDBSQLParams(sql="")

        with pytest.raises(ValidationError):
            DuckDBSQLParams(sql="   ")

    def test_duckdb_params_sql_stripped(self):
        """Test that SQL is stripped of whitespace."""
        params = DuckDBSQLParams(sql="  SELECT 1  ")
        assert params.sql == "SELECT 1"

    def test_duckdb_execute_simple(self):
        """Test simple DuckDB execution."""
        transform = DuckDBSQLTransform()
        table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})

        result = transform.execute([table], DuckDBSQLParams(sql="SELECT x + y AS z FROM input0"))

        assert result.num_rows == 3
        assert result.column_names == ["z"]
        assert result.column("z").to_pylist() == [5, 7, 9]

    def test_duckdb_execute_aggregation(self):
        """Test DuckDB aggregation."""
        transform = DuckDBSQLTransform()
        table = pa.table({"category": ["a", "a", "b"], "value": [10, 20, 30]})

        result = transform.execute(
            [table],
            DuckDBSQLParams(sql="SELECT category, SUM(value) as total FROM input0 GROUP BY 1"),
        )

        assert result.num_rows == 2
        data = {r["category"]: r["total"] for r in result.to_pylist()}
        assert data["a"] == 30
        assert data["b"] == 30

    def test_duckdb_execute_multiple_inputs(self):
        """Test DuckDB with multiple input tables."""
        transform = DuckDBSQLTransform()
        events = pa.table({"id": [1, 2, 3], "user_id": [10, 20, 10]})
        users = pa.table({"id": [10, 20], "name": ["Alice", "Bob"]})

        result = transform.execute(
            [events, users],
            DuckDBSQLParams(
                sql="SELECT e.id, u.name FROM input0 e JOIN input1 u ON e.user_id = u.id"
            ),
        )

        assert result.num_rows == 3
        names = result.column("name").to_pylist()
        assert names.count("Alice") == 2
        assert names.count("Bob") == 1

    def test_duckdb_execute_no_inputs(self):
        """Test DuckDB can execute queries without inputs."""
        transform = DuckDBSQLTransform()
        result = transform.execute([], DuckDBSQLParams(sql="SELECT 42 as answer"))

        assert result.num_rows == 1
        assert result.column("answer").to_pylist() == [42]

    def test_duckdb_run_method(self):
        """Test the high-level run() method."""
        transform = DuckDBSQLTransform()
        table = pa.table({"x": [1, 2, 3]})

        result = transform.run([table], {"sql": "SELECT x * 2 AS doubled FROM input0"})

        assert result.column("doubled").to_pylist() == [2, 4, 6]

    def test_build_duckdb_sql_transform(self):
        """Test build_duckdb_sql_transform helper."""
        spec = build_duckdb_sql_transform("SELECT * FROM input0")

        assert spec["executor"] == "duckdb_sql@v1"
        assert spec["params"]["sql"] == "SELECT * FROM input0"


class TestRunTransform:
    """Tests for run_transform function."""

    def test_run_transform_duckdb(self):
        """Test run_transform with duckdb_sql@v1."""
        table = pa.table({"x": [1, 2, 3]})

        result = run_transform(
            "duckdb_sql@v1",
            inputs=[table],
            params={"sql": "SELECT SUM(x) as total FROM input0"},
        )

        assert result.column("total").to_pylist() == [6]

    def test_run_transform_unknown(self):
        """Test run_transform with unknown executor."""
        with pytest.raises(ValueError, match="Unknown transform"):
            run_transform("unknown@v1", inputs=[], params={})

    def test_run_transform_with_prefix(self):
        """Test run_transform strips local:// prefix."""
        table = pa.table({"x": [1]})

        result = run_transform(
            "local://duckdb_sql@v1",
            inputs=[table],
            params={"sql": "SELECT x FROM input0"},
        )

        assert result.num_rows == 1


class TestRunLocal:
    """Tests for run_local function (build_spec based execution)."""

    def test_run_local_duckdb(self):
        """Test run_local with DuckDB build_spec."""
        table_uri = "test://input"
        table = pa.table({"value": [10, 20, 30]})

        build_spec = {
            "executor": "duckdb_sql@v1",
            "params": {"sql": "SELECT AVG(value) as avg_val FROM input0"},
            "input_uris": [table_uri],
        }
        input_tables = {table_uri: table}

        result = run_local(build_spec, input_tables)

        assert result.column("avg_val").to_pylist() == [20.0]

    def test_run_local_multiple_inputs(self):
        """Test run_local with multiple inputs."""
        uri1 = "test://events"
        uri2 = "test://users"
        events = pa.table({"event_id": [1, 2], "user_id": [10, 20]})
        users = pa.table({"user_id": [10, 20], "name": ["A", "B"]})

        build_spec = {
            "executor": "duckdb_sql@v1",
            "params": {
                "sql": (
                    "SELECT e.event_id, u.name FROM input0 e JOIN input1 u ON e.user_id = u.user_id"
                )
            },
            "input_uris": [uri1, uri2],
        }
        input_tables = {uri1: events, uri2: users}

        result = run_local(build_spec, input_tables)

        assert result.num_rows == 2
        assert set(result.column("name").to_pylist()) == {"A", "B"}

    def test_run_local_missing_input(self):
        """Test run_local fails on missing input."""
        build_spec = {
            "executor": "duckdb_sql@v1",
            "params": {"sql": "SELECT 1"},
            "input_uris": ["test://missing"],
        }
        input_tables = {}

        with pytest.raises(ValueError, match="Missing input table"):
            run_local(build_spec, input_tables)

    def test_run_local_with_local_prefix(self):
        """Test run_local handles local:// prefix."""
        table = pa.table({"x": [1]})

        build_spec = {
            "executor": "local://duckdb_sql@v1",
            "params": {"sql": "SELECT x FROM input0"},
            "input_uris": ["test://table"],
        }
        input_tables = {"test://table": table}

        result = run_local(build_spec, input_tables)

        assert result.num_rows == 1

    def test_run_local_preserves_input_order(self):
        """Test that run_local preserves input order from input_uris."""
        first = pa.table({"val": [1]})
        second = pa.table({"val": [2]})

        # Dict order differs from input_uris on purpose.
        build_spec = {
            "executor": "duckdb_sql@v1",
            "params": {
                "sql": (
                    "SELECT (SELECT val FROM input0) as first, (SELECT val FROM input1) as second"
                )
            },
            "input_uris": ["uri://first", "uri://second"],
        }
        # Dict order doesn't matter; input_uris order does.
        input_tables = {"uri://second": second, "uri://first": first}

        result = run_local(build_spec, input_tables)

        assert result.column("first").to_pylist() == [1]
        assert result.column("second").to_pylist() == [2]

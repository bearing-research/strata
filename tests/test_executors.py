"""Unit tests for the embedded local executors."""

import pyarrow as pa
import pytest

from strata.transforms.base import _run_transform as run_transform


class TestDuckDBExecutor:
    def test_simple_query(self):
        result = run_transform(
            "duckdb_sql@v1",
            inputs=[],
            params={"sql": "SELECT 1 as a, 'hello' as b"},
        )
        assert result.num_rows == 1
        assert result.to_pydict() == {"a": [1], "b": ["hello"]}

    def test_single_input_table(self):
        input_table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
        result = run_transform(
            "duckdb_sql@v1",
            inputs=[input_table],
            params={"sql": "SELECT sum(value) as total FROM input0"},
        )
        assert result.to_pydict() == {"total": [60]}

    def test_multiple_input_tables(self):
        users = pa.table({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        orders = pa.table({"user_id": [1, 1, 2], "amount": [100, 200, 150]})

        result = run_transform(
            "duckdb_sql@v1",
            inputs=[users, orders],
            params={
                "sql": """
                SELECT u.name, sum(o.amount) as total
                FROM input0 u
                JOIN input1 o ON u.user_id = o.user_id
                GROUP BY u.name
                ORDER BY u.name
                """
            },
        )
        assert result.to_pydict() == {"name": ["Alice", "Bob"], "total": [300, 150]}

    def test_missing_sql_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            run_transform(
                "duckdb_sql@v1",
                inputs=[],
                params={},
            )

    def test_empty_sql_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            run_transform(
                "duckdb_sql@v1",
                inputs=[],
                params={"sql": ""},
            )

    def test_query_with_missing_table_raises(self):
        """A query referencing a missing table raises a DuckDB error."""
        import duckdb

        with pytest.raises(duckdb.CatalogException):
            run_transform(
                "duckdb_sql@v1",
                inputs=[],
                params={"sql": "SELECT * FROM input0"},
            )

    def test_filter_and_aggregate(self):
        events = pa.table(
            {
                "event_type": ["click", "view", "click", "purchase", "view"],
                "count": [1, 2, 3, 4, 5],
            }
        )
        result = run_transform(
            "duckdb_sql@v1",
            inputs=[events],
            params={
                "sql": """
                SELECT event_type, sum(count) as total
                FROM input0
                WHERE event_type != 'purchase'
                GROUP BY event_type
                ORDER BY event_type
                """
            },
        )
        assert result.to_pydict() == {
            "event_type": ["click", "view"],
            "total": [4, 7],
        }

    def test_preserves_arrow_types(self):
        input_table = pa.table(
            {
                "int_col": pa.array([1, 2, 3], type=pa.int64()),
                "float_col": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
                "str_col": pa.array(["a", "b", "c"], type=pa.string()),
            }
        )
        result = run_transform(
            "duckdb_sql@v1",
            inputs=[input_table],
            params={"sql": "SELECT * FROM input0"},
        )
        assert result.num_rows == 3
        assert "int_col" in result.column_names
        assert "float_col" in result.column_names
        assert "str_col" in result.column_names

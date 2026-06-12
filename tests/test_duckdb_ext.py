"""Tests for DuckDB integration module."""

import duckdb
import pyarrow as pa
import pytest
from strata_client.integration.duckdb import StrataScanner, StrataTableParams, register_strata_scan

from strata.types import Filter, FilterOp


class TestStrataTableParams:
    """Tests for StrataTableParams TypedDict."""

    def test_required_field(self):
        """table_uri is the only required field."""
        params: StrataTableParams = {"table_uri": "file:///warehouse#db.table"}
        assert params["table_uri"] == "file:///warehouse#db.table"

    def test_all_fields(self):
        """All optional fields can be specified."""
        params: StrataTableParams = {
            "table_uri": "file:///warehouse#db.table",
            "snapshot_id": 123456789,
            "columns": ["id", "value"],
            "filters": [Filter(column="id", op=FilterOp.GT, value=100)],
        }
        assert params["table_uri"] == "file:///warehouse#db.table"
        assert params["snapshot_id"] == 123456789
        assert params["columns"] == ["id", "value"]
        assert len(params["filters"]) == 1


class TestRegisterStrataScan:
    """Tests for register_strata_scan function."""

    def test_returns_arrow_table(self, server_with_client):
        """register_strata_scan returns the Arrow table for reference retention."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        conn = duckdb.connect(database=":memory:")
        try:
            result = register_strata_scan(
                conn=conn,
                name="test_table",
                table_uri=table_uri,
                base_url=f"http://127.0.0.1:{config.port}",
            )

            assert isinstance(result, pa.Table)
            assert result.num_rows > 0
        finally:
            conn.close()

    def test_registers_as_queryable_view(self, server_with_client):
        """Registered table can be queried via DuckDB."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        conn = duckdb.connect(database=":memory:")
        try:
            register_strata_scan(
                conn=conn,
                name="events",
                table_uri=table_uri,
                columns=["id", "value"],
                base_url=f"http://127.0.0.1:{config.port}",
            )

            result = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert result is not None
            assert result[0] > 0
        finally:
            conn.close()

    def test_column_projection(self, server_with_client):
        """Column projection limits returned columns."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        conn = duckdb.connect(database=":memory:")
        try:
            table = register_strata_scan(
                conn=conn,
                name="events",
                table_uri=table_uri,
                columns=["id", "value"],
                base_url=f"http://127.0.0.1:{config.port}",
            )

            assert table.num_columns == 2
            assert "id" in table.column_names
            assert "value" in table.column_names
            assert "name" not in table.column_names
        finally:
            conn.close()

    def test_overwrites_existing_registration(self, server_with_client):
        """Re-registering with same name overwrites."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        conn = duckdb.connect(database=":memory:")
        try:
            # First registration with all columns
            register_strata_scan(
                conn=conn,
                name="events",
                table_uri=table_uri,
                base_url=f"http://127.0.0.1:{config.port}",
            )

            # Get column count
            result1 = conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'events'"
            ).fetchone()

            # Second registration with fewer columns
            register_strata_scan(
                conn=conn,
                name="events",
                table_uri=table_uri,
                columns=["id"],
                base_url=f"http://127.0.0.1:{config.port}",
            )

            result2 = conn.execute(
                "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'events'"
            ).fetchone()
            assert result1 is not None
            assert result2 is not None

            # Second registration should have fewer columns
            assert result2[0] < result1[0]
        finally:
            conn.close()


class TestStrataScanner:
    """Tests for StrataScanner class."""

    def test_context_manager(self, server_with_client):
        """StrataScanner works as context manager."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            scanner.register("events", table_uri)
            result = scanner.query("SELECT COUNT(*) as cnt FROM events")
            assert result.num_rows == 1

    def test_method_chaining(self, server_with_client):
        """register() returns self for method chaining."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            result = scanner.register("events", table_uri)
            assert result is scanner
        finally:
            scanner.close()

    def test_registered_tables_property(self, server_with_client):
        """registered_tables returns list of registered names."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            assert scanner.registered_tables == []

            scanner.register("events", table_uri)
            assert scanner.registered_tables == ["events"]

            scanner.register("events2", table_uri)
            assert set(scanner.registered_tables) == {"events", "events2"}
        finally:
            scanner.close()

    def test_unregister(self, server_with_client):
        """unregister() removes a table."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)
            assert "events" in scanner.registered_tables

            result = scanner.unregister("events")
            assert result is scanner  # Returns self
            assert "events" not in scanner.registered_tables
        finally:
            scanner.close()

    def test_unregister_nonexistent_is_safe(self, server_with_client):
        """unregister() on nonexistent table doesn't raise."""
        config = server_with_client["config"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            # Should not raise
            scanner.unregister("nonexistent")
        finally:
            scanner.close()

    def test_replace_false_raises_on_duplicate(self, server_with_client):
        """replace=False raises ValueError on duplicate registration."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)

            with pytest.raises(ValueError, match="already registered"):
                scanner.register("events", table_uri, replace=False)
        finally:
            scanner.close()

    def test_replace_true_allows_overwrite(self, server_with_client):
        """replace=True (default) allows overwriting."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)
            # Should not raise
            scanner.register("events", table_uri, replace=True)
            assert "events" in scanner.registered_tables
        finally:
            scanner.close()

    def test_query_returns_arrow_table(self, server_with_client):
        """query() returns Arrow Table."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)
            result = scanner.query("SELECT id, value FROM events LIMIT 10")

            assert isinstance(result, pa.Table)
            assert result.num_rows <= 10
            assert "id" in result.column_names
        finally:
            scanner.close()

    def test_query_df_returns_dataframe(self, server_with_client):
        """query_df() returns pandas DataFrame."""
        pytest.importorskip("pandas")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)
            result = scanner.query_df("SELECT id, value FROM events LIMIT 10")

            # Check it's a DataFrame (without importing pandas)
            assert hasattr(result, "shape")
            assert result.shape[0] <= 10
        finally:
            scanner.close()

    def test_duckdb_filter_after_fetch(self, server_with_client):
        """DuckDB WHERE clause filters already-fetched data."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)

            all_rows = scanner.query("SELECT COUNT(*) as cnt FROM events")
            filtered_rows = scanner.query("SELECT COUNT(*) as cnt FROM events WHERE id < 50")

            # Filtered should have fewer rows
            all_count = all_rows.to_pydict()["cnt"][0]
            filtered_count = filtered_rows.to_pydict()["cnt"][0]
            assert filtered_count < all_count
        finally:
            scanner.close()

    def test_join_multiple_tables(self, server_with_client):
        """Can join multiple registered tables."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            # Register same table twice with different names
            scanner.register("t1", table_uri, columns=["id", "value"])
            scanner.register("t2", table_uri, columns=["id", "name"])

            result = scanner.query("""
                SELECT t1.id, t1.value, t2.name
                FROM t1
                JOIN t2 ON t1.id = t2.id
                LIMIT 5
            """)

            assert result.num_rows <= 5
            assert "value" in result.column_names
            assert "name" in result.column_names
        finally:
            scanner.close()

    def test_aggregation(self, server_with_client):
        """DuckDB aggregations work on registered tables."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)

            result = scanner.query("""
                SELECT
                    COUNT(*) as cnt,
                    SUM(value) as total,
                    AVG(value) as avg_val
                FROM events
            """)

            assert result.num_rows == 1
            row = result.to_pydict()
            assert row["cnt"][0] > 0
            assert row["total"][0] is not None
        finally:
            scanner.close()


class TestTableReferenceRetention:
    """Tests for Arrow table reference retention to prevent GC."""

    def test_scanner_keeps_table_references(self, server_with_client):
        """StrataScanner keeps references to prevent GC."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner.register("events", table_uri)

            # Internal _tables dict should have the reference
            assert "events" in scanner._tables
            assert isinstance(scanner._tables["events"], pa.Table)
        finally:
            scanner.close()

    def test_close_clears_references(self, server_with_client):
        """close() clears table references."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataScanner(base_url=f"http://127.0.0.1:{config.port}")
        scanner.register("events", table_uri)
        assert len(scanner._tables) == 1

        scanner.close()
        assert len(scanner._tables) == 0

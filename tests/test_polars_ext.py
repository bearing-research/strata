"""Tests for Polars integration module."""

import pyarrow as pa
import pytest
from strata_client.integration.polars import StrataPolarsScanner, scan_to_lazy, scan_to_polars


class TestScanToPolars:
    """Tests for scan_to_polars function."""

    def test_returns_polars_dataframe(self, server_with_client):
        """scan_to_polars returns a Polars DataFrame."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = scan_to_polars(
            table_uri=table_uri,
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.height > 0

    def test_column_projection(self, server_with_client):
        """Column projection limits returned columns."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = scan_to_polars(
            table_uri=table_uri,
            columns=["id", "value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["id", "value"]
        assert "name" not in result.columns

    def test_returns_expected_row_count(self, server_with_client):
        """Returns all rows from the table."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = scan_to_polars(
            table_uri=table_uri,
            base_url=f"http://127.0.0.1:{config.port}",
        )

        # temp_warehouse creates 500 rows
        assert result.height == 500


class TestScanToLazy:
    """Tests for scan_to_lazy function."""

    def test_returns_lazy_frame(self, server_with_client):
        """scan_to_lazy returns a Polars LazyFrame."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = scan_to_lazy(
            table_uri=table_uri,
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert isinstance(result, pl.LazyFrame)

    def test_lazy_operations_work(self, server_with_client):
        """Lazy operations can be chained and collected."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        lf = scan_to_lazy(
            table_uri=table_uri,
            columns=["id", "value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        # Chain lazy operations
        result = lf.filter(pl.col("id") < 100).select(pl.col("value")).collect()

        assert isinstance(result, pl.DataFrame)
        assert result.height < 500  # Filtered result


class TestStrataPolarsScanner:
    """Tests for StrataPolarsScanner class."""

    def test_context_manager(self, server_with_client):
        """StrataPolarsScanner works as context manager."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            result = scanner.scan(table_uri)
            assert isinstance(result, pl.DataFrame)
            assert result.height > 0

    def test_scan_returns_dataframe(self, server_with_client):
        """scan() returns Polars DataFrame."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            result = scanner.scan(table_uri)
            assert isinstance(result, pl.DataFrame)
        finally:
            scanner.close()

    def test_scan_with_columns(self, server_with_client):
        """scan() respects column projection."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            result = scanner.scan(table_uri, columns=["id", "name"])
            assert result.columns == ["id", "name"]
        finally:
            scanner.close()

    def test_scan_lazy_returns_lazyframe(self, server_with_client):
        """scan_lazy() returns Polars LazyFrame."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            result = scanner.scan_lazy(table_uri)
            assert isinstance(result, pl.LazyFrame)
        finally:
            scanner.close()

    def test_scan_batches_yields_record_batches(self, server_with_client):
        """scan_batches() yields Arrow RecordBatches."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            batches = list(scanner.scan_batches(table_uri))

            assert len(batches) > 0
            for batch in batches:
                assert isinstance(batch, pa.RecordBatch)
        finally:
            scanner.close()

    def test_scan_batches_total_rows(self, server_with_client):
        """scan_batches() returns all rows across batches."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            total_rows = sum(batch.num_rows for batch in scanner.scan_batches(table_uri))
            assert total_rows == 500
        finally:
            scanner.close()

    def test_scan_batches_with_columns(self, server_with_client):
        """scan_batches() respects column projection."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            batches = list(scanner.scan_batches(table_uri, columns=["id", "value"]))

            assert len(batches) > 0
            for batch in batches:
                assert batch.num_columns == 2
                assert "id" in batch.schema.names
                assert "value" in batch.schema.names
        finally:
            scanner.close()

    def test_multiple_scans_same_scanner(self, server_with_client):
        """Multiple scans can use the same scanner."""
        pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        scanner = StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}")
        try:
            # First scan
            result1 = scanner.scan(table_uri, columns=["id"])

            # Second scan with different columns
            result2 = scanner.scan(table_uri, columns=["value", "name"])

            assert result1.columns == ["id"]
            assert result2.columns == ["value", "name"]
        finally:
            scanner.close()

    def test_polars_operations_on_scanned_data(self, server_with_client):
        """Polars operations work on scanned data."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df = scanner.scan(table_uri, columns=["id", "value"])

            # Test filtering
            filtered = df.filter(pl.col("id") < 100)
            assert filtered.height < df.height

            # Test aggregation
            agg = df.select(
                pl.col("value").sum().alias("total"),
                pl.col("value").mean().alias("avg"),
            )
            assert agg.height == 1

    def test_streaming_processing_pattern(self, server_with_client):
        """Demonstrates streaming processing pattern with scan_batches."""
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPolarsScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            # Process batches incrementally (memory-efficient pattern)
            batch_counts = []
            for batch in scanner.scan_batches(table_uri, columns=["id", "value"]):
                # Convert to Polars and process each batch
                df = pl.from_arrow(batch)
                batch_counts.append(df.height)

            # Verify we processed all data
            assert sum(batch_counts) == 500

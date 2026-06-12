"""Tests for pandas integration."""

import pytest

pd = pytest.importorskip("pandas")

from strata_client.client import gt, lt  # noqa: E402
from strata_client.integration.pandas import StrataPandasScanner, scan_to_pandas  # noqa: E402


class TestScanToPandas:
    """Tests for scan_to_pandas function."""

    def test_basic_scan(self, server_with_client):
        """scan_to_pandas returns a pandas DataFrame."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(table_uri, base_url=f"http://127.0.0.1:{config.port}")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # temp_warehouse creates 500 rows

    def test_column_projection(self, server_with_client):
        """scan_to_pandas respects column projection."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["id", "value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert list(df.columns) == ["id", "value"]
        assert "name" not in df.columns

    def test_with_filters(self, server_with_client):
        """scan_to_pandas accepts filters for row-group pruning."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        # Filters are for row-group pruning, not row-level filtering
        df = scan_to_pandas(
            table_uri,
            filters=[lt("id", 100)],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        # Just verify scan succeeds with filters
        assert isinstance(df, pd.DataFrame)


class TestStrataPandasScanner:
    """Tests for StrataPandasScanner class."""

    def test_context_manager(self, server_with_client):
        """StrataPandasScanner works as context manager."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df = scanner.scan(table_uri)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 500

    def test_multiple_scans(self, server_with_client):
        """Scanner can perform multiple scans with same connection."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df1 = scanner.scan(table_uri, columns=["id"])
            df2 = scanner.scan(table_uri, columns=["value"])

            assert list(df1.columns) == ["id"]
            assert list(df2.columns) == ["value"]

    def test_scan_with_filters(self, server_with_client):
        """Scanner accepts filters for row-group pruning."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df = scanner.scan(table_uri, filters=[gt("id", 99), lt("id", 200)])
            assert isinstance(df, pd.DataFrame)

    def test_scan_batches(self, server_with_client):
        """scan_batches yields Arrow RecordBatches."""
        import pyarrow as pa

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            batches = list(scanner.scan_batches(table_uri))

            assert len(batches) > 0
            for batch in batches:
                assert isinstance(batch, pa.RecordBatch)

            total_rows = sum(b.num_rows for b in batches)
            assert total_rows == 500

    def test_scan_batches_to_pandas(self, server_with_client):
        """scan_batches can be converted to pandas incrementally."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            dfs = [
                batch.to_pandas()
                for batch in scanner.scan_batches(table_uri, columns=["id", "value"])
            ]

            # Concatenate all batches
            result = pd.concat(dfs, ignore_index=True)
            assert len(result) == 500
            assert list(result.columns) == ["id", "value"]


class TestPandasDataTypes:
    """Tests for pandas data type handling."""

    def test_integer_columns(self, server_with_client):
        """Integer columns are properly converted."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["id"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        # Should be numeric type
        assert pd.api.types.is_integer_dtype(df["id"]) or pd.api.types.is_numeric_dtype(df["id"])

    def test_float_columns(self, server_with_client):
        """Float columns are properly converted."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert pd.api.types.is_float_dtype(df["value"])

    def test_string_columns(self, server_with_client):
        """String columns are properly converted."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["name"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        # Should be object or string dtype
        assert df["name"].dtype == object or pd.api.types.is_string_dtype(df["name"])

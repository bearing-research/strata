"""Tests for pandas integration."""

import pytest

pd = pytest.importorskip("pandas")

from strata_client.client import gt, lt  # noqa: E402
from strata_client.integration.pandas import StrataPandasScanner, scan_to_pandas  # noqa: E402


class TestScanToPandas:
    def test_basic_scan(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(table_uri, base_url=f"http://127.0.0.1:{config.port}")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 500  # temp_warehouse creates 500 rows

    def test_column_projection(self, server_with_client):
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
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        # Filters prune row groups; they do not filter rows.
        df = scan_to_pandas(
            table_uri,
            filters=[lt("id", 100)],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert isinstance(df, pd.DataFrame)


class TestStrataPandasScanner:
    def test_context_manager(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df = scanner.scan(table_uri)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 500

    def test_multiple_scans(self, server_with_client):
        """One scanner performs several scans over the same connection."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df1 = scanner.scan(table_uri, columns=["id"])
            df2 = scanner.scan(table_uri, columns=["value"])

            assert list(df1.columns) == ["id"]
            assert list(df2.columns) == ["value"]

    def test_scan_with_filters(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            df = scanner.scan(table_uri, filters=[gt("id", 99), lt("id", 200)])
            assert isinstance(df, pd.DataFrame)

    def test_scan_batches(self, server_with_client):
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
        """scan_batches converts to pandas incrementally."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataPandasScanner(base_url=f"http://127.0.0.1:{config.port}") as scanner:
            dfs = [
                batch.to_pandas()
                for batch in scanner.scan_batches(table_uri, columns=["id", "value"])
            ]

            result = pd.concat(dfs, ignore_index=True)
            assert len(result) == 500
            assert list(result.columns) == ["id", "value"]


class TestPandasDataTypes:
    def test_integer_columns(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["id"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert pd.api.types.is_integer_dtype(df["id"]) or pd.api.types.is_numeric_dtype(df["id"])

    def test_float_columns(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert pd.api.types.is_float_dtype(df["value"])

    def test_string_columns(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        df = scan_to_pandas(
            table_uri,
            columns=["name"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert df["name"].dtype == object or pd.api.types.is_string_dtype(df["name"])

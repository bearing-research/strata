"""Tests for PyArrow Dataset/Scanner integration."""

import pyarrow as pa
import pytest
from strata_client.client import gt, lt
from strata_client.integration.arrow import StrataDataset, dataset


class TestStrataDataset:
    def test_context_manager(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}") as ds:
            table = ds.to_table()
            assert isinstance(table, pa.Table)
            assert table.num_rows > 0

    def test_table_uri_property(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            assert ds.table_uri == table_uri
        finally:
            ds.close()

    def test_schema_property(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            schema = ds.schema
            assert isinstance(schema, pa.Schema)
            assert "id" in schema.names
            assert "value" in schema.names
        finally:
            ds.close()

    def test_to_table(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            table = ds.to_table()
            assert isinstance(table, pa.Table)
            assert table.num_rows == 500  # temp_warehouse creates 500 rows
        finally:
            ds.close()

    def test_to_table_with_columns(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            table = ds.to_table(columns=["id", "value"])
            assert table.num_columns == 2
            assert "id" in table.column_names
            assert "value" in table.column_names
            assert "name" not in table.column_names
        finally:
            ds.close()

    def test_to_batches(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            batches = list(ds.to_batches())
            assert len(batches) > 0
            for batch in batches:
                assert isinstance(batch, pa.RecordBatch)

            total_rows = sum(b.num_rows for b in batches)
            assert total_rows == 500
        finally:
            ds.close()

    def test_count_rows(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            count = ds.count_rows()
            assert count == 500
        finally:
            ds.close()

    def test_head(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            table = ds.head(10)
            assert isinstance(table, pa.Table)
            assert table.num_rows == 10
        finally:
            ds.close()

    def test_head_with_columns(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            table = ds.head(5, columns=["id"])
            assert table.num_rows == 5
            assert table.num_columns == 1
            assert table.column_names == ["id"]
        finally:
            ds.close()


class TestStrataScanner:
    def test_scanner_to_batches(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner = ds.scanner()
            batches = list(scanner.to_batches())
            assert len(batches) > 0
            assert all(isinstance(b, pa.RecordBatch) for b in batches)
        finally:
            ds.close()

    def test_scanner_to_table(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner = ds.scanner(columns=["id", "value"])
            table = scanner.to_table()
            assert isinstance(table, pa.Table)
            assert table.num_columns == 2
        finally:
            ds.close()

    def test_scanner_to_reader(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner = ds.scanner()
            reader = scanner.to_reader()
            assert isinstance(reader, pa.RecordBatchReader)

            table = reader.read_all()
            assert table.num_rows == 500
        finally:
            ds.close()

    def test_scanner_with_filter(self, server_with_client):
        """Filters prune row groups by min/max stats, not rows; a filter matching the range keeps
        all.
        """
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            # The filter is passed to the server for pruning.
            scanner = ds.scanner(filter=lt("id", 100))
            table = scanner.to_table()
            # Row-group pruning doesn't filter individual rows, so only check the scan works.
            assert isinstance(table, pa.Table)
        finally:
            ds.close()

    def test_scanner_with_multiple_filters(self, server_with_client):
        """Multiple filters are combined for row-group pruning."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            # Multiple filters are passed for pruning
            scanner = ds.scanner(filter=[gt("id", 99), lt("id", 200)])
            table = scanner.to_table()
            assert isinstance(table, pa.Table)
        finally:
            ds.close()

    def test_scanner_count_rows(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner = ds.scanner()
            count = scanner.count_rows()
            assert count == 500
        finally:
            ds.close()

    def test_scanner_head(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            scanner = ds.scanner()
            table = scanner.head(20)
            assert table.num_rows == 20
        finally:
            ds.close()


class TestDatasetFunction:
    def test_dataset_function(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = dataset(table_uri, base_url=f"http://127.0.0.1:{config.port}")
        try:
            assert isinstance(ds, StrataDataset)
            assert ds.table_uri == table_uri
        finally:
            ds.close()

    def test_dataset_with_snapshot_id(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ds = dataset(table_uri, snapshot_id=12345, base_url=f"http://127.0.0.1:{config.port}")
        try:
            assert ds.snapshot_id == 12345
        finally:
            ds.close()


class TestIntegrationWithOtherLibraries:
    """Integration patterns with other libraries."""

    def test_reader_to_polars(self, server_with_client):
        pl = pytest.importorskip("polars")

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}") as ds:
            reader = ds.scanner(columns=["id", "value"]).to_reader()

            df = pl.from_arrow(reader.read_all())
            assert df.height == 500
            assert df.columns == ["id", "value"]

    def test_reader_to_duckdb(self, server_with_client):
        import duckdb

        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataset(table_uri, base_url=f"http://127.0.0.1:{config.port}") as ds:
            table = ds.scanner(columns=["id", "value"]).to_table()

            conn = duckdb.connect()
            conn.register("events", table)

            result = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            assert result is not None
            assert result[0] == 500

            conn.close()

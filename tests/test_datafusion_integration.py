"""Tests for DataFusion integration."""

import pytest

datafusion = pytest.importorskip("datafusion")

from strata_client.client import gt, lt  # noqa: E402
from strata_client.integration.datafusion import (  # noqa: E402
    StrataDataFusionContext,
    register_strata_table,
    strata_query,
)


class TestRegisterStrataTable:
    def test_basic_registration(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ctx = register_strata_table(
            "events",
            table_uri,
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert "events" in ctx.catalog().schema("public").table_names()

        result = ctx.sql("SELECT COUNT(*) as cnt FROM events").collect()
        assert len(result) == 1
        assert result[0].column("cnt")[0].as_py() == 500

    def test_with_column_projection(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ctx = register_strata_table(
            "events",
            table_uri,
            columns=["id", "value"],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        result = ctx.sql("SELECT id, value FROM events LIMIT 5").collect()
        assert len(result) == 1
        assert result[0].num_rows == 5

    def test_with_existing_context(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ctx = datafusion.SessionContext()
        assert len(ctx.catalog().schema("public").table_names()) == 0

        returned_ctx = register_strata_table(
            "events",
            table_uri,
            ctx=ctx,
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert returned_ctx is ctx
        assert "events" in ctx.catalog().schema("public").table_names()

    def test_with_filters(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        ctx = register_strata_table(
            "events",
            table_uri,
            filters=[lt("id", 100)],
            base_url=f"http://127.0.0.1:{config.port}",
        )

        result = ctx.sql("SELECT * FROM events").collect()
        assert len(result) > 0


class TestStrataQuery:
    def test_single_table_query(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = strata_query(
            "SELECT id, value FROM events WHERE id < 10 ORDER BY id",
            tables={"events": table_uri},
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert len(result) > 0
        total_rows = sum(batch.num_rows for batch in result)
        assert total_rows == 10

    def test_with_column_projection(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = strata_query(
            "SELECT id FROM events LIMIT 5",
            tables={"events": table_uri},
            columns={"events": ["id", "value"]},
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert len(result) > 0

    def test_aggregation_query(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        result = strata_query(
            "SELECT COUNT(*) as cnt, AVG(value) as avg_val FROM events",
            tables={"events": table_uri},
            base_url=f"http://127.0.0.1:{config.port}",
        )

        assert len(result) == 1
        assert result[0].column("cnt")[0].as_py() == 500


class TestStrataDataFusionContext:
    def test_context_manager(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)
            result = ctx.sql("SELECT COUNT(*) as cnt FROM events").collect()
            assert result[0].column("cnt")[0].as_py() == 500

    def test_multiple_table_registration(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events1", table_uri, columns=["id"])
            ctx.register("events2", table_uri, columns=["value"])

            assert "events1" in ctx.tables()
            assert "events2" in ctx.tables()

    def test_method_chaining(self, server_with_client):
        """register() returns self."""
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            result = ctx.register("events", table_uri).sql("SELECT * FROM events LIMIT 5").collect()
            assert len(result) > 0

    def test_table_method(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)
            df = ctx.table("events")

            assert hasattr(df, "select")
            assert hasattr(df, "filter")
            assert hasattr(df, "collect")

    def test_deregister(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)
            assert "events" in ctx.tables()

            ctx.deregister("events")
            assert "events" not in ctx.tables()

    def test_with_filters(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register(
                "events",
                table_uri,
                filters=[gt("id", 99), lt("id", 200)],
            )

            result = ctx.sql("SELECT * FROM events").collect()
            assert len(result) > 0


class TestDataFusionDataFrameAPI:
    def test_select(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)
            result = ctx.table("events").select("id", "value").limit(5).collect()

            assert len(result) > 0
            batch = result[0]
            assert "id" in batch.schema.names
            assert "value" in batch.schema.names

    def test_filter(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)

            result = ctx.table("events").filter(datafusion.col("id") < datafusion.lit(10)).collect()

            total_rows = sum(batch.num_rows for batch in result)
            assert total_rows == 10

    def test_aggregate(self, server_with_client):
        config = server_with_client["config"]
        table_uri = server_with_client["warehouse"]["table_uri"]

        with StrataDataFusionContext(base_url=f"http://127.0.0.1:{config.port}") as ctx:
            ctx.register("events", table_uri)

            result = (
                ctx.table("events")
                .aggregate([], [datafusion.functions.count(datafusion.col("id"))])
                .collect()
            )

            assert len(result) == 1

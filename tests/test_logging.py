"""Tests for structured logging with correlation IDs."""

import json
from typing import cast


class TestRequestContext:
    def test_request_context_sets_and_clears(self):
        from strata.logging import RequestContext, get_request_context

        assert get_request_context() == {}

        with RequestContext(request_id="test-123", scan_id="scan-456"):
            ctx = get_request_context()
            assert ctx["request_id"] == "test-123"
            assert ctx["scan_id"] == "scan-456"

        assert get_request_context() == {}

    def test_nested_request_contexts(self):
        """Nested contexts restore the parent context."""
        from strata.logging import RequestContext, get_request_context

        with RequestContext(request_id="outer"):
            assert get_request_context()["request_id"] == "outer"

            with RequestContext(scan_id="inner-scan"):
                ctx = get_request_context()
                assert ctx["request_id"] == "outer"
                assert ctx["scan_id"] == "inner-scan"

            ctx = get_request_context()
            assert ctx["request_id"] == "outer"
            assert "scan_id" not in ctx

    def test_generate_request_id(self):
        from strata.logging import generate_request_id

        id1 = generate_request_id()
        id2 = generate_request_id()

        assert len(id1) == 16
        assert len(id2) == 16
        assert id1 != id2


class TestStructuredFormatter:
    def test_json_format_basic(self):
        import io
        import logging

        from strata.logging import StructuredFormatter

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(include_timestamp=False))

        logger = logging.getLogger("test.json.basic")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info("Test message")

        output = stream.getvalue()
        log_entry = json.loads(output.strip())

        assert log_entry["level"] == "info"
        assert log_entry["message"] == "Test message"
        assert log_entry["logger"] == "test.json.basic"

    def test_json_format_with_context(self):
        import io
        import logging

        from strata.logging import RequestContext, StructuredFormatter

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(include_timestamp=False))

        logger = logging.getLogger("test.json.context")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        with RequestContext(request_id="req-123", scan_id="scan-456"):
            logger.info("With context")

        output = stream.getvalue()
        log_entry = json.loads(output.strip())

        assert log_entry["request_id"] == "req-123"
        assert log_entry["scan_id"] == "scan-456"


class TestStructuredLogger:
    def test_logger_with_structured_data(self):
        """The logger accepts structured data as kwargs."""
        import io
        import logging

        from strata.logging import StructuredFormatter, StructuredLogger

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter(include_timestamp=False))

        logging.setLoggerClass(StructuredLogger)
        logger = cast(StructuredLogger, logging.getLogger("test.structured.data"))
        logger.handlers = [handler]
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info("Row group fetched", file_path="/data/file.parquet", rows=1000)

        output = stream.getvalue()
        log_entry = json.loads(output.strip())

        assert log_entry["message"] == "Row group fetched"
        assert log_entry["file_path"] == "/data/file.parquet"
        assert log_entry["rows"] == 1000


class TestTraceContextIntegration:
    """OpenTelemetry trace context integration."""

    def test_trace_context_when_disabled(self, monkeypatch):
        """Trace context is empty when tracing is disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.logging import get_trace_context

        ctx = get_trace_context()
        assert ctx == {}

    def test_trace_context_when_enabled(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "true")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.logging import get_trace_context
        from strata.tracing import init_tracing, trace_span

        init_tracing()

        with trace_span("test_span"):
            ctx = get_trace_context()
            if ctx:  # Only if OTel is installed
                assert "trace_id" in ctx
                assert "span_id" in ctx
                assert len(ctx["trace_id"]) == 32  # 128 bits in hex
                assert len(ctx["span_id"]) == 16  # 64 bits in hex


class TestMiddleware:
    def test_middleware_sets_request_id(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from strata.logging import get_request_context, request_context_middleware

        app = FastAPI()
        app.middleware("http")(request_context_middleware)

        @app.get("/test")
        async def endpoint():
            ctx = get_request_context()
            return ctx

        client = TestClient(app)
        response = client.get("/test")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 16

        data = response.json()
        assert "request_id" in data
        assert data["request_id"] == response.headers["X-Request-ID"]

    def test_middleware_uses_provided_request_id(self):
        """The middleware uses the X-Request-ID header when provided."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from strata.logging import get_request_context, request_context_middleware

        app = FastAPI()
        app.middleware("http")(request_context_middleware)

        @app.get("/test")
        async def endpoint():
            ctx = get_request_context()
            return ctx

        client = TestClient(app)
        response = client.get("/test", headers={"X-Request-ID": "custom-id-123"})

        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "custom-id-123"
        data = response.json()
        assert data["request_id"] == "custom-id-123"


class TestScanMetricsCorrelation:
    def test_scan_metrics_includes_request_id(self):
        from strata.metrics import ScanMetrics

        metrics = ScanMetrics(
            scan_id="scan-123",
            snapshot_id=12345,
            request_id="req-abc",
        )

        data = metrics.to_dict()
        assert data["request_id"] == "req-abc"

    def test_scan_metrics_excludes_empty_request_id(self):
        from strata.metrics import ScanMetrics

        metrics = ScanMetrics(
            scan_id="scan-123",
            snapshot_id=12345,
        )

        data = metrics.to_dict()
        assert "request_id" not in data

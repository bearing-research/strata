"""Tests for OpenTelemetry tracing integration."""

import pytest


class TestTracingModule:
    """Tests for the tracing module when OTel is not installed."""

    def test_is_tracing_available_returns_bool(self):
        """Test that is_tracing_available returns a boolean."""
        from strata.tracing import is_tracing_available

        result = is_tracing_available()
        assert isinstance(result, bool)

    def test_is_tracing_enabled_returns_bool(self):
        """Test that is_tracing_enabled returns a boolean."""
        from strata.tracing import is_tracing_enabled

        result = is_tracing_enabled()
        assert isinstance(result, bool)

    def test_get_tracer_returns_none_when_disabled(self, monkeypatch):
        """Test that get_tracer returns None when tracing is disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        # Reset module state
        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import get_tracer

        result = get_tracer()
        assert result is None

    def test_trace_span_yields_noop_span_when_disabled(self, monkeypatch):
        """Test that trace_span yields a NoOpSpan when tracing is disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        # Reset module state
        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import NoOpSpan, trace_span

        with trace_span("test_operation", attr1="value1") as span:
            assert isinstance(span, NoOpSpan)
            # NoOpSpan methods should be no-ops (not raise)
            span.set_attribute("key", "value")
            span.add_event("event_name")
            span.record_exception(ValueError("test"))

    def test_noop_span_methods_are_silent(self):
        """Test that NoOpSpan methods don't raise exceptions."""
        from strata.tracing import NoOpSpan

        span = NoOpSpan()
        # All methods should be no-ops
        span.set_attribute("key", "value")
        span.set_attribute("int_key", 42)
        span.set_attribute("float_key", 3.14)
        span.add_event("event", {"attr": "value"})
        span.record_exception(RuntimeError("test error"))
        span.set_status("OK")

    def test_init_tracing_returns_false_when_disabled(self, monkeypatch):
        """Test that init_tracing returns False when tracing is disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        # Reset module state
        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import init_tracing

        result = init_tracing()
        assert result is False

    def test_instrument_fastapi_is_silent_when_disabled(self, monkeypatch):
        """Test that instrument_fastapi doesn't raise when tracing is disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        from fastapi import FastAPI

        from strata.tracing import instrument_fastapi

        app = FastAPI()
        # Should not raise
        instrument_fastapi(app)


class TestTracingContextManager:
    """Tests for trace_span context manager behavior."""

    def test_trace_span_propagates_exceptions(self, monkeypatch):
        """Test that exceptions are propagated from trace_span."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        # Reset module state
        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import trace_span

        with pytest.raises(ValueError, match="test error"):
            with trace_span("failing_operation"):
                raise ValueError("test error")

    def test_trace_span_with_attributes(self, monkeypatch):
        """Test that trace_span accepts initial attributes."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        # Reset module state
        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import NoOpSpan, trace_span

        with trace_span(
            "operation",
            table_id="test.table",
            snapshot_id=12345,
            columns_count=5,
        ) as span:
            assert isinstance(span, NoOpSpan)


def _is_otel_available() -> bool:
    """Check if OpenTelemetry is installed."""
    try:
        import opentelemetry.trace  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_otel_available(), reason="OpenTelemetry not installed")
class TestTracingWithOTelEnabled:
    """Tests for tracing when OpenTelemetry is installed and enabled."""

    @pytest.fixture
    def reset_tracing(self, monkeypatch):
        """Reset tracing state and enable tracing."""
        import strata.tracing

        # Store original state
        original_tracer = strata.tracing._tracer
        original_initialized = strata.tracing._initialized

        # Reset state
        strata.tracing._tracer = None
        strata.tracing._initialized = False

        # Enable tracing
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "true")

        yield

        # Restore state
        strata.tracing._tracer = original_tracer
        strata.tracing._initialized = original_initialized

    def test_is_tracing_available_returns_true(self):
        """Test that is_tracing_available returns True when OTel is installed."""
        from strata.tracing import is_tracing_available

        # OTel should be installed in test environment with extras
        assert is_tracing_available() is True

    def test_is_tracing_enabled_returns_true_when_enabled(self, reset_tracing):
        """Test that is_tracing_enabled returns True when OTel is installed and enabled."""
        from strata.tracing import is_tracing_enabled

        assert is_tracing_enabled() is True

    def test_init_tracing_returns_true(self, reset_tracing):
        """Test that init_tracing returns True when OTel is installed."""
        from strata.tracing import init_tracing

        result = init_tracing()
        assert result is True

    def test_get_tracer_returns_tracer(self, reset_tracing):
        """Test that get_tracer returns a real Tracer when enabled."""
        from opentelemetry.trace import Tracer

        from strata.tracing import get_tracer, init_tracing

        init_tracing()
        tracer = get_tracer()
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_trace_span_yields_real_span(self, reset_tracing):
        """Test that trace_span yields a real OTel Span when enabled."""
        from opentelemetry.trace import Span

        from strata.tracing import NoOpSpan, init_tracing, trace_span

        init_tracing()

        with trace_span("test_operation", key="value") as span:
            assert not isinstance(span, NoOpSpan)
            assert isinstance(span, Span)
            # Real span methods should work
            span.set_attribute("dynamic_attr", 42)
            span.add_event("test_event", {"event_key": "event_value"})

    def test_trace_span_records_exception_on_error(self, reset_tracing):
        """Test that trace_span records exceptions when they occur."""
        from strata.tracing import init_tracing, trace_span

        init_tracing()

        with pytest.raises(RuntimeError, match="test exception"):
            with trace_span("failing_op"):
                raise RuntimeError("test exception")

    def test_trace_span_with_in_memory_exporter(self, monkeypatch):
        """Test that spans are actually captured using in-memory exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        import strata.tracing

        # Reset module state
        strata.tracing._tracer = None
        strata.tracing._initialized = False
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "true")

        # Set up in-memory exporter with a fresh provider
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Create tracer directly from this provider (don't use global)
        tracer = provider.get_tracer("strata", "0.1.0")

        # Inject the tracer directly
        strata.tracing._tracer = tracer
        strata.tracing._initialized = True

        from strata.tracing import trace_span

        # Create a span
        with trace_span("test_operation", table_id="ns.table") as span:
            span.set_attribute("rows_count", 100)

        # Verify span was captured
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test_operation"
        assert spans[0].attributes["table_id"] == "ns.table"
        assert spans[0].attributes["rows_count"] == 100

    def test_instrument_fastapi_works_when_enabled(self, reset_tracing):
        """Test that instrument_fastapi instruments the app when enabled."""
        from fastapi import FastAPI

        from strata.tracing import init_tracing, instrument_fastapi

        init_tracing()
        app = FastAPI()

        # Should not raise
        instrument_fastapi(app)


class TestTracingIntegration:
    """Integration tests for tracing in server components."""

    def test_server_starts_with_tracing_disabled(self, tmp_path, monkeypatch):
        """Test that server starts correctly with tracing disabled."""
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        from strata.config import StrataConfig
        from strata.server import ServerState

        config = StrataConfig(cache_dir=tmp_path / "cache")
        state = ServerState(config)

        # Server state should be initialized
        assert state.config == config
        assert state.planner is not None
        assert state.fetcher is not None

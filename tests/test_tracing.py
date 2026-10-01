"""Tests for OpenTelemetry tracing integration."""

import pytest


class TestTracingModule:
    """The tracing module when OTel is not installed."""

    def test_is_tracing_available_returns_bool(self):
        from strata.tracing import is_tracing_available

        result = is_tracing_available()
        assert isinstance(result, bool)

    def test_is_tracing_enabled_returns_bool(self):
        from strata.tracing import is_tracing_enabled

        result = is_tracing_enabled()
        assert isinstance(result, bool)

    def test_get_tracer_returns_none_when_disabled(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import get_tracer

        result = get_tracer()
        assert result is None

    def test_trace_span_yields_noop_span_when_disabled(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import NoOpSpan, trace_span

        with trace_span("test_operation", attr1="value1") as span:
            assert isinstance(span, NoOpSpan)
            # NoOpSpan methods must not raise.
            span.set_attribute("key", "value")
            span.add_event("event_name")
            span.record_exception(ValueError("test"))

    def test_noop_span_methods_are_silent(self):
        """NoOpSpan methods do not raise."""
        from strata.tracing import NoOpSpan

        span = NoOpSpan()
        span.set_attribute("key", "value")
        span.set_attribute("int_key", 42)
        span.set_attribute("float_key", 3.14)
        span.add_event("event", {"attr": "value"})
        span.record_exception(RuntimeError("test error"))
        span.set_status("OK")

    def test_init_tracing_returns_false_when_disabled(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import init_tracing

        result = init_tracing()
        assert result is False

    def test_instrument_fastapi_is_silent_when_disabled(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        from fastapi import FastAPI

        from strata.tracing import instrument_fastapi

        app = FastAPI()
        instrument_fastapi(app)


class TestTracingContextManager:
    """trace_span context manager behavior."""

    def test_trace_span_propagates_exceptions(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        from strata.tracing import trace_span

        with pytest.raises(ValueError, match="test error"):
            with trace_span("failing_operation"):
                raise ValueError("test error")

    def test_trace_span_with_attributes(self, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

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
    try:
        import opentelemetry.trace  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _is_otel_available(), reason="OpenTelemetry not installed")
class TestTracingWithOTelEnabled:
    """Tracing when OpenTelemetry is installed and enabled."""

    @pytest.fixture
    def reset_tracing(self, monkeypatch):
        """Reset tracing state and enable tracing."""
        import strata.tracing

        original_tracer = strata.tracing._tracer
        original_initialized = strata.tracing._initialized

        strata.tracing._tracer = None
        strata.tracing._initialized = False

        monkeypatch.setenv("STRATA_TRACING_ENABLED", "true")

        yield

        strata.tracing._tracer = original_tracer
        strata.tracing._initialized = original_initialized

    def test_is_tracing_available_returns_true(self):
        from strata.tracing import is_tracing_available

        # OTel is installed in the test environment with extras.
        assert is_tracing_available() is True

    def test_is_tracing_enabled_returns_true_when_enabled(self, reset_tracing):
        from strata.tracing import is_tracing_enabled

        assert is_tracing_enabled() is True

    def test_init_tracing_returns_true(self, reset_tracing):
        from strata.tracing import init_tracing

        result = init_tracing()
        assert result is True

    def test_get_tracer_returns_tracer(self, reset_tracing):
        from opentelemetry.trace import Tracer

        from strata.tracing import get_tracer, init_tracing

        init_tracing()
        tracer = get_tracer()
        assert tracer is not None
        assert isinstance(tracer, Tracer)

    def test_trace_span_yields_real_span(self, reset_tracing):
        from opentelemetry.trace import Span

        from strata.tracing import NoOpSpan, init_tracing, trace_span

        init_tracing()

        with trace_span("test_operation", key="value") as span:
            assert not isinstance(span, NoOpSpan)
            assert isinstance(span, Span)
            span.set_attribute("dynamic_attr", 42)
            span.add_event("test_event", {"event_key": "event_value"})

    def test_trace_span_records_exception_on_error(self, reset_tracing):
        from strata.tracing import init_tracing, trace_span

        init_tracing()

        with pytest.raises(RuntimeError, match="test exception"):
            with trace_span("failing_op"):
                raise RuntimeError("test exception")

    def test_trace_span_with_in_memory_exporter(self, monkeypatch):
        """Spans are actually captured by an in-memory exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        import strata.tracing

        strata.tracing._tracer = None
        strata.tracing._initialized = False
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "true")

        # In-memory exporter with a fresh provider.
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # Tracer straight from this provider, not the global one.
        tracer = provider.get_tracer("strata", "0.1.0")

        strata.tracing._tracer = tracer
        strata.tracing._initialized = True

        from strata.tracing import trace_span

        with trace_span("test_operation", table_id="ns.table") as span:
            span.set_attribute("rows_count", 100)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "test_operation"
        assert spans[0].attributes["table_id"] == "ns.table"
        assert spans[0].attributes["rows_count"] == 100

    def test_instrument_fastapi_works_when_enabled(self, reset_tracing):
        from fastapi import FastAPI

        from strata.tracing import init_tracing, instrument_fastapi

        init_tracing()
        app = FastAPI()

        instrument_fastapi(app)


class TestTracingIntegration:
    """Tracing in server components."""

    def test_server_starts_with_tracing_disabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

        from strata.config import StrataConfig
        from strata.server import ServerState

        config = StrataConfig(cache_dir=tmp_path / "cache")
        state = ServerState(config)

        assert state.config == config
        assert state.planner is not None
        assert state.fetcher is not None

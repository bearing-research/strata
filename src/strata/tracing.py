"""OpenTelemetry tracing for Strata.

This module provides distributed tracing integration using OpenTelemetry.
Tracing is optional and only enabled when:
1. The 'otel' extras are installed: pip install strata-notebook[otel]
2. OTEL_EXPORTER_OTLP_ENDPOINT is set (or tracing is explicitly configured)

Usage:
    from strata.tracing import get_tracer, trace_span

    tracer = get_tracer()
    with trace_span("my_operation", table_id="ns.table") as span:
        # ... do work ...
        span.set_attribute("rows_returned", 1000)

Environment variables (standard OpenTelemetry):
    OTEL_EXPORTER_OTLP_ENDPOINT - OTLP endpoint (e.g., http://localhost:4317)
    OTEL_SERVICE_NAME - Service name (default: strata)
    OTEL_TRACES_SAMPLER - Sampler type (default: parentbased_always_on)
    OTEL_TRACES_SAMPLER_ARG - Sampler argument (e.g., 0.1 for 10% sampling)

Strata-specific:
    STRATA_TRACING_ENABLED - Set to "false" to disable tracing even if OTel is installed
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

# Check if OpenTelemetry is available
_OTEL_AVAILABLE = False
try:
    from opentelemetry.trace import Span, Tracer  # noqa: F401

    _OTEL_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

# Module-level state
_tracer: "Tracer | None" = None
_initialized = False


def is_tracing_available() -> bool:
    """Check if OpenTelemetry is installed."""
    return _OTEL_AVAILABLE


def is_tracing_enabled() -> bool:
    """Check if tracing is both available and enabled."""
    if not _OTEL_AVAILABLE:
        return False
    # Check explicit disable
    if os.environ.get("STRATA_TRACING_ENABLED", "true").lower() == "false":
        return False
    return True


def init_tracing(
    service_name: str = "strata",
    otlp_endpoint: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing.

    This should be called once at server startup. If OpenTelemetry is not
    installed or tracing is disabled, this is a no-op.

    Args:
        service_name: Name of the service (default: strata)
        otlp_endpoint: OTLP endpoint URL. If None, uses OTEL_EXPORTER_OTLP_ENDPOINT

    Returns:
        True if tracing was initialized, False otherwise
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer is not None

    _initialized = True

    if not is_tracing_enabled():
        return False

    # Get endpoint from env if not provided
    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        # Create resource with service name
        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
                "service.version": "0.2.0",
            }
        )

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if endpoint is configured
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("strata", "0.2.0")

        return True

    except Exception:
        # Silently fail - tracing is optional
        return False


def get_tracer() -> "Tracer | None":
    """Get the configured tracer, or None if tracing is not available."""
    global _tracer

    if not is_tracing_enabled():
        return None

    if _tracer is None and not _initialized:
        # Auto-initialize on first use
        init_tracing()

    return _tracer


class NoOpSpan:
    """A no-op span for when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: Any, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: dict | None = None) -> None:
        pass


@contextmanager
def trace_span(
    name: str,
    **attributes: Any,
) -> Iterator["Span | NoOpSpan"]:
    """Create a traced span with attributes.

    This is a convenience wrapper that handles the case where tracing
    is not available. When tracing is disabled, yields a no-op span.

    Args:
        name: Span name (e.g., "plan_scan", "fetch_row_group")
        **attributes: Initial span attributes

    Yields:
        The span (or a no-op span if tracing is disabled)

    Example:
        with trace_span("fetch_row_group", file_path=path, row_group_id=0) as span:
            data = fetch(path, 0)
            span.set_attribute("bytes_read", len(data))
    """
    tracer = get_tracer()

    if tracer is None:
        yield NoOpSpan()
        return

    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI app with OpenTelemetry.

    This adds automatic tracing for all HTTP endpoints including:
    - Request/response timing
    - HTTP method, path, status code
    - Request headers (configurable)

    Args:
        app: FastAPI application instance
    """
    if not is_tracing_enabled():
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        # Silently fail - tracing is optional
        pass

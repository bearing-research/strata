"""Optional OpenTelemetry tracing.

Active when the ``[otel]`` extra is installed and ``STRATA_TRACING_ENABLED`` is not
``false``; spans are exported only when an OTLP endpoint is set. The standard ``OTEL_*``
variables apply.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

_OTEL_AVAILABLE = False
try:
    from opentelemetry.trace import Span, Tracer  # noqa: F401

    _OTEL_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

_tracer: "Tracer | None" = None
_initialized = False


def is_tracing_available() -> bool:
    """Check if OpenTelemetry is installed."""
    return _OTEL_AVAILABLE


def is_tracing_enabled() -> bool:
    """Check if tracing is both available and enabled."""
    if not _OTEL_AVAILABLE:
        return False
    if os.environ.get("STRATA_TRACING_ENABLED", "true").lower() == "false":
        return False
    return True


def init_tracing(
    service_name: str = "strata",
    otlp_endpoint: str | None = None,
) -> bool:
    """Initialize OpenTelemetry tracing once at startup; return whether it is active.

    No-op when OpenTelemetry is missing or tracing is disabled. ``otlp_endpoint`` defaults to
    ``OTEL_EXPORTER_OTLP_ENDPOINT``; with neither, spans are recorded but not exported.
    """
    global _tracer, _initialized

    if _initialized:
        return _tracer is not None

    _initialized = True

    if not is_tracing_enabled():
        return False

    endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from strata.health import _package_version

        resource = Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
                "service.version": _package_version(),
            }
        )

        provider = TracerProvider(resource=resource)

        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("strata", _package_version())

        return True

    except Exception:
        # Tracing is optional; never fail the caller.
        return False


def get_tracer() -> "Tracer | None":
    """Get the configured tracer, or None if tracing is not available."""
    global _tracer

    if not is_tracing_enabled():
        return None

    if _tracer is None and not _initialized:
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
    """Yield a span with *attributes*, or a no-op span when tracing is off.

    An exception raised in the block is recorded on the span and re-raised.
    """
    tracer = get_tracer()

    if tracer is None:
        yield NoOpSpan()
        return

    from opentelemetry.trace import Status, StatusCode

    with tracer.start_as_current_span(name) as span:
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
    """Instrument a FastAPI app with OpenTelemetry HTTP tracing."""
    if not is_tracing_enabled():
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception:
        # Tracing is optional; never fail the caller.
        pass


# W3C trace context, carried across process boundaries.

TRACE_CONTEXT_KEYS = ("traceparent", "tracestate")


def current_trace_context() -> dict[str, str]:
    """The active span's W3C ``traceparent`` / ``tracestate`` for an outgoing request or manifest.

    Empty when tracing is off or no span is active, so callers can merge it unconditionally.
    """
    if get_tracer() is None:
        return {}
    from opentelemetry.propagate import inject

    carrier: dict[str, str] = {}
    inject(carrier)
    return {key: carrier[key] for key in TRACE_CONTEXT_KEYS if carrier.get(key)}


@contextmanager
def trace_span_from(
    name: str,
    carrier: dict[str, Any] | None,
    **attributes: Any,
) -> Iterator["Span | NoOpSpan"]:
    """A span whose parent is the trace context another process sent.

    ``carrier`` holds ``traceparent`` (and optionally ``tracestate``); without
    one the span starts a trace of its own, as ``trace_span`` does.
    """
    tracer = get_tracer()
    if tracer is None:
        yield NoOpSpan()
        return

    from opentelemetry.propagate import extract
    from opentelemetry.trace import Status, StatusCode

    context = extract(
        {key: str(carrier[key]) for key in TRACE_CONTEXT_KEYS if carrier and carrier.get(key)}
    )
    with tracer.start_as_current_span(name, context=context) as span:
        for key, value in attributes.items():
            if value is not None:
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise

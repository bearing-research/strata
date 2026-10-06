"""Structured logging with request-scoped correlation IDs.

Log entries carry the request (or build) context and, when tracing is on, the
OpenTelemetry ``trace_id``/``span_id``. ``STRATA_LOG_LEVEL`` (default INFO) and
``STRATA_LOG_FORMAT`` (``json`` or ``text``, default json) configure output.
"""

import contextvars
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import TracebackType
from typing import cast

from strata.json_types import JsonObject, JsonValue

type ExcInfo = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)

_request_context: contextvars.ContextVar[JsonObject] = contextvars.ContextVar(
    "request_context", default={}
)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex[:16]


def get_request_context() -> JsonObject:
    """Get the current request context."""
    return _request_context.get()


def set_request_context(**kwargs: JsonValue) -> contextvars.Token[JsonObject]:
    """Merge values into the request context; returns the token for reset."""
    current = _request_context.get().copy()
    current.update(kwargs)
    return _request_context.set(current)


def clear_request_context() -> None:
    """Clear all request context."""
    _request_context.set({})


@contextmanager
def RequestContext(**kwargs: JsonValue) -> Iterator[None]:
    """Context manager that adds its keyword arguments to every log entry inside it."""
    token = set_request_context(**kwargs)
    try:
        yield
    finally:
        _request_context.reset(token)


@contextmanager
def BuildContext(
    build_id: str,
    tenant_id: str | None = None,
    transform_ref: str | None = None,
    provenance_hash: str | None = None,
    **kwargs: JsonValue,
) -> Iterator[None]:
    """Context manager that adds build identity to every log entry inside it.

    Takes ``build_id``, ``tenant_id``, ``transform_ref``, ``provenance_hash`` and
    any extra keyword fields.
    """
    ctx: JsonObject = {"build_id": build_id}
    if tenant_id:
        ctx["tenant_id"] = tenant_id
    if transform_ref:
        ctx["transform_ref"] = transform_ref
    if provenance_hash:
        ctx["provenance_hash"] = provenance_hash
    ctx.update(kwargs)

    token = set_request_context(**ctx)
    try:
        yield
    finally:
        _request_context.reset(token)


def get_trace_context() -> dict[str, str]:
    """Return ``trace_id`` and ``span_id`` of the active span, or ``{}`` without tracing."""
    try:
        from strata.tracing import is_tracing_enabled

        if not is_tracing_enabled():
            return {}

        from opentelemetry import trace

        span = trace.get_current_span()
        if span is None:
            return {}

        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return {}

        return {
            "trace_id": format(ctx.trace_id, "032x"),
            "span_id": format(ctx.span_id, "016x"),
        }
    except ImportError:
        return {}


def _normalize_exc_info(
    exc_info: ExcInfo,
) -> (
    tuple[type[BaseException], BaseException, TracebackType | None] | tuple[None, None, None] | None
):
    """Normalize logging exc_info into the stdlib makeRecord shape."""
    if exc_info is True:
        return cast(
            tuple[type[BaseException], BaseException, TracebackType | None]
            | tuple[None, None, None],
            sys.exc_info(),
        )
    if exc_info in (False, None):
        return None
    if isinstance(exc_info, BaseException):
        return (type(exc_info), exc_info, exc_info.__traceback__)
    return exc_info


class StructuredFormatter(logging.Formatter):
    """JSON formatter that includes request context and trace IDs."""

    def __init__(self, include_timestamp: bool = True):
        super().__init__()
        self.include_timestamp = include_timestamp

    def format(self, record: logging.LogRecord) -> str:
        log_entry: JsonObject = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if self.include_timestamp:
            log_entry["timestamp"] = time.time()

        ctx = get_request_context()
        if ctx:
            log_entry.update(ctx)

        trace_ctx = get_trace_context()
        if trace_ctx:
            log_entry.update(trace_ctx)

        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict):
            log_entry.update(cast(JsonObject, structured_data))

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        if record.levelno >= logging.WARNING:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_entry)


class TextFormatter(logging.Formatter):
    """Human-readable text formatter with context for development."""

    def format(self, record: logging.LogRecord) -> str:
        ctx_parts = []

        ctx = get_request_context()
        request_id = ctx.get("request_id")
        if isinstance(request_id, str):
            ctx_parts.append(f"req={request_id[:8]}")
        scan_id = ctx.get("scan_id")
        if isinstance(scan_id, str):
            ctx_parts.append(f"scan={scan_id[:8]}")
        build_id = ctx.get("build_id")
        if isinstance(build_id, str):
            ctx_parts.append(f"build={build_id[:8]}")
        tenant_id = ctx.get("tenant_id")
        if isinstance(tenant_id, str):
            ctx_parts.append(f"tenant={tenant_id}")
        transform_ref = ctx.get("transform_ref")
        if isinstance(transform_ref, str):
            ctx_parts.append(f"transform={transform_ref}")

        trace_ctx = get_trace_context()
        if "trace_id" in trace_ctx:
            ctx_parts.append(f"trace={trace_ctx['trace_id'][:8]}")

        ctx_str = f"[{' '.join(ctx_parts)}] " if ctx_parts else ""

        data_str = ""
        structured_data = getattr(record, "structured_data", None)
        if isinstance(structured_data, dict) and structured_data:
            data_parts = [f"{k}={v}" for k, v in structured_data.items()]
            data_str = " | " + ", ".join(data_parts)

        return f"{record.levelname:7} {ctx_str}{record.name}: {record.getMessage()}{data_str}"


class StructuredLogger(logging.Logger):
    """Logger that takes structured fields as keyword arguments (``logger.info(msg, rows=n)``)."""

    def _log_with_data(
        self,
        level: int,
        msg: object,
        args: tuple[object, ...],
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 2,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        """Log with keyword fields attached as structured data."""
        if self.isEnabledFor(level):
            normalized_exc_info = _normalize_exc_info(exc_info)
            record = self.makeRecord(
                self.name,
                level,
                "(unknown file)",
                0,
                msg,
                args,
                normalized_exc_info,
                func=None,
                extra=None,
                sinfo=None,
            )
            structured_data: JsonObject = {}
            if extra is not None:
                structured_data.update(cast(JsonObject, dict(extra)))
            structured_data.update(kwargs)
            if structured_data:
                setattr(record, "structured_data", structured_data)
            self.handle(record)

    def debug(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.DEBUG,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

    def info(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.INFO,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

    def warning(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.WARNING,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

    def error(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.ERROR,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

    def critical(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.CRITICAL,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )

    def exception(
        self,
        msg: object,
        *args: object,
        exc_info: ExcInfo = True,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, object] | None = None,
        **kwargs: JsonValue,
    ) -> None:
        self._log_with_data(
            logging.ERROR,
            msg,
            args,
            exc_info=exc_info,
            stack_info=stack_info,
            stacklevel=stacklevel,
            extra=extra,
            **kwargs,
        )


logging.setLoggerClass(StructuredLogger)

_configured = False
_log_format = os.environ.get("STRATA_LOG_FORMAT", "json").lower()


def configure_logging(
    level: str | None = None,
    format: str | None = None,
) -> None:
    """Configure the ``strata`` and uvicorn loggers to write to stderr; idempotent.

    ``level`` and ``format`` default to ``STRATA_LOG_LEVEL`` (INFO) and
    ``STRATA_LOG_FORMAT`` (json).
    """
    global _configured, _log_format

    if _configured:
        return

    level = level or os.environ.get("STRATA_LOG_LEVEL", "INFO").upper()
    format = format or os.environ.get("STRATA_LOG_FORMAT", "json").lower()
    _log_format = format

    handler = logging.StreamHandler(sys.stderr)

    if format == "text":
        handler.setFormatter(TextFormatter())
    else:
        handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger("strata")
    root_logger.setLevel(getattr(logging, level, logging.INFO))
    root_logger.addHandler(handler)
    root_logger.propagate = False

    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers = [handler]

    _configured = True


def log_format() -> str:
    """The configured log format, ``json`` or ``text``."""
    return _log_format


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger, configuring logging on first use."""
    if not _configured:
        configure_logging()

    return cast(StructuredLogger, logging.getLogger(name))


async def request_context_middleware(request, call_next):
    """Bind ``request_id``, method, path and tenant to the log context for each request.

    ``request_id`` comes from ``X-Request-ID`` or is generated, and is echoed in the
    response header.
    """
    # Lazy import to avoid a circular dependency.
    from strata.tenant import get_tenant_id

    request_id = request.headers.get("X-Request-ID") or generate_request_id()

    # Set by the tenant middleware: registered later, so it wraps this one and runs first.
    tenant_id = get_tenant_id()

    token = set_request_context(
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        tenant_id=tenant_id,
    )

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _request_context.reset(token)

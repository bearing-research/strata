"""HTTP request counts by route template and status class, and request durations.

Rendered by ``/metrics/prometheus``. Labels stay bounded: a route is its template, never the
raw path, and a method outside the standard set is ``other``.
"""

from __future__ import annotations

import bisect
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from fastapi.routing import APIRoute
from starlette.routing import BaseRoute, Match, Mount, Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

UNMATCHED_ROUTE = "unmatched"
"""The route label of a request answered before routing (a refused Host, auth, the rate
limiter) or matching no route."""

DURATION_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    300.0,
)
"""Upper bounds in seconds; streams and builds run long, hence the tail past 10 s."""

_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


@dataclass
class DurationHistogram:
    """Per-bucket (non-cumulative) counts plus the sum, for one route."""

    bucket_counts: list[int] = field(default_factory=lambda: [0] * (len(DURATION_BUCKETS) + 1))
    total_seconds: float = 0.0
    count: int = 0


@dataclass
class RequestMetrics:
    """Request counters and duration histograms, safe to update from any thread."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _requests: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _durations: dict[str, DurationHistogram] = field(default_factory=dict)

    def record(self, method: str, route: str, status_code: int, seconds: float) -> None:
        """Count one finished request and add its duration to the route's histogram."""
        method = method if method in _METHODS else "other"
        key = (method, route, f"{status_code // 100}xx")
        with self._lock:
            self._requests[key] = self._requests.get(key, 0) + 1
            histogram = self._durations.setdefault(route, DurationHistogram())
            histogram.bucket_counts[bisect.bisect_left(DURATION_BUCKETS, seconds)] += 1
            histogram.total_seconds += seconds
            histogram.count += 1

    def snapshot(self) -> tuple[dict[tuple[str, str, str], int], dict[str, DurationHistogram]]:
        """Copies of the counters and histograms, consistent with each other."""
        with self._lock:
            durations = {
                route: DurationHistogram(list(h.bucket_counts), h.total_seconds, h.count)
                for route, h in self._durations.items()
            }
            return dict(self._requests), durations


_request_metrics = RequestMetrics()


def get_request_metrics() -> RequestMetrics:
    """The process-wide request metrics."""
    return _request_metrics


def _route_template(scope: Scope, match_scope: Scope, routes: Sequence[BaseRoute]) -> str:
    """The template of the route that served *scope*, or ``UNMATCHED_ROUTE``.

    The router leaves an API route in the scope. Plain Starlette routes and mounts (the docs
    pages, ``/mcp``, the frontend's ``/assets``) leave none, or a mounted app's own inner
    route, so those are matched again against the scope as it arrived.
    """
    route = scope.get("route")
    if isinstance(route, APIRoute) and scope.get("root_path", "") == match_scope["root_path"]:
        return route.path_format
    for candidate in routes:
        if isinstance(candidate, (Route, Mount)):
            match, _ = candidate.matches(match_scope)
            if match is not Match.NONE:
                return candidate.path_format
    return UNMATCHED_ROUTE


class RequestMetricsMiddleware:
    """Count and time every HTTP request.

    The duration runs until the app returns, so a streamed response is timed to its last
    chunk. A request that raised before its response started counts as a 5xx (the server
    answers it 500); one that raised after keeps the status already sent.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Routing mutates the scope (a mount moves root_path), so keep what it arrived as.
        match_scope = {
            "type": "http",
            "path": scope["path"],
            "root_path": scope.get("root_path", ""),
            "method": scope["method"],
        }
        # A mounted Starlette app replaces scope["app"] with itself.
        routes = getattr(scope.get("app"), "routes", ())
        status_code: int | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _request_metrics.record(
                scope["method"],
                _route_template(scope, match_scope, routes),
                status_code if status_code is not None else 500,
                time.perf_counter() - start,
            )

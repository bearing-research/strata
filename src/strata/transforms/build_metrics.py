"""Build metrics: lifecycle counts, duration and queue-wait percentiles, bytes in/out.

Broken down per tenant and per transform, and exported in Prometheus format.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class BuildEvent:
    """Single build event for metrics tracking."""

    build_id: str
    tenant_id: str | None
    transform_ref: str
    event_type: str  # "started", "succeeded", "failed", "cancelled"
    timestamp: float
    duration_ms: float | None = None  # Only for completed events
    queue_wait_ms: float | None = None
    bytes_in: int = 0
    bytes_out: int = 0
    error_code: str | None = None


@dataclass
class TransformStats:
    """Aggregated stats per transform reference."""

    transform_ref: str
    started: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    total_duration_ms: float = 0.0
    total_bytes_in: int = 0
    total_bytes_out: int = 0
    durations: deque = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self, duration_ms: float, bytes_in: int, bytes_out: int) -> None:
        self.succeeded += 1
        self.total_duration_ms += duration_ms
        self.total_bytes_in += bytes_in
        self.total_bytes_out += bytes_out
        self.durations.append(duration_ms)

    def record_failure(self, duration_ms: float) -> None:
        self.failed += 1
        self.total_duration_ms += duration_ms
        self.durations.append(duration_ms)

    def get_duration_percentiles(self) -> dict[str, float | None]:
        """Get p50, p95, p99 duration in ms."""
        if not self.durations:
            return {"p50_ms": None, "p95_ms": None, "p99_ms": None}

        sorted_durations = sorted(self.durations)
        n = len(sorted_durations)

        def pct(p: float) -> float:
            idx = max(0, int(n * p) - 1)
            return round(sorted_durations[idx], 2)

        return {
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
        }


@dataclass
class TenantBuildStats:
    """Aggregated stats per tenant."""

    tenant_id: str
    started: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    total_bytes_in: int = 0
    total_bytes_out: int = 0
    total_duration_ms: float = 0.0


class BuildMetricsCollector:
    """Thread-safe collector of build events, aggregated overall, per transform and per tenant.

    Per-transform and per-tenant tables are bounded; the oldest entry is evicted.
    """

    def __init__(self, max_events: int = 1000, max_transforms: int = 100, max_tenants: int = 100):
        self._lock = threading.RLock()  # Reentrant: locked methods call each other

        self._builds_started = 0
        self._builds_succeeded = 0
        self._builds_failed = 0
        self._builds_cancelled = 0

        self._total_duration_ms = 0.0
        self._durations: deque[float] = deque(maxlen=1000)

        self._total_queue_wait_ms = 0.0
        self._queue_wait_count = 0
        self._queue_waits: deque[float] = deque(maxlen=1000)

        self._total_bytes_in = 0
        self._total_bytes_out = 0

        # Bounded; the oldest entry is evicted first.
        self._transform_stats: dict[str, TransformStats] = {}
        self._max_transforms = max_transforms

        self._tenant_stats: dict[str, TenantBuildStats] = {}
        self._max_tenants = max_tenants

        self._recent_events: deque[BuildEvent] = deque(maxlen=max_events)

        self._error_codes: dict[str, int] = {}

    def record_started(
        self,
        build_id: str,
        tenant_id: str | None,
        transform_ref: str,
        queue_wait_ms: float | None = None,
    ) -> None:
        """Record a build start event."""
        with self._lock:
            self._builds_started += 1

            if queue_wait_ms is not None:
                self._total_queue_wait_ms += queue_wait_ms
                self._queue_wait_count += 1
                self._queue_waits.append(queue_wait_ms)

            if transform_ref not in self._transform_stats:
                if len(self._transform_stats) >= self._max_transforms:
                    oldest = next(iter(self._transform_stats))
                    del self._transform_stats[oldest]
                self._transform_stats[transform_ref] = TransformStats(transform_ref=transform_ref)
            self._transform_stats[transform_ref].started += 1

            tenant_key = tenant_id or "__default__"
            if tenant_key not in self._tenant_stats:
                if len(self._tenant_stats) >= self._max_tenants:
                    oldest = next(iter(self._tenant_stats))
                    del self._tenant_stats[oldest]
                self._tenant_stats[tenant_key] = TenantBuildStats(tenant_id=tenant_key)
            self._tenant_stats[tenant_key].started += 1

            event = BuildEvent(
                build_id=build_id,
                tenant_id=tenant_id,
                transform_ref=transform_ref,
                event_type="started",
                timestamp=time.time(),
                queue_wait_ms=queue_wait_ms,
            )
            self._recent_events.append(event)

    def record_succeeded(
        self,
        build_id: str,
        tenant_id: str | None,
        transform_ref: str,
        duration_ms: float,
        bytes_in: int = 0,
        bytes_out: int = 0,
    ) -> None:
        """Record a successful build completion."""
        with self._lock:
            self._builds_succeeded += 1
            self._total_duration_ms += duration_ms
            self._durations.append(duration_ms)
            self._total_bytes_in += bytes_in
            self._total_bytes_out += bytes_out

            if transform_ref in self._transform_stats:
                self._transform_stats[transform_ref].record_success(
                    duration_ms, bytes_in, bytes_out
                )

            tenant_key = tenant_id or "__default__"
            if tenant_key in self._tenant_stats:
                stats = self._tenant_stats[tenant_key]
                stats.succeeded += 1
                stats.total_bytes_in += bytes_in
                stats.total_bytes_out += bytes_out
                stats.total_duration_ms += duration_ms

            event = BuildEvent(
                build_id=build_id,
                tenant_id=tenant_id,
                transform_ref=transform_ref,
                event_type="succeeded",
                timestamp=time.time(),
                duration_ms=duration_ms,
                bytes_in=bytes_in,
                bytes_out=bytes_out,
            )
            self._recent_events.append(event)

    def record_failed(
        self,
        build_id: str,
        tenant_id: str | None,
        transform_ref: str,
        duration_ms: float,
        error_code: str | None = None,
    ) -> None:
        """Record a failed build."""
        with self._lock:
            self._builds_failed += 1
            self._total_duration_ms += duration_ms
            self._durations.append(duration_ms)

            if error_code:
                self._error_codes[error_code] = self._error_codes.get(error_code, 0) + 1

            if transform_ref in self._transform_stats:
                self._transform_stats[transform_ref].record_failure(duration_ms)

            tenant_key = tenant_id or "__default__"
            if tenant_key in self._tenant_stats:
                self._tenant_stats[tenant_key].failed += 1
                self._tenant_stats[tenant_key].total_duration_ms += duration_ms

            event = BuildEvent(
                build_id=build_id,
                tenant_id=tenant_id,
                transform_ref=transform_ref,
                event_type="failed",
                timestamp=time.time(),
                duration_ms=duration_ms,
                error_code=error_code,
            )
            self._recent_events.append(event)

    def record_cancelled(
        self,
        build_id: str,
        tenant_id: str | None,
        transform_ref: str,
    ) -> None:
        """Record a cancelled build."""
        with self._lock:
            self._builds_cancelled += 1

            if transform_ref in self._transform_stats:
                self._transform_stats[transform_ref].cancelled += 1

            tenant_key = tenant_id or "__default__"
            if tenant_key in self._tenant_stats:
                self._tenant_stats[tenant_key].cancelled += 1

            event = BuildEvent(
                build_id=build_id,
                tenant_id=tenant_id,
                transform_ref=transform_ref,
                event_type="cancelled",
                timestamp=time.time(),
            )
            self._recent_events.append(event)

    def get_duration_percentiles(self) -> dict[str, float | None]:
        """Get global duration percentiles (p50, p95, p99)."""
        with self._lock:
            if not self._durations:
                return {"p50_ms": None, "p95_ms": None, "p99_ms": None}

            sorted_durations = sorted(self._durations)
            n = len(sorted_durations)

            def pct(p: float) -> float:
                idx = max(0, int(n * p) - 1)
                return round(sorted_durations[idx], 2)

            return {
                "p50_ms": pct(0.50),
                "p95_ms": pct(0.95),
                "p99_ms": pct(0.99),
            }

    def get_queue_wait_percentiles(self) -> dict[str, float | None]:
        """Get queue wait percentiles (p50, p95, p99)."""
        with self._lock:
            if not self._queue_waits:
                return {"p50_ms": None, "p95_ms": None, "p99_ms": None}

            sorted_waits = sorted(self._queue_waits)
            n = len(sorted_waits)

            def pct(p: float) -> float:
                idx = max(0, int(n * p) - 1)
                return round(sorted_waits[idx], 2)

            return {
                "p50_ms": pct(0.50),
                "p95_ms": pct(0.95),
                "p99_ms": pct(0.99),
            }

    def get_stats(self) -> dict:
        """Get aggregate build statistics."""
        with self._lock:
            avg_duration = (
                self._total_duration_ms / (self._builds_succeeded + self._builds_failed)
                if (self._builds_succeeded + self._builds_failed) > 0
                else 0.0
            )
            avg_queue_wait = (
                self._total_queue_wait_ms / self._queue_wait_count
                if self._queue_wait_count > 0
                else 0.0
            )

            return {
                "builds_started": self._builds_started,
                "builds_succeeded": self._builds_succeeded,
                "builds_failed": self._builds_failed,
                "builds_cancelled": self._builds_cancelled,
                "builds_in_flight": self._builds_started
                - self._builds_succeeded
                - self._builds_failed
                - self._builds_cancelled,
                "total_duration_ms": round(self._total_duration_ms, 2),
                "avg_duration_ms": round(avg_duration, 2),
                "duration_percentiles": self.get_duration_percentiles(),
                "avg_queue_wait_ms": round(avg_queue_wait, 2),
                "queue_wait_percentiles": self.get_queue_wait_percentiles(),
                "total_bytes_in": self._total_bytes_in,
                "total_bytes_out": self._total_bytes_out,
                "error_codes": dict(self._error_codes),
                "transforms_tracked": len(self._transform_stats),
                "tenants_tracked": len(self._tenant_stats),
            }

    def get_transform_stats(self, transform_ref: str) -> dict | None:
        """Get stats for a specific transform."""
        with self._lock:
            stats = self._transform_stats.get(transform_ref)
            if stats is None:
                return None

            return {
                "transform_ref": stats.transform_ref,
                "started": stats.started,
                "succeeded": stats.succeeded,
                "failed": stats.failed,
                "cancelled": stats.cancelled,
                "total_bytes_in": stats.total_bytes_in,
                "total_bytes_out": stats.total_bytes_out,
                "duration_percentiles": stats.get_duration_percentiles(),
            }

    def get_all_transform_stats(self) -> list[dict]:
        """Get stats for all tracked transforms."""
        with self._lock:
            return [
                {
                    "transform_ref": stats.transform_ref,
                    "started": stats.started,
                    "succeeded": stats.succeeded,
                    "failed": stats.failed,
                    "cancelled": stats.cancelled,
                    "total_bytes_in": stats.total_bytes_in,
                    "total_bytes_out": stats.total_bytes_out,
                    "duration_percentiles": stats.get_duration_percentiles(),
                }
                for stats in self._transform_stats.values()
            ]

    def get_tenant_stats(self, tenant_id: str) -> dict | None:
        """Get stats for a specific tenant."""
        with self._lock:
            stats = self._tenant_stats.get(tenant_id)
            if stats is None:
                return None

            return {
                "tenant_id": stats.tenant_id,
                "started": stats.started,
                "succeeded": stats.succeeded,
                "failed": stats.failed,
                "cancelled": stats.cancelled,
                "total_bytes_in": stats.total_bytes_in,
                "total_bytes_out": stats.total_bytes_out,
                "total_duration_ms": round(stats.total_duration_ms, 2),
            }

    def get_all_tenant_stats(self) -> list[dict]:
        """Get stats for all tracked tenants."""
        with self._lock:
            return [
                {
                    "tenant_id": stats.tenant_id,
                    "started": stats.started,
                    "succeeded": stats.succeeded,
                    "failed": stats.failed,
                    "cancelled": stats.cancelled,
                    "total_bytes_in": stats.total_bytes_in,
                    "total_bytes_out": stats.total_bytes_out,
                    "total_duration_ms": round(stats.total_duration_ms, 2),
                }
                for stats in self._tenant_stats.values()
            ]

    def get_recent_events(self, limit: int = 20) -> list[dict]:
        """Get recent build events for debugging."""
        with self._lock:
            events = list(self._recent_events)[-limit:]
            return [
                {
                    "build_id": e.build_id,
                    "tenant_id": e.tenant_id,
                    "transform_ref": e.transform_ref,
                    "event_type": e.event_type,
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                    "queue_wait_ms": e.queue_wait_ms,
                    "bytes_in": e.bytes_in,
                    "bytes_out": e.bytes_out,
                    "error_code": e.error_code,
                }
                for e in events
            ]

    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics text for builds."""
        lines = []

        with self._lock:
            lines.append("# HELP strata_builds_started_total Total builds started")
            lines.append("# TYPE strata_builds_started_total counter")
            lines.append(f"strata_builds_started_total {self._builds_started}")

            lines.append("# HELP strata_builds_succeeded_total Total builds succeeded")
            lines.append("# TYPE strata_builds_succeeded_total counter")
            lines.append(f"strata_builds_succeeded_total {self._builds_succeeded}")

            lines.append("# HELP strata_builds_failed_total Total builds failed")
            lines.append("# TYPE strata_builds_failed_total counter")
            lines.append(f"strata_builds_failed_total {self._builds_failed}")

            lines.append("# HELP strata_builds_cancelled_total Total builds cancelled")
            lines.append("# TYPE strata_builds_cancelled_total counter")
            lines.append(f"strata_builds_cancelled_total {self._builds_cancelled}")

            in_flight = (
                self._builds_started
                - self._builds_succeeded
                - self._builds_failed
                - self._builds_cancelled
            )
            lines.append("# HELP strata_builds_in_flight Current builds in progress")
            lines.append("# TYPE strata_builds_in_flight gauge")
            lines.append(f"strata_builds_in_flight {in_flight}")

            lines.append("# HELP strata_builds_duration_total_ms Total build duration in ms")
            lines.append("# TYPE strata_builds_duration_total_ms counter")
            lines.append(f"strata_builds_duration_total_ms {round(self._total_duration_ms, 2)}")

            duration_pcts = self.get_duration_percentiles()
            if duration_pcts["p50_ms"] is not None:
                lines.append("# HELP strata_builds_duration_p50_ms Build duration p50")
                lines.append("# TYPE strata_builds_duration_p50_ms gauge")
                lines.append(f"strata_builds_duration_p50_ms {duration_pcts['p50_ms']}")

                lines.append("# HELP strata_builds_duration_p95_ms Build duration p95")
                lines.append("# TYPE strata_builds_duration_p95_ms gauge")
                lines.append(f"strata_builds_duration_p95_ms {duration_pcts['p95_ms']}")

                lines.append("# HELP strata_builds_duration_p99_ms Build duration p99")
                lines.append("# TYPE strata_builds_duration_p99_ms gauge")
                lines.append(f"strata_builds_duration_p99_ms {duration_pcts['p99_ms']}")

            lines.append("# HELP strata_builds_queue_wait_total_ms Total queue wait time in ms")
            lines.append("# TYPE strata_builds_queue_wait_total_ms counter")
            lines.append(f"strata_builds_queue_wait_total_ms {round(self._total_queue_wait_ms, 2)}")

            queue_pcts = self.get_queue_wait_percentiles()
            if queue_pcts["p50_ms"] is not None:
                lines.append("# HELP strata_builds_queue_wait_p95_ms Queue wait p95")
                lines.append("# TYPE strata_builds_queue_wait_p95_ms gauge")
                lines.append(f"strata_builds_queue_wait_p95_ms {queue_pcts['p95_ms']}")

            lines.append("# HELP strata_builds_bytes_in_total Total input bytes processed")
            lines.append("# TYPE strata_builds_bytes_in_total counter")
            lines.append(f"strata_builds_bytes_in_total {self._total_bytes_in}")

            lines.append("# HELP strata_builds_bytes_out_total Total output bytes produced")
            lines.append("# TYPE strata_builds_bytes_out_total counter")
            lines.append(f"strata_builds_bytes_out_total {self._total_bytes_out}")

            transform_list = sorted(
                self._transform_stats.values(),
                key=lambda s: s.started,
                reverse=True,
            )[:20]

            if transform_list:
                lines.append(
                    "# HELP strata_build_transform_started_total Builds started by transform"
                )
                lines.append("# TYPE strata_build_transform_started_total counter")
                for stats in transform_list:
                    ref = stats.transform_ref.replace('"', '\\"')
                    lines.append(
                        f'strata_build_transform_started_total{{transform="{ref}"}} {stats.started}'
                    )

                lines.append(
                    "# HELP strata_build_transform_succeeded_total Builds succeeded by transform"
                )
                lines.append("# TYPE strata_build_transform_succeeded_total counter")
                for stats in transform_list:
                    ref = stats.transform_ref.replace('"', '\\"')
                    val = stats.succeeded
                    lines.append(
                        f'strata_build_transform_succeeded_total{{transform="{ref}"}} {val}'
                    )

                lines.append(
                    "# HELP strata_build_transform_failed_total Builds failed by transform"
                )
                lines.append("# TYPE strata_build_transform_failed_total counter")
                for stats in transform_list:
                    ref = stats.transform_ref.replace('"', '\\"')
                    lines.append(
                        f'strata_build_transform_failed_total{{transform="{ref}"}} {stats.failed}'
                    )

            tenant_list = sorted(
                self._tenant_stats.values(),
                key=lambda s: s.started,
                reverse=True,
            )[:20]

            if tenant_list:
                lines.append("# HELP strata_build_tenant_started_total Builds started by tenant")
                lines.append("# TYPE strata_build_tenant_started_total counter")
                for stats in tenant_list:
                    tenant = stats.tenant_id.replace('"', '\\"')
                    lines.append(
                        f'strata_build_tenant_started_total{{tenant="{tenant}"}} {stats.started}'
                    )

                lines.append("# HELP strata_build_tenant_bytes_out_total Output bytes by tenant")
                lines.append("# TYPE strata_build_tenant_bytes_out_total counter")
                for stats in tenant_list:
                    tenant = stats.tenant_id.replace('"', '\\"')
                    val = stats.total_bytes_out
                    lines.append(f'strata_build_tenant_bytes_out_total{{tenant="{tenant}"}} {val}')

            if self._error_codes:
                lines.append("# HELP strata_builds_errors_total Build errors by code")
                lines.append("# TYPE strata_builds_errors_total counter")
                for code, count in sorted(self._error_codes.items())[:20]:
                    code_escaped = code.replace('"', '\\"')
                    lines.append(
                        f'strata_builds_errors_total{{error_code="{code_escaped}"}} {count}'
                    )

        return "\n".join(lines)


_build_metrics: BuildMetricsCollector | None = None


def get_build_metrics() -> BuildMetricsCollector | None:
    """Get the build metrics collector singleton."""
    return _build_metrics


def init_build_metrics() -> BuildMetricsCollector:
    """Initialize and return the build metrics collector."""
    global _build_metrics
    if _build_metrics is None:
        _build_metrics = BuildMetricsCollector()
    return _build_metrics


def reset_build_metrics() -> None:
    """Reset the build metrics collector (for testing)."""
    global _build_metrics
    _build_metrics = None

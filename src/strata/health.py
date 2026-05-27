"""Health check utilities for dependency status monitoring.

Provides comprehensive health checks for all server dependencies:
- Disk cache (accessibility, space)
- Metadata store (SQLite connectivity)
- S3 connectivity (if configured)
- Memory pressure
- Thread pool saturation
"""

import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TypedDict

import pyarrow as pa

from strata.json_types import JsonObject


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyCheckDict(TypedDict, total=False):
    """Serialized dependency check payload."""

    name: str
    status: str
    latency_ms: float
    message: str
    details: JsonObject


class HealthSummaryDict(TypedDict):
    """Serialized health summary payload."""

    total: int
    healthy: int
    degraded: int
    unhealthy: int


class HealthReportDict(TypedDict):
    """Serialized top-level health report payload."""

    status: str
    timestamp: float
    version: str
    checks: list[DependencyCheckDict]
    summary: HealthSummaryDict


@dataclass
class DependencyCheck:
    """Result of a single dependency health check."""

    name: str
    status: HealthStatus
    latency_ms: float
    message: str | None = None
    details: JsonObject = field(default_factory=dict)

    def to_dict(self) -> DependencyCheckDict:
        result: DependencyCheckDict = {
            "name": self.name,
            "status": self.status.value,
            "latency_ms": round(self.latency_ms, 2),
        }
        if self.message:
            result["message"] = self.message
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class HealthReport:
    """Comprehensive health report for all dependencies."""

    status: HealthStatus
    checks: list[DependencyCheck]
    timestamp: float
    version: str = "0.2.0"

    def to_dict(self) -> HealthReportDict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "version": self.version,
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
            },
        }


def check_disk_cache(cache_dir: Path, max_size_bytes: int) -> DependencyCheck:
    """Check disk cache health."""
    start = time.perf_counter()

    try:
        # Check directory exists and is writable
        if not cache_dir.exists():
            return DependencyCheck(
                name="disk_cache",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Cache directory does not exist",
            )

        # Check we can write a test file
        test_file = cache_dir / ".health_check"
        try:
            test_file.write_text("health_check")
            test_file.unlink()
        except Exception as e:
            return DependencyCheck(
                name="disk_cache",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Cache directory not writable: {e}",
            )

        # Check available disk space. shutil.disk_usage is cross-platform
        # (Windows uses GetDiskFreeSpaceEx; POSIX uses statvfs). Prefer it
        # over os.statvfs, which doesn't exist on Windows.
        usage = shutil.disk_usage(cache_dir)
        available_bytes = usage.free
        total_bytes = usage.total
        usage_percent = ((total_bytes - available_bytes) / total_bytes) * 100

        details: JsonObject = {
            "path": str(cache_dir),
            "available_bytes": available_bytes,
            "total_bytes": total_bytes,
            "usage_percent": round(usage_percent, 1),
            "max_cache_bytes": max_size_bytes,
        }

        # Degraded if disk is >90% full
        if usage_percent > 90:
            return DependencyCheck(
                name="disk_cache",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Disk usage at {usage_percent:.1f}%",
                details=details,
            )

        return DependencyCheck(
            name="disk_cache",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details=details,
        )

    except Exception as e:
        return DependencyCheck(
            name="disk_cache",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def check_metadata_store(cache_dir: Path) -> DependencyCheck:
    """Check metadata store (SQLite) health."""
    start = time.perf_counter()

    try:
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store(cache_dir)
        stats = store.stats()

        return DependencyCheck(
            name="metadata_store",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details={
                "parquet_meta_entries": stats.get("parquet_meta_entries", 0),
                "manifest_entries": stats.get("manifest_entries", 0),
            },
        )

    except Exception as e:
        return DependencyCheck(
            name="metadata_store",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def check_arrow_memory() -> DependencyCheck:
    """Check Arrow memory pool health."""
    start = time.perf_counter()

    try:
        pool = pa.default_memory_pool()
        bytes_allocated = pool.bytes_allocated()
        max_memory = pool.max_memory()

        details: JsonObject = {
            "backend": pool.backend_name,
            "bytes_allocated": bytes_allocated,
            "max_memory": max_memory,
        }

        # Degraded if using >80% of max observed memory
        if max_memory > 0 and bytes_allocated > max_memory * 0.8:
            return DependencyCheck(
                name="arrow_memory",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.perf_counter() - start) * 1000,
                message="High Arrow memory usage",
                details=details,
            )

        return DependencyCheck(
            name="arrow_memory",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details=details,
        )

    except Exception as e:
        return DependencyCheck(
            name="arrow_memory",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def check_thread_pools(planning_executor, fetch_executor) -> DependencyCheck:
    """Check thread pool health."""
    start = time.perf_counter()

    try:
        from strata.pool_metrics import get_pool_tracker

        tracker = get_pool_tracker()
        summary = tracker.get_summary()

        pools = summary.get("thread_pools", {})
        total_active = 0
        total_max = 0
        pool_details: JsonObject = {}

        for name, stats in pools.items():
            active = stats.get("active_workers", 0)
            max_workers = stats.get("max_workers", 1)
            total_active += active
            total_max += max_workers
            pool_details[name] = {
                "active": active,
                "max": max_workers,
                "utilization": round(stats.get("utilization_pct", 0), 2),
            }

        overall_utilization = (total_active / total_max * 100) if total_max > 0 else 0

        details: JsonObject = {
            "pools": pool_details,
            "overall_utilization": round(overall_utilization, 1),
        }

        # Degraded if >90% utilized
        if overall_utilization > 90:
            return DependencyCheck(
                name="thread_pools",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Thread pools at {overall_utilization:.1f}% utilization",
                details=details,
            )

        return DependencyCheck(
            name="thread_pools",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details=details,
        )

    except Exception as e:
        return DependencyCheck(
            name="thread_pools",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def check_rate_limiter() -> DependencyCheck:
    """Check rate limiter health."""
    start = time.perf_counter()

    try:
        from strata.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()
        if limiter is None:
            return DependencyCheck(
                name="rate_limiter",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Rate limiter not initialized",
                details={"enabled": False},
            )

        stats = limiter.get_stats()

        # Check rejection rate
        total = stats.get("total_requests", 0)
        rejected = (
            stats.get("rejected_global", 0)
            + stats.get("rejected_client", 0)
            + stats.get("rejected_endpoint", 0)
        )
        rejection_rate = (rejected / total * 100) if total > 0 else 0

        details: JsonObject = {
            "enabled": stats.get("enabled", False),
            "active_clients": stats.get("active_clients", 0),
            "total_requests": total,
            "rejected_requests": rejected,
            "rejection_rate": round(rejection_rate, 2),
        }

        # Degraded if >10% rejection rate
        if rejection_rate > 10:
            return DependencyCheck(
                name="rate_limiter",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"High rejection rate: {rejection_rate:.1f}%",
                details=details,
            )

        return DependencyCheck(
            name="rate_limiter",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details=details,
        )

    except Exception as e:
        return DependencyCheck(
            name="rate_limiter",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def check_cache_evictions() -> DependencyCheck:
    """Check cache eviction pressure."""
    start = time.perf_counter()

    try:
        from strata.cache_metrics import get_eviction_tracker

        tracker = get_eviction_tracker()
        stats = tracker.get_stats()

        details: JsonObject = {
            "total_evictions": stats.total_evictions,
            "evictions_last_hour": stats.evictions_last_hour,
            "eviction_rate_per_minute": stats.eviction_rate_per_minute,
            "pressure_level": stats.pressure_level,
        }

        if stats.pressure_level == "critical":
            return DependencyCheck(
                name="cache_evictions",
                status=HealthStatus.UNHEALTHY,
                latency_ms=(time.perf_counter() - start) * 1000,
                message="Cache thrashing - critical eviction rate",
                details=details,
            )
        elif stats.pressure_level in ("high", "medium"):
            return DependencyCheck(
                name="cache_evictions",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"Elevated eviction pressure: {stats.pressure_level}",
                details=details,
            )

        return DependencyCheck(
            name="cache_evictions",
            status=HealthStatus.HEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            details=details,
        )

    except Exception as e:
        return DependencyCheck(
            name="cache_evictions",
            status=HealthStatus.UNHEALTHY,
            latency_ms=(time.perf_counter() - start) * 1000,
            message=str(e),
        )


def run_health_checks(
    cache_dir: Path,
    max_cache_size_bytes: int,
    planning_executor,
    fetch_executor,
) -> HealthReport:
    """Run all health checks and return a comprehensive report."""
    checks = [
        check_disk_cache(cache_dir, max_cache_size_bytes),
        check_metadata_store(cache_dir),
        check_arrow_memory(),
        check_thread_pools(planning_executor, fetch_executor),
        check_rate_limiter(),
        check_cache_evictions(),
    ]

    # Determine overall status (worst of all checks)
    if any(c.status == HealthStatus.UNHEALTHY for c in checks):
        overall = HealthStatus.UNHEALTHY
    elif any(c.status == HealthStatus.DEGRADED for c in checks):
        overall = HealthStatus.DEGRADED
    else:
        overall = HealthStatus.HEALTHY

    return HealthReport(
        status=overall,
        checks=checks,
        timestamp=time.time(),
    )

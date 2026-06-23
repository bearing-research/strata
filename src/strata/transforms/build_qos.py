"""Build QoS (Quality of Service) admission control.

Provides quotas and backpressure for the build system:
- Per-tenant build slots (concurrency limits)
- Global build slots (overall system limit)
- Per-tenant bytes/day quota (optional)
- Priority queues: interactive vs bulk builds
- Early rejection with clear 429/403 errors

Unlike the build runner's semaphores (which control execution), this module
controls admission at the API layer - rejecting builds before they're created
if the system is at capacity.

Design principles:
- Fail fast with 429 rather than letting the queue explode
- Return Retry-After headers for client backoff
- Track queue wait time for observability
- Support dynamic slot resizing (Netflix-style adaptive control)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from strata.adaptive_concurrency import ResizableLimiter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BuildPriority(Enum):
    """Build priority classification.

    Interactive builds (dashboards, ad-hoc queries) get priority.
    Bulk builds (ETL, batch jobs) go to a separate queue.
    """

    INTERACTIVE = "interactive"
    BULK = "bulk"


class BuildQoSError(Exception):
    """Base error for QoS rejections."""

    def __init__(self, message: str, status_code: int = 429, retry_after: float | None = None):
        self.message = message
        self.status_code = status_code
        self.retry_after = retry_after
        super().__init__(message)

    def to_dict(self) -> dict:
        """Convert error to dictionary representation."""
        return {
            "error": "build_qos_error",
            "message": self.message,
            "retry_after_seconds": self.retry_after,
        }


class TenantAtCapacityError(BuildQoSError):
    """Raised when a tenant has reached their concurrent build limit."""

    def __init__(
        self,
        tenant_id: str,
        limit: int,
        active: int,
        retry_after: float = 5.0,
    ):
        self.tenant_id = tenant_id
        self.limit = limit
        self.active = active
        super().__init__(
            f"Tenant {tenant_id} at capacity ({active}/{limit} concurrent builds)",
            status_code=429,
            retry_after=retry_after,
        )

    def to_dict(self) -> dict:
        return {
            "error": "tenant_at_capacity",
            "message": self.message,
            "tenant_id": self.tenant_id,
            "limit": self.limit,
            "active": self.active,
            "retry_after_seconds": self.retry_after,
        }


class GlobalCapacityError(BuildQoSError):
    """Raised when global build capacity is exhausted."""

    def __init__(
        self,
        tier: str,
        slots: int,
        queue_timeout_seconds: float,
        queue_wait_ms: float,
        retry_after: float = 5.0,
    ):
        self.tier = tier
        self.slots = slots
        self.queue_timeout_seconds = queue_timeout_seconds
        self.queue_wait_ms = queue_wait_ms
        super().__init__(
            f"Server at capacity ({slots} {tier} build slots)",
            status_code=429,
            retry_after=retry_after,
        )

    def to_dict(self) -> dict:
        return {
            "error": "too_many_requests",
            "message": self.message,
            "tier": self.tier,
            "slots": self.slots,
            "queue_timeout_seconds": self.queue_timeout_seconds,
            "queue_wait_ms": round(self.queue_wait_ms, 1),
            "retry_after_seconds": self.retry_after,
        }


class TenantQuotaExceededError(BuildQoSError):
    """Raised when a tenant exceeds their daily byte quota."""

    def __init__(
        self,
        tenant_id: str,
        used_bytes: int,
        limit_bytes: int,
        reset_in_seconds: float,
    ):
        self.tenant_id = tenant_id
        self.used_bytes = used_bytes
        self.limit_bytes = limit_bytes
        self.reset_in_seconds = reset_in_seconds
        super().__init__(
            f"Tenant {tenant_id} exceeded daily quota ({used_bytes}/{limit_bytes} bytes)",
            status_code=429,
            retry_after=reset_in_seconds,
        )

    def to_dict(self) -> dict:
        return {
            "error": "quota_exceeded",
            "message": self.message,
            "tenant_id": self.tenant_id,
            "used_bytes": self.used_bytes,
            "limit_bytes": self.limit_bytes,
            "reset_in_seconds": round(self.reset_in_seconds, 1),
            "retry_after_seconds": self.retry_after,
        }


@dataclass
class BuildQoSConfig:
    """Configuration for build QoS.

    Attributes:
        interactive_slots: Max concurrent interactive builds (globally)
        bulk_slots: Max concurrent bulk builds (globally)
        per_tenant_interactive: Max concurrent interactive builds per tenant
        per_tenant_bulk: Max concurrent bulk builds per tenant
        interactive_queue_timeout: Max queue wait for interactive (seconds)
        bulk_queue_timeout: Max queue wait for bulk (seconds)
        per_tenant_timeout: Max wait for per-tenant slot (seconds)
        bytes_per_day_limit: Per-tenant daily byte limit (None = unlimited)
        classify_by_estimated_bytes: Threshold for bulk classification
        classify_by_input_count: Threshold for bulk classification
    """

    interactive_slots: int = 16
    bulk_slots: int = 8
    per_tenant_interactive: int = 4
    per_tenant_bulk: int = 2
    interactive_queue_timeout: float = 5.0
    bulk_queue_timeout: float = 15.0
    per_tenant_timeout: float = 1.0
    bytes_per_day_limit: int | None = None  # None = unlimited
    classify_by_estimated_bytes: int = 100 * 1024 * 1024  # 100MB
    classify_by_input_count: int = 5  # >5 inputs = bulk


@dataclass
class TenantQuota:
    """Per-tenant daily quota tracking.

    Tracks bytes produced by builds for quota enforcement.
    Resets daily at midnight UTC.
    """

    tenant_id: str
    bytes_today: int = 0
    last_reset: float = field(default_factory=time.time)

    def reset_if_new_day(self) -> None:
        """Reset quota if we've crossed midnight UTC."""
        now = time.time()
        # Calculate days since epoch
        current_day = int(now // 86400)
        last_day = int(self.last_reset // 86400)
        if current_day > last_day:
            self.bytes_today = 0
            self.last_reset = now

    def seconds_until_reset(self) -> float:
        """Seconds until next midnight UTC."""
        now = time.time()
        current_day = int(now // 86400)
        next_midnight = (current_day + 1) * 86400
        return max(0.0, next_midnight - now)


@dataclass
class TenantLimiters:
    """Per-tenant limiters for build concurrency."""

    interactive: ResizableLimiter
    bulk: ResizableLimiter
    quota: TenantQuota


class BuildQoS:
    """Build QoS admission controller.

    Provides fair scheduling, backpressure, and quota enforcement.
    Use `acquire()` before starting a build and `release()` when done.

    Example:
        qos = BuildQoS(config)
        await qos.start()

        try:
            async with qos.acquire(tenant_id="acme", priority=BuildPriority.INTERACTIVE):
                # Build runs here
                ...
        except BuildQoSError as e:
            return JSONResponse(status_code=e.status_code, content=e.to_dict())
    """

    def __init__(self, config: BuildQoSConfig):
        self.config = config

        # Global limiters (shared across all tenants)
        self._interactive_limiter = ResizableLimiter(config.interactive_slots)
        self._bulk_limiter = ResizableLimiter(config.bulk_slots)

        # Per-tenant limiters (lazy created)
        self._tenant_limiters: dict[str, TenantLimiters] = {}
        self._tenant_lock = asyncio.Lock()

        # Metrics
        self._lock = threading.Lock()
        self._interactive_rejected = 0
        self._bulk_rejected = 0
        self._tenant_rejected = 0
        self._quota_rejected = 0
        self._interactive_queue_wait_total_ms = 0.0
        self._interactive_queue_wait_count = 0
        self._bulk_queue_wait_total_ms = 0.0
        self._bulk_queue_wait_count = 0

    async def _get_tenant_limiters(self, tenant_id: str) -> TenantLimiters:
        """Get or create limiters for a tenant."""
        async with self._tenant_lock:
            if tenant_id not in self._tenant_limiters:
                self._tenant_limiters[tenant_id] = TenantLimiters(
                    interactive=ResizableLimiter(self.config.per_tenant_interactive),
                    bulk=ResizableLimiter(self.config.per_tenant_bulk),
                    quota=TenantQuota(tenant_id=tenant_id),
                )
            return self._tenant_limiters[tenant_id]

    def classify_build(
        self,
        estimated_output_bytes: int | None = None,
        input_count: int = 0,
        explicit_priority: BuildPriority | None = None,
    ) -> BuildPriority:
        """Classify a build as interactive or bulk.

        Classification criteria (in order):
        1. Explicit priority (if specified)
        2. Large output estimate (> threshold bytes)
        3. Many inputs (> threshold count)
        4. Default to interactive

        Args:
            estimated_output_bytes: Estimated output size (from transform def)
            input_count: Number of input artifacts/tables
            explicit_priority: Client-specified priority (takes precedence)

        Returns:
            BuildPriority (INTERACTIVE or BULK)
        """
        if explicit_priority is not None:
            return explicit_priority

        # Large outputs are bulk
        if estimated_output_bytes is not None:
            if estimated_output_bytes > self.config.classify_by_estimated_bytes:
                return BuildPriority.BULK

        # Many inputs are bulk (likely joins/aggregations)
        if input_count > self.config.classify_by_input_count:
            return BuildPriority.BULK

        return BuildPriority.INTERACTIVE

    async def check_quota(self, tenant_id: str, estimated_bytes: int) -> None:
        """Check if tenant has quota remaining.

        Args:
            tenant_id: Tenant identifier
            estimated_bytes: Estimated output bytes for this build

        Raises:
            TenantQuotaExceededError: If adding this build would exceed quota
        """
        if self.config.bytes_per_day_limit is None:
            return

        limiters = await self._get_tenant_limiters(tenant_id)
        quota = limiters.quota

        # Reset if new day
        quota.reset_if_new_day()

        # Check if adding this build would exceed quota
        if quota.bytes_today + estimated_bytes > self.config.bytes_per_day_limit:
            with self._lock:
                self._quota_rejected += 1
            raise TenantQuotaExceededError(
                tenant_id=tenant_id,
                used_bytes=quota.bytes_today,
                limit_bytes=self.config.bytes_per_day_limit,
                reset_in_seconds=quota.seconds_until_reset(),
            )

    async def record_bytes(self, tenant_id: str, bytes_produced: int) -> None:
        """Record bytes produced by a build (for quota tracking).

        Call this after a build completes with the actual output size.
        """
        if self.config.bytes_per_day_limit is None:
            return

        limiters = await self._get_tenant_limiters(tenant_id)
        quota = limiters.quota
        quota.reset_if_new_day()
        quota.bytes_today += bytes_produced

    async def acquire(
        self,
        tenant_id: str,
        priority: BuildPriority,
    ) -> BuildSlot:
        """Acquire a build slot.

        This is the main admission control entry point. It:
        1. Checks per-tenant concurrency limit
        2. Queues for global tier slot (interactive or bulk)
        3. Fails fast with 429 if capacity is exhausted

        Args:
            tenant_id: Tenant identifier
            priority: Build priority (interactive or bulk)

        Returns:
            BuildSlot context manager

        Raises:
            TenantAtCapacityError: Tenant has too many concurrent builds
            GlobalCapacityError: System is at capacity
        """
        limiters = await self._get_tenant_limiters(tenant_id)

        # Select limiters based on priority
        if priority == BuildPriority.INTERACTIVE:
            tenant_limiter = limiters.interactive
            global_limiter = self._interactive_limiter
            queue_timeout = self.config.interactive_queue_timeout
            tier_name = "interactive"
        else:
            tenant_limiter = limiters.bulk
            global_limiter = self._bulk_limiter
            queue_timeout = self.config.bulk_queue_timeout
            tier_name = "bulk"

        # Step 1: Try per-tenant slot (short timeout - fail fast)
        if not await tenant_limiter.acquire(timeout=self.config.per_tenant_timeout):
            with self._lock:
                self._tenant_rejected += 1
            max_per_tenant = (
                self.config.per_tenant_interactive
                if priority == BuildPriority.INTERACTIVE
                else self.config.per_tenant_bulk
            )
            raise TenantAtCapacityError(
                tenant_id=tenant_id,
                limit=max_per_tenant,
                active=tenant_limiter.in_use,
                retry_after=self.config.per_tenant_timeout,
            )

        # Step 2: Try global tier slot (with queue wait tracking)
        queue_start = time.time()
        tenant_slot_released = False
        try:
            acquired = await global_limiter.acquire(timeout=queue_timeout)
            queue_wait_ms = (time.time() - queue_start) * 1000

            # Track queue wait metrics
            with self._lock:
                if priority == BuildPriority.INTERACTIVE:
                    self._interactive_queue_wait_total_ms += queue_wait_ms
                    self._interactive_queue_wait_count += 1
                else:
                    self._bulk_queue_wait_total_ms += queue_wait_ms
                    self._bulk_queue_wait_count += 1

            if not acquired:
                # Release tenant slot since we failed to get global slot
                await tenant_limiter.release()
                tenant_slot_released = True
                with self._lock:
                    if priority == BuildPriority.INTERACTIVE:
                        self._interactive_rejected += 1
                    else:
                        self._bulk_rejected += 1
                raise GlobalCapacityError(
                    tier=tier_name,
                    slots=global_limiter.capacity,
                    queue_timeout_seconds=queue_timeout,
                    queue_wait_ms=queue_wait_ms,
                    retry_after=5.0,
                )

            return BuildSlot(
                qos=self,
                tenant_id=tenant_id,
                priority=priority,
                tenant_limiter=tenant_limiter,
                global_limiter=global_limiter,
            )

        except BaseException:
            # Release the tenant slot on ANY exit (if not already released),
            # including asyncio.CancelledError — a BaseException, not Exception.
            # The global-slot acquire above is a cancellation point (client
            # disconnect / shutdown while queued), and `except Exception` let a
            # cancel escape without releasing the per-tenant slot grabbed at
            # step 1 — a permanent per-tenant slot leak that bleeds into
            # TenantAtCapacityError for every later build (effectively a
            # per-tenant DoS). ResizableLimiter has no timeout-based reclaim.
            if not tenant_slot_released:
                await tenant_limiter.release()
            raise

    def get_metrics(self) -> dict:
        """Get QoS metrics for observability."""
        with self._lock:
            interactive_avg_wait = (
                self._interactive_queue_wait_total_ms / self._interactive_queue_wait_count
                if self._interactive_queue_wait_count > 0
                else 0.0
            )
            bulk_avg_wait = (
                self._bulk_queue_wait_total_ms / self._bulk_queue_wait_count
                if self._bulk_queue_wait_count > 0
                else 0.0
            )

            return {
                "interactive": {
                    "slots": self._interactive_limiter.capacity,
                    "active": self._interactive_limiter.in_use,
                    "available": self._interactive_limiter.available,
                    "rejected": self._interactive_rejected,
                    "queue_wait_avg_ms": round(interactive_avg_wait, 1),
                    "queue_wait_count": self._interactive_queue_wait_count,
                },
                "bulk": {
                    "slots": self._bulk_limiter.capacity,
                    "active": self._bulk_limiter.in_use,
                    "available": self._bulk_limiter.available,
                    "rejected": self._bulk_rejected,
                    "queue_wait_avg_ms": round(bulk_avg_wait, 1),
                    "queue_wait_count": self._bulk_queue_wait_count,
                },
                "per_tenant": {
                    "rejected": self._tenant_rejected,
                    "tenants_tracked": len(self._tenant_limiters),
                },
                "quota": {
                    "rejected": self._quota_rejected,
                    "limit_bytes_per_day": self.config.bytes_per_day_limit,
                },
            }

    def get_tenant_metrics(self, tenant_id: str) -> dict | None:
        """Get metrics for a specific tenant."""
        if tenant_id not in self._tenant_limiters:
            return None

        limiters = self._tenant_limiters[tenant_id]
        return {
            "tenant_id": tenant_id,
            "interactive": {
                "slots": limiters.interactive.capacity,
                "active": limiters.interactive.in_use,
                "available": limiters.interactive.available,
            },
            "bulk": {
                "slots": limiters.bulk.capacity,
                "active": limiters.bulk.in_use,
                "available": limiters.bulk.available,
            },
            "quota": {
                "bytes_today": limiters.quota.bytes_today,
                "limit_bytes": self.config.bytes_per_day_limit,
                "seconds_until_reset": round(limiters.quota.seconds_until_reset(), 1),
            },
        }


class BuildSlot:
    """Context manager for a build slot.

    Ensures proper release of both tenant and global slots.
    """

    def __init__(
        self,
        qos: BuildQoS,
        tenant_id: str,
        priority: BuildPriority,
        tenant_limiter: ResizableLimiter,
        global_limiter: ResizableLimiter,
    ):
        self._qos = qos
        self._tenant_id = tenant_id
        self._priority = priority
        self._tenant_limiter = tenant_limiter
        self._global_limiter = global_limiter
        self._released = False

    async def release(self) -> None:
        """Release the slot (idempotent)."""
        if self._released:
            return
        self._released = True

        # Release in reverse order (global first, then tenant)
        await self._global_limiter.release()
        await self._tenant_limiter.release()

    async def __aenter__(self) -> BuildSlot:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.release()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_build_qos: BuildQoS | None = None


def get_build_qos() -> BuildQoS | None:
    """Get the build QoS singleton."""
    return _build_qos


def set_build_qos(qos: BuildQoS | None) -> None:
    """Set the build QoS singleton."""
    global _build_qos
    _build_qos = qos


def reset_build_qos() -> None:
    """Reset the build QoS singleton (for testing)."""
    global _build_qos
    _build_qos = None


def normalized_build_qos_tenant_id(tenant_id: str | None) -> str:
    """Normalize tenant IDs for build QoS accounting."""
    return tenant_id or "__default__"


async def record_build_output_bytes(tenant_id: str | None, bytes_produced: int) -> None:
    """Best-effort quota accounting for completed builds."""
    build_qos = get_build_qos()
    if build_qos is None:
        return

    await build_qos.record_bytes(
        normalized_build_qos_tenant_id(tenant_id),
        max(0, bytes_produced),
    )

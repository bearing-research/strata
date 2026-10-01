"""Two-tier (interactive/bulk) QoS admission for scan streaming.

Admission spans the whole streaming response: acquire the per-client semaphore,
then the tenant limiter, and release exactly once on whichever exit path runs.
Leaf module: never imports ``strata.server``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from strata.adaptive_concurrency import ResizableLimiter
from strata.tenant import get_tenant_id
from strata.tenant_registry import get_tenant_registry

if TYPE_CHECKING:
    from starlette.requests import Request

    from strata.adaptive_concurrency import AdaptiveConcurrencyController
    from strata.config import StrataConfig


class QoSRejected(Exception):
    """Admission refused; the handler maps ``error``, ``tier`` and ``retry_after`` to a 429."""

    def __init__(self, error: str, tier: str, retry_after: int) -> None:
        super().__init__(error)
        self.error = error
        self.tier = tier
        self.retry_after = retry_after


class Admission:
    """A held QoS slot for one scan; call ``release()`` on every exit path (idempotent)."""

    def __init__(
        self,
        qos: QoSAdmission,
        scan_id: str,
        tier: str,
        limiter: Any,
        client_id: str,
        client_semaphore_acquired: bool,
        client_semaphore: Any = None,
    ) -> None:
        self._qos = qos
        self.scan_id = scan_id
        self.tier = tier
        self.limiter = limiter
        self.client_id = client_id
        self.client_semaphore_acquired = client_semaphore_acquired
        # Slot-held duration is the adaptive controller's latency signal, and
        # release() is the one point every exit path funnels through.
        self.admitted_at = time.perf_counter()
        # Release this exact semaphore, not one re-derived from client_id, so release stays correct
        # under LRU eviction and two admissions sharing a scan_id (see ``QoSAdmission._release``).
        self.client_semaphore = client_semaphore
        self._released = False

    async def release(self) -> None:
        """Release the tenant limiter + per-client semaphore and clear counters."""
        await self._qos._release(self)


class QoSAdmission:
    """Two-tier (interactive/bulk) admission control for scan streaming.

    Owns the per-client fairness semaphores and the counters; the tenant limiters
    live in the tenant registry and are only acquired and released here.
    """

    def __init__(self, config: StrataConfig) -> None:
        self._config = config
        # Attached in the lifespan once the controller exists (ServerState is
        # built first). None = adaptive control off, and admission just keeps
        # its own counters.
        self._controller: AdaptiveConcurrencyController | None = None
        # scan_id -> "interactive" | "bulk"
        self._scan_tier: dict[str, str] = {}
        # scan_id -> (client_id, semaphore_acquired)
        self._scan_client: dict[str, tuple[str, bool]] = {}
        # Approximate active counters (observability only; +=/-= not atomic).
        self._active_scans = 0
        self._active_interactive = 0
        self._active_bulk = 0
        # Rejection counters (429 when queue deadline / per-client cap exceeded).
        self._interactive_rejected = 0
        self._bulk_rejected = 0
        self._client_rejected = 0
        self._interactive_queue_wait_total_ms = 0.0
        self._interactive_queue_wait_count = 0
        self._bulk_queue_wait_total_ms = 0.0
        self._bulk_queue_wait_count = 0
        # Per-client fairness: LRU dict of client_id -> Semaphore per tier.
        self._client_interactive_semaphores: dict[str, asyncio.Semaphore] = {}
        self._client_bulk_semaphores: dict[str, asyncio.Semaphore] = {}
        self._client_semaphore_max_entries = 10000

    def attach_controller(self, controller: AdaptiveConcurrencyController | None) -> None:
        """Feed observed queue waits and slot-held durations to *controller*.

        Attached by the lifespan, since the controller needs the tenant registry's
        limiters; until then the control loop has no inputs.
        """
        self._controller = controller

    @property
    def active_scans(self) -> int:
        """Approximate in-flight scan count, for metrics only.

        Use :meth:`active_scan_count` for control flow such as draining.
        """
        return self._active_scans

    def classify(self, plan: Any) -> str:
        """Classify a plan as 'interactive' or 'bulk'.

        Interactive needs an explicit projection within ``interactive_max_columns``
        and an estimate within ``interactive_max_bytes``.
        """
        config = self._config
        if plan.estimated_bytes > config.interactive_max_bytes:
            return "bulk"
        if plan.columns is None:
            return "bulk"
        if len(plan.columns) > config.interactive_max_columns:
            return "bulk"
        return "interactive"

    def _get_client_semaphore(self, client_id: str, tier: str) -> asyncio.Semaphore | None:
        """Get or create a per-client semaphore for the tier, LRU-bounded; None when disabled."""
        if tier == "interactive":
            max_concurrent = self._config.per_client_interactive
            client_semaphores = self._client_interactive_semaphores
        else:
            max_concurrent = self._config.per_client_bulk
            client_semaphores = self._client_bulk_semaphores

        if max_concurrent <= 0:  # 0 = disabled
            return None

        if client_id in client_semaphores:
            # Move to the LRU tail.
            sem = client_semaphores.pop(client_id)
            client_semaphores[client_id] = sem
            return sem

        sem = asyncio.Semaphore(max_concurrent)
        client_semaphores[client_id] = sem
        while len(client_semaphores) > self._client_semaphore_max_entries:
            oldest_client = next(iter(client_semaphores))
            del client_semaphores[oldest_client]
        return sem

    async def admit(self, plan: Any, request: Request, scan_id: str) -> Admission:
        """Acquire a tier slot for *scan_id*, or raise :class:`QoSRejected`.

        Waits 1s for the per-client semaphore, then up to the tier's queue timeout
        for the tenant limiter. A cancelled acquire releases the per-client
        semaphore before propagating.
        """
        tier = self.classify(plan)
        tenant_id = get_tenant_id()
        if tier == "interactive":
            queue_timeout = self._config.interactive_queue_timeout
        else:
            queue_timeout = self._config.bulk_queue_timeout

        client_id = request.client.host if request.client else "unknown"
        client_semaphore = self._get_client_semaphore(client_id, tier)
        client_semaphore_acquired = False

        if client_semaphore is not None:
            try:
                await asyncio.wait_for(client_semaphore.acquire(), timeout=1.0)
                client_semaphore_acquired = True
            except TimeoutError:
                self._client_rejected += 1
                raise QoSRejected("per_client_limit", tier, 1)

        # Queue with deadline. A cancelled acquire (disconnect or shutdown while queued) must
        # release the per-client semaphore grabbed above: CancelledError is a BaseException, and the
        # `if not acquired:` path below only handles the timeout.
        #
        # Look the limiters up here, with no await before the acquire: an idle tenant can be evicted
        # while its request waits for the per-client slot, leaving a limiter the registry no longer
        # counts.
        interactive_limiter, bulk_limiter = get_tenant_registry().get_or_create_limiters(tenant_id)
        limiter = interactive_limiter if tier == "interactive" else bulk_limiter
        queue_start = time.perf_counter()
        try:
            acquired = await limiter.acquire(timeout=queue_timeout)
        except BaseException:
            if client_semaphore_acquired and client_semaphore is not None:
                client_semaphore.release()
            raise
        queue_wait_ms = (time.perf_counter() - queue_start) * 1000

        if not acquired:
            if client_semaphore_acquired and client_semaphore is not None:
                client_semaphore.release()
            if tier == "interactive":
                self._interactive_rejected += 1
            else:
                self._bulk_rejected += 1
            raise QoSRejected("too_many_requests", tier, max(1, int(queue_timeout / 2)))

        # Queue wait is both an operator metric and the controller's demand
        # signal: latency under target only justifies more slots if requests
        # are actually waiting for them.
        if tier == "interactive":
            self._interactive_queue_wait_total_ms += queue_wait_ms
            self._interactive_queue_wait_count += 1
        else:
            self._bulk_queue_wait_total_ms += queue_wait_ms
            self._bulk_queue_wait_count += 1
        if self._controller is not None:
            self._controller.record_queue_wait(tier, queue_wait_ms)

        self._scan_tier[scan_id] = tier
        self._scan_client[scan_id] = (client_id, client_semaphore_acquired)
        if tier == "interactive":
            self._active_interactive += 1
        else:
            self._active_bulk += 1
        self._active_scans += 1

        return Admission(
            self,
            scan_id,
            tier,
            limiter,
            client_id,
            client_semaphore_acquired,
            client_semaphore,
        )

    async def _release(self, admission: Admission) -> None:
        """Reverse an :meth:`admit` once, releasing exactly the objects it acquired.

        Never re-derive the semaphore from ``scan_id`` or ``client_id``: two
        admissions can share a ``scan_id`` (a retried stream GET), and LRU eviction
        would hand back a fresh semaphore, so either would leak or inflate a slot.
        """
        if admission._released:
            return
        admission._released = True

        if self._controller is not None:
            held_ms = (time.perf_counter() - admission.admitted_at) * 1000
            self._controller.record_latency(admission.tier, held_ms)

        await admission.limiter.release()
        self._scan_tier.pop(admission.scan_id, None)
        self._scan_client.pop(admission.scan_id, None)
        if admission.client_semaphore_acquired and admission.client_semaphore is not None:
            admission.client_semaphore.release()
        if admission.tier == "interactive":
            self._active_interactive -= 1
        else:
            self._active_bulk -= 1
        self._active_scans -= 1

    def active_scan_count(self) -> int:
        """Authoritative in-flight scan count, from the per-tenant limiters.

        Graceful shutdown drains on this so it never cuts off a live stream.
        """
        i_in_use, _, b_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
        return i_in_use + b_in_use

    def qos_metrics(self) -> dict[str, Any]:
        """QoS tier metrics: capacity/usage, rejections, queue waits, per-tenant."""
        # The per-tenant limiters stream admission actually acquires, aggregated (single-tenant
        # reduces to the _default tenant).
        i_in_use, i_avail, b_in_use, b_avail = get_tenant_registry().aggregate_limiter_usage()

        interactive_avg_wait_ms = (
            self._interactive_queue_wait_total_ms / self._interactive_queue_wait_count
            if self._interactive_queue_wait_count > 0
            else 0.0
        )
        bulk_avg_wait_ms = (
            self._bulk_queue_wait_total_ms / self._bulk_queue_wait_count
            if self._bulk_queue_wait_count > 0
            else 0.0
        )

        # Per-tenant QoS metrics (only for tenants with resizable limiters).
        tenant_registry = get_tenant_registry()
        per_tenant_qos: dict[str, Any] = {}
        with tenant_registry._lock:
            for tenant_id, quotas in tenant_registry._quotas.items():
                interactive = quotas.interactive_limiter
                bulk = quotas.bulk_limiter
                if isinstance(interactive, ResizableLimiter) and isinstance(bulk, ResizableLimiter):
                    per_tenant_qos[tenant_id] = {
                        "interactive_capacity": interactive.capacity,
                        "interactive_in_use": interactive.in_use,
                        "bulk_capacity": bulk.capacity,
                        "bulk_in_use": bulk.in_use,
                    }

        return {
            "interactive_slots": i_in_use + i_avail,
            "interactive_active": i_in_use,
            "interactive_available": i_avail,
            "interactive_rejected": self._interactive_rejected,
            "interactive_queue_timeout_seconds": self._config.interactive_queue_timeout,
            "interactive_queue_wait_avg_ms": round(interactive_avg_wait_ms, 2),
            "interactive_queue_wait_total_ms": round(self._interactive_queue_wait_total_ms, 2),
            "interactive_queue_wait_count": self._interactive_queue_wait_count,
            "bulk_slots": b_in_use + b_avail,
            "bulk_active": b_in_use,
            "bulk_available": b_avail,
            "bulk_rejected": self._bulk_rejected,
            "bulk_queue_timeout_seconds": self._config.bulk_queue_timeout,
            "bulk_queue_wait_avg_ms": round(bulk_avg_wait_ms, 2),
            "bulk_queue_wait_total_ms": round(self._bulk_queue_wait_total_ms, 2),
            "bulk_queue_wait_count": self._bulk_queue_wait_count,
            "per_client_interactive": self._config.per_client_interactive,
            "per_client_bulk": self._config.per_client_bulk,
            "client_rejected": self._client_rejected,
            "tracked_clients": len(self._client_interactive_semaphores),
            "per_tenant": per_tenant_qos,
        }

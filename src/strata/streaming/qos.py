"""Two-tier QoS admission for scan streaming (#302 phase 3).

Extracted from ``server.py`` so the ``get_stream`` handler can express admission
structurally (``admission = await state.qos.admit(...)`` / ``admission.release()``)
and eventually move into ``api/routers/streams.py`` without a ``from strata.server
import _private`` scatter.

Admission spans the entire streaming response: acquire the per-client semaphore
then the tenant limiter (with the #238 cancel-safety), then release exactly once
in whichever exit path runs (429, build-cancel, or after the blob finishes). The
:class:`Admission` token carries the acquired state; ``release()`` reverses it,
cancel-safe.

Leaf module: imports only ``strata.adaptive_concurrency`` / ``strata.tenant`` /
``strata.tenant_registry`` / ``strata.config``, never ``strata.server``.
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
    """Admission refused — the handler maps this to a 429 JSON response.

    Carries the exact ``error`` code, ``tier``, and ``retry_after`` the inline
    handler used to return, so the 429 body/headers are unchanged.
    """

    def __init__(self, error: str, tier: str, retry_after: int) -> None:
        super().__init__(error)
        self.error = error
        self.tier = tier
        self.retry_after = retry_after


class Admission:
    """A held QoS slot for one scan. ``release()`` runs the accounting once.

    Returned by :meth:`QoSAdmission.admit`; the handler calls ``release()`` in
    whichever exit path runs. Release is delegated back to the owning
    :class:`QoSAdmission` so all the shared counters/tables mutate in one place.
    """

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
        # The exact semaphore object this admission acquired. Releasing THIS
        # object (rather than re-deriving one from client_id at release time)
        # is what makes release correct under both an LRU eviction and two
        # admissions sharing a scan_id — see ``QoSAdmission._release``.
        self.client_semaphore = client_semaphore
        self._released = False

    async def release(self) -> None:
        """Release the tenant limiter + per-client semaphore and clear counters."""
        await self._qos._release(self)


class QoSAdmission:
    """Two-tier (interactive/bulk) admission control for scan streaming.

    Owns the admission runtime that used to live on ``ServerState``: the
    per-scan tier/client tables, the active/rejection/queue-wait counters, and
    the per-client fairness semaphores. The tenant limiters themselves live in
    the tenant registry (per-tenant, #185); this only acquires/releases them.
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
        # Queue wait tracking (observability).
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

        The controller cannot be constructed at ``ServerState`` time (it needs
        the tenant registry's limiters), so the lifespan hands it over here.
        Until it does, the control loop has no inputs at all — which is how it
        ran as a no-op timer for eight months (#549).
        """
        self._controller = controller

    @property
    def active_scans(self) -> int:
        """Approximate in-flight scan count (observability only, not authoritative).

        Bumped on admit / released on release; ``+=``/``-=`` are not atomic under
        async, so it's for metrics/logging, not control flow — use
        :meth:`active_scan_count` (limiter-derived) for the drain signal.
        """
        return self._active_scans

    def classify(self, plan: Any) -> str:
        """Classify a query as 'interactive' or 'bulk' based on its plan.

        Interactive queries are small, fast dashboard-style queries: estimated
        response size <= ``interactive_max_bytes`` and column count <=
        ``interactive_max_columns``. Everything else is bulk (a ``None``
        projection = all columns = likely bulk).
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
        """Get or create a per-client semaphore for the tier (None = disabled).

        Simple LRU: touch on hit, evict oldest past ``_client_semaphore_max_entries``.
        """
        if tier == "interactive":
            max_concurrent = self._config.per_client_interactive
            client_semaphores = self._client_interactive_semaphores
        else:
            max_concurrent = self._config.per_client_bulk
            client_semaphores = self._client_bulk_semaphores

        if max_concurrent <= 0:  # 0 = disabled
            return None

        if client_id in client_semaphores:
            # Move to end for LRU (dict keeps insertion order).
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

        Mirrors the previous inline acquire verbatim: classify the tier, acquire
        the per-client semaphore (1s) then the tenant limiter (queue deadline).
        The #238 property is preserved — if the limiter acquire is cancelled
        (``CancelledError`` is a ``BaseException``), the per-client semaphore is
        released before propagating. On success it records the tier/client and
        bumps the active counters, returning an :class:`Admission` to release.
        """
        tier = self.classify(plan)
        tenant_id = get_tenant_id()
        interactive_limiter, bulk_limiter = get_tenant_registry().get_or_create_limiters(tenant_id)

        if tier == "interactive":
            limiter = interactive_limiter
            queue_timeout = self._config.interactive_queue_timeout
        else:
            limiter = bulk_limiter
            queue_timeout = self._config.bulk_queue_timeout

        # Per-client fairness
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

        # Queue with deadline. If this acquire is cancelled (client disconnect /
        # shutdown while queued for a tenant slot), release the per-client
        # semaphore grabbed above before propagating — CancelledError is a
        # BaseException, and the `if not acquired:` path below only handles the
        # timeout (False) case, so the semaphore would otherwise leak a slot.
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
        """Reverse an :meth:`admit`: release the limiter + client semaphore once.

        Runs on every exit path including cancellation — the #238 leak fix.

        Releases exactly what this admission acquired. The previous version
        re-derived both from ``scan_id`` / ``client_id`` at release time, which
        leaked a per-client slot two ways:

        - ``_scan_client`` is keyed by ``scan_id``, and nothing stops two
          admissions sharing one (``GET /v1/streams/{id}`` has no
          already-consumed guard, so a client retry or a proxy retry produces
          two handlers for the same stream). The first release popped the
          entry; the second found ``None`` and never released its semaphore.
          That client permanently loses a slot, and after
          ``per_client_interactive`` such events it is 429'd forever.
        - ``_get_client_semaphore`` *creates* a semaphore when the entry is
          missing, so if the LRU evicted it between acquire and release, the
          release landed on a brand-new ``Semaphore(max_concurrent)`` and
          ratcheted its value to ``max_concurrent + 1`` — repeatable, so the
          per-client cap grew without bound.

        Idempotent: a double release would otherwise drive the counters
        negative and hand back a slot twice.
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
        """Authoritative in-flight scan count from the admission limiters.

        Derives from the per-tenant limiters (what admission actually acquires),
        not the approximate ``_active_*`` counters — so graceful shutdown never
        drains past a live stream (#185).
        """
        i_in_use, _, b_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
        return i_in_use + b_in_use

    def qos_metrics(self) -> dict[str, Any]:
        """QoS tier metrics: capacity/usage, rejections, queue waits, per-tenant."""
        # Top-line capacity/usage reflects the per-tenant admission limiters that
        # stream admission actually acquires (aggregated; a single-tenant
        # deployment reduces to the _default tenant). Reading the never-acquired
        # global limiters always reported 0 (#185).
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
            # Per-client fairness metrics
            "per_client_interactive": self._config.per_client_interactive,
            "per_client_bulk": self._config.per_client_bulk,
            "client_rejected": self._client_rejected,
            "tracked_clients": len(self._client_interactive_semaphores),
            # Per-tenant QoS metrics
            "per_tenant": per_tenant_qos,
        }

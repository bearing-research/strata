"""Unit tests for QoSAdmission — the acquire/release lifecycle in isolation.

These pin the #238 cancel-safety property deterministically, which the
HTTP-level QoS characterization tests (test_qos.py) can't do reliably: a small
streaming response buffers fully before the client reads, so a mid-flight
disconnect never actually cancels the held-slot path. Extracting the admission
logic into QoSAdmission makes it directly callable, so we drive the cancel path
with a fake limiter instead of a racy socket.
"""

import asyncio
import threading
import time

import pytest

from strata.config import StrataConfig
from strata.streaming.qos import QoSAdmission, QoSRejected


class _FakeLimiter:
    """Stand-in tenant limiter: records acquire/release, scriptable outcome."""

    def __init__(self, *, acquire_result: bool = True, raise_cancel: bool = False):
        self._acquire_result = acquire_result
        self._raise_cancel = raise_cancel
        self.released = 0

    async def acquire(self, timeout=None):
        if self._raise_cancel:
            raise asyncio.CancelledError()
        return self._acquire_result

    async def release(self):
        self.released += 1


class _FakeRegistry:
    def __init__(self, interactive, bulk):
        self._interactive = interactive
        self._bulk = bulk
        # qos_metrics() reads these directly off the registry.
        self._lock = threading.Lock()
        self._quotas: dict[str, object] = {}

    def get_or_create_limiters(self, tenant_id):
        return self._interactive, self._bulk

    def aggregate_limiter_usage(self):
        return 0, 0, 0, 0


class _FakePlan:
    # Small + few columns → classifies "interactive".
    estimated_bytes = 100
    columns = ["a"]


class _FakeRequest:
    class _Client:
        host = "test-client"

    client = _Client()


def _install_registry(monkeypatch, interactive, bulk):
    monkeypatch.setattr("strata.streaming.qos.get_tenant_id", lambda: "_default")
    monkeypatch.setattr(
        "strata.streaming.qos.get_tenant_registry",
        lambda: _FakeRegistry(interactive, bulk),
    )


@pytest.mark.asyncio
async def test_admit_then_release_frees_limiter_and_counters(monkeypatch):
    limiter = _FakeLimiter()
    _install_registry(monkeypatch, limiter, _FakeLimiter())
    qos = QoSAdmission(StrataConfig())

    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-1")
    assert admission.tier == "interactive"
    assert qos._active_interactive == 1
    assert qos.active_scans == 1
    assert qos._scan_tier["scan-1"] == "interactive"

    await admission.release()
    assert limiter.released == 1
    assert qos._active_interactive == 0
    assert qos.active_scans == 0
    assert "scan-1" not in qos._scan_tier
    assert "scan-1" not in qos._scan_client


@pytest.mark.asyncio
async def test_client_semaphore_released_when_limiter_acquire_cancelled(monkeypatch):
    # #238: if the tenant-limiter acquire is cancelled (client disconnect /
    # shutdown while queued) while the per-client semaphore is already held, the
    # semaphore must be released before the CancelledError propagates — else that
    # client leaks a slot forever.
    limiter = _FakeLimiter(raise_cancel=True)
    _install_registry(monkeypatch, limiter, _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=1))

    with pytest.raises(asyncio.CancelledError):
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-1")

    # The per-client semaphore is back to full capacity — not stranded.
    sem = qos._get_client_semaphore("test-client", "interactive")
    assert sem is not None
    assert sem._value == 1
    # No admission was recorded.
    assert qos.active_scans == 0
    assert "scan-1" not in qos._scan_tier


@pytest.mark.asyncio
async def test_queue_timeout_rejects_and_releases_client_semaphore(monkeypatch):
    # Limiter acquire returns False (queue deadline exceeded) → 429, and the
    # per-client semaphore grabbed first must be released, not leaked.
    limiter = _FakeLimiter(acquire_result=False)
    _install_registry(monkeypatch, limiter, _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=1))

    with pytest.raises(QoSRejected) as excinfo:
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-1")
    assert excinfo.value.error == "too_many_requests"
    assert excinfo.value.tier == "interactive"

    sem = qos._get_client_semaphore("test-client", "interactive")
    assert sem is not None
    assert sem._value == 1  # released, not leaked
    assert qos.active_scans == 0


@pytest.mark.asyncio
async def test_per_client_cap_rejects_second_concurrent_admit(monkeypatch):
    # With per_client_interactive=1, a second admit from the same client while
    # the first still holds its slot is rejected (per_client_limit), and the
    # rejection counter increments.
    _install_registry(monkeypatch, _FakeLimiter(), _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=1))

    first = await qos.admit(_FakePlan(), _FakeRequest(), "scan-1")
    with pytest.raises(QoSRejected) as excinfo:
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-2")
    assert excinfo.value.error == "per_client_limit"
    assert qos._client_rejected == 1

    # The first admission still releases cleanly.
    await first.release()
    assert qos.active_scans == 0


@pytest.mark.asyncio
async def test_two_admissions_sharing_a_scan_id_both_release_their_slot(monkeypatch):
    """``_scan_client`` is keyed by scan_id, and nothing stops two admissions
    sharing one — ``GET /v1/streams/{id}`` has no already-consumed guard, so a
    client retry or a proxy retry produces two handlers for the same stream.

    Release used to re-derive the semaphore from that table, so the first
    release popped the entry and the second found nothing and never released
    its slot: the client permanently lost capacity and was eventually 429'd
    forever."""
    _install_registry(monkeypatch, _FakeLimiter(), _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=2))

    first = await qos.admit(_FakePlan(), _FakeRequest(), "same-scan")
    second = await qos.admit(_FakePlan(), _FakeRequest(), "same-scan")

    await first.release()
    await second.release()

    # Both per-client slots are back: a fresh pair of admits still succeeds.
    third = await qos.admit(_FakePlan(), _FakeRequest(), "scan-a")
    fourth = await qos.admit(_FakePlan(), _FakeRequest(), "scan-b")
    await third.release()
    await fourth.release()

    assert qos.active_scans == 0


@pytest.mark.asyncio
async def test_release_survives_client_semaphore_eviction(monkeypatch):
    """``_get_client_semaphore`` *creates* one when the entry is missing, so a
    release after an LRU eviction landed on a brand-new semaphore and ratcheted
    its value above the cap — repeatable, so the per-client limit grew without
    bound. Releasing the held object instead is immune."""
    _install_registry(monkeypatch, _FakeLimiter(), _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=1))

    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-evict")

    # Simulate the LRU dropping this client's entry while the scan is in flight.
    qos._client_interactive_semaphores.clear()
    qos._client_bulk_semaphores.clear()

    await admission.release()

    # The cap still holds: one concurrent admit, and the second is rejected.
    held = await qos.admit(_FakePlan(), _FakeRequest(), "scan-after-1")
    with pytest.raises(QoSRejected) as excinfo:
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-after-2")
    assert excinfo.value.error == "per_client_limit"
    await held.release()


@pytest.mark.asyncio
async def test_double_release_is_idempotent(monkeypatch):
    """A second release would otherwise drive the counters negative and hand
    the same slot back twice."""
    limiter = _FakeLimiter()
    _install_registry(monkeypatch, limiter, _FakeLimiter())
    qos = QoSAdmission(StrataConfig(per_client_interactive=1))

    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-double")
    await admission.release()
    await admission.release()

    assert limiter.released == 1
    assert qos.active_scans == 0


class _SlowLimiter(_FakeLimiter):
    """Limiter whose acquire blocks, so queue wait is a real observable."""

    def __init__(self, delay: float):
        super().__init__()
        self._delay = delay

    async def acquire(self, timeout=None):
        await asyncio.sleep(self._delay)
        return True


class _BulkPlan:
    # No projection → all columns → classifies "bulk".
    estimated_bytes = 100
    columns = None


class _RecordingController:
    """Stands in for AdaptiveConcurrencyController, capturing both feeds."""

    def __init__(self):
        self.queue_waits: list[tuple[str, float]] = []
        self.latencies: list[tuple[str, float]] = []

    def record_queue_wait(self, tier, wait_ms):
        self.queue_waits.append((tier, wait_ms))

    def record_latency(self, tier, latency_ms):
        self.latencies.append((tier, latency_ms))


@pytest.mark.asyncio
async def test_queue_wait_metric_reflects_the_actual_wait(monkeypatch):
    """``/metrics`` reported ``queue_wait_avg_ms: 0`` under every load.

    The accounting lived on ``POST /v1/scan`` and was deleted with that endpoint;
    the fields survived on the metrics payload, so the number an operator reads
    to size their deployment has been a constant zero since the unified
    materialize API landed.

    Asserted against the wall time the admit call actually took, not against a
    fixed threshold: the recorded wait is a sub-interval of that call, so it can
    be neither zero nor larger than the whole.
    """
    _install_registry(monkeypatch, _SlowLimiter(0.05), _FakeLimiter())
    qos = QoSAdmission(StrataConfig())

    started = time.perf_counter()
    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-wait")
    observed_ms = (time.perf_counter() - started) * 1000

    metrics = qos.qos_metrics()
    assert metrics["interactive_queue_wait_count"] == 1
    recorded = metrics["interactive_queue_wait_avg_ms"]
    assert 0 < recorded <= observed_ms
    await admission.release()


@pytest.mark.asyncio
async def test_tier_rejections_are_counted(monkeypatch):
    """``interactive_rejected`` / ``bulk_rejected`` were never incremented, so
    the metric that says "we are shedding load" read zero while shedding load."""
    _install_registry(monkeypatch, _FakeLimiter(acquire_result=False), _FakeLimiter())
    qos = QoSAdmission(StrataConfig())

    with pytest.raises(QoSRejected):
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-r1")

    metrics = qos.qos_metrics()
    assert metrics["interactive_rejected"] == 1
    assert metrics["bulk_rejected"] == 0


@pytest.mark.asyncio
async def test_bulk_tier_rejections_are_counted_separately(monkeypatch):
    _install_registry(monkeypatch, _FakeLimiter(), _FakeLimiter(acquire_result=False))
    qos = QoSAdmission(StrataConfig())

    with pytest.raises(QoSRejected) as excinfo:
        await qos.admit(_BulkPlan(), _FakeRequest(), "scan-r2")
    assert excinfo.value.tier == "bulk"

    metrics = qos.qos_metrics()
    assert metrics["bulk_rejected"] == 1
    assert metrics["interactive_rejected"] == 0


@pytest.mark.asyncio
async def test_attached_controller_receives_both_signals(monkeypatch):
    """The adaptive control loop has two inputs and both come from here.

    Without them ``get_p95()`` returns None forever and ``_evaluate_and_adjust``
    returns early on every tick — a 5-second timer that logs "started" and
    adjusts nothing (#549).
    """
    _install_registry(monkeypatch, _SlowLimiter(0.01), _FakeLimiter())
    qos = QoSAdmission(StrataConfig())
    controller = _RecordingController()
    qos.attach_controller(controller)

    started = time.perf_counter()
    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-fed")
    await asyncio.sleep(0.01)
    await admission.release()
    observed_ms = (time.perf_counter() - started) * 1000

    assert [tier for tier, _ in controller.queue_waits] == ["interactive"]
    assert [tier for tier, _ in controller.latencies] == ["interactive"]
    # Both are sub-intervals of the same observed span.
    assert 0 < controller.queue_waits[0][1] <= observed_ms
    assert 0 < controller.latencies[0][1] <= observed_ms


@pytest.mark.asyncio
async def test_rejected_admissions_feed_no_latency(monkeypatch):
    """A 429 never held a slot, so it has no slot-held duration to report."""
    _install_registry(monkeypatch, _FakeLimiter(acquire_result=False), _FakeLimiter())
    qos = QoSAdmission(StrataConfig())
    controller = _RecordingController()
    qos.attach_controller(controller)

    with pytest.raises(QoSRejected):
        await qos.admit(_FakePlan(), _FakeRequest(), "scan-rej")

    assert controller.latencies == []
    assert controller.queue_waits == []


@pytest.mark.asyncio
async def test_admission_works_with_no_controller_attached(monkeypatch):
    """Adaptive control is opt-in; the default deployment attaches nothing."""
    limiter = _FakeLimiter()
    _install_registry(monkeypatch, limiter, _FakeLimiter())
    qos = QoSAdmission(StrataConfig())

    admission = await qos.admit(_FakePlan(), _FakeRequest(), "scan-none")
    await admission.release()

    assert limiter.released == 1
    assert qos.qos_metrics()["interactive_queue_wait_count"] == 1

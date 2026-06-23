"""Unit tests for ScanBuildManager's scan table + prefetch (#302, phase 2a).

The build-side (``build_identity_artifact``) still lives in ``server.py`` and is
covered by the streaming e2e suites; these drive the scan-table + prefetch logic
directly with fake plans — no TestClient, no ServerState.
"""

from __future__ import annotations

from types import SimpleNamespace

from strata.streaming import ScanBuildManager


def _plan(scan_id: str = "scan-1", *, prefetched=None, tasks=(1,)) -> SimpleNamespace:
    return SimpleNamespace(scan_id=scan_id, prefetched_first=prefetched, tasks=list(tasks))


def test_scan_table_register_get_pop_contains():
    m = ScanBuildManager()
    plan = _plan()

    assert "scan-1" not in m
    assert m.get_scan("scan-1") is None

    m.register_scan(plan)
    assert "scan-1" in m
    assert m.get_scan("scan-1") is plan

    assert m.pop_scan("scan-1") is plan
    assert "scan-1" not in m
    assert m.pop_scan("scan-1") is None  # idempotent


def test_prefetch_metrics_initial_zero():
    assert ScanBuildManager().prefetch_metrics() == {
        "started": 0,
        "used": 0,
        "wasted": 0,
        "skipped": 0,
        "in_flight": 0,
    }


async def test_consume_prefetched_first_returns_warm_chunk_and_counts_used():
    m = ScanBuildManager()
    plan = _plan(prefetched=b"chunk0")
    m.register_scan(plan)

    chunk = await m.consume_prefetched_first(plan, "scan-1")
    assert chunk == b"chunk0"
    assert plan.prefetched_first is None  # consumed
    assert m.prefetch_metrics()["used"] == 1


async def test_consume_prefetched_first_none_when_cold_and_no_task():
    m = ScanBuildManager()
    plan = _plan(prefetched=None)
    m.register_scan(plan)

    # No warm chunk and no in-flight prefetch task → None (build does a direct fetch).
    assert await m.consume_prefetched_first(plan, "scan-1") is None
    assert m.prefetch_metrics()["used"] == 0


def test_discard_prefetch_counts_wasted_when_chunk_was_ready():
    m = ScanBuildManager()
    plan = _plan(prefetched=b"warm")
    m.register_scan(plan)

    m.discard_prefetch("scan-1", count_wasted=True)
    assert plan.prefetched_first is None
    assert m.prefetch_metrics()["wasted"] == 1


def test_discard_prefetch_no_waste_when_nothing_ready():
    m = ScanBuildManager()
    m.register_scan(_plan(prefetched=None))

    m.discard_prefetch("scan-1", count_wasted=True)
    assert m.prefetch_metrics()["wasted"] == 0


def test_expire_scan_discards_and_drops_the_scan():
    m = ScanBuildManager()
    plan = _plan(prefetched=b"warm")
    m.register_scan(plan)

    m.expire_scan("scan-1")
    assert "scan-1" not in m  # scan dropped
    assert m.prefetch_metrics()["wasted"] == 1  # prefetched chunk discarded


def test_start_prefetch_skips_when_already_warm():
    m = ScanBuildManager()
    plan = _plan(prefetched=b"already")
    state = SimpleNamespace(_draining=False)

    # A plan whose first chunk is already warm registers no prefetch task.
    m.start_prefetch(state, plan)
    assert m._prefetch_futures == {}
    assert m.prefetch_metrics()["started"] == 0


def test_start_prefetch_skips_empty_plan():
    m = ScanBuildManager()
    plan = _plan(prefetched=None, tasks=())
    state = SimpleNamespace(_draining=False)

    m.start_prefetch(state, plan)
    assert m._prefetch_futures == {}

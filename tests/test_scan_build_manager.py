"""Unit tests for ScanBuildManager's scan table and prefetch, driven with fake plans.

``build_identity_artifact`` still lives in ``server.py`` and is covered by the streaming e2e suites.
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

    # No warm chunk and no in-flight prefetch, so None (the build fetches directly).
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


def _fetching_state(fetched: list):
    def fetch(task):
        fetched.append(task)
        return b"first row group"

    return SimpleNamespace(
        _draining=False, _fetch_executor=None, fetcher=SimpleNamespace(fetch_as_stream_bytes=fetch)
    )


async def _prefetch(m: ScanBuildManager, state, plan) -> None:
    m.register_scan(plan)
    m.start_prefetch(state, plan)
    await m._prefetch_futures[plan.scan_id]


async def test_prefetch_holds_at_most_its_budget_for_streams_nobody_has_read():
    """A POST in stream mode prefetches its first row group, and nothing reads it until a GET
    that may never come; past the budget the POST skips it and the GET fetches it."""
    m = ScanBuildManager(max_held_prefetch_bytes=10)
    fetched: list = []
    state = _fetching_state(fetched)
    m.register_scan(_plan("unread", prefetched=b"x" * 10))

    skipped = _plan("next")
    await _prefetch(m, state, skipped)

    assert skipped.prefetched_first is None
    assert fetched == []
    assert m.prefetch_metrics()["skipped"] == 1


async def test_prefetch_resumes_once_held_bytes_are_streamed():
    m = ScanBuildManager(max_held_prefetch_bytes=10)
    fetched: list = []
    state = _fetching_state(fetched)
    unread = _plan("unread", prefetched=b"x" * 10)
    m.register_scan(unread)
    await m.consume_prefetched_first(unread, "unread")

    plan = _plan("next")
    await _prefetch(m, state, plan)

    assert plan.prefetched_first == b"first row group"
    assert fetched == [1]

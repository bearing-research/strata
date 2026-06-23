"""Unit tests for StreamRegistry (#302, phase 1).

The registry owns the live stream table + per-stream TTL cleanup tasks that
``server.py`` used to hold inline. These drive it directly — no TestClient, no
ServerState — with a tiny TTL so the expiry path runs in real time.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from strata.streaming import StreamRegistry, StreamState


def _stream(stream_id: str = "s1") -> StreamState:
    return StreamState(
        stream_id=stream_id,
        plan=SimpleNamespace(scan_id=f"scan-{stream_id}"),
        artifact_id=f"art-{stream_id}",
        artifact_version=1,
        created_at=0.0,
    )


def test_register_get_contains_pop():
    reg = StreamRegistry(ttl_seconds=60)
    st = _stream()

    assert reg.get("s1") is None
    assert "s1" not in reg

    reg.register(st)
    assert reg.get("s1") is st
    assert "s1" in reg
    assert reg.active_streams() == [st]

    assert reg.pop("s1") is st
    assert reg.get("s1") is None
    assert reg.pop("s1") is None  # idempotent


async def test_schedule_cleanup_expires_stream_and_runs_on_expire():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    await asyncio.sleep(0.05)

    # TTL elapsed → stream dropped + scan-side cleanup ran with the scan id.
    assert reg.get("s1") is None
    assert expired == ["scan-s1"]


async def test_schedule_cleanup_without_scan_skips_on_expire():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1")  # no scan_id (e.g. background-build finally)
    await asyncio.sleep(0.05)

    assert reg.get("s1") is None
    assert expired == []  # on_expire only fires when a scan id is supplied


async def test_cancel_cleanup_keeps_the_stream():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    reg.cancel_cleanup("s1")
    await asyncio.sleep(0.08)

    # Cleanup was cancelled before the TTL elapsed → stream survives, no expire.
    assert reg.get("s1") is not None
    assert expired == []


async def test_reschedule_supersedes_the_prior_timer():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    # Re-arm before the first fires; only the latest timer should remain.
    reg.schedule_cleanup("s1", scan_id="scan-s1")
    await asyncio.sleep(0.09)

    assert reg.get("s1") is None
    assert expired == ["scan-s1"]  # exactly once, not twice


async def test_shutdown_cleanups_cancels_pending_timers():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())
    reg.schedule_cleanup("s1", scan_id="scan-s1")

    reg.shutdown_cleanups()
    await asyncio.sleep(0.08)

    # Pending cleanup cancelled at shutdown → stream not expired by the timer.
    assert reg.get("s1") is not None
    assert expired == []


def test_shutdown_cleanups_is_safe_with_no_tasks():
    reg = StreamRegistry(ttl_seconds=60)
    reg.shutdown_cleanups()  # no pending tasks → no error


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])

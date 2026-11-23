"""Unit tests for StreamRegistry: the live stream table and per-stream TTL cleanup.

Driven directly, with a tiny TTL so the expiry path runs in real time.
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


async def _timers_done(*tasks: asyncio.Task) -> None:
    """Wait for cleanup timers to finish or settle their cancel, however slow the runner."""
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_schedule_cleanup_expires_stream_and_runs_on_expire():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    await _timers_done(reg._cleanup_tasks["s1"])

    # TTL elapsed: stream dropped, and scan-side cleanup ran with the scan id.
    assert reg.get("s1") is None
    assert expired == ["scan-s1"]


async def test_schedule_cleanup_recovers_the_scan_id_when_omitted():
    """Omitting scan_id must not leak the ReadPlan for the process lifetime.

    ``schedule_cleanup`` replaces the pending cleanup, and only a scan-aware one frees the plan, so
    the registry recovers the id from the registered stream.
    """
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1")  # no scan_id (e.g. background-build finally)
    await _timers_done(reg._cleanup_tasks["s1"])

    assert reg.get("s1") is None
    assert expired == ["scan-s1"], "the scan must still be expired, not leaked"


@pytest.mark.asyncio
async def test_schedule_cleanup_without_a_registered_stream_is_a_noop():
    """Nothing to recover an id from; must not raise."""
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_expire=expired.append)

    reg.schedule_cleanup("unknown")
    await _timers_done(reg._cleanup_tasks["unknown"])

    assert expired == []


async def test_cancel_cleanup_keeps_the_stream():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    timer = reg._cleanup_tasks["s1"]
    reg.cancel_cleanup("s1")
    await _timers_done(timer)

    # Cancelled before the TTL elapsed: the stream survives, no expire.
    assert reg.get("s1") is not None
    assert expired == []


async def test_reschedule_supersedes_the_prior_timer():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    first = reg._cleanup_tasks["s1"]
    # Re-arm before the first fires; only the latest timer remains.
    reg.schedule_cleanup("s1", scan_id="scan-s1")
    await _timers_done(first, reg._cleanup_tasks["s1"])

    assert reg.get("s1") is None
    assert expired == ["scan-s1"]  # exactly once, not twice


async def test_shutdown_cleanups_cancels_pending_timers():
    expired: list[str] = []
    reg = StreamRegistry(ttl_seconds=0.05, on_expire=expired.append)
    reg.register(_stream())
    reg.schedule_cleanup("s1", scan_id="scan-s1")
    timer = reg._cleanup_tasks["s1"]

    reg.shutdown_cleanups()
    await _timers_done(timer)

    # Pending cleanup cancelled at shutdown, so the timer never expires the stream.
    assert reg.get("s1") is not None
    assert expired == []


def test_shutdown_cleanups_is_safe_with_no_tasks():
    reg = StreamRegistry(ttl_seconds=60)
    reg.shutdown_cleanups()  # no pending tasks, no error


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


def test_a_cleanup_whose_loop_closed_under_it_closes_without_raising():
    """A bare ``TestClient(app)`` closes each request's loop, stranding the TTL task.

    When its coroutine is later closed outside any loop, the bookkeeping (which calls
    ``asyncio.current_task()``) must not run.
    """
    reg = StreamRegistry(ttl_seconds=60)
    reg.register(_stream())

    async def schedule() -> None:
        reg.schedule_cleanup("s1", scan_id="scan-s1")
        await asyncio.sleep(0)  # the task starts and parks in its sleep

    loop = asyncio.new_event_loop()
    loop.run_until_complete(schedule())
    loop.close()

    task = reg._cleanup_tasks["s1"]
    task._log_destroy_pending = False  # stranded on purpose; asyncio would log it
    task.get_coro().close()


async def test_an_expired_stream_is_handed_to_on_drop():
    dropped: list[StreamState] = []
    reg = StreamRegistry(ttl_seconds=0.01, on_drop=dropped.append)
    st = _stream()
    reg.register(st)

    reg.schedule_cleanup("s1", scan_id="scan-s1")
    await reg._cleanup_tasks["s1"]

    assert dropped == [st]


def test_a_stream_miss_nobody_fetches_fails_its_artifact(tmp_path, temp_warehouse):
    """Its build starts only when the stream is fetched, so without this the row stays
    ``building`` for good, holding its chain against collection.
    """
    import time

    import httpx
    from vestigo.store import ArtifactStore

    from tests.conftest import run_server_with_context

    (tmp_path / "cache").mkdir()
    (tmp_path / "artifacts").mkdir()
    with run_server_with_context(
        tmp_path / "cache", tmp_path / "artifacts", "personal", stream_state_ttl_seconds=0.1
    ) as ctx:
        response = httpx.post(
            f"{ctx.base_url}/v1/materialize",
            json={
                "inputs": [temp_warehouse["table_uri"]],
                "transform": {"executor": "scan@v1", "params": {}},
                "mode": "stream",
            },
            timeout=30,
        )
        response.raise_for_status()
        assert response.json()["state"] == "building"
        artifact_id, version = response.json()["artifact_uri"].split("/")[-1].split("@v=")
        store = ArtifactStore(tmp_path / "artifacts")

        deadline = time.monotonic() + 30
        while store.get_artifact(artifact_id, int(version)).state == "building":
            if time.monotonic() > deadline:
                break
            time.sleep(0.05)

        assert store.get_artifact(artifact_id, int(version)).state == "failed"


async def test_a_stream_is_not_expired_under_a_get_waiting_for_a_slot(
    in_process_server, temp_warehouse
):
    """A retry's 429 re-arms the cleanup while the first GET still waits for a slot. Expiring
    the stream then failed its unbuilt artifact, and the first GET, once admitted, built into
    the failed row and answered 500 (finding 17)."""
    from httpx import ASGITransport, AsyncClient
    from vestigo.store import get_artifact_store

    import strata.server as server_module
    from strata.tenancy.registry import get_tenant_registry
    from strata.tenancy.tenant import get_tenant_id

    state = in_process_server(stream_state_ttl_seconds=0.3, bulk_queue_timeout=10.0)
    store = get_artifact_store(state.config.artifact_dir)
    _, bulk = get_tenant_registry().get_or_create_limiters(get_tenant_id())
    for _ in range(bulk.capacity):  # other scans hold every bulk slot
        assert await bulk.acquire(timeout=1)

    async with AsyncClient(
        transport=ASGITransport(app=server_module.app), base_url="http://test"
    ) as client:
        materialized = (
            await client.post(
                "/v1/materialize",
                json={
                    "inputs": [temp_warehouse["table_uri"]],
                    "transform": {"executor": "scan@v1", "params": {}},
                    "mode": "stream",
                },
            )
        ).json()
        artifact_id, version = materialized["artifact_uri"].split("/")[-1].split("@v=")

        waiting = asyncio.create_task(client.get(materialized["stream_url"]))
        await asyncio.sleep(0.1)
        # The retry meets the per-client cap the first GET holds.
        assert (await client.get(materialized["stream_url"])).status_code == 429
        await asyncio.sleep(1.0)  # past the TTL the 429 re-armed

        for _ in range(bulk.capacity):
            await bulk.release()
        served = await waiting
        assert served.status_code == 200
        assert store.get_artifact(artifact_id, int(version)).state == "ready"

        # Once nobody waits, the stream is still expired.
        await asyncio.sleep(1.0)
        assert state.streams.get(materialized["stream_id"]) is None

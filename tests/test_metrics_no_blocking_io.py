"""The Prometheus scrape must not block the event loop.

``_get_cache_size_bytes`` rglobs and stats every cache file;
``_get_cache_entry_count`` rglobs and ``read_text()``s every ``.meta`` sidecar.
Called inline from an async handler they froze the loop for the duration of a
full cache walk on every scrape (Prometheus polls every ~15s), stalling
in-flight Arrow streams and risking a ``/health/ready`` timeout. The sibling
``/metrics`` handler already offloads exactly these two calls.
"""

from __future__ import annotations

import asyncio
import time

import pytest


@pytest.mark.asyncio
async def test_prometheus_scrape_does_not_block_the_event_loop(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.api.routers import metrics_health
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)

    # Stand in for a large cache: a slow, synchronous filesystem walk.
    def slow_walk(_state):
        time.sleep(0.4)
        return 123

    monkeypatch.setattr(server_module, "_get_cache_size_bytes", slow_walk)
    monkeypatch.setattr(server_module, "_get_cache_entry_count", slow_walk)

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    try:
        await metrics_health.metrics_prometheus()
    finally:
        beat.cancel()

    # If the walk ran inline, the loop would have been frozen for ~0.4s and the
    # heartbeat could not have advanced. Offloaded, it keeps ticking.
    assert ticks >= 5, f"event loop appears to have been blocked (ticks={ticks})"


@pytest.mark.asyncio
async def test_prometheus_reports_the_offloaded_values(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.api.routers import metrics_health
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"), cache_dir=tmp_path / "cache")
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)
    monkeypatch.setattr(server_module, "_get_cache_size_bytes", lambda _s: 4242)
    monkeypatch.setattr(server_module, "_get_cache_entry_count", lambda _s: 17)

    body = await metrics_health.metrics_prometheus()
    text = body.body.decode() if hasattr(body, "body") else str(body)

    assert "strata_cache_bytes_current 4242" in text
    assert "strata_cache_entries_current 17" in text

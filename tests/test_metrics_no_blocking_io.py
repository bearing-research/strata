"""The Prometheus scrape must not block the event loop.

The cache size and entry count walk every cache file; inline in an async handler that freezes the
loop on every scrape, stalling streams and risking a ``/health/ready`` timeout.
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

    # Run inline, the walk would freeze the loop for ~0.4s and the heartbeat could not
    # advance.
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


class TestPrometheusLabelEscaping:
    """Label values are escaped per the exposition format.

    An unescaped quote fails the scrape parse and drops the entire payload; a newline could inject
    fabricated series.
    """

    def test_quote_is_escaped(self):
        from strata.api.routers.metrics_health import _prom_label

        assert _prom_label('ns.a"b') == 'ns.a\\"b'

    def test_backslash_is_escaped(self):
        from strata.api.routers.metrics_health import _prom_label

        assert _prom_label("a\\b") == "a\\\\b"

    def test_newline_is_escaped(self):
        from strata.api.routers.metrics_health import _prom_label

        assert _prom_label("line1\nline2") == "line1\\nline2"
        assert "\n" not in _prom_label("line1\nline2")

    def test_ordinary_names_are_unchanged(self):
        from strata.api.routers.metrics_health import _prom_label

        assert _prom_label("test_db.events") == "test_db.events"

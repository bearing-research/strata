"""Tests for cache eviction metrics."""

import pytest


class TestEvictionEvent:
    """Tests for EvictionEvent dataclass."""

    def test_eviction_event_creation(self):
        """Test creating an eviction event."""
        from strata.cache_metrics import EvictionEvent

        event = EvictionEvent(
            timestamp=1234567890.0,
            files_evicted=5,
            bytes_evicted=1024 * 1024,
            cache_size_before=10 * 1024 * 1024,
            cache_size_after=9 * 1024 * 1024,
            reason="size_limit",
        )

        assert event.timestamp == 1234567890.0
        assert event.files_evicted == 5
        assert event.bytes_evicted == 1024 * 1024
        assert event.cache_size_before == 10 * 1024 * 1024
        assert event.cache_size_after == 9 * 1024 * 1024
        assert event.reason == "size_limit"


class TestEvictionStats:
    """Tests for EvictionStats dataclass."""

    def test_asdict(self):
        """EvictionStats serializes to a plain field mapping via asdict."""
        from dataclasses import asdict

        from strata.cache_metrics import EvictionStats

        stats = EvictionStats(
            total_evictions=100,
            total_files_evicted=500,
            total_bytes_evicted=1024 * 1024 * 100,
            evictions_last_minute=5,
            evictions_last_hour=50,
            bytes_evicted_last_minute=1024 * 1024,
            bytes_evicted_last_hour=10 * 1024 * 1024,
            eviction_rate_per_minute=0.833,
            last_eviction_at=1234567890.0,
            pressure_level="low",
        )

        d = asdict(stats)
        assert d["total_evictions"] == 100
        assert d["total_files_evicted"] == 500
        # Raw value — rounding for display is the consumer's concern.
        assert d["eviction_rate_per_minute"] == 0.833
        assert d["pressure_level"] == "low"


class TestCacheEvictionTracker:
    """Tests for CacheEvictionTracker."""

    def test_initial_state(self):
        """Test tracker starts with zero stats."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()
        stats = tracker.get_stats()

        assert stats.total_evictions == 0
        assert stats.total_files_evicted == 0
        assert stats.total_bytes_evicted == 0
        assert stats.pressure_level == "low"

    def test_record_eviction(self):
        """Test recording an eviction event."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()
        tracker.record_eviction(
            files_evicted=10,
            bytes_evicted=1024 * 1024,
            cache_size_before=100 * 1024 * 1024,
            cache_size_after=99 * 1024 * 1024,
        )

        stats = tracker.get_stats()
        assert stats.total_evictions == 1
        assert stats.total_files_evicted == 10
        assert stats.total_bytes_evicted == 1024 * 1024

    def test_multiple_evictions(self):
        """Test recording multiple eviction events."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()
        for i in range(5):
            tracker.record_eviction(
                files_evicted=i + 1,
                bytes_evicted=1024 * (i + 1),
                cache_size_before=100 * 1024,
                cache_size_after=99 * 1024,
            )

        stats = tracker.get_stats()
        assert stats.total_evictions == 5
        assert stats.total_files_evicted == 1 + 2 + 3 + 4 + 5
        assert stats.total_bytes_evicted == 1024 * (1 + 2 + 3 + 4 + 5)

    def test_pressure_levels(self):
        """Test pressure level calculation based on eviction rate."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()

        # No evictions = low pressure
        assert tracker.get_stats().pressure_level == "low"

        # Simulate high eviction rate (many events in last hour)
        # Rate = evictions_hour / 60, need rate >= 10 for critical
        for _ in range(600):  # 600 in last hour = 10/minute
            tracker.record_eviction(
                files_evicted=1,
                bytes_evicted=1024,
                cache_size_before=1000,
                cache_size_after=999,
            )

        stats = tracker.get_stats()
        assert stats.pressure_level == "critical"

    def test_recent_events(self):
        """Test getting recent eviction events."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()

        for i in range(5):
            tracker.record_eviction(
                files_evicted=i + 1,
                bytes_evicted=1024 * (i + 1),
                cache_size_before=100 * 1024,
                cache_size_after=99 * 1024,
                reason=f"reason_{i}",
            )

        events = tracker.get_recent_events(limit=3)
        assert len(events) == 3
        # Most recent first
        assert events[0]["reason"] == "reason_4"
        assert events[1]["reason"] == "reason_3"
        assert events[2]["reason"] == "reason_2"

    def test_max_events_limit(self):
        """Test that events are limited to max_events."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker(max_events=5)

        for i in range(10):
            tracker.record_eviction(
                files_evicted=i,
                bytes_evicted=i * 1024,
                cache_size_before=1000,
                cache_size_after=999,
            )

        events = tracker.get_recent_events(limit=100)
        assert len(events) == 5  # Max events is 5

    def test_reset(self):
        """Test resetting the tracker."""
        from strata.cache_metrics import CacheEvictionTracker

        tracker = CacheEvictionTracker()
        tracker.record_eviction(
            files_evicted=10,
            bytes_evicted=1024,
            cache_size_before=1000,
            cache_size_after=999,
        )

        tracker.reset()
        stats = tracker.get_stats()
        assert stats.total_evictions == 0
        assert stats.total_files_evicted == 0


class TestGlobalTracker:
    """Tests for global tracker functions."""

    def test_get_and_reset(self):
        """Test getting and resetting global tracker."""
        from strata.cache_metrics import (
            get_eviction_tracker,
            reset_eviction_tracker,
        )

        reset_eviction_tracker()
        tracker1 = get_eviction_tracker()
        tracker2 = get_eviction_tracker()

        # Should return same instance
        assert tracker1 is tracker2

        reset_eviction_tracker()
        tracker3 = get_eviction_tracker()

        # After reset, should be new instance
        assert tracker3 is not tracker1


class TestCacheEvictionIntegration:
    """Integration tests for cache eviction with server."""

    @pytest.mark.asyncio
    async def test_evictions_endpoint(self, tmp_path):
        """Test /v1/cache/evictions endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.cache_metrics import reset_eviction_tracker
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        reset_eviction_tracker()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/cache/evictions")
                assert response.status_code == 200
                data = response.json()

                assert "total_evictions" in data
                assert "total_files_evicted" in data
                assert "total_bytes_evicted" in data
                assert "evictions_last_minute" in data
                assert "evictions_last_hour" in data
                assert "pressure_level" in data
                assert data["pressure_level"] == "low"
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_evictions_endpoint_with_events(self, tmp_path):
        """Test /v1/cache/evictions with include_events=true."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.cache_metrics import get_eviction_tracker, reset_eviction_tracker
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        reset_eviction_tracker()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        # Add some eviction events
        tracker = get_eviction_tracker()
        for i in range(3):
            tracker.record_eviction(
                files_evicted=i + 1,
                bytes_evicted=1024 * (i + 1),
                cache_size_before=10000,
                cache_size_after=9000,
            )

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get(
                    "/v1/cache/evictions", params={"include_events": True, "limit": 2}
                )
                assert response.status_code == 200
                data = response.json()

                assert "recent_events" in data
                assert len(data["recent_events"]) == 2
                # Most recent first
                assert data["recent_events"][0]["files_evicted"] == 3
                assert data["recent_events"][1]["files_evicted"] == 2
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

"""Tests for rate limiting functionality."""

import pytest


class MockClock:
    """Mock clock for testing time-dependent behavior."""

    def __init__(self, start_time: float = 0.0):
        self._time = start_time

    def time(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_initial_tokens(self):
        """Test bucket starts with full capacity."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.tokens_available() == 10.0

    def test_acquire_success(self):
        """Test acquiring tokens when available."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.acquire() is True
        assert bucket.tokens_available() == 9.0

    def test_acquire_multiple(self):
        """Test acquiring multiple tokens."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.acquire(5.0) is True
        assert bucket.tokens_available() == 5.0

    def test_acquire_failure(self):
        """Test acquiring fails when not enough tokens."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        # Exhaust tokens
        for _ in range(10):
            bucket.acquire()

        assert bucket.acquire() is False
        assert bucket.tokens_available() == 0.0

    def test_refill_over_time(self):
        """Test tokens refill over time."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0, _clock=clock)

        # Exhaust tokens
        bucket.acquire(10.0)
        assert bucket.tokens_available() == 0.0

        # Advance time by 3 seconds (should add 6 tokens at 2/s)
        clock.advance(3.0)
        assert bucket.tokens_available() == 6.0

    def test_refill_caps_at_capacity(self):
        """Test refill doesn't exceed capacity."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=100.0, _clock=clock)

        bucket.acquire(5.0)
        clock.advance(10.0)  # Would add 1000 tokens

        assert bucket.tokens_available() == 10.0

    def test_time_until_available(self):
        """Test calculating time until tokens available."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0, _clock=clock)

        bucket.acquire(10.0)
        # Need 1 token, refill rate is 2/s, so 0.5s
        assert bucket.time_until_available(1.0) == pytest.approx(0.5)

        # Need 4 tokens, so 2s
        assert bucket.time_until_available(4.0) == pytest.approx(2.0)


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_default_allows_requests(self):
        """Test default config allows requests."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        config = RateLimitConfig()
        limiter = RateLimiter(config)

        result = limiter.check("client1")
        assert result.allowed is True

    def test_disabled_always_allows(self):
        """Test disabled limiter always allows."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        # Even with aggressive limits, disabled should allow
        for _ in range(1000):
            result = limiter.check("client1")
            assert result.allowed is True

    def test_global_limit_rejection(self):
        """Test global limit rejects requests."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            global_requests_per_second=1.0,
            global_burst=2.0,
            client_requests_per_second=1000.0,  # High to not interfere
            client_burst=1000.0,
        )
        limiter = RateLimiter(config, clock=clock)

        # First 2 requests allowed (burst)
        assert limiter.check("client1").allowed is True
        assert limiter.check("client1").allowed is True

        # Third request rejected
        result = limiter.check("client1")
        assert result.allowed is False
        assert result.limit_type == "global"

    def test_client_limit_rejection(self):
        """Test per-client limit rejects requests."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            global_requests_per_second=1000.0,  # High to not interfere
            global_burst=1000.0,
            client_requests_per_second=1.0,
            client_burst=2.0,
        )
        limiter = RateLimiter(config, clock=clock)

        # First 2 requests from client1 allowed
        assert limiter.check("client1").allowed is True
        assert limiter.check("client1").allowed is True

        # Third request from client1 rejected
        result = limiter.check("client1")
        assert result.allowed is False
        assert result.limit_type == "client"

        # Different client still allowed
        assert limiter.check("client2").allowed is True

    def test_endpoint_limit_rejection(self):
        """Test per-endpoint limit rejects requests."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            global_requests_per_second=1000.0,
            global_burst=1000.0,
            client_requests_per_second=1000.0,
            client_burst=1000.0,
            scan_requests_per_second=1.0,
            scan_burst=2.0,
        )
        limiter = RateLimiter(config, clock=clock)

        # First 2 materialize requests allowed
        assert limiter.check("client1", endpoint="/v1/materialize").allowed is True
        assert limiter.check("client1", endpoint="/v1/materialize").allowed is True

        # Third materialize request rejected
        result = limiter.check("client1", endpoint="/v1/materialize")
        assert result.allowed is False
        assert result.limit_type == "endpoint"

        # Other endpoints still allowed
        assert limiter.check("client1", endpoint="/health").allowed is True

    def test_retry_after_header(self):
        """Test retry-after is calculated correctly."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            client_requests_per_second=2.0,
            client_burst=1.0,
        )
        limiter = RateLimiter(config, clock=clock)

        limiter.check("client1")  # Use the one token
        result = limiter.check("client1")  # Rejected

        assert result.allowed is False
        assert result.retry_after_seconds == pytest.approx(0.5)  # 1 token / 2 per sec

    def test_stats_tracking(self):
        """Test statistics are tracked correctly."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            client_requests_per_second=1.0,
            client_burst=1.0,
        )
        limiter = RateLimiter(config, clock=clock)

        limiter.check("client1")  # Allowed
        limiter.check("client2")  # Allowed
        limiter.check("client1")  # Rejected (client limit)

        stats = limiter.get_stats()
        assert stats["total_requests"] == 3
        assert stats["allowed_requests"] == 2
        assert stats["rejected_client"] == 1
        assert stats["active_clients"] == 2

    def test_cleanup_stale_clients(self):
        """Test stale client cleanup."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(client_ttl_seconds=60.0)
        limiter = RateLimiter(config, clock=clock)

        limiter.check("client1")
        limiter.check("client2")
        assert limiter.get_stats()["active_clients"] == 2

        # Advance time past TTL
        clock.advance(61.0)

        # Adding a new client now sweeps the idle ones on the request path —
        # cleanup_stale_clients used to have no caller at all, so buckets
        # accumulated forever.
        limiter.check("client3")
        assert limiter.get_stats()["active_clients"] == 1

        # The explicit sweep is still available and idempotent.
        assert limiter.cleanup_stale_clients() == 0
        assert limiter.get_stats()["active_clients"] == 1

    def test_reset_stats(self):
        """Test resetting statistics."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        config = RateLimitConfig()
        limiter = RateLimiter(config)

        limiter.check("client1")
        limiter.check("client2")

        limiter.reset_stats()
        stats = limiter.get_stats()
        assert stats["total_requests"] == 0
        assert stats["allowed_requests"] == 0


class TestRateLimiterGlobals:
    """Tests for global rate limiter functions."""

    def test_init_and_get(self):
        """Test initializing and getting global rate limiter."""
        from strata.rate_limiter import (
            RateLimitConfig,
            get_rate_limiter,
            init_rate_limiter,
            reset_rate_limiter,
        )

        reset_rate_limiter()
        assert get_rate_limiter() is None

        config = RateLimitConfig()
        limiter = init_rate_limiter(config)

        assert get_rate_limiter() is limiter
        assert limiter.config == config

        reset_rate_limiter()
        assert get_rate_limiter() is None


class TestRateLimiterIntegration:
    """Integration tests for rate limiting with server."""

    @pytest.mark.asyncio
    async def test_rate_limit_endpoint(self, tmp_path):
        """Test /v1/debug/rate-limits endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        # Initialize rate limiter manually for test
        from strata.rate_limiter import RateLimitConfig, init_rate_limiter

        init_rate_limiter(RateLimitConfig())

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 200
                data = response.json()
                assert "total_requests" in data
                assert "allowed_requests" in data
                assert "enabled" in data
                assert data["enabled"] is True
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_rate_limiter()

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_allows(self, tmp_path):
        """Test middleware allows requests under limit."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import RateLimitConfig, init_rate_limiter, reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)
        init_rate_limiter(RateLimitConfig())

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Multiple requests should be allowed
                for _ in range(5):
                    response = await client.get("/health")
                    # Health endpoint skips rate limiting
                    assert response.status_code == 200

                # Check rate limit stats endpoint (not skipped)
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 200
                assert "X-RateLimit-Remaining" in response.headers
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_rate_limiter()

    @pytest.mark.asyncio
    async def test_rate_limit_middleware_rejects(self, tmp_path):
        """Test middleware rejects requests over limit."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import RateLimitConfig, init_rate_limiter, reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        # Very restrictive config
        init_rate_limiter(
            RateLimitConfig(
                client_requests_per_second=1.0,
                client_burst=1.0,
            )
        )

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # First request allowed
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 200

                # Second request should be rate limited
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 429
                assert "Retry-After" in response.headers
                assert "Rate limit exceeded" in response.text
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_rate_limiter()


class TestClientBucketGrowthIsBounded:
    """The per-client bucket tables must not grow without bound.

    ``cleanup_stale_clients`` existed with a TTL config but had **no caller**
    anywhere in the codebase, and nothing else bounded the two dicts. The
    client id is derived from ``X-Forwarded-For``, which the caller controls
    whenever a proxy appends rather than replaces it — so a client sending a
    distinct forwarded address per request created a permanent bucket plus a
    timestamp entry every time: unbounded RSS growth, and (because each new id
    starts with a full burst) a per-client limit that never limited.
    """

    def test_idle_buckets_are_reclaimed_without_an_explicit_call(self):
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        limiter = RateLimiter(
            RateLimitConfig(client_ttl_seconds=60.0, cleanup_interval_seconds=10.0),
            clock=clock,
        )

        for i in range(50):
            limiter.check(f"client-{i}")
        assert limiter.get_stats()["active_clients"] == 50

        clock.advance(61.0)
        limiter.check("fresh")

        assert limiter.get_stats()["active_clients"] == 1

    def test_distinct_ids_cannot_grow_past_the_ceiling(self):
        """The TTL sweep alone is not a bound — ids can be minted faster than
        they age out, which is exactly what a spoofed X-Forwarded-For does."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        limiter = RateLimiter(
            RateLimitConfig(
                client_ttl_seconds=3600.0,  # nothing ages out during the test
                max_tracked_clients=25,
            ),
            clock=clock,
        )

        for i in range(500):
            limiter.check(f"10.0.0.{i}")

        assert limiter.get_stats()["active_clients"] <= 25

    def test_active_client_keeps_its_bucket_under_eviction_pressure(self):
        """Eviction drops the least-recently-seen, so a steadily active client
        is not the one sacrificed."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        limiter = RateLimiter(
            RateLimitConfig(client_ttl_seconds=3600.0, max_tracked_clients=10),
            clock=clock,
        )

        for i in range(40):
            limiter.check("steady")  # touched throughout
            clock.advance(1.0)
            limiter.check(f"churn-{i}")

        assert "steady" in limiter._client_buckets


class TestRetryAfterIsNeverZero:
    """A 429 must never tell the client to retry immediately.

    ``Retry-After``'s grammar is whole seconds, and the middleware used to
    render the limiter's precise wait with ``int()``. The largest wait the
    limiter can ever compute is one token's worth of refill -- 0.01s at the
    default 100 requests/sec -- so truncation did not merely lose precision,
    it produced ``Retry-After: 0`` on every single rejection. The one existing
    middleware test used a 1 req/sec limit, the only rate whose wait survives
    truncation, and asserted the header was present rather than what it said.
    """

    def test_subsecond_waits_round_up(self):
        from strata.server import _retry_after_header

        assert _retry_after_header(0.001, 1.0) == "1"  # global default
        assert _retry_after_header(0.01, 1.0) == "1"  # per-client default
        assert _retry_after_header(0.1, 1.0) == "1"  # warm endpoint default

    def test_whole_and_partial_seconds_round_up(self):
        from strata.server import _retry_after_header

        assert _retry_after_header(1.0, 5.0) == "1"
        assert _retry_after_header(1.2, 5.0) == "2"
        assert _retry_after_header(30.0, 5.0) == "30"

    def test_missing_wait_uses_the_fallback(self):
        from strata.server import _retry_after_header

        assert _retry_after_header(None, 5.0) == "5"

    def test_zero_wait_still_floors_at_one_second(self):
        """``time_until_available`` can refill to 0.0 between the failed
        acquire and the header being built."""
        from strata.server import _retry_after_header

        assert _retry_after_header(0.0, 1.0) == "1"

    @pytest.mark.asyncio
    async def test_middleware_sends_a_usable_retry_after(self, tmp_path):
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import RateLimitConfig, init_rate_limiter, reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        # The default refill rate, not the 1/sec the older test uses -- the bug
        # only appears once the rate is above one token per second.
        init_rate_limiter(
            RateLimitConfig(
                client_requests_per_second=100.0,
                client_burst=1.0,
            )
        )

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                assert (await client.get("/v1/debug/rate-limits")).status_code == 200

                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 429
                assert int(response.headers["Retry-After"]) >= 1
                assert "Retry after 0s" not in response.text
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_rate_limiter()

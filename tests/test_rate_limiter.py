"""Tests for rate limiting functionality."""

import pytest


class MockClock:
    """A controllable clock for time-dependent tests."""

    def __init__(self, start_time: float = 0.0):
        self._time = start_time

    def time(self) -> float:
        return self._time

    def advance(self, seconds: float) -> None:
        self._time += seconds


class TestTokenBucket:
    def test_initial_tokens(self):
        """The bucket starts full."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.tokens_available() == 10.0

    def test_acquire_success(self):
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.acquire() is True
        assert bucket.tokens_available() == 9.0

    def test_acquire_multiple(self):
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        assert bucket.acquire(5.0) is True
        assert bucket.tokens_available() == 5.0

    def test_acquire_failure(self):
        """Acquire fails without enough tokens."""
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=1.0, _clock=clock)

        for _ in range(10):
            bucket.acquire()

        assert bucket.acquire() is False
        assert bucket.tokens_available() == 0.0

    def test_refill_over_time(self):
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0, _clock=clock)

        bucket.acquire(10.0)
        assert bucket.tokens_available() == 0.0

        # 3 seconds at 2/s adds 6 tokens.
        clock.advance(3.0)
        assert bucket.tokens_available() == 6.0

    def test_refill_caps_at_capacity(self):
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=100.0, _clock=clock)

        bucket.acquire(5.0)
        clock.advance(10.0)  # Would add 1000 tokens

        assert bucket.tokens_available() == 10.0

    def test_time_until_available(self):
        from strata.rate_limiter import TokenBucket

        clock = MockClock()
        bucket = TokenBucket(capacity=10.0, refill_rate=2.0, _clock=clock)

        bucket.acquire(10.0)
        # 1 token at 2/s takes 0.5s.
        assert bucket.time_until_available(1.0) == pytest.approx(0.5)

        # 4 tokens takes 2s.
        assert bucket.time_until_available(4.0) == pytest.approx(2.0)


class TestRateLimiter:
    def test_default_allows_requests(self):
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        config = RateLimitConfig()
        limiter = RateLimiter(config)

        result = limiter.check("client1")
        assert result.allowed is True

    def test_disabled_always_allows(self):
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        # Even aggressive limits allow everything when disabled.
        for _ in range(1000):
            result = limiter.check("client1")
            assert result.allowed is True

    def test_global_limit_rejection(self):
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            global_requests_per_second=1.0,
            global_burst=2.0,
            client_requests_per_second=1000.0,  # High, so it does not interfere
            client_burst=1000.0,
        )
        limiter = RateLimiter(config, clock=clock)

        # First 2 allowed (burst).
        assert limiter.check("client1").allowed is True
        assert limiter.check("client1").allowed is True

        result = limiter.check("client1")
        assert result.allowed is False
        assert result.limit_type == "global"

    def test_client_limit_rejection(self):
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            global_requests_per_second=1000.0,  # High, so it does not interfere
            global_burst=1000.0,
            client_requests_per_second=1.0,
            client_burst=2.0,
        )
        limiter = RateLimiter(config, clock=clock)

        assert limiter.check("client1").allowed is True
        assert limiter.check("client1").allowed is True

        result = limiter.check("client1")
        assert result.allowed is False
        assert result.limit_type == "client"

        assert limiter.check("client2").allowed is True

    def test_endpoint_limit_rejection(self):
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

        assert limiter.check("client1", endpoint="/v1/materialize").allowed is True
        assert limiter.check("client1", endpoint="/v1/materialize").allowed is True

        result = limiter.check("client1", endpoint="/v1/materialize")
        assert result.allowed is False
        assert result.limit_type == "endpoint"

        assert limiter.check("client1", endpoint="/health").allowed is True

    def test_retry_after_header(self):
        """retry-after is calculated correctly."""
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(
            client_requests_per_second=2.0,
            client_burst=1.0,
        )
        limiter = RateLimiter(config, clock=clock)

        limiter.check("client1")  # Uses the one token
        result = limiter.check("client1")  # Rejected

        assert result.allowed is False
        assert result.retry_after_seconds == pytest.approx(0.5)  # 1 token / 2 per sec

    def test_stats_tracking(self):
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
        from strata.rate_limiter import RateLimitConfig, RateLimiter

        clock = MockClock()
        config = RateLimitConfig(client_ttl_seconds=60.0)
        limiter = RateLimiter(config, clock=clock)

        limiter.check("client1")
        limiter.check("client2")
        assert limiter.get_stats()["active_clients"] == 2

        clock.advance(61.0)

        # Adding a new client sweeps the idle ones on the request path, so buckets do
        # not accumulate forever.
        limiter.check("client3")
        assert limiter.get_stats()["active_clients"] == 1

        # The explicit sweep still works and is idempotent.
        assert limiter.cleanup_stale_clients() == 0
        assert limiter.get_stats()["active_clients"] == 1

    def test_reset_stats(self):
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
    def test_init_and_get(self):
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


class _SteppableClock:
    """A clock that moves only when the test says so, so no bucket refills unseen."""

    def __init__(self, now: float = 1000.0) -> None:
        self._now = now

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


class TestRateLimiterIntegration:
    """Rate limiting through the server."""

    @pytest.mark.asyncio
    async def test_rate_limit_endpoint(self, tmp_path):
        """/v1/debug/rate-limits."""
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
        """The middleware allows requests under the limit."""
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
                for _ in range(5):
                    response = await client.get("/health")
                    # Health skips rate limiting.
                    assert response.status_code == 200

                # The stats endpoint is not skipped.
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
        """The middleware rejects requests over the limit."""
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

        # Restrictive config on a hand-stepped clock. The bucket refills on wall-clock
        # time, so with a real clock a slow ASGI startup between the two requests refills
        # the token and the second legitimately succeeds.
        clock = _SteppableClock()
        init_rate_limiter(
            RateLimitConfig(
                client_requests_per_second=1.0,
                client_burst=1.0,
            ),
            clock=clock,
        )

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 200

                # The bucket is empty, and no time has passed.
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 429
                assert "Retry-After" in response.headers
                assert "Rate limit exceeded" in response.text

                # One token per second, so advancing past the refill re-admits. This also proves
                # the limiter reads the injected clock: a frozen clock alone would pass even if
                # the limiter used system time.
                clock.advance(1.0)
                response = await client.get("/v1/debug/rate-limits")
                assert response.status_code == 200
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_rate_limiter()


class TestClientBucketGrowthIsBounded:
    """The per-client bucket tables must not grow without bound.

    The client id comes from ``X-Forwarded-For``, which a caller may control, so a distinct address
    per request would mint a bucket each time: unbounded memory and a limit that never limits.
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
        """The TTL sweep is not a bound: spoofed ids arrive faster than they age out."""
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
        """Eviction drops the least recently seen, not a steadily active client."""
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

    ``Retry-After`` is whole seconds, and the longest wait at the default 100 req/s is 0.01s, so
    truncating with ``int()`` gives 0 on every rejection. Only a 1 req/s limit hides it.
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
        """The bucket can refill to 0.0 between the failed acquire and the header."""
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

        # The default refill rate, not 1/sec: the bug only appears above one token per
        # second. At 100/sec the token refills in 10ms, so a fixed clock pins the bucket
        # empty rather than racing the refill between the two requests.
        init_rate_limiter(
            RateLimitConfig(
                client_requests_per_second=100.0,
                client_burst=1.0,
            ),
            clock=MockClock(start_time=1000.0),
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

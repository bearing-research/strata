"""Tests for circuit breaker."""

import time

import pytest


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_circuit_state_values(self):
        """Test CircuitState enum values."""
        from strata.circuit_breaker import CircuitState

        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from strata.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.success_threshold == 3
        assert config.reset_timeout_seconds == 30.0
        assert config.name == "default"

    def test_custom_config(self):
        """Test custom configuration values."""
        from strata.circuit_breaker import CircuitBreakerConfig

        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            reset_timeout_seconds=10.0,
            name="test_breaker",
        )
        assert config.failure_threshold == 3
        assert config.success_threshold == 2
        assert config.reset_timeout_seconds == 10.0
        assert config.name == "test_breaker"


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_initial_state_closed(self):
        """Test circuit starts in closed state."""
        from strata.circuit_breaker import CircuitBreaker, CircuitState

        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_allows_requests_when_closed(self):
        """Test requests are allowed when circuit is closed."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()
        assert breaker.allow_request() is True

    def test_opens_after_failure_threshold(self):
        """Test circuit opens after reaching failure threshold."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        # Record failures
        for _ in range(3):
            breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

    def test_rejects_requests_when_open(self):
        """Test requests are rejected when circuit is open."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.allow_request() is False

        stats = breaker.get_stats()
        assert stats.total_rejections == 1

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to half-open after reset timeout."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2, reset_timeout_seconds=0.1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should transition to half-open
        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.allow_request() is True

    def test_closes_after_success_threshold_in_half_open(self):
        """Test circuit closes after successes in half-open state."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2, success_threshold=2, reset_timeout_seconds=0.1
        )
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Record successes
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_opens_immediately_on_failure_in_half_open(self):
        """Test circuit opens immediately on failure in half-open state."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2, reset_timeout_seconds=0.1)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()

        # Wait for half-open
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Single failure should re-open
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count_when_closed(self):
        """Test success resets failure count in closed state."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(config)

        # Some failures
        breaker.record_failure()
        breaker.record_failure()

        # Success should reset
        breaker.record_success()

        # More failures shouldn't open yet
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        # Now it should open
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerCallMethod:
    """Tests for CircuitBreaker.call() method."""

    def test_call_success(self):
        """Test successful call through circuit breaker."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        def success_func():
            return "success"

        result = breaker.call(success_func)
        assert result == "success"

        stats = breaker.get_stats()
        assert stats.total_successes == 1
        assert stats.total_failures == 0

    def test_call_failure(self):
        """Test failed call through circuit breaker."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        def failure_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            breaker.call(failure_func)

        stats = breaker.get_stats()
        assert stats.total_failures == 1

    def test_call_when_open_raises(self):
        """Test call raises CircuitOpenError when circuit is open."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitOpenError,
        )

        config = CircuitBreakerConfig(failure_threshold=1, name="test")
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()

        def success_func():
            return "success"

        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(success_func)

        assert exc_info.value.name == "test"


class TestCircuitBreakerContextManager:
    """Tests for CircuitBreaker context manager."""

    def test_context_manager_success(self):
        """Test context manager records success."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        with breaker:
            pass  # Success

        stats = breaker.get_stats()
        assert stats.total_successes == 1

    def test_context_manager_failure(self):
        """Test context manager records failure on exception."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("test")

        stats = breaker.get_stats()
        assert stats.total_failures == 1

    def test_context_manager_when_open_raises(self):
        """Test context manager raises when circuit is open."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitOpenError,
        )

        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)
        breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            with breaker:
                pass


class TestCircuitBreakerDecorator:
    """Tests for CircuitBreaker as decorator."""

    def test_decorator_success(self):
        """Test decorator records success."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        @breaker
        def my_func():
            return "result"

        result = my_func()
        assert result == "result"

        stats = breaker.get_stats()
        assert stats.total_successes == 1

    def test_decorator_failure(self):
        """Test decorator records failure."""
        from strata.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker()

        @breaker
        def my_func():
            raise ValueError("error")

        with pytest.raises(ValueError):
            my_func()

        stats = breaker.get_stats()
        assert stats.total_failures == 1

    def test_decorator_when_open_raises(self):
        """Test decorator raises when circuit is open."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitOpenError,
        )

        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(config)

        @breaker
        def my_func():
            return "result"

        # Open the circuit
        breaker.record_failure()

        with pytest.raises(CircuitOpenError):
            my_func()


class TestCircuitStats:
    """Tests for CircuitStats."""

    def test_stats_asdict(self):
        """CircuitStats serializes to a plain field mapping via asdict."""
        from dataclasses import asdict

        from strata.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        breaker.record_success()
        breaker.record_failure()

        stats = breaker.get_stats()
        d = asdict(stats)

        assert d["state"] == "closed"
        assert d["total_calls"] == 2
        assert d["total_successes"] == 1
        assert d["total_failures"] == 1
        assert "last_success_at" in d
        assert "last_failure_at" in d


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset."""

    def test_reset(self):
        """Test resetting circuit breaker."""
        from strata.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker(config)

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        stats = breaker.get_stats()
        assert stats.total_calls == 0
        assert stats.total_failures == 0


class TestCircuitBreakerRegistry:
    """Tests for CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test getting or creating circuit breakers."""
        from strata.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("test1")
        cb2 = registry.get_or_create("test1")

        assert cb1 is cb2  # Same instance

        cb3 = registry.get_or_create("test2", CircuitBreakerConfig(failure_threshold=10))
        assert cb3 is not cb1
        assert cb3.config.failure_threshold == 10

    def test_get_all_stats(self):
        """Test getting stats for all breakers."""
        from strata.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("breaker1")
        cb2 = registry.get_or_create("breaker2")

        cb1.record_success()
        cb2.record_failure()

        all_stats = registry.get_all_stats()

        assert "breaker1" in all_stats
        assert "breaker2" in all_stats
        assert all_stats["breaker1"]["total_successes"] == 1
        assert all_stats["breaker2"]["total_failures"] == 1

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        from strata.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("breaker1")
        cb1.record_success()

        registry.reset_all()

        assert registry.get("breaker1") is None


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def test_get_and_reset(self):
        """Test getting and resetting global registry."""
        from strata.circuit_breaker import (
            get_circuit_breaker,
            get_circuit_breaker_registry,
            reset_circuit_breakers,
        )

        reset_circuit_breakers()

        cb1 = get_circuit_breaker("test")
        cb2 = get_circuit_breaker("test")

        assert cb1 is cb2

        registry = get_circuit_breaker_registry()
        assert registry.get("test") is cb1

        reset_circuit_breakers()
        cb3 = get_circuit_breaker("test")
        assert cb3 is not cb1


class TestCircuitBreakerIntegration:
    """Integration tests for circuit breaker with server."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_endpoint(self, tmp_path):
        """Test /v1/debug/circuit-breakers endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.cache_metrics import reset_eviction_tracker
        from strata.circuit_breaker import get_circuit_breaker, reset_circuit_breakers
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.rate_limiter import reset_rate_limiter
        from strata.server import ServerState, app

        reset_metrics()
        reset_rate_limiter()
        reset_eviction_tracker()
        reset_circuit_breakers()

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        config = StrataConfig(cache_dir=cache_dir)
        server_module._state = ServerState(config)

        # Create some circuit breakers
        cb1 = get_circuit_breaker("s3")
        cb2 = get_circuit_breaker("metadata")
        cb1.record_success()
        cb2.record_failure()

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/circuit-breakers")
                assert response.status_code == 200
                data = response.json()

                assert "breakers" in data
                assert "s3" in data["breakers"]
                assert "metadata" in data["breakers"]
                assert data["breakers"]["s3"]["total_successes"] == 1
                assert data["breakers"]["metadata"]["total_failures"] == 1
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None
            reset_circuit_breakers()

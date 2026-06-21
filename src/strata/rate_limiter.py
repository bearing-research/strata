"""Rate limiting for request throttling.

Implements a token bucket algorithm for per-client rate limiting.
This complements the existing QoS admission control by adding
request-rate throttling to prevent abuse and ensure fair usage.
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Protocol


class Clock(Protocol):
    """Protocol for time sources (for testing)."""

    def time(self) -> float:
        """Return current time in seconds."""
        ...


class SystemClock:
    """Default clock using system time."""

    def time(self) -> float:
        return time.time()


@dataclass
class TokenBucket:
    """Token bucket for rate limiting.

    Tokens refill at a fixed rate up to ``capacity``; each request consumes one
    token, and a request with no token available is rejected.

    Attributes
    ----------
    capacity : float
        Maximum tokens (the burst ceiling).
    refill_rate : float
        Tokens added per second.
    tokens : float
        Current token count (initialized to ``capacity``).
    last_update : float
        Timestamp of the last refill.
    """

    capacity: float
    refill_rate: float
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _clock: Clock = field(default_factory=SystemClock)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self.last_update = self._clock.time()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = self._clock.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to consume ``tokens``, refilling first.

        Parameters
        ----------
        tokens : float, optional
            Tokens to consume (default 1.0).

        Returns
        -------
        bool
            ``True`` if enough tokens were available and consumed.
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def tokens_available(self) -> float:
        """Return current number of available tokens."""
        self._refill()
        return self.tokens

    def time_until_available(self, tokens: float = 1.0) -> float:
        """Return seconds until ``tokens`` will have refilled.

        Parameters
        ----------
        tokens : float, optional
            Tokens needed (default 1.0).

        Returns
        -------
        float
            Seconds to wait (0.0 if already available).
        """
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        needed = tokens - self.tokens
        return needed / self.refill_rate


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Global rate limits (all clients combined)
    global_requests_per_second: float = 1000.0
    global_burst: float = 100.0  # Max burst above rate

    # Per-client rate limits
    client_requests_per_second: float = 100.0
    client_burst: float = 20.0

    # Per-endpoint rate limits (optional overrides)
    scan_requests_per_second: float = 50.0
    scan_burst: float = 10.0
    warm_requests_per_second: float = 10.0
    warm_burst: float = 5.0

    # Cleanup settings
    client_ttl_seconds: float = 300.0  # Remove idle clients after 5 minutes

    # Whether rate limiting is enabled
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Outcome of a rate-limit check.

    Attributes
    ----------
    allowed : bool
        Whether the request may proceed.
    limit_type : str or None
        Which limit rejected the request: ``"global"`` / ``"client"`` /
        ``"endpoint"`` (``None`` when allowed).
    retry_after_seconds : float or None
        Seconds to wait before retrying, when rejected.
    tokens_remaining : float or None
        Client tokens left after the check.
    """

    allowed: bool
    limit_type: str | None = None
    retry_after_seconds: float | None = None
    tokens_remaining: float | None = None


class RateLimiter:
    """Multi-level rate limiter with global, per-client, and per-endpoint limits."""

    def __init__(
        self,
        config: RateLimitConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self.config = config or RateLimitConfig()
        self._clock = clock or SystemClock()
        self._lock = Lock()

        # Global bucket
        self._global_bucket = TokenBucket(
            capacity=self.config.global_burst,
            refill_rate=self.config.global_requests_per_second,
            _clock=self._clock,
        )

        # Per-client buckets
        self._client_buckets: dict[str, TokenBucket] = {}
        self._client_last_seen: dict[str, float] = {}

        # Per-endpoint buckets
        self._endpoint_buckets: dict[str, TokenBucket] = {}

        # Stats
        self._stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rejected_global": 0,
            "rejected_client": 0,
            "rejected_endpoint": 0,
        }

    def _get_client_bucket(self, client_id: str) -> TokenBucket:
        """Get or create a bucket for a client."""
        if client_id not in self._client_buckets:
            self._client_buckets[client_id] = TokenBucket(
                capacity=self.config.client_burst,
                refill_rate=self.config.client_requests_per_second,
                _clock=self._clock,
            )
        self._client_last_seen[client_id] = self._clock.time()
        return self._client_buckets[client_id]

    def _get_endpoint_bucket(self, endpoint: str) -> TokenBucket | None:
        """Get or create a bucket for an endpoint (if configured)."""
        # Both /v1/scan (legacy) and /v1/materialize use the same rate limit
        if endpoint in ("/v1/scan", "/v1/materialize"):
            key = "/v1/materialize"  # Use single bucket for both endpoints
            if key not in self._endpoint_buckets:
                self._endpoint_buckets[key] = TokenBucket(
                    capacity=self.config.scan_burst,
                    refill_rate=self.config.scan_requests_per_second,
                    _clock=self._clock,
                )
            return self._endpoint_buckets[key]
        elif endpoint.startswith("/v1/cache/warm"):
            key = "/v1/cache/warm"
            if key not in self._endpoint_buckets:
                self._endpoint_buckets[key] = TokenBucket(
                    capacity=self.config.warm_burst,
                    refill_rate=self.config.warm_requests_per_second,
                    _clock=self._clock,
                )
            return self._endpoint_buckets[key]
        return None

    def check(
        self,
        client_id: str,
        endpoint: str | None = None,
    ) -> RateLimitResult:
        """Check a request against the global, per-client, and endpoint limits.

        Parameters
        ----------
        client_id : str
            Unique client identifier (e.g. IP address).
        endpoint : str or None, optional
            Endpoint path for endpoint-specific limits.

        Returns
        -------
        RateLimitResult
            Whether the request is allowed, and why if not.
        """
        if not self.config.enabled:
            return RateLimitResult(allowed=True)

        with self._lock:
            self._stats["total_requests"] += 1

            # Check global limit first
            if not self._global_bucket.acquire():
                self._stats["rejected_global"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit_type="global",
                    retry_after_seconds=self._global_bucket.time_until_available(),
                    tokens_remaining=0,
                )

            # Check per-client limit
            client_bucket = self._get_client_bucket(client_id)
            if not client_bucket.acquire():
                self._stats["rejected_client"] += 1
                return RateLimitResult(
                    allowed=False,
                    limit_type="client",
                    retry_after_seconds=client_bucket.time_until_available(),
                    tokens_remaining=0,
                )

            # Check per-endpoint limit (if applicable)
            if endpoint:
                endpoint_bucket = self._get_endpoint_bucket(endpoint)
                if endpoint_bucket and not endpoint_bucket.acquire():
                    self._stats["rejected_endpoint"] += 1
                    return RateLimitResult(
                        allowed=False,
                        limit_type="endpoint",
                        retry_after_seconds=endpoint_bucket.time_until_available(),
                        tokens_remaining=0,
                    )

            self._stats["allowed_requests"] += 1
            return RateLimitResult(
                allowed=True,
                tokens_remaining=client_bucket.tokens_available(),
            )

    def cleanup_stale_clients(self) -> int:
        """Drop per-client buckets idle longer than ``client_ttl_seconds``.

        Returns
        -------
        int
            Number of client buckets removed.
        """
        with self._lock:
            now = self._clock.time()
            stale = [
                client_id
                for client_id, last_seen in self._client_last_seen.items()
                if now - last_seen > self.config.client_ttl_seconds
            ]
            for client_id in stale:
                del self._client_buckets[client_id]
                del self._client_last_seen[client_id]
            return len(stale)

    def get_stats(self) -> dict[str, Any]:
        """Return request/rejection counters and current bucket state.

        Returns
        -------
        dict
            Counters plus ``active_clients``, ``global_tokens_available``, and
            ``enabled``.
        """
        with self._lock:
            return {
                **self._stats,
                "active_clients": len(self._client_buckets),
                "global_tokens_available": self._global_bucket.tokens_available(),
                "enabled": self.config.enabled,
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._stats = {
                "total_requests": 0,
                "allowed_requests": 0,
                "rejected_global": 0,
                "rejected_client": 0,
                "rejected_endpoint": 0,
            }


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter | None:
    """Return the global rate limiter, or ``None`` if not initialized.

    Returns
    -------
    RateLimiter or None
        The process-wide limiter.
    """
    return _rate_limiter


def init_rate_limiter(config: RateLimitConfig | None = None) -> RateLimiter:
    """Create and install the global rate limiter.

    Parameters
    ----------
    config : RateLimitConfig or None, optional
        Configuration; defaults are used when ``None``.

    Returns
    -------
    RateLimiter
        The newly installed limiter.
    """
    global _rate_limiter
    _rate_limiter = RateLimiter(config)
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None

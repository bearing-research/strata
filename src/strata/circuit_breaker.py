"""Circuit breaker for external dependency protection.

Implements the circuit breaker pattern to protect against cascading failures
when external dependencies (S3, the metadata store) become unavailable. A
breaker is ``CLOSED`` in normal operation, trips to ``OPEN`` (fail fast) after
enough failures, and probes recovery in ``HALF_OPEN`` before closing again.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from threading import Lock
from typing import Any, TypeVar

T = TypeVar("T")


class CircuitState(StrEnum):
    """State of a circuit breaker."""

    CLOSED = "closed"  # normal operation, requests pass through
    OPEN = "open"  # failing fast, requests rejected
    HALF_OPEN = "half_open"  # probing whether the dependency has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker.

    Attributes
    ----------
    failure_threshold : int
        Consecutive failures (while closed) that trip the circuit open.
    success_threshold : int
        Consecutive successes (while half-open) that close the circuit.
    reset_timeout_seconds : float
        Time to wait while open before probing recovery (half-open).
    name : str
        Identifier used in errors and metrics.
    """

    failure_threshold: int = 5
    success_threshold: int = 3
    reset_timeout_seconds: float = 30.0
    name: str = "default"


@dataclass
class CircuitStats:
    """Point-in-time snapshot of a circuit breaker's state and counters.

    Attributes
    ----------
    state : CircuitState
        Current state.
    failure_count : int
        Failures in the current closed/half-open window.
    success_count : int
        Successes in the current half-open window.
    total_calls : int
        Lifetime calls recorded.
    total_failures : int
        Lifetime failures recorded.
    total_successes : int
        Lifetime successes recorded.
    total_rejections : int
        Lifetime calls rejected while open.
    last_failure_at : float or None
        Timestamp of the most recent failure, or ``None``.
    last_success_at : float or None
        Timestamp of the most recent success, or ``None``.
    opened_at : float or None
        When the circuit last opened, or ``None``.
    half_opened_at : float or None
        When the circuit last entered half-open, or ``None``.
    """

    state: CircuitState
    failure_count: int
    success_count: int
    total_calls: int
    total_failures: int
    total_successes: int
    total_rejections: int
    last_failure_at: float | None
    last_success_at: float | None
    opened_at: float | None
    half_opened_at: float | None


class CircuitOpenError(Exception):
    """Raised when a call is rejected because the circuit is open.

    Parameters
    ----------
    name : str
        Name of the circuit breaker that rejected the call.
    message : str, optional
        Override message; a default is built from ``name`` when omitted.
    """

    def __init__(self, name: str, message: str = ""):
        self.name = name
        self.message = message or f"Circuit breaker '{name}' is open"
        super().__init__(self.message)


class CircuitBreaker:
    """Protects an external dependency with the circuit breaker pattern.

    Examples
    --------
    >>> breaker = CircuitBreaker(CircuitBreakerConfig(name="s3"))
    >>> with breaker:                       # context manager
    ...     result = call_external_service()
    >>> @breaker                            # decorator
    ... def call_external_service():
    ...     ...
    >>> result = breaker.call(call_external_service)   # call method
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        """Initialize a breaker in the closed state.

        Parameters
        ----------
        config : CircuitBreakerConfig, optional
            Thresholds and timeout. Defaults to ``CircuitBreakerConfig()``.
        """
        self.config = config or CircuitBreakerConfig()
        self._lock = Lock()

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0

        # Timestamps
        self._opened_at: float | None = None
        self._half_opened_at: float | None = None
        self._last_failure_at: float | None = None
        self._last_success_at: float | None = None

        # Lifetime counters
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        """Current state, transitioning open → half-open if the timeout expired.

        Returns
        -------
        CircuitState
            The (possibly just-transitioned) state.
        """
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
            return self._state

    def _should_attempt_reset(self) -> bool:
        """Return whether the reset timeout has elapsed since opening."""
        if self._opened_at is None:
            return False
        elapsed = time.time() - self._opened_at
        return elapsed >= self.config.reset_timeout_seconds

    def _transition_to_half_open(self) -> None:
        """Move to half-open and reset the window counters."""
        self._state = CircuitState.HALF_OPEN
        self._half_opened_at = time.time()
        self._success_count = 0
        self._failure_count = 0

    def _transition_to_open(self) -> None:
        """Move to open and stamp the open time."""
        self._state = CircuitState.OPEN
        self._opened_at = time.time()
        self._failure_count = 0
        self._success_count = 0

    def _transition_to_closed(self) -> None:
        """Move to closed and clear the open/half-open timestamps."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._opened_at = None
        self._half_opened_at = None

    def record_success(self) -> None:
        """Record a successful call and update state.

        In half-open, enough consecutive successes close the circuit; in closed,
        a success resets the failure window.
        """
        with self._lock:
            self._total_calls += 1
            self._total_successes += 1
            self._last_success_at = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call and update state.

        Any failure in half-open re-opens the circuit; in closed, the circuit
        opens once the failure threshold is reached.
        """
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._last_failure_at = time.time()
            self._failure_count += 1

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to_open()
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def allow_request(self) -> bool:
        """Return whether a request may proceed, counting rejections when open.

        Returns
        -------
        bool
            ``True`` when closed or half-open; ``False`` when open.
        """
        current_state = self.state  # may transition open → half-open

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            with self._lock:
                self._total_rejections += 1
            return False
        else:  # HALF_OPEN
            return True  # allow requests to test recovery

    def call(self, func: Callable[[], T]) -> T:
        """Execute ``func`` under circuit-breaker protection.

        Parameters
        ----------
        func : callable
            Zero-argument function to execute.

        Returns
        -------
        T
            The function's result.

        Raises
        ------
        CircuitOpenError
            If the circuit is open.
        Exception
            Whatever ``func`` raises (also recorded as a failure).
        """
        if not self.allow_request():
            raise CircuitOpenError(self.config.name)

        try:
            result = func()
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    def __enter__(self) -> "CircuitBreaker":
        """Enter the protected block, raising if the circuit is open.

        Returns
        -------
        CircuitBreaker
            This breaker.

        Raises
        ------
        CircuitOpenError
            If the circuit is open.
        """
        if not self.allow_request():
            raise CircuitOpenError(self.config.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Record success or failure on exit; never suppress the exception."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()
        return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Wrap ``func`` so each call is circuit-breaker protected.

        Parameters
        ----------
        func : callable
            Function to wrap.

        Returns
        -------
        callable
            The protected wrapper.
        """

        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitOpenError(self.config.name)
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception:
                self.record_failure()
                raise

        return wrapper

    def get_stats(self) -> CircuitStats:
        """Return a snapshot of the breaker's current state and counters.

        Returns
        -------
        CircuitStats
            The snapshot.
        """
        with self._lock:
            return CircuitStats(
                state=self._state,
                failure_count=self._failure_count,
                success_count=self._success_count,
                total_calls=self._total_calls,
                total_failures=self._total_failures,
                total_successes=self._total_successes,
                total_rejections=self._total_rejections,
                last_failure_at=self._last_failure_at,
                last_success_at=self._last_success_at,
                opened_at=self._opened_at,
                half_opened_at=self._half_opened_at,
            )

    def reset(self) -> None:
        """Reset to the initial closed state and clear all counters."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._opened_at = None
            self._half_opened_at = None
            self._last_failure_at = None
            self._last_success_at = None
            self._total_calls = 0
            self._total_failures = 0
            self._total_successes = 0
            self._total_rejections = 0


@dataclass
class CircuitBreakerRegistry:
    """Holds and reuses circuit breakers by name.

    Attributes
    ----------
    breakers : dict of str to CircuitBreaker
        Registered breakers keyed by name.
    """

    breakers: dict[str, CircuitBreaker] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def get_or_create(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Return the breaker for ``name``, creating it on first use.

        Parameters
        ----------
        name : str
            Breaker name.
        config : CircuitBreakerConfig, optional
            Configuration, used only when creating a new breaker.

        Returns
        -------
        CircuitBreaker
            The existing or newly created breaker.
        """
        with self._lock:
            if name not in self.breakers:
                cfg = config or CircuitBreakerConfig(name=name)
                self.breakers[name] = CircuitBreaker(cfg)
            return self.breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Return the breaker for ``name``, or ``None`` if unregistered."""
        return self.breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return each breaker's stats snapshot, keyed by name.

        Returns
        -------
        dict of str to dict
            ``name -> asdict(CircuitStats)`` for every registered breaker.
        """
        with self._lock:
            return {name: asdict(cb.get_stats()) for name, cb in self.breakers.items()}

    def reset_all(self) -> None:
        """Reset and drop every registered breaker."""
        with self._lock:
            for cb in self.breakers.values():
                cb.reset()
            self.breakers.clear()


# The module-global registry, guarded by its own lock: the lock protects the
# lazy creation of the singleton itself, so it cannot live on the instance.
_registry: CircuitBreakerRegistry | None = None
_registry_lock = Lock()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Return the process-wide registry, creating it on first use.

    Returns
    -------
    CircuitBreakerRegistry
        The shared registry.
    """
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = CircuitBreakerRegistry()
        return _registry


def reset_circuit_breakers() -> None:
    """Reset and drop the process-wide registry (for testing)."""
    global _registry
    with _registry_lock:
        if _registry is not None:
            _registry.reset_all()
        _registry = None


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name from the global registry.

    Parameters
    ----------
    name : str
        Unique breaker name.
    config : CircuitBreakerConfig, optional
        Configuration, used only when creating a new breaker.

    Returns
    -------
    CircuitBreaker
        The existing or newly created breaker.
    """
    registry = get_circuit_breaker_registry()
    return registry.get_or_create(name, config)

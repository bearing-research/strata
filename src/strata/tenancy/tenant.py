"""Tenant config, per-tenant runtime state, id validation and request-scoped tenant context.

``DEFAULT_TENANT_ID`` is the fallback for single-tenant deployments.
"""

from __future__ import annotations

import contextvars
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.adaptive_concurrency import ResizableLimiter
    from strata.rate_limiter import TokenBucket

# Request-scoped; set by middleware.
_tenant_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_id", default=None
)

# Tenant used in single-tenant mode.
DEFAULT_TENANT_ID = "_default"

MAX_TENANT_ID_LENGTH = 64
TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def validate_tenant_id(tenant_id: str) -> tuple[bool, str | None]:
    """Validate a tenant id: 1-64 chars, alphanumeric first, then alphanumerics, ``_`` and ``-``.

    Returns ``(is_valid, error_message)``; the message is ``None`` when valid.
    """
    if not tenant_id:
        return False, "Tenant ID cannot be empty"

    if len(tenant_id) > MAX_TENANT_ID_LENGTH:
        return False, f"Tenant ID exceeds maximum length of {MAX_TENANT_ID_LENGTH} characters"

    if not TENANT_ID_PATTERN.match(tenant_id):
        return False, (
            "Tenant ID must start with alphanumeric and contain only "
            "alphanumeric characters, underscores, and hyphens"
        )

    return True, None


@dataclass(frozen=True)
class TenantConfig:
    """Per-tenant configuration and limits.

    Every optional limit defaults to ``None``, meaning "use the global default".
    ``enabled=False`` disables the tenant without deleting it.
    """

    tenant_id: str

    interactive_slots: int | None = None
    bulk_slots: int | None = None
    per_client_interactive: int | None = None
    per_client_bulk: int | None = None

    requests_per_second: float | None = None
    burst: float | None = None

    max_cache_size_bytes: int | None = None
    max_response_bytes: int | None = None

    enabled: bool = True

    def effective_interactive_slots(self, default: int) -> int:
        """Return ``interactive_slots``, or ``default`` when unset."""
        return self.interactive_slots if self.interactive_slots is not None else default

    def effective_bulk_slots(self, default: int) -> int:
        """Return ``bulk_slots``, or ``default`` when unset."""
        return self.bulk_slots if self.bulk_slots is not None else default

    def effective_per_client_interactive(self, default: int) -> int:
        """Return ``per_client_interactive``, or ``default`` when unset."""
        return self.per_client_interactive if self.per_client_interactive is not None else default

    def effective_per_client_bulk(self, default: int) -> int:
        """Return ``per_client_bulk``, or ``default`` when unset."""
        return self.per_client_bulk if self.per_client_bulk is not None else default


@dataclass
class TenantQuotas:
    """Per-tenant runtime state and aggregate metrics.

    Created on a tenant's first request; limiters and rate bucket are created
    lazily by the server. ``last_access`` drives LRU eviction.
    """

    tenant_id: str

    interactive_limiter: ResizableLimiter | None = None
    bulk_limiter: ResizableLimiter | None = None
    rate_bucket: TokenBucket | None = None

    total_scans: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bytes_from_cache: int = 0
    bytes_from_storage: int = 0
    rows_returned: int = 0

    last_access: float = field(default_factory=time.time)

    def is_idle(self) -> bool:
        """No request holds or awaits a slot on this tenant's limiters."""
        return all(
            limiter is None or limiter.idle
            for limiter in (self.interactive_limiter, self.bulk_limiter)
        )

    def touch(self) -> None:
        """Update ``last_access`` to now for LRU tracking."""
        self.last_access = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Return the API-facing metrics: counters plus a derived ``cache_hit_rate``.

        Limiters, rate bucket and ``last_access`` are omitted; the rate is unrounded.
        """
        total_requests = self.cache_hits + self.cache_misses
        return {
            "tenant_id": self.tenant_id,
            "total_scans": self.total_scans,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (self.cache_hits / total_requests if total_requests > 0 else 0.0),
            "bytes_from_cache": self.bytes_from_cache,
            "bytes_from_storage": self.bytes_from_storage,
            "rows_returned": self.rows_returned,
        }


def get_tenant_id() -> str:
    """Return the current request's tenant id, or ``DEFAULT_TENANT_ID`` when none is set."""
    return _tenant_context.get() or DEFAULT_TENANT_ID


def set_tenant_id(tenant_id: str) -> contextvars.Token:
    """Bind the tenant id for the current request context.

    Returns the ``contextvars.Token`` for :func:`reset_tenant_id`.
    """
    return _tenant_context.set(tenant_id)


def reset_tenant_id(token: contextvars.Token) -> None:
    """Restore the tenant context to its value before :func:`set_tenant_id`."""
    _tenant_context.reset(token)


def clear_tenant_context() -> None:
    """Clear the tenant context (set it to ``None``)."""
    _tenant_context.set(None)

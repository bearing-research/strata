"""Tenant types and request-scoped context for multi-tenancy.

Provides :class:`TenantConfig` (per-tenant QoS / rate / size limits and feature
flags), :class:`TenantQuotas` (per-tenant runtime state and metrics), tenant-id
validation, and the request-scoped tenant context accessors. ``DEFAULT_TENANT_ID``
is the fallback for single-tenant deployments.
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

# Context variable for tenant-scoped data (request-scoped via middleware)
_tenant_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_id", default=None
)

# Default tenant for backward compatibility (single-tenant mode)
DEFAULT_TENANT_ID = "_default"

# Tenant ID validation constraints
MAX_TENANT_ID_LENGTH = 64
TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def validate_tenant_id(tenant_id: str) -> tuple[bool, str | None]:
    """Validate a tenant id's format.

    A valid id is 1–64 characters, starts with an alphanumeric, and otherwise
    contains only alphanumerics, underscores, and hyphens (e.g. ``"acme-corp"``,
    ``"tenant_123"``, ``"MyTenant"``). Empty, over-length, leading ``_``/``-``,
    or special characters are rejected.

    Parameters
    ----------
    tenant_id : str
        The candidate tenant id.

    Returns
    -------
    tuple of (bool, str or None)
        ``(is_valid, error_message)``; the message is ``None`` when valid.
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

    Loaded from the tenant registry on startup or from external config. Every
    optional field defaults to ``None``, meaning "use the global default".

    Attributes
    ----------
    tenant_id : str
        Tenant this config applies to.
    interactive_slots, bulk_slots : int or None
        QoS slot quotas; ``None`` uses the global default.
    per_client_interactive, per_client_bulk : int or None
        Per-client QoS limits; ``None`` uses the global default.
    requests_per_second, burst : float or None
        Rate-limit quota; ``None`` uses the global default.
    max_cache_size_bytes, max_response_bytes : int or None
        Size limits; ``None`` uses the global default.
    enabled : bool
        When ``False``, the tenant is disabled without being deleted.
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

    Holds the lazily-created QoS limiters and rate bucket plus running metrics.
    Created on a tenant's first request and LRU-evictable via ``last_access``.

    Attributes
    ----------
    tenant_id : str
        Tenant this state belongs to.
    interactive_limiter, bulk_limiter : ResizableLimiter or None
        QoS limiters, created lazily by the server.
    rate_bucket : TokenBucket or None
        Rate-limiter bucket, created lazily.
    total_scans, cache_hits, cache_misses : int
        Running request counters.
    bytes_from_cache, bytes_from_storage, rows_returned : int
        Running data-volume counters.
    last_access : float
        Unix timestamp of the most recent access, used for LRU eviction.
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

    def touch(self) -> None:
        """Update ``last_access`` to now for LRU tracking."""
        self.last_access = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Return the API-facing metrics projection.

        A curated subset — the runtime limiters / rate bucket and ``last_access``
        are omitted, and a derived ``cache_hit_rate`` is added. The rate is
        emitted at full precision; rounding for display is the consumer's
        concern.

        Returns
        -------
        dict
            Tenant id, request/volume counters, and ``cache_hit_rate``.
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
    """Return the current request's tenant id, or ``DEFAULT_TENANT_ID``.

    Reads the value set by the tenant middleware; falls back to
    ``DEFAULT_TENANT_ID`` when no tenant context is set (single-tenant mode).

    Returns
    -------
    str
        The active tenant id.
    """
    return _tenant_context.get() or DEFAULT_TENANT_ID


def set_tenant_id(tenant_id: str) -> contextvars.Token:
    """Set the tenant id in the request context.

    Called by the tenant middleware at the start of each request.

    Parameters
    ----------
    tenant_id : str
        Tenant id to bind to the current context.

    Returns
    -------
    contextvars.Token
        Token for restoring the previous context via :func:`reset_tenant_id`.
    """
    return _tenant_context.set(tenant_id)


def reset_tenant_id(token: contextvars.Token) -> None:
    """Restore the tenant context to its prior value.

    Parameters
    ----------
    token : contextvars.Token
        The token returned by :func:`set_tenant_id`.
    """
    _tenant_context.reset(token)


def clear_tenant_context() -> None:
    """Clear the tenant context (set it to ``None``)."""
    _tenant_context.set(None)

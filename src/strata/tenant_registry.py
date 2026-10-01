"""Tenant registry: thread-safe, LRU-bounded tenant configs and runtime quotas."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.tenant import DEFAULT_TENANT_ID, TenantConfig, TenantQuotas

if TYPE_CHECKING:
    from strata.adaptive_concurrency import ResizableLimiter

MAX_TRACKED_TENANTS = 1000


@dataclass
class TenantRegistry:
    """Registry for tenant configurations and runtime quotas.

    Thread-safe; runtime state is LRU-evicted beyond ``MAX_TRACKED_TENANTS``.
    """

    # Static configs, from the config file.
    _configs: dict[str, TenantConfig] = field(default_factory=dict)

    # Runtime quota state; LRU-evictable.
    _quotas: dict[str, TenantQuotas] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Global defaults, from StrataConfig.
    default_interactive_slots: int = 32
    default_bulk_slots: int = 8
    default_per_client_interactive: int = 2
    default_per_client_bulk: int = 1

    def __post_init__(self) -> None:
        """Register the default tenant."""
        self._configs[DEFAULT_TENANT_ID] = TenantConfig(tenant_id=DEFAULT_TENANT_ID)

    def register_tenant(self, config: TenantConfig) -> None:
        """Register a tenant configuration, at startup or from the admin API."""
        with self._lock:
            self._configs[config.tenant_id] = config

    def unregister_tenant(self, tenant_id: str) -> bool:
        """Unregister a tenant configuration; return whether it existed.

        The default tenant is never removed.
        """
        if tenant_id == DEFAULT_TENANT_ID:
            return False

        with self._lock:
            if tenant_id in self._configs:
                del self._configs[tenant_id]
                self._quotas.pop(tenant_id, None)
                return True
            return False

    def get_config(self, tenant_id: str) -> TenantConfig | None:
        """Get a tenant's configuration, or None if it is not registered."""
        return self._configs.get(tenant_id)

    def get_or_create_quotas(self, tenant_id: str) -> TenantQuotas:
        """Get or lazily create runtime quotas for a tenant.

        May LRU-evict another tenant's quotas beyond ``MAX_TRACKED_TENANTS``.
        """
        with self._lock:
            if tenant_id in self._quotas:
                # Re-insert to move to the LRU tail.
                quotas = self._quotas.pop(tenant_id)
                quotas.touch()
                self._quotas[tenant_id] = quotas
                return quotas

            quotas = TenantQuotas(tenant_id=tenant_id)
            self._quotas[tenant_id] = quotas

            # Evict oldest first, idle tenants only. Evicting a tenant whose limiters hold or await
            # a slot would give its next request a fresh pair (up to twice its quota) and drop the
            # old pair's streams from aggregate_limiter_usage, which the shutdown drain waits on.
            # The registry runs over the cap while that many tenants are busy.
            for candidate in list(self._quotas):
                if len(self._quotas) <= MAX_TRACKED_TENANTS:
                    break
                if candidate != tenant_id and self._quotas[candidate].is_idle():
                    del self._quotas[candidate]

            return quotas

    def get_or_create_limiters(self, tenant_id: str) -> tuple[ResizableLimiter, ResizableLimiter]:
        """Get or lazily create a tenant's ``(interactive, bulk)`` QoS limiters.

        Each tenant has its own limiters, stored on its ``TenantQuotas``, so they are
        evicted with it.
        """
        from strata.adaptive_concurrency import ResizableLimiter

        quotas = self.get_or_create_quotas(tenant_id)

        with self._lock:
            if quotas.interactive_limiter is None:
                config = self.get_config(tenant_id)
                if config:
                    interactive_slots = config.effective_interactive_slots(
                        self.default_interactive_slots
                    )
                    bulk_slots = config.effective_bulk_slots(self.default_bulk_slots)
                else:
                    interactive_slots = self.default_interactive_slots
                    bulk_slots = self.default_bulk_slots

                quotas.interactive_limiter = ResizableLimiter(interactive_slots)
                quotas.bulk_limiter = ResizableLimiter(bulk_slots)

            interactive = quotas.interactive_limiter
            bulk = quotas.bulk_limiter
            if not isinstance(interactive, ResizableLimiter):
                msg = f"interactive_limiter not a ResizableLimiter for tenant {tenant_id}"
                raise RuntimeError(msg)
            if not isinstance(bulk, ResizableLimiter):
                msg = f"bulk_limiter not a ResizableLimiter for tenant {tenant_id}"
                raise RuntimeError(msg)
            return interactive, bulk

    def aggregate_limiter_usage(self) -> tuple[int, int, int, int]:
        """Sum live admission-limiter usage across all tracked tenants.

        Stream admission acquires these per-tenant limiters, not the global
        ``ServerState`` ones, so this is the true active-scan count.

        Returns ``(interactive_in_use, interactive_available, bulk_in_use,
        bulk_available)``.
        """
        from strata.adaptive_concurrency import ResizableLimiter

        i_in_use = i_avail = b_in_use = b_avail = 0
        with self._lock:
            for quotas in self._quotas.values():
                interactive = quotas.interactive_limiter
                bulk = quotas.bulk_limiter
                if isinstance(interactive, ResizableLimiter):
                    i_in_use += interactive.in_use
                    i_avail += interactive.available
                if isinstance(bulk, ResizableLimiter):
                    b_in_use += bulk.in_use
                    b_avail += bulk.available
        return i_in_use, i_avail, b_in_use, b_avail

    def is_tenant_enabled(self, tenant_id: str) -> bool:
        """Check whether a tenant may make requests: unknown tenants are allowed."""
        config = self._configs.get(tenant_id)
        if config is None:
            # Unknown tenants are allowed by default; the server config can require
            # pre-registration.
            return True
        return config.enabled

    def is_tenant_registered(self, tenant_id: str) -> bool:
        """Check if tenant is registered (has explicit config)."""
        return tenant_id in self._configs

    def list_tenants(self) -> list[str]:
        """List all registered tenant IDs."""
        return list(self._configs.keys())

    def get_all_tenant_configs(self) -> list[TenantConfig]:
        """Get all registered tenant configurations."""
        return list(self._configs.values())

    def get_all_tenant_metrics(self) -> list[dict]:
        """Get metrics for all tenants with runtime state."""
        with self._lock:
            return [q.to_dict() for q in self._quotas.values()]

    def get_tenant_metrics(self, tenant_id: str) -> dict | None:
        """Get metrics for a specific tenant."""
        with self._lock:
            quotas = self._quotas.get(tenant_id)
            return quotas.to_dict() if quotas else None

    def record_scan(
        self,
        tenant_id: str,
        cache_hits: int,
        cache_misses: int,
        bytes_from_cache: int,
        bytes_from_storage: int,
        rows_returned: int,
    ) -> None:
        """Record scan metrics for a tenant."""
        quotas = self.get_or_create_quotas(tenant_id)
        with self._lock:
            quotas.total_scans += 1
            quotas.cache_hits += cache_hits
            quotas.cache_misses += cache_misses
            quotas.bytes_from_cache += bytes_from_cache
            quotas.bytes_from_storage += bytes_from_storage
            quotas.rows_returned += rows_returned
            quotas.touch()

    def reset_tenant_metrics(self, tenant_id: str) -> bool:
        """Reset metrics for a specific tenant. Returns True if tenant existed."""
        with self._lock:
            if tenant_id in self._quotas:
                quotas = self._quotas[tenant_id]
                quotas.total_scans = 0
                quotas.cache_hits = 0
                quotas.cache_misses = 0
                quotas.bytes_from_cache = 0
                quotas.bytes_from_storage = 0
                quotas.rows_returned = 0
                return True
            return False

    def reset_all_metrics(self) -> None:
        """Reset metrics for all tenants."""
        with self._lock:
            for quotas in self._quotas.values():
                quotas.total_scans = 0
                quotas.cache_hits = 0
                quotas.cache_misses = 0
                quotas.bytes_from_cache = 0
                quotas.bytes_from_storage = 0
                quotas.rows_returned = 0


_registry: TenantRegistry | None = None
_registry_lock = threading.Lock()


def get_tenant_registry() -> TenantRegistry:
    """Get the global tenant registry, creating it with defaults on first use (thread-safe)."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TenantRegistry()
    return _registry


def init_tenant_registry(
    default_interactive_slots: int = 32,
    default_bulk_slots: int = 8,
    default_per_client_interactive: int = 2,
    default_per_client_bulk: int = 1,
    tenant_configs: list[TenantConfig] | None = None,
) -> TenantRegistry:
    """Initialize the global tenant registry with custom defaults; call once at startup."""
    global _registry
    with _registry_lock:
        _registry = TenantRegistry(
            default_interactive_slots=default_interactive_slots,
            default_bulk_slots=default_bulk_slots,
            default_per_client_interactive=default_per_client_interactive,
            default_per_client_bulk=default_per_client_bulk,
        )
        if tenant_configs:
            for config in tenant_configs:
                _registry.register_tenant(config)
        return _registry


def reset_tenant_registry() -> None:
    """Reset the global tenant registry (for tests)."""
    global _registry
    with _registry_lock:
        _registry = None

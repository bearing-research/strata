"""Tenant registry for managing tenant configurations and runtime state.

Provides:
- TenantRegistry: Thread-safe registry with LRU eviction for tenant quotas
- Global registry access via get_tenant_registry() / init_tenant_registry()
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.tenant import DEFAULT_TENANT_ID, TenantConfig, TenantQuotas

if TYPE_CHECKING:
    from strata.adaptive_concurrency import ResizableLimiter

# Max tenants to track runtime state for (LRU eviction beyond this)
MAX_TRACKED_TENANTS = 1000


@dataclass
class TenantRegistry:
    """Registry for tenant configurations and runtime quotas.

    Thread-safe with LRU eviction for runtime state.

    Usage:
        registry = get_tenant_registry()
        config = registry.get_config("tenant-a")
        quotas = registry.get_or_create_quotas("tenant-a")
    """

    # Static tenant configurations (loaded from config file)
    _configs: dict[str, TenantConfig] = field(default_factory=dict)

    # Runtime quota state (LRU evictable)
    _quotas: dict[str, TenantQuotas] = field(default_factory=dict)

    # Lock for thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Global defaults (from StrataConfig)
    default_interactive_slots: int = 32
    default_bulk_slots: int = 8
    default_per_client_interactive: int = 2
    default_per_client_bulk: int = 1

    def __post_init__(self) -> None:
        """Initialize with default tenant for backward compatibility."""
        self._configs[DEFAULT_TENANT_ID] = TenantConfig(tenant_id=DEFAULT_TENANT_ID)

    def register_tenant(self, config: TenantConfig) -> None:
        """Register a tenant configuration.

        Can be called at startup from config file or dynamically via admin API.
        """
        with self._lock:
            self._configs[config.tenant_id] = config

    def unregister_tenant(self, tenant_id: str) -> bool:
        """Unregister a tenant configuration.

        Returns True if tenant was found and removed, False otherwise.
        Does not remove the default tenant.
        """
        if tenant_id == DEFAULT_TENANT_ID:
            return False

        with self._lock:
            if tenant_id in self._configs:
                del self._configs[tenant_id]
                # Also clean up runtime state
                self._quotas.pop(tenant_id, None)
                return True
            return False

    def get_config(self, tenant_id: str) -> TenantConfig | None:
        """Get tenant configuration.

        Returns None if tenant is not registered.
        """
        return self._configs.get(tenant_id)

    def get_or_create_quotas(self, tenant_id: str) -> TenantQuotas:
        """Get or create runtime quotas for a tenant.

        Creates quotas lazily on first access. Uses LRU eviction
        when more than MAX_TRACKED_TENANTS are active.
        """
        with self._lock:
            if tenant_id in self._quotas:
                # Move to end for LRU ordering (Python 3.7+ dict maintains insertion order)
                quotas = self._quotas.pop(tenant_id)
                quotas.touch()
                self._quotas[tenant_id] = quotas
                return quotas

            # Create new quotas
            quotas = TenantQuotas(tenant_id=tenant_id)
            self._quotas[tenant_id] = quotas

            # LRU eviction if over limit
            while len(self._quotas) > MAX_TRACKED_TENANTS:
                # Remove oldest (first) entry
                oldest = next(iter(self._quotas))
                del self._quotas[oldest]

            return quotas

    def get_or_create_limiters(self, tenant_id: str) -> tuple[ResizableLimiter, ResizableLimiter]:
        """Get or create per-tenant QoS limiters.

        Each tenant gets their own ResizableLimiter instances for interactive
        and bulk tiers, providing complete QoS isolation between tenants.

        Limiters are created lazily on first access and stored in TenantQuotas.
        When tenant quotas are LRU-evicted, their limiters go with them.

        Returns:
            Tuple of (interactive_limiter, bulk_limiter)
        """
        from strata.adaptive_concurrency import ResizableLimiter

        quotas = self.get_or_create_quotas(tenant_id)

        with self._lock:
            if quotas.interactive_limiter is None:
                # Get tenant-specific config or use defaults
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

            # Cast and return (both should now be ResizableLimiter instances)
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
        """Aggregate live admission-limiter usage across all tracked tenants.

        Stream admission acquires the per-tenant limiters returned by
        ``get_or_create_limiters`` — not the global ServerState limiters, which
        are never acquired — so these are the real source of truth for active
        scans and saturation. Sums in-use / available across every tenant that
        has live limiters. Single-tenant deployments have exactly one (the
        ``_default`` tenant), so this reduces to that tenant's limiter.

        Returns:
            ``(interactive_in_use, interactive_available, bulk_in_use,
            bulk_available)`` summed over live limiters.
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
        """Check if tenant is enabled (exists and not disabled).

        Unknown tenants are allowed by default (dynamic registration).
        Set require_tenant_registration in config to require pre-registration.
        """
        config = self._configs.get(tenant_id)
        if config is None:
            # Unknown tenants allowed by default for flexibility
            # The server config can require pre-registration if needed
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
        """Get metrics for all tracked tenants (those with runtime state)."""
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


# Global registry instance
_registry: TenantRegistry | None = None
_registry_lock = threading.Lock()


def get_tenant_registry() -> TenantRegistry:
    """Get the global tenant registry.

    Initializes with defaults if not already initialized.
    Thread-safe via double-checked locking.
    """
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
    """Initialize the global tenant registry with custom defaults.

    Should be called once at server startup before handling requests.
    """
    global _registry
    with _registry_lock:
        _registry = TenantRegistry(
            default_interactive_slots=default_interactive_slots,
            default_bulk_slots=default_bulk_slots,
            default_per_client_interactive=default_per_client_interactive,
            default_per_client_bulk=default_per_client_bulk,
        )
        # Register any provided tenant configs
        if tenant_configs:
            for config in tenant_configs:
                _registry.register_tenant(config)
        return _registry


def reset_tenant_registry() -> None:
    """Reset the global tenant registry. Primarily for testing."""
    global _registry
    with _registry_lock:
        _registry = None

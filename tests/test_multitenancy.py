"""Tests for multi-tenancy functionality."""

import asyncio

import pytest

from strata.tenant import (
    DEFAULT_TENANT_ID,
    MAX_TENANT_ID_LENGTH,
    TenantConfig,
    TenantQuotas,
    clear_tenant_context,
    get_tenant_id,
    reset_tenant_id,
    set_tenant_id,
    validate_tenant_id,
)
from strata.tenant_registry import (
    MAX_TRACKED_TENANTS,
    TenantRegistry,
    get_tenant_registry,
    init_tenant_registry,
    reset_tenant_registry,
)
from strata.types import CacheKey, TableIdentity


class TestTenantContext:
    def setup_method(self):
        clear_tenant_context()

    def teardown_method(self):
        clear_tenant_context()

    def test_default_tenant_when_not_set(self):
        """The tenant is _default when no context is set."""
        assert get_tenant_id() == DEFAULT_TENANT_ID

    def test_tenant_context_set_and_get(self):
        token = set_tenant_id("tenant-a")
        assert get_tenant_id() == "tenant-a"
        reset_tenant_id(token)
        assert get_tenant_id() == DEFAULT_TENANT_ID

    def test_tenant_context_clear(self):
        set_tenant_id("tenant-a")
        clear_tenant_context()
        assert get_tenant_id() == DEFAULT_TENANT_ID


class TestTenantConfig:
    def test_default_values(self):
        config = TenantConfig(tenant_id="test-tenant")
        assert config.tenant_id == "test-tenant"
        assert config.interactive_slots is None
        assert config.bulk_slots is None
        assert config.enabled is True

    def test_effective_slots_with_defaults(self):
        config = TenantConfig(tenant_id="test-tenant")
        assert config.effective_interactive_slots(32) == 32
        assert config.effective_bulk_slots(8) == 8

    def test_effective_slots_with_overrides(self):
        config = TenantConfig(
            tenant_id="test-tenant",
            interactive_slots=16,
            bulk_slots=4,
        )
        assert config.effective_interactive_slots(32) == 16
        assert config.effective_bulk_slots(8) == 4


class TestTenantQuotas:
    def test_default_values(self):
        quotas = TenantQuotas(tenant_id="test-tenant")
        assert quotas.total_scans == 0
        assert quotas.cache_hits == 0
        assert quotas.cache_misses == 0

    def test_to_dict(self):
        quotas = TenantQuotas(
            tenant_id="test-tenant",
            total_scans=10,
            cache_hits=8,
            cache_misses=2,
            bytes_from_cache=1000,
            bytes_from_storage=200,
            rows_returned=500,
        )
        result = quotas.to_dict()
        assert result["tenant_id"] == "test-tenant"
        assert result["total_scans"] == 10
        assert result["cache_hit_rate"] == 0.8
        assert result["bytes_from_cache"] == 1000

    def test_touch_updates_last_access(self):
        quotas = TenantQuotas(tenant_id="test-tenant")
        old_time = quotas.last_access
        import time

        time.sleep(0.01)
        quotas.touch()
        assert quotas.last_access > old_time


class TestTenantRegistry:
    def setup_method(self):
        reset_tenant_registry()

    def teardown_method(self):
        reset_tenant_registry()

    def test_default_tenant_exists(self):
        registry = TenantRegistry()
        assert registry.get_config(DEFAULT_TENANT_ID) is not None

    def test_register_and_get_tenant(self):
        registry = TenantRegistry()
        config = TenantConfig(
            tenant_id="tenant-a",
            interactive_slots=10,
            bulk_slots=5,
        )
        registry.register_tenant(config)

        retrieved = registry.get_config("tenant-a")
        assert retrieved is not None
        assert retrieved.interactive_slots == 10
        assert retrieved.bulk_slots == 5

    def test_unregister_tenant(self):
        registry = TenantRegistry()
        config = TenantConfig(tenant_id="tenant-a")
        registry.register_tenant(config)
        assert registry.get_config("tenant-a") is not None

        registry.unregister_tenant("tenant-a")
        assert registry.get_config("tenant-a") is None

    def test_cannot_unregister_default_tenant(self):
        registry = TenantRegistry()
        result = registry.unregister_tenant(DEFAULT_TENANT_ID)
        assert result is False
        assert registry.get_config(DEFAULT_TENANT_ID) is not None

    def test_get_or_create_quotas(self):
        registry = TenantRegistry()
        quotas = registry.get_or_create_quotas("new-tenant")
        assert quotas.tenant_id == "new-tenant"
        assert quotas.total_scans == 0

        # Same object again (after the LRU move).
        quotas2 = registry.get_or_create_quotas("new-tenant")
        assert quotas2.tenant_id == quotas.tenant_id

    def test_is_tenant_enabled(self):
        registry = TenantRegistry()

        assert registry.is_tenant_enabled("unknown-tenant") is True

        disabled_config = TenantConfig(tenant_id="disabled-tenant", enabled=False)
        registry.register_tenant(disabled_config)
        assert registry.is_tenant_enabled("disabled-tenant") is False

    def test_record_scan(self):
        registry = TenantRegistry()
        registry.record_scan(
            tenant_id="tenant-a",
            cache_hits=5,
            cache_misses=3,
            bytes_from_cache=500,
            bytes_from_storage=300,
            rows_returned=100,
        )

        quotas = registry.get_or_create_quotas("tenant-a")
        assert quotas.total_scans == 1
        assert quotas.cache_hits == 5
        assert quotas.cache_misses == 3

    def test_lru_eviction(self):
        """The registry evicts the oldest tenants when over its limit."""
        registry = TenantRegistry()

        # More tenants than the max, to trigger eviction.
        for i in range(MAX_TRACKED_TENANTS + 100):
            registry.get_or_create_quotas(f"tenant-{i}")

        assert len(registry._quotas) <= MAX_TRACKED_TENANTS

    def test_global_registry(self):
        registry = init_tenant_registry(
            default_interactive_slots=16,
            default_bulk_slots=4,
        )
        assert registry.default_interactive_slots == 16
        assert registry.default_bulk_slots == 4

        assert get_tenant_registry() is registry


class TestEvictionSparesBusyTenants:
    """Evicting a tenant that holds or awaits a slot would double its quota.

    Its old limiters keep admitting while the next request gets a fresh pair, and the old streams
    are invisible to the shutdown drain.
    """

    @staticmethod
    def _crowd_out(registry):
        for i in range(MAX_TRACKED_TENANTS):
            registry.get_or_create_limiters(f"other-{i}")

    @staticmethod
    def _registry():
        return TenantRegistry(default_interactive_slots=1, default_bulk_slots=1)

    @pytest.mark.asyncio
    async def test_a_tenant_holding_a_slot_keeps_its_quota(self):
        registry = self._registry()
        first, _ = registry.get_or_create_limiters("a")
        assert await first.acquire(timeout=0.1)

        self._crowd_out(registry)

        again, _ = registry.get_or_create_limiters("a")
        assert again is first
        assert not await again.acquire(timeout=0.01)  # still one slot

    @pytest.mark.asyncio
    async def test_the_shutdown_drain_counts_a_busy_tenants_stream(self):
        registry = self._registry()
        limiter, _ = registry.get_or_create_limiters("a")
        assert await limiter.acquire(timeout=0.1)

        self._crowd_out(registry)

        interactive_in_use, _, bulk_in_use, _ = registry.aggregate_limiter_usage()
        assert interactive_in_use + bulk_in_use == 1

    @pytest.mark.asyncio
    async def test_a_tenant_waiting_for_a_slot_keeps_its_quota(self):
        registry = self._registry()
        limiter, _ = registry.get_or_create_limiters("a")
        assert await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)  # queued on the limiter
        await limiter.release()  # free, but the waiter has not run yet

        self._crowd_out(registry)

        assert registry.get_or_create_limiters("a")[0] is limiter
        assert await waiter

    @pytest.mark.asyncio
    async def test_idle_tenants_are_still_evicted(self):
        registry = self._registry()
        busy, _ = registry.get_or_create_limiters("busy")
        assert await busy.acquire()
        registry.get_or_create_limiters("idle")

        self._crowd_out(registry)
        assert "idle" not in registry._quotas
        assert "busy" in registry._quotas  # a newer idle tenant went instead
        assert len(registry._quotas) == MAX_TRACKED_TENANTS

        await busy.release()
        registry.get_or_create_limiters("newcomer")
        assert "busy" not in registry._quotas
        assert len(registry._quotas) == MAX_TRACKED_TENANTS


class TestCacheKeyTenantIsolation:
    def test_different_tenants_different_cache_keys(self):
        table_identity = TableIdentity("catalog", "ns", "table")

        key_a = CacheKey(
            tenant_id="tenant-a",
            table_identity=table_identity,
            snapshot_id=1,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc",
        )
        key_b = CacheKey(
            tenant_id="tenant-b",
            table_identity=table_identity,
            snapshot_id=1,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc",
        )

        assert key_a.to_hex() != key_b.to_hex()

    def test_same_tenant_same_cache_key(self):
        table_identity = TableIdentity("catalog", "ns", "table")

        key_a = CacheKey(
            tenant_id="tenant-a",
            table_identity=table_identity,
            snapshot_id=1,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc",
        )
        key_b = CacheKey(
            tenant_id="tenant-a",
            table_identity=table_identity,
            snapshot_id=1,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc",
        )

        assert key_a.to_hex() == key_b.to_hex()

    def test_tenant_id_in_cache_key(self):
        table_identity = TableIdentity("catalog", "ns", "table")

        key = CacheKey(
            tenant_id="my-tenant",
            table_identity=table_identity,
            snapshot_id=1,
            file_path="/data/file.parquet",
            row_group_id=0,
            projection_fingerprint="abc",
        )

        assert key.tenant_id == "my-tenant"


class TestTenantIdValidation:
    @pytest.mark.parametrize(
        "tenant_id",
        ["acme", "acme-corp", "tenant_123", "MyTenantName", "123tenant"],
    )
    def test_valid_tenant_id_character_classes(self, tenant_id):
        """Alphanumerics, hyphens, underscores, mixed case and leading digits are valid."""
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is True
        assert error is None

    def test_valid_single_character(self):
        is_valid, error = validate_tenant_id("a")
        assert is_valid is True
        assert error is None

    def test_valid_max_length(self):
        tenant_id = "a" * MAX_TENANT_ID_LENGTH
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is True
        assert error is None

    def test_invalid_empty(self):
        is_valid, error = validate_tenant_id("")
        assert is_valid is False
        assert error is not None
        assert "cannot be empty" in error

    def test_invalid_too_long(self):
        tenant_id = "a" * (MAX_TENANT_ID_LENGTH + 1)
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is False
        assert error is not None
        assert "exceeds maximum length" in error

    def test_invalid_starts_with_underscore(self):
        is_valid, error = validate_tenant_id("_private")
        assert is_valid is False
        assert error is not None
        assert "start with alphanumeric" in error

    def test_invalid_starts_with_hyphen(self):
        is_valid, error = validate_tenant_id("-bad")
        assert is_valid is False
        assert error is not None
        assert "start with alphanumeric" in error

    def test_invalid_contains_space(self):
        is_valid, error = validate_tenant_id("has spaces")
        assert is_valid is False
        assert error is not None
        assert "alphanumeric" in error

    def test_invalid_contains_special_chars(self):
        invalid_ids = [
            "has@at",
            "has.dot",
            "has/slash",
            "has:colon",
            "has;semicolon",
            "has'quote",
            'has"doublequote',
            "has<angle>",
            "has[bracket]",
            "has{brace}",
            "has|pipe",
            "has\\backslash",
        ]
        for tenant_id in invalid_ids:
            is_valid, error = validate_tenant_id(tenant_id)
            assert is_valid is False, f"Expected {tenant_id!r} to be invalid"

    def test_invalid_unicode(self):
        is_valid, error = validate_tenant_id("café")
        assert is_valid is False

    def test_invalid_newline(self):
        """A newline would allow header injection."""
        is_valid, error = validate_tenant_id("tenant\nX-Evil: header")
        assert is_valid is False

    def test_invalid_null_byte(self):
        is_valid, error = validate_tenant_id("tenant\x00evil")
        assert is_valid is False


class TestPerTenantQoS:
    def setup_method(self):
        reset_tenant_registry()

    def teardown_method(self):
        reset_tenant_registry()

    def test_tenant_gets_own_limiters(self):
        registry = TenantRegistry()

        lim_a_int, lim_a_bulk = registry.get_or_create_limiters("tenant-a")
        lim_b_int, lim_b_bulk = registry.get_or_create_limiters("tenant-b")

        assert lim_a_int is not lim_b_int
        assert lim_a_bulk is not lim_b_bulk

    def test_same_tenant_gets_same_limiters(self):
        """The same tenant gets the same limiter instances on later calls."""
        registry = TenantRegistry()

        lim1_int, lim1_bulk = registry.get_or_create_limiters("tenant-a")
        lim2_int, lim2_bulk = registry.get_or_create_limiters("tenant-a")

        assert lim1_int is lim2_int
        assert lim1_bulk is lim2_bulk

    def test_tenant_limiter_uses_config_slots(self):
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        config = TenantConfig(
            tenant_id="premium",
            interactive_slots=64,
            bulk_slots=16,
        )
        registry.register_tenant(config)

        lim_int, lim_bulk = registry.get_or_create_limiters("premium")
        assert lim_int.capacity == 64
        assert lim_bulk.capacity == 16

    def test_unknown_tenant_uses_defaults(self):
        """Unknown tenants get the default slot counts."""
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        lim_int, lim_bulk = registry.get_or_create_limiters("unknown")
        assert lim_int.capacity == 32
        assert lim_bulk.capacity == 8

    def test_default_tenant_uses_global_defaults(self):
        registry = TenantRegistry(
            default_interactive_slots=16,
            default_bulk_slots=4,
        )

        lim_int, lim_bulk = registry.get_or_create_limiters(DEFAULT_TENANT_ID)
        assert lim_int.capacity == 16
        assert lim_bulk.capacity == 4

    def test_limiter_persists_across_quotas_access(self):
        """Limiters persist across repeated quota access."""
        registry = TenantRegistry()

        lim1_int, lim1_bulk = registry.get_or_create_limiters("tenant-a")

        quotas = registry.get_or_create_quotas("tenant-a")

        lim2_int, lim2_bulk = registry.get_or_create_limiters("tenant-a")

        assert lim1_int is lim2_int
        assert lim1_bulk is lim2_bulk
        assert quotas.interactive_limiter is lim1_int
        assert quotas.bulk_limiter is lim1_bulk

    def test_tenant_config_partial_override(self):
        """A config overriding one slot type uses the default for the other."""
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        config = TenantConfig(
            tenant_id="partial",
            interactive_slots=100,
            # bulk_slots=None means use the default.
        )
        registry.register_tenant(config)

        lim_int, lim_bulk = registry.get_or_create_limiters("partial")
        assert lim_int.capacity == 100
        assert lim_bulk.capacity == 8  # Default

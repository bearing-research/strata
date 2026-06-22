"""Tests for multi-tenancy functionality."""

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
    """Tests for tenant context management."""

    def setup_method(self):
        """Clear tenant context before each test."""
        clear_tenant_context()

    def teardown_method(self):
        """Clear tenant context after each test."""
        clear_tenant_context()

    def test_default_tenant_when_not_set(self):
        """Default tenant should be _default when no context set."""
        assert get_tenant_id() == DEFAULT_TENANT_ID

    def test_tenant_context_set_and_get(self):
        """Tenant context should be settable and gettable."""
        token = set_tenant_id("tenant-a")
        assert get_tenant_id() == "tenant-a"
        reset_tenant_id(token)
        assert get_tenant_id() == DEFAULT_TENANT_ID

    def test_tenant_context_clear(self):
        """Clear should reset to default tenant."""
        set_tenant_id("tenant-a")
        clear_tenant_context()
        assert get_tenant_id() == DEFAULT_TENANT_ID


class TestTenantConfig:
    """Tests for TenantConfig dataclass."""

    def test_default_values(self):
        """TenantConfig should have sensible defaults."""
        config = TenantConfig(tenant_id="test-tenant")
        assert config.tenant_id == "test-tenant"
        assert config.interactive_slots is None
        assert config.bulk_slots is None
        assert config.enabled is True

    def test_effective_slots_with_defaults(self):
        """Effective slots should fall back to defaults."""
        config = TenantConfig(tenant_id="test-tenant")
        assert config.effective_interactive_slots(32) == 32
        assert config.effective_bulk_slots(8) == 8

    def test_effective_slots_with_overrides(self):
        """Effective slots should use tenant-specific values when set."""
        config = TenantConfig(
            tenant_id="test-tenant",
            interactive_slots=16,
            bulk_slots=4,
        )
        assert config.effective_interactive_slots(32) == 16
        assert config.effective_bulk_slots(8) == 4


class TestTenantQuotas:
    """Tests for TenantQuotas runtime state."""

    def test_default_values(self):
        """TenantQuotas should initialize with zero metrics."""
        quotas = TenantQuotas(tenant_id="test-tenant")
        assert quotas.total_scans == 0
        assert quotas.cache_hits == 0
        assert quotas.cache_misses == 0

    def test_to_dict(self):
        """to_dict should return proper representation."""
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
        """touch() should update last_access time."""
        quotas = TenantQuotas(tenant_id="test-tenant")
        old_time = quotas.last_access
        import time

        time.sleep(0.01)
        quotas.touch()
        assert quotas.last_access > old_time


class TestTenantRegistry:
    """Tests for TenantRegistry."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_tenant_registry()

    def teardown_method(self):
        """Reset global registry after each test."""
        reset_tenant_registry()

    def test_default_tenant_exists(self):
        """Default tenant should be pre-registered."""
        registry = TenantRegistry()
        assert registry.get_config(DEFAULT_TENANT_ID) is not None

    def test_register_and_get_tenant(self):
        """Should be able to register and retrieve tenant config."""
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
        """Should be able to unregister a tenant."""
        registry = TenantRegistry()
        config = TenantConfig(tenant_id="tenant-a")
        registry.register_tenant(config)
        assert registry.get_config("tenant-a") is not None

        registry.unregister_tenant("tenant-a")
        assert registry.get_config("tenant-a") is None

    def test_cannot_unregister_default_tenant(self):
        """Default tenant should not be unregisterable."""
        registry = TenantRegistry()
        result = registry.unregister_tenant(DEFAULT_TENANT_ID)
        assert result is False
        assert registry.get_config(DEFAULT_TENANT_ID) is not None

    def test_get_or_create_quotas(self):
        """get_or_create_quotas should create quotas on first access."""
        registry = TenantRegistry()
        quotas = registry.get_or_create_quotas("new-tenant")
        assert quotas.tenant_id == "new-tenant"
        assert quotas.total_scans == 0

        # Second call should return same object (after LRU move)
        quotas2 = registry.get_or_create_quotas("new-tenant")
        assert quotas2.tenant_id == quotas.tenant_id

    def test_is_tenant_enabled(self):
        """is_tenant_enabled should check enabled flag."""
        registry = TenantRegistry()

        # Unknown tenant is enabled by default
        assert registry.is_tenant_enabled("unknown-tenant") is True

        # Disabled tenant should return False
        disabled_config = TenantConfig(tenant_id="disabled-tenant", enabled=False)
        registry.register_tenant(disabled_config)
        assert registry.is_tenant_enabled("disabled-tenant") is False

    def test_record_scan(self):
        """record_scan should update tenant metrics."""
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
        """Registry should evict oldest tenants when over limit."""
        registry = TenantRegistry()

        # Create more tenants than max to trigger eviction
        for i in range(MAX_TRACKED_TENANTS + 100):
            registry.get_or_create_quotas(f"tenant-{i}")

        # Should be at or below max
        assert len(registry._quotas) <= MAX_TRACKED_TENANTS

    def test_global_registry(self):
        """init_tenant_registry should create global registry."""
        registry = init_tenant_registry(
            default_interactive_slots=16,
            default_bulk_slots=4,
        )
        assert registry.default_interactive_slots == 16
        assert registry.default_bulk_slots == 4

        # get_tenant_registry should return same instance
        assert get_tenant_registry() is registry


class TestCacheKeyTenantIsolation:
    """Tests for cache key isolation between tenants."""

    def test_different_tenants_different_cache_keys(self):
        """Different tenants should produce different cache key hashes."""
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

        # Same data but different tenants = different cache keys
        assert key_a.to_hex() != key_b.to_hex()

    def test_same_tenant_same_cache_key(self):
        """Same tenant with same data should produce same cache key."""
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
        """CacheKey should include tenant_id field."""
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
    """Tests for tenant ID validation."""

    @pytest.mark.parametrize(
        "tenant_id",
        ["acme", "acme-corp", "tenant_123", "MyTenantName", "123tenant"],
    )
    def test_valid_tenant_id_character_classes(self, tenant_id):
        """Alphanumerics, hyphens, underscores, mixed case, leading digits are valid."""
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is True
        assert error is None

    def test_valid_single_character(self):
        """Single character tenant ID should be valid."""
        is_valid, error = validate_tenant_id("a")
        assert is_valid is True
        assert error is None

    def test_valid_max_length(self):
        """Tenant ID at max length should be valid."""
        tenant_id = "a" * MAX_TENANT_ID_LENGTH
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is True
        assert error is None

    def test_invalid_empty(self):
        """Empty tenant ID should be invalid."""
        is_valid, error = validate_tenant_id("")
        assert is_valid is False
        assert error is not None
        assert "cannot be empty" in error

    def test_invalid_too_long(self):
        """Tenant ID exceeding max length should be invalid."""
        tenant_id = "a" * (MAX_TENANT_ID_LENGTH + 1)
        is_valid, error = validate_tenant_id(tenant_id)
        assert is_valid is False
        assert error is not None
        assert "exceeds maximum length" in error

    def test_invalid_starts_with_underscore(self):
        """Tenant ID starting with underscore should be invalid."""
        is_valid, error = validate_tenant_id("_private")
        assert is_valid is False
        assert error is not None
        assert "start with alphanumeric" in error

    def test_invalid_starts_with_hyphen(self):
        """Tenant ID starting with hyphen should be invalid."""
        is_valid, error = validate_tenant_id("-bad")
        assert is_valid is False
        assert error is not None
        assert "start with alphanumeric" in error

    def test_invalid_contains_space(self):
        """Tenant ID with spaces should be invalid."""
        is_valid, error = validate_tenant_id("has spaces")
        assert is_valid is False
        assert error is not None
        assert "alphanumeric" in error

    def test_invalid_contains_special_chars(self):
        """Tenant ID with special characters should be invalid."""
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
        """Tenant ID with unicode characters should be invalid."""
        is_valid, error = validate_tenant_id("café")
        assert is_valid is False

    def test_invalid_newline(self):
        """Tenant ID with newline should be invalid (prevents header injection)."""
        is_valid, error = validate_tenant_id("tenant\nX-Evil: header")
        assert is_valid is False

    def test_invalid_null_byte(self):
        """Tenant ID with null byte should be invalid."""
        is_valid, error = validate_tenant_id("tenant\x00evil")
        assert is_valid is False


class TestPerTenantQoS:
    """Tests for per-tenant QoS enforcement."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_tenant_registry()

    def teardown_method(self):
        """Reset global registry after each test."""
        reset_tenant_registry()

    def test_tenant_gets_own_limiters(self):
        """Each tenant should get separate limiter instances."""
        registry = TenantRegistry()

        lim_a_int, lim_a_bulk = registry.get_or_create_limiters("tenant-a")
        lim_b_int, lim_b_bulk = registry.get_or_create_limiters("tenant-b")

        # Different tenants should get different instances
        assert lim_a_int is not lim_b_int
        assert lim_a_bulk is not lim_b_bulk

    def test_same_tenant_gets_same_limiters(self):
        """Same tenant should get same limiter instances on subsequent calls."""
        registry = TenantRegistry()

        lim1_int, lim1_bulk = registry.get_or_create_limiters("tenant-a")
        lim2_int, lim2_bulk = registry.get_or_create_limiters("tenant-a")

        # Same tenant should get same instances
        assert lim1_int is lim2_int
        assert lim1_bulk is lim2_bulk

    def test_tenant_limiter_uses_config_slots(self):
        """Tenant limiters should use configured slot counts."""
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        # Register tenant with custom slots
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
        """Unknown tenants should get default slot counts."""
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        lim_int, lim_bulk = registry.get_or_create_limiters("unknown")
        assert lim_int.capacity == 32
        assert lim_bulk.capacity == 8

    def test_default_tenant_uses_global_defaults(self):
        """Default tenant should use global default slot counts."""
        registry = TenantRegistry(
            default_interactive_slots=16,
            default_bulk_slots=4,
        )

        lim_int, lim_bulk = registry.get_or_create_limiters(DEFAULT_TENANT_ID)
        assert lim_int.capacity == 16
        assert lim_bulk.capacity == 4

    def test_limiter_persists_across_quotas_access(self):
        """Limiters should persist when quotas are accessed multiple times."""
        registry = TenantRegistry()

        # Create limiters
        lim1_int, lim1_bulk = registry.get_or_create_limiters("tenant-a")

        # Access quotas separately
        quotas = registry.get_or_create_quotas("tenant-a")

        # Get limiters again
        lim2_int, lim2_bulk = registry.get_or_create_limiters("tenant-a")

        # Should be same instances
        assert lim1_int is lim2_int
        assert lim1_bulk is lim2_bulk
        assert quotas.interactive_limiter is lim1_int
        assert quotas.bulk_limiter is lim1_bulk

    def test_tenant_config_partial_override(self):
        """Tenant config with only one slot type should use defaults for the other."""
        registry = TenantRegistry(
            default_interactive_slots=32,
            default_bulk_slots=8,
        )

        # Register tenant with only interactive slots override
        config = TenantConfig(
            tenant_id="partial",
            interactive_slots=100,  # Override
            # bulk_slots=None means use default
        )
        registry.register_tenant(config)

        lim_int, lim_bulk = registry.get_or_create_limiters("partial")
        assert lim_int.capacity == 100  # Custom
        assert lim_bulk.capacity == 8  # Default

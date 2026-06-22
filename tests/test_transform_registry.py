"""Tests for transform registry (server-mode transforms)."""

from strata.transforms.registry import (
    TransformDefinition,
    TransformRegistry,
    get_transform_registry,
    reset_transform_registry,
    set_transform_registry,
)


class TestTransformDefinition:
    """Tests for TransformDefinition matching."""

    def test_exact_match(self):
        """Exact ref matches correctly."""
        defn = TransformDefinition(ref="duckdb_sql@v1", executor_url="http://exec:8080")
        assert defn.matches("duckdb_sql@v1")
        assert defn.matches("local://duckdb_sql@v1")  # Strips URI prefix
        assert not defn.matches("duckdb_sql@v2")
        assert not defn.matches("pandas_script@v1")

    def test_wildcard_version(self):
        """Wildcard version matching."""
        defn = TransformDefinition(ref="duckdb_sql@*", executor_url="http://exec:8080")
        assert defn.matches("duckdb_sql@v1")
        assert defn.matches("duckdb_sql@v2")
        assert defn.matches("duckdb_sql@latest")
        assert not defn.matches("pandas_script@v1")

    def test_wildcard_executor(self):
        """Wildcard executor matching."""
        defn = TransformDefinition(ref="*_sql@v1", executor_url="http://exec:8080")
        assert defn.matches("duckdb_sql@v1")
        assert defn.matches("sqlite_sql@v1")
        assert not defn.matches("duckdb_sql@v2")

    def test_full_wildcard(self):
        """Full wildcard matches anything."""
        defn = TransformDefinition(ref="*", executor_url="http://exec:8080")
        assert defn.matches("duckdb_sql@v1")
        assert defn.matches("anything@any_version")


class TestTransformRegistry:
    """Tests for TransformRegistry."""

    def test_disabled_registry_returns_none(self):
        """Disabled registry returns None for all lookups."""
        registry = TransformRegistry(enabled=False, definitions=[])
        assert registry.get("duckdb_sql@v1") is None
        assert not registry.is_allowed("duckdb_sql@v1")

    def test_enabled_with_no_definitions(self):
        """Enabled registry with no definitions returns None."""
        registry = TransformRegistry(enabled=True, definitions=[])
        assert registry.get("duckdb_sql@v1") is None
        assert not registry.is_allowed("duckdb_sql@v1")

    def test_get_returns_matching_definition(self):
        """Get returns matching definition."""
        defn1 = TransformDefinition(ref="duckdb_sql@v1", executor_url="http://duck:8080")
        defn2 = TransformDefinition(ref="pandas_script@*", executor_url="http://pandas:8080")
        registry = TransformRegistry(enabled=True, definitions=[defn1, defn2])

        result = registry.get("local://duckdb_sql@v1")
        assert result is defn1

        result = registry.get("pandas_script@v2")
        assert result is defn2

    def test_first_match_wins(self):
        """First matching definition wins."""
        defn1 = TransformDefinition(ref="*", executor_url="http://catch-all:8080")
        defn2 = TransformDefinition(ref="duckdb_sql@v1", executor_url="http://duck:8080")
        registry = TransformRegistry(enabled=True, definitions=[defn1, defn2])

        # First definition (catch-all) wins
        result = registry.get("duckdb_sql@v1")
        assert result is defn1

    def test_is_allowed(self):
        """is_allowed returns True only for registered transforms."""
        defn = TransformDefinition(ref="duckdb_sql@v1", executor_url="http://exec:8080")
        registry = TransformRegistry(enabled=True, definitions=[defn])

        assert registry.is_allowed("duckdb_sql@v1")
        assert not registry.is_allowed("unknown@v1")

    def test_from_config_empty_embedded_mode(self):
        """from_config with empty config returns embedded registry by default."""
        registry = TransformRegistry.from_config({})
        assert registry.enabled
        # Should have embedded duckdb_sql@v1
        assert registry.is_allowed("duckdb_sql@v1")
        defn = registry.get("duckdb_sql@v1")
        assert defn is not None
        assert defn.executor_url == "embedded://local"

    def test_from_config_empty_registry_mode(self):
        """from_config with empty config and embedded_mode=False returns disabled registry."""
        registry = TransformRegistry.from_config({}, embedded_mode=False)
        assert not registry.enabled
        assert len(registry.definitions) == 0

    def test_from_config_enabled(self):
        """from_config parses full configuration."""
        config = {
            "enabled": True,
            "registry": [
                {
                    "ref": "duckdb_sql@v1",
                    "executor_url": "http://executor:8080/execute",
                    "timeout_seconds": 300,
                    "max_output_bytes": 1073741824,
                },
                {
                    "ref": "pandas_script@*",
                    "executor_url": "http://python:8080/execute",
                    "max_input_bytes": 536870912,
                    "requires_scope": "transform:pandas",
                },
            ],
        }

        registry = TransformRegistry.from_config(config)
        assert registry.enabled
        assert len(registry.definitions) == 2

        # Check first definition
        defn1 = registry.definitions[0]
        assert defn1.ref == "duckdb_sql@v1"
        assert defn1.executor_url == "http://executor:8080/execute"
        assert defn1.timeout_seconds == 300
        assert defn1.max_output_bytes == 1073741824
        assert defn1.requires_scope is None

        # Check second definition
        defn2 = registry.definitions[1]
        assert defn2.ref == "pandas_script@*"
        assert defn2.max_input_bytes == 536870912
        assert defn2.requires_scope == "transform:pandas"


class TestSingletons:
    """Tests for module-level singleton functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_transform_registry()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_transform_registry()

    def test_default_registry_disabled(self):
        """Default registry is disabled."""
        registry = get_transform_registry()
        assert not registry.enabled

    def test_set_and_get_registry(self):
        """set_transform_registry updates the singleton."""
        defn = TransformDefinition(ref="test@v1", executor_url="http://test:8080")
        custom = TransformRegistry(enabled=True, definitions=[defn])

        set_transform_registry(custom)
        retrieved = get_transform_registry()

        assert retrieved is custom
        assert retrieved.enabled
        assert retrieved.is_allowed("test@v1")

    def test_reset_clears_registry(self):
        """reset_transform_registry clears the singleton."""
        defn = TransformDefinition(ref="test@v1", executor_url="http://test:8080")
        custom = TransformRegistry(enabled=True, definitions=[defn])
        set_transform_registry(custom)

        reset_transform_registry()

        # After reset, get returns a new disabled registry
        registry = get_transform_registry()
        assert not registry.enabled

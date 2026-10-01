"""Transform registry for server-mode execution.

The transform registry is an allowlist of approved transforms that can be
executed in server mode. Each transform definition specifies:
- How to identify the transform (ref pattern matching)
- Where to execute it (executor URL)
- Resource limits (timeout, max output size)

In service mode, only registered transforms can be materialized.
In personal mode, the registry is bypassed (client executes locally).

Example configuration in pyproject.toml:

    [tool.strata.transforms]
    enabled = true

    [[tool.strata.transforms.registry]]
    ref = "duckdb_sql@v1"
    executor_url = "http://executor:8080/execute"
    timeout_seconds = 300
    max_output_bytes = 1073741824  # 1 GB

    [[tool.strata.transforms.registry]]
    ref = "pandas_script@*"  # Wildcard version matching
    executor_url = "http://python-executor:8080/execute"
    timeout_seconds = 600
    max_output_bytes = 536870912  # 512 MB
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransformDefinition:
    """Definition of an approved transform.

    Attributes:
        ref: Transform reference pattern (e.g., "duckdb_sql@v1" or "pandas_*@*")
            Supports glob-style wildcards for flexible matching.
        executor_url: HTTP URL for the executor endpoint.
            The executor receives input streams and returns output stream.
        timeout_seconds: Maximum execution time in seconds.
        max_output_bytes: Maximum output size in bytes (0 = unlimited).
        max_input_bytes: Maximum total input size in bytes (0 = unlimited).
        requires_scope: Optional scope required to use this transform.
            If set, the principal must have this scope to materialize.
    """

    ref: str
    executor_url: str
    timeout_seconds: float = 300.0
    max_output_bytes: int = 0  # 0 = unlimited
    max_input_bytes: int = 0  # 0 = unlimited
    requires_scope: str | None = None

    def matches(self, executor_ref: str) -> bool:
        """Check if this definition matches an executor reference.

        Args:
            executor_ref: Executor reference to match (e.g., "duckdb_sql@v1")

        Returns:
            True if this definition matches the reference
        """
        # Strip a scheme: "local://duckdb_sql@v1" -> "duckdb_sql@v1".
        if "://" in executor_ref:
            executor_ref = executor_ref.split("://", 1)[1]

        return fnmatch.fnmatch(executor_ref, self.ref)


@dataclass
class TransformRegistry:
    """Registry of approved transforms for server-mode execution.

    The registry acts as an allowlist - only transforms with matching
    definitions can be executed in server mode.

    Thread-safe: all operations are read-only after initialization.
    """

    enabled: bool = False

    definitions: list[TransformDefinition] = field(default_factory=list)

    def get(self, executor_ref: str) -> TransformDefinition | None:
        """Look up a transform definition by executor reference.

        Args:
            executor_ref: Executor reference (e.g., "local://duckdb_sql@v1")

        Returns:
            Matching TransformDefinition, or None if not found
        """
        if not self.enabled:
            return None

        for defn in self.definitions:
            if defn.matches(executor_ref):
                return defn

        return None

    def is_allowed(self, executor_ref: str) -> bool:
        """Check if a transform is allowed.

        Args:
            executor_ref: Executor reference to check

        Returns:
            True if transform is registered and allowed
        """
        return self.get(executor_ref) is not None

    @classmethod
    def create_embedded_registry(cls) -> TransformRegistry:
        """Create a registry with default embedded executors.

        This is used for local deployment where no external executor
        services are configured. Common transforms like duckdb_sql@v1
        run directly in the server process.

        Returns:
            TransformRegistry with embedded executor definitions
        """
        # Transforms that can run in-process (no external HTTP calls).
        embedded_transforms = ["duckdb_sql@v1"]

        definitions = [
            TransformDefinition(
                ref=ref,
                executor_url="embedded://local",
                timeout_seconds=300.0,
                max_output_bytes=1024 * 1024 * 1024,  # 1GB
                max_input_bytes=1024 * 1024 * 1024,  # 1GB
            )
            for ref in embedded_transforms
        ]

        logger.info(f"Created embedded registry with transforms: {embedded_transforms}")

        return cls(enabled=True, definitions=definitions)

    @classmethod
    def from_config(cls, config: dict, embedded_mode: bool = True) -> TransformRegistry:
        """Create registry from configuration dictionary.

        Expected format (from pyproject.toml [tool.strata.transforms]):
            {
                "enabled": true,
                "registry": [
                    {
                        "ref": "duckdb_sql@v1",
                        "executor_url": "http://executor:8080/execute",
                        "timeout_seconds": 300,
                        "max_output_bytes": 1073741824
                    },
                    ...
                ]
            }

        Args:
            config: Configuration dictionary
            embedded_mode: If True and no config provided, enable embedded
                           executors for common transforms (default for local deployment)

        Returns:
            Configured TransformRegistry
        """
        if not config:
            if embedded_mode:
                return cls.create_embedded_registry()
            return cls(enabled=False, definitions=[])

        enabled = config.get("enabled", False)
        definitions = []

        for entry in config.get("registry", []):
            defn = TransformDefinition(
                ref=entry["ref"],
                executor_url=entry.get("executor_url", ""),
                timeout_seconds=entry.get("timeout_seconds", 300.0),
                max_output_bytes=entry.get("max_output_bytes", 0),
                max_input_bytes=entry.get("max_input_bytes", 0),
                requires_scope=entry.get("requires_scope"),
            )
            definitions.append(defn)
            logger.debug(f"Registered transform: {defn.ref} -> {defn.executor_url}")

        logger.info(
            f"Transform registry initialized: enabled={enabled}, definitions={len(definitions)}"
        )

        return cls(enabled=enabled, definitions=definitions)


_registry: TransformRegistry | None = None


def get_transform_registry() -> TransformRegistry:
    """Get the transform registry singleton.

    Returns:
        TransformRegistry instance (may be disabled if not configured)
    """
    global _registry
    if _registry is None:
        _registry = TransformRegistry(enabled=False, definitions=[])
    return _registry


def set_transform_registry(registry: TransformRegistry) -> None:
    """Set the transform registry singleton.

    Args:
        registry: TransformRegistry to use
    """
    global _registry
    _registry = registry


def reset_transform_registry() -> None:
    """Reset the transform registry singleton (for testing)."""
    global _registry
    _registry = None

"""Transform registry: the allowlist of transforms that may run in service mode.

Each definition matches transform refs (glob, e.g. ``pandas_script@*``) and gives the
executor URL and resource limits. Personal mode bypasses the allowlist. Configured in
pyproject.toml:

    [tool.strata.transforms]
    enabled = true

    [[tool.strata.transforms.registry]]
    ref = "duckdb_sql@v1"
    executor_url = "http://executor:8080/execute"
    timeout_seconds = 300
    max_output_bytes = 1073741824  # 1 GB
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
    """An approved transform: a glob ``ref`` pattern, its executor URL, and limits.

    Byte limits of 0 mean unlimited. When ``requires_scope`` is set, the principal must hold
    that scope to materialize.
    """

    ref: str
    executor_url: str
    timeout_seconds: float = 300.0
    max_output_bytes: int = 0  # 0 = unlimited
    max_input_bytes: int = 0  # 0 = unlimited
    requires_scope: str | None = None

    def matches(self, executor_ref: str) -> bool:
        """Whether *executor_ref* matches ``ref``, ignoring any scheme (``local://``)."""
        # Strip a scheme: "local://duckdb_sql@v1" -> "duckdb_sql@v1".
        if "://" in executor_ref:
            executor_ref = executor_ref.split("://", 1)[1]

        return fnmatch.fnmatch(executor_ref, self.ref)


@dataclass
class TransformRegistry:
    """Allowlist of approved transforms; thread-safe because it is read-only after init."""

    enabled: bool = False

    definitions: list[TransformDefinition] = field(default_factory=list)

    def get(self, executor_ref: str) -> TransformDefinition | None:
        """Return the first definition matching *executor_ref*; None when unmatched or disabled."""
        if not self.enabled:
            return None

        for defn in self.definitions:
            if defn.matches(executor_ref):
                return defn

        return None

    def is_allowed(self, executor_ref: str) -> bool:
        """Whether *executor_ref* is registered and the registry is enabled."""
        return self.get(executor_ref) is not None

    @classmethod
    def create_embedded_registry(cls) -> TransformRegistry:
        """Create an enabled registry of in-process transforms (``duckdb_sql@v1``)."""
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
        """Create a registry from the ``[tool.strata.transforms]`` dict (``enabled``, ``registry``).

        An empty config gives the embedded registry when ``embedded_mode``, else a disabled one.
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
    """Get the transform registry singleton (disabled if not configured)."""
    global _registry
    if _registry is None:
        _registry = TransformRegistry(enabled=False, definitions=[])
    return _registry


def set_transform_registry(registry: TransformRegistry) -> None:
    """Set the transform registry singleton."""
    global _registry
    _registry = registry


def reset_transform_registry() -> None:
    """Reset the transform registry singleton (for testing)."""
    global _registry
    _registry = None

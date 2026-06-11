"""Strata: Snapshot-aware serving layer for Iceberg tables."""

from typing import TYPE_CHECKING

from strata.client import AsyncStrataClient, RetryConfig, StrataClient
from strata.filters import Filter, FilterOp, FilterValue

if TYPE_CHECKING:
    from strata.config import StrataConfig
    from strata.integration.duckdb import register_strata_scan
    from strata.types import CacheKey, ReadPlan, Task

__all__ = [
    "AsyncStrataClient",
    "CacheKey",
    "Filter",
    "FilterOp",
    "FilterValue",
    "ReadPlan",
    "RetryConfig",
    "StrataClient",
    "StrataConfig",
    "Task",
    "register_strata_scan",
]

# Heavy / server-side exports are resolved lazily (PEP 562) so a plain
# ``import strata`` stays light — it must not pull in duckdb (via
# ``register_strata_scan``) or pydantic (via ``StrataConfig`` / ``strata.types``).
# This is what lets the client be installed without the server's dependency
# stack; see docs/internal/design-strata-client.md.
_LAZY_EXPORTS = {
    "StrataConfig": ("strata.config", "StrataConfig"),
    "register_strata_scan": ("strata.integration.duckdb", "register_strata_scan"),
    "CacheKey": ("strata.types", "CacheKey"),
    "ReadPlan": ("strata.types", "ReadPlan"),
    "Task": ("strata.types", "Task"),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])

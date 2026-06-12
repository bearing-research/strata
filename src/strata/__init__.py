"""Strata: Snapshot-aware serving layer for Iceberg tables.

This is the **server** distribution. The HTTP *client* is a separate, slim
package — ``pip install strata-client`` then ``from strata_client import
StrataClient``. The server and client are independent: they share only the JSON
wire protocol, not code (each owns its copy of the ``Filter`` wire types).
See docs/internal/design-strata-client.md.
"""

from typing import TYPE_CHECKING

from strata.filters import Filter, FilterOp, FilterValue

if TYPE_CHECKING:
    from strata.config import StrataConfig
    from strata.types import CacheKey, ReadPlan, Task

__all__ = [
    "CacheKey",
    "Filter",
    "FilterOp",
    "FilterValue",
    "ReadPlan",
    "StrataConfig",
    "Task",
]

# Server-side exports resolved lazily (PEP 562) so a plain ``import strata``
# stays cheap and doesn't eagerly pull pydantic (via ``StrataConfig`` /
# ``strata.types``).
_LAZY_EXPORTS = {
    "StrataConfig": ("strata.config", "StrataConfig"),
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

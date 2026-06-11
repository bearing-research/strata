"""Backward-compatible alias for the Filter types.

The filter types now live in the standalone ``strata-client`` distribution
(``strata_client.filters``). This module re-exports them so existing
``from strata.filters import Filter`` / ``from strata.types import Filter``
imports keep working — a single implementation, re-exported. See
docs/internal/design-strata-client.md.
"""

from strata_client.filters import (
    Filter,
    FilterOp,
    FilterValue,
    SupportsOrdering,
    compute_filter_fingerprint,
)

__all__ = [
    "Filter",
    "FilterOp",
    "FilterValue",
    "SupportsOrdering",
    "compute_filter_fingerprint",
]

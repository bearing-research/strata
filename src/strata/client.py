"""Backward-compatible alias for the Strata client.

The client now lives in the standalone ``strata-client`` distribution
(``strata_client``), which depends only on httpx + pyarrow. This module
re-exports it so existing ``from strata.client import StrataClient`` imports
keep working. There is a single implementation — this is a re-export, not a
copy. See docs/internal/design-strata-client.md.
"""

from strata_client.client import (
    Artifact,
    AsyncStrataClient,
    RetryConfig,
    StrataClient,
    eq,
    ge,
    gt,
    le,
    lt,
    ne,
)

__all__ = [
    "Artifact",
    "AsyncStrataClient",
    "RetryConfig",
    "StrataClient",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "ne",
]

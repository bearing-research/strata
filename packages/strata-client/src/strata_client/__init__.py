"""strata_client — a lightweight Python client for a Strata server.

Depends only on ``httpx`` + ``pyarrow`` — none of the Strata server's stack
(no pyiceberg / fastapi / duckdb / pydantic, no Rust extension). Install it
anywhere you want to *use* Strata as a library::

    pip install strata-client

    from strata_client import StrataClient

    with StrataClient() as client:
        art = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(art.uri)

The server distribution (``strata-notebook``) depends on this package and
re-exports it as ``strata.client`` / ``strata.filters`` for backward
compatibility. See docs/internal/design-strata-client.md.
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
from strata_client.filters import Filter, FilterOp, FilterValue, compute_filter_fingerprint

__all__ = [
    "Artifact",
    "AsyncStrataClient",
    "Filter",
    "FilterOp",
    "FilterValue",
    "RetryConfig",
    "StrataClient",
    "compute_filter_fingerprint",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "ne",
]

"""strata_client: a lightweight Python client for a Strata server.

Depends only on ``httpx`` and ``pyarrow``, none of the server's stack::

    pip install strata-client

    from strata_client import StrataClient

    with StrataClient() as client:
        art = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(art.uri)

The server distribution (``strata-notebook``) re-exports it as
``strata.client`` / ``strata.filters``.
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

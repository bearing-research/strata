"""Model tokens used by this server, by tenant, principal and model.

Every provider response already carried its token counts, and each route
returned them for the one call; nothing added them up, so metering model use
per organization or per member meant logging every response and summing it
elsewhere. Recorded where each response is read, and exported as counters on
``/metrics/prometheus``.

The caller is whoever is in the request context (``strata.auth``): the
principal an HTTP route or WebSocket was opened as, or no one in personal
mode. Agent runs and prompt cells run in tasks that copy that context.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

from strata.auth import get_principal


class UsageRow(NamedTuple):
    tenant: str
    principal: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int


_lock = threading.Lock()
# (tenant, principal, model) -> [calls, input tokens, output tokens]
_totals: dict[tuple[str, str, str], list[int]] = {}


def record_llm_usage(model: str, input_tokens: int, output_tokens: int) -> None:
    """Add one provider response's counts to the caller's totals."""
    principal = get_principal()
    key = (
        (principal.tenant or "") if principal is not None else "",
        principal.id if principal is not None else "",
        model or "",
    )
    with _lock:
        totals = _totals.setdefault(key, [0, 0, 0])
        totals[0] += 1
        totals[1] += int(input_tokens or 0)
        totals[2] += int(output_tokens or 0)


def llm_usage() -> list[UsageRow]:
    with _lock:
        return [UsageRow(*key, *totals) for key, totals in sorted(_totals.items())]


def reset_llm_usage() -> None:
    with _lock:
        _totals.clear()

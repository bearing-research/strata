"""Secret-provider protocol + shared return type."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol


class SecretProviderError(RuntimeError):
    """Raised for unrecoverable provider-config errors (unknown provider, bad shape).

    Network and auth failures come back in ``SecretFetchResult.error`` instead, so
    they do not break the session open.
    """


@dataclass(frozen=True)
class SecretFetchResult:
    """What a provider returns from a ``fetch`` call.

    ``secrets`` is always a dict, empty on error. ``error`` is a user-facing message
    for the Runtime panel; ``fetched_at`` is an ISO-8601 UTC timestamp. A partial
    fetch is success: a non-empty ``secrets`` with ``error = None``.
    """

    secrets: dict[str, str] = field(default_factory=dict)
    source: str = ""
    fetched_at: str = ""
    error: str | None = None

    @classmethod
    def failure(cls, source: str, message: str) -> SecretFetchResult:
        return cls(
            secrets={},
            source=source,
            fetched_at=_now_iso(),
            error=message,
        )


def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


# Applied by a provider to each request, and by the server to the whole fetch.
SECRET_FETCH_TIMEOUT_SECONDS = 20.0


class SecretProvider(Protocol):
    """Minimal interface every secret-manager integration implements.

    ``fetch(config, timeout=...)`` runs on session open and on explicit refresh. It
    must not raise on network or auth errors; it returns them in ``error`` and the
    session keeps running. No single request may wait longer than ``timeout``
    seconds.
    """

    name: str

    def fetch(self, config: dict[str, Any], *, timeout: float) -> SecretFetchResult: ...

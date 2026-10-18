"""Run tokens: how a cell's ambient client calls its own server under principal auth.

A cell holds no proxy token or API key, so under ``trusted_proxy`` or ``api_key`` its
``strata.materialize`` would get 401. For each local run the server mints a random token,
hands it to the harness over a pipe (never a file, argv or the environment, which other
cells under the same harness user could read), and accepts it in place of the proxy token
while the run lives. The request acts as whoever started the run, without ``admin:``
scopes, and reaches only the routes the ambient client calls.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from strata.types import Principal

RUN_TOKEN_HEADER = "X-Strata-Run-Token"

# A run that outlives its timeout is being killed; this covers the kill and the last request.
_EXPIRY_MARGIN_SECONDS = 60.0

# What ``notebook_client.StrataClient`` calls. Nothing that edits notebooks, environments
# or workers.
_CELL_ROUTES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (method, re.compile(pattern))
    for method, pattern in (
        ("POST", r"/v1/materialize"),
        ("GET", r"/v1/streams/[^/]+"),
        ("GET", r"/v1/artifacts/[^/]+/v/\d+/data"),
        ("PUT", r"/v1/artifacts"),
        ("GET", r"/v1/artifacts/[^/]+/v/\d+/tags"),
        ("PUT", r"/v1/artifacts/[^/]+/v/\d+/tags"),
        ("POST", r"/v1/names"),
        ("GET", r"/v1/names/.+"),
        ("PUT", r"/v1/names/.+/aliases/[^/]+"),
        ("GET", r"/v1/registry/audit"),
    )
)
_PROMOTE_ROUTE = re.compile(r"/v1/notebooks/([^/]+)/artifacts/[^/]+/v/\d+/promote")


@dataclass(frozen=True)
class RunGrant:
    """What a live run's token lets its cell do."""

    principal: Principal
    session_id: str
    expires_at: float

    def allows(self, method: str, path: str) -> bool:
        """Whether the run's cell may call ``method path``: its routes, promote for its session."""
        if any(m == method and p.fullmatch(path) for m, p in _CELL_ROUTES):
            return True
        promote = _PROMOTE_ROUTE.fullmatch(path) if method == "POST" else None
        return promote is not None and promote.group(1) == self.session_id


# Keyed by the token's digest, so a lookup compares no secret byte by byte.
_live: dict[str, RunGrant] = {}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@contextmanager
def run_token(principal: Principal, session_id: str, timeout_seconds: float) -> Generator[str]:
    """A token for one run, live until the block exits or the timeout and a margin pass."""
    token = secrets.token_urlsafe(32)
    key = _digest(token)
    _live[key] = RunGrant(
        principal=replace(
            principal,
            scopes=frozenset(s for s in principal.scopes if not s.startswith("admin:")),
        ),
        session_id=session_id,
        expires_at=time.monotonic() + timeout_seconds + _EXPIRY_MARGIN_SECONDS,
    )
    try:
        yield token
    finally:
        _live.pop(key, None)


def grant_for(token: str) -> RunGrant | None:
    """The live run's grant, or ``None`` once its run has ended or expired."""
    grant = _live.get(_digest(token))
    if grant is None or time.monotonic() > grant.expires_at:
        return None
    return grant

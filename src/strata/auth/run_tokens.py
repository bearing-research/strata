"""Run tokens: how a cell's ambient client calls its own server under principal auth.

A cell holds no proxy token or API key, so under ``trusted_proxy`` or ``api_key`` its
``strata.materialize`` would get 401. For each local run the server mints a random token,
hands it to the harness over a pipe (never a file, argv or the environment, which other
cells under the same harness user could read), and accepts it in place of the proxy token
while the run lives. The request acts as whoever started the run, without ``admin:``
scopes, and reaches only the routes the ambient client calls.

Every local run gets a token, authenticated or not, so the server can also count what the
run's scans read from storage, by cell and table, for the notebook's reads index.
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
# Which cell of the run made a call, for counting only: a batch runs many under one token.
RUN_CELL_HEADER = "X-Strata-Cell-Id"

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
# A run's read meter, ``{cell id: {table uri: bytes read from storage}}``, owned by the
# caller, which reads it after the run. Same keys.
_meters: dict[str, dict[str, dict[str, int]]] = {}


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@contextmanager
def run_token(
    principal: Principal | None,
    session_id: str,
    timeout_seconds: float,
    meter: dict[str, dict[str, int]] | None = None,
) -> Generator[str]:
    """A token for one run, live until the block exits or the timeout and a margin pass.

    Without a *principal* the token authenticates nothing; it only counts the run's scans
    into *meter*.
    """
    token = secrets.token_urlsafe(32)
    key = _digest(token)
    if principal is not None:
        _live[key] = RunGrant(
            principal=replace(
                principal,
                scopes=frozenset(s for s in principal.scopes if not s.startswith("admin:")),
            ),
            session_id=session_id,
            expires_at=time.monotonic() + timeout_seconds + _EXPIRY_MARGIN_SECONDS,
        )
    if meter is not None:
        _meters[key] = meter
    try:
        yield token
    finally:
        _live.pop(key, None)
        _meters.pop(key, None)


def grant_for(token: str) -> RunGrant | None:
    """The live run's grant, or ``None`` once its run has ended or expired."""
    grant = _live.get(_digest(token))
    if grant is None or time.monotonic() > grant.expires_at:
        return None
    return grant


def run_reader(token: str | None, cell_id: str | None) -> tuple[str, str] | None:
    """Who a scan's storage reads count for: the run's digest and its cell, if metered."""
    if token is None or not cell_id:
        return None
    key = _digest(token)
    return (key, cell_id) if key in _meters else None


def count_storage_read(reader: tuple[str, str], table_uri: str, nbytes: int) -> None:
    """Add a finished scan's storage bytes to its run's meter, if the run still lives."""
    key, cell_id = reader
    meter = _meters.get(key)
    if meter is not None:
        tables = meter.setdefault(cell_id, {})
        tables[table_uri] = tables.get(table_uri, 0) + nbytes

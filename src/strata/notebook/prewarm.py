"""Pre-warm hints: tell a remote worker's dispatcher that a run is near.

A dispatcher in front of a pool (``strata-pool``'s ``POST /v1/hints``) can boot a
machine for the worker while the user is still typing, instead of in their first
run. The notebook knows the moments: a cell bound to the worker gains focus, the
worker becomes the notebook's default, a cascade or Run All is about to dispatch.

Best effort and silent: a worker that serves no ``/v1/hints`` (a plain
``strata-worker``) answers 404 and nothing changes. At most one hint per session and
worker a minute, since a dispatcher treats repeats as no-ops anyway.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

import httpx

from strata.notebook.models import WorkerBackendType
from strata.notebook.workers import one_policy_read, read_worker_policy, resolve_worker_spec
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

logger = logging.getLogger(__name__)

HintReason = Literal["focus", "select", "cascade", "run_all"]

HINT_INTERVAL_SECONDS = 60.0
HINT_TIMEOUT_SECONDS = 5.0

# (session id, worker name) -> when its last hint went out.
_last_sent: dict[tuple[str, str], float] = {}
# Hints in flight, held so they are not garbage collected mid-send.
_in_flight: set[asyncio.Task] = set()

_EXECUTE_SUFFIXES = ("/v1/execute", "/v1/notebook-execute", "/v1/execute-manifest", "/execute")


def hints_url(executor_url: str) -> str:
    """The ``/v1/hints`` route beside a worker's execute URL, as manifests are mapped."""
    parsed = urlparse(executor_url)
    path = parsed.path or ""
    for suffix in _EXECUTE_SUFFIXES:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    path = f"{path.rstrip('/')}/v1/hints"
    return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))


async def send_hint(session: Any, worker_name: str | None, reason: HintReason) -> bool:
    """Send one hint for *worker_name* if it is a remote worker this session may use.

    Returns whether a dispatcher accepted it. Never raises.
    """
    if not worker_name:
        return False
    key = (session.id, worker_name)
    now = time.monotonic()
    if now - _last_sent.get(key, float("-inf")) < HINT_INTERVAL_SECONDS:
        return False
    state = session.notebook_state
    try:
        # The run's own resolution: a worker the policy refuses gets no machine started.
        policy = await read_worker_policy(state, worker_name)
        with one_policy_read(state, policy):
            spec = resolve_worker_spec(state, worker_name)
    except Exception:
        logger.debug("No pre-warm hint for worker %s", worker_name, exc_info=True)
        return False
    if spec is None or spec.backend != WorkerBackendType.EXECUTOR or not spec.config.url:
        return False
    _last_sent[key] = now

    from strata.notebook.executor import _resolve_worker_token

    headers = {EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION}
    token = _resolve_worker_token(spec)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "worker": spec.name,
        "runtime_id": spec.runtime_id,
        "notebook_id": state.id,
        "reason": reason,
    }
    try:
        async with httpx.AsyncClient(timeout=HINT_TIMEOUT_SECONDS) as client:
            response = await client.post(
                hints_url(str(spec.config.url)), json=body, headers=headers
            )
    except httpx.HTTPError as exc:
        logger.debug("Pre-warm hint to %s failed: %s", spec.name, exc)
        return False
    return response.status_code < 300


def hint_in_background(
    session: Any, worker_names: Iterable[str | None], reason: HintReason
) -> None:
    """Send a hint for each distinct worker without holding up the caller."""
    for name in dict.fromkeys(n for n in worker_names if n):
        task = asyncio.get_running_loop().create_task(send_hint(session, name, reason))
        _in_flight.add(task)
        task.add_done_callback(_in_flight.discard)


async def settled() -> None:
    """Wait for the hints in flight: for tests and shutdown."""
    while _in_flight:
        await asyncio.gather(*list(_in_flight), return_exceptions=True)


def forget_session(session_id: str) -> None:
    """Drop a closed session's rate-limit entries."""
    for key in [k for k in _last_sent if k[0] == session_id]:
        del _last_sent[key]

"""Forward registry routes to the team store when ``notebook_remote_store_url`` is set.

Cells write named artifacts there, so the local store would show an empty registry.
Forwarding happens server-side so ``notebook_remote_store_headers`` never reach the
browser. The caller's principal is forwarded when configured, but the registry shown
is the organization's, not a per-user view.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from strata.artifact_transfer import detail_of
from strata.logging import get_logger

logger = get_logger(__name__)

# The dashboard is an interactive panel: a slow team store should say so rather
# than hang a tab. Longer than a keystroke, far shorter than a copy.
REGISTRY_TIMEOUT_SECONDS = 20.0


def remote_registry() -> tuple[str, dict[str, str]] | None:
    """Return the team store's ``(base_url, headers)``, or ``None`` to use the local store."""
    from strata.server import get_state

    try:
        config = get_state().config
    except RuntimeError:
        # No server state (a direct handler call in a test, say). The local
        # store is the only one there is.
        return None
    url = getattr(config, "notebook_remote_store_url", None)
    if not url:
        return None
    from strata.auth import remote_store_headers

    return str(url).rstrip("/"), remote_store_headers(config)


async def _send(
    target: tuple[str, dict[str, str]],
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> httpx.Response:
    """One request against the team store; a failure to ask becomes a 502."""
    base_url, headers = target
    try:
        async with httpx.AsyncClient(timeout=REGISTRY_TIMEOUT_SECONDS) as client:
            response = await client.request(
                method,
                f"{base_url}{path}",
                params={k: v for k, v in (params or {}).items() if v is not None},
                json=json_body,
                headers=headers,
            )
    except httpx.HTTPError as exc:
        logger.warning("Registry request to the team store failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"The team store at {base_url}: {exc}")

    if response.status_code >= 400:
        raise HTTPException(status_code=response.status_code, detail=detail_of(response))
    return response


async def forward(
    target: tuple[str, dict[str, str]],
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    """Make one registry request against the team store and return its JSON body.

    Error statuses from the store pass through as ``HTTPException``; only a failure
    to reach it becomes a 502. Use :func:`relay` to keep a success status too.
    """
    response = await _send(target, method, path, params=params, json_body=json_body)
    return response.json()


async def relay(
    target: tuple[str, dict[str, str]],
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> JSONResponse:
    """Forward a request and return the team store's answer, status included.

    A protected alias answers 202 (pending); a client deciding by status, such as
    ``RemoteStore.set_alias``, must not see that as an applied 200.
    """
    response = await _send(target, method, path, params=params, json_body=json_body)
    return JSONResponse(status_code=response.status_code, content=response.json())


def quoted(segment: str, *, path: bool = False) -> str:
    """URL-quote a path segment; with ``path=True`` slashes survive (names like ``taxi/model``)."""
    return quote(segment, safe="/" if path else "")

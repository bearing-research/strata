"""Answer the registry routes from the store the cells actually write to.

With ``notebook_remote_store_url`` set, a cell's ``strata.put(name=...)`` lands
in the team's store and a promotion copies a chain there. The dashboard read
the *local* store, so it showed an empty registry on exactly the deployment
where the registry is most useful — everything the notebook names is somewhere
else.

The forwarding happens here rather than in the browser because the credentials
do. ``notebook_remote_store_headers`` is the trusted-proxy identity the server
holds; handing it to a page so the page could call the team store directly
would put it in every user's devtools.

The team store sees the server's identity, with the caller's principal
forwarded when there is one (``notebook_remote_store_forward_principal``), so an
approval from the Registry tab is the member's. The registry a viewer sees is
still the organization's: a personal server is one person, and a shared one is
one organization. It is not a per-user view of a shared registry, and pointing
several organizations at one server would not make it one.
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
    """The team store the dashboard should describe, or ``None`` for this one.

    ``None`` is the ordinary single-machine case, not a failure: with no team
    store configured the local store *is* the registry the cells write to.
    """
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
    """Make one registry request against the team store and return its body.

    The far side's status codes are passed through rather than flattened: a
    protected alias answering 403 for separation of duty, or 404 for a pending
    change someone else already approved, are answers the dashboard knows how
    to show. Only "could not ask" becomes a 502, because that is this server's
    news to report and not the store's.

    For a caller that consumes the body. A route that hands the answer straight
    back uses :func:`relay`, which keeps a success status too.
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
    """Forward a request and return the team store's answer as this route's own.

    Keeps the success status, which :func:`forward` drops. A protected alias
    answers 202 with ``status: pending``; the dashboard reads the body and would
    cope, but a client that decides by the status — ``RemoteStore.set_alias``
    does — would take a queued change for an applied one.
    """
    response = await _send(target, method, path, params=params, json_body=json_body)
    return JSONResponse(status_code=response.status_code, content=response.json())


def quoted(segment: str, *, path: bool = False) -> str:
    """A path segment made safe to put back into a URL.

    Names are paths (``taxi/model``), so their slashes survive; anything that
    would end the path or start a query does not.
    """
    return quote(segment, safe="/" if path else "")

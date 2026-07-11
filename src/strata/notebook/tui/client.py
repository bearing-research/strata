"""REST + WS plumbing for the notebook TUI spectator.

Thin async client over the notebook server's HTTP surface: list the caller's
running sessions, open/reuse a notebook by path, and derive the WS URL. Kept
separate from the Textual app so the bootstrap flow is easy to read and the app
stays a renderer.
"""

from __future__ import annotations

from typing import Any

import httpx


class TuiClientError(RuntimeError):
    """A server call failed; carries a human-readable detail for the UI."""


class TuiClient:
    """Async REST client for the spectator's bootstrap calls."""

    def __init__(self, server_url: str, auth_headers: dict[str, str] | None = None) -> None:
        self.server_url = server_url.rstrip("/")
        self.auth_headers = auth_headers or {}

    async def list_sessions(self) -> list[dict[str, Any]]:
        """Return the caller's currently-open sessions (id, name, path, …)."""
        data = await self._get("/v1/notebooks/sessions")
        sessions = data.get("sessions")
        return sessions if isinstance(sessions, list) else []

    async def open_notebook(self, path: str) -> dict[str, Any]:
        """``POST /v1/notebooks/open`` — resolve/reuse a session for *path*."""
        return await self._post("/v1/notebooks/open", {"path": path})

    def ws_url(self, session_id: str) -> str:
        """WS URL for a session, derived by swapping the HTTP scheme."""
        base = self.server_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return f"{base}/v1/notebooks/ws/{session_id}"

    async def get_cell_data_page(
        self,
        notebook_id: str,
        cell_id: str,
        artifact_uri: str,
        *,
        offset: int = 0,
        limit: int = 50,
        sort_by: str | None = None,
        sort_dir: str = "asc",
    ) -> dict[str, Any]:
        """Fetch one page of a cell output's cached DataFrame (data viewer)."""
        params: dict[str, str] = {
            "artifact_uri": artifact_uri,
            "offset": str(offset),
            "limit": str(limit),
        }
        if sort_by:
            params["sort_by"] = sort_by
            params["sort_dir"] = sort_dir
        return await self._get(f"/v1/notebooks/{notebook_id}/cells/{cell_id}/data", params)

    async def export_cell_data(
        self,
        notebook_id: str,
        cell_id: str,
        artifact_uri: str,
        *,
        fmt: str = "csv",
        sort_by: str | None = None,
        sort_dir: str = "asc",
    ) -> bytes:
        """Download a cell output's DataFrame as raw CSV/Parquet bytes."""
        params: dict[str, str] = {"artifact_uri": artifact_uri, "fmt": fmt}
        if sort_by:
            params["sort_by"] = sort_by
            params["sort_dir"] = sort_dir
        path = f"/v1/notebooks/{notebook_id}/cells/{cell_id}/data/export"
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.get(
                    self.server_url + path, params=params, headers=self.auth_headers
                )
            except httpx.HTTPError as exc:
                raise TuiClientError(f"cannot reach {self.server_url}: {exc}") from exc
            if response.is_error:
                raise TuiClientError(f"export failed ({response.status_code})")
            return response.content

    # -- internals -----------------------------------------------------------

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    self.server_url + path, params=params, headers=self.auth_headers
                )
            except httpx.HTTPError as exc:
                raise TuiClientError(f"cannot reach {self.server_url}: {exc}") from exc
            return _json_or_error(response)

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    self.server_url + path, json=body, headers=self.auth_headers
                )
            except httpx.HTTPError as exc:
                raise TuiClientError(f"cannot reach {self.server_url}: {exc}") from exc
            return _json_or_error(response)


def _json_or_error(response: httpx.Response) -> dict[str, Any]:
    """Return the JSON body, or raise with the server's ``detail`` surfaced.

    Without surfacing ``detail`` the user sees a bare "400 Bad Request" and has
    to dig through server logs for e.g. "personal mode only" or "Notebook not
    found".
    """
    if response.is_error:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("detail") or "")
        except ValueError:
            detail = response.text.strip()
        suffix = f": {detail}" if detail else ""
        raise TuiClientError(f"server returned {response.status_code}{suffix}")
    body = response.json()
    return body if isinstance(body, dict) else {}

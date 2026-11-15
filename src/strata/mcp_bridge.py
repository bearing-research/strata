"""``strata mcp``: a project's scratchpad as a stdio MCP server, for an agent's config.

It answers ``initialize`` and ``tools/list`` itself, from the guidance and a copy of the
daemon's tool list (``scratch_tools.json``), so a session that never calls a tool starts
nothing. The first tool call finds or starts the project's daemon (``strata.scratch``),
reporting each setup step as MCP progress, and every call is forwarded to the daemon's
``/mcp/`` endpoint. A call that had to set the scratchpad up carries ``setup_ms``.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Any

import anyio
import anyio.from_thread
import anyio.to_thread
import httpx
import mcp_types as types
import pydantic_core
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from strata import scratch
from strata.agent_guidance import section

TOOLS_FILE = Path(__file__).with_name("scratch_tools.json")
_PROTOCOL_VERSION = "2025-06-18"
_SESSION_HEADER = "mcp-session-id"


@cache
def listed_tools() -> list[dict[str, Any]]:
    """The tools the daemon lists, as ``tools/list`` gives them."""
    return json.loads(TOOLS_FILE.read_text())


class UpstreamError(RuntimeError):
    """The daemon answered a forwarded request with something other than its result."""


def _message(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """The JSON-RPC response to *request_id*, from a JSON body or an SSE stream."""
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data:"):
            message = json.loads(line[5:])
            if message.get("id") == request_id:
                return message
    raise UpstreamError(f"the scratchpad's server sent no answer: {response.text[:300]}")


class Upstream:
    """One MCP session on a daemon's ``/mcp/``, opened again when the daemon changes."""

    def __init__(self) -> None:
        # A cell may run for hours; its own timeout ends it, not this client.
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5.0))
        self._url: str | None = None
        self._session: str | None = None
        self._ids = itertools.count(1)

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": _PROTOCOL_VERSION,
        }
        if self._session is not None:
            headers[_SESSION_HEADER] = self._session
        return headers

    async def _send(self, method: str, params: dict[str, Any]) -> tuple[int, httpx.Response]:
        assert self._url is not None
        request_id = next(self._ids)
        response = await self._client.post(
            f"{self._url}/mcp/",
            headers=self._headers(),
            json={"jsonrpc": "2.0", "id": request_id, "method": method, "params": params},
        )
        return request_id, response

    async def open(self, url: str) -> None:
        if url == self._url and self._session is not None:
            return
        self._url, self._session = url, None
        _, response = await self._send(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "strata-mcp", "version": "1"},
            },
        )
        response.raise_for_status()
        self._session = response.headers.get(_SESSION_HEADER)
        await self._client.post(
            f"{url}/mcp/",
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The daemon's ``tools/call`` result, as JSON."""
        assert self._url is not None
        params = {"name": name, "arguments": arguments}
        request_id, response = await self._send("tools/call", params)
        if response.status_code == 404:
            # The daemon forgot an idle session; the call never ran.
            url, self._url = self._url, None
            await self.open(url)
            request_id, response = await self._send("tools/call", params)
        response.raise_for_status()
        message = _message(response, request_id)
        if "error" in message:
            raise UpstreamError(message["error"].get("message", str(message["error"])))
        return message["result"]


def _with_setup(result: dict[str, Any], setup_ms: dict[str, int]) -> dict[str, Any]:
    """*result* with ``setup_ms`` added to its structured value and the text showing it."""
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        return result
    structured = {**structured, "setup_ms": setup_ms}
    # As the daemon renders it, so the text and the structured value agree.
    text = pydantic_core.to_json(structured, fallback=str, indent=2).decode()
    return {**result, "structuredContent": structured, "content": [{"type": "text", "text": text}]}


def _failure(error: str, fix: str) -> types.CallToolResult:
    value = {"status": "error", "error": error, "fix": fix}
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(value, indent=2))],
        is_error=True,
    )


def build_server(root: Path) -> Server:
    """The server for the scratchpad of the project at *root*."""
    upstream = Upstream()
    # One setup at a time: parallel first calls wait for one daemon, not race two.
    setting_up = anyio.Lock()

    async def list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult.model_validate({"tools": listed_tools()})

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        steps = itertools.count(1)

        def progress(step: str) -> None:
            anyio.from_thread.run(ctx.session.report_progress, next(steps), None, step)

        async with setting_up:
            try:
                daemon = await anyio.to_thread.run_sync(scratch.ensure_daemon, root, progress)
            except scratch.ScratchError as exc:
                log = scratch.scratch_dir(root.resolve()) / ".strata" / "server.log"
                return _failure(str(exc), f"read {log} for why; the next call tries again")
        try:
            async with setting_up:
                # The daemon can stop between being found and this handshake.
                await upstream.open(daemon.url)
            result = await upstream.call_tool(params.name, dict(params.arguments or {}))
        except (httpx.HTTPError, UpstreamError) as exc:
            return _failure(
                f"the scratchpad's server failed the call: {exc}",
                "call again; a server that stopped is started again on the next call",
            )
        if daemon.setup_ms:
            result = _with_setup(result, daemon.setup_ms)
        return types.CallToolResult.model_validate(result)

    @asynccontextmanager
    async def lifespan(server: Server) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            await upstream.aclose()

    return Server(
        "strata-scratchpad",
        lifespan=lifespan,
        instructions=section("instructions"),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def serve(root: Path) -> None:
    server = build_server(root)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def mcp_main(root: Path) -> int:
    anyio.run(serve, root)
    return 0

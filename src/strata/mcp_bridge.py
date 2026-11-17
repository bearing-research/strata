"""``strata mcp``: a project's scratchpad as a stdio MCP server, for an agent's config.

It answers ``initialize`` and ``tools/list`` itself, from the guidance and a copy of the
daemon's tool list (``scratch_tools.json``), so a session that never calls a tool starts
nothing. The first tool call finds or starts the project's daemon (``strata.scratch``),
reporting each setup step as MCP progress, and every call is forwarded to the daemon's
``/mcp/`` endpoint. A call that had to set the scratchpad up carries ``setup_ms``, and
one served by another Strata version's daemon, busy so not replaced, carries ``note``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import cache
from importlib.metadata import version
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
            # Not a request of the server's own that happens to reuse the id.
            if message.get("id") == request_id and ("result" in message or "error" in message):
                return message
    raise UpstreamError(f"the scratchpad's server sent no answer: {response.text[:300]}")


class Upstream:
    """One MCP session on a daemon's ``/mcp/``, opened again when the daemon changes."""

    def __init__(self) -> None:
        # A cell may run for hours; its own timeout ends it, not this client. No kept-alive
        # connections: one to a daemon that died fails as if the call were sent, where a
        # new one finds nothing listening and the call is sent again.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=5.0),
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._url: str | None = None
        self._session: str | None = None
        self._ids = itertools.count(1)
        # One handshake at a time; a call waits out one in progress.
        self._opening = anyio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    def _headers(self, session: str | None = None) -> dict[str, str]:
        headers = {
            "accept": "application/json, text/event-stream",
            "mcp-protocol-version": _PROTOCOL_VERSION,
        }
        session = session or self._session
        if session is not None:
            headers[_SESSION_HEADER] = session
        return headers

    async def _initialize(self, url: str) -> None:
        self._url, self._session = url, None
        response = await self._client.post(
            f"{url}/mcp/",
            headers=self._headers(),
            json={
                "jsonrpc": "2.0",
                "id": next(self._ids),
                "method": "initialize",
                "params": {
                    "protocolVersion": _PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "strata-mcp", "version": "1"},
                },
            },
        )
        response.raise_for_status()
        session = response.headers.get(_SESSION_HEADER)
        await self._client.post(
            f"{url}/mcp/",
            headers=self._headers(session),
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        # Only now: a call sent under it before the handshake ends would be refused.
        self._session = session

    async def open(self, url: str) -> None:
        async with self._opening:
            if url != self._url or self._session is None:
                await self._initialize(url)

    async def _reopen(self, forgotten: str | None) -> None:
        async with self._opening:
            # A parallel call that met the same 404 may have opened one already.
            if self._session == forgotten:
                assert self._url is not None
                await self._initialize(self._url)

    async def _call(self, params: dict[str, Any]) -> tuple[str | None, int, httpx.Response]:
        async with self._opening:
            url, session = self._url, self._session
        assert url is not None
        request_id = next(self._ids)
        try:
            response = await self._client.post(
                f"{url}/mcp/",
                headers=self._headers(session),
                json={"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params},
            )
        except anyio.get_cancelled_exc_class():
            # The agent cancelled it: so does the daemon, or the cell runs on for nobody.
            with anyio.CancelScope(shield=True), contextlib.suppress(httpx.HTTPError):
                await self._client.post(
                    f"{url}/mcp/",
                    headers=self._headers(session),
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": request_id, "reason": "cancelled by the client"},
                    },
                    timeout=5.0,
                )
            raise
        return session, request_id, response

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The daemon's ``tools/call`` result, as JSON."""
        params = {"name": name, "arguments": arguments}
        session, request_id, response = await self._call(params)
        if response.status_code == 404:
            # The daemon forgot an idle session; the call never ran.
            await self._reopen(session)
            _, request_id, response = await self._call(params)
        response.raise_for_status()
        message = _message(response, request_id)
        if "error" in message:
            raise UpstreamError(message["error"].get("message", str(message["error"])))
        return message["result"]


def _with_fields(result: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """*result* with *fields* added to its structured value and the text showing it."""
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        return result
    structured = {**structured, **fields}
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

    # The daemon forwarded to: found or started on the first call, and again only once
    # a call cannot reach it, or while another version's is used until it is idle.
    found: list[scratch.Daemon] = []
    current = version("strata-notebook")

    async def connect(ctx: Any, unreachable: scratch.Daemon | None) -> dict[str, int]:
        """Connect to the daemon; return the setup steps this took, if it set one up."""
        steps = itertools.count(1)

        def progress(step: str) -> None:
            anyio.from_thread.run(ctx.session.report_progress, next(steps), None, step)

        setup_ms: dict[str, int] = {}
        async with setting_up:
            other = bool(found) and found[0] is not unreachable and found[0].version != current
            if not found or found[0] is unreachable or other:
                try:
                    daemon = await anyio.to_thread.run_sync(scratch.ensure_daemon, root, progress)
                except scratch.NotAnswering:
                    # Another version's, too busy to answer whether it is: it serves the call.
                    if not other:
                        raise
                else:
                    found[:] = [daemon]
                    setup_ms = daemon.setup_ms
        await upstream.open(found[0].url)
        return setup_ms

    async def call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
        arguments = dict(params.arguments or {})
        setup_ms: dict[str, int] = {}
        unreachable: scratch.Daemon | None = None
        try:
            for attempt in range(2):
                try:
                    setup_ms = await connect(ctx, unreachable) or setup_ms
                    result = await upstream.call_tool(params.name, arguments)
                    break
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    # Nothing listens there now (it stopped idle, or was replaced): the
                    # call never ran, so find the daemon again and send it once more.
                    # Windows may time such a connection out rather than refuse it.
                    if attempt:
                        raise
                    unreachable = found[0] if found else None
        except scratch.ScratchError as exc:
            return _failure(str(exc), "the error says what to look at; the next call tries again")
        except (httpx.HTTPError, UpstreamError) as exc:
            return _failure(
                f"the scratchpad's server failed the call: {type(exc).__name__}: {exc}",
                "call again; a server that stopped is started again on the next call",
            )
        fields: dict[str, Any] = {"setup_ms": setup_ms} if setup_ms else {}
        if found[0].version != current:
            fields["note"] = (
                f"served by the scratchpad's daemon of Strata {found[0].version}, not this "
                f"client's {current}: it was busy, so it is replaced once idle"
            )
        if fields:
            result = _with_fields(result, fields)
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

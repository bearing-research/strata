"""``strata mcp``: a project's scratchpad as a stdio MCP server, for an agent's config.

It answers ``initialize`` and ``tools/list`` itself, from the guidance and a copy of the
daemon's tool list (``scratch_tools.json``). In a project that has a scratchpad, the
session's first tool list starts the project's daemon (``strata.scratch``) in the
background, so the first call does not wait for it; elsewhere nothing starts until the
first tool call, which finds or starts the daemon. A call that sets the scratchpad up
hears each setup step as MCP progress, one that waits for the background start hears that
it waits, and both carry ``setup_ms``. Every call is forwarded to the daemon's ``/mcp/``
endpoint; one served by another Strata version's daemon, busy so not replaced, carries
``note``.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import signal
import sys
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import version
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
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


@dataclass
class _Opening:
    """The daemon's background start, which calls made meanwhile wait for."""

    # Begun once: one that fails leaves the start to the first call.
    begun: bool = False
    running: bool = False
    daemon: scratch.Daemon | None = None
    # The lifespan's task group, and the cancel its end sets (killing a daemon the start
    # began and had not recorded).
    group: anyio.abc.TaskGroup | None = None
    cancel: scratch.SetupCancel = field(default_factory=scratch.SetupCancel)


def build_server(root: Path) -> Server:
    """The server for the scratchpad of the project at *root*, for one session (as
    ``strata mcp`` serves one): its client and its start are not made again."""
    upstream = Upstream()
    # One setup at a time: parallel first calls wait for one daemon, not race two.
    setting_up = anyio.Lock()

    opening = _Opening()

    async def list_tools(
        ctx: Any, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        # An agent's session lists the tools as it opens, where a health check only
        # connects. Only where a scratchpad was made: elsewhere a session that never
        # calls a tool creates nothing.
        if (
            opening.group is not None
            and not opening.begun
            and not found
            and (scratch.scratch_dir(root) / "notebook.toml").is_file()
        ):
            opening.begun = opening.running = True
            opening.group.start_soon(prestart)
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
        # A background start is running: this call waits for it, and says so.
        waits = opening.running
        if waits:
            await ctx.session.report_progress(
                next(steps), None, "waiting for the scratchpad to start"
            )
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
            elif waits and found[0] is opening.daemon:
                # What the start it waited for took.
                setup_ms = dict(found[0].setup_ms)
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

    async def prestart() -> None:
        """Start the daemon before the first call; a call meanwhile waits for it."""
        async with setting_up:
            try:
                if found:
                    return
                try:
                    daemon = await anyio.to_thread.run_sync(
                        scratch.ensure_daemon, root, None, opening.cancel, abandon_on_cancel=True
                    )
                except Exception as exc:
                    # Never the session's end: the first call tries again, and says why.
                    print(f"strata mcp: the scratchpad did not start: {exc!r}", file=sys.stderr)
                    return
                found[:] = [daemon]
                opening.daemon = daemon
            finally:
                # Under the lock: a call that sees it running waits for this start.
                opening.running = False
        if found[0] is not daemon:
            # A call replaced it meanwhile, and opened its own.
            return
        try:
            # Its handshake too; one that fails is made again by the call.
            await upstream.open(daemon.url)
        except Exception as exc:
            print(f"strata mcp: the scratchpad's handshake failed: {exc!r}", file=sys.stderr)

    @asynccontextmanager
    async def lifespan(server: Server) -> AsyncGenerator[None]:
        own: BaseException | None = None
        try:
            try:
                async with anyio.create_task_group() as group:
                    opening.group = group
                    try:
                        yield
                    finally:
                        opening.group = None
                        opening.cancel.set()
                        group.cancel_scope.cancel()
            except BaseExceptionGroup as error:
                # The server's own error: the start raises none.
                if len(error.exceptions) != 1:
                    raise
                own = error.exceptions[0]
            if own is not None:
                # Out of the handler, so its own cause and context stay as they were.
                raise own
        finally:
            await upstream.aclose()

    return Server(
        "strata-scratchpad",
        lifespan=lifespan,
        instructions=section("instructions"),
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


def call_once(
    root: Path,
    name: str,
    arguments: dict[str, Any],
    progress: Callable[[str], None] | None = None,
) -> tuple[Any, bool]:
    """One tool call on the scratchpad of *root*, for the CLI: its value, and whether it failed.

    Finds or starts the daemon as the bridge does, and sends the call once more if the
    daemon it found stopped before the call reached it.
    """

    # Set on a SIGTERM or Ctrl-C: the setup thread, abandoned so the answer is said at
    # once, stops soon after (the exit waits for it), killing a daemon it had not recorded.
    cancel = scratch.SetupCancel()

    async def call() -> dict[str, Any]:
        upstream = Upstream()
        try:
            for attempt in range(2):
                daemon = await anyio.to_thread.run_sync(
                    scratch.ensure_daemon, root, progress, cancel, abandon_on_cancel=True
                )
                try:
                    await upstream.open(daemon.url)
                    result = await upstream.call_tool(name, arguments)
                except (httpx.ConnectError, httpx.ConnectTimeout):
                    if attempt:
                        raise
                    continue
                if daemon.setup_ms:
                    result = _with_fields(result, {"setup_ms": daemon.setup_ms})
                return result
            raise AssertionError("unreachable")
        finally:
            await upstream.aclose()

    async def cancellable() -> dict[str, Any] | None:
        """*call*, cancelled on SIGTERM as on Ctrl-C, so the daemon cancels the cell too:
        an agent's shell tool ends a command that outlives its timeout so."""
        result: dict[str, Any] | None = None
        failure: Exception | None = None
        with contextlib.ExitStack() as stack:
            # Received from before the call starts: a SIGTERM never finds the default action.
            signals = (
                stack.enter_context(anyio.open_signal_receiver(signal.SIGTERM))
                if sys.platform != "win32"
                else None
            )
            async with anyio.create_task_group() as group:

                async def run() -> None:
                    nonlocal result, failure
                    try:
                        result = await call()
                    except Exception as exc:
                        # Raised after the group, as itself: the group would wrap it in an
                        # ExceptionGroup, which the caller's handlers do not catch.
                        failure = exc
                    group.cancel_scope.cancel()

                async def on_terminate(received: AsyncIterator[int]) -> None:
                    async for _ in received:
                        cancel.set()
                        group.cancel_scope.cancel()
                        return

                group.start_soon(run)
                if signals is not None:
                    group.start_soon(on_terminate, signals)
        if failure is not None:
            raise failure
        return result

    try:
        try:
            result = anyio.run(cancellable)
        except KeyboardInterrupt:
            cancel.set()
            raise
    except scratch.ScratchError as exc:
        return {"status": "error", "error": str(exc), "fix": "the error says what to look at"}, True
    except (httpx.HTTPError, UpstreamError) as exc:
        error = f"the scratchpad's server failed the call: {type(exc).__name__}: {exc}"
        return {"status": "error", "error": error, "fix": "run the command again"}, True
    if result is None:
        # A result that arrived with the signal stands: the cell ran.
        error = "terminated: the call was cancelled, with its cell if one had started"
        return {"status": "error", "error": error, "fix": "run the command again"}, True
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured, bool(result.get("isError")) or structured.get("status") == "error"
    text = "\n".join(part.get("text", "") for part in result.get("content") or [])
    return text, bool(result.get("isError"))


async def serve(root: Path) -> None:
    server = build_server(root)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def mcp_main(root: Path) -> int:
    anyio.run(serve, root)
    return 0

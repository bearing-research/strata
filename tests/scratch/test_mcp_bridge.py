"""``strata mcp``: tools listed at once, the daemon started on the first call, calls forwarded."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys

import httpx
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from strata import mcp_bridge, scratch
from strata.agent_guidance import section
from strata.notebook.mcp_server import build_mcp_app
from strata.notebook.session import SessionManager


def test_the_listed_tools_are_the_daemons():
    app = build_mcp_app(SessionManager(), profile="scratchpad")
    assert app is not None
    tools = asyncio.run(app.state.mcp_server.list_tools())
    daemons = [t.model_dump(mode="json", by_alias=True, exclude_none=True) for t in tools]
    # When this fails, write `daemons` to the file: the bridge lists that copy.
    assert mcp_bridge.listed_tools() == daemons, mcp_bridge.TOOLS_FILE


def _bridge(project) -> Client:
    # Legacy: the initialize handshake, as Claude Code connects.
    return Client(mcp_bridge.build_server(project), mode="legacy")


def _value(result) -> dict:
    assert not result.is_error, result.content
    (text,) = result.content
    assert json.loads(text.text) == result.structured_content
    return result.structured_content


@pytest.mark.asyncio
async def test_strata_mcp_lists_its_tools_without_starting_anything(project):
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "strata.cli", "mcp", "--root", str(project)],
        env=dict(os.environ),
    )
    async with Client(server, mode="legacy") as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == ["run", "scratchpad"]
        assert client.instructions == section("instructions")
    assert not (project / ".strata-scratch").exists()


@pytest.mark.asyncio
async def test_the_first_call_sets_the_scratchpad_up_and_says_so(project):
    steps: list[str | None] = []

    async def progress(done, total, message):
        steps.append(message)

    async with _bridge(project) as client:
        first = _value(
            await client.call_tool("run", {"source": "x = 41\nx + 1"}, progress_callback=progress)
        )
        assert first["status"] == "ok"
        assert first["value"] == 42
        assert (
            list(first["setup_ms"])
            == steps
            == [
                "creating the scratchpad",
                "starting the scratchpad",
                "opening the scratchpad",
                "syncing the scratchpad's environment",
            ]
        )

        # The daemon holds the variable; a later call finds it with no setup.
        held = _value(await client.call_tool("scratchpad", {"name": "x"}))
        assert "x = 41" in held["source"]
        assert "setup_ms" not in held


@pytest.mark.asyncio
async def test_a_daemon_that_stopped_is_started_again_on_the_next_call(project):
    async with _bridge(project) as client:
        _value(await client.call_tool("run", {"source": "1"}))
        daemon = scratch.find_daemon(project.resolve())
        os.kill(daemon.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        # Gone before the next call, as after an idle exit.
        while scratch._alive(daemon.pid):
            await asyncio.sleep(0.05)

        again = _value(await client.call_tool("run", {"source": "2"}))
        assert again["status"] == "ok"
        assert "starting the scratchpad" in again["setup_ms"]
        assert "creating the scratchpad" not in again["setup_ms"]


@pytest.mark.asyncio
async def test_a_session_the_daemon_forgot_is_opened_again(project):
    daemon = scratch.ensure_daemon(project)
    upstream = mcp_bridge.Upstream()
    try:
        await upstream.open(daemon.url)
        first = await upstream.call_tool("scratchpad", {})
        # As the daemon's idle timeout does.
        httpx.delete(f"{daemon.url}/mcp/", headers=upstream._headers()).raise_for_status()

        second = await upstream.call_tool("scratchpad", {})
        assert second["structuredContent"]["cells"] == first["structuredContent"]["cells"]
    finally:
        await upstream.aclose()


@pytest.mark.asyncio
async def test_a_scratchpad_that_cannot_start_says_where_to_look(project, monkeypatch):
    monkeypatch.setattr(
        scratch, "_server_command", lambda: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    async with _bridge(project) as client:
        result = await client.call_tool("run", {"source": "1"})
    assert result.is_error
    failed = json.loads(result.content[0].text)
    assert "exited while starting" in failed["error"]
    assert "server.log" in failed["error"]


@pytest.mark.asyncio
async def test_a_daemon_gone_before_the_handshake_is_a_failure_with_its_fix(project, monkeypatch):
    async def refused(self, url):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(mcp_bridge.Upstream, "open", refused)
    async with _bridge(project) as client:
        result = await client.call_tool("scratchpad", {})
    assert result.is_error
    assert "call again" in json.loads(result.content[0].text)["fix"]


@pytest.mark.asyncio
async def test_later_calls_do_not_probe_the_daemon_again(project, monkeypatch):
    found = []
    ensure = scratch.ensure_daemon
    monkeypatch.setattr(
        scratch, "ensure_daemon", lambda *args, **kwargs: found.append(1) or ensure(*args, **kwargs)
    )
    async with _bridge(project) as client:
        for source in ("1", "2", "3"):
            _value(await client.call_tool("run", {"source": source}))
    # A busy daemon is never probed (and taken for gone) per call.
    assert found == [1]


@pytest.mark.asyncio
async def test_a_call_that_could_not_connect_is_sent_again_once(project, monkeypatch):
    call = mcp_bridge.Upstream.call_tool
    refused = []

    async def once_refused(self, name, arguments):
        if not refused:
            refused.append(name)
            raise httpx.ConnectError("refused")
        return await call(self, name, arguments)

    monkeypatch.setattr(mcp_bridge.Upstream, "call_tool", once_refused)
    async with _bridge(project) as client:
        out = _value(await client.call_tool("run", {"source": "1 + 1"}))
    assert (refused, out["value"]) == (["run"], 2)


@pytest.mark.asyncio
async def test_parallel_calls_on_a_forgotten_session_open_one_again(project):
    daemon = scratch.ensure_daemon(project)
    upstream = mcp_bridge.Upstream()
    try:
        await upstream.open(daemon.url)
        httpx.delete(f"{daemon.url}/mcp/", headers=upstream._headers()).raise_for_status()
        opened = []
        initialize = upstream._initialize

        async def counted(url):
            opened.append(url)
            await initialize(url)

        upstream._initialize = counted
        results = await asyncio.gather(*(upstream.call_tool("scratchpad", {}) for _ in range(3)))
        assert opened == [daemon.url]
        assert all("cells" in result["structuredContent"] for result in results)
    finally:
        await upstream.aclose()


def test_a_request_of_the_servers_with_the_answers_id_is_not_the_answer():
    body = (
        'data: {"jsonrpc": "2.0", "id": 1, "method": "elicitation/create", "params": {}}\n\n'
        'data: {"jsonrpc": "2.0", "id": 1, "result": {"ok": true}}\n\n'
    )
    response = httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)
    assert mcp_bridge._message(response, 1) == {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}


@pytest.mark.asyncio
async def test_a_cancelled_call_is_cancelled_on_the_daemon_too():
    seen = []
    calling = asyncio.Event()

    async def daemon(request):
        body = json.loads(request.content)
        seen.append(body)
        if body.get("method") == "tools/call":
            calling.set()
            await asyncio.sleep(3600)
        return httpx.Response(202)

    upstream = mcp_bridge.Upstream()
    await upstream.aclose()
    upstream._client = httpx.AsyncClient(transport=httpx.MockTransport(daemon))
    upstream._url, upstream._session = "http://daemon", "s"
    try:
        task = asyncio.create_task(upstream.call_tool("run", {"source": "1"}))
        await calling.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await upstream.aclose()
    (call,) = [body for body in seen if body.get("method") == "tools/call"]
    assert [body for body in seen if body.get("method") == "notifications/cancelled"] == [
        {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": call["id"], "reason": "cancelled by the client"},
        }
    ]

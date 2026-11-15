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
        os.kill(scratch.find_daemon(project.resolve()).pid, signal.SIGKILL)

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
    assert "server.log" in failed["fix"]

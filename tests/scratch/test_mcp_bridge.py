"""``strata mcp``: tools listed at once, the daemon started on the first call, calls forwarded."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import threading
import time

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


def _made(project) -> None:
    """A scratchpad made by an earlier session, its daemon since stopped."""
    scratch.ensure_daemon(project)
    assert scratch.stop_daemon(project)


@pytest.mark.asyncio
async def test_a_project_with_a_scratchpad_starts_its_daemon_as_the_session_opens(
    project, monkeypatch
):
    _made(project)
    starts = _counted(monkeypatch)
    async with _bridge(project) as client:
        # As an agent's session opens: the tool list, no call yet.
        await client.list_tools()
        async with asyncio.timeout(120):
            while scratch.find_daemon(project.resolve()) is None:
                await asyncio.sleep(0.05)
        first = _value(await client.call_tool("run", {"source": "1 + 1"}))
    assert first["value"] == 2
    # The opening's start (no progress sink), none of the call's.
    assert starts == [None]


def _counted(
    monkeypatch,
    *,
    entered: threading.Event | None = None,
    release: threading.Event | None = None,
    fails: int = 0,
) -> list:
    """Each ensure_daemon's progress sink, in order. Each one sets *entered* and waits for
    *release*, if given; the first *fails* raise."""
    starts: list[object] = []
    ensure = scratch.ensure_daemon

    def counted(*args, **kwargs):
        starts.append(args[1])
        if entered is not None:
            entered.set()
        if release is not None:
            assert release.wait(60)
        if len(starts) <= fails:
            raise PermissionError("the lock")
        return ensure(*args, **kwargs)

    monkeypatch.setattr(scratch, "ensure_daemon", counted)
    return starts


@pytest.mark.asyncio
async def test_a_call_while_the_daemon_starts_waits_for_it_and_says_so(project, monkeypatch):
    _made(project)
    entered, release = threading.Event(), threading.Event()
    starts = _counted(monkeypatch, entered=entered, release=release)
    steps: list[str | None] = []

    async def progress(done, total, message):
        steps.append(message)

    async with _bridge(project) as client:
        await client.list_tools()
        assert await asyncio.to_thread(entered.wait, 60)
        call = asyncio.create_task(
            client.call_tool("run", {"source": "1 + 1"}, progress_callback=progress)
        )
        async with asyncio.timeout(60):
            while not steps:
                await asyncio.sleep(0.01)
        release.set()
        first = _value(await call)
    assert first["value"] == 2
    assert starts == [None]
    assert steps == ["waiting for the scratchpad to start"]
    assert "starting the scratchpad" in first["setup_ms"]


@pytest.mark.asyncio
async def test_a_health_check_that_only_connects_starts_nothing(project, monkeypatch):
    _made(project)
    starts = _counted(monkeypatch)
    async with _bridge(project):
        await asyncio.sleep(0.2)
    assert starts == []


@pytest.mark.asyncio
async def test_a_project_with_no_scratchpad_starts_nothing_until_a_call(project, monkeypatch):
    starts = _counted(monkeypatch)
    async with _bridge(project) as client:
        await client.list_tools()
        await asyncio.sleep(0.2)
    assert starts == []
    assert not (project / ".strata-scratch").exists()


@pytest.mark.asyncio
async def test_a_start_that_fails_as_the_session_opens_is_left_to_the_call(project, monkeypatch):
    _made(project)
    # Not a ScratchError: still the start alone fails, not the session.
    starts = _counted(monkeypatch, fails=1)
    async with _bridge(project) as client:
        await client.list_tools()
        async with asyncio.timeout(60):
            while not starts:
                await asyncio.sleep(0.05)
        # Once a session: a later tool list does not start it again.
        await client.list_tools()
        await asyncio.sleep(0.2)
        assert len(starts) == 1
        first = _value(await client.call_tool("run", {"source": "1 + 1"}))
    assert first["value"] == 2
    assert len(starts) == 2


@pytest.mark.asyncio
async def test_a_call_after_a_failed_start_that_fails_too_says_where_to_look(project, monkeypatch):
    _made(project)
    monkeypatch.setattr(
        scratch, "_server_command", lambda: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    async with _bridge(project) as client:
        await client.list_tools()
        result = await client.call_tool("run", {"source": "1"})
    assert result.is_error
    assert "exited while starting" in json.loads(result.content[0].text)["error"]


@pytest.mark.asyncio
async def test_a_session_that_ends_during_the_start_cancels_it(project, monkeypatch):
    _made(project)
    ended = asyncio.Event()
    seen: list[bool] = []
    loop = asyncio.get_running_loop()

    def waits(root, progress, cancel):
        # As a start would: until the session's end sets the cancel (bounded, so a cancel
        # never set fails the test rather than leaving the thread spinning).
        deadline = time.monotonic() + 30
        while not cancel.is_set() and time.monotonic() < deadline:
            time.sleep(0.01)
        seen.append(cancel.is_set())
        loop.call_soon_threadsafe(ended.set)
        raise scratch.ScratchError("cancelled")

    monkeypatch.setattr(scratch, "ensure_daemon", waits)
    async with _bridge(project) as client:
        await client.list_tools()
    await asyncio.wait_for(ended.wait(), 60)
    assert seen == [True]


@pytest.mark.asyncio
async def test_the_servers_own_error_leaves_its_lifespan_as_itself(project):
    server = mcp_bridge.build_server(project)
    with pytest.raises(LookupError, match="the server's"):
        async with server.lifespan(server):
            raise LookupError("the server's")


def _refused(url: str) -> bool:
    """Whether a connection to *url* is refused: nothing listens there."""
    import socket
    from urllib.parse import urlsplit

    address = urlsplit(url)
    try:
        socket.create_connection((address.hostname, address.port), timeout=5).close()
    except ConnectionRefusedError:
        return True
    except OSError:
        return False
    return False


@pytest.mark.asyncio
async def test_a_daemon_that_stopped_is_started_again_on_the_next_call(project):
    async with _bridge(project) as client:
        _value(await client.call_tool("run", {"source": "1"}))
        daemon = scratch.find_daemon(project.resolve())
        os.kill(daemon.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        # Gone before the next call, as after an idle exit: its port refuses. Not its pid's
        # exit: on Windows that is the venv's launcher, and the server it started dies after.
        while not await asyncio.to_thread(_refused, daemon.url):
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


def _another_version(project) -> None:
    record_path = project / ".strata-scratch" / ".strata" / "server.json"
    record = json.loads(record_path.read_text())
    record["version"] = "0.0.0"
    record_path.write_text(json.dumps(record))


@pytest.mark.asyncio
async def test_another_versions_busy_daemon_serves_the_call_until_it_is_idle(project, monkeypatch):
    first = scratch.ensure_daemon(project)
    _another_version(project)
    monkeypatch.setattr(scratch, "_busy", lambda url: True)
    async with _bridge(project) as client:
        served = _value(await client.call_tool("run", {"source": "1"}))
        assert served["status"] == "ok"
        assert "Strata 0.0.0" in served["note"]
        assert scratch._healthy(first.url)

        # Asked again on the next call: idle now, so replaced.
        monkeypatch.setattr(scratch, "_busy", lambda url: False)
        again = _value(await client.call_tool("run", {"source": "2"}))
        assert again["status"] == "ok"
        assert "note" not in again
        assert "starting the scratchpad" in again["setup_ms"]
        assert not scratch._healthy(first.url)


@pytest.mark.asyncio
async def test_another_versions_daemon_too_busy_to_answer_still_serves_the_call(
    project, monkeypatch
):
    scratch.ensure_daemon(project)
    _another_version(project)
    monkeypatch.setattr(scratch, "_busy", lambda url: True)
    async with _bridge(project) as client:
        _value(await client.call_tool("run", {"source": "1"}))

        def not_answering(*args, **kwargs):
            raise scratch.NotAnswering("is not answering; try again shortly")

        monkeypatch.setattr(scratch, "ensure_daemon", not_answering)
        served = _value(await client.call_tool("run", {"source": "2"}))
        assert served["status"] == "ok"
        assert "Strata 0.0.0" in served["note"]

        # Any other failure, such as a replacement that would not start, is the call's.
        def failed(*args, **kwargs):
            raise scratch.ScratchError("the scratchpad's server did not open it")

        monkeypatch.setattr(scratch, "ensure_daemon", failed)
        result = await client.call_tool("run", {"source": "3"})
        assert result.is_error
        assert "did not open it" in result.content[0].text


@pytest.mark.asyncio
# Windows may time a connection to a port nobody holds out instead of refusing it.
@pytest.mark.parametrize("unconnected", [httpx.ConnectError, httpx.ConnectTimeout])
async def test_a_call_that_could_not_connect_is_sent_again_once(project, monkeypatch, unconnected):
    call = mcp_bridge.Upstream.call_tool
    refused = []

    async def once_refused(self, name, arguments):
        if not refused:
            refused.append(name)
            raise unconnected("refused")
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


@pytest.mark.asyncio
async def test_each_call_opens_its_own_connection():
    """None is kept alive: reused after the daemon died, one fails as if the call were sent."""
    connections = 0

    async def serve(reader, writer):
        nonlocal connections
        connections += 1
        while head := await reader.readuntil(b"\r\n\r\n"):
            length = next(
                int(line.split(b":")[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            request = json.loads(await reader.readexactly(length))
            body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {}}).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n"
                b"content-length: %d\r\n\r\n%s" % (len(body), body)
            )
            await writer.drain()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    upstream = mcp_bridge.Upstream()
    upstream._url, upstream._session = f"http://127.0.0.1:{port}", "s"
    try:
        await upstream.call_tool("scratchpad", {})
        await upstream.call_tool("scratchpad", {})
    finally:
        await upstream.aclose()
        server.close()
    assert connections == 2

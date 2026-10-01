"""MCP on a server that authenticates its callers.

Each tool call runs as the principal its HTTP request names, checked against the
same notebook scopes as REST and WebSocket. Driven with the real MCP client over
streamable HTTP so identity is read from each call's own request.
"""

from __future__ import annotations

import json
import threading

import pytest

pytest.importorskip("mcp")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402

from tests.conftest import find_free_port, wait_for_server  # noqa: E402
from tests.notebook.test_cli import _build_notebook  # noqa: E402

TOKEN = "sekrit"


def _headers(principal: str, scopes: str, token: str = TOKEN) -> dict[str, str]:
    return {
        "X-Strata-Proxy-Token": token,
        "X-Strata-Principal": principal,
        "X-Strata-Scopes": scopes,
    }


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A service-mode, trusted-proxy server with one open notebook session."""
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.notebook.mcp_server import build_mcp_app
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession, SessionManager
    from strata.server import ServerState

    config = StrataConfig(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token=TOKEN,
        mcp_enabled=True,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    monkeypatch.setattr(server_module, "_state", ServerState(config))

    nb_dir = _build_notebook(tmp_path, cells=[("a", "x = 1", None)])
    sessions = SessionManager()
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    sessions._sessions[session.id] = session

    mcp_app = build_mcp_app(sessions)
    app = Starlette(
        routes=[
            Route("/health", lambda request: PlainTextResponse("ok")),
            Mount("/mcp", app=mcp_app),
        ],
        lifespan=mcp_app.router.lifespan_context,
    )
    port = find_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_for_server(port, thread=thread)
    try:
        yield f"http://127.0.0.1:{port}/mcp", session
    finally:
        server.should_exit = True
        thread.join(timeout=5)


async def _call(url: str, headers: dict[str, str], tool: str, arguments: dict):
    async with (
        httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as http,
        streamable_http_client(url, http_client=http) as (read, write),
        ClientSession(read, write) as client,
    ):
        await client.initialize()
        return await client.call_tool(tool, arguments)


def _text(result) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


async def test_a_viewer_reads_but_cannot_run_a_cell(served):
    url, session = served
    viewer = _headers("vera", "notebook:read")

    listed = await _call(url, viewer, "list_notebooks", {})
    refused = await _call(url, viewer, "run_cell", {"session_id": session.id, "cell_id": "a"})

    assert not listed.is_error
    assert session.id in _text(listed)
    assert refused.is_error
    assert "notebook:execute" in _text(refused)


async def test_a_writer_authors_as_themselves(served):
    """Authorship records the calling principal, not what the client declared."""
    url, session = served
    writer = _headers("wes", "notebook:read notebook:write")

    added = await _call(
        url, writer, "add_cell", {"session_id": session.id, "source": "y = 2", "author": "someone"}
    )

    assert not added.is_error, _text(added)
    assert json.loads(_text(added))["created_by"] == "wes"


async def test_two_callers_on_one_server_each_act_as_themselves(served):
    url, session = served

    ana = await _call(
        url,
        _headers("ana", "notebook:read notebook:write"),
        "add_cell",
        {"session_id": session.id, "source": "p = 1"},
    )
    ben = await _call(
        url,
        _headers("ben", "notebook:read notebook:write"),
        "add_cell",
        {"session_id": session.id, "source": "q = 1"},
    )

    assert json.loads(_text(ana))["created_by"] == "ana"
    assert json.loads(_text(ben))["created_by"] == "ben"


async def test_a_call_without_valid_credentials_is_refused(served):
    url, _ = served

    forged = await _call(url, _headers("mallory", "admin:*", token="wrong"), "list_notebooks", {})

    assert forged.is_error
    assert "notebook:read" in _text(forged)


def test_every_tool_is_classified():
    """A new tool must be classified explicitly rather than default to notebook:execute."""
    import asyncio

    from strata.notebook.mcp_server import build_mcp_app
    from strata.notebook.scopes import CLASSIFIED_TOOLS
    from strata.notebook.session import SessionManager

    app = build_mcp_app(SessionManager())
    names = {tool.name for tool in asyncio.run(app.state.mcp_server.list_tools())}

    assert len(names) > 20
    assert names - CLASSIFIED_TOOLS == set()


def test_service_mode_accepts_mcp_only_with_principal_auth():
    from pydantic import ValidationError

    from strata.config import StrataConfig

    StrataConfig(
        deployment_mode="service", auth_mode="trusted_proxy", proxy_token=TOKEN, mcp_enabled=True
    )
    with pytest.raises(ValidationError, match="mcp_enabled"):
        StrataConfig(deployment_mode="service", auth_mode="none", mcp_enabled=True)

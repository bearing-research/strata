"""``/mcp`` answers to the hosts the server answers to, guarded by the server's own checks.

The SDK's DNS-rebinding check is off; the server's Host allowlist and origin guard cover
``/mcp`` as they cover every route, so a client on the private network reaches it by the
server's name or address.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("mcp")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from mcp.types import LATEST_PROTOCOL_VERSION  # noqa: E402

from tests.conftest import find_free_port, wait_for_server  # noqa: E402

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"},
    },
}


@pytest.fixture
def served(tmp_path, monkeypatch):
    """The real server app with MCP mounted as in production, naming one extra host."""
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.notebook.mcp_server import build_mcp_app
    from strata.notebook.session import SessionManager
    from strata.server import ServerState, _mount_mcp, app

    config = StrataConfig(
        deployment_mode="personal",
        allowed_hosts=["strata-box"],
        mcp_enabled=True,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr(server_module, "_state", ServerState(config))

    mcp_app = build_mcp_app(SessionManager())
    original_routes = list(app.router.routes)
    _mount_mcp(app, mcp_app)
    # Production mounts MCP at import, ahead of the SPA catch-all.
    added = app.router.routes[len(original_routes) :]
    app.router.routes[:] = added + original_routes

    async def server_app(scope, receive, send):
        # Only the MCP lifespan: the requests under test need no other startup state.
        target = mcp_app if scope["type"] == "lifespan" else app
        await target(scope, receive, send)

    port = find_free_port()
    server = uvicorn.Server(
        uvicorn.Config(server_app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        assert wait_for_server(port, thread=thread)
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        app.router.routes[:] = original_routes


def _initialize(port: int, path: str, host: str, origin: str | None = None) -> httpx.Response:
    headers = {"Host": host, "Accept": "application/json, text/event-stream"}
    if origin is not None:
        headers["Origin"] = origin
    return httpx.post(
        f"http://127.0.0.1:{port}{path}", json=INITIALIZE, headers=headers, timeout=30
    )


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
@pytest.mark.parametrize("host", ["strata-box", "10.0.0.5", "localhost"])
def test_a_host_the_server_answers_to_reaches_mcp(served, path, host):
    response = _initialize(served, path, f"{host}:8765")

    assert response.status_code == 200, response.text
    assert "serverInfo" in response.text


def test_a_host_the_server_does_not_answer_to_is_refused(served):
    response = _initialize(served, "/mcp", "evil.example:8765")

    # The server's Host allowlist, not the SDK's 421.
    assert response.status_code == 400
    assert "STRATA_ALLOWED_HOSTS" in response.text


def test_a_same_origin_page_may_call_mcp(served):
    response = _initialize(served, "/mcp", "strata-box:8765", origin="http://strata-box:8765")

    assert response.status_code == 200, response.text


def test_a_page_on_another_origin_is_refused(served):
    response = _initialize(served, "/mcp", "strata-box:8765", origin="http://evil.example")

    assert response.status_code == 403
    assert "Cross-origin request" in response.text

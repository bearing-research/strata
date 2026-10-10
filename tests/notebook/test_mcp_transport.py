"""The MCP transport answers to the hosts the server answers to.

The SDK's own Host and Origin check follows ``STRATA_ALLOWED_HOSTS`` and the server's
host rules, so a client on the private network reaches ``/mcp`` by the server's name.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("mcp")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from mcp.types import LATEST_PROTOCOL_VERSION  # noqa: E402
from starlette.responses import PlainTextResponse  # noqa: E402

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
    """A personal server's MCP endpoint, mounted as the server mounts it, naming one host."""
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.notebook.mcp_server import build_mcp_app
    from strata.notebook.session import SessionManager
    from strata.server import ServerState, _mount_mcp

    config = StrataConfig(
        allowed_hosts=["strata-box"],
        mcp_enabled=True,
        artifact_dir=str(tmp_path / "artifacts"),
    )
    monkeypatch.setattr(server_module, "_state", ServerState(config))

    mcp_app = build_mcp_app(SessionManager())
    app = FastAPI(lifespan=lambda _app: mcp_app.router.lifespan_context(mcp_app))
    app.add_api_route("/health", lambda: PlainTextResponse("ok"))
    _mount_mcp(app, mcp_app)
    port = find_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_for_server(port, thread=thread)
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


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

    assert response.status_code == 421


def test_a_page_on_an_allowed_host_may_call_mcp(served):
    response = _initialize(served, "/mcp", "strata-box:8765", origin="http://strata-box:8765")

    assert response.status_code == 200, response.text


def test_a_page_on_another_host_is_refused(served):
    response = _initialize(served, "/mcp", "strata-box:8765", origin="http://evil.example")

    assert response.status_code == 403

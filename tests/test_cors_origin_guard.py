"""A web page must not be able to drive the loopback API.

Personal mode has no auth, so a cross-origin page that can add and execute a cell is remote code
execution. ``POST .../execute`` has no body, which makes it a CORS simple request with no preflight,
so a disallowed ``Origin`` must also fail closed on unsafe methods.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.config import StrataConfig
from strata.server import ServerState, app

EVIL = "https://evil.example"

# Every mutating notebook route the drive-by chain walked.
ATTACK_ROUTES = [
    "/v1/notebooks/create",
    "/v1/notebooks/open",
    "/v1/notebooks/sess-1/cells",
    "/v1/notebooks/sess-1/cells/abc/execute",
]


@pytest.fixture
def client(tmp_path):
    def _make(**overrides):
        config = StrataConfig(
            cache_dir=Path(tempfile.mkdtemp(dir=tmp_path)),
            deployment_mode="personal",
            **overrides,
        )
        server_module._state = ServerState(config)
        return TestClient(app)

    return _make


class TestACrossOriginPageCannotDriveTheApi:
    @pytest.mark.parametrize("route", ATTACK_ROUTES)
    def test_preflight_from_another_site_is_refused(self, client, route):
        c = client()
        resp = c.options(
            route,
            headers={
                "Origin": EVIL,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 403
        assert "access-control-allow-origin" not in resp.headers

    def test_a_simple_request_that_needs_no_preflight_is_refused(self, client):
        """The execute route takes no body, so CORS alone could never stop it."""
        c = client()
        resp = c.post(
            "/v1/notebooks/sess-1/cells/abc/execute",
            headers={"Origin": EVIL, "Content-Type": "text/plain;charset=UTF-8"},
        )
        assert resp.status_code == 403

    def test_a_cross_origin_read_is_not_made_readable(self, client):
        """GET goes through, but without the header the browser hides the body."""
        c = client()
        resp = c.get("/v1/notebooks/discover", headers={"Origin": EVIL})
        assert "access-control-allow-origin" not in resp.headers


class TestLegitimateCallersAreUnaffected:
    def test_a_caller_with_no_origin_header_is_untouched(self, client):
        """The CLI, the MCP server and the SDK never send Origin."""
        c = client()
        assert c.get("/health").status_code == 200

    def test_the_bundled_frontend_is_same_origin(self, client):
        c = client()
        resp = c.get("/health", headers={"Origin": "http://testserver"})
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://testserver"

    def test_a_same_origin_preflight_succeeds(self, client):
        c = client()
        resp = c.options(
            "/v1/notebooks/open",
            headers={
                "Origin": "http://testserver",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://testserver"

    def test_a_configured_dev_origin_is_allowed(self, client):
        """`npm run dev` serves the UI from another port via VITE_STRATA_URL."""
        c = client(cors_allow_origins=["http://localhost:5173"])
        resp = c.options(
            "/v1/notebooks/open",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == "http://localhost:5173"

    def test_configuring_a_dev_origin_does_not_admit_anyone_else(self, client):
        c = client(cors_allow_origins=["http://localhost:5173"])
        resp = c.post(
            "/v1/notebooks/sess-1/cells/abc/execute",
            headers={"Origin": EVIL},
        )
        assert resp.status_code == 403


async def _ws_upgrade(origin: str | None) -> list[dict]:
    """Drive a WS upgrade through the real app (no TestClient portal) and return what it sent."""
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    headers = [(b"host", b"127.0.0.1:8765")]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/v1/notebooks/ws/no-such-session",
        "raw_path": b"/v1/notebooks/ws/no-such-session",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
        "subprotocols": [],
    }
    await app(scope, receive, send)
    return sent


def _reached_the_session_lookup(sent: list[dict]) -> bool:
    # A guessed session id gets this reason; the origin check closes before the lookup.
    return any(m.get("reason") == "Notebook not found" for m in sent)


class TestTheNotebookWebSocketChecksOrigin:
    """Browsers apply no CORS to WebSockets, so a cross-site page can open one to loopback."""

    @pytest.mark.asyncio
    async def test_a_foreign_origin_is_closed_before_the_session_lookup(self, client):
        client()
        sent = await _ws_upgrade(EVIL)
        assert sent[0]["type"] == "websocket.close"
        assert sent[0]["code"] == 1008
        assert not _reached_the_session_lookup(sent)

    @pytest.mark.asyncio
    async def test_same_origin_is_allowed(self, client):
        client()
        assert _reached_the_session_lookup(await _ws_upgrade("http://127.0.0.1:8765"))

    @pytest.mark.asyncio
    async def test_no_origin_is_allowed(self, client):
        """The TUI and SDK clients send no Origin."""
        client()
        assert _reached_the_session_lookup(await _ws_upgrade(None))

    @pytest.mark.asyncio
    async def test_a_configured_dev_origin_is_allowed(self, client):
        client(cors_allow_origins=["http://localhost:5173"])
        assert _reached_the_session_lookup(await _ws_upgrade("http://localhost:5173"))

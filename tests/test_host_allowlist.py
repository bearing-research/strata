"""A DNS-rebinding page must not reach a personal-mode server.

The page's own name resolves to 127.0.0.1, so its requests carry ``Host`` and ``Origin`` for that
name: same-origin to the browser, and same-origin to the origin guard. Only the Host check stops
it, and it must cover WebSocket upgrades, which no ``@app.middleware("http")`` sees.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.config import StrataConfig
from strata.server import ServerState, app

EVIL = "evil.example:8765"


@pytest.fixture
def state(tmp_path):
    def _make(**overrides):
        overrides.setdefault("allowed_hosts", [])
        overrides.setdefault("deployment_mode", "personal")
        if overrides["deployment_mode"] == "service":
            overrides.setdefault("artifact_dir", tmp_path / "artifacts")
        config = StrataConfig(cache_dir=Path(tempfile.mkdtemp(dir=tmp_path)), **overrides)
        server_module._state = ServerState(config)

    return _make


def _client(host: str) -> TestClient:
    return TestClient(app, base_url=f"http://{host}")


async def _ws_upgrade(host: str) -> list[dict]:
    """Drive a WS upgrade through the real app (no TestClient portal) and return what it sent."""
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0"},
        "scheme": "ws",
        "path": "/v1/notebooks/ws/no-such-session",
        "raw_path": b"/v1/notebooks/ws/no-such-session",
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", host.encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
        "subprotocols": [],
    }
    await app(scope, receive, send)
    return sent


def _reached_the_route(sent: list[dict]) -> bool:
    # The route closes an unknown session with this reason; the Host check closes first.
    return any(m.get("reason") == "Notebook not found" for m in sent)


class TestARebindingPageIsRefused:
    def test_the_rebinding_attack_chain_is_refused(self, state):
        """Host and Origin agree, so the origin guard alone lets this through."""
        state()
        resp = _client(EVIL).post(
            "/v1/notebooks/create",
            headers={"Origin": f"http://{EVIL}"},
            json={"parent_path": "/tmp", "name": "x"},
        )
        assert resp.status_code == 400
        assert "not allowed" in resp.text

    def test_the_refusal_names_the_setting_that_admits_a_host(self, state):
        # An operator reaching a personal server by a LAN name meets this first.
        state()
        resp = _client(EVIL).get("/health")
        assert resp.status_code == 400
        assert "STRATA_ALLOWED_HOSTS" in resp.text

    def test_a_safe_read_is_refused_too(self, state):
        """A rebound page can read same-origin responses, so GETs leak state."""
        state()
        assert _client(EVIL).get("/v1/notebooks/sessions").status_code == 400

    @pytest.mark.asyncio
    async def test_a_websocket_upgrade_is_closed(self, state):
        state()
        sent = await _ws_upgrade(EVIL)
        assert sent[0]["type"] == "websocket.close"
        assert sent[0]["code"] == 1008
        assert not _reached_the_route(sent)

    def test_a_lookalike_suffix_is_not_a_configured_host(self, state):
        state(allowed_hosts=[".example.com"])
        assert _client("evil-example.com").get("/health").status_code == 400


class TestLegitimateHostsAreAllowed:
    @pytest.mark.parametrize(
        "host", ["localhost:8765", "127.0.0.1:8765", "[::1]:8765", "LOCALHOST", "localhost:5173"]
    )
    def test_loopback_names_on_any_port(self, state, host):
        state()
        assert _client(host).get("/health").status_code == 200

    def test_an_ip_literal_cannot_be_rebound(self, state):
        """Docker on a LAN address or a probe hitting the pod IP."""
        state()
        assert _client("192.168.1.5:8765").get("/health").status_code == 200

    @pytest.mark.parametrize("host", ["strata.example.com", "nb.team.example.com:443"])
    def test_a_configured_host(self, state, host):
        state(allowed_hosts=["strata.example.com", ".team.example.com"])
        assert _client(host).get("/health").status_code == 200

    def test_the_bind_host(self, state):
        state(host="devbox.lan", allow_remote_clients_in_personal=True)
        assert _client("devbox.lan:8765").get("/health").status_code == 200

    def test_a_wildcard_turns_the_check_off(self, state):
        state(allowed_hosts=["*"])
        assert _client(EVIL).get("/health").status_code == 200

    def test_the_setting_reads_a_comma_separated_env_value(self, monkeypatch):
        monkeypatch.setenv("STRATA_ALLOWED_HOSTS", "a.example.com, .b.example.com")
        assert StrataConfig().allowed_hosts == ["a.example.com", ".b.example.com"]

    @pytest.mark.asyncio
    async def test_a_loopback_websocket_upgrade_reaches_the_route(self, state):
        state()
        assert _reached_the_route(await _ws_upgrade("127.0.0.1:8765"))


class TestServiceMode:
    def test_unconfigured_service_mode_does_not_check(self, state):
        """It sits behind a proxy that may present any public name."""
        state(deployment_mode="service", auth_mode="trusted_proxy", proxy_token="t")
        assert _client(EVIL).get("/health").status_code == 200

    def test_configured_service_mode_checks(self, state):
        state(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="t",
            allowed_hosts=["strata.example.com"],
        )
        assert _client(EVIL).get("/health").status_code == 400
        assert _client("strata.example.com").get("/health").status_code == 200

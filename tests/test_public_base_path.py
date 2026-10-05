"""A server behind a reverse proxy at a non-root path (``public_base_path``).

A proxy may strip the prefix or pass it through; both must reach the same routes, and every URL
the server hands out must carry the prefix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.config import StrataConfig
from strata.server import ServerState, _mount_frontend, app
from tests.test_publications import _ready_artifact

BASE = "/o/acme/lab"
INDEX = '<!doctype html><html><head><script src="./assets/app.js"></script></head></html>'
SCRIPT = "console.log('strata')"
# Stripped by the proxy, or passed through as the request path.
PROXY_STYLES = pytest.mark.parametrize("prefix", ["", BASE], ids=["stripped", "passed-through"])


def _config(tmp_path: Path, **overrides) -> StrataConfig:
    return StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        **overrides,
    )


class TestTheSetting:
    @pytest.mark.parametrize(
        ("value", "stored"),
        [
            ("/o/acme/lab", BASE),
            ("o/acme/lab/", BASE),
            (" /o/acme/lab/ ", BASE),
            ("/", ""),
            ("", ""),
        ],
    )
    def test_it_is_normalized(self, value, stored):
        assert StrataConfig(public_base_path=value).public_base_path == stored

    @pytest.mark.parametrize("value", ["/o/acme?x=1", "/o/#lab", "/o/a b", "/o/%2e"])
    def test_anything_but_a_plain_path_is_refused(self, value):
        with pytest.raises(ValueError, match="public_base_path"):
            StrataConfig(public_base_path=value)


@pytest.fixture
def served(tmp_path: Path) -> Iterator[Callable[..., TestClient]]:
    """Install a ServerState and a throwaway dist; returns ``make(**config) -> TestClient``."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX)
    (dist / "assets" / "app.js").write_text(SCRIPT)

    original_state = server_module._state
    original_routes = list(app.router.routes)
    # A built frontend may have been mounted at import; its catch-all would answer first.
    app.router.routes[:] = [
        route
        for route in original_routes
        if getattr(route, "name", None) not in ("frontend-assets", "spa_fallback")
    ]
    _mount_frontend(app, dist)

    def make(**overrides) -> TestClient:
        server_module._state = ServerState(_config(tmp_path, **overrides))
        return TestClient(app)

    try:
        yield make
    finally:
        app.router.routes[:] = original_routes
        server_module._state = original_state


class TestTheBundledUi:
    @PROXY_STYLES
    def test_index_tells_the_ui_its_base_path(self, served, prefix):
        client = served(public_base_path=BASE)

        response = client.get(f"{prefix}/")

        assert response.status_code == 200
        assert f'<meta name="strata-base-path" content="{BASE}">' in response.text
        assert 'src="./assets/app.js"' in response.text
        assert response.headers["cache-control"] == "no-cache"

    @PROXY_STYLES
    def test_its_relative_assets_resolve(self, served, prefix):
        client = served(public_base_path=BASE)

        assert client.get(f"{prefix}/assets/app.js").text == SCRIPT

    @PROXY_STYLES
    def test_the_api_answers(self, served, prefix):
        client = served(public_base_path=BASE)

        # JSON, not the SPA catch-all's index.html, which answers any unrouted GET with 200.
        assert client.get(f"{prefix}/health").json()["status"] == "ok"
        assert client.get(f"{prefix}/v1/notebooks/sessions").json() == {"sessions": []}

    def test_without_a_base_path_index_is_served_as_built(self, served):
        assert served().get("/").text == INDEX

    def test_the_api_docs_load_the_schema_under_the_base_path(self, served):
        client = served(public_base_path=BASE)

        assert f"{BASE}/openapi.json" in client.get(f"{BASE}/docs").text
        assert client.get(f"{BASE}/openapi.json").json()["servers"] == [{"url": BASE}]


async def _ws_upgrade(path: str) -> list[dict]:
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
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1:8765")],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
        "subprotocols": [],
    }
    await app(scope, receive, send)
    return sent


@pytest.mark.asyncio
@PROXY_STYLES
async def test_the_notebook_websocket_is_reached(tmp_path, prefix):
    original_state = server_module._state
    server_module._state = ServerState(_config(tmp_path, public_base_path=BASE))
    try:
        sent = await _ws_upgrade(f"{prefix}/v1/notebooks/ws/no-such-session")
    finally:
        server_module._state = original_state

    # The route closes an unknown session with this reason; an unrouted upgrade closes without it.
    assert any(m.get("reason") == "Notebook not found" for m in sent)


@pytest.fixture(params=[None, "https://app.example.com"], ids=["request-origin", "public-origin"])
def published(request, tmp_path):
    """A running server under BASE with one published figure; yields ``(origin, public, token)``.

    ``public`` is the origin and base path the server's links must start with.
    """
    from strata.artifact_store import ArtifactStore
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(
        tmp_path / "cache",
        artifact_dir,
        "personal",
        public_base_path=BASE,
        public_base_url=request.param,
    ) as ctx:
        version = _ready_artifact(ArtifactStore(artifact_dir), "fig", b"\x89PNG\r\n\x1a\n bytes")
        response = httpx.post(
            f"{ctx.base_url}{BASE}/v1/artifacts/fig/v/{version}/publish",
            json={"title": "Figure 3"},
            timeout=10,
        )
        response.raise_for_status()
        yield ctx.base_url, f"{request.param or ctx.base_url}{BASE}", response.json()["token"]


class TestPublicationsUnderABasePath:
    @PROXY_STYLES
    def test_the_page_links_carry_the_base_path(self, published, prefix):
        origin, public, token = published

        page = httpx.get(f"{origin}{prefix}/p/{token}", timeout=10)

        assert page.status_code == 200
        assert f"href='{public}/p/{token}/data'" in page.text
        assert f"href='{public}/p/{token}/verify'" in page.text
        assert f"{public}/p/{token}/badge.svg" in page.text
        assert f"{public}/p/{token}/embed" in page.text
        assert f"{public}/oembed?url=" in page.text

    @PROXY_STYLES
    def test_the_bytes_and_card_are_served(self, published, prefix):
        origin, public, token = published

        assert httpx.get(f"{origin}{prefix}/p/{token}/data", timeout=10).status_code == 200
        card = httpx.get(f"{origin}{prefix}/p/{token}/embed", timeout=10)
        assert f"{public}/p/{token}" in card.text

    def test_oembed_unfurls_a_link_under_the_base_path(self, published):
        origin, public, token = published

        response = httpx.get(
            f"{origin}{BASE}/oembed", params={"url": f"{public}/p/{token}"}, timeout=10
        )

        assert response.status_code == 200
        assert f'src="{public}/p/{token}/embed"' in response.json()["html"]

    def test_oembed_refuses_a_link_outside_the_base_path(self, published):
        origin, public, token = published
        elsewhere = public.removesuffix(BASE)

        response = httpx.get(
            f"{origin}{BASE}/oembed", params={"url": f"{elsewhere}/p/{token}"}, timeout=10
        )

        assert response.status_code == 404


class TestClientsKeepThePath:
    """Clients are handed the full URL; none may resolve a route against the origin alone."""

    def test_the_tui_websocket_url(self):
        from strata.notebook.tui.client import TuiClient

        client = TuiClient(f"https://app.example.com{BASE}/")

        assert client.ws_url("s1") == f"wss://app.example.com{BASE}/v1/notebooks/ws/s1"

    def test_the_sdk_client(self):
        from strata_client.client import StrataClient

        seen: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"status": "ok"})

        client = StrataClient(base_url=f"https://app.example.com{BASE}")
        # The real constructor's URL; only the wire is swapped out.
        client._client = httpx.Client(
            transport=httpx.MockTransport(answer), base_url=client._client.base_url
        )

        assert client.health() == {"status": "ok"}
        assert seen == [f"https://app.example.com{BASE}/health"]

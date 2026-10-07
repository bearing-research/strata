"""The bundled frontend: the SPA catch-all and the headers its responses carry."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.config import StrataConfig
from strata.server import ServerState, _mount_frontend, app

INDEX = "<!doctype html><title>strata-index</title>"


@pytest.fixture
def spa(tmp_path: Path) -> Iterator[tuple[TestClient, Path]]:
    """The real app with a throwaway dist mounted; yields ``(client, outside_file)``."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX)
    (dist / "favicon.svg").write_text("<svg/>")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside the dist")

    config = StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    original_state = server_module._state
    original_routes = list(app.router.routes)
    server_module._state = ServerState(config)
    # A built frontend (src/strata/_frontend or frontend/dist) was mounted at import;
    # its catch-all would answer before this dist's.
    app.router.routes[:] = [
        route
        for route in original_routes
        if getattr(route, "name", None) not in ("frontend-assets", "spa_fallback")
    ]
    _mount_frontend(app, dist)
    try:
        # No ``with``: lifespan never runs; the routes under test need no lifespan state.
        yield TestClient(app), secret
    finally:
        app.router.routes[:] = original_routes
        server_module._state = original_state


@pytest.mark.parametrize(
    "template",
    [
        "/{abs}",  # "//etc/passwd"
        "/{abs_enc}",  # "/%2fetc%2fpasswd"
        "/{up}{abs}",
        "/{up_enc}{abs_enc}",
    ],
)
def test_a_file_outside_the_dist_is_never_served(spa, template):
    client, secret = spa
    absolute = secret.as_posix()
    # httpx drops literal ``..`` segments; ``%2e%2e`` reaches the app as ``..``.
    up = "/".join(["%2e%2e"] * (len(secret.parts) + 2))
    path = template.format(
        abs=absolute,
        abs_enc=absolute.replace("/", "%2f"),
        up=up,
        up_enc=up.replace("/", "%2f"),
    )

    # Absolute URL: httpx reads a bare "//x/..." as a host, not a path.
    response = client.get("http://testserver" + path)

    assert response.text == INDEX


def test_a_file_inside_the_dist_is_served(spa):
    client, _ = spa
    assert client.get("/favicon.svg").text == "<svg/>"
    assert client.get("/notebook/anything").text == INDEX


@pytest.mark.parametrize("path", ["/p/x/embed/", "/p//x/embed", "/p/x/embed%2f"])
def test_only_the_embed_route_itself_may_be_framed_anywhere(spa, path):
    """These miss the embed route and fall to index.html, the live app behind its hash route."""
    client, _ = spa

    response = client.get("http://testserver" + path)

    assert response.text == INDEX
    assert response.headers["content-security-policy"] == "frame-ancestors 'self'"


@pytest.mark.parametrize("path", ["/", "/index.html", "/notebook/anything"])
def test_index_is_revalidated(spa, path):
    """A cached index.html outlives an upgrade and names hashed assets that no longer exist."""
    client, _ = spa

    response = client.get(path)

    assert response.text == INDEX
    assert response.headers["cache-control"] == "no-cache"


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_mcp_without_the_trailing_slash_reaches_the_mcp_app(spa, method):
    """The documented URL is ``/mcp``; the mount alone answers only ``/mcp/``."""
    from starlette.applications import Starlette
    from starlette.responses import PlainTextResponse
    from starlette.routing import Route

    from strata.server import _mount_mcp

    async def endpoint(request):
        return PlainTextResponse(f"mcp {request.method} {request.url.path}")

    fake_mcp = Starlette(routes=[Route("/", endpoint, methods=["GET", "POST", "DELETE"])])
    client, _ = spa
    # Production order: the MCP routes are added at import, before the SPA catch-all.
    routes = app.router.routes
    spa_route = next(r for r in routes if getattr(r, "name", None) == "spa_fallback")
    routes.remove(spa_route)
    _mount_mcp(app, fake_mcp)
    routes.append(spa_route)

    for path in ("/mcp", "/mcp/"):
        response = client.request(method, path, follow_redirects=False)
        assert response.status_code == 200, (path, response.status_code)
        assert response.text == f"mcp {method} /mcp/"

"""The frame-ancestors middleware controls who may embed the app view.

Strata sends no framing header by default historically; the embed feature adds
``Content-Security-Policy: frame-ancestors`` so an operator opts specific host
origins into iframing a notebook's app view (and, as a side effect, closes the
gap where any page could silently frame Strata).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.config import StrataConfig
from strata.server import ServerState, app


def _client(tmp_path, **overrides) -> Iterator[TestClient]:
    config = StrataConfig(
        host="127.0.0.1",
        port=8765,
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        **overrides,
    )
    original = server_module._state
    server_module._state = ServerState(config)
    try:
        # No ``with`` — lifespan never runs; the middleware reads _state directly.
        yield TestClient(app)
    finally:
        server_module._state = original


def _csp(tmp_path, **overrides) -> str:
    for client in _client(tmp_path, **overrides):
        return client.get("/health").headers["content-security-policy"]
    raise AssertionError("client fixture yielded nothing")


def test_default_is_same_origin_only(tmp_path):
    assert _csp(tmp_path) == "frame-ancestors 'self'"


def test_configured_origins_are_appended(tmp_path):
    csp = _csp(tmp_path, embed_frame_ancestors=["https://a.example", "https://b.example"])
    assert csp == "frame-ancestors 'self' https://a.example https://b.example"


def test_star_allows_any_host(tmp_path):
    assert _csp(tmp_path, embed_frame_ancestors=["*"]) == "frame-ancestors *"


@pytest.mark.parametrize("path", ["/health", "/v1/notebooks"])
def test_header_present_on_every_route(tmp_path, path):
    for client in _client(tmp_path):
        resp = client.get(path)
        assert "content-security-policy" in resp.headers
        assert "frame-ancestors" in resp.headers["content-security-policy"]

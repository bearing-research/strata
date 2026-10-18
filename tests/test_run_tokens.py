"""A cell run's token stands in for the proxy on the ambient client's routes, and only there."""

from __future__ import annotations

import time

import httpx
import pytest

from strata.auth import run_tokens
from strata.auth.run_tokens import RUN_TOKEN_HEADER, run_token
from strata.config import StrataConfig
from strata.types import Principal
from tests.conftest import LIVE_SERVER_TIMEOUT, find_free_port, run_server

ANA = Principal(id="ana", scopes=frozenset({"admin:*", "notebook:write"}))


def _proxy(principal: str, scopes: str = "") -> dict[str, str]:
    return {
        "X-Strata-Proxy-Token": "proxy-secret",
        "X-Strata-Principal": principal,
        "X-Strata-Scopes": scopes,
    }


def _token(token: str) -> dict[str, str]:
    return {RUN_TOKEN_HEADER: token}


@pytest.fixture
def server(tmp_path):
    config = StrataConfig(
        host="127.0.0.1",
        port=find_free_port(),
        cache_dir=tmp_path / "cache",
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="proxy-secret",
        rate_limit_enabled=False,
    )
    with run_server(config) as base:
        with httpx.Client(base_url=base, timeout=LIVE_SERVER_TIMEOUT) as client:
            yield client


def _scan(client: httpx.Client, temp_warehouse, headers: dict[str, str]) -> httpx.Response:
    return client.post(
        "/v1/materialize",
        json={
            "inputs": [temp_warehouse["table_uri"]],
            "transform": {"executor": "scan@v1", "params": {}},
            "mode": "stream",
        },
        headers=headers,
    )


def test_a_live_runs_token_scans_as_the_member(server, temp_warehouse):
    with run_token(ANA, "s1", 60) as token:
        response = _scan(server, temp_warehouse, _token(token))
        assert response.status_code == 200, response.text
        assert server.get(response.json()["stream_url"], headers=_token(token)).status_code == 200


def test_an_unknown_token_is_refused(server, temp_warehouse):
    assert _scan(server, temp_warehouse, _token("made-up")).status_code == 401


def test_a_token_is_refused_once_its_run_ends(server, temp_warehouse):
    with run_token(ANA, "s1", 60) as token:
        pass

    assert _scan(server, temp_warehouse, _token(token)).status_code == 401


def test_a_token_is_refused_past_its_hard_expiry(server, temp_warehouse, monkeypatch):
    with run_token(ANA, "s1", 60) as token:
        later = time.monotonic() + 60 + 61
        monkeypatch.setattr(run_tokens.time, "monotonic", lambda: later)

        assert _scan(server, temp_warehouse, _token(token)).status_code == 401


def test_a_route_outside_the_cells_list_is_refused(server):
    """Even for a member whose own request reaches it."""
    assert server.post("/v1/cache/clear", headers=_proxy("ana", "admin:*")).status_code == 200
    with run_token(ANA, "s1", 60) as token:
        response = server.post("/v1/cache/clear", headers=_token(token))

    assert response.status_code == 403
    assert "run token" in response.json()["detail"]


def test_promote_reaches_only_the_runs_own_session(server):
    path = "/v1/notebooks/{}/artifacts/a1/v/1/promote"
    with run_token(ANA, "s1", 60) as token:
        other = server.post(path.format("s2"), json={"name": "n"}, headers=_token(token))
        own = server.post(path.format("s1"), json={"name": "n"}, headers=_token(token))

    assert other.status_code == 403
    assert "run token" in other.json()["detail"]
    # Past the token: the route itself answers, here for a session that is not open.
    assert own.status_code == 404


def test_a_cell_does_not_carry_its_members_admin_scopes(server, temp_warehouse):
    """An ``admin:*`` member reads anyone's stream; their cell reads only its own."""
    stream_url = _scan(server, temp_warehouse, _proxy("bob")).json()["stream_url"]
    with run_token(ANA, "s1", 60) as token:
        # Not found rather than forbidden: hide_forbidden_as_not_found is on by default.
        assert server.get(stream_url, headers=_token(token)).status_code == 404
    assert server.get(stream_url, headers=_proxy("ana", "admin:*")).status_code == 200

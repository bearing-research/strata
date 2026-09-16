"""Notebook REST routes check the same scopes the WebSocket frames do. Item 7.

Frames checked ``notebook:read`` / ``notebook:write`` / ``notebook:execute``;
the REST routes checked nothing, so a principal with only ``notebook:read``
could run a cell with ``POST /cells/{id}/execute``. Driven through the real app
and its auth middleware.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from strata.notebook.scopes import required_scope_for_route


@pytest.fixture
def server(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(artifact_dir=str(tmp_path / "artifacts"))
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)
    return config


def _headers(scopes: str) -> dict[str, str]:
    return {
        "x-strata-proxy-token": "sekrit",
        "x-strata-principal": "alice",
        "x-strata-scopes": scopes,
    }


EXECUTE = "/v1/notebooks/nb1/cells/c1/execute"
CREATE_CELL = "/v1/notebooks/nb1/cells"
SESSIONS = "/v1/notebooks/sessions"


class TestUnderPrincipalAuth:
    @pytest.fixture(autouse=True)
    def _trusted_proxy(self, server):
        server.auth_mode = "trusted_proxy"
        server.proxy_token = "sekrit"

    def _client(self):
        from strata.server import app

        return TestClient(app)

    def test_read_scope_cannot_execute_or_create_a_cell(self):
        client = self._client()

        execute = client.post(EXECUTE, json={}, headers=_headers("notebook:read"))
        create = client.post(CREATE_CELL, json={}, headers=_headers("notebook:read"))

        assert execute.status_code == 403
        assert "notebook:execute" in execute.json()["detail"]
        assert create.status_code == 403
        assert "notebook:write" in create.json()["detail"]

    def test_read_scope_still_reads(self):
        response = self._client().get(SESSIONS, headers=_headers("notebook:read"))

        assert response.status_code == 200

    def test_write_scope_edits_but_does_not_execute(self):
        client = self._client()
        scopes = "notebook:read notebook:write"

        create = client.post(CREATE_CELL, json={}, headers=_headers(scopes))
        execute = client.post(EXECUTE, json={}, headers=_headers(scopes))

        # 404 is past the gate: the notebook does not exist, which is the
        # handler's answer rather than the gate's.
        assert create.status_code == 404
        assert execute.status_code == 403

    def test_execute_scope_passes_the_gate(self):
        response = self._client().post(
            EXECUTE, json={}, headers=_headers("notebook:read notebook:execute")
        )

        assert response.status_code == 404

    def test_admin_passes_every_gate(self):
        response = self._client().post(EXECUTE, json={}, headers=_headers("admin:*"))

        assert response.status_code == 404


def test_without_principal_auth_nothing_is_gated(server):
    from strata.server import app

    response = TestClient(app).post(EXECUTE, json={})

    assert response.status_code == 404


class TestTheTable:
    def test_code_and_environment_routes_need_execute(self):
        for method, path in (
            ("POST", "/v1/notebooks/{notebook_id}/cells/{cell_id}/execute"),
            ("POST", "/v1/notebooks/{notebook_id}/cells/{cell_id}/tests"),
            ("POST", "/v1/notebooks/{notebook_id}/dependencies"),
            ("PUT", "/v1/notebooks/{notebook_id}/python-version"),
            ("POST", "/v1/notebooks/{notebook_id}/ai/agent"),
        ):
            assert required_scope_for_route(method, path) == "notebook:execute", path

    def test_an_unclassified_mutation_fails_closed(self):
        assert (
            required_scope_for_route("POST", "/v1/notebooks/{notebook_id}/something-new")
            == "notebook:execute"
        )

    def test_every_notebook_route_is_behind_the_gate(self):
        """The gate is one router-level dependency; a route mounted on another
        router would escape it silently."""
        from strata.notebook.routes import _require_notebook_scope
        from strata.server import app

        ungated = [
            (sorted(route.methods), route.path)
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path.startswith(("/v1/notebooks", "/v1/projects"))
            and not any(d.dependency is _require_notebook_scope for d in route.dependencies)
        ]

        assert ungated == []

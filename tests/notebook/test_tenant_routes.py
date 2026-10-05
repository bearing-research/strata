"""Notebook REST routes keep one tenant out of another's sessions and notebooks.

A session records the tenant that opened it; another tenant gets the 404 an unknown
session gets. On a multi-tenant server each tenant's notebooks live under its own
subdir of the storage root. Driven through the real app and its auth middleware.
"""

from __future__ import annotations

import pytest
from fastapi.routing import APIRoute, iter_route_contexts
from fastapi.testclient import TestClient

from strata.notebook.routes import get_notebook_session, get_session_manager
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture(autouse=True)
def _close_opened_sessions():
    """The manager is a module global; a session left open counts against later tests."""
    manager = get_session_manager()
    before = set(manager.list_sessions())
    yield
    for session_id in manager.list_sessions():
        if session_id not in before:
            manager.close_session(session_id)


@pytest.fixture
def server(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState

    config = StrataConfig(
        artifact_dir=str(tmp_path / "artifacts"), notebook_storage_dir=tmp_path / "notebooks"
    )
    monkeypatch.setattr(server_module, "_state", ServerState(config), raising=False)
    return config


@pytest.fixture
def trusted_proxy(server):
    server.deployment_mode = "service"
    server.auth_mode = "trusted_proxy"
    server.proxy_token = "sekrit"
    return server


def _headers(tenant: str, scopes: str = "notebook:read notebook:write notebook:execute"):
    return {
        "x-strata-proxy-token": "sekrit",
        "x-strata-principal": f"someone-at-{tenant}",
        "x-strata-scopes": scopes,
        "x-tenant-id": tenant,
    }


def _client() -> TestClient:
    from strata.server import app

    return TestClient(app)


@pytest.fixture
def acme_session(server):
    """A session acme's member opened, with one cell."""
    notebook_dir = create_notebook(server.notebook_storage_dir, "acme_nb")
    add_cell_to_notebook(notebook_dir, "c1")
    write_cell(notebook_dir, "c1", "x = 1")
    session = get_session_manager().open_notebook(notebook_dir, opened_by=("ana", "acme"))
    return session


class TestSessionRoutes:
    def test_another_tenant_cannot_read_edit_or_run_the_session(self, trusted_proxy, acme_session):
        client = _client()
        globex = _headers("globex")
        sid = acme_session.id

        read = client.get(f"/v1/notebooks/{sid}/cells", headers=globex)
        write = client.post(f"/v1/notebooks/{sid}/cells", json={}, headers=globex)
        execute = client.post(f"/v1/notebooks/{sid}/cells/c1/execute", json={}, headers=globex)
        close = client.post(f"/v1/notebooks/{sid}/close", headers=globex)

        assert [r.status_code for r in (read, write, execute, close)] == [404] * 4
        assert len(acme_session.notebook_state.cells) == 1
        assert get_session_manager().get_session(sid) is acme_session

    def test_the_same_tenant_reads_and_edits(self, trusted_proxy, acme_session):
        client = _client()
        acme = _headers("acme")
        sid = acme_session.id

        read = client.get(f"/v1/notebooks/{sid}/cells", headers=acme)
        write = client.post(f"/v1/notebooks/{sid}/cells", json={}, headers=acme)

        assert read.status_code == 200, read.text
        assert write.status_code == 200, write.text
        assert len(acme_session.notebook_state.cells) == 2

    def test_an_admin_reaches_every_tenants_session(self, trusted_proxy, acme_session):
        response = _client().get(
            f"/v1/notebooks/{acme_session.id}/cells", headers=_headers("ops", "admin:*")
        )

        assert response.status_code == 200, response.text

    def test_personal_mode_is_unaffected(self, server, acme_session):
        response = _client().get(f"/v1/notebooks/{acme_session.id}/cells")

        assert response.status_code == 200, response.text

    def test_the_session_table_leaves_out_another_tenants_session(self, server, acme_session):
        """The /sessions routes run only in personal mode, which has no tenants; they
        filter anyway, so the rule does not depend on that gate."""
        server.auth_mode = "trusted_proxy"
        server.proxy_token = "sekrit"
        client = _client()

        listed = client.get("/v1/notebooks/sessions", headers=_headers("globex"))
        one = client.get(f"/v1/notebooks/sessions/{acme_session.id}", headers=_headers("globex"))
        own = client.get("/v1/notebooks/sessions", headers=_headers("acme"))

        assert listed.status_code == 200, listed.text
        assert acme_session.id not in [s["session_id"] for s in listed.json()["sessions"]]
        assert one.status_code == 404
        assert acme_session.id in [s["session_id"] for s in own.json()["sessions"]]

    def test_every_session_route_resolves_through_the_tenant_check(self):
        """A route that looked its session up itself would skip the tenant check."""
        from strata.server import app

        def depends_on_session(dependant) -> bool:
            return any(
                d.call is get_notebook_session or depends_on_session(d)
                for d in dependant.dependencies
            )

        session_routes = [
            route
            for route in iter_route_contexts(app.routes)
            if isinstance(route.original_route, APIRoute) and "{notebook_id}" in route.path
        ]
        unchecked = [
            (sorted(route.methods), route.path)
            for route in session_routes
            if not depends_on_session(route.original_route.dependant)
        ]

        assert len(session_routes) > 40
        assert unchecked == []


class TestTenantStorage:
    @pytest.fixture(autouse=True)
    def _multi_tenant(self, trusted_proxy):
        trusted_proxy.multi_tenant_enabled = True

    @pytest.fixture
    def acme_notebook(self, server):
        return create_notebook(server.notebook_storage_dir / "acme", "acme_nb")

    def test_another_tenant_cannot_list_or_open_the_notebook(self, acme_notebook):
        client = _client()
        globex = _headers("globex")

        listed = client.get("/v1/notebooks/discover", headers=globex)
        opened = client.post(
            "/v1/notebooks/open", json={"path": str(acme_notebook)}, headers=globex
        )

        assert listed.status_code == 200
        assert listed.json()["notebooks"] == []
        assert opened.status_code == 400
        assert "must be inside configured notebook storage" in opened.json()["detail"]

    def test_the_tenant_lists_and_opens_its_own_notebook(self, acme_notebook):
        client = _client()
        acme = _headers("acme")

        listed = client.get("/v1/notebooks/discover", headers=acme)
        opened = client.post("/v1/notebooks/open", json={"path": str(acme_notebook)}, headers=acme)

        assert [n["path"] for n in listed.json()["notebooks"]] == [str(acme_notebook.resolve())]
        assert opened.status_code == 200, opened.text
        session = get_session_manager().get_session(opened.json()["session_id"])
        assert session is not None
        assert session.opened_by == ("someone-at-acme", "acme")

    def test_a_new_notebook_lands_in_the_creators_tenant_dir(self, server):
        response = _client().post(
            "/v1/notebooks/create",
            json={"parent_path": "", "name": "fresh"},
            headers=_headers("acme"),
        )

        assert response.status_code == 200, response.text
        assert (server.notebook_storage_dir / "acme" / "fresh" / "notebook.toml").is_file()

    def test_an_admin_sees_every_tenants_notebooks(self, acme_notebook):
        listed = _client().get("/v1/notebooks/discover", headers=_headers("ops", "admin:*"))

        assert [n["path"] for n in listed.json()["notebooks"]] == [str(acme_notebook.resolve())]

    def test_one_tenant_server_keeps_the_shared_root(self, trusted_proxy, server):
        """Without multi_tenant_enabled a tenant header does not move anyone's notebooks."""
        trusted_proxy.multi_tenant_enabled = False
        notebook = create_notebook(server.notebook_storage_dir, "shared_nb")

        listed = _client().get("/v1/notebooks/discover", headers=_headers("acme"))

        assert [n["path"] for n in listed.json()["notebooks"]] == [str(notebook.resolve())]

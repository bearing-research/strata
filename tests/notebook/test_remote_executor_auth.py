"""Bearer-token auth on the remote executor's /v1/* endpoints.

With ``STRATA_WORKER_TOKEN`` set they require ``Authorization: Bearer <token>``;
unset, they stay open. ``/health`` is always open for liveness probes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.notebook.remote_executor import create_notebook_executor_app


@pytest.fixture
def client_without_token(monkeypatch):
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    return TestClient(create_notebook_executor_app())


@pytest.fixture
def client_with_token(monkeypatch):
    monkeypatch.setenv("STRATA_WORKER_TOKEN", "test-secret-xyz")
    return TestClient(create_notebook_executor_app())


def test_health_open_when_token_unset(client_without_token):
    resp = client_without_token.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_health_open_even_when_token_set(client_with_token):
    """/health stays open with auth on; platform probes cannot always carry the header."""
    resp = client_with_token.get("/health")
    assert resp.status_code == 200


@pytest.mark.parametrize("launch_id", [None, "lid-123"])
def test_health_reports_the_launch_id_the_worker_started_with(monkeypatch, launch_id):
    """The SSH supervisor checks it to tell its worker from another listener on the port."""
    if launch_id is None:
        monkeypatch.delenv("STRATA_WORKER_LAUNCH_ID", raising=False)
    else:
        monkeypatch.setenv("STRATA_WORKER_LAUNCH_ID", launch_id)
    client = TestClient(create_notebook_executor_app())
    assert client.get("/health").json()["launch_id"] == launch_id


def test_v1_execute_rejects_no_auth_when_token_set(client_with_token):
    """No token gives 401 before the body is read, so payload errors leak nothing."""
    resp = client_with_token.post("/v1/execute", content=b"")
    assert resp.status_code == 401
    detail = resp.json()["detail"]
    assert "Bearer" in detail or "Missing" in detail


def test_v1_execute_rejects_wrong_token(client_with_token):
    resp = client_with_token.post(
        "/v1/execute",
        content=b"",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401
    assert "Invalid worker token" in resp.json()["detail"]


def test_v1_execute_accepts_correct_token(client_with_token):
    """The correct token passes auth; the 400 for missing ``metadata`` proves parsing began."""
    resp = client_with_token.post(
        "/v1/execute",
        content=b"",
        headers={"Authorization": "Bearer test-secret-xyz"},
    )
    assert resp.status_code == 400
    assert "metadata" in resp.json()["detail"].lower()


def test_v1_execute_open_when_token_unset(client_without_token):
    """With no token configured, a request without auth reaches body parsing."""
    resp = client_without_token.post("/v1/execute", content=b"")
    assert resp.status_code == 400


def test_v1_notebook_execute_also_gated(client_with_token):
    resp = client_with_token.post("/v1/notebook-execute", content=b"")
    assert resp.status_code == 401


def test_v1_execute_manifest_also_gated(client_with_token):
    resp = client_with_token.post("/v1/execute-manifest", json={})
    assert resp.status_code == 401


def test_malformed_authorization_header_rejected(client_with_token):
    """A header not starting with ``Bearer `` fails like a missing one, hiding the scheme."""
    for bad in ["", "Token foo", "bearer test-secret-xyz", "Basic test-secret-xyz"]:
        resp = client_with_token.post(
            "/v1/execute",
            content=b"",
            headers={"Authorization": bad},
        )
        assert resp.status_code == 401, (bad, resp.text)


class TestPoolContractPath:
    """``POST /execute``, the path ``strata-pool`` dispatches to.

    The pool forwards the job payload verbatim with no content type; these pin that
    real wire shape, not a well-formed JSON request.
    """

    def test_requires_the_bearer_token(self, client_with_token):
        resp = client_with_token.post("/execute", content=b"{}")
        assert resp.status_code == 401

    def test_accepts_the_token_the_pool_mints(self, client_with_token):
        """The pool's per-machine ``STRATA_WORKER_TOKEN`` passes auth (400: empty manifest)."""
        resp = client_with_token.post(
            "/execute",
            content=b"{}",
            headers={"Authorization": "Bearer test-secret-xyz"},
        )
        assert resp.status_code == 400

    def test_parses_a_body_sent_without_a_content_type(self, client_without_token):
        """A body sent with no Content-Type is still parsed as JSON.

        A 400 naming the manifest proves the body was read, not rejected unparsed.
        """
        resp = client_without_token.post("/execute", content=b'{"metadata": {}}')
        assert "content-type" not in {k.lower() for k in resp.request.headers}
        assert resp.status_code == 400
        assert "executor ref" in resp.json()["detail"].lower()

    def test_rejects_a_body_that_is_not_json(self, client_without_token):
        resp = client_without_token.post("/execute", content=b"not json at all")
        assert resp.status_code == 400
        assert "invalid manifest payload" in resp.json()["detail"].lower()

    def test_matches_the_v1_manifest_endpoint(self, client_without_token):
        """The alias must not drift from the endpoint it aliases."""
        body = b'{"metadata": {"executor_ref": "nope"}}'
        alias = client_without_token.post("/execute", content=body)
        canonical = client_without_token.post("/v1/execute-manifest", content=body)
        assert alias.status_code == canonical.status_code
        assert alias.json() == canonical.json()

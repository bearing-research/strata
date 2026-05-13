"""Bearer-token auth on the remote executor's /v1/* endpoints.

The worker app exposes a public HTTP surface. When ``STRATA_WORKER_TOKEN``
is set the v1 execution endpoints require ``Authorization: Bearer
<token>``. When unset the endpoints stay open, matching the original
behaviour (backward compatibility with deployments that don't enforce
a token yet).

``/health`` always stays open — platform liveness probes can't carry
the secret.
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
    """Platform health probes (Fly, Cloudflare, k8s) can't always carry
    the auth header, so /health stays open even with auth enabled."""
    resp = client_with_token.get("/health")
    assert resp.status_code == 200


def test_v1_execute_rejects_no_auth_when_token_set(client_with_token):
    """Without a token, the endpoint returns 401 immediately —
    request body isn't even read, so misformed payloads don't leak
    error details that fingerprint the protocol version."""
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
    """Correct token gets past auth; the request fails downstream
    with 400 because the body has no ``metadata`` form field. That
    400 is the proof auth passed and the endpoint started parsing."""
    resp = client_with_token.post(
        "/v1/execute",
        content=b"",
        headers={"Authorization": "Bearer test-secret-xyz"},
    )
    assert resp.status_code == 400
    assert "metadata" in resp.json()["detail"].lower()


def test_v1_execute_open_when_token_unset(client_without_token):
    """Backward compat: deployments without STRATA_WORKER_TOKEN reach
    the endpoint without any Authorization header. The request still
    fails downstream on missing metadata, proving auth was bypassed."""
    resp = client_without_token.post("/v1/execute", content=b"")
    assert resp.status_code == 400


def test_v1_notebook_execute_also_gated(client_with_token):
    """All v1 endpoints share the auth dependency; check the older
    /v1/notebook-execute endpoint also enforces."""
    resp = client_with_token.post("/v1/notebook-execute", content=b"")
    assert resp.status_code == 401


def test_v1_execute_manifest_also_gated(client_with_token):
    """And the signed-URL manifest endpoint."""
    resp = client_with_token.post("/v1/execute-manifest", json={})
    assert resp.status_code == 401


def test_malformed_authorization_header_rejected(client_with_token):
    """Anything that doesn't start with ``Bearer `` fails the same way
    as a missing header — no leakage of expected scheme."""
    for bad in ["", "Token foo", "bearer test-secret-xyz", "Basic test-secret-xyz"]:
        resp = client_with_token.post(
            "/v1/execute",
            content=b"",
            headers={"Authorization": bad},
        )
        assert resp.status_code == 401, (bad, resp.text)

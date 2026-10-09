"""``/metrics/prometheus`` is open to scrapers, so it must not name who called a model.

Under principal auth the AI usage series sum every tenant and principal per model; in
personal mode they keep the ``tenant`` and ``principal`` labels.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.auth import set_principal
from strata.notebook.llm.usage import record_llm_usage, reset_llm_usage
from strata.types import Principal

SERVICE = {
    "deployment_mode": "service",
    "auth_mode": "trusted_proxy",
    "proxy_token": "test-token",
    "multi_tenant_enabled": True,
}


@pytest.fixture
def scrape(tmp_path):
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    original = server_module._state

    def _scrape(**overrides) -> str:
        config = StrataConfig(
            host="127.0.0.1",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            metadata_db=tmp_path / "meta.sqlite",
            **overrides,
        )
        server_module._state = ServerState(config)
        for principal, tenant, model, tokens in (
            ("ana", "acme", "m-1", 10),
            ("ben", "acme", "m-1", 5),
            ("ben", "acme", "m-2", 4),
            ("cy", "globex", "m-1", 3),
        ):
            set_principal(Principal(id=principal, tenant=tenant))
            record_llm_usage(model, tokens, 1)
        set_principal(None)
        # No proxy headers: a scraper calls the route as it is.
        response = TestClient(app).get("/metrics/prometheus")
        assert response.status_code == 200, response.text
        return response.text

    reset_artifact_store()
    reset_llm_usage()
    try:
        yield _scrape
    finally:
        set_principal(None)
        reset_llm_usage()
        server_module._state = original
        reset_artifact_store()


@pytest.mark.parametrize("multi_tenant", [True, False])
def test_a_service_with_principal_auth_names_no_caller(scrape, multi_tenant):
    body = scrape(**{**SERVICE, "multi_tenant_enabled": multi_tenant})

    assert "principal=" not in body and "tenant=" not in body
    assert "ana" not in body and "ben" not in body
    assert "acme" not in body and "globex" not in body
    assert 'strata_ai_calls_total{model="m-1"} 3' in body
    assert 'strata_ai_input_tokens_total{model="m-1"} 18' in body
    assert 'strata_ai_output_tokens_total{model="m-1"} 3' in body
    assert 'strata_ai_input_tokens_total{model="m-2"} 4' in body
    # The rest of the scrape is still there.
    assert "strata_scans_total" in body


def test_personal_mode_keeps_the_principal_label(scrape):
    body = scrape(deployment_mode="personal")

    assert 'strata_ai_calls_total{tenant="acme",principal="ana",model="m-1"} 1' in body
    assert 'strata_ai_input_tokens_total{tenant="acme",principal="ben",model="m-1"} 5' in body
    assert 'strata_ai_input_tokens_total{tenant="globex",principal="cy",model="m-1"} 3' in body

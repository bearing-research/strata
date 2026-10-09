"""``/metrics/prometheus`` is open to scrapers, so under principal auth it names no tenant.

The ``strata_tenant_*`` and ``strata_build_tenant_*`` series are left out there, as the
per-table ones are; personal mode keeps them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

TENANT = "acme_payroll"

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
    from strata.tenant_registry import get_tenant_registry, reset_tenant_registry
    from strata.transforms.build_metrics import init_build_metrics, reset_build_metrics

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
        get_tenant_registry().record_scan(TENANT, 1, 0, 10, 0, 5)
        init_build_metrics().record_started("b1", TENANT, "duckdb_sql@v1")
        # No proxy headers: a scraper calls the route as it is.
        response = TestClient(app).get("/metrics/prometheus")
        assert response.status_code == 200, response.text
        return response.text

    reset_artifact_store()
    reset_tenant_registry()
    reset_build_metrics()
    try:
        yield _scrape
    finally:
        server_module._state = original
        reset_build_metrics()
        reset_tenant_registry()
        reset_artifact_store()


@pytest.mark.parametrize("multi_tenant", [True, False])
def test_a_service_with_principal_auth_names_no_tenant(scrape, multi_tenant):
    body = scrape(**{**SERVICE, "multi_tenant_enabled": multi_tenant})

    assert TENANT not in body
    assert "tenant=" not in body
    # The rest of the scrape is still there, builds included.
    assert "strata_scans_total" in body
    assert "strata_builds_started_total 1" in body


def test_personal_mode_keeps_the_tenant_series(scrape):
    body = scrape(deployment_mode="personal")

    assert f'strata_tenant_scans_total{{tenant="{TENANT}"}} 1' in body
    assert f'strata_build_tenant_started_total{{tenant="{TENANT}"}} 1' in body

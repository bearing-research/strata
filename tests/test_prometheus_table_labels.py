"""``/metrics/prometheus`` is open to scrapers, so it must not hand out table names.

Under principal auth the per-table JSON routes need ``admin:*`` (the names span
tenants); the scrape leaves the ``strata_table_*`` series out there and keeps them
in personal mode.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from strata.metrics import ScanMetrics

TABLE = "strata.acme_payroll.salaries"

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
        server_module._state.metrics.log_scan_complete(
            ScanMetrics(scan_id="s1", snapshot_id=1, table_id=TABLE)
        )
        # No proxy headers: a scraper calls the route as it is.
        response = TestClient(app).get("/metrics/prometheus")
        assert response.status_code == 200, response.text
        return response.text

    reset_artifact_store()
    try:
        yield _scrape
    finally:
        server_module._state = original
        reset_artifact_store()


def test_a_multi_tenant_service_scrape_names_no_table(scrape):
    body = scrape(**SERVICE)

    assert "acme_payroll" not in body
    assert "strata_table_" not in body
    # The rest of the scrape is still there.
    assert "strata_scans_total" in body


def test_a_single_tenant_service_with_auth_names_no_table(scrape):
    body = scrape(**{**SERVICE, "multi_tenant_enabled": False})

    assert "acme_payroll" not in body
    assert "strata_table_" not in body


def test_personal_mode_keeps_the_table_series(scrape):
    body = scrape(deployment_mode="personal")

    assert f'strata_table_scans_total{{table="{TABLE}"}} 1' in body

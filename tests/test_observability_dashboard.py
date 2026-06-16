"""Guard against Grafana dashboard / metrics drift.

The provisioned dashboard silently rotted: QoS metrics gained a ``qos_`` prefix
(``strata_bulk_slots_used`` → ``strata_qos_bulk_slots``) but the dashboard kept
the old names, so its panels showed "No data" with nothing to flag it. This test
asserts every ``strata_*`` metric the dashboard references is one the server
actually exposes on ``/metrics/prometheus`` — so the next rename fails CI instead
of a dashboard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from strata.config import StrataConfig
from tests.conftest import find_free_port, run_server

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "observability/grafana/provisioning/dashboards/strata.json"
DATASOURCES = REPO_ROOT / "observability/grafana/provisioning/datasources/datasources.yml"

# Metric families emitted only under specific conditions — multi-tenant traffic,
# a registered circuit breaker, per-table activity — so they are absent from an
# idle server's scrape. Allowlisted by prefix rather than required to be live.
CONDITIONAL_PREFIXES = ("strata_tenant_", "strata_circuit_breaker_", "strata_table_")

_METRIC_RE = re.compile(r"strata_[a-z0-9_]+")


def _referenced_metrics() -> set[str]:
    """Every strata_* metric referenced by a panel target or template variable."""
    dashboard = json.loads(DASHBOARD.read_text())
    found: set[str] = set()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("expr", "definition") and isinstance(value, str):
                    found.update(_METRIC_RE.findall(value))
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(dashboard)
    return found


def test_dashboard_is_valid_and_uses_pinned_datasource():
    dashboard = json.loads(DASHBOARD.read_text())
    assert dashboard["panels"], "dashboard has no panels"
    # The datasource template variable resolves to the pinned Prometheus uid.
    ds_var = next(v for v in dashboard["templating"]["list"] if v.get("name") == "datasource")
    assert ds_var["current"]["value"] == "prometheus"
    assert "uid: prometheus" in DATASOURCES.read_text()


def test_dashboard_only_references_exposed_metrics(tmp_path):
    config = StrataConfig(
        host="127.0.0.1",
        port=find_free_port(),
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        deployment_mode="personal",
    )
    with run_server(config) as base_url:
        text = httpx.get(f"{base_url}/metrics/prometheus", timeout=10.0).text

    exposed = set(re.findall(r"(?m)^(strata_[a-z0-9_]+)", text))
    referenced = _referenced_metrics()
    assert referenced, "no metrics parsed from the dashboard"

    missing = sorted(
        m for m in referenced if m not in exposed and not m.startswith(CONDITIONAL_PREFIXES)
    )
    assert not missing, (
        "dashboard references metrics the server does not expose: "
        f"{missing}. Either the metric was renamed/removed (fix the dashboard), "
        "or add its family to CONDITIONAL_PREFIXES if it's only emitted under load."
    )

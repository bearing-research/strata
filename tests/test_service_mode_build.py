"""Service-mode build/persistence edge cases."""

import requests

from tests.conftest import run_server_with_context


def _artifact_mode_scan(table_uri: str) -> dict:
    return {
        "inputs": [table_uri],
        "transform": {"executor": "scan@v1", "params": {}},
        "mode": "artifact",
    }


def test_artifact_mode_without_store_is_rejected(temp_warehouse, tmp_path):
    """mode='artifact' needs a store, so without one it is a 400 rather than a hang.

    The background build would no-op and the build_id would never resolve.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    with run_server_with_context(cache_dir, None, "service") as ctx:
        resp = requests.post(
            f"{ctx.base_url}/v1/materialize",
            json=_artifact_mode_scan(temp_warehouse["table_uri"]),
        )
        assert resp.status_code == 400
        assert "artifact" in resp.json()["detail"].lower()


def test_artifact_mode_with_store_succeeds(temp_warehouse, tmp_path):
    """With an artifact_dir, artifact mode returns a build id."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    with run_server_with_context(cache_dir, artifact_dir, "service") as ctx:
        resp = requests.post(
            f"{ctx.base_url}/v1/materialize",
            json=_artifact_mode_scan(temp_warehouse["table_uri"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] in ("pending", "building", "ready")
        assert data["build_id"] is not None

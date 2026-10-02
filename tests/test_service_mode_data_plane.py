"""Scan and stream in genuine service mode (deployment_mode="service", writes off).

Covers the read-gateway pass-through, the identity-scan cache-hit read-back over /data, and table
ACLs on the scan path under trusted-proxy auth.
"""

import pyarrow.ipc as ipc
import requests

from strata.config import AclConfig, AclRule
from tests.conftest import run_server_with_context

ARROW = {"Accept": "application/vnd.apache.arrow.stream"}


def _scan(table_uri: str) -> dict:
    return {
        "inputs": [table_uri],
        "transform": {"executor": "scan@v1", "params": {}},
        "mode": "stream",
    }


def _auth(principal: str) -> dict:
    return {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": principal,
        "X-Tenant-ID": "team-a",
    }


class TestServiceModeScanStream:
    """The read gateway: scans stream in service mode."""

    def test_scan_stream_works_without_store(self, temp_warehouse, tmp_path):
        """Scan to bounded stream pass-through, with no persistence."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with run_server_with_context(cache_dir, None, "service") as ctx:
            table_uri = temp_warehouse["table_uri"]
            resp = requests.post(f"{ctx.base_url}/v1/materialize", json=_scan(table_uri))
            assert resp.status_code == 200
            data = resp.json()
            assert data["stream_url"].startswith("/v1/streams/")

            stream = requests.get(f"{ctx.base_url}{data['stream_url']}", headers=ARROW)
            assert stream.status_code == 200
            table = ipc.open_stream(stream.content).read_all()
            assert table.num_rows == 500
            assert set(table.column_names) == {"id", "value", "name", "timestamp"}

    def test_scan_cache_hit_readable_with_store(self, temp_warehouse, tmp_path):
        """An identity-scan cache hit returns a /data URL readable in service mode."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        with run_server_with_context(cache_dir, artifact_dir, "service") as ctx:
            table_uri = temp_warehouse["table_uri"]

            # Miss: stream (and persist the artifact via the wait-then-serve build).
            miss = requests.post(f"{ctx.base_url}/v1/materialize", json=_scan(table_uri)).json()
            assert miss["hit"] is False
            assert miss["stream_url"].startswith("/v1/streams/")
            s1 = requests.get(f"{ctx.base_url}{miss['stream_url']}", headers=ARROW)
            assert s1.status_code == 200

            # Hit: the response points at /data...
            hit = requests.post(f"{ctx.base_url}/v1/materialize", json=_scan(table_uri)).json()
            assert hit["hit"] is True
            assert "/data" in hit["stream_url"]

            # ...and the read-back works.
            s2 = requests.get(f"{ctx.base_url}{hit['stream_url']}", headers=ARROW)
            assert s2.status_code == 200
            table = ipc.open_stream(s2.content).read_all()
            assert table.num_rows == 500


class TestServiceModeScanAcl:
    """Table ACL is enforced on the scan path under trusted-proxy auth."""

    def test_scan_denied_table_rejected(self, temp_warehouse, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        acl = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:other.*",))],
        )
        with run_server_with_context(
            cache_dir,
            None,
            "service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            acl_config=acl,
            hide_forbidden_as_not_found=False,
        ) as ctx:
            resp = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=_scan(temp_warehouse["table_uri"]),
                headers=_auth("intruder"),
            )
            assert resp.status_code == 403

    def test_scan_allowed_table_streams(self, temp_warehouse, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        acl = AclConfig(
            default="deny",
            deny_rules=[],
            allow_rules=[AclRule(principal="analyst", tables=("file:test_db.*",))],
        )
        with run_server_with_context(
            cache_dir,
            None,
            "service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            acl_config=acl,
        ) as ctx:
            resp = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=_scan(temp_warehouse["table_uri"]),
                headers=_auth("analyst"),
            )
            assert resp.status_code == 200
            stream = requests.get(
                f"{ctx.base_url}{resp.json()['stream_url']}",
                headers={**_auth("analyst"), **ARROW},
            )
            assert stream.status_code == 200
            assert ipc.open_stream(stream.content).read_all().num_rows == 500


class TestMaterializeNameNeedsWriteScope:
    """``name`` on materialize writes a registry name, so it takes the ``POST /v1/names`` gate."""

    @staticmethod
    def _headers(principal: str, scopes: str = "") -> dict:
        return {**_auth(principal), "X-Strata-Scopes": scopes}

    @staticmethod
    def _named_scan(table_uri: str, name: str, mode: str) -> dict:
        return {**_scan(table_uri), "name": name, "mode": mode}

    def test_name_without_artifacts_write_is_refused(self, temp_warehouse, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with run_server_with_context(
            cache_dir,
            tmp_path / "artifacts",
            "service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            service_writes_enabled=True,
        ) as ctx:
            table_uri = temp_warehouse["table_uri"]
            writer = self._headers("alice", "artifacts:write")
            reader = self._headers("carol")

            # Miss: refused before anything is built or named.
            miss = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=self._named_scan(table_uri, "prod", "artifact"),
                headers=reader,
            )
            assert miss.status_code == 403
            assert miss.json()["detail"]["error"] == "missing_scope"

            # Hit: the writer's scan is cached, and naming it still needs the scope.
            first = requests.post(
                f"{ctx.base_url}/v1/materialize", json=_scan(table_uri), headers=writer
            )
            assert first.status_code == 200
            stream = requests.get(f"{ctx.base_url}{first.json()['stream_url']}", headers=writer)
            assert stream.status_code == 200
            hit = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=self._named_scan(table_uri, "prod", "stream"),
                headers=reader,
            )
            assert hit.status_code == 403
            assert requests.get(f"{ctx.base_url}/v1/names/prod", headers=writer).status_code == 404

            # The writer may name it.
            named = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=self._named_scan(table_uri, "prod", "stream"),
                headers=writer,
            )
            assert named.status_code == 200
            assert named.json()["hit"] is True
            resolved = requests.get(f"{ctx.base_url}/v1/names/prod", headers=writer)
            assert resolved.status_code == 200
            assert resolved.json()["artifact_uri"] == named.json()["artifact_uri"]

    def test_name_without_service_writes_is_refused(self, temp_warehouse, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with run_server_with_context(
            cache_dir,
            tmp_path / "artifacts",
            "service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
        ) as ctx:
            resp = requests.post(
                f"{ctx.base_url}/v1/materialize",
                json=self._named_scan(temp_warehouse["table_uri"], "prod", "artifact"),
                headers=self._headers("alice", "artifacts:write"),
            )
            assert resp.status_code == 403
            assert resp.json()["detail"]["error"] == "writes_disabled"

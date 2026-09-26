"""The import in two steps: the bytes, then the record that names them.

``PUT /v1/artifacts/import/blobs/{sha256}`` uploads an artifact's bytes ahead
of its record, and ``POST /v1/artifacts/import`` with a JSON body imports the
record whose ``content_sha256`` names them. This is the interface a publisher
copying a chain into a central store codes against (upstream item 1), and it
carries a large artifact without holding it in memory.
"""

from __future__ import annotations

import hashlib

import httpx
import pytest

from strata.artifact_store import IMPORT_STAGING_TTL_SECONDS, ArtifactStore

PROXY_TOKEN = "import-proxy-token"
BLOB = b"the bytes of a figure"
DIGEST = hashlib.sha256(BLOB).hexdigest()


def _record(artifact_id: str = "fig", version: int = 1, **extra) -> dict:
    return {
        "id": artifact_id,
        "version": version,
        "state": "ready",
        "provenance_hash": hashlib.sha256(artifact_id.encode()).hexdigest(),
        "created_at": 1.0,
        "content_sha256": DIGEST,
        **extra,
    }


def _stage(base: str, blob: bytes = BLOB, digest: str = DIGEST, headers=None):
    return httpx.put(
        f"{base}/v1/artifacts/import/blobs/{digest}", content=blob, headers=headers, timeout=10
    )


def _import(base: str, record: dict, headers=None, **params):
    return httpx.post(
        f"{base}/v1/artifacts/import", json=record, params=params, headers=headers, timeout=10
    )


@pytest.fixture
def served(tmp_path):
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "served"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        yield ctx.base_url, artifact_dir


class TestTwoSteps:
    def test_the_upload_then_the_record_imports_the_bytes(self, served):
        base, artifact_dir = served

        staged = _stage(base)
        assert staged.status_code == 201, staged.text
        assert staged.json() == {"content_sha256": DIGEST, "byte_size": len(BLOB)}

        imported = _import(base, _record())
        assert imported.status_code == 200, imported.text
        assert imported.json()["written"] is True

        store = ArtifactStore(artifact_dir)
        assert store.read_blob("fig", 1) == BLOB
        assert store.get_artifact("fig", 1).content_sha256 == DIGEST

    def test_the_upload_is_let_go_once_imported(self, served):
        base, artifact_dir = served
        _stage(base)
        _import(base, _record())

        assert ArtifactStore(artifact_dir).open_staged_import(None, DIGEST) is None

    def test_a_retry_after_the_import_went_through_writes_nothing(self, served):
        """The upload is gone by then, and the record needs none."""
        base, _ = served
        _stage(base)
        _import(base, _record())

        again = _import(base, _record())
        assert again.status_code == 200, again.text
        assert again.json()["written"] is False


class TestRefusals:
    def test_bytes_that_do_not_match_the_path_are_not_kept(self, served):
        base, _ = served

        assert _stage(base, blob=b"something else").status_code == 400
        assert _import(base, _record()).status_code == 400

    def test_a_record_with_nothing_uploaded_is_refused_naming_the_upload(self, served):
        base, artifact_dir = served

        response = _import(base, _record())
        assert response.status_code == 400
        assert f"/v1/artifacts/import/blobs/{DIGEST}" in response.json()["detail"]
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_record_without_its_creation_time_is_a_400(self, served):
        """The column is NOT NULL, so the database refused it as a 500."""
        base, artifact_dir = served
        _stage(base)
        record = _record()
        del record["created_at"]

        assert _import(base, record).status_code == 400
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_json_import_names_its_bytes(self, served):
        base, _ = served
        record = _record()
        del record["content_sha256"]

        assert _import(base, record).status_code == 400

    @pytest.mark.parametrize("version", [0, -1])
    def test_a_version_below_one_is_refused(self, served, version):
        """Staged uploads wait under version 0, which no artifact has."""
        base, _ = served
        _stage(base)

        assert _import(base, _record(version=version)).status_code == 400

    def test_the_digest_in_the_path_must_be_one(self, served):
        base, _ = served

        assert _stage(base, digest="not-a-digest").status_code == 422


class TestAbandonedUploads:
    def test_an_upload_nothing_imports_is_dropped_a_day_later(self, tmp_path):
        store = ArtifactStore(tmp_path / "store")
        source = tmp_path / "blob"
        source.write_bytes(BLOB)
        other = hashlib.sha256(b"later").hexdigest()

        store.stage_import_blob(None, DIGEST, source, len(BLOB), now=0.0)
        store.stage_import_blob(None, other, source, len(BLOB), now=IMPORT_STAGING_TTL_SECONDS - 1)
        assert store.open_staged_import(None, DIGEST) is not None

        store.stage_import_blob(None, other, source, len(BLOB), now=IMPORT_STAGING_TTL_SECONDS + 1)
        assert store.open_staged_import(None, DIGEST) is None
        assert store.open_staged_import(None, other) is not None


def _headers(tenant: str, principal: str = "publisher", scopes: str = "artifacts:write"):
    return {
        "X-Strata-Proxy-Token": PROXY_TOKEN,
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
        "X-Strata-Scopes": scopes,
    }


@pytest.fixture
def central(tmp_path):
    """A central store: service mode behind a trusted proxy, one tenant per team."""
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "central"
    with run_server_with_context(
        tmp_path / "cache",
        artifact_dir,
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
        hide_forbidden_as_not_found=False,
    ) as ctx:
        yield ctx.base_url, artifact_dir


class TestInACentralStore:
    def test_the_callers_tenant_is_stamped_whatever_the_record_says(self, central):
        base, artifact_dir = central
        _stage(base, headers=_headers("team-a"))

        response = _import(base, _record(tenant="team-b"), headers=_headers("team-a"))
        assert response.status_code == 200, response.text
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1).tenant == "team-a"

    def test_one_tenants_upload_does_not_satisfy_anothers_import(self, central):
        """A digest is printed on every publication's page, so knowing one is
        no proof of holding the bytes it names."""
        base, artifact_dir = central
        assert _stage(base, headers=_headers("team-a")).status_code == 201

        response = _import(base, _record(), headers=_headers("team-b"))
        assert response.status_code == 400
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_version_another_tenant_holds_is_a_409_and_remap_mints_a_fresh_id(self, central):
        base, _ = central
        _stage(base, headers=_headers("team-a"))
        _import(base, _record(), headers=_headers("team-a"))
        _stage(base, headers=_headers("team-b"))

        clash = _import(base, _record(), headers=_headers("team-b"))
        assert clash.status_code == 409

        remapped = _import(base, _record(), headers=_headers("team-b"), remap="true")
        assert remapped.status_code == 200, remapped.text
        assert remapped.json()["remapped"] is True
        assert remapped.json()["id"] != "fig"

    @pytest.mark.parametrize("step", ["stage", "import"])
    def test_without_the_write_scope_nothing_is_taken(self, central, step):
        base, _ = central
        reader = _headers("team-a", scopes="artifacts:read")

        response = (
            _stage(base, headers=reader)
            if step == "stage"
            else _import(base, _record(), headers=reader)
        )
        assert response.status_code == 403

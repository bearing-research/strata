"""Pull-model endpoints: build, manifest with signed URLs, download, upload, finalize."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from fastapi.testclient import TestClient

import strata.server as server_module
from strata.artifact_store import get_artifact_store, reset_artifact_store
from strata.config import StrataConfig
from strata.server import app
from strata.transforms.build_qos import (
    BuildQoS,
    BuildQoSConfig,
    reset_build_qos,
    set_build_qos,
)
from strata.transforms.build_store import (
    get_build_store,
    reset_build_store,
)
from strata.transforms.signed_urls import URLSigner

_TEST_SECRET = b"test-secret-key-12345678901234"
_TEST_SIGNER = URLSigner(_TEST_SECRET)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def config(temp_dir):
    """A test config with server transforms enabled."""
    return StrataConfig(
        cache_dir=temp_dir / "cache",
        deployment_mode="service",
        transforms_config={"enabled": True},
        artifact_dir=temp_dir / "artifacts",
        signed_url_expiry_seconds=600.0,
        # The signed build-transport routes mint upload + finalize capabilities, so in
        # service mode they need trusted-proxy auth; otherwise anyone who learned a build
        # id could forge an artifact.
        auth_mode="trusted_proxy",
        proxy_token="test-token",
    )


@pytest.fixture
def artifact_store(config):
    reset_artifact_store()
    store = get_artifact_store(config.artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(config):
    reset_build_store()
    db_path = config.artifact_dir / "artifacts.sqlite"
    store = get_build_store(db_path)
    yield store
    reset_build_store()


@pytest.fixture
def client(config, artifact_store, build_store):
    """A test client for the pull-model routes."""

    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    mock_state.url_signer = _TEST_SIGNER

    original_state = server_module._state
    server_module._state = mock_state

    # admin:* so these tests exercise the transport, not per-build ownership (builds
    # here have no owner).
    yield TestClient(
        app,
        headers={
            "X-Strata-Proxy-Token": "test-token",
            "X-Strata-Principal": "test-executor",
            "X-Strata-Scopes": "admin:*",
        },
    )

    server_module._state = original_state


@pytest.fixture
def trusted_proxy_client(temp_dir, artifact_store, build_store):
    """A trusted-proxy client for signed URL and tenant ACL tests."""
    config = StrataConfig(
        cache_dir=temp_dir / "cache-auth",
        deployment_mode="service",
        transforms_config={"enabled": True},
        artifact_dir=temp_dir / "artifacts",
        signed_url_expiry_seconds=600.0,
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        hide_forbidden_as_not_found=True,
    )

    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    mock_state.url_signer = _TEST_SIGNER

    original_state = server_module._state
    server_module._state = mock_state

    yield TestClient(app)

    server_module._state = original_state


def create_test_arrow_blob() -> bytes:
    """A small Arrow IPC stream."""
    schema = pa.schema([("id", pa.int64()), ("value", pa.string())])
    data = [
        pa.array([1, 2, 3], type=pa.int64()),
        pa.array(["a", "b", "c"], type=pa.string()),
    ]
    batch = pa.RecordBatch.from_arrays(data, schema=schema)

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)

    return sink.getvalue().to_pybytes()


def _auth_headers(
    tenant: str,
    principal: str = "user-1",
    scopes: str | None = None,
) -> dict[str, str]:
    headers = {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
    }
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


def create_test_artifact(artifact_store, artifact_id: str, finalize: bool = True) -> int:
    """Create an artifact (finalized unless ``finalize`` is false) and return its version."""
    provenance_hash = f"test-hash-{artifact_id}"
    version = artifact_store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
    )

    if finalize:
        blob = create_test_arrow_blob()
        artifact_store.write_blob(artifact_id, version, blob)
        artifact_store.finalize_artifact(artifact_id, version, "test-schema", 3, len(blob))

    return version


class TestBuildManifestEndpoint:
    """GET /v1/builds/{build_id}/manifest."""

    def test_get_manifest_for_pending_build(self, client, build_store, artifact_store):
        input_version = create_test_artifact(artifact_store, "input1", finalize=True)

        output_version = create_test_artifact(artifact_store, "output1", finalize=False)

        build_store.create_build(
            build_id="build-001",
            artifact_id="output1",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            input_uris=[f"strata://artifact/input1@v={input_version}"],
            params={"sql": "SELECT * FROM input"},
        )

        response = client.get("/v1/builds/build-001/manifest")
        assert response.status_code == 200

        data = response.json()
        assert data["build_id"] == "build-001"
        assert data["metadata"]["artifact_id"] == "output1"
        assert data["metadata"]["executor_ref"] == "duckdb_sql@v1"
        assert len(data["inputs"]) == 1
        assert data["inputs"][0]["artifact_id"] == "input1"
        assert data["inputs"][0]["version"] == 1
        assert "url" in data["inputs"][0]
        assert "signature=" in data["inputs"][0]["url"]
        assert data["output"]["max_bytes"] > 0
        assert "url" in data["output"]
        assert "finalize" in data["finalize_url"]

    def test_presigning_runs_off_the_event_loop(
        self, client, config, build_store, artifact_store, monkeypatch
    ):
        """Presigning can call the cloud (a role, IAM signBlob), so it must not block the loop."""
        import asyncio

        config.artifact_presigned_urls = True
        on_loop: list[bool] = []

        def presign_get(artifact_id, version, ttl_seconds):
            try:
                asyncio.get_running_loop()
                on_loop.append(True)
            except RuntimeError:
                on_loop.append(False)
            return None

        monkeypatch.setattr(artifact_store.blob_store, "presign_get", presign_get)
        input_version = create_test_artifact(artifact_store, "presign-in", finalize=True)
        output_version = create_test_artifact(artifact_store, "presign-out", finalize=False)
        build_store.create_build(
            build_id="build-presign",
            artifact_id="presign-out",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            input_uris=[f"strata://artifact/presign-in@v={input_version}"],
        )

        assert client.get("/v1/builds/build-presign/manifest").status_code == 200
        assert on_loop == [False]

    def test_get_manifest_for_building_build(self, client, build_store, artifact_store):
        """A manifest is available for a build that has already started."""
        input_version = create_test_artifact(artifact_store, "input-building", finalize=True)
        output_version = create_test_artifact(artifact_store, "output-building", finalize=False)

        build_store.create_build(
            build_id="build-building-001",
            artifact_id="output-building",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            input_uris=[f"strata://artifact/input-building@v={input_version}"],
            params={"sql": "SELECT * FROM input"},
        )
        build_store.start_build("build-building-001")

        response = client.get("/v1/builds/build-building-001/manifest")
        assert response.status_code == 200
        assert response.json()["build_id"] == "build-building-001"

    def test_get_manifest_resolves_name_inputs_with_build_tenant(
        self,
        client,
        build_store,
        artifact_store,
    ):
        """Tenant-scoped name inputs resolve within the build's own tenant."""
        version_a = create_test_artifact(artifact_store, "tenant-a-input", finalize=True)
        version_b = create_test_artifact(artifact_store, "tenant-b-input", finalize=True)
        artifact_store.set_name("shared-input", "tenant-a-input", version_a, tenant="team-a")
        artifact_store.set_name("shared-input", "tenant-b-input", version_b, tenant="team-b")

        output_version = create_test_artifact(artifact_store, "output-name", finalize=False)
        build_store.create_build(
            build_id="build-name-tenant",
            artifact_id="output-name",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            tenant_id="team-a",
            input_uris=["strata://name/shared-input"],
            params={"sql": "SELECT * FROM input"},
        )

        response = client.get("/v1/builds/build-name-tenant/manifest")
        assert response.status_code == 200
        data = response.json()
        assert data["inputs"][0]["artifact_id"] == "tenant-a-input"
        assert data["inputs"][0]["version"] == version_a

    def test_get_manifest_is_tenant_scoped_even_for_same_principal_id(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """Manifest access requires both the owning principal and tenant."""
        output_version = create_test_artifact(artifact_store, "output-authz", finalize=False)
        build_store.create_build(
            build_id="build-authz-001",
            artifact_id="output-authz",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            tenant_id="team-a",
            principal_id="shared-user",
        )

        response = trusted_proxy_client.get(
            "/v1/builds/build-authz-001/manifest",
            headers=_auth_headers("team-b", principal="shared-user"),
        )
        assert response.status_code == 404

    def test_get_manifest_not_found(self, client):
        response = client.get("/v1/builds/nonexistent/manifest")
        assert response.status_code == 404

    def test_get_manifest_completed_build_rejected(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "output2", finalize=False)
        build_store.create_build(
            build_id="build-002",
            artifact_id="output2",
            version=version,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build("build-002")
        build_store.complete_build("build-002")

        response = client.get("/v1/builds/build-002/manifest")
        assert response.status_code == 400
        assert "not in pending or building state" in response.json()["detail"]


class TestDownloadEndpoint:
    """GET /v1/artifacts/download."""

    def test_download_with_valid_signature(self, client, artifact_store):
        version = create_test_artifact(artifact_store, "dl-test", finalize=True)

        blob = artifact_store.read_blob("dl-test", version)

        signed = _TEST_SIGNER.generate_download_url(
            base_url="http://testserver",
            artifact_id="dl-test",
            version=version,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.get(
            "/v1/artifacts/download",
            params={
                "artifact_id": params["artifact_id"][0],
                "version": params["version"][0],
                "build_id": params["build_id"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
        assert response.content == blob

    def test_download_expired_signature_rejected(self, client, artifact_store):
        version = create_test_artifact(artifact_store, "dl-test2", finalize=True)

        signed = _TEST_SIGNER.generate_download_url(
            base_url="http://testserver",
            artifact_id="dl-test2",
            version=version,
            build_id="build-123",
            expiry_seconds=-1.0,  # Already expired
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.get(
            "/v1/artifacts/download",
            params={
                "artifact_id": params["artifact_id"][0],
                "version": params["version"][0],
                "build_id": params["build_id"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
        )

        assert response.status_code == 403
        assert "Invalid or expired signature" in response.json()["detail"]

    def test_download_tampered_signature_rejected(self, client, artifact_store):
        """Tampered parameters are rejected."""
        version = create_test_artifact(artifact_store, "dl-test3", finalize=True)

        signed = _TEST_SIGNER.generate_download_url(
            base_url="http://testserver",
            artifact_id="dl-test3",
            version=version,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.get(
            "/v1/artifacts/download",
            params={
                "artifact_id": "different-artifact",
                "version": params["version"][0],
                "build_id": params["build_id"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
        )

        assert response.status_code == 403


class TestUploadEndpoint:
    """POST /v1/artifacts/upload."""

    def test_upload_with_valid_signature(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "up-output", finalize=False)
        build_store.create_build(
            build_id="up-build-001",
            artifact_id="up-output",
            version=version,
            executor_ref="test@v1",
        )

        blob = create_test_arrow_blob()
        signed = _TEST_SIGNER.generate_upload_url(
            base_url="http://testserver",
            build_id="up-build-001",
            max_bytes=len(blob) + 1000,
            expiry_seconds=300.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.post(
            "/v1/artifacts/upload",
            params={
                "build_id": params["build_id"][0],
                "max_bytes": params["max_bytes"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
            content=blob,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "uploaded"
        assert data["byte_size"] == len(blob)

        stored_blob = artifact_store.read_blob("up-output", version)
        assert stored_blob == blob

    def test_upload_with_valid_signature_for_building_build(
        self, client, build_store, artifact_store
    ):
        """Upload works for a build that has already started."""
        version = create_test_artifact(artifact_store, "up-output-building", finalize=False)
        build_store.create_build(
            build_id="up-build-building-001",
            artifact_id="up-output-building",
            version=version,
            executor_ref="test@v1",
        )
        build_store.start_build("up-build-building-001")

        blob = create_test_arrow_blob()
        signed = _TEST_SIGNER.generate_upload_url(
            base_url="http://testserver",
            build_id="up-build-building-001",
            max_bytes=len(blob) + 1000,
            expiry_seconds=300.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)
        response = client.post(
            "/v1/artifacts/upload",
            params={
                "build_id": params["build_id"][0],
                "max_bytes": params["max_bytes"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
            content=blob,
        )

        assert response.status_code == 200
        assert artifact_store.read_blob("up-output-building", version) == blob

    def test_upload_exceeds_max_bytes_rejected(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "up-output2", finalize=False)
        build_store.create_build(
            build_id="up-build-002",
            artifact_id="up-output2",
            version=version,
            executor_ref="test@v1",
        )

        blob = create_test_arrow_blob()
        signed = _TEST_SIGNER.generate_upload_url(
            base_url="http://testserver",
            build_id="up-build-002",
            max_bytes=10,
            expiry_seconds=300.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.post(
            "/v1/artifacts/upload",
            params={
                "build_id": params["build_id"][0],
                "max_bytes": params["max_bytes"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
            content=blob,
        )

        assert response.status_code == 413
        assert "exceeds maximum size" in response.json()["detail"]

    def test_upload_oversize_does_not_commit_partial_blob(
        self, client, build_store, artifact_store
    ):
        """A 413 upload must not leave a partial blob in the store."""
        version = create_test_artifact(artifact_store, "up-output3", finalize=False)
        build_store.create_build(
            build_id="up-build-003",
            artifact_id="up-output3",
            version=version,
            executor_ref="test@v1",
        )

        blob = create_test_arrow_blob()
        signed = _TEST_SIGNER.generate_upload_url(
            base_url="http://testserver",
            build_id="up-build-003",
            max_bytes=10,
            expiry_seconds=300.0,
        )
        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.post(
            "/v1/artifacts/upload",
            params={
                "build_id": params["build_id"][0],
                "max_bytes": params["max_bytes"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
            content=blob,
        )

        assert response.status_code == 413
        assert not artifact_store.blob_exists("up-output3", version)

    def test_upload_expired_signature_rejected(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "up-output3", finalize=False)
        build_store.create_build(
            build_id="up-build-003",
            artifact_id="up-output3",
            version=version,
            executor_ref="test@v1",
        )

        signed = _TEST_SIGNER.generate_upload_url(
            base_url="http://testserver",
            build_id="up-build-003",
            max_bytes=10000,
            expiry_seconds=-1.0,
        )

        parsed = urlparse(signed.url)
        params = parse_qs(parsed.query)

        response = client.post(
            "/v1/artifacts/upload",
            params={
                "build_id": params["build_id"][0],
                "max_bytes": params["max_bytes"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
            content=b"test data",
        )

        assert response.status_code == 403


class TestFinalizeEndpoint:
    """POST /v1/builds/{build_id}/finalize."""

    def test_finalize_after_upload(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "fin-output", finalize=False)
        build_store.create_build(
            build_id="fin-build-001",
            artifact_id="fin-output",
            version=version,
            executor_ref="test@v1",
            name="my-result",
        )

        blob = create_test_arrow_blob()
        artifact_store.write_blob("fin-output", version, blob)

        response = client.post("/v1/builds/fin-build-001/finalize")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "finalized"
        assert data["build_id"] == "fin-build-001"
        assert f"fin-output@v={version}" in data["artifact_uri"]
        assert data["name_uri"] == "strata://name/my-result"
        assert data["row_count"] == 3  # The test blob has 3 rows.

        build = build_store.get_build("fin-build-001")
        assert build.state == "ready"

        artifact = artifact_store.get_artifact("fin-output", version)
        assert artifact.state == "ready"

    def test_finalize_refuses_an_output_over_the_limit(
        self, client, config, build_store, artifact_store
    ):
        """A presigned upload bypasses the upload route's byte count, so finalize checks size."""
        version = create_test_artifact(artifact_store, "big-output", finalize=False)
        build_store.create_build(
            build_id="big-build-001",
            artifact_id="big-output",
            version=version,
            executor_ref="test@v1",
        )
        blob = create_test_arrow_blob()
        artifact_store.write_blob("big-output", version, blob)
        config.build_runner_default_max_output = len(blob) - 1

        response = client.post("/v1/builds/big-build-001/finalize")

        assert response.status_code == 413
        build = build_store.get_build("big-build-001")
        assert build.state == "failed"
        assert build.error_code == "OUTPUT_TOO_LARGE"
        assert artifact_store.get_artifact("big-output", version).state == "failed"

    def test_finalize_arrow_validation_runs_off_event_loop(
        self, client, build_store, artifact_store, monkeypatch
    ):
        """Arrow schema and row-count validation runs in a worker thread."""
        import threading

        version = create_test_artifact(artifact_store, "fin-output-offload", finalize=False)
        build_store.create_build(
            build_id="fin-build-offload",
            artifact_id="fin-output-offload",
            version=version,
            executor_ref="test@v1",
        )

        blob = create_test_arrow_blob()
        artifact_store.write_blob("fin-output-offload", version, blob)

        main_thread_ident = threading.get_ident()
        reader_thread_idents: list[int] = []

        real_open_blob_reader = artifact_store.open_blob_reader

        def _recording_open_blob_reader(artifact_id, version_, attempt=None):
            reader_thread_idents.append(threading.get_ident())
            return real_open_blob_reader(artifact_id, version_, attempt)

        monkeypatch.setattr(artifact_store, "open_blob_reader", _recording_open_blob_reader)

        response = client.post("/v1/builds/fin-build-offload/finalize")
        assert response.status_code == 200

        assert reader_thread_idents, "Arrow validation did not open the blob reader"
        assert all(tid != main_thread_ident for tid in reader_thread_idents), (
            "Arrow validation reader ran on the event loop thread"
        )

    def test_finalize_records_quota_bytes(self, client, build_store, artifact_store):
        """Finalize charges the produced bytes to the build's QoS quota."""
        qos = BuildQoS(BuildQoSConfig(bytes_per_day_limit=10 * 1024 * 1024))
        set_build_qos(qos)

        try:
            version = create_test_artifact(artifact_store, "fin-output-quota", finalize=False)
            build_store.create_build(
                build_id="fin-build-quota-001",
                artifact_id="fin-output-quota",
                version=version,
                executor_ref="test@v1",
                tenant_id="tenant-finalize",
            )

            blob = create_test_arrow_blob()
            artifact_store.write_blob("fin-output-quota", version, blob)

            response = client.post("/v1/builds/fin-build-quota-001/finalize")
            assert response.status_code == 200

            tenant_metrics = qos.get_tenant_metrics("tenant-finalize")
            assert tenant_metrics is not None
            assert tenant_metrics["quota"]["bytes_today"] == len(blob)
        finally:
            reset_build_qos()

    def test_finalize_after_upload_for_building_build(self, client, build_store, artifact_store):
        """Finalize works after the build has moved to building."""
        version = create_test_artifact(artifact_store, "fin-output-building", finalize=False)
        build_store.create_build(
            build_id="fin-build-building-001",
            artifact_id="fin-output-building",
            version=version,
            executor_ref="test@v1",
            name="my-result-building",
        )
        build_store.start_build("fin-build-building-001")

        blob = create_test_arrow_blob()
        artifact_store.write_blob("fin-output-building", version, blob)

        response = client.post("/v1/builds/fin-build-building-001/finalize")
        assert response.status_code == 200
        assert response.json()["artifact_uri"].endswith(f"fin-output-building@v={version}")

    def test_finalize_duplicate_provenance_repoints_build(
        self, client, build_store, artifact_store
    ):
        """A duplicate returns the canonical artifact URI and repoints the build."""
        blob = create_test_arrow_blob()
        existing_version = artifact_store.create_artifact("canonical-output", "shared-hash")
        artifact_store.write_blob("canonical-output", existing_version, blob)
        artifact_store.finalize_artifact("canonical-output", existing_version, "{}", 3, 100)

        duplicate_version = artifact_store.create_artifact("duplicate-output", "shared-hash")
        build_store.create_build(
            build_id="fin-build-duplicate-001",
            artifact_id="duplicate-output",
            version=duplicate_version,
            executor_ref="test@v1",
            name="duplicate-result",
        )
        build_store.start_build("fin-build-duplicate-001")

        artifact_store.write_blob("duplicate-output", duplicate_version, blob)

        response = client.post("/v1/builds/fin-build-duplicate-001/finalize")
        assert response.status_code == 200
        data = response.json()
        assert data["artifact_uri"] == f"strata://artifact/canonical-output@v={existing_version}"
        assert data["name_uri"] == "strata://name/duplicate-result"

        build = build_store.get_build("fin-build-duplicate-001")
        assert build is not None
        assert build.state == "ready"
        assert build.artifact_id == "canonical-output"
        assert build.version == existing_version

        # Overtaken, not failed: the URI materialize handed out still serves its rows, read from
        # the canonical's blob; its own bytes are gone.
        duplicate_artifact = artifact_store.get_artifact("duplicate-output", duplicate_version)
        assert duplicate_artifact is not None
        assert duplicate_artifact.state == "superseded"
        assert artifact_store.read_blob("duplicate-output", duplicate_version) == blob
        assert not artifact_store.blob_store.blob_exists("duplicate-output", duplicate_version)

    def test_finalize_without_upload_rejected(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "fin-output2", finalize=False)
        build_store.create_build(
            build_id="fin-build-002",
            artifact_id="fin-output2",
            version=version,
            executor_ref="test@v1",
        )

        response = client.post("/v1/builds/fin-build-002/finalize")
        assert response.status_code == 400
        assert "Blob not uploaded" in response.json()["detail"]

    def test_finalize_already_complete_rejected(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "fin-output3", finalize=False)
        build_store.create_build(
            build_id="fin-build-003",
            artifact_id="fin-output3",
            version=version,
            executor_ref="test@v1",
        )

        build_store.start_build("fin-build-003")
        build_store.complete_build("fin-build-003")

        response = client.post("/v1/builds/fin-build-003/finalize")
        assert response.status_code == 400
        assert "not in pending or building state" in response.json()["detail"]

    def test_finalize_invalid_arrow_fails_build(self, client, build_store, artifact_store):
        """Invalid Arrow data marks the build failed."""
        version = create_test_artifact(artifact_store, "fin-output4", finalize=False)
        build_store.create_build(
            build_id="fin-build-004",
            artifact_id="fin-output4",
            version=version,
            executor_ref="test@v1",
        )

        artifact_store.write_blob("fin-output4", version, b"not valid arrow data")

        response = client.post("/v1/builds/fin-build-004/finalize")
        assert response.status_code == 400
        assert "Invalid Arrow IPC format" in response.json()["detail"]

        build = build_store.get_build("fin-build-004")
        assert build.state == "failed"
        assert build.error_code == "INVALID_ARROW_FORMAT"

    def test_finalize_is_tenant_scoped_even_for_same_principal_id(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """Unsigned finalize requires both the owning principal and tenant."""
        version = create_test_artifact(artifact_store, "fin-authz-output", finalize=False)
        build_store.create_build(
            build_id="fin-authz-001",
            artifact_id="fin-authz-output",
            version=version,
            executor_ref="test@v1",
            tenant_id="team-a",
            principal_id="shared-user",
        )
        artifact_store.write_blob("fin-authz-output", version, create_test_arrow_blob())

        response = trusted_proxy_client.post(
            "/v1/builds/fin-authz-001/finalize",
            headers=_auth_headers("team-b", principal="shared-user"),
        )
        assert response.status_code == 404


class TestBuildStatusEndpoint:
    """Build status access control."""

    def test_build_status_is_tenant_scoped_even_for_same_principal_id(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """Build polling requires both the owning principal and tenant."""
        version = create_test_artifact(artifact_store, "status-output", finalize=False)
        build_store.create_build(
            build_id="status-build-001",
            artifact_id="status-output",
            version=version,
            executor_ref="test@v1",
            tenant_id="team-a",
            principal_id="shared-user",
        )

        response = trusted_proxy_client.get(
            "/v1/artifacts/builds/status-build-001",
            headers=_auth_headers("team-b", principal="shared-user"),
        )
        assert response.status_code == 404


class TestPullModelEndToEnd:
    def test_complete_pull_model_flow(self, client, build_store, artifact_store):
        input_version = create_test_artifact(artifact_store, "e2e-input", finalize=True)
        input_blob = artifact_store.read_blob("e2e-input", input_version)

        output_version = create_test_artifact(artifact_store, "e2e-output", finalize=False)
        build_store.create_build(
            build_id="e2e-build-001",
            artifact_id="e2e-output",
            version=output_version,
            executor_ref="test@v1",
            input_uris=[f"strata://artifact/e2e-input@v={input_version}"],
            params={"query": "SELECT * FROM input"},
            name="e2e-result",
        )

        response = client.get("/v1/builds/e2e-build-001/manifest")
        assert response.status_code == 200
        manifest = response.json()

        input_url = manifest["inputs"][0]["url"]
        parsed = urlparse(input_url)
        params = parse_qs(parsed.query)
        response = client.get("/v1/artifacts/download", params={k: v[0] for k, v in params.items()})
        assert response.status_code == 200
        assert response.content == input_blob

        # "Execute" the transform by reusing the same blob.
        output_blob = create_test_arrow_blob()

        output_url = manifest["output"]["url"]
        parsed = urlparse(output_url)
        params = parse_qs(parsed.query)
        response = client.post(
            "/v1/artifacts/upload",
            params={k: v[0] for k, v in params.items()},
            content=output_blob,
        )
        assert response.status_code == 200

        response = client.post(manifest["finalize_url"].replace("http://testserver", ""))
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "finalized"
        assert result["name_uri"] == "strata://name/e2e-result"

        build = build_store.get_build("e2e-build-001")
        assert build.state == "ready"

        artifact = artifact_store.get_artifact("e2e-output", output_version)
        assert artifact.state == "ready"

        name_info = artifact_store.get_name("e2e-result")
        assert name_info is not None
        assert name_info.artifact_id == "e2e-output"
        assert name_info.version == output_version

    def test_signed_urls_are_self_sufficient_in_trusted_proxy_mode(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """Signed URLs work without proxy auth headers."""
        input_version = create_test_artifact(artifact_store, "auth-e2e-input", finalize=True)
        output_version = create_test_artifact(artifact_store, "auth-e2e-output", finalize=False)
        build_store.create_build(
            build_id="auth-e2e-build-001",
            artifact_id="auth-e2e-output",
            version=output_version,
            executor_ref="test@v1",
            tenant_id="team-a",
            principal_id="user-1",
            input_uris=[f"strata://artifact/auth-e2e-input@v={input_version}"],
        )

        manifest_response = trusted_proxy_client.get(
            "/v1/builds/auth-e2e-build-001/manifest",
            headers=_auth_headers("team-a", principal="user-1"),
        )
        assert manifest_response.status_code == 200
        manifest = manifest_response.json()

        input_url = manifest["inputs"][0]["url"]
        parsed = urlparse(input_url)
        params = parse_qs(parsed.query)
        download_response = trusted_proxy_client.get(
            "/v1/artifacts/download",
            params={k: v[0] for k, v in params.items()},
        )
        assert download_response.status_code == 200

        output_blob = create_test_arrow_blob()
        output_url = manifest["output"]["url"]
        parsed = urlparse(output_url)
        params = parse_qs(parsed.query)
        upload_response = trusted_proxy_client.post(
            "/v1/artifacts/upload",
            params={k: v[0] for k, v in params.items()},
            content=output_blob,
        )
        assert upload_response.status_code == 200

        finalize_response = trusted_proxy_client.post(
            manifest["finalize_url"].replace("http://testserver", ""),
        )
        assert finalize_response.status_code == 200
        assert finalize_response.json()["status"] == "finalized"

    def test_a_signed_log_url_is_self_sufficient_in_trusted_proxy_mode(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """A worker posts console chunks holding only the log capability, no proxy token."""
        output_version = create_test_artifact(artifact_store, "auth-log-output", finalize=False)
        build_store.create_build(
            build_id="auth-log-build-001",
            artifact_id="auth-log-output",
            version=output_version,
            executor_ref="test@v1",
            tenant_id="team-a",
            principal_id="user-1",
        )
        log_url = _TEST_SIGNER.generate_log_url(
            base_url="http://testserver", build_id="auth-log-build-001"
        )

        response = trusted_proxy_client.post(
            log_url.replace("http://testserver", "") + "&stream=stdout&seq=0",
            content=b"epoch 1\n",
        )
        assert response.status_code == 202

        unsigned = trusted_proxy_client.post(
            "/v1/builds/auth-log-build-001/log?stream=stdout&seq=0",
            content=b"epoch 1\n",
        )
        assert unsigned.status_code == 401


@pytest.fixture
def unauthenticated_service_client(temp_dir, artifact_store, build_store):
    """A service-mode server with ``auth_mode="none"`` and no loopback restriction."""
    config = StrataConfig(
        cache_dir=temp_dir / "cache-noauth",
        deployment_mode="service",
        transforms_config={"enabled": True},
        artifact_dir=temp_dir / "artifacts",
        signed_url_expiry_seconds=600.0,
    )

    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    mock_state.url_signer = _TEST_SIGNER

    original_state = server_module._state
    server_module._state = mock_state
    yield TestClient(app)
    server_module._state = original_state


class TestManifestMintingRequiresAuth:
    """The manifest route mints capabilities, so it requires auth.

    Without it, anyone with a build id could upload and finalize forged bytes under the build's
    provenance hash, served to every later identical materialize as a cache hit.
    """

    def _seed_build(self, build_store, artifact_store, build_id="build-mint"):
        output_version = create_test_artifact(artifact_store, "mint-out", finalize=False)
        build_store.create_build(
            build_id=build_id,
            artifact_id="mint-out",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )

    def test_unauthenticated_service_mode_cannot_mint(
        self, unauthenticated_service_client, build_store, artifact_store
    ):
        self._seed_build(build_store, artifact_store)
        resp = unauthenticated_service_client.get("/v1/builds/build-mint/manifest")
        assert resp.status_code == 404
        # An error body, not a manifest: no signed capability of any kind.
        assert "signature=" not in resp.text
        assert "output" not in resp.json()
        assert "inputs" not in resp.json()

    def test_trusted_proxy_service_mode_can_mint(self, client, build_store, artifact_store):
        """The authenticated client (see the ``config`` fixture) still mints."""
        self._seed_build(build_store, artifact_store, build_id="build-mint-ok")
        resp = client.get("/v1/builds/build-mint-ok/manifest")
        assert resp.status_code == 200, resp.text
        assert resp.json()["output"]["url"]

    def test_redeeming_stays_signature_authed(
        self, unauthenticated_service_client, build_store, artifact_store
    ):
        """Redeeming needs only the signature: a worker holds a signed URL, not an identity.

        An unsigned redeem is still refused.
        """
        self._seed_build(build_store, artifact_store, build_id="build-redeem")
        resp = unauthenticated_service_client.post("/v1/builds/build-redeem/finalize")
        # Refused for want of a signature (401/403), not for want of a principal and not
        # with the manifest route's 404.
        assert resp.status_code in (400, 401, 403), resp.text


class TestManifestClaimsTheBuild:
    """Issuing a manifest claims the build, so the local runner cannot also run it.

    An unclaimed build would have two writers on one ``(artifact_id, version)`` blob, and finalize
    could validate one blob while completing another.
    """

    def _pending(self, build_store, artifact_store, build_id):
        version = create_test_artifact(artifact_store, f"out-{build_id}", finalize=False)
        build_store.create_build(
            build_id=build_id,
            artifact_id=f"out-{build_id}",
            version=version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )

    def test_manifest_moves_the_build_out_of_pending(self, client, build_store, artifact_store):
        self._pending(build_store, artifact_store, "claim-1")
        assert build_store.get_build("claim-1").state == "pending"

        assert client.get("/v1/builds/claim-1/manifest").status_code == 200

        claimed = build_store.get_build("claim-1")
        assert claimed.state == "building", "runner would still have polled this build"
        assert claimed.lease_owner == "external:manifest"

    def test_runner_cannot_claim_a_build_handed_to_an_executor(
        self, client, build_store, artifact_store
    ):
        """The runner's claim is the same atomic ``WHERE state = 'pending'`` update."""
        self._pending(build_store, artifact_store, "claim-2")
        client.get("/v1/builds/claim-2/manifest")

        assert build_store.claim_build("claim-2", lease_owner="runner-abc") is False

    def test_refetching_a_manifest_extends_the_lease_past_the_new_urls(
        self, client, build_store, artifact_store
    ):
        """A re-fetch mints fresh URLs, so the lease must cover them.

        Otherwise the orphan sweep could reclaim the build while the executor can still finalize.
        """
        self._pending(build_store, artifact_store, "claim-refetch")
        assert client.get("/v1/builds/claim-refetch/manifest").status_code == 200
        first_lease = build_store.get_build("claim-refetch").lease_expires_at

        # Simulate the executor working most of its window before re-fetching.
        conn = build_store._get_connection()
        conn.execute(
            "UPDATE artifact_builds SET lease_expires_at = ? WHERE build_id = ?",
            (time.time() + 5.0, "claim-refetch"),
        )
        conn.commit()
        conn.close()
        near_expiry = build_store.get_build("claim-refetch").lease_expires_at
        assert near_expiry < first_lease

        assert client.get("/v1/builds/claim-refetch/manifest").status_code == 200

        renewed = build_store.get_build("claim-refetch")
        assert renewed.lease_owner == "external:manifest"
        # The lease now covers the freshly minted URLs instead of expiring while they are
        # still usable.
        assert renewed.lease_expires_at > near_expiry
        assert renewed.lease_expires_at > time.time() + 5.0

    def test_finalize_refused_once_the_lease_was_reclaimed(
        self, client, build_store, artifact_store, monkeypatch
    ):
        """An executor whose lease was reclaimed must not publish its result.

        Finalize passes a fencing owner to complete_build.
        """
        version = create_test_artifact(artifact_store, "fin-fenced", finalize=False)
        build_store.create_build(
            build_id="fin-fenced-001",
            artifact_id="fin-fenced",
            version=version,
            executor_ref="test@v1",
        )
        artifact_store.write_blob("fin-fenced", version, create_test_arrow_blob())

        # The executor holds the build when its finalize request starts.
        assert build_store.claim_build("fin-fenced-001", lease_owner="external:manifest")

        # The sweep reclaims it *while* that request is in flight. Artifact finalization
        # is the real work mid-handler, so a takeover there is the interleaving the fence
        # exists for.
        store = artifact_store
        real_finalize = store.finalize_and_set_name

        def reclaim_then_finalize(*args, **kwargs):
            conn = build_store._get_connection()
            conn.execute(
                "UPDATE artifact_builds SET lease_expires_at = ? WHERE build_id = ?",
                (time.time() - 1.0, "fin-fenced-001"),
            )
            conn.commit()
            conn.close()
            build_store.reclaim_expired_build("fin-fenced-001", new_lease_owner="runner-9")
            return real_finalize(*args, **kwargs)

        monkeypatch.setattr(store, "finalize_and_set_name", reclaim_then_finalize)

        response = client.post("/v1/builds/fin-fenced-001/finalize")

        assert response.status_code == 409
        # The build still belongs to the runner that took it over.
        build = build_store.get_build("fin-fenced-001")
        assert build.state == "building"
        assert build.lease_owner == "runner-9"

    def test_a_capability_from_a_previous_claim_publishes_nothing(
        self, client, build_store, artifact_store
    ):
        """Fencing at the end is too late: the artifact and name commit first.

        The lease token in the signed URL says which claim the request belongs to, so a stale one is
        rejected before publishing.
        """
        self._pending(build_store, artifact_store, "stale-1")
        manifest = client.get("/v1/builds/stale-1/manifest").json()
        finalize_url = manifest["finalize_url"]
        artifact_store.write_blob("out-stale-1", 1, create_test_arrow_blob())

        # The sweep hands the build to a runner: same build, new claim.
        conn = build_store._get_connection()
        conn.execute(
            "UPDATE artifact_builds SET lease_expires_at = ? WHERE build_id = ?",
            (time.time() - 1.0, "stale-1"),
        )
        conn.commit()
        conn.close()
        assert build_store.reclaim_expired_build("stale-1", new_lease_owner="runner-9")

        response = client.post(finalize_url)

        assert response.status_code == 409
        # Refused *before* anything was written.
        assert artifact_store.get_latest_version("out-stale-1") is None
        assert build_store.get_build("stale-1").lease_owner == "runner-9"

    def test_the_current_holder_can_still_finalize(self, client, build_store, artifact_store):
        self._pending(build_store, artifact_store, "fresh-1")
        manifest = client.get("/v1/builds/fresh-1/manifest").json()
        _upload(client, manifest, create_test_arrow_blob())

        response = client.post(manifest["finalize_url"])

        assert response.status_code == 200
        assert artifact_store.get_latest_version("out-fresh-1") is not None

    def test_refetching_a_manifest_retires_the_previous_capability(
        self, client, build_store, artifact_store
    ):
        """A re-fetch renews the lease, so the earlier URLs stop verifying.

        One live capability set at a time makes two writers impossible by construction.
        """
        self._pending(build_store, artifact_store, "refetch-1")
        first_manifest = client.get("/v1/builds/refetch-1/manifest").json()
        second_manifest = client.get("/v1/builds/refetch-1/manifest").json()
        first, second = first_manifest["finalize_url"], second_manifest["finalize_url"]
        assert first != second
        _upload(client, second_manifest, create_test_arrow_blob())

        # 409, not 403: the older URL is properly signed, so not a forgery; it names a
        # claim that is no longer current.
        assert client.post(first).status_code == 409
        assert artifact_store.get_latest_version("out-refetch-1") is None
        assert client.post(second).status_code == 200

    def test_manifest_refused_once_the_runner_holds_the_lease(
        self, client, build_store, artifact_store
    ):
        self._pending(build_store, artifact_store, "claim-3")
        assert build_store.claim_build("claim-3", lease_owner="runner-abc") is True

        resp = client.get("/v1/builds/claim-3/manifest")
        assert resp.status_code == 409
        assert "local runner" in resp.text

    def test_refetching_our_own_manifest_still_works(self, client, build_store, artifact_store):
        """An executor retrying its fetch is not a second writer."""
        self._pending(build_store, artifact_store, "claim-4")
        assert client.get("/v1/builds/claim-4/manifest").status_code == 200
        assert client.get("/v1/builds/claim-4/manifest").status_code == 200


class TestARetiredManifestCannotWriteTheOutput:
    """Found by formal verification (BuildLease.tla, NoStaleBytesPublished).

    An executor holding a retired manifest could upload into the current slot and have the current
    holder publish it. Each claim uploads under its own attempt key, and finalize reads only its
    own.
    """

    def test_finalize_publishes_the_current_claims_bytes(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "out-pull", finalize=False)
        build_store.create_build(
            build_id="pull-1",
            artifact_id="out-pull",
            version=version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )
        first = client.get("/v1/builds/pull-1/manifest").json()
        second = client.get("/v1/builds/pull-1/manifest").json()

        _upload(client, second, _ipc_values([2]))
        _upload(client, first, _ipc_values([1]))  # still signed, but its own key

        assert client.post(second["finalize_url"]).status_code == 200

        data = artifact_store.read_blob("out-pull", version)
        assert pa.ipc.open_stream(data).read_all().column("x").to_pylist() == [2]
        assert not [f for f in artifact_store.verify_artifacts() if f["artifact_id"] == "out-pull"]

    def test_a_losing_duplicate_finalize_keeps_the_published_bytes(
        self, client, build_store, artifact_store
    ):
        """Two finalizes under one lease share an attempt id.

        The loser's cleanup targets the attempt the winner just published, so it must leave it
        alone.
        """
        version = create_test_artifact(artifact_store, "out-pull-3", finalize=False)
        build_store.create_build(
            build_id="pull-3",
            artifact_id="out-pull-3",
            version=version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )
        first = client.get("/v1/builds/pull-3/manifest").json()
        second = client.get("/v1/builds/pull-3/manifest").json()
        _upload(client, first, _ipc_values([1]))
        _upload(client, second, _ipc_values([2]))
        assert client.post(second["finalize_url"]).status_code == 200

        def attempt_of(manifest):
            return parse_qs(urlparse(manifest["output"]["url"]).query)["attempt"][0]

        artifact_store.delete_attempt_blob("out-pull-3", version, attempt_of(second))
        artifact_store.delete_attempt_blob("out-pull-3", version, attempt_of(first))

        data = artifact_store.read_blob("out-pull-3", version)
        assert pa.ipc.open_stream(data).read_all().column("x").to_pylist() == [2]
        assert not artifact_store.blob_exists("out-pull-3", version, attempt_of(first))

    def test_an_attempt_the_url_did_not_sign_is_refused(self, client, build_store, artifact_store):
        version = create_test_artifact(artifact_store, "out-pull-2", finalize=False)
        build_store.create_build(
            build_id="pull-2",
            artifact_id="out-pull-2",
            version=version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )
        url = client.get("/v1/builds/pull-2/manifest").json()["output"]["url"]
        assert "attempt=" in url
        forged = re.sub(r"attempt=[0-9a-f]+", "attempt=" + "0" * 32, url)

        assert client.post(forged, content=_ipc_values([9])).status_code == 403


class TestAnUploadThatIsNeverFinalizedIsSwept:
    """An attempt uploaded but never finalized is swept once its build is over.

    Blob stores cannot list keys, so every attempt is recorded when its manifest is issued.
    """

    def _uploaded_and_abandoned(self, client, build_store, artifact_store, artifact_id):
        version = create_test_artifact(artifact_store, artifact_id, finalize=False)
        build_store.create_build(
            build_id=f"build-{artifact_id}",
            artifact_id=artifact_id,
            version=version,
            executor_ref="duckdb_sql@v1",
            input_uris=[],
            params={},
        )
        manifest = client.get(f"/v1/builds/build-{artifact_id}/manifest").json()
        _upload(client, manifest, _ipc_values([1]))
        attempt = parse_qs(urlparse(manifest["output"]["url"]).query)["attempt"][0]
        assert artifact_store.blob_exists(artifact_id, version, attempt)
        return version, attempt

    def test_the_abandoned_upload_is_deleted_once_its_url_has_expired(
        self, client, build_store, artifact_store, monkeypatch
    ):
        from strata.transforms.runner import sweep_settled_attempts

        version, attempt = self._uploaded_and_abandoned(
            client, build_store, artifact_store, "out-abandoned"
        )
        assert build_store.fail_build("build-out-abandoned", "executor gave up")

        # The URL is still signed: an upload could still land, so wait.
        sweep_settled_attempts(build_store, artifact_store)
        assert artifact_store.blob_exists("out-abandoned", version, attempt)

        monkeypatch.setattr(build_store, "_clock", lambda: time.time() + 86_400)
        sweep_settled_attempts(build_store, artifact_store)
        assert not artifact_store.blob_exists("out-abandoned", version, attempt)
        assert build_store.settled_attempts() == []

    def test_a_build_still_running_keeps_its_upload(
        self, client, build_store, artifact_store, monkeypatch
    ):
        from strata.transforms.runner import sweep_settled_attempts

        version, attempt = self._uploaded_and_abandoned(
            client, build_store, artifact_store, "out-running"
        )
        monkeypatch.setattr(build_store, "_clock", lambda: time.time() + 86_400)
        sweep_settled_attempts(build_store, artifact_store)
        assert artifact_store.blob_exists("out-running", version, attempt)


def _ipc_values(values):
    sink = pa.BufferOutputStream()
    table = pa.table({"x": values})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _upload(client, manifest, blob):
    """Upload through the manifest's signed URL, as an executor does."""
    response = client.post(manifest["output"]["url"], content=blob)
    assert response.status_code == 200, response.text

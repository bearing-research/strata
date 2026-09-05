"""Tests for pull model endpoints (Stage 2).

Tests the complete pull model flow:
1. Create a build
2. Get manifest with signed URLs
3. Download inputs via signed URL
4. Upload output via signed URL
5. Finalize build
"""

from __future__ import annotations

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
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def config(temp_dir):
    """Create a test config with server transforms enabled."""
    return StrataConfig(
        cache_dir=temp_dir / "cache",
        deployment_mode="service",
        transforms_config={"enabled": True},
        artifact_dir=temp_dir / "artifacts",
        signed_url_expiry_seconds=600.0,
        # The signed build-transport routes mint upload + finalize
        # capabilities, so in service mode they require trusted-proxy auth
        # (an unauthenticated, network-reachable server would let anyone who
        # learned a build id forge an artifact). The client below sends the
        # matching headers.
        auth_mode="trusted_proxy",
        proxy_token="test-token",
    )


@pytest.fixture
def artifact_store(config):
    """Create an artifact store for testing."""
    reset_artifact_store()
    store = get_artifact_store(config.artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(config):
    """Create a build store for testing."""
    reset_build_store()
    db_path = config.artifact_dir / "artifacts.sqlite"
    store = get_build_store(db_path)
    yield store
    reset_build_store()


@pytest.fixture
def client(config, artifact_store, build_store):
    """Create a test client with pull model enabled."""
    # Set signing secret for reproducible tests

    # Create mock state
    mock_state = MagicMock()
    mock_state.config = config
    mock_state.planner = MagicMock()
    mock_state.fetcher = MagicMock()
    mock_state.scans = {}
    mock_state.metrics = MagicMock()
    mock_state.url_signer = _TEST_SIGNER

    # Patch _state on server module
    original_state = server_module._state
    server_module._state = mock_state

    # admin:* so these transport tests exercise the routes rather than the
    # per-build ownership rules (builds here are created without an owner).
    yield TestClient(
        app,
        headers={
            "X-Strata-Proxy-Token": "test-token",
            "X-Strata-Principal": "test-executor",
            "X-Strata-Scopes": "admin:*",
        },
    )

    # Restore
    server_module._state = original_state


@pytest.fixture
def trusted_proxy_client(temp_dir, artifact_store, build_store):
    """Create a trusted-proxy client for signed URL and tenant ACL tests."""
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
    """Create a small Arrow IPC stream for testing."""
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
    """Helper to create an artifact for testing.

    Args:
        artifact_store: The artifact store
        artifact_id: Artifact ID to create
        finalize: Whether to finalize the artifact

    Returns:
        Version number
    """
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
    """Tests for GET /v1/builds/{build_id}/manifest."""

    def test_get_manifest_for_pending_build(self, client, build_store, artifact_store):
        """Can get manifest for a pending build."""
        # Create an input artifact first
        input_version = create_test_artifact(artifact_store, "input1", finalize=True)

        # Create output artifact placeholder
        output_version = create_test_artifact(artifact_store, "output1", finalize=False)

        # Create a build with input_uris
        build_store.create_build(
            build_id="build-001",
            artifact_id="output1",
            version=output_version,
            executor_ref="duckdb_sql@v1",
            input_uris=[f"strata://artifact/input1@v={input_version}"],
            params={"sql": "SELECT * FROM input"},
        )

        # Get manifest
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

    def test_get_manifest_for_building_build(self, client, build_store, artifact_store):
        """Can get manifest for a build that has already started."""
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
        """Tenant-scoped name inputs should resolve within the owning build tenant."""
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
        """Returns 404 for non-existent build."""
        response = client.get("/v1/builds/nonexistent/manifest")
        assert response.status_code == 404

    def test_get_manifest_completed_build_rejected(self, client, build_store, artifact_store):
        """Cannot get manifest for completed build."""
        version = create_test_artifact(artifact_store, "output2", finalize=False)
        build_store.create_build(
            build_id="build-002",
            artifact_id="output2",
            version=version,
            executor_ref="duckdb_sql@v1",
        )
        # Mark as running then complete
        build_store.start_build("build-002")
        build_store.complete_build("build-002")

        response = client.get("/v1/builds/build-002/manifest")
        assert response.status_code == 400
        assert "not in pending or building state" in response.json()["detail"]


class TestDownloadEndpoint:
    """Tests for GET /v1/artifacts/download."""

    def test_download_with_valid_signature(self, client, artifact_store):
        """Can download artifact with valid signed URL."""
        # Create artifact
        version = create_test_artifact(artifact_store, "dl-test", finalize=True)

        # Read the blob that was written
        blob = artifact_store.read_blob("dl-test", version)

        # Generate signed URL
        signed = _TEST_SIGNER.generate_download_url(
            base_url="http://testserver",
            artifact_id="dl-test",
            version=version,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        # Extract query params
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
        """Expired signature is rejected."""
        version = create_test_artifact(artifact_store, "dl-test2", finalize=True)

        # Generate expired URL
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

        # Tamper with artifact_id
        response = client.get(
            "/v1/artifacts/download",
            params={
                "artifact_id": "different-artifact",  # Tampered!
                "version": params["version"][0],
                "build_id": params["build_id"][0],
                "expires_at": params["expires_at"][0],
                "signature": params["signature"][0],
            },
        )

        assert response.status_code == 403


class TestUploadEndpoint:
    """Tests for POST /v1/artifacts/upload."""

    def test_upload_with_valid_signature(self, client, build_store, artifact_store):
        """Can upload artifact with valid signed URL."""
        # Create build
        version = create_test_artifact(artifact_store, "up-output", finalize=False)
        build_store.create_build(
            build_id="up-build-001",
            artifact_id="up-output",
            version=version,
            executor_ref="test@v1",
        )

        # Generate signed upload URL
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

        # Verify blob was written
        stored_blob = artifact_store.read_blob("up-output", version)
        assert stored_blob == blob

    def test_upload_with_valid_signature_for_building_build(
        self, client, build_store, artifact_store
    ):
        """Can upload artifact for a build that has already started."""
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
        """Upload exceeding max_bytes is rejected."""
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
            max_bytes=10,  # Very small limit
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
        """Expired upload signature is rejected."""
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
            expiry_seconds=-1.0,  # Expired
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
    """Tests for POST /v1/builds/{build_id}/finalize."""

    def test_finalize_after_upload(self, client, build_store, artifact_store):
        """Can finalize a build after uploading blob."""
        # Create artifact and build
        version = create_test_artifact(artifact_store, "fin-output", finalize=False)
        build_store.create_build(
            build_id="fin-build-001",
            artifact_id="fin-output",
            version=version,
            executor_ref="test@v1",
            name="my-result",
        )

        # Upload blob
        blob = create_test_arrow_blob()
        artifact_store.write_blob("fin-output", version, blob)

        # Finalize
        response = client.post("/v1/builds/fin-build-001/finalize")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "finalized"
        assert data["build_id"] == "fin-build-001"
        assert f"fin-output@v={version}" in data["artifact_uri"]
        assert data["name_uri"] == "strata://name/my-result"
        assert data["row_count"] == 3  # Our test blob has 3 rows

        # Verify build is complete
        build = build_store.get_build("fin-build-001")
        assert build.state == "ready"

        # Verify artifact is ready
        artifact = artifact_store.get_artifact("fin-output", version)
        assert artifact.state == "ready"

    def test_finalize_arrow_validation_runs_off_event_loop(
        self, client, build_store, artifact_store, monkeypatch
    ):
        """Arrow schema/row-count validation must run in a worker thread."""
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

        def _recording_open_blob_reader(artifact_id, version_):
            reader_thread_idents.append(threading.get_ident())
            return real_open_blob_reader(artifact_id, version_)

        monkeypatch.setattr(artifact_store, "open_blob_reader", _recording_open_blob_reader)

        response = client.post("/v1/builds/fin-build-offload/finalize")
        assert response.status_code == 200

        assert reader_thread_idents, "Arrow validation did not open the blob reader"
        assert all(tid != main_thread_ident for tid in reader_thread_idents), (
            "Arrow validation reader ran on the event loop thread"
        )

    def test_finalize_records_quota_bytes(self, client, build_store, artifact_store):
        """Pull-model finalize should account for produced bytes against build QoS quota."""
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
        """Can finalize a build after it has already transitioned to building."""
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
        """Duplicate finalization returns the canonical artifact URI and repoints the build."""
        existing_version = artifact_store.create_artifact("canonical-output", "shared-hash")
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

        blob = create_test_arrow_blob()
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

        duplicate_artifact = artifact_store.get_artifact("duplicate-output", duplicate_version)
        assert duplicate_artifact is not None
        assert duplicate_artifact.state == "failed"

    def test_finalize_without_upload_rejected(self, client, build_store, artifact_store):
        """Cannot finalize without uploading blob first."""
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
        """Cannot finalize an already complete build."""
        version = create_test_artifact(artifact_store, "fin-output3", finalize=False)
        build_store.create_build(
            build_id="fin-build-003",
            artifact_id="fin-output3",
            version=version,
            executor_ref="test@v1",
        )

        # Complete the build
        build_store.start_build("fin-build-003")
        build_store.complete_build("fin-build-003")

        response = client.post("/v1/builds/fin-build-003/finalize")
        assert response.status_code == 400
        assert "not in pending or building state" in response.json()["detail"]

    def test_finalize_invalid_arrow_fails_build(self, client, build_store, artifact_store):
        """Finalizing with invalid Arrow data marks build as failed."""
        version = create_test_artifact(artifact_store, "fin-output4", finalize=False)
        build_store.create_build(
            build_id="fin-build-004",
            artifact_id="fin-output4",
            version=version,
            executor_ref="test@v1",
        )

        # Write invalid Arrow data
        artifact_store.write_blob("fin-output4", version, b"not valid arrow data")

        response = client.post("/v1/builds/fin-build-004/finalize")
        assert response.status_code == 400
        assert "Invalid Arrow IPC format" in response.json()["detail"]

        # Build should be marked as failed
        build = build_store.get_build("fin-build-004")
        assert build.state == "failed"
        assert build.error_code == "INVALID_ARROW_FORMAT"

    def test_finalize_is_tenant_scoped_even_for_same_principal_id(
        self,
        trusted_proxy_client,
        build_store,
        artifact_store,
    ):
        """Unsigned finalize access requires both the owning principal and tenant."""
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
    """Tests for build status access control."""

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
    """End-to-end test of the complete pull model flow."""

    def test_complete_pull_model_flow(self, client, build_store, artifact_store):
        """Test the complete pull model workflow."""
        # Step 1: Create input artifact
        input_version = create_test_artifact(artifact_store, "e2e-input", finalize=True)
        input_blob = artifact_store.read_blob("e2e-input", input_version)

        # Step 2: Create build with input_uris
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

        # Step 3: Get manifest
        response = client.get("/v1/builds/e2e-build-001/manifest")
        assert response.status_code == 200
        manifest = response.json()

        # Step 4: Download input using signed URL from manifest
        input_url = manifest["inputs"][0]["url"]
        parsed = urlparse(input_url)
        params = parse_qs(parsed.query)
        response = client.get("/v1/artifacts/download", params={k: v[0] for k, v in params.items()})
        assert response.status_code == 200
        assert response.content == input_blob

        # Step 5: "Execute" transform (just use the same blob for testing)
        output_blob = create_test_arrow_blob()

        # Step 6: Upload output using signed URL from manifest
        output_url = manifest["output"]["url"]
        parsed = urlparse(output_url)
        params = parse_qs(parsed.query)
        response = client.post(
            "/v1/artifacts/upload",
            params={k: v[0] for k, v in params.items()},
            content=output_blob,
        )
        assert response.status_code == 200

        # Step 7: Finalize build
        response = client.post(manifest["finalize_url"].replace("http://testserver", ""))
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "finalized"
        assert result["name_uri"] == "strata://name/e2e-result"

        # Verify final state
        build = build_store.get_build("e2e-build-001")
        assert build.state == "ready"

        artifact = artifact_store.get_artifact("e2e-output", output_version)
        assert artifact.state == "ready"

        # Verify name pointer was set
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
        """Signed pull-model URLs should work without proxy auth headers."""
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


@pytest.fixture
def unauthenticated_service_client(temp_dir, artifact_store, build_store):
    """A service-mode server with ``auth_mode="none"`` — a config the coherence
    validator accepts, and one with no loopback restriction."""
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
    """The manifest route MINTS capabilities — a signed upload URL plus a
    finalize URL, with nothing binding the uploaded bytes to the executor's
    identity. In an unauthenticated service-mode deployment (no loopback
    restriction), anyone who learned a build id could PUT arbitrary Arrow IPC
    and finalize it; the forged artifact is keyed by the build's provenance
    hash, so every later identical materialize serves it as a dedup cache hit.
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
        """Redeeming a capability must NOT require a principal — a worker holds
        a signed URL, not an identity. That is the whole point of the pull
        model, and is how the notebook's signed remote workers operate: the
        notebook assembles the manifest in-process and the worker only redeems.
        An unsigned redeem is still refused, by the signature check."""
        self._seed_build(build_store, artifact_store, build_id="build-redeem")
        resp = unauthenticated_service_client.post("/v1/builds/build-redeem/finalize")
        # Refused for want of a signature (401/403), not for want of a principal
        # and not with the manifest route's 404.
        assert resp.status_code in (400, 401, 403), resp.text


class TestManifestClaimsTheBuild:
    """Issuing a manifest must take the build, so the local runner can't also
    execute it.

    Previously the manifest route left the build in ``pending`` and recorded
    nothing, so ``BuildRunner``'s poll loop picked the same build up and ran it
    via v1-push while the pull executor was uploading — two writers on the same
    ``(artifact_id, version)`` blob, with ``finalize_build`` then validating one
    blob's schema/row-count while completing a different one.
    ``_is_runner_managed_build`` only skips builds whose *user-supplied* params
    carry ``_dispatch_mode == "external"``, which only the notebook path sets.
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
        """The runner's own claim must now fail — it is the same atomic
        ``WHERE state = 'pending'`` update the manifest route used."""
        self._pending(build_store, artifact_store, "claim-2")
        client.get("/v1/builds/claim-2/manifest")

        assert build_store.claim_build("claim-2", lease_owner="runner-abc") is False

    def test_refetching_a_manifest_extends_the_lease_past_the_new_urls(
        self, client, build_store, artifact_store
    ):
        """A re-fetch mints a fresh set of signed URLs, so the lease has to be
        pushed out to cover them.

        The claim only runs while the build is still 'pending', so a re-fetch
        used to leave the lease on its original deadline while handing out URLs
        good for a full window past it. That gap is the orphan sweep's window to
        reclaim the build while the executor can still finalize it.
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
        # The lease now covers the freshly minted URLs rather than expiring
        # while they are still usable.
        assert renewed.lease_expires_at > near_expiry
        assert renewed.lease_expires_at > time.time() + 5.0

    def test_finalize_refused_once_the_lease_was_reclaimed(
        self, client, build_store, artifact_store, monkeypatch
    ):
        """An executor whose lease was reclaimed must not publish its result.

        complete_build takes a fencing owner for exactly this ("a runner whose
        lease was stolen ... published over the runner that legitimately took
        the build over"), but the finalize route called it without one, so a
        late executor still recorded its result as authoritative.
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

        # The sweep reclaims it *while* that request is in flight. Artifact
        # finalization is the real work in the middle of the handler, so a
        # takeover landing there is the interleaving the fence exists for.
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
        """The ordering #582 could not fix by fencing.

        finalize_and_set_name commits the artifact and moves the name pointer
        before the fence at the end of the handler runs, so a stale executor
        used to publish its bytes and only then be told it had lost the build.
        Rejecting it needs the request to say *which claim* it belongs to,
        which is what the lease token in the signed URL is for.
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
        # The point of the change: refused *before* anything was written.
        assert artifact_store.get_latest_version("out-stale-1") is None
        assert build_store.get_build("stale-1").lease_owner == "runner-9"

    def test_the_current_holder_can_still_finalize(self, client, build_store, artifact_store):
        self._pending(build_store, artifact_store, "fresh-1")
        manifest = client.get("/v1/builds/fresh-1/manifest").json()
        artifact_store.write_blob("out-fresh-1", 1, create_test_arrow_blob())

        response = client.post(manifest["finalize_url"])

        assert response.status_code == 200
        assert artifact_store.get_latest_version("out-fresh-1") is not None

    def test_refetching_a_manifest_retires_the_previous_capability(
        self, client, build_store, artifact_store
    ):
        """The open question in #583, answered by the deadline moving.

        A re-fetch renews the lease to cover its fresh URLs, so the earlier
        set stops verifying instead of staying usable alongside them. One live
        capability set at a time is the property that makes two writers
        impossible by construction rather than by durations lining up.
        """
        self._pending(build_store, artifact_store, "refetch-1")
        first = client.get("/v1/builds/refetch-1/manifest").json()["finalize_url"]
        second = client.get("/v1/builds/refetch-1/manifest").json()["finalize_url"]
        assert first != second
        artifact_store.write_blob("out-refetch-1", 1, create_test_arrow_blob())

        # 409, not 403: the older URL is properly signed, so it is not a
        # forgery — it names a claim that is no longer current.
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

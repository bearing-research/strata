"""Tests for blob storage backends."""

from pathlib import Path

import pytest
from vestigo.blob_store import (
    GCSBlobStore,
    LocalBlobStore,
    S3BlobStore,
    create_blob_store,
)

from strata.config import StrataConfig


class TestCreateBlobStore:
    def test_creates_local_store_by_default(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("STRATA_ARTIFACT_BLOB_BACKEND", raising=False)
        monkeypatch.delenv("STRATA_ARTIFACT_S3_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        store = create_blob_store(config)

        assert isinstance(store, LocalBlobStore)

    def test_creates_s3_store_from_env(self, tmp_path: Path, monkeypatch):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="my-bucket")

            monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "s3")
            monkeypatch.setenv("STRATA_ARTIFACT_S3_BUCKET", "my-bucket")
            monkeypatch.setenv("STRATA_ARTIFACT_S3_PREFIX", "custom-prefix")

            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                s3_region="us-east-1",
            )

            store = create_blob_store(config)

            assert isinstance(store, S3BlobStore)
            assert store.bucket == "my-bucket"
            assert store.prefix == "custom-prefix"

    def test_raises_without_s3_bucket(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "s3")
        monkeypatch.delenv("STRATA_ARTIFACT_S3_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="S3 blob backend requires"):
            create_blob_store(config)

    def test_raises_without_artifact_dir(self, monkeypatch):
        monkeypatch.delenv("STRATA_ARTIFACT_BLOB_BACKEND", raising=False)

        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="test-token",
            artifact_dir=None,
        )

        with pytest.raises(ValueError, match="requires artifact_dir"):
            create_blob_store(config)

    def test_creates_gcs_store_from_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "gcs")
        monkeypatch.setenv("STRATA_ARTIFACT_GCS_BUCKET", "my-gcs-bucket")
        monkeypatch.setenv("STRATA_ARTIFACT_GCS_PREFIX", "custom-prefix")
        monkeypatch.setenv("STRATA_GCS_ANONYMOUS", "true")

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            gcs_anonymous=True,
        )

        store = create_blob_store(config)

        assert isinstance(store, GCSBlobStore)
        assert store.bucket == "my-gcs-bucket"
        assert store.prefix == "custom-prefix"

    def test_raises_without_gcs_bucket(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "gcs")
        monkeypatch.delenv("STRATA_ARTIFACT_GCS_BUCKET", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="GCS blob backend requires"):
            create_blob_store(config)

    def test_creates_azure_store_from_env(self, tmp_path: Path, monkeypatch):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "azure")
        monkeypatch.setenv("STRATA_ARTIFACT_AZURE_CONTAINER", "my-container")
        monkeypatch.setenv("STRATA_ARTIFACT_AZURE_PREFIX", "custom-prefix")

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )

        store = create_blob_store(config)

        assert isinstance(store, AzureBlobStore)
        assert store.container_name == "my-container"
        assert store.prefix == "custom-prefix"

    def test_raises_without_azure_container(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_BLOB_BACKEND", "azure")
        monkeypatch.delenv("STRATA_ARTIFACT_AZURE_CONTAINER", raising=False)

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )

        with pytest.raises(ValueError, match="Azure blob backend requires"):
            create_blob_store(config)


class TestConfigCreateBlobStore:
    def test_creates_local_store(self, tmp_path: Path):
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="local",
        )

        store = config.create_blob_store()

        assert isinstance(store, LocalBlobStore)

    def test_creates_s3_store(self, tmp_path: Path):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws

        with mock_aws():
            conn = boto3.client("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="my-bucket")

            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                artifact_blob_backend="s3",
                artifact_s3_bucket="my-bucket",
                artifact_s3_prefix="my-prefix",
                s3_region="us-east-1",
            )

            store = config.create_blob_store()

            assert isinstance(store, S3BlobStore)
            assert store.bucket == "my-bucket"
            assert store.prefix == "my-prefix"

    def test_raises_without_s3_bucket(self, tmp_path: Path):
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="s3",
            artifact_s3_bucket=None,
        )

        with pytest.raises(ValueError, match="requires artifact_s3_bucket"):
            config.create_blob_store()

    def test_creates_gcs_store(self, tmp_path: Path):
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="gcs",
            artifact_gcs_bucket="my-gcs-bucket",
            artifact_gcs_prefix="my-prefix",
            gcs_anonymous=True,
        )

        store = config.create_blob_store()

        assert isinstance(store, GCSBlobStore)
        assert store.bucket == "my-gcs-bucket"
        assert store.prefix == "my-prefix"

    def test_raises_without_gcs_bucket(self, tmp_path: Path):
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="gcs",
            artifact_gcs_bucket=None,
        )

        with pytest.raises(ValueError, match="requires artifact_gcs_bucket"):
            config.create_blob_store()

    def test_creates_azure_store(self, tmp_path: Path):
        pytest.importorskip("azure.storage.blob")
        from vestigo.blob_store import AzureBlobStore

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="azure",
            artifact_azure_container="my-container",
            artifact_azure_prefix="my-prefix",
            azure_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdA==;EndpointSuffix=core.windows.net",
        )

        store = config.create_blob_store()

        assert isinstance(store, AzureBlobStore)
        assert store.container_name == "my-container"
        assert store.prefix == "my-prefix"

    def test_raises_without_azure_container(self, tmp_path: Path):
        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="azure",
            artifact_azure_container=None,
        )

        with pytest.raises(ValueError, match="requires artifact_azure_container"):
            config.create_blob_store()


class TestConfiguredBackendIsActuallyWired:
    """The configured blob backend must reach the artifact store.

    ``ArtifactStore`` falls back to ``LocalBlobStore`` when ``blob_store`` is omitted, so an unwired
    ``STRATA_ARTIFACT_BLOB_BACKEND=s3`` silently writes to local disk.
    """

    def _init(self, config):
        from vestigo.store import reset_artifact_store

        from strata.server import _init_configured_artifact_store

        reset_artifact_store()
        try:
            _init_configured_artifact_store(config)
            from vestigo.store import get_artifact_store

            return get_artifact_store(config.artifact_dir)
        finally:
            pass

    def test_s3_backend_reaches_the_artifact_store(self, tmp_path: Path):
        pytest.importorskip("moto")
        import boto3
        from moto import mock_aws
        from vestigo.store import reset_artifact_store

        with mock_aws():
            boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="wired-bucket")
            config = StrataConfig(
                deployment_mode="personal",
                artifact_dir=tmp_path / "artifacts",
                artifact_blob_backend="s3",
                artifact_s3_bucket="wired-bucket",
                s3_region="us-east-1",
            )
            try:
                store = self._init(config)
                assert isinstance(store.blob_store, S3BlobStore)
                assert store.blob_store.bucket == "wired-bucket"
            finally:
                reset_artifact_store()

    def test_local_backend_is_unchanged(self, tmp_path: Path):
        from vestigo.store import reset_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
        )
        assert config.artifact_blob_backend == "local"
        try:
            store = self._init(config)
            assert isinstance(store.blob_store, LocalBlobStore)
        finally:
            reset_artifact_store()

    def test_misconfigured_backend_raises_rather_than_degrading(self, tmp_path: Path):
        """Falling back to local disk hid the misconfiguration and lost artifacts with the pod, so
        startup fails instead.
        """
        from vestigo.store import reset_artifact_store

        config = StrataConfig(
            deployment_mode="personal",
            artifact_dir=tmp_path / "artifacts",
            artifact_blob_backend="s3",
            artifact_s3_bucket=None,
        )
        try:
            with pytest.raises(ValueError, match="requires artifact_s3_bucket"):
                self._init(config)
        finally:
            reset_artifact_store()

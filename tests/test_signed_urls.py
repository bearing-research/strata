"""Tests for signed URL generation and verification."""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlparse

from strata.transforms.signed_urls import (
    BuildManifest,
    SignedDownloadURL,
    SignedFinalizeURL,
    SignedUploadURL,
    URLSigner,
)

_SECRET = b"test-secret-key-12345678901234"


class TestURLSigner:
    """The signer is keyed by its secret, with no shared process state."""

    def test_same_secret_verifies_different_signer_rejects(self):
        """A URL verifies under any signer holding the same secret, and is
        rejected by a signer with a different secret — the property that makes a
        configured (pinned) secret survive restarts and match across replicas."""
        signer = URLSigner(b"stable-deployment-secret-000001")
        url = signer.generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="art",
            version=1,
            build_id="build-1",
            expiry_seconds=300.0,
        )
        params = parse_qs(urlparse(url.url).query)

        def _verify(s: URLSigner) -> bool:
            return s.verify_download_signature(
                artifact_id=params["artifact_id"][0],
                version=int(params["version"][0]),
                build_id=params["build_id"][0],
                expires_at=float(params["expires_at"][0]),
                signature=params["signature"][0],
            )

        # A different secret (e.g. a random per-process one) rejects it.
        assert _verify(URLSigner(b"a-different-random-process-secret")) is False
        # A fresh signer with the same secret accepts it.
        assert _verify(URLSigner(b"stable-deployment-secret-000001")) is True


class TestSigningSecretConfig:
    """The signing secret is configurable (so it can be pinned across restarts)."""

    def test_field_default_is_none(self):
        from strata.config import StrataConfig

        assert StrataConfig().transform_signing_secret is None

    def test_field_set_explicitly(self):
        from strata.config import StrataConfig

        config = StrataConfig(transform_signing_secret="s3cr3t")
        assert config.transform_signing_secret == "s3cr3t"

    def test_field_from_env(self, monkeypatch):
        from strata.config import StrataConfig

        monkeypatch.setenv("STRATA_TRANSFORM_SIGNING_SECRET", "from-env")
        assert StrataConfig.load().transform_signing_secret == "from-env"


class TestDownloadURL:
    """Tests for download URL generation and verification."""

    def setup_method(self):
        self.signer = URLSigner(_SECRET)

    def test_generate_download_url(self):
        """Generate a signed download URL."""
        url = self.signer.generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        assert isinstance(url, SignedDownloadURL)
        assert url.artifact_id == "test-artifact"
        assert url.version == 1
        assert url.expires_at > time.time()
        assert "artifact_id=test-artifact" in url.url
        assert "version=1" in url.url
        assert "signature=" in url.url

    def test_verify_download_signature_valid(self):
        """Verify a valid download signature."""
        url = self.signer.generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=300.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_download_signature(
            artifact_id=params["artifact_id"][0],
            version=int(params["version"][0]),
            build_id=params["build_id"][0],
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_download_signature_expired(self):
        """Expired signatures are rejected."""
        url = self.signer.generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=-1.0,  # Already expired
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_download_signature(
            artifact_id=params["artifact_id"][0],
            version=int(params["version"][0]),
            build_id=params["build_id"][0],
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_download_signature_tampered(self):
        """Tampered parameters are rejected."""
        url = self.signer.generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=300.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_download_signature(
            artifact_id="different-artifact",  # Tampered!
            version=int(params["version"][0]),
            build_id=params["build_id"][0],
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False


class TestUploadURL:
    """Tests for upload URL generation and verification."""

    def setup_method(self):
        self.signer = URLSigner(_SECRET)

    def test_generate_upload_url(self):
        """Generate a signed upload URL."""
        url = self.signer.generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=600.0,
        )

        assert isinstance(url, SignedUploadURL)
        assert url.build_id == "build-123"
        assert url.max_bytes == 1024 * 1024
        assert url.expires_at > time.time()
        assert "build_id=build-123" in url.url
        assert "signature=" in url.url

    def test_verify_upload_signature_valid(self):
        """Verify a valid upload signature."""
        url = self.signer.generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=600.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=int(params["max_bytes"][0]),
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_upload_signature_expired(self):
        """Expired upload signatures are rejected."""
        url = self.signer.generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=-1.0,  # Already expired
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=int(params["max_bytes"][0]),
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_upload_signature_tampered_max_bytes(self):
        """Tampered max_bytes is rejected."""
        url = self.signer.generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=600.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=10 * 1024 * 1024,  # Tampered!
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False


class TestBuildManifest:
    """Tests for build manifest generation."""

    def setup_method(self):
        self.signer = URLSigner(_SECRET)

    def test_generate_build_manifest(self):
        """Generate a complete build manifest."""
        manifest = self.signer.generate_build_manifest(
            base_url="http://localhost:8765",
            build_id="build-123",
            metadata={"executor": "duckdb_sql@v1", "params": {"sql": "SELECT 1"}},
            input_artifacts=[("input1", 1), ("input2", 3)],
            max_output_bytes=10 * 1024 * 1024,
            url_expiry_seconds=600.0,
        )

        assert isinstance(manifest, BuildManifest)
        assert manifest.build_id == "build-123"
        assert len(manifest.input_urls) == 2
        assert manifest.input_urls[0].artifact_id == "input1"
        assert manifest.input_urls[0].version == 1
        assert manifest.input_urls[1].artifact_id == "input2"
        assert manifest.input_urls[1].version == 3
        assert manifest.output_url.build_id == "build-123"
        assert manifest.output_url.max_bytes == 10 * 1024 * 1024
        assert manifest.finalize_url.startswith(
            "http://localhost:8765/v1/builds/build-123/finalize?"
        )
        assert "signature=" in manifest.finalize_url

    def test_build_manifest_to_dict(self):
        """Build manifest can be serialized to dict."""
        manifest = self.signer.generate_build_manifest(
            base_url="http://localhost:8765",
            build_id="build-123",
            metadata={"executor": "duckdb_sql@v1"},
            input_artifacts=[("input1", 1)],
            max_output_bytes=1024,
            url_expiry_seconds=600.0,
        )

        d = manifest.to_dict()

        assert d["build_id"] == "build-123"
        assert d["metadata"]["executor"] == "duckdb_sql@v1"
        assert len(d["inputs"]) == 1
        assert d["inputs"][0]["artifact_id"] == "input1"
        assert d["inputs"][0]["version"] == 1
        assert "url" in d["inputs"][0]
        assert "expires_at" in d["inputs"][0]
        assert d["output"]["max_bytes"] == 1024
        assert "url" in d["output"]
        assert d["finalize_url"].startswith("http://localhost:8765/v1/builds/build-123/finalize?")

    def test_build_manifest_empty_inputs(self):
        """Build manifest can have no inputs."""
        manifest = self.signer.generate_build_manifest(
            base_url="http://localhost:8765",
            build_id="build-123",
            metadata={"executor": "noop@v1"},
            input_artifacts=[],
            max_output_bytes=1024,
            url_expiry_seconds=600.0,
        )

        assert len(manifest.input_urls) == 0
        d = manifest.to_dict()
        assert d["inputs"] == []


class TestFinalizeURL:
    """Tests for finalize URL generation and verification."""

    def setup_method(self):
        self.signer = URLSigner(_SECRET)

    def test_generate_finalize_url(self):
        """Generate a signed finalize URL."""
        url = self.signer.generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=600.0,
        )

        assert isinstance(url, SignedFinalizeURL)
        assert url.build_id == "build-123"
        assert url.expires_at > time.time()
        assert url.url.startswith("http://localhost:8765/v1/builds/build-123/finalize?")
        assert "signature=" in url.url

    def test_verify_finalize_signature_valid(self):
        """Verify a valid finalize signature."""
        url = self.signer.generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=600.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_finalize_signature(
            build_id="build-123",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_finalize_signature_expired(self):
        """Expired finalize signatures are rejected."""
        url = self.signer.generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=-1.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_finalize_signature(
            build_id="build-123",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_finalize_signature_tampered(self):
        """Tampered finalize parameters are rejected."""
        url = self.signer.generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=600.0,
        )
        params = parse_qs(urlparse(url.url).query)

        valid = self.signer.verify_finalize_signature(
            build_id="different-build",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

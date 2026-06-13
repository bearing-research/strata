"""Tests for signed URL generation and verification."""

from __future__ import annotations

import time

from strata.transforms.signed_urls import (
    BuildManifest,
    SignedDownloadURL,
    SignedFinalizeURL,
    SignedUploadURL,
    generate_build_manifest,
    generate_download_url,
    generate_finalize_url,
    generate_upload_url,
    get_signing_secret,
    reset_signing_secret,
    set_signing_secret,
    verify_download_signature,
    verify_finalize_signature,
    verify_upload_signature,
)


class TestSigningSecret:
    """Tests for signing secret management."""

    def setup_method(self):
        """Reset signing secret before each test."""
        reset_signing_secret()

    def teardown_method(self):
        """Reset signing secret after each test."""
        reset_signing_secret()

    def test_get_signing_secret_generates_secret(self):
        """First call to get_signing_secret generates a secret."""
        secret = get_signing_secret()
        assert secret is not None
        assert len(secret) == 32  # 256 bits

    def test_get_signing_secret_returns_same_secret(self):
        """Multiple calls return the same secret."""
        secret1 = get_signing_secret()
        secret2 = get_signing_secret()
        assert secret1 == secret2

    def test_set_signing_secret(self):
        """Can explicitly set the signing secret."""
        custom_secret = b"my-secret-key-1234567890123456"
        set_signing_secret(custom_secret)
        assert get_signing_secret() == custom_secret

    def test_reset_signing_secret(self):
        """Reset clears the signing secret."""
        _ = get_signing_secret()
        reset_signing_secret()
        # After reset, a new secret is generated
        new_secret = get_signing_secret()
        assert new_secret is not None

    def test_signed_url_survives_restart_when_secret_pinned(self):
        """A signed URL stays valid across a 'restart' iff the secret is re-pinned
        from config (the point of transform_signing_secret) — with a random
        per-process secret it would not."""
        from urllib.parse import parse_qs, urlparse

        secret = b"stable-deployment-secret-000001"
        set_signing_secret(secret)
        url = generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="art",
            version=1,
            build_id="build-1",
            expiry_seconds=300.0,
        )
        params = parse_qs(urlparse(url.url).query)

        def _verify() -> bool:
            return verify_download_signature(
                artifact_id=params["artifact_id"][0],
                version=int(params["version"][0]),
                build_id=params["build_id"][0],
                expires_at=float(params["expires_at"][0]),
                signature=params["signature"][0],
            )

        # "Restart" loses the in-memory secret; a fresh random one would reject it.
        reset_signing_secret()
        set_signing_secret(b"a-different-random-process-secret")
        assert _verify() is False

        # Re-pinning the same configured secret keeps the URL valid.
        reset_signing_secret()
        set_signing_secret(secret)
        assert _verify() is True


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
        """Set a known secret for reproducible tests."""
        set_signing_secret(b"test-secret-key-12345678901234")

    def teardown_method(self):
        reset_signing_secret()

    def test_generate_download_url(self):
        """Generate a signed download URL."""
        url = generate_download_url(
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
        url = generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        # Extract signature from URL
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_download_signature(
            artifact_id=params["artifact_id"][0],
            version=int(params["version"][0]),
            build_id=params["build_id"][0],
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_download_signature_expired(self):
        """Expired signatures are rejected."""
        url = generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=-1.0,  # Already expired
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_download_signature(
            artifact_id=params["artifact_id"][0],
            version=int(params["version"][0]),
            build_id=params["build_id"][0],
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_download_signature_tampered(self):
        """Tampered parameters are rejected."""
        url = generate_download_url(
            base_url="http://localhost:8765",
            artifact_id="test-artifact",
            version=1,
            build_id="build-123",
            expiry_seconds=300.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        # Try with different artifact_id
        valid = verify_download_signature(
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
        set_signing_secret(b"test-secret-key-12345678901234")

    def teardown_method(self):
        reset_signing_secret()

    def test_generate_upload_url(self):
        """Generate a signed upload URL."""
        url = generate_upload_url(
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
        url = generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=600.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=int(params["max_bytes"][0]),
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_upload_signature_expired(self):
        """Expired upload signatures are rejected."""
        url = generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=-1.0,  # Already expired
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=int(params["max_bytes"][0]),
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_upload_signature_tampered_max_bytes(self):
        """Tampered max_bytes is rejected."""
        url = generate_upload_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            max_bytes=1024 * 1024,
            expiry_seconds=600.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        # Try with larger max_bytes
        valid = verify_upload_signature(
            build_id=params["build_id"][0],
            max_bytes=10 * 1024 * 1024,  # Tampered!
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False


class TestBuildManifest:
    """Tests for build manifest generation."""

    def setup_method(self):
        set_signing_secret(b"test-secret-key-12345678901234")

    def teardown_method(self):
        reset_signing_secret()

    def test_generate_build_manifest(self):
        """Generate a complete build manifest."""
        manifest = generate_build_manifest(
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
        manifest = generate_build_manifest(
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
        manifest = generate_build_manifest(
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
        set_signing_secret(b"test-secret-key-12345678901234")

    def teardown_method(self):
        reset_signing_secret()

    def test_generate_finalize_url(self):
        """Generate a signed finalize URL."""
        url = generate_finalize_url(
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
        url = generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=600.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_finalize_signature(
            build_id="build-123",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is True

    def test_verify_finalize_signature_expired(self):
        """Expired finalize signatures are rejected."""
        url = generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=-1.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_finalize_signature(
            build_id="build-123",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

    def test_verify_finalize_signature_tampered(self):
        """Tampered finalize parameters are rejected."""
        url = generate_finalize_url(
            base_url="http://localhost:8765",
            build_id="build-123",
            expiry_seconds=600.0,
        )

        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(url.url)
        params = parse_qs(parsed.query)

        valid = verify_finalize_signature(
            build_id="different-build",
            expires_at=float(params["expires_at"][0]),
            signature=params["signature"][0],
        )

        assert valid is False

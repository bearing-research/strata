"""Build manifests with object-store URLs where the blob store can sign them. Item 12."""

from __future__ import annotations

from strata.transforms.signed_urls import URLSigner

BASE = "https://strata.example"


class _PresigningStore:
    def presign_get(self, artifact_id, version, ttl_seconds):
        return f"https://bucket.s3.example/{artifact_id}@v={version}?sig=get"

    def presign_post(self, artifact_id, version, max_bytes, ttl_seconds):
        return "https://bucket.s3.example", {"key": f"{artifact_id}@v={version}", "policy": "p"}


class _LocalStore:
    def presign_get(self, artifact_id, version, ttl_seconds):
        return None

    def presign_post(self, artifact_id, version, max_bytes, ttl_seconds):
        return None


def _manifest(blob_store):
    return (
        URLSigner(b"s" * 32)
        .generate_build_manifest(
            base_url=BASE,
            build_id="b1",
            metadata={"artifact_id": "out", "version": 3},
            input_artifacts=[("in", 1)],
            max_output_bytes=1024,
            blob_store=blob_store,
        )
        .to_dict()
    )


def test_a_presigning_store_puts_object_store_urls_in_the_manifest():
    manifest = _manifest(_PresigningStore())

    assert manifest["inputs"][0]["url"] == "https://bucket.s3.example/in@v=1?sig=get"
    assert manifest["output"]["url"] == "https://bucket.s3.example"
    assert manifest["output"]["fields"] == {"key": "out@v=3", "policy": "p"}
    assert manifest["output"]["max_bytes"] == 1024
    # Finalize stays a Strata route: it is where the server publishes.
    assert manifest["finalize_url"].startswith(f"{BASE}/v1/builds/b1/finalize")


def test_a_store_that_cannot_sign_keeps_the_strata_routes_and_the_old_wire_shape():
    for blob_store in (None, _LocalStore()):
        manifest = _manifest(blob_store)

        assert manifest["inputs"][0]["url"].startswith(f"{BASE}/v1/artifacts/download")
        assert manifest["output"]["url"].startswith(f"{BASE}/v1/artifacts/upload")
        assert "fields" not in manifest["output"]

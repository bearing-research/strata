"""Presigned object-store URLs against a real S3 implementation. Item 12.

MinIO checks SigV4 signatures and POST policies the way S3 does, so a URL it
accepts is one S3 would; moto accepts anything. Needs Docker.
"""

from __future__ import annotations

import docker
import httpx
import pytest

from strata.blob_store import S3BlobStore


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

from testcontainers.community.minio import MinioContainer  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BUCKET = "strata-presign"


@pytest.fixture(scope="module")
def minio():
    with MinioContainer("quay.io/minio/minio:RELEASE.2024-11-07T00-52-20Z") as container:
        client = container.get_client()
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
        config = container.get_config()
        yield {
            "endpoint": f"http://{config['endpoint']}",
            "access_key": config["access_key"],
            "secret_key": config["secret_key"],
        }


@pytest.fixture
def store(minio):
    return S3BlobStore(
        bucket=BUCKET,
        prefix="artifacts",
        region="us-east-1",
        endpoint_url=minio["endpoint"],
        access_key=minio["access_key"],
        secret_key=minio["secret_key"],
    )


def test_a_presigned_get_reads_the_blob_and_a_tampered_one_does_not(store):
    store.write_blob("fig@1", 1, b"bytes of the figure")

    url = store.presign_get("fig@1", 1, ttl_seconds=60)

    assert url is not None
    assert httpx.get(url).content == b"bytes of the figure"
    tampered = url.replace("X-Amz-Expires=60", "X-Amz-Expires=6000")
    assert httpx.get(tampered).status_code == 403


def test_a_presigned_post_uploads_within_the_bound_and_refuses_beyond_it(store):
    signed = store.presign_post("out", 7, max_bytes=16, ttl_seconds=60)
    assert signed is not None
    url, fields = signed

    small = httpx.post(url, data=fields, files={"file": ("bundle.tar", b"0123456789")})
    assert small.status_code in (200, 201, 204), small.text
    assert store.blob_size("out", 7) == 10

    big = store.presign_post("big", 1, max_bytes=16, ttl_seconds=60)
    assert big is not None
    too_big = httpx.post(big[0], data=big[1], files={"file": ("bundle.tar", b"x" * 64)})
    assert too_big.status_code == 400
    assert "EntityTooLarge" in too_big.text
    assert not store.blob_exists("big", 1)


def test_a_store_without_credentials_does_not_presign(minio, monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    anonymous = S3BlobStore(bucket=BUCKET, endpoint_url=minio["endpoint"], anonymous=True)

    assert anonymous.presign_get("fig", 1, 60) is None
    assert anonymous.presign_post("fig", 1, 16, 60) is None


def test_a_worker_runs_a_job_whose_bytes_never_touch_strata(store, monkeypatch, tmp_path):
    """Input read from and output written to the object store; the only request
    that reaches the server's side is finalize."""
    import http.server
    import json
    import threading

    from fastapi.testclient import TestClient

    from strata.notebook.remote_bundle import unpack_notebook_output_bundle
    from strata.notebook.remote_executor import (
        NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        create_notebook_executor_app,
    )
    from strata.transforms.signed_urls import URLSigner

    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    store.write_blob("zones", 1, b"zone,borough\n1,Queens\n")

    seen: list[str] = []

    class Strata(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        do_GET = do_POST  # noqa: N815

        def log_message(self, *args):
            return None

    strata = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Strata)
    threading.Thread(target=strata.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{strata.server_address[1]}"

    manifest = (
        URLSigner(b"s" * 32)
        .generate_build_manifest(
            base_url=base,
            build_id="b1",
            metadata={
                "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                "artifact_id": "result",
                "version": 1,
                "params": {
                    "source": "rows = len(zones.read_text().splitlines())",
                    "input_specs": {
                        "zones": {"uri": "strata://artifact/zones@v=1", "content_type": "file/path"}
                    },
                    "mounts": [],
                    "env": {},
                },
            },
            input_artifacts=[("zones", 1)],
            max_output_bytes=10 * 1024 * 1024,
            blob_store=store,
        )
        .to_dict()
    )
    manifest["schema_version"] = NOTEBOOK_EXECUTOR_MANIFEST_VERSION

    try:
        with TestClient(create_notebook_executor_app()) as worker:
            response = worker.post("/v1/execute-manifest", json=manifest)
    finally:
        strata.shutdown()

    assert response.status_code == 200, response.text
    assert [path.split("?")[0] for path in seen] == ["/v1/builds/b1/finalize"]
    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(store.read_blob("result", 1))
    result = unpack_notebook_output_bundle(bundle, tmp_path / "out")
    assert result["success"], result.get("error")
    assert json.dumps(result["variables"]["rows"]["preview"]) == "2"

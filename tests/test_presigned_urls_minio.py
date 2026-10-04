"""Presigned object-store URLs against a real S3 implementation.

MinIO checks SigV4 signatures and POST policies as S3 does; moto accepts anything. Needs Docker.
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

from tests.conftest import MINIO_IMAGE  # noqa: E402
from tests.presign_helpers import role_credentials, run_presigned_job  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BUCKET = "strata-presign"


@pytest.fixture(scope="module")
def minio():
    with MinioContainer(MINIO_IMAGE) as container:
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
    """Input and output go through the object store; only finalize reaches the server."""
    assert run_presigned_job(store, monkeypatch, tmp_path) == ["/v1/builds/b1/finalize"]


def test_role_credentials_sign_urls_the_store_accepts(minio, store, monkeypatch, tmp_path):
    """Temporary credentials from a role endpoint, session token and all, as on EC2 or ECS."""
    import boto3

    sts = boto3.client(
        "sts",
        endpoint_url=minio["endpoint"],
        region_name="us-east-1",
        aws_access_key_id=minio["access_key"],
        aws_secret_access_key=minio["secret_key"],
    )
    temporary = sts.assume_role(
        RoleArn="arn:aws:iam::000000000000:role/strata", RoleSessionName="strata"
    )["Credentials"]
    store.write_blob("fig", 1, b"bytes of the figure")

    with role_credentials(
        monkeypatch,
        tmp_path,
        temporary["AccessKeyId"],
        temporary["SecretAccessKey"],
        temporary["SessionToken"],
    ):
        role = S3BlobStore(
            bucket=BUCKET, prefix="artifacts", region="us-east-1", endpoint_url=minio["endpoint"]
        )
        url = role.presign_get("fig", 1, ttl_seconds=60)
        signed = role.presign_post("out", 1, max_bytes=16, ttl_seconds=60)

    assert url is not None and "X-Amz-Security-Token=" in url
    assert httpx.get(url).content == b"bytes of the figure"
    assert signed is not None
    upload = httpx.post(signed[0], data=signed[1], files={"file": ("bundle.tar", b"0123456789")})
    assert upload.status_code in (200, 201, 204), upload.text
    assert store.blob_size("out", 1) == 10

"""Each object-store backend against its emulator, started through Docker.

MinIO checks SigV4 signatures and POST policies as S3 does; fake-gcs-server and Azurite
answer the real SDKs. The unit tests in test_blob_store.py cover the backends' logic over
fakes; these cover the wire. Needs Docker (CI requires it, locally the module skips).
"""

from __future__ import annotations

import os
import socket
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest
from conftest import docker_or_skip
from testcontainers.core.container import DockerContainer
from testcontainers.core.docker_client import DockerClient
from testcontainers.core.wait_strategies import HttpWaitStrategy, LogMessageWaitStrategy

from vestigo.blob_store import AzureBlobStore, GCSBlobStore, S3BlobStore

docker_or_skip()

pytestmark = [pytest.mark.integration, pytest.mark.slow]

MINIO_IMAGE = (
    "cgr.dev/chainguard/minio"
    "@sha256:bd014394a80898e68c149f2311fdf8d5a2c2f3bb2c33b9327ae6d02b4b065ae1"
)
FAKE_GCS_IMAGE = (
    "fsouza/fake-gcs-server@sha256:797ce226d62f947c009dc40246b30cfb456b8473d8241407f9d6f2c04e4d69ef"
)
AZURITE_IMAGE = (
    "mcr.microsoft.com/azure-storage/azurite"
    "@sha256:830430c1da1a2d537e08f3e6764dd1f5ae00cf0346bcaf625b968ec3f0971fd5"
)
BUCKET = "vestigo-blobs"
LARGE = os.urandom(1024) * (9 * 1024)  # 9 MiB: past the S3 multipart and GCS resumable thresholds


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def minio():
    # Not testcontainers' MinioContainer: that imports the minio client package.
    container = (
        DockerContainer(MINIO_IMAGE)
        .with_env("MINIO_ROOT_USER", "minioadmin")
        .with_env("MINIO_ROOT_PASSWORD", "minioadmin")
        .with_command("server /data --address :9000")
        .with_exposed_ports(9000)
        .waiting_for(HttpWaitStrategy(9000, "/minio/health/live"))
    )
    with container:
        endpoint = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(9000)}"
        store = S3BlobStore(
            bucket=BUCKET,
            prefix="artifacts",
            region="us-east-1",
            endpoint_url=endpoint,
            access_key="minioadmin",
            secret_key="minioadmin",
        )
        store._client.create_bucket(Bucket=BUCKET)
        yield store


@pytest.fixture(scope="module")
def fake_gcs():
    # A resumable upload's session URL is built from the external URL, so the port is
    # chosen here and bound explicitly rather than mapped at random.
    host = DockerClient().host()
    port = _free_port()
    container = (
        DockerContainer(FAKE_GCS_IMAGE)
        .with_bind_ports(4443, port)
        .with_command(
            f"-scheme http -port 4443 -external-url http://{host}:{port} -public-host {host}:{port}"
        )
        .waiting_for(LogMessageWaitStrategy("server started"))
    )
    with container:
        store = GCSBlobStore(
            bucket=BUCKET,
            prefix="artifacts",
            anonymous=True,
            endpoint_override=f"http://{host}:{port}",
        )
        store._client.create_bucket(BUCKET)
        yield store


@pytest.fixture(scope="module")
def azurite():
    from testcontainers.community.azurite import AzuriteContainer

    # The SDK speaks a newer API version than the emulator admits to; the check is skipped.
    container = AzuriteContainer(AZURITE_IMAGE).with_command(
        "azurite --skipApiVersionCheck --blobHost 0.0.0.0 --queueHost 0.0.0.0 --tableHost 0.0.0.0"
    )
    with container:
        store = AzureBlobStore(
            account_name=container.account_name,
            container_name=BUCKET,
            prefix="artifacts",
            connection_string=container.get_connection_string(),
        )
        store._client.create_container()
        yield store


@pytest.fixture(params=["minio", "fake_gcs", "azurite"])
def store(request):
    return request.getfixturevalue(request.param)


def test_a_blob_round_trips_and_is_gone_after_delete(store, tmp_path: Path):
    artifact = f"art-{uuid.uuid4().hex[:8]}"

    assert store.read_blob(artifact, 1) is None
    assert store.blob_exists(artifact, 1) is False
    assert store.blob_size(artifact, 1) is None
    assert store.delete_blob(artifact, 1) is False

    store.write_blob(artifact, 1, b"streamed")
    assert store.read_blob(artifact, 1) == b"streamed"
    assert store.blob_exists(artifact, 1) is True
    assert store.blob_size(artifact, 1) == len(b"streamed")

    source = tmp_path / "incoming.bin"
    source.write_bytes(b"published")
    store.publish_blob_from_path(artifact, 1, source)
    assert store.read_blob(artifact, 1) == b"published"

    assert store.delete_blob(artifact, 1) is True
    assert store.blob_exists(artifact, 1) is False
    assert store.delete_blob(artifact, 1) is False


def test_a_large_blob_round_trips_in_parts(store, tmp_path: Path):
    artifact = f"big-{uuid.uuid4().hex[:8]}"
    source = tmp_path / "big.bin"
    source.write_bytes(LARGE)

    store.publish_blob_from_path(artifact, 1, source)

    assert store.blob_size(artifact, 1) == len(LARGE)
    reader = store.open_blob_reader(artifact, 1)
    assert reader is not None
    with reader as f:
        chunks = iter(lambda: f.read(1 << 20), b"")
        assert b"".join(chunks) == LARGE


def test_a_presigned_get_reads_the_blob_from_minio(minio):
    minio.write_blob("fig", 1, b"bytes of the figure")

    url = minio.presign_get("fig", 1, ttl_seconds=60)

    assert url is not None
    with urllib.request.urlopen(url) as response:
        assert response.read() == b"bytes of the figure"


def test_a_presigned_post_uploads_within_the_bound_and_minio_refuses_beyond_it(minio):
    signed = minio.presign_post("out", 1, max_bytes=16, ttl_seconds=60)
    assert signed is not None
    url, fields = signed

    status, _ = _post_form(url, fields, b"0123456789")
    assert status == 204
    assert minio.read_blob("out", 1) == b"0123456789"

    status, body = _post_form(url, fields, b"x" * 17)
    assert status == 400
    assert b"EntityTooLarge" in body


def _post_form(url: str, fields: dict[str, str], payload: bytes) -> tuple[int, bytes]:
    """POST ``fields`` and ``payload`` as the ``file`` part of a multipart form."""
    boundary = uuid.uuid4().hex
    body = b""
    for name, value in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="f"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    body += payload + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()

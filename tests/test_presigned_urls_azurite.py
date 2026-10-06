"""Presigned Azure Blob URLs against Azurite, which checks SAS signatures as Azure does.

Needs Docker.
"""

from __future__ import annotations

import docker
import httpx
import pytest


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

from testcontainers.community.azurite import AzuriteContainer  # noqa: E402

from strata.blob_store import AzureBlobStore  # noqa: E402
from strata.config import StrataConfig  # noqa: E402
from tests.conftest import start_container_or_skip  # noqa: E402
from tests.presign_helpers import run_presigned_job  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.slow]

CONTAINER = "strata-presign"


@pytest.fixture(scope="module")
def azurite():
    # :latest because older Azurite rejects the installed SDK's API version.
    container = start_container_or_skip(
        AzuriteContainer("mcr.microsoft.com/azure-storage/azurite:latest"), label="Azurite"
    )
    try:
        from azure.storage.blob import BlobServiceClient

        connection_string = container.get_connection_string()
        BlobServiceClient.from_connection_string(connection_string).create_container(CONTAINER)
        yield container
    finally:
        container.stop()


@pytest.fixture
def store(azurite):
    return AzureBlobStore(
        account_name="devstoreaccount1",
        container_name=CONTAINER,
        connection_string=azurite.get_connection_string(),
    )


def test_a_presigned_get_reads_the_blob_and_a_tampered_one_does_not(store):
    store.write_blob("fig@1", 1, b"bytes of the figure")

    url = store.presign_get("fig@1", 1, ttl_seconds=60)

    assert url is not None
    assert httpx.get(url).content == b"bytes of the figure"
    assert "sp=r&" in url
    assert httpx.get(url.replace("sp=r&", "sp=rw&")).status_code == 403


def test_a_presigned_put_uploads_and_a_read_url_cannot(store):
    signed = store.presign_put("out", 7, ttl_seconds=60)
    assert signed is not None
    url, headers = signed

    response = httpx.put(url, content=b"0123456789", headers=headers)

    assert response.status_code == 201, response.text
    assert store.blob_size("out", 7) == 10
    read_only = store.presign_get("other", 1, ttl_seconds=60)
    assert read_only is not None
    assert httpx.put(read_only, content=b"x", headers=headers).status_code == 403
    assert not store.blob_exists("other", 1)


def test_a_worker_runs_a_job_whose_bytes_never_touch_strata(store, monkeypatch, tmp_path):
    """Input and output go through Azure; only finalize reaches the server."""
    assert run_presigned_job(store, monkeypatch, tmp_path) == ["/v1/builds/b1/finalize"]


def test_the_store_built_from_the_documented_endpoint_setting_reaches_azurite(
    azurite, monkeypatch, tmp_path
):
    """``STRATA_AZURE_ENDPOINT_URL`` is the blob host; the account is its first path
    segment, as the lake path and the doc read it. No connection string is set."""
    host = f"http://{azurite.get_container_host_ip()}:{azurite.get_exposed_port(10000)}"
    monkeypatch.setenv("STRATA_AZURE_ACCOUNT_NAME", azurite.account_name)
    monkeypatch.setenv("STRATA_AZURE_ACCOUNT_KEY", azurite.account_key)
    monkeypatch.setenv("STRATA_AZURE_ENDPOINT_URL", host)
    monkeypatch.delenv("STRATA_AZURE_CONNECTION_STRING", raising=False)
    config = StrataConfig(cache_dir=tmp_path / "cache", metadata_db=tmp_path / "meta.sqlite")
    assert config.azure_connection_string is None

    store = AzureBlobStore.from_config(config, container_name=CONTAINER, prefix="by-endpoint")

    store.write_blob("blob", 3, b"reached through the host form")
    assert store.blob_exists("blob", 3)
    with store.open_blob_reader("blob", 3) as reader:
        assert reader.read() == b"reached through the host form"
    assert "by-endpoint/blob@v=3.arrow" in list(store._client.list_blob_names())
    url = store.presign_get("blob", 3, ttl_seconds=60)
    assert url is not None and url.startswith(f"{host}/{azurite.account_name}/")
    assert httpx.get(url).content == b"reached through the host form"

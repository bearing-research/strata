"""Artifact write routes refuse a body over ``max_upload_bytes`` with 413.

The count is kept as the body arrives, so a chunked body with no Content-Length is held to it too,
and nothing the refused upload wrote is left behind.
"""

from __future__ import annotations

import hashlib
import json
import tempfile

import httpx
import pyarrow as pa
import pytest

from strata.artifact_store import ArtifactStore, TransformSpec
from tests.conftest import run_server_with_context, table_to_ipc_bytes

CAP = 4096
OVERSIZED = b"x" * (CAP * 4)
DIGEST = hashlib.sha256(OVERSIZED).hexdigest()
PROVENANCE = "a" * 64


@pytest.fixture
def served(tmp_path, monkeypatch):
    # The server runs in this process, so its temp files land here where the test can see them.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(
        tmp_path / "cache", artifact_dir, "personal", max_upload_bytes=CAP
    ) as ctx:
        yield ctx.base_url, artifact_dir, scratch


def _multipart(metadata: dict) -> dict:
    return {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.bin", OVERSIZED, "application/octet-stream"),
    }


def _chunked(method: str, url: str, files: dict) -> httpx.Response:
    """Send a multipart body as one chunk with no Content-Length."""
    built = httpx.Request(method, url, files=files)
    body = built.read()
    headers = {"content-type": built.headers["content-type"]}
    return httpx.request(method, url, content=iter([body]), headers=headers, timeout=10)


_PUT_METADATA = {"inputs": [], "transform": {"executor": "local@v1", "params": {}}}
_IMPORT_METADATA = {
    "id": "big",
    "version": 1,
    "provenance_hash": PROVENANCE,
    "created_at": 1.0,
}
_ROUTES = [
    pytest.param("PUT", "/v1/artifacts", _PUT_METADATA, id="put"),
    pytest.param(
        "PUT",
        f"/v1/artifacts/by-provenance/{PROVENANCE}",
        {"content_type": "pickle/object"},
        id="by-provenance",
    ),
    pytest.param("POST", "/v1/artifacts/import", _IMPORT_METADATA, id="import"),
]


def _nothing_stored(artifact_dir) -> bool:
    store = ArtifactStore(artifact_dir)
    return store.find_by_provenance(PROVENANCE) is None and store.get_artifact("big", 1) is None


@pytest.mark.parametrize(("method", "path", "metadata"), _ROUTES)
def test_an_oversized_multipart_upload_is_a_413(served, method, path, metadata):
    base, artifact_dir, scratch = served

    response = httpx.request(method, f"{base}{path}", files=_multipart(metadata), timeout=10)

    assert response.status_code == 413, response.text
    assert _nothing_stored(artifact_dir)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(("method", "path", "metadata"), _ROUTES)
def test_an_oversized_chunked_multipart_upload_is_a_413(served, method, path, metadata):
    """No Content-Length to check up front: the count while parsing refuses it."""
    base, artifact_dir, scratch = served

    response = _chunked(method, f"{base}{path}", _multipart(metadata))

    assert response.status_code == 413, response.text
    assert _nothing_stored(artifact_dir)
    assert list(scratch.iterdir()) == []


def test_an_oversized_chunked_blob_stage_is_a_413_and_leaves_nothing(served):
    base, artifact_dir, scratch = served

    response = httpx.put(
        f"{base}/v1/artifacts/import/blobs/{DIGEST}",
        content=iter([OVERSIZED[:CAP], OVERSIZED[CAP:]]),
        timeout=10,
    )

    assert response.status_code == 413, response.text
    assert ArtifactStore(artifact_dir).open_staged_import(None, DIGEST) is None
    assert list(scratch.iterdir()) == []


def test_an_oversized_json_put_is_a_413(served):
    base, artifact_dir, _ = served
    body = {**_PUT_METADATA, "data": {"x": ["y" * CAP]}}

    response = httpx.put(f"{base}/v1/artifacts", json=body, timeout=10)

    assert response.status_code == 413, response.text


def test_an_oversized_personal_upload_is_a_413(served):
    base, artifact_dir, scratch = served
    store = ArtifactStore(artifact_dir)
    version = store.create_artifact(
        artifact_id="building",
        provenance_hash=PROVENANCE,
        transform_spec=TransformSpec(executor="local@v1", params={}, inputs=[]),
    )

    response = httpx.post(
        f"{base}/v1/artifacts/upload/building/v/{version}",
        content=iter([OVERSIZED]),
        timeout=10,
    )

    assert response.status_code == 413, response.text
    assert not store.blob_exists("building", version)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize(("method", "path", "metadata"), _ROUTES)
def test_an_upload_under_the_cap_is_stored(served, method, path, metadata):
    base, _, _ = served
    data = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
    files = {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", data, "application/vnd.apache.arrow.stream"),
    }

    response = _chunked(method, f"{base}{path}", files)

    assert response.status_code == 200, response.text

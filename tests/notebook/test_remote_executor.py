"""Tests for the notebook HTTP executor."""

from __future__ import annotations

import io
import json
import tarfile
import uuid
from pathlib import Path

import httpx

from strata.artifact_store import TransformSpec
from strata.notebook.remote_bundle import SCHEMA_VERSION, unpack_notebook_output_bundle
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    NOTEBOOK_EXECUTOR_TRANSFORM_REF,
)
from strata.server import get_state
from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION


def test_remote_executor_health(notebook_executor_server):
    """The notebook executor should expose its protocol and health."""
    response = httpx.get(f"{notebook_executor_server['base_url']}/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert EXECUTOR_PROTOCOL_VERSION in payload["capabilities"]["protocol_versions"]
    assert NOTEBOOK_EXECUTOR_TRANSFORM_REF in payload["capabilities"]["transform_refs"]
    assert (
        payload["capabilities"]["features"]["notebook_protocol_version"]
        == NOTEBOOK_EXECUTOR_PROTOCOL_VERSION
    )
    assert payload["capabilities"]["features"]["pull_model"] is True


def test_remote_executor_executes_simple_cell_via_notebook_endpoint(
    tmp_path,
    notebook_executor_server,
):
    """The compatibility notebook endpoint should return a notebook result bundle."""
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 1 + 1",
        "timeout_seconds": 10.0,
        "inputs": {},
        "mounts": [],
        "env": {},
    }

    response = httpx.post(
        notebook_executor_server["notebook_execute_url"],
        files={
            "metadata": (
                "metadata.json",
                json.dumps(metadata).encode("utf-8"),
                "application/json",
            ),
        },
        timeout=10.0,
    )

    assert response.status_code == 200
    assert (
        response.headers["X-Strata-Notebook-Executor-Protocol"]
        == NOTEBOOK_EXECUTOR_PROTOCOL_VERSION
    )

    bundle_path = tmp_path / "bundle.tar"
    bundle_path.write_bytes(response.content)
    unpacked = unpack_notebook_output_bundle(bundle_path, tmp_path / "unpacked")

    assert unpacked["success"] is True
    assert unpacked["variables"]["x"]["preview"] == 2
    assert (tmp_path / "unpacked" / "x.json").exists()


def test_remote_executor_executes_simple_cell_via_executor_v1(
    tmp_path,
    notebook_executor_server,
):
    """The standard executor v1 endpoint should support notebook_cell@v1."""
    metadata = {
        "protocol_version": EXECUTOR_PROTOCOL_VERSION,
        "build_id": "b-test",
        "tenant": None,
        "principal": None,
        "provenance_hash": "abc123",
        "transform": {
            "ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "code_hash": "def456",
            "params": {
                "source": "x = 1 + 1",
                "timeout_seconds": 10.0,
                "mounts": [],
                "env": {},
            },
        },
        "inputs": [],
    }

    response = httpx.post(
        notebook_executor_server["execute_url"],
        files={
            "metadata": (
                "metadata.json",
                json.dumps(metadata).encode("utf-8"),
                "application/json",
            ),
        },
        headers={EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION},
        timeout=10.0,
    )

    assert response.status_code == 200
    assert response.headers[EXECUTOR_PROTOCOL_HEADER] == EXECUTOR_PROTOCOL_VERSION
    assert (
        response.headers["X-Strata-Notebook-Executor-Protocol"]
        == NOTEBOOK_EXECUTOR_PROTOCOL_VERSION
    )

    bundle_path = tmp_path / "bundle-v1.tar"
    bundle_path.write_bytes(response.content)
    unpacked = unpack_notebook_output_bundle(bundle_path, tmp_path / "unpacked-v1")

    assert unpacked["success"] is True
    assert unpacked["variables"]["x"]["preview"] == 2


def test_remote_executor_rejects_file_mounts(notebook_executor_server):
    """Remote execution should reject notebook-declared file:// mounts."""
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 1",
        "timeout_seconds": 10.0,
        "inputs": {},
        "mounts": [
            {
                "name": "raw_data",
                "uri": "file:///tmp/data",
                "mode": "ro",
                "pin": None,
            }
        ],
        "env": {},
    }

    response = httpx.post(
        notebook_executor_server["notebook_execute_url"],
        files={
            "metadata": (
                "metadata.json",
                json.dumps(metadata).encode("utf-8"),
                "application/json",
            ),
        },
        timeout=10.0,
    )

    assert response.status_code == 400
    assert "does not support file:// mount" in response.json()["detail"]


def test_remote_executor_executes_signed_manifest_build(
    notebook_executor_server,
    notebook_build_server,
):
    """The manifest endpoint should execute, upload, and finalize a notebook bundle."""
    artifact_store = notebook_build_server["artifact_store"]
    build_store = notebook_build_server["build_store"]
    base_url = notebook_build_server["base_url"]

    input_artifact_id = f"input-{uuid.uuid4().hex[:8]}"
    input_version = artifact_store.create_artifact(
        artifact_id=input_artifact_id,
        provenance_hash=f"prov-{input_artifact_id}",
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": "json/object"},
            inputs=[],
        ),
    )
    input_bytes = json.dumps(2).encode("utf-8")
    artifact_store.write_blob(input_artifact_id, input_version, input_bytes)
    artifact_store.finalize_artifact(
        input_artifact_id,
        input_version,
        schema_json="",
        row_count=0,
        byte_size=len(input_bytes),
    )

    output_artifact_id = f"output-{uuid.uuid4().hex[:8]}"
    output_version = artifact_store.create_artifact(
        artifact_id=output_artifact_id,
        provenance_hash=f"prov-{output_artifact_id}",
        transform_spec=TransformSpec(
            executor=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            params={},
            inputs=[f"strata://artifact/{input_artifact_id}@v={input_version}"],
        ),
    )

    build_id = f"build-{uuid.uuid4().hex[:8]}"
    params = {
        "source": "result = value + 1",
        "timeout_seconds": 10.0,
        "mounts": [],
        "env": {},
        "input_specs": {
            "value": {
                "uri": f"strata://artifact/{input_artifact_id}@v={input_version}",
                "content_type": "json/object",
            }
        },
        "output_format": "notebook-output-bundle@v1",
        "_dispatch_mode": "external",
    }
    build_store.create_build(
        build_id=build_id,
        artifact_id=output_artifact_id,
        version=output_version,
        executor_ref=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        executor_url=notebook_executor_server["execute_url"],
        input_uris=[f"strata://artifact/{input_artifact_id}@v={input_version}"],
        params=params,
    )
    build_store.start_build(build_id)

    manifest = get_state().url_signer.generate_build_manifest(
        base_url=base_url,
        build_id=build_id,
        metadata={
            "build_id": build_id,
            "artifact_id": output_artifact_id,
            "version": output_version,
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": params,
        },
        input_artifacts=[(input_artifact_id, input_version)],
        max_output_bytes=notebook_build_server["config"].max_transform_output_bytes,
        url_expiry_seconds=notebook_build_server["config"].signed_url_expiry_seconds,
    )

    response = httpx.post(
        notebook_executor_server["manifest_execute_url"],
        json=manifest.to_dict(),
        timeout=20.0,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["protocol_version"] == NOTEBOOK_EXECUTOR_MANIFEST_VERSION

    build = build_store.get_build(build_id)
    assert build is not None
    assert build.state == "ready"

    blob = artifact_store.read_blob(output_artifact_id, output_version)
    assert blob is not None
    bundle_path = notebook_build_server["config"].cache_dir / "manifest-bundle.tar"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(blob)
    unpacked = unpack_notebook_output_bundle(bundle_path, bundle_path.parent / "manifest-bundle")
    assert unpacked["success"] is True
    assert unpacked["variables"]["result"]["preview"] == 3


def test_remote_executor_manifest_reports_finalize_failure(
    notebook_executor_server,
    notebook_build_server,
):
    """Manifest execution should surface finalize failures as 502s."""
    artifact_store = notebook_build_server["artifact_store"]
    build_store = notebook_build_server["build_store"]
    base_url = notebook_build_server["base_url"]

    input_artifact_id = f"input-{uuid.uuid4().hex[:8]}"
    input_version = artifact_store.create_artifact(
        artifact_id=input_artifact_id,
        provenance_hash=f"prov-{input_artifact_id}",
        transform_spec=TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": "json/object"},
            inputs=[],
        ),
    )
    input_bytes = json.dumps(2).encode("utf-8")
    artifact_store.write_blob(input_artifact_id, input_version, input_bytes)
    artifact_store.finalize_artifact(
        input_artifact_id,
        input_version,
        schema_json="",
        row_count=0,
        byte_size=len(input_bytes),
    )

    output_artifact_id = f"output-{uuid.uuid4().hex[:8]}"
    output_version = artifact_store.create_artifact(
        artifact_id=output_artifact_id,
        provenance_hash=f"prov-{output_artifact_id}",
        transform_spec=TransformSpec(
            executor=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            params={},
            inputs=[f"strata://artifact/{input_artifact_id}@v={input_version}"],
        ),
    )

    build_id = f"build-{uuid.uuid4().hex[:8]}"
    params = {
        "source": "result = value + 1",
        "timeout_seconds": 10.0,
        "mounts": [],
        "env": {},
        "input_specs": {
            "value": {
                "uri": f"strata://artifact/{input_artifact_id}@v={input_version}",
                "content_type": "json/object",
            }
        },
        "output_format": "notebook-output-bundle@v1",
        "_dispatch_mode": "external",
    }
    build_store.create_build(
        build_id=build_id,
        artifact_id=output_artifact_id,
        version=output_version,
        executor_ref=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        executor_url=notebook_executor_server["execute_url"],
        input_uris=[f"strata://artifact/{input_artifact_id}@v={input_version}"],
        params=params,
    )
    build_store.start_build(build_id)

    manifest = (
        get_state()
        .url_signer.generate_build_manifest(
            base_url=base_url,
            build_id=build_id,
            metadata={
                "build_id": build_id,
                "artifact_id": output_artifact_id,
                "version": output_version,
                "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                "params": params,
            },
            input_artifacts=[(input_artifact_id, input_version)],
            max_output_bytes=notebook_build_server["config"].max_transform_output_bytes,
            url_expiry_seconds=notebook_build_server["config"].signed_url_expiry_seconds,
        )
        .to_dict()
    )
    manifest["finalize_url"] = f"{base_url}/v1/builds/{build_id}/missing-finalize"

    response = httpx.post(
        notebook_executor_server["manifest_execute_url"],
        json=manifest,
        timeout=20.0,
    )

    assert response.status_code == 502
    assert "Failed to finalize notebook bundle build" in response.json()["detail"]


def test_service_finalize_rejects_incomplete_notebook_bundle(notebook_build_server):
    """Finalize should reject malformed notebook bundles and fail the artifact."""
    artifact_store = notebook_build_server["artifact_store"]
    build_store = notebook_build_server["build_store"]
    base_url = notebook_build_server["base_url"]

    artifact_id = f"invalid-bundle-{uuid.uuid4().hex[:8]}"
    version = artifact_store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=f"prov-{artifact_id}",
        transform_spec=TransformSpec(
            executor=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            params={},
            inputs=[],
        ),
    )
    build_id = f"build-{uuid.uuid4().hex[:8]}"
    build_store.create_build(
        build_id=build_id,
        artifact_id=artifact_id,
        version=version,
        executor_ref=NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        executor_url="http://executor.invalid/v1/execute",
        input_uris=[],
        params={"output_format": "notebook-output-bundle@v1"},
    )
    build_store.start_build(build_id)

    bundle_bytes = io.BytesIO()
    with tarfile.open(fileobj=bundle_bytes, mode="w") as tar:
        manifest_bytes = json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "success": True,
                "variables": {
                    "result": {
                        "content_type": "json/object",
                        "file": "files/result.json",
                    }
                },
                "stdout_file": "stdout.txt",
                "stderr_file": "stderr.txt",
            }
        ).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        tar.addfile(info, io.BytesIO(manifest_bytes))
        for name in ("stdout.txt", "stderr.txt"):
            payload = b""
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    artifact_store.write_blob(artifact_id, version, bundle_bytes.getvalue())

    response = httpx.post(
        f"{base_url}/v1/builds/{build_id}/finalize",
        json={"output_format": "notebook-output-bundle@v1"},
        timeout=10.0,
    )

    assert response.status_code == 400
    assert "Invalid notebook output bundle" in response.json()["detail"]

    build = build_store.get_build(build_id)
    assert build is not None
    assert build.state == "failed"

    artifact = artifact_store.get_artifact(artifact_id, version)
    assert artifact is not None
    assert artifact.state == "failed"


def test_remote_executor_cleans_up_tempdir_after_streamed_response(
    tmp_path,
    notebook_executor_server,
    monkeypatch,
):
    """FileResponse + BackgroundTask should remove the executor tempdir after streaming."""
    import tempfile as stdlib_tempfile

    from strata.notebook import remote_executor as remote_executor_module

    recorded_paths: list[str] = []
    real_mkdtemp = stdlib_tempfile.mkdtemp

    def _recording_mkdtemp(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        if kwargs.get("prefix", "").startswith("strata_notebook_executor_"):
            recorded_paths.append(path)
        return path

    monkeypatch.setattr(remote_executor_module.tempfile, "mkdtemp", _recording_mkdtemp)

    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 42",
        "timeout_seconds": 10.0,
        "inputs": {},
        "mounts": [],
        "env": {},
    }

    with httpx.Client(timeout=10.0) as client:
        with client.stream(
            "POST",
            notebook_executor_server["notebook_execute_url"],
            files={
                "metadata": (
                    "metadata.json",
                    json.dumps(metadata).encode("utf-8"),
                    "application/json",
                ),
            },
        ) as response:
            assert response.status_code == 200
            bundle_path = tmp_path / "cleanup-bundle.tar"
            with open(bundle_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)

    unpacked = unpack_notebook_output_bundle(bundle_path, tmp_path / "cleanup-unpacked")
    assert unpacked["success"] is True

    import time as _time

    deadline = _time.time() + 2.0
    while _time.time() < deadline:
        if all(not Path(p).exists() for p in recorded_paths):
            break
        _time.sleep(0.05)

    assert recorded_paths, "Executor did not create a tempdir"
    for path in recorded_paths:
        assert not Path(path).exists(), f"Executor tempdir {path} was not cleaned up"


def test_remote_executor_error_detail_survives_streaming_client(
    notebook_executor_server,
):
    """Streaming clients that aread() on non-200 should see the executor's JSON detail."""
    import asyncio

    async def _run():
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream(
                "POST",
                notebook_executor_server["notebook_execute_url"],
                files={
                    "metadata": (
                        "metadata.json",
                        json.dumps(
                            {
                                "protocol_version": "wrong-version",
                                "source": "x = 1",
                                "timeout_seconds": 10.0,
                                "inputs": {},
                                "mounts": [],
                                "env": {},
                            }
                        ).encode("utf-8"),
                        "application/json",
                    ),
                },
            ) as response:
                assert response.status_code == 400
                await response.aread()
                payload = response.json()
                assert "Unsupported protocol version" in payload["detail"]

    asyncio.run(_run())


def test_remote_executor_round_trips_large_input_bundle(
    tmp_path,
    notebook_executor_server,
):
    """Large inputs should flow through the streaming transport without corruption."""
    import hashlib
    import pickle

    payload_size = 8 * 1024 * 1024
    payload_bytes = (b"\xab\xcd\xef\x01" * (payload_size // 4))[:payload_size]
    expected_digest = hashlib.sha256(payload_bytes).hexdigest()
    pickled_payload = pickle.dumps(payload_bytes)

    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": ("import hashlib\ndigest = hashlib.sha256(blob).hexdigest()\nsize = len(blob)\n"),
        "timeout_seconds": 30.0,
        "inputs": {
            "blob": {
                "content_type": "pickle/object",
                "file": "blob.pickle",
            }
        },
        "mounts": [],
        "env": {},
    }

    response = httpx.post(
        notebook_executor_server["notebook_execute_url"],
        files={
            "metadata": (
                "metadata.json",
                json.dumps(metadata).encode("utf-8"),
                "application/json",
            ),
            "blob.pickle": ("blob.pickle", pickled_payload, "application/octet-stream"),
        },
        timeout=60.0,
    )

    assert response.status_code == 200
    bundle_path = tmp_path / "large-bundle.tar"
    bundle_path.write_bytes(response.content)
    unpacked = unpack_notebook_output_bundle(bundle_path, tmp_path / "large-unpacked")

    assert unpacked["success"] is True, unpacked.get("error") or unpacked.get("traceback")
    assert unpacked["variables"]["digest"]["preview"] == expected_digest
    assert unpacked["variables"]["size"]["preview"] == payload_size


def test_manifest_rejects_input_url_with_disallowed_scheme(notebook_executor_server):
    """A compromised orchestrator could put file:// or ftp:// in a manifest;
    the worker must reject up front before httpx fetches anything.
    """
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [
            {
                "artifact_id": "a",
                "version": 1,
                "url": "file:///etc/passwd",
            }
        ],
        "output": {"url": "https://example.invalid/upload"},
        "finalize_url": "https://example.invalid/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    assert "disallowed scheme" in response.json()["detail"]
    assert "'file'" in response.json()["detail"]


def test_manifest_rejects_upload_url_with_disallowed_scheme(notebook_executor_server):
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [],
        "output": {"url": "javascript:alert(1)"},
        "finalize_url": "https://example.invalid/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    assert "output.url" in response.json()["detail"]


def test_manifest_rejects_finalize_url_with_disallowed_scheme(notebook_executor_server):
    # example.com is the only sanctioned "real but uninteresting" host
    # that resolves to a public IP; the validator now requires hosts
    # to resolve to a non-private address, so example.invalid would
    # fail at host resolution before reaching the scheme check.
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [],
        "output": {"url": "https://example.com/upload"},
        "finalize_url": "ftp://example.com/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    assert "finalize_url" in response.json()["detail"]


def test_manifest_rejects_url_resolving_to_loopback(notebook_executor_server, monkeypatch):
    """SSRF guard: a hostname like ``localhost`` resolves to 127.0.0.1 / ::1
    and would let a compromised orchestrator coerce the worker into
    fetching internal services on the worker's loopback interface.

    The fixture sets STRATA_WORKER_ALLOW_LOCAL_HOSTS so the build-server
    tests pass against 127.0.0.1; delete it here to exercise the
    production SSRF guard.
    """
    monkeypatch.delenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", raising=False)
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [
            {
                "artifact_id": "a",
                "version": 1,
                "url": "http://localhost/secret",
            }
        ],
        "output": {"url": "https://example.com/upload"},
        "finalize_url": "https://example.com/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "non-routable" in detail
    assert "localhost" in detail


def test_manifest_rejects_url_with_metadata_ip(notebook_executor_server, monkeypatch):
    """SSRF guard: 169.254.169.254 is the AWS / Azure / GCP cloud-metadata
    endpoint and must never be reachable from worker code paths.
    """
    monkeypatch.delenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", raising=False)
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [],
        "output": {"url": "http://169.254.169.254/latest/meta-data/"},
        "finalize_url": "https://example.com/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "non-routable" in detail


def test_manifest_rejects_url_with_private_ip(notebook_executor_server, monkeypatch):
    """SSRF guard: RFC1918 private addresses (10/8, 172.16/12, 192.168/16)
    typically belong to the worker's VPC and host internal services
    that should not be reachable from manifest URLs.
    """
    monkeypatch.delenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", raising=False)
    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
        },
        "inputs": [],
        "output": {"url": "http://10.0.0.1/upload"},
        "finalize_url": "https://example.com/finalize",
    }
    response = httpx.post(
        notebook_executor_server["manifest_execute_url"], json=manifest, timeout=10.0
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "non-routable" in detail

"""Shared pieces of the presigned-URL tests: a role credential endpoint and a worker job."""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def _serve(handler: type[http.server.BaseHTTPRequestHandler]) -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def role_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, access_key: str, secret_key: str, token: str
) -> Iterator[list[str]]:
    """Serve temporary credentials the way an ECS task role does, with no other source.

    Yields the list of requests the endpoint saw, so a test can count resolutions.
    """
    seen: list[str] = []
    body = json.dumps(
        {
            "AccessKeyId": access_key,
            "SecretAccessKey": secret_key,
            "Token": token,
            "Expiration": "2099-01-01T00:00:00Z",
        }
    ).encode()

    class Role(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return None

    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    with _serve(Role) as base:
        monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", f"{base}/role")
        yield seen


def run_presigned_job(store, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Run one worker job whose manifest ``store`` presigns; return the Strata paths it hit.

    Strata is a stub that answers every request, so any byte transfer through it would show up
    in the returned paths. Asserts the job succeeded and its output reads back from ``store``.
    """
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
        do_PUT = do_POST  # noqa: N815

        def log_message(self, *args):
            return None

    with _serve(Strata) as base:
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
                            "zones": {
                                "uri": "strata://artifact/zones@v=1",
                                "content_type": "file/path",
                            }
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
        with TestClient(create_notebook_executor_app()) as worker:
            response = worker.post("/v1/execute-manifest", json=manifest)

    assert response.status_code == 200, response.text
    bundle = tmp_path / "bundle.tar"
    bundle.write_bytes(store.read_blob("result", 1))
    result = unpack_notebook_output_bundle(bundle, tmp_path / "out")
    assert result["success"], result.get("error")
    assert json.dumps(result["variables"]["rows"]["preview"]) == "2"
    return [path.split("?")[0] for path in seen]

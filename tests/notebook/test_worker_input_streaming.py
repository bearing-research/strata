"""A worker writes signed-URL inputs to disk as they arrive. Item 19.

The download used to collect each input in a ``bytearray`` before writing it,
so ``STRATA_WORKER_MAX_INPUT_BYTES`` was really a memory bound: a 2 GiB input
on a 4 GiB machine failed.
"""

from __future__ import annotations

import hashlib
import http.server
import threading
import tracemalloc

import pytest
from fastapi.testclient import TestClient

from strata.notebook.remote_bundle import unpack_notebook_output_bundle
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
    NOTEBOOK_EXECUTOR_TRANSFORM_REF,
    create_notebook_executor_app,
)

_BLOCK = bytes(range(256)) * 256  # 64 KiB


class _Store:
    """Serves one close-delimited input of ``size`` bytes (no Content-Length,
    so only the running count can enforce the cap) and accepts the upload and
    finalize calls."""

    def __init__(self, size: int):
        self.size = size
        self.sent = 0
        self.uploads: list[bytes] = []
        store = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.end_headers()
                try:
                    while store.sent < store.size:
                        block = _BLOCK[: store.size - store.sent]
                        self.wfile.write(block)
                        store.sent += len(block)
                except (BrokenPipeError, ConnectionResetError):
                    return

            def do_POST(self):  # noqa: N802
                received = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path == "/upload":
                    store.uploads.append(received)
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def digest(self) -> str:
        whole, rest = divmod(self.size, len(_BLOCK))
        sha = hashlib.sha256()
        for _ in range(whole):
            sha.update(_BLOCK)
        sha.update(_BLOCK[:rest])
        return sha.hexdigest()


def _manifest(store: _Store, source: str) -> dict:
    uri = "strata://artifact/big@v=1"
    return {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {
                "source": source,
                "timeout_seconds": 120,
                "input_specs": {"data": {"uri": uri, "content_type": "file/path"}},
                "mounts": [],
                "env": {},
            },
        },
        "inputs": [{"artifact_id": "big", "version": 1, "url": f"{store.base}/input"}],
        "output": {"url": f"{store.base}/upload"},
        "finalize_url": f"{store.base}/finalize",
    }


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    with TestClient(create_notebook_executor_app()) as client:
        yield client


def test_an_input_larger_than_what_the_worker_holds_in_memory_arrives_whole(worker, tmp_path):
    size = 64 * 1024 * 1024
    store = _Store(size)
    source = (
        "import hashlib\n"
        "size = data.stat().st_size\n"
        "digest = hashlib.sha256(data.read_bytes()).hexdigest()\n"
        "print(size, digest)"
    )

    tracemalloc.start()
    try:
        response = worker.post("/v1/execute-manifest", json=_manifest(store, source))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert response.status_code == 200, response.text
    (bundle,) = store.uploads
    (tmp_path / "bundle.tar").write_bytes(bundle)
    result = unpack_notebook_output_bundle(tmp_path / "bundle.tar", tmp_path / "out")
    assert result["success"], result.get("error")
    assert result["variables"]["size"]["preview"] == size
    assert result["variables"]["digest"]["preview"] == store.digest()
    # The cell ran in a subprocess, reading the file the worker wrote; the
    # worker process itself never held more than a few chunks of it.
    assert peak < size // 4, f"worker process peaked at {peak} bytes for a {size}-byte input"


def test_an_input_over_the_cap_is_refused_at_the_cap(worker, monkeypatch):
    cap = 1024 * 1024
    monkeypatch.setenv("STRATA_WORKER_MAX_INPUT_BYTES", str(cap))
    store = _Store(256 * 1024 * 1024)

    response = worker.post("/v1/execute-manifest", json=_manifest(store, "x = 1"))

    assert response.status_code == 413
    assert "cap during download" in response.json()["detail"]
    assert store.uploads == []
    # The worker stopped reading at the cap; what the store got out before
    # the connection closed is socket buffers, not the input.
    assert store.sent < store.size // 4


def test_an_input_named_to_leave_the_run_directory_is_refused(worker):
    """The name comes from the request and is cut to its last component,
    and ``..`` is a last component."""
    store = _Store(1024)
    manifest = _manifest(store, "x = 1")
    manifest["metadata"]["params"]["input_specs"]["data"]["file"] = ".."

    response = worker.post("/v1/execute-manifest", json=manifest)

    assert response.status_code == 400
    assert "not a plain file name" in response.json()["detail"]
    assert store.sent == 0

"""Fixtures and helpers for the ledger's own suite.

Nothing here imports Strata: the suite must run from a checkout of this package alone.
"""

from __future__ import annotations

import http.server
import json
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
import pytest
from vestigo.build_store import reset_build_store
from vestigo.store import register_validator, reset_artifact_store


def _arrow_rows(data: bytes) -> int:
    """Rows in exactly one Arrow IPC stream; trailing bytes (a second stream) are an error."""
    if not data:
        return 0
    buf = pa.BufferReader(pa.py_buffer(data))
    rows = sum(batch.num_rows for batch in pa.ipc.open_stream(buf))
    if buf.tell() != len(data):
        raise ValueError(f"{len(data) - buf.tell()} bytes after the stream's end")
    return rows


# The application registers its formats at startup; this suite stands in for one.
register_validator("arrow/ipc", _arrow_rows)
register_validator("", _arrow_rows)


@pytest.fixture(autouse=True)
def _reset_singletons():
    """The store singletons are process globals shared by every test on a worker."""
    yield
    reset_artifact_store()
    reset_build_store()


BUILD_TARGET_IDS: tuple[str, ...] = (
    "a",
    "art-123",
    "canonical-artifact",
    *(f"art-{i}" for i in range(4)),
    *(f"artifact-{i}" for i in range(1, 4)),
)


def seed_build_targets(
    artifact_dir: Path,
    artifact_ids: Iterable[str] = BUILD_TARGET_IDS,
    versions: Iterable[int] = range(1, 8),
) -> None:
    """Create the artifact versions that build rows are allowed to reference.

    ``artifact_builds`` has a foreign key on ``(artifact_id, version)`` and production always
    creates the version first. Opening the artifact store also creates ``artifact_versions``
    in the shared database, without which the key cannot resolve.
    """
    from vestigo.store import ArtifactStore

    store = ArtifactStore(artifact_dir)
    try:
        for artifact_id in artifact_ids:
            for version in versions:
                created = store.create_artifact(artifact_id, f"seed-{artifact_id}-{version}")
                store.blob_store.write_blob(artifact_id, created, b"seed")
                store.finalize_artifact(artifact_id, created, "{}", 1, 4)
    finally:
        store.close()


def race_after_pending_read(store, competitor, first, second) -> tuple[object, object]:
    """Run ``first(store)``, landing ``second(competitor)`` right after it reads the pending row.

    ``competitor`` is a second store whose connections do not wait for a lock. One that finds it
    held runs ``second`` again once ``first`` is done, as a second process waiting on it would.
    Returns each call's result, or the ``ValueError`` it raised.
    """
    outcomes: dict[str, object] = {}
    state = {"read": False, "locked": False}

    def call(key, fn, target):
        try:
            outcomes[key] = fn(target)
        except ValueError as exc:
            outcomes[key] = exc

    class Interleaved:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, attr):
            return getattr(self._inner, attr)

        def execute(self, sql, params=()):
            if state["read"] and not state["locked"] and "second" not in outcomes:
                try:
                    call("second", second, competitor)
                except competitor.dialect.operational_error:
                    state["locked"] = True
            cursor = self._inner.execute(sql, params)
            if "FROM registry_pending" in sql:
                state["read"] = True
            return cursor

    connect = store._get_connection
    store._get_connection = lambda: Interleaved(connect())
    try:
        call("first", first, store)
    finally:
        store._get_connection = connect
    if state["locked"]:
        call("second", second, competitor)
    return outcomes["first"], outcomes["second"]


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

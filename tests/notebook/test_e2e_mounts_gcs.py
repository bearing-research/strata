"""GCS mount integration tests against a fake-gcs-server testcontainer.

Phase 3 of issue #19. Mirrors ``test_e2e_mounts_s3.py`` and
``test_e2e_mounts_azure.py`` against ``fsouza/fake-gcs-server`` — a
read-write GCS emulator that speaks the JSON API.

``testcontainers.google`` only ships Datastore + PubSub emulators (no
GCS), so this file uses ``DockerContainer`` directly and runs the
emulator with ``-scheme http`` so we don't need a self-signed cert.
``gcsfs`` is the fsspec backend (mapped from URI scheme ``gs`` →
fsspec protocol ``gcs`` by ``mounts._scheme_to_fsspec_protocol``).

Three scopes:

- **Scope A — Annotation-only.** ``# @mount data gs://bucket/key ro``
  with no ``[[mounts]]`` block. Credentials reach fsspec via the
  ``CellExecutor.mount_credentials`` kwarg from Phase 0.
- **Scope B — Read-write.** A cell mounts ``rw`` and writes; a separate
  cell mounts ``ro`` and reads back, asserting sync-back actually pushed
  bytes to the emulator.
- **Scope C — Storage options via TOML.** ``[[mounts]] options = {...}``
  carries the same ``endpoint_url`` / ``token`` / ``project`` per-mount;
  ``CellExecutor`` constructed *without* ``mount_credentials``.

Requires Docker. Skipped at collection time when the daemon is unreachable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import docker
import httpx
import pytest
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from strata.notebook.executor import CellExecutor
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mounts import MountCredentials
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)
from tests.conftest import start_container_or_skip


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


_GCS_TEST_PROJECT = "strata-mount-test"
_FAKE_GCS_PORT = 4443


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fake_gcs_container():
    """Module-scoped fake-gcs-server emulator.

    ``-scheme http`` keeps it on plain HTTP so we don't fight self-signed
    certs in CI. ``-public-host`` makes the emulator return signed-URL
    redirects against the host:port that the test process sees rather
    than the in-container ``0.0.0.0:4443`` (which gcsfs cannot reach).
    """
    container = DockerContainer("fsouza/fake-gcs-server:latest")
    container.with_command(
        f"-scheme http -port {_FAKE_GCS_PORT} -public-host localhost:{_FAKE_GCS_PORT}"
    )
    container.with_exposed_ports(_FAKE_GCS_PORT)
    start_container_or_skip(
        container,
        label="fake-gcs-server",
        ready=lambda c: wait_for_logs(c, "server started at"),
    )
    try:
        yield container
    finally:
        container.stop()


def _endpoint(fake_gcs_container: DockerContainer) -> str:
    host = fake_gcs_container.get_container_host_ip()
    port = fake_gcs_container.get_exposed_port(_FAKE_GCS_PORT)
    return f"http://{host}:{port}"


def _gcsfs_options(fake_gcs_container: DockerContainer) -> dict[str, object]:
    """fsspec/gcsfs storage_options for the fake-gcs-server emulator.

    ``token="anon"`` skips OAuth — the emulator doesn't validate
    credentials. ``endpoint_url`` overrides the production GCS endpoint.
    ``project`` can be any non-empty string; the emulator doesn't care.
    """
    return {
        "endpoint_url": _endpoint(fake_gcs_container),
        "token": "anon",
        "project": _GCS_TEST_PROJECT,
    }


@pytest.fixture
def gs_credentials(fake_gcs_container) -> MountCredentials:
    """Per-scheme credentials map for ``CellExecutor.mount_credentials``."""
    return {"gs": _gcsfs_options(fake_gcs_container)}


@pytest.fixture
def fresh_bucket(fake_gcs_container, request) -> str:
    """Make-and-return a unique bucket per test via fake-gcs-server's REST API.

    GCS bucket names: lowercase, hyphens/digits, 3–63 chars — same
    munging as the S3/Azure test fixtures.
    """
    raw = request.node.name.lower().replace("_", "-").replace(".", "-")
    name = f"mt-{raw}"[:63].rstrip("-")
    endpoint = _endpoint(fake_gcs_container)
    response = httpx.post(
        f"{endpoint}/storage/v1/b",
        params={"project": _GCS_TEST_PROJECT},
        json={"name": name},
        timeout=10.0,
    )
    # 409 is "bucket already exists" — fine in module-scoped retries
    if response.status_code not in (200, 409):
        response.raise_for_status()
    return name


def _put(fake_gcs_container: DockerContainer, bucket: str, key: str, content: bytes) -> None:
    """Upload a blob via fake-gcs-server's multipart upload endpoint."""
    endpoint = _endpoint(fake_gcs_container)
    response = httpx.post(
        f"{endpoint}/upload/storage/v1/b/{bucket}/o",
        params={"uploadType": "media", "name": key},
        content=content,
        timeout=10.0,
    )
    response.raise_for_status()


def _make_session(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "GcsMountTest")
    prev: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


# ---------------------------------------------------------------------------
# Scope A — Annotation-only mount, credentials via mount_credentials kwarg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_annotation_only_mount_reads_via_credentials_kwarg(
    tmp_path: Path,
    fake_gcs_container,
    gs_credentials: MountCredentials,
    fresh_bucket: str,
) -> None:
    """Phase 0's mount_credentials kwarg drives an annotation-only mount end-to-end."""
    _put(fake_gcs_container, fresh_bucket, "data/hello.txt", b"hello from fake-gcs")

    source = textwrap.dedent(
        f"""
        # @mount data gs://{fresh_bucket}/data ro
        content = (data / "hello.txt").read_text()
        """
    ).strip()
    session = _make_session(tmp_path, [("c1", source)])
    executor = CellExecutor(session, mount_credentials=gs_credentials)

    result = await executor.execute_cell("c1", source)

    assert result.success, f"cell errored: {result.error}"
    assert result.outputs["content"]["preview"] == "hello from fake-gcs"


# ---------------------------------------------------------------------------
# Scope B — Read-write mount: write in one cell, read in another
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rw_mount_writes_then_separate_ro_cell_reads_back(
    tmp_path: Path,
    fake_gcs_container,
    gs_credentials: MountCredentials,
    fresh_bucket: str,
) -> None:
    """RW sync-back actually pushes bytes; a downstream RO mount sees them."""
    write_source = textwrap.dedent(
        f"""
        # @mount scratch gs://{fresh_bucket}/out rw
        (scratch / "report.txt").write_text("hello from rw")
        wrote_bytes = 13
        """
    ).strip()
    read_source = textwrap.dedent(
        f"""
        # @mount data gs://{fresh_bucket}/out ro
        content = (data / "report.txt").read_text()
        """
    ).strip()

    session = _make_session(tmp_path, [("c_write", write_source), ("c_read", read_source)])
    executor = CellExecutor(session, mount_credentials=gs_credentials)

    write_result = await executor.execute_cell("c_write", write_source)
    assert write_result.success, f"write cell errored: {write_result.error}"

    read_result = await executor.execute_cell("c_read", read_source)
    assert read_result.success, f"read cell errored: {read_result.error}"
    assert read_result.outputs["content"]["preview"] == "hello from rw"


# ---------------------------------------------------------------------------
# Scope C — Storage options via TOML [[mounts]] (no mount_credentials kwarg)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_toml_mount_options_carry_endpoint_credentials(
    tmp_path: Path,
    fake_gcs_container,
    fresh_bucket: str,
) -> None:
    """Per-mount ``options = {...}`` reaches fsspec without a scheme-level credentials hook."""
    _put(fake_gcs_container, fresh_bucket, "tbl/v.txt", b"hello from toml options")

    notebook_dir = create_notebook(tmp_path, "TomlOptionsTest")
    update_notebook_mounts(
        notebook_dir,
        [
            MountSpec(
                name="tbl",
                uri=f"gs://{fresh_bucket}/tbl",
                mode=MountMode.READ_ONLY,
                options=_gcsfs_options(fake_gcs_container),
            ),
        ],
    )
    add_cell_to_notebook(notebook_dir, "c1")
    source = 'content = (tbl / "v.txt").read_text()'
    write_cell(notebook_dir, "c1", source)

    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    executor = CellExecutor(session)  # no mount_credentials kwarg — TOML carries it

    result = await executor.execute_cell("c1", source)

    assert result.success, f"cell errored: {result.error}"
    assert result.outputs["content"]["preview"] == "hello from toml options"

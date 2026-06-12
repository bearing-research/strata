"""S3 mount integration tests against a MinIO testcontainer.

Phase 1 of issue #19. Three scopes exercised against a real S3-protocol
backend (MinIO):

- **Scope A — Annotation-only.** ``# @mount data s3://bucket/key ro`` with
  no ``[[mounts]]`` block. Credentials reach fsspec via the
  ``CellExecutor.mount_credentials`` kwarg added in Phase 0.
- **Scope B — Read-write.** A cell mounts ``rw`` and writes; a separate
  cell mounts ``ro`` and reads back, asserting sync-back actually pushed
  bytes to the backend.
- **Scope C — Storage options via TOML.** ``[[mounts]] options = {...}``
  carries the same ``endpoint_url`` / ``key`` / ``secret`` per-mount;
  ``CellExecutor`` constructed *without* ``mount_credentials``.

Requires Docker. Skipped at collection time when Docker or
``testcontainers[minio]`` is unavailable.
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import docker
import pytest
from testcontainers.minio import MinioContainer

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
    """Skip when the daemon is unreachable — the docker package itself is a
    transitive dev dep, but the daemon may not be running on a contributor's
    laptop. CI always has Docker, so this only triggers locally.
    """
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minio_container():
    """Module-scoped MinIO container shared across mount tests.

    Pinned image: matches ``tests/test_s3_integration.py`` so both modules
    can hit the registry cache.
    """
    container = start_container_or_skip(
        MinioContainer("minio/minio:RELEASE.2024-11-07T00-52-20Z"), label="MinIO"
    )
    try:
        yield container
    finally:
        container.stop()


def _s3_endpoint(minio_container) -> str:
    """MinIO returns an endpoint without scheme; s3fs needs HTTP for local."""
    endpoint = minio_container.get_config()["endpoint"]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint


def _fsspec_options(minio_container) -> dict[str, object]:
    """Build s3fs storage_options pointing at the MinIO emulator."""
    config = minio_container.get_config()
    return {
        "endpoint_url": _s3_endpoint(minio_container),
        "key": config["access_key"],
        "secret": config["secret_key"],
        "client_kwargs": {"region_name": "us-east-1"},
    }


@pytest.fixture
def s3_credentials(minio_container) -> MountCredentials:
    """Per-scheme credentials map for ``CellExecutor.mount_credentials``."""
    return {"s3": _fsspec_options(minio_container)}


@pytest.fixture
def fresh_bucket(minio_container, request) -> str:
    """Make-and-return a unique bucket per test, cleaned up by the container teardown."""
    # Bucket names: lowercase, hyphens only, ≤63 chars.
    raw = request.node.name.lower().replace("_", "-").replace(".", "-")
    bucket = f"mt-{raw}"[:63].rstrip("-")
    client = minio_container.get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
    return bucket


def _put(minio_container, bucket: str, key: str, content: bytes) -> None:
    client = minio_container.get_client()
    client.put_object(bucket, key, io.BytesIO(content), length=len(content))


def _make_session(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    """Create a notebook on disk with the given (cell_id, source) pairs."""
    notebook_dir = create_notebook(tmp_path, "S3MountTest")
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
    minio_container,
    s3_credentials: MountCredentials,
    fresh_bucket: str,
) -> None:
    """Phase 0's mount_credentials kwarg drives an annotation-only mount end-to-end."""
    _put(minio_container, fresh_bucket, "data/hello.txt", b"hello from minio")

    source = textwrap.dedent(
        f"""
        # @mount data s3://{fresh_bucket}/data ro
        content = (data / "hello.txt").read_text()
        """
    ).strip()
    session = _make_session(tmp_path, [("c1", source)])
    executor = CellExecutor(session, mount_credentials=s3_credentials)

    result = await executor.execute_cell("c1", source)

    assert result.success, f"cell errored: {result.error}"
    assert result.outputs["content"]["preview"] == "hello from minio"


# ---------------------------------------------------------------------------
# Scope B — Read-write mount: write in one cell, read in another
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rw_mount_writes_then_separate_ro_cell_reads_back(
    tmp_path: Path,
    minio_container,
    s3_credentials: MountCredentials,
    fresh_bucket: str,
) -> None:
    """RW sync-back actually pushes bytes; a downstream RO mount sees them."""
    write_source = textwrap.dedent(
        f"""
        # @mount scratch s3://{fresh_bucket}/out rw
        (scratch / "report.txt").write_text("hello from rw")
        wrote_bytes = 13
        """
    ).strip()
    read_source = textwrap.dedent(
        f"""
        # @mount data s3://{fresh_bucket}/out ro
        content = (data / "report.txt").read_text()
        """
    ).strip()

    session = _make_session(tmp_path, [("c_write", write_source), ("c_read", read_source)])
    executor = CellExecutor(session, mount_credentials=s3_credentials)

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
    minio_container,
    fresh_bucket: str,
) -> None:
    """Per-mount ``options = {...}`` reaches fsspec without a scheme-level credentials hook."""
    _put(minio_container, fresh_bucket, "tbl/v.txt", b"hello from toml options")

    notebook_dir = create_notebook(tmp_path, "TomlOptionsTest")
    update_notebook_mounts(
        notebook_dir,
        [
            MountSpec(
                name="tbl",
                uri=f"s3://{fresh_bucket}/tbl",
                mode=MountMode.READ_ONLY,
                options=_fsspec_options(minio_container),
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

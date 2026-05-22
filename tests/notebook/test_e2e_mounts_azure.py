"""Azure Blob mount integration tests against an Azurite testcontainer.

Phase 2 of issue #19. Mirrors ``test_e2e_mounts_s3.py`` against Azurite,
Microsoft's emulator for the Blob/Queue/Table services. ``adlfs`` is the
fsspec backend (mapped from URI scheme ``az`` → fsspec protocol ``abfs``
by ``mounts._scheme_to_fsspec_protocol``).

Three scopes:

- **Scope A — Annotation-only.** ``# @mount data az://container/key ro``
  with no ``[[mounts]]`` block. Credentials reach fsspec via the
  ``CellExecutor.mount_credentials`` kwarg from Phase 0.
- **Scope B — Read-write.** A cell mounts ``rw`` and writes; a separate
  cell mounts ``ro`` and reads back, asserting sync-back actually pushed
  bytes to the Blob service.
- **Scope C — Storage options via TOML.** ``[[mounts]] options = {...}``
  carries the same ``connection_string`` per-mount; ``CellExecutor``
  constructed *without* ``mount_credentials``.

Requires Docker. Skipped at collection time when the Docker daemon is
unreachable (CI always has it).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import docker
import pytest
from azure.storage.blob import BlobServiceClient
from testcontainers.azurite import AzuriteContainer

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


def _docker_daemon_reachable() -> bool:
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
def azurite_container():
    """Module-scoped Azurite container exposing the Blob endpoint.

    ``:latest`` is intentional — the installed ``azure-storage-blob``
    SDK ships a recent API version (``2025-11-05+``) that older Azurite
    builds reject with ``InvalidHeaderValue``. Microsoft's error
    response for that mismatch literally says "upgrade Azurite to
    latest version and retry," so we track latest. The tradeoff
    (occasional CI breakage if Microsoft ships a regression) is
    smaller than the SDK/emulator drift problem with a fixed pin.
    """
    with AzuriteContainer("mcr.microsoft.com/azure-storage/azurite:latest") as az:
        yield az


def _adlfs_options(azurite_container: AzuriteContainer) -> dict[str, object]:
    """fsspec/adlfs storage_options for the Azurite emulator.

    ``connection_string`` carries the BlobEndpoint and account credentials;
    ``account_name`` is set explicitly so adlfs resolves ``abfs://<container>/...``
    URIs without needing the full FQDN form.
    """
    return {
        "connection_string": azurite_container.get_connection_string(),
        "account_name": azurite_container.account_name,
    }


@pytest.fixture
def az_credentials(azurite_container) -> MountCredentials:
    """Per-scheme credentials map for ``CellExecutor.mount_credentials``."""
    return {"az": _adlfs_options(azurite_container)}


@pytest.fixture
def fresh_container(azurite_container, request) -> str:
    """Make-and-return a unique Blob container per test.

    Azure container names: lowercase, 3–63 chars, hyphens/digits — same
    constraints satisfied by the S3 test's bucket-name munging.
    """
    raw = request.node.name.lower().replace("_", "-").replace(".", "-")
    name = f"mt-{raw}"[:63].rstrip("-")
    client = BlobServiceClient.from_connection_string(azurite_container.get_connection_string())
    try:
        client.create_container(name)
    except Exception:
        # ResourceExistsError if container already exists from a prior test run
        # in the same module-scoped Azurite container (e.g., test retries).
        pass
    return name


def _put(azurite_container: AzuriteContainer, container: str, blob: str, content: bytes) -> None:
    client = BlobServiceClient.from_connection_string(azurite_container.get_connection_string())
    blob_client = client.get_blob_client(container=container, blob=blob)
    blob_client.upload_blob(content, overwrite=True)


def _make_session(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "AzureMountTest")
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
    azurite_container,
    az_credentials: MountCredentials,
    fresh_container: str,
) -> None:
    """Phase 0's mount_credentials kwarg drives an annotation-only mount end-to-end."""
    _put(azurite_container, fresh_container, "data/hello.txt", b"hello from azurite")

    source = textwrap.dedent(
        f"""
        # @mount data az://{fresh_container}/data ro
        content = (data / "hello.txt").read_text()
        """
    ).strip()
    session = _make_session(tmp_path, [("c1", source)])
    executor = CellExecutor(session, mount_credentials=az_credentials)

    result = await executor.execute_cell("c1", source)

    assert result.success, f"cell errored: {result.error}"
    assert result.outputs["content"]["preview"] == "hello from azurite"


# ---------------------------------------------------------------------------
# Scope B — Read-write mount: write in one cell, read in another
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rw_mount_writes_then_separate_ro_cell_reads_back(
    tmp_path: Path,
    azurite_container,
    az_credentials: MountCredentials,
    fresh_container: str,
) -> None:
    """RW sync-back actually pushes bytes; a downstream RO mount sees them."""
    write_source = textwrap.dedent(
        f"""
        # @mount scratch az://{fresh_container}/out rw
        (scratch / "report.txt").write_text("hello from rw")
        wrote_bytes = 13
        """
    ).strip()
    read_source = textwrap.dedent(
        f"""
        # @mount data az://{fresh_container}/out ro
        content = (data / "report.txt").read_text()
        """
    ).strip()

    session = _make_session(tmp_path, [("c_write", write_source), ("c_read", read_source)])
    executor = CellExecutor(session, mount_credentials=az_credentials)

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
    azurite_container,
    fresh_container: str,
) -> None:
    """Per-mount ``options = {...}`` reaches fsspec without a scheme-level credentials hook."""
    _put(azurite_container, fresh_container, "tbl/v.txt", b"hello from toml options")

    notebook_dir = create_notebook(tmp_path, "TomlOptionsTest")
    update_notebook_mounts(
        notebook_dir,
        [
            MountSpec(
                name="tbl",
                uri=f"az://{fresh_container}/tbl",
                mode=MountMode.READ_ONLY,
                options=_adlfs_options(azurite_container),
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

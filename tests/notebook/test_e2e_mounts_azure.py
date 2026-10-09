"""Azure Blob mount integration tests against an Azurite testcontainer.

Mirrors ``test_e2e_mounts_s3.py`` with ``adlfs`` as the fsspec backend (scheme ``az`` maps
to protocol ``abfs``). Covers an annotation-only mount fed by
``CellExecutor.mount_credentials``, an ``rw`` mount read back by a separate ``ro`` cell, and
``[[mounts]] options`` carrying the connection string with no ``mount_credentials``.
Skipped when the Docker daemon is unreachable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import docker
import pytest
from azure.storage.blob import BlobServiceClient
from testcontainers.community.azurite import AzuriteContainer

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
from tests.conftest import make_azurite, start_container_or_skip


def _docker_daemon_reachable() -> bool:
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)


pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Fixtures


@pytest.fixture(scope="module")
def azurite_container():
    """Module-scoped Azurite container exposing the Blob endpoint."""
    container = start_container_or_skip(make_azurite(), label="Azurite")
    try:
        yield container
    finally:
        container.stop()


def _adlfs_options(azurite_container: AzuriteContainer) -> dict[str, object]:
    """fsspec/adlfs storage_options for Azurite.

    ``account_name`` lets adlfs resolve ``abfs://<container>/...`` without the FQDN form.
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
    """A unique Blob container per test."""
    raw = request.node.name.lower().replace("_", "-").replace(".", "-")
    name = f"mt-{raw}"[:63].rstrip("-")
    client = BlobServiceClient.from_connection_string(azurite_container.get_connection_string())
    try:
        client.create_container(name)
    except Exception:
        # The container may already exist from a retry in the same module-scoped Azurite.
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
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


# Scope A: annotation-only mount, credentials via mount_credentials kwarg


@pytest.mark.asyncio
async def test_annotation_only_mount_reads_via_credentials_kwarg(
    tmp_path: Path,
    azurite_container,
    az_credentials: MountCredentials,
    fresh_container: str,
) -> None:
    """The ``mount_credentials`` kwarg drives an annotation-only mount end to end."""
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


# Scope B: read-write mount, write in one cell and read in another


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


# Scope C: storage options via TOML [[mounts]] (no mount_credentials kwarg)


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
    session.refresh_environment_runtime()
    executor = CellExecutor(session)  # no mount_credentials kwarg; TOML carries it

    result = await executor.execute_cell("c1", source)

    assert result.success, f"cell errored: {result.error}"
    assert result.outputs["content"]["preview"] == "hello from toml options"

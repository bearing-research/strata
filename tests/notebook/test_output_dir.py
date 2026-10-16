"""``# @output_dir <name>``: a folder a cell fills, kept as files and a manifest output.

What is in the folder when the run ends is the output: each file stored once by digest,
the manifest stored as the variable, and a downstream cell gets the folder back as a local
path. A failed run keeps its files under a failed version that is never a cache hit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from strata.notebook.annotations import parse_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

WRITER = (
    "# @output_dir ckpt\n"
    "(ckpt / 'epochs').mkdir()\n"
    "for i in range(2):\n"
    "    (ckpt / 'epochs' / f'e{i}.bin').write_bytes(bytes([i]) * 10)\n"
    "(ckpt / 'final.txt').write_text('done')\n"
)
READER = "seen = sorted(p.relative_to(ckpt).as_posix() for p in ckpt.rglob('*') if p.is_file())\n"


def _session(tmp_path: Path, writer: str = WRITER, reader: str | None = READER):
    nb = create_notebook(tmp_path, "Folders")
    add_cell_to_notebook(nb, "w", None)
    write_cell(nb, "w", writer)
    if reader is not None:
        add_cell_to_notebook(nb, "r", "w")
        write_cell(nb, "r", reader)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _manifest(session, version: int | None = None) -> tuple[dict, str]:
    manager = session.get_artifact_manager()
    artifact_id = manager.cell_artifact_id("w", "ckpt")
    store = manager.artifact_store
    artifact = (
        store.get_latest_version(artifact_id)
        if version is None
        else store.get_artifact(artifact_id, version)
    )
    return json.loads(store.read_blob(artifact.id, artifact.version)), artifact.state


class TestTheAnnotation:
    def test_it_parses(self):
        assert parse_annotations("# @output_dir ckpt\n# @output_dir logs\nx = 1").output_dirs == [
            "ckpt",
            "logs",
        ]

    def test_a_name_that_is_not_an_identifier_is_dropped(self):
        assert parse_annotations("# @output_dir 2ckpt\nx = 1").output_dirs == []

    def test_the_folder_is_a_define_of_its_cell(self, tmp_path):
        session = _session(tmp_path)

        assert "ckpt" in session.notebook_state.get_cell("w").defines
        assert "w" in session.notebook_state.get_cell("r").upstream_ids


class TestAFolderOutput:
    async def test_a_downstream_cell_reads_the_folder(self, tmp_path):
        session = _session(tmp_path)
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", WRITER)
        assert written.success, written.error
        read = await executor.execute_cell("r", READER)

        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]
        manifest, state = _manifest(session)
        assert state == "ready"
        assert manifest["files"]["final.txt"]["size"] == 4

    async def test_a_leaf_folder_is_kept(self, tmp_path):
        """Nothing reads it: a checkpoint is the point of the cell, not a downstream value."""
        session = _session(tmp_path, reader=None)

        assert (await CellExecutor(session).execute_cell("w", WRITER)).success

        manifest, state = _manifest(session)
        assert state == "ready"
        assert sorted(manifest["files"]) == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]

    async def test_a_second_run_is_a_cache_hit(self, tmp_path):
        session = _session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success

        again = await executor.execute_cell("w", WRITER)

        assert again.cache_hit

    async def test_a_file_that_did_not_change_is_stored_once(self, tmp_path):
        session = _session(tmp_path, reader=None)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success
        edited = WRITER + "(ckpt / 'extra.txt').write_text('more')\n"
        cell = session.notebook_state.get_cell("w")
        cell.source = edited
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", edited)).success

        manifest, _ = _manifest(session)
        manager = session.get_artifact_manager()
        sha = manifest["files"]["final.txt"]["sha256"]
        file_id = manager.cell_artifact_id("w", f"__file__{sha}")
        assert manager.artifact_store.get_latest_version(file_id).version == 1
        assert sorted(manifest["files"]) == [
            "epochs/e0.bin",
            "epochs/e1.bin",
            "extra.txt",
            "final.txt",
        ]

    async def test_a_failed_run_keeps_what_it_wrote_and_is_not_cached(self, tmp_path):
        failing = (
            "# @output_dir ckpt\n"
            "(ckpt / 'e0.bin').write_bytes(b'x' * 5)\n"
            "raise RuntimeError('lost')\n"
        )
        session = _session(tmp_path, writer=failing, reader=None)
        executor = CellExecutor(session)

        result = await executor.execute_cell("w", failing)

        assert not result.success
        # A failed version is not the latest value; it is read by its number.
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]
        manager = session.get_artifact_manager()
        file_id = manager.cell_artifact_id("w", f"__file__{manifest['files']['e0.bin']['sha256']}")
        assert manager.artifact_store.read_blob(file_id, 1) == b"x" * 5
        assert not (await executor.execute_cell("w", failing)).cache_hit

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    async def test_a_link_in_the_folder_is_not_kept(self, tmp_path):
        secret = tmp_path / "server-only.txt"
        secret.write_text("secret")
        writer = (
            "# @output_dir ckpt\n"
            "(ckpt / 'mine.txt').write_text('mine')\n"
            f"(ckpt / 'leak.txt').symlink_to({str(secret)!r})\n"
        )
        session = _session(tmp_path, writer=writer, reader=None)

        assert (await CellExecutor(session).execute_cell("w", writer)).success

        manifest, _ = _manifest(session)
        assert list(manifest["files"]) == ["mine.txt"]


class TestNotYet:
    async def test_a_loop_cell_is_refused(self, tmp_path):
        source = "# @loop max_iter=2 carry=n\n# @output_dir ckpt\nn = 1\n"
        session = _session(tmp_path, writer=source, reader=None)

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "@output_dir does not work in a # @loop cell" in result.error

    async def test_a_worker_that_cannot_be_asked_is_refused_before_it_runs(self, tmp_path):
        source = "# @worker gpu\n# @output_dir ckpt\n(ckpt / 'a').write_text('a')\n"
        session = _session(tmp_path, writer=source, reader=None)
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "http://127.0.0.1:9/v1/execute"},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "could not be asked whether it returns # @output_dir folders" in result.error

    async def test_an_older_worker_is_refused_before_it_runs(self, tmp_path, monkeypatch):
        source = "# @worker gpu\n# @output_dir ckpt\n(ckpt / 'a').write_text('a')\n"
        session = _session(tmp_path, writer=source, reader=None)
        from strata.notebook import workers
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        async def older(worker, feature):
            return False

        monkeypatch.setattr(workers, "worker_advertises", older)
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "http://127.0.0.1:9/v1/execute"},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "does not return # @output_dir folders" in result.error

    async def test_run_all_runs_it_on_its_own(self, tmp_path):
        """A batch shares one process and has no run directory per cell to collect from."""
        from strata.notebook.executor import is_cell_batchable

        session = _session(tmp_path)
        executor = CellExecutor(session)

        assert not is_cell_batchable(executor, session.notebook_state.get_cell("w"))
        assert is_cell_batchable(executor, session.notebook_state.get_cell("r"))


class TestTheBundle:
    """A worker returns each folder's files in the bundle, streamed past the member cap."""

    def test_files_larger_than_the_member_cap_round_trip(self, tmp_path, monkeypatch):
        from strata.notebook.output_dirs import OUTPUT_DIRS
        from strata.notebook.remote_bundle import (
            pack_notebook_output_bundle,
            unpack_notebook_output_bundle,
        )

        monkeypatch.setenv("STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES", "4096")
        run = tmp_path / "run"
        folder = run / OUTPUT_DIRS / "ckpt"
        (folder / "deep").mkdir(parents=True)
        big = bytes(range(256)) * 40  # 10240 bytes, past the 4096-byte cap
        (folder / "deep" / "big.bin").write_bytes(big)
        (folder / "empty.txt").write_bytes(b"")
        if sys.platform != "win32":
            (folder / "leak").symlink_to(tmp_path / "elsewhere")
        bundle = tmp_path / "bundle.tar"

        pack_notebook_output_bundle(bundle, {"success": True, "variables": {}}, run)
        out = tmp_path / "unpacked"
        unpack_notebook_output_bundle(bundle, out)

        got = out / OUTPUT_DIRS / "ckpt"
        assert (got / "deep" / "big.bin").read_bytes() == big
        assert (got / "empty.txt").read_bytes() == b""
        assert sorted(p.relative_to(got).as_posix() for p in got.rglob("*") if p.is_file()) == [
            "deep/big.bin",
            "empty.txt",
        ]

    @pytest.mark.parametrize("rel", ["../escape.bin", "/abs.bin"])
    def test_a_path_leaving_the_folder_is_refused(self, tmp_path, rel):
        import io
        import tarfile

        from strata.notebook.remote_bundle import SCHEMA_VERSION, unpack_notebook_output_bundle

        bundle = tmp_path / "bundle.tar"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "success": True,
            "variables": {},
            "stdout_file": "stdout.txt",
            "stderr_file": "stderr.txt",
            "output_dirs": {"ckpt": {rel: 1}},
        }
        with tarfile.open(bundle, "w") as tar:
            for name, data in (
                ("manifest.json", json.dumps(manifest).encode()),
                ("stdout.txt", b""),
                ("stderr.txt", b""),
                (f"dirs/ckpt/{rel}", b"x"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Invalid bundle output file"):
            unpack_notebook_output_bundle(bundle, tmp_path / "unpacked")
        assert not (tmp_path / "escape.bin").exists()


class TestOnAWorker:
    @pytest.mark.parametrize("transport", ["direct", "signed"])
    async def test_a_worker_returns_the_folder_and_a_local_cell_reads_it(
        self, tmp_path, transport, notebook_executor_server, notebook_personal_server
    ):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        writer = "# @worker remote\n" + WRITER
        session = _session(tmp_path, writer=writer)
        config = {"url": notebook_executor_server["execute_url"], "transport": transport}
        if transport == "signed":
            config["strata_url"] = notebook_personal_server["base_url"]
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=config
            )
        ]
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)
        assert written.success, written.error
        assert written.execution_method == "executor"
        read = await executor.execute_cell("r", READER)

        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]

    async def test_a_failed_run_on_a_worker_keeps_what_it_wrote(
        self, tmp_path, notebook_executor_server
    ):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        failing = (
            "# @worker remote\n# @output_dir ckpt\n"
            "(ckpt / 'e0.bin').write_bytes(b'x' * 5)\n"
            "raise RuntimeError('lost')\n"
        )
        session = _session(tmp_path, writer=failing, reader=None)
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="r",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", failing)

        assert not result.success
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]

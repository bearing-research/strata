"""``# @output_dir <name>``: a folder a cell fills, kept as files and a manifest output.

What is in the folder when the run ends is the output: each file stored once by digest,
the manifest stored as the variable, and a downstream cell gets the folder back as a local
path. A failed run keeps its files under a failed version that is never a cache hit.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path

import pytest

from strata.notebook import output_dirs
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


def _manifest(session, version: int | None = None, cell_id: str = "w") -> tuple[dict, str]:
    manager = session.get_artifact_manager()
    artifact_id = manager.cell_artifact_id(cell_id, "ckpt")
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

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    async def test_a_folder_swapped_for_a_link_keeps_nothing_it_points_at(self, tmp_path):
        """The harness user owns the run directory, so it can turn ``output_dirs`` into a link."""
        server = tmp_path / "server-only"
        (server / "ckpt").mkdir(parents=True)
        (server / "ckpt" / "secret.txt").write_text("secret")
        writer = (
            "# @output_dir ckpt\n"
            "import os\n"
            "os.rename(ckpt.parent, ckpt.parent.with_name('moved'))\n"
            f"os.symlink({str(server)!r}, ckpt.parent)\n"
        )
        session = _session(tmp_path, writer=writer, reader=None)

        assert (await CellExecutor(session).execute_cell("w", writer)).success

        manifest, _ = _manifest(session)
        assert manifest["files"] == {}


def _listings(displays) -> list[str]:
    """The markdown displays: a trailing ``write_text`` call shows its return value too."""
    return [d["markdown_text"] for d in displays if d["content_type"] == "text/markdown"]


LISTING = (
    "**ckpt**: 3 files, 24 B\n"
    "\n"
    "| File | Size |\n"
    "| --- | ---: |\n"
    "| `epochs/e0.bin` | 10 B |\n"
    "| `epochs/e1.bin` | 10 B |\n"
    "| `final.txt` | 4 B |"
)


class TestTheCellShowsTheFolder:
    async def test_a_run_shows_its_files_and_sizes_before_the_cells_own_display(self, tmp_path):
        writer = WRITER + "'trained'\n"
        session = _session(tmp_path, writer=writer)

        result = await CellExecutor(session).execute_cell("w", writer)

        assert result.success, result.error
        shown = [d.get("markdown_text") or d.get("preview") for d in result.display_outputs]
        assert shown == [LISTING, "trained"]
        assert result.display_output["preview"] == "trained"

    async def test_two_folders_are_shown_in_name_order(self, tmp_path):
        writer = "# @output_dir b\n# @output_dir a\n(a / 'x').write_text('1')\n"
        session = _session(tmp_path, writer=writer, reader=None)

        result = await CellExecutor(session).execute_cell("w", writer)

        listed = _listings(result.display_outputs)
        assert [text.split(":")[0] for text in listed] == ["**a**", "**b**"]
        assert listed[1] == "**b**: 0 files, 0 B"

    async def test_a_cache_hit_and_a_reopen_show_it_from_the_store(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("w", WRITER)).success

        again = await CellExecutor(session).execute_cell("w", WRITER)
        reopened = NotebookSession(parse_notebook(session.path), session.path)
        reopened.compute_staleness()

        assert again.cache_hit
        assert _listings(again.display_outputs) == [LISTING]
        cell = reopened.serialize_cell(reopened.notebook_state.get_cell("w"))
        assert _listings(cell["display_outputs"]) == [LISTING]

    async def test_a_failed_runs_folder_is_not_shown_by_the_next_run(self, tmp_path):
        failing = "# @output_dir old\n(old / 'x').write_text('1')\nraise ValueError('no')\n"
        session = _session(tmp_path, writer=failing, reader=None)
        executor = CellExecutor(session)
        assert not (await executor.execute_cell("w", failing)).success

        cell = session.notebook_state.get_cell("w")
        cell.source = WRITER
        session.re_analyze_cell("w")
        result = await executor.execute_cell("w", WRITER)

        assert result.success, result.error
        assert _listings(result.display_outputs) == [LISTING]

    def test_a_long_folder_lists_the_first_files_and_counts_the_rest(self):
        files = {f"f{i:02d}": {"size": 1536} for i in range(25)}
        shown = output_dirs.listing("big", json.dumps({"cell_id": "w", "files": files}).encode())

        assert shown.startswith("**big**: 25 files, 37.5 KiB\n")
        assert "| `f19` | 1.5 KiB |" in shown
        assert "f20" not in shown
        assert shown.endswith("\n\n... and 5 more")

    def test_a_path_cannot_break_out_of_its_table_cell(self):
        files = {"a|b": {"size": 1}, "c`d": {"size": 2}}
        shown = output_dirs.listing("odd", json.dumps({"cell_id": "w", "files": files}).encode())

        assert "| `a\\|b` | 1 B |" in shown
        assert "| `` c`d `` | 2 B |" in shown


_SAME = "# @output_dir ckpt\n(ckpt / 'same.txt').write_text('same')\n"
OLD = _SAME + "(ckpt / 'old.txt').write_text('o')\n"
NEW = _SAME + "(ckpt / 'new.txt').write_text('n')\n"


class TestTheFileSweep:
    """A folder file goes once no manifest version left in the store names it."""

    @staticmethod
    async def _two_runs(tmp_path):
        session = _session(tmp_path, writer=OLD)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", OLD)).success
        cell = session.notebook_state.get_cell("w")
        cell.source = NEW
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", NEW)).success
        return session, executor

    @staticmethod
    def _stored(session, content: bytes) -> bool:
        import hashlib

        manager = session.get_artifact_manager()
        sha256 = hashlib.sha256(content).hexdigest()
        artifact_id = output_dirs.file_artifact_id(manager, "w", sha256)
        return manager.artifact_store.get_latest_version(artifact_id) is not None

    async def test_a_file_only_a_pruned_manifest_named_goes(self, tmp_path):
        session, executor = await self._two_runs(tmp_path)

        session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)

        assert not self._stored(session, b"o")
        assert self._stored(session, b"same")
        assert self._stored(session, b"n")
        read = await executor.execute_cell("r", READER)
        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["new.txt", "same.txt"]

    async def test_a_store_from_before_edges_declares_its_folders_when_opened(self, tmp_path):
        """A ledger upgrade creates the edges table empty; Strata reads the manifests it
        stored before and declares their files, wherever it opens the store, until the
        store records that it has."""
        import sqlite3

        from strata.artifact.retention import FOLDER_EDGES_UPGRADE
        from strata.notebook.artifact_integration import NotebookArtifactManager

        session, _ = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()
        manifest_id = manager.cell_artifact_id("w", "ckpt")
        conn = sqlite3.connect(manager.artifact_store.db_path)
        try:
            conn.execute("DELETE FROM retention_edges")
            conn.execute("DELETE FROM store_upgrades")
            conn.commit()
        finally:
            conn.close()

        reopened = NotebookArtifactManager(manager.notebook_id, manager.artifact_dir).artifact_store

        assert reopened.upgrade_applied(FOLDER_EDGES_UPGRADE)
        for version in (1, 2):
            assert reopened.requirements_met(reopened.get_artifact(manifest_id, version))
        conn = sqlite3.connect(reopened.db_path)
        try:
            edges = conn.execute(
                "SELECT from_version, COUNT(*) FROM retention_edges "
                "WHERE from_artifact_id = ? GROUP BY from_version ORDER BY from_version",
                (manifest_id,),
            ).fetchall()
        finally:
            conn.close()
        assert edges == [(1, 2), (2, 2)]

    async def test_a_pinned_earlier_manifest_keeps_its_files(self, tmp_path):
        session, _ = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()
        manager.artifact_store.retain(
            manager.cell_artifact_id("w", "ckpt"), 1, kind="pin", holder="review"
        )

        manager.prune(keep_superseded=0, min_idle_seconds=0)

        assert self._stored(session, b"o")

    async def test_nothing_goes_while_a_run_is_assembling_a_folder(self, tmp_path):
        session, _ = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()

        with output_dirs.assembling():
            manager.prune(keep_superseded=0, min_idle_seconds=0)
        assert self._stored(session, b"o")

        manager.prune(keep_superseded=0, min_idle_seconds=0)
        assert not self._stored(session, b"o")

    async def test_a_run_holds_off_a_sweep_until_its_manifest_names_its_files(
        self, tmp_path, monkeypatch
    ):
        session, executor = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()
        collect = output_dirs.collect

        def prune_before_the_manifest(*args, **kwargs):
            # The run's files are stored and named by no manifest yet.
            manifest = collect(*args, **kwargs)
            manager.prune(keep_superseded=0, min_idle_seconds=0)
            return manifest

        monkeypatch.setattr(output_dirs, "collect", prune_before_the_manifest)
        third = "# @output_dir ckpt\n(ckpt / 'third.txt').write_text('3')\n"
        cell = session.notebook_state.get_cell("w")
        cell.source = third
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", third)).success

        assert self._stored(session, b"3")

    async def test_a_manifest_that_cannot_be_read_stops_the_sweep(self, tmp_path):
        session, _ = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()
        current = manager.artifact_store.get_latest_version(manager.cell_artifact_id("w", "ckpt"))
        manager.artifact_store.blob_store.delete_blob(current.id, current.version)

        manager.prune(keep_superseded=0, min_idle_seconds=0)

        assert self._stored(session, b"o")

    async def test_a_failed_manifest_a_crash_left_without_bytes_does_not_stop_the_sweep(
        self, tmp_path
    ):
        from vestigo.provenance import TransformSpec

        session, _ = await self._two_runs(tmp_path)
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        artifact_id = manager.cell_artifact_id("w", "ckpt")
        spec = TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": output_dirs.CONTENT_TYPE},
            inputs=[],
        )
        version = store.create_artifact(artifact_id, provenance_hash="crash", transform_spec=spec)
        store.fail_artifact(artifact_id, version)

        manager.prune(keep_superseded=0, min_idle_seconds=0)

        assert store.get_artifact(artifact_id, version) is not None  # the newest is spared
        assert not self._stored(session, b"o")

    async def test_a_file_swept_while_a_run_reuses_it_is_stored_again(self, tmp_path, monkeypatch):
        session = _session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success
        store = session.get_artifact_manager().artifact_store
        reused = []

        def swept_meanwhile(artifact_id, version):
            reused.append(artifact_id)
            store.delete_artifact(artifact_id, version)

        monkeypatch.setattr(store, "record_use", swept_meanwhile)
        rerun = await executor.execute_cell_rerun("w", WRITER)

        assert rerun.success, rerun.error
        assert reused  # each unchanged file was reused, and marked as used first
        manifest, _ = _manifest(session)
        manager = session.get_artifact_manager()
        for entry in manifest["files"].values():
            artifact_id = output_dirs.file_artifact_id(manager, "w", entry["sha256"])
            assert store.get_artifact(artifact_id, entry["version"]).state == "ready"

    async def test_two_cells_holding_the_same_bytes_each_keep_their_file(self, tmp_path):
        first = "# @output_dir a\n(a / 'x.txt').write_text('same')\n"
        second = "# @output_dir b\n(b / 'x.txt').write_text('same')\n"
        reader = "seen = (b / 'x.txt').read_text()\n"
        nb = create_notebook(tmp_path, "Folders")
        for cell_id, after, source in (("a", None, first), ("b", "a", second), ("r", "b", reader)):
            add_cell_to_notebook(nb, cell_id, after)
            write_cell(nb, cell_id, source)
        session = NotebookSession(parse_notebook(nb), nb)
        session.refresh_environment_runtime()
        executor = CellExecutor(session)
        assert (await executor.execute_cell("a", first)).success
        assert (await executor.execute_cell("b", second)).success
        # Each failed run would store the file again if it were not ready under b's id.
        failing = second + "raise RuntimeError\n"
        cell = session.notebook_state.get_cell("b")
        cell.source = failing
        session.re_analyze_cell("b")
        for _ in range(3):
            assert not (await executor.execute_cell("b", failing)).success

        session.get_artifact_manager().prune(keep_superseded=0, min_idle_seconds=0)

        cell.source = second
        session.re_analyze_cell("b")
        read = await executor.execute_cell("r", reader)
        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == "same"

    async def test_a_crashs_folder_stays_as_long_as_the_good_one_it_follows(self, tmp_path):
        """Pruned with superseded values kept, a failed version stays, files and all, until a
        later run stores a good one and the good one it follows is pruned too."""
        crash = (
            "# @output_dir ckpt\n"
            "import os\n"
            "(ckpt / 'e.pt').write_bytes(os.urandom(16))\n"
            "raise RuntimeError('lost')\n"
        )
        session = _session(tmp_path, writer=OLD, reader=None)
        executor = CellExecutor(session)
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        artifact_id = manager.cell_artifact_id("w", "ckpt")
        assert (await executor.execute_cell("w", OLD)).success
        cell = session.notebook_state.get_cell("w")
        cell.source = crash
        session.re_analyze_cell("w")
        # Two crashes: the newer is the id's top version, which is spared anyway.
        for _ in range(2):
            assert not (await executor.execute_cell("w", crash)).success
        first_crash, _ = _manifest(session, version=2)

        manager.prune(keep_superseded=1, min_idle_seconds=0)

        assert store.get_artifact(artifact_id, 2).state == "failed"
        [entry] = first_crash["files"].values()
        file_id = output_dirs.file_artifact_id(manager, "w", entry["sha256"])
        assert store.read_blob(file_id, entry["version"]) is not None

        cell.source = NEW
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", NEW)).success
        manager.prune(keep_superseded=1, min_idle_seconds=0)

        # v1 is the one earlier value kept, and the crashes follow it.
        assert store.get_artifact(artifact_id, 1) is not None
        assert store.get_artifact(artifact_id, 2).state == "failed"
        assert store.read_blob(file_id, entry["version"]) is not None

        newer = NEW + "(ckpt / 'more.txt').write_text('m')\n"
        cell.source = newer
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", newer)).success
        manager.prune(keep_superseded=1, min_idle_seconds=0)
        assert store.get_artifact(artifact_id, 1) is None
        # Spared as v1 was when the sweep began; the next sweep takes them.
        manager.prune(keep_superseded=1, min_idle_seconds=0)

        assert store.get_artifact(artifact_id, 2) is None
        assert store.get_artifact(artifact_id, 3) is None
        assert store.get_latest_version(file_id) is None

    def test_a_failed_version_follows_the_good_one_just_before_it(self, tmp_path):
        """Spared while the good version just below it is stored, not while an older one is."""
        from vestigo.provenance import TransformSpec

        from strata.notebook.artifact_integration import NotebookArtifactManager

        manager = NotebookArtifactManager("nb", tmp_path)
        store = manager.artifact_store
        artifact_id = manager.cell_artifact_id("w", "ckpt")
        spec = TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": output_dirs.CONTENT_TYPE},
            inputs=[],
        )
        # A folder whose first runs crashed has no good version yet: those stay.
        crashed = manager.cell_artifact_id("c", "ckpt")
        runs = {artifact_id: [True, True, False, False, True], crashed: [False, False]}
        for folder, outcomes in runs.items():
            for good in outcomes:
                version = store.create_artifact(
                    folder, provenance_hash=f"{folder}{good}", transform_spec=spec
                )
                if good:
                    store.finalize_artifact(folder, version, "{}", row_count=0, byte_size=0)
                else:
                    store.fail_artifact(folder, version)

        assert sorted(output_dirs.failed_since_ready(manager)) == sorted(
            [(artifact_id, 3), (artifact_id, 4), (crashed, 1), (crashed, 2)]
        )

        store.delete_artifact(artifact_id, 2)  # pruned; v1 (say, pinned) is not what they follow

        assert sorted(output_dirs.failed_since_ready(manager)) == [(crashed, 1), (crashed, 2)]

    async def test_a_file_a_kept_manifest_names_stays_though_stored_again(
        self, tmp_path, monkeypatch
    ):
        session = _session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success
        manager = session.get_artifact_manager()
        manager.artifact_store.retain(
            manager.cell_artifact_id("w", "ckpt"), 1, kind="pin", holder="review"
        )
        # A second copy of each file under the same id, as two runs storing it at once make.
        monkeypatch.setattr(output_dirs, "_reused", lambda store, artifact_id: None)
        assert (await executor.execute_cell_rerun("w", WRITER)).success

        manager.prune(keep_superseded=0, min_idle_seconds=0)

        pinned, _ = _manifest(session, version=1)
        for entry in pinned["files"].values():
            artifact_id = output_dirs.file_artifact_id(manager, "w", entry["sha256"])
            assert manager.artifact_store.get_artifact(artifact_id, entry["version"]) is not None


LOOP = (
    "# @loop max_iter=3 carry=n\n"
    "# @output_dir ckpt\n"
    "(ckpt / f'e{n}.bin').write_bytes(bytes([n]) * 4)\n"
    "(ckpt / 'latest.txt').write_text(str(n))\n"
    "n = n + 1\n"
)


def _loop_session(tmp_path: Path, loop: str = LOOP, fork: str | None = None):
    nb = create_notebook(tmp_path, "Loop folders")
    add_cell_to_notebook(nb, "s", None)
    write_cell(nb, "s", "n = 0\n")
    add_cell_to_notebook(nb, "w", "s")
    write_cell(nb, "w", loop)
    add_cell_to_notebook(nb, "r", "w")
    write_cell(nb, "r", READER)
    if fork is not None:
        add_cell_to_notebook(nb, "f", "r")
        write_cell(nb, "f", fork)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _files_at(session, iteration: int) -> list[str]:
    manager = session.get_artifact_manager()
    artifact = manager.get_iteration_artifact("w", "ckpt", iteration)
    return sorted(
        json.loads(manager.artifact_store.read_blob(artifact.id, artifact.version))["files"]
    )


class TestALoopCell:
    """One folder for the whole loop: each iteration sees, and is stored with, what came before."""

    async def test_each_iteration_adds_to_the_folder_and_the_last_is_the_output(
        self, tmp_path, caplog
    ):
        session = _loop_session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success

        result = await executor.execute_cell("w", LOOP)
        read = await executor.execute_cell("r", READER)

        assert result.success, result.error
        assert [_files_at(session, k) for k in range(3)] == [
            ["e0.bin", "latest.txt"],
            ["e0.bin", "e1.bin", "latest.txt"],
            ["e0.bin", "e1.bin", "e2.bin", "latest.txt"],
        ]
        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["e0.bin", "e1.bin", "e2.bin", "latest.txt"]
        assert _listings(result.display_outputs)[0].startswith("**ckpt**: 4 files, 13 B")
        assert "did not produce it" not in caplog.text  # the folder is not a carried value

    async def test_a_file_is_read_again_only_when_it_changed(self, tmp_path, monkeypatch):
        stored: list[str] = []
        real = output_dirs._store_opened

        def counting(artifact_mgr, cell_id, opener, shown):
            stored.append(shown)
            return real(artifact_mgr, cell_id, opener, shown)

        monkeypatch.setattr(output_dirs, "_store_opened", counting)
        session = _loop_session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success

        assert (await executor.execute_cell("w", LOOP)).success

        # Each checkpoint once; latest.txt, rewritten at the same size, every time.
        assert sorted(stored) == ["e0.bin", "e1.bin", "e2.bin"] + ["latest.txt"] * 3
        manifest, _ = _manifest(session)
        manager = session.get_artifact_manager()
        latest = manifest["files"]["latest.txt"]
        artifact_id = output_dirs.file_artifact_id(manager, "w", latest["sha256"])
        assert manager.artifact_store.read_blob(artifact_id, latest["version"]) == b"2"

    async def test_a_failed_iteration_keeps_what_the_folder_held(self, tmp_path):
        failing = LOOP + "if n == 2:\n    raise RuntimeError('lost')\n"
        session = _loop_session(tmp_path, failing)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success

        result = await executor.execute_cell("w", failing)

        assert not result.success
        assert _files_at(session, 0) == ["e0.bin", "latest.txt"]
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert sorted(manifest["files"]) == ["e0.bin", "e1.bin", "latest.txt"]

    async def test_a_second_run_is_a_cache_hit_that_shows_the_folder(self, tmp_path):
        session = _loop_session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success
        first = await executor.execute_cell("w", LOOP)

        second = await executor.execute_cell("w", LOOP)

        assert second.cache_hit
        assert _listings(second.display_outputs) == _listings(first.display_outputs)

    async def test_a_fork_starts_from_that_iterations_folder(self, tmp_path):
        fork = LOOP.replace("max_iter=3 carry=n", "max_iter=1 carry=n start_from=w@iter=0")
        session = _loop_session(tmp_path, fork=fork)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success
        assert (await executor.execute_cell("w", LOOP)).success

        result = await executor.execute_cell("f", fork)

        assert result.success, result.error
        manifest, _ = _manifest(session, cell_id="f")
        # Iteration 0 left e0 and n == 1, so the resumed iteration adds e1.
        assert sorted(manifest["files"]) == ["e0.bin", "e1.bin", "latest.txt"]

    async def test_a_folder_from_an_older_run_is_not_the_seed(self, tmp_path):
        """The seed carry comes from the latest run, which kept no folder."""
        fork = LOOP.replace("max_iter=3 carry=n", "max_iter=1 carry=n start_from=w@iter=0")
        session = _loop_session(tmp_path, fork=fork)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("s", "n = 0\n")).success
        assert (await executor.execute_cell("w", LOOP)).success
        assert (await executor.execute_cell("w", "# @loop max_iter=3 carry=n\nn = n + 1\n")).success

        result = await executor.execute_cell("f", fork)

        assert result.success, result.error
        manifest, _ = _manifest(session, cell_id="f")
        assert sorted(manifest["files"]) == ["e1.bin", "latest.txt"]


def _held_until_the_loop_runs(monkeypatch, name: str) -> list[bool]:
    """Patch ``output_dirs.<name>`` to wait, while it runs, for a coroutine to run.

    It cannot while the call blocks the event loop, so the wait then ends unanswered.
    Returns, per call, whether the loop ran meanwhile; :func:`_beat` is the coroutine.
    """
    real = getattr(output_dirs, name)
    answered: list[bool] = []

    def held(*args, **kwargs):
        _started.set()
        answered.append(_loop_ran.wait(timeout=10))
        return real(*args, **kwargs)

    _started.clear()
    _loop_ran.clear()
    monkeypatch.setattr(output_dirs, name, held)
    return answered


_started = threading.Event()
_loop_ran = threading.Event()


async def _beat() -> None:
    while not _started.is_set():
        await asyncio.sleep(0.01)
    _loop_ran.set()


class TestOffTheLoop:
    """A folder can hold gigabytes, so storing or rebuilding one leaves the loop free."""

    async def test_a_folder_is_stored_off_the_loop(self, tmp_path, monkeypatch):
        session = _session(tmp_path, reader=None)
        answered = _held_until_the_loop_runs(monkeypatch, "collect")

        beat = asyncio.create_task(_beat())
        written = await CellExecutor(session).execute_cell("w", WRITER)
        await beat

        assert written.success, written.error
        assert answered == [True]

    async def test_an_input_folder_is_rebuilt_off_the_loop(self, tmp_path, monkeypatch):
        session = _session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success
        answered = _held_until_the_loop_runs(monkeypatch, "materialize")

        beat = asyncio.create_task(_beat())
        read = await executor.execute_cell("r", READER)
        await beat

        assert read.success, read.error
        assert answered == [True]


class TestNotYet:
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
        assert written.display_outputs[0]["markdown_text"] == LISTING

    async def test_an_embedded_worker_returns_the_folder(self, tmp_path):
        writer = "# @worker remote\n" + WRITER
        session = _session(tmp_path, writer=writer)
        _remote(session, {"url": "embedded://local"})
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


REMOTE_READER = (
    "# @worker remote\n" + READER + "text = (ckpt / 'final.txt').read_text()\nname = ckpt.name\n"
)


def _remote(session, config: dict) -> None:
    from strata.notebook.models import WorkerBackendType, WorkerSpec

    session.notebook_state.workers = [
        WorkerSpec(name="remote", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=config)
    ]


class TestAFolderInputOnAWorker:
    @pytest.mark.parametrize("transport", ["direct", "signed"])
    async def test_a_worker_cell_reads_a_local_cells_folder(
        self, tmp_path, transport, notebook_executor_server, notebook_personal_server
    ):
        session = _session(tmp_path, reader=REMOTE_READER)
        config = {"url": notebook_executor_server["execute_url"], "transport": transport}
        if transport == "signed":
            config["strata_url"] = notebook_personal_server["base_url"]
        _remote(session, config)
        executor = CellExecutor(session)

        assert (await executor.execute_cell("w", WRITER)).success
        read = await executor.execute_cell("r", REMOTE_READER)

        assert read.success, read.error
        assert read.execution_method == "executor"
        assert read.outputs["seen"]["preview"] == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]
        assert read.outputs["text"]["preview"] == "done"
        assert read.outputs["name"]["preview"] == "ckpt.dir"  # as on the server

    async def test_a_worker_that_cannot_be_asked_is_refused_before_it_runs(self, tmp_path):
        session = _session(tmp_path, reader=REMOTE_READER)
        _remote(session, {"url": "http://127.0.0.1:9/v1/execute"})
        executor = CellExecutor(session)

        assert (await executor.execute_cell("w", WRITER)).success
        read = await executor.execute_cell("r", REMOTE_READER)

        assert not read.success
        assert "could not be asked whether it takes # @output_dir folders as inputs" in read.error

    async def test_an_older_worker_is_refused_before_it_runs(self, tmp_path, monkeypatch):
        from strata.notebook import workers

        asked = []

        async def older(worker, feature):
            asked.append(feature)
            return False

        monkeypatch.setattr(workers, "worker_advertises", older)
        session = _session(tmp_path, reader=REMOTE_READER)
        _remote(session, {"url": "http://127.0.0.1:9/v1/execute"})
        executor = CellExecutor(session)

        assert (await executor.execute_cell("w", WRITER)).success
        read = await executor.execute_cell("r", REMOTE_READER)

        assert not read.success
        assert "cannot take 'ckpt', an # @output_dir folder, as an input" in read.error
        assert asked == ["folder_inputs"]

    def test_the_folder_goes_as_a_tar_the_harness_unpacks(self, tmp_path):
        from strata.notebook.serializer import deserialize_value

        folder = tmp_path / "src"
        (folder / "deep").mkdir(parents=True)
        (folder / "deep" / "a.bin").write_bytes(b"abc")
        (folder / "empty.txt").write_bytes(b"")
        if sys.platform != "win32":
            (folder / "leak").symlink_to(tmp_path / "elsewhere")
        tar = tmp_path / "run" / f"ckpt{output_dirs.TAR_EXTENSION}"
        tar.parent.mkdir()
        output_dirs.pack(folder, tar)

        got = deserialize_value(output_dirs.TAR_CONTENT_TYPE, tar)

        assert got == tmp_path / "run" / "ckpt.dir"
        assert sorted(p.relative_to(got).as_posix() for p in got.rglob("*")) == [
            "deep",
            "deep/a.bin",
            "empty.txt",
        ]
        assert (got / "deep" / "a.bin").read_bytes() == b"abc"

    @pytest.mark.parametrize("kind", ["path", "link"])
    def test_a_tar_cannot_write_outside_its_folder(self, tmp_path, kind):
        import io
        import tarfile

        from strata.notebook.serializer import deserialize_value

        tar = tmp_path / "run" / f"ckpt{output_dirs.TAR_EXTENSION}"
        tar.parent.mkdir()
        with tarfile.open(tar, "w") as out:
            if kind == "path":
                info = tarfile.TarInfo("../escape.bin")
                info.size = 1
                out.addfile(info, io.BytesIO(b"x"))
            else:
                info = tarfile.TarInfo("leak")
                info.type = tarfile.SYMTYPE
                info.linkname = str(tmp_path / "escape.bin")
                out.addfile(info)

        with pytest.raises(tarfile.FilterError):
            deserialize_value(output_dirs.TAR_CONTENT_TYPE, tar)
        assert not (tmp_path / "run" / "escape.bin").exists()
        assert not (tmp_path / "run" / "ckpt.dir" / "leak").is_symlink()


def test_a_folder_name_cannot_leave_the_run_directory(tmp_path, monkeypatch):
    """The parser takes identifiers only; the check holds even if it did not."""
    from strata.notebook import annotations
    from strata.notebook.output_dirs import mount_entries

    real = annotations.parse_annotations

    def loose(source):
        parsed = real(source)
        parsed.output_dirs = ["../escape"]
        return parsed

    monkeypatch.setattr(annotations, "parse_annotations", loose)

    with pytest.raises(ValueError, match="leaves the run directory"):
        mount_entries("x = 1", tmp_path / "run")
    assert not (tmp_path / "escape").exists()


class TestTheUploadUrl:
    def test_it_verifies_and_is_not_a_log_url(self):
        from urllib.parse import parse_qs, urlparse

        from strata.transforms.signed_urls import URLSigner

        signer = URLSigner(b"k" * 32)
        url = signer.generate_output_file_url(base_url="http://s", build_id="b1")
        query = parse_qs(urlparse(url).query)
        expires, signature = float(query["expires_at"][0]), query["signature"][0]

        assert "/v1/builds/b1/output-file" in url
        assert signer.verify_output_file_signature("b1", expires, signature)
        assert not signer.verify_output_file_signature("b2", expires, signature)
        assert not signer.verify_log_signature("b1", expires, signature)

        log = parse_qs(urlparse(signer.generate_log_url(base_url="http://s", build_id="b1")).query)
        assert not signer.verify_output_file_signature(
            "b1", float(log["expires_at"][0]), log["signature"][0]
        )


class TestTheUploadRoute:
    """``POST /v1/builds/{id}/output-file`` stores a file for the run receiving it."""

    @staticmethod
    def _receiver(tmp_path):
        from strata.notebook.output_file_relay import Receiver

        session = _session(tmp_path, reader=None)
        return session, Receiver(
            session.get_artifact_manager(),
            "w",
            upload_dir=tmp_path / "uploads",
            folders=frozenset({"ckpt"}),
        )

    @staticmethod
    def _url(build_id: str, **params: str) -> str:
        from urllib.parse import urlencode

        from strata.server import get_state

        url = get_state().url_signer.generate_output_file_url(
            base_url="http://127.0.0.1", build_id=build_id
        )
        return url.split("http://127.0.0.1", 1)[1] + "&" + urlencode(params)

    def test_a_file_reaches_the_receiving_run(self, tmp_path, notebook_personal_server):
        import httpx

        from strata.notebook.output_file_relay import receiving

        session, receiver = self._receiver(tmp_path)
        base = notebook_personal_server["base_url"]
        with receiving("b1", receiver):
            response = httpx.post(
                base + self._url("b1", folder="ckpt", path="deep/e0.bin"), content=b"x" * 7
            )

        assert response.status_code == 201, response.text
        entry = receiver.received()["ckpt"]["deep/e0.bin"]
        manager = session.get_artifact_manager()
        file_id = manager.cell_artifact_id("w", f"__file__{entry['sha256']}")
        assert manager.artifact_store.read_blob(file_id, entry["version"]) == b"x" * 7
        assert list((tmp_path / "uploads").iterdir()) == []

    def test_an_older_upload_stored_last_does_not_replace_a_newer_one(self, tmp_path):
        import hashlib

        _, receiver = self._receiver(tmp_path)
        older, newer = receiver.begin(), receiver.begin()
        for begun, body in ((newer, b"new"), (older, b"old")):
            path = tmp_path / f"upload-{begun}"
            path.write_bytes(body)
            sha256 = hashlib.sha256(body).hexdigest()
            receiver.accept("ckpt", "e.bin", path, sha256, len(body), begun)

        entry = receiver.received()["ckpt"]["e.bin"]
        assert entry["sha256"] == hashlib.sha256(b"new").hexdigest()

    def test_opening_the_notebook_removes_what_a_crash_left_mid_upload(self, tmp_path):
        import os
        import time

        from strata.notebook import output_file_relay

        session = _session(tmp_path, reader=None)
        uploads = session.path / ".strata" / "uploads"
        uploads.mkdir()
        crashed, running = uploads / ".upload-crashed", uploads / ".upload-running"
        crashed.write_bytes(b"x")
        running.write_bytes(b"y")
        before = time.time() - output_file_relay._STALE_UPLOAD_SECONDS - 60
        os.utime(crashed, (before, before))

        NotebookSession(parse_notebook(session.path), session.path)

        assert not crashed.exists()
        assert running.exists()

    @pytest.mark.parametrize(
        ("build", "params", "status"),
        [
            ("b1", {"folder": "ckpt", "path": "../escape"}, 400),
            ("b1", {"folder": "../x", "path": "e0.bin"}, 400),
            ("b1", {"folder": "ckpt", "path": "/abs"}, 400),
            ("other", {"folder": "ckpt", "path": "e0.bin"}, 409),
            ("b1", {"folder": "undeclared", "path": "e0.bin"}, 400),
        ],
    )
    def test_a_bad_upload_is_refused(
        self, tmp_path, notebook_personal_server, build, params, status
    ):
        import httpx

        from strata.notebook.output_file_relay import receiving

        _, receiver = self._receiver(tmp_path)
        base = notebook_personal_server["base_url"]
        with receiving("b1", receiver):
            response = httpx.post(base + self._url(build, **params), content=b"x")

        assert response.status_code == status, response.text
        assert receiver.received() == {}

    def test_a_forged_signature_is_refused(self, tmp_path, notebook_personal_server):
        import httpx

        from strata.notebook.output_file_relay import receiving

        _, receiver = self._receiver(tmp_path)
        url = self._url("b1", folder="ckpt", path="e0.bin").replace("signature=", "signature=0")
        with receiving("b1", receiver):
            response = httpx.post(notebook_personal_server["base_url"] + url, content=b"x")

        assert response.status_code == 403
        assert receiver.received() == {}


class TestStreamingFromAWorker:
    """On the signed transport a worker uploads settled files while the cell runs."""

    @staticmethod
    def _session_on_worker(tmp_path, writer, executor_server, personal_server):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        session = _session(tmp_path, writer=writer)
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="r",
                config={
                    "url": executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": personal_server["base_url"],
                },
            )
        ]
        return session

    async def test_files_arrive_while_it_runs_and_the_result_only_names_them(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        # e0 is uploaded, then deleted: the result's list keeps it out of the folder.
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            "(ckpt / 'keep.bin').write_bytes(b'k' * 9)\n"
            "time.sleep(1)\n"
            "(ckpt / 'e0.bin').unlink()\n"
            "(ckpt / 'e1.bin').write_bytes(b'1' * 9)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)

        assert written.success, written.error
        assert sorted(executor._received_output_files["w"]["ckpt"]) == [
            "e0.bin",
            "e1.bin",
            "keep.bin",
        ]
        manifest, state = _manifest(session)
        assert state == "ready"
        assert sorted(manifest["files"]) == ["e1.bin", "keep.bin"]

    async def test_a_lost_worker_leaves_what_it_uploaded(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        """The run dies on the worker: no result comes back, the uploads stay.

        Only the uploads: a folder of the same name where the server runs is not the run's.
        """
        import asyncio

        from strata.notebook import output_file_relay

        stray = tmp_path / "cwd" / "output_dirs" / "ckpt"
        stray.mkdir(parents=True)
        (stray / "stray.bin").write_bytes(b"not the run's")
        monkeypatch.chdir(tmp_path / "cwd")
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        # The cell is killed (no result file) only once the server holds e0, so the
        # loss cannot beat the upload; the timeout is a backstop for a broken upload.
        lose = tmp_path / "lose"
        writer = (
            "# @worker remote\n# @timeout 60\n# @output_dir ckpt\n"
            "import os, pathlib, signal, time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            f"while not pathlib.Path({str(lose)!r}).exists():\n"
            "    time.sleep(0.05)\n"
            "os.kill(os.getpid(), signal.SIGKILL)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        run = asyncio.create_task(CellExecutor(session).execute_cell("w", writer))
        while not any(r.received() for r in output_file_relay._receivers.values()):
            assert not run.done(), (await run).error
            await asyncio.sleep(0.05)
        lose.touch()

        result = await run

        assert not result.success
        assert result.remote_error_code == "EXECUTOR_HTTP_ERROR", result.error
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]

    async def test_a_run_past_its_timeout_on_the_worker_fails_as_a_timeout(
        self, tmp_path, notebook_executor_server, notebook_personal_server
    ):
        """The worker answers 408 and the server records the run as timed out."""
        writer = "# @worker remote\n# @timeout 2\nimport time\ntime.sleep(60)\n"
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )

        result = await CellExecutor(session).execute_cell("w", writer)

        assert not result.success
        assert result.remote_error_code == "TIMEOUT", result.error

    async def test_a_folder_emptied_before_the_end_is_empty(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        """Every file sent was deleted: the result lists the folder with none of them."""
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'tmp.bin').write_bytes(b't' * 9)\n"
            "time.sleep(1)\n"
            "(ckpt / 'tmp.bin').unlink()\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)

        assert written.success, written.error
        assert list(executor._received_output_files["w"]["ckpt"]) == ["tmp.bin"]
        manifest, _ = _manifest(session)
        assert manifest["files"] == {}

    async def test_a_cancelled_run_keeps_what_arrived(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        import asyncio

        from strata.notebook import output_file_relay

        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            "time.sleep(60)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        run = asyncio.create_task(CellExecutor(session).execute_cell("w", writer))
        for _ in range(200):
            if any(r.received() for r in output_file_relay._receivers.values()):
                break
            await asyncio.sleep(0.05)

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]


class TestTheStreamer:
    """The worker's poller: never fails the run, and does not resend what is refused."""

    async def test_a_file_gone_between_listing_and_stat_is_skipped(self, tmp_path, monkeypatch):
        from strata.notebook import output_dirs as module
        from strata.notebook.remote_executor import _OutputDirStreamer

        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "here.bin").write_bytes(b"x")
        monkeypatch.setattr(module, "regular_files", lambda folder: ["gone.tmp", "here.bin"])

        states = _OutputDirStreamer(tmp_path, "http://s/u?x=1")._states()

        assert list(states) == [("ckpt", "here.bin")]

    async def test_a_refused_file_is_not_sent_again(self, tmp_path):
        import httpx

        from strata.notebook.remote_executor import _OutputDirStreamer

        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "big.bin").write_bytes(b"x" * 10)
        posts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(str(request.url))
            return httpx.Response(413, json={"detail": "too large"})

        streamer = _OutputDirStreamer(tmp_path, "http://s/u?x=1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            key, state = next(iter(streamer._states().items()))
            for _ in range(3):
                if streamer._due(key, state):
                    await streamer._upload(client, key, state)

        assert len(posts) == 1


def test_a_result_that_lists_no_files_keeps_none_of_the_uploads(tmp_path):
    """An empty list is an answer (nothing held), unlike no list at all (a lost worker)."""
    import json as json_module

    from strata.notebook.output_dirs import STREAMED

    session = _session(tmp_path, reader=None)
    executor = CellExecutor(session)
    executor._received_output_files["w"] = {
        "ckpt": {"gone.bin": {"sha256": "0" * 64, "size": 1, "version": 1}}
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / STREAMED).write_text(json_module.dumps({}))

    listed_none = executor._stage_output_dir(
        "w", "ckpt", run, "p" * 64, {}, source_hash="", source="", env_hash=""
    )
    lost = executor._stage_output_dir(
        "w", "ckpt", None, "q" * 64, {}, source_hash="", source="", env_hash=""
    )

    manager = session.get_artifact_manager()
    read = lambda staged: json_module.loads(  # noqa: E731
        manager.artifact_store.blob_store.read_blob(staged.artifact_id, staged.version)
    )
    assert read(listed_none)["files"] == {}
    assert list(read(lost)["files"]) == ["gone.bin"]

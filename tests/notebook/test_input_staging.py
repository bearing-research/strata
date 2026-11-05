"""Staging an upstream value for a run: bounded memory, the pinned version, and its failures.

What a failure does is pinned as it was before staging streamed: ``_load_input_blobs`` logs and
goes on without that input, and only a missing ``@each`` item is raised.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tracemalloc
from pathlib import Path

import pytest

from strata.notebook import each
from strata.notebook.artifact_integration import BLOB_COPY_CHUNK_BYTES
from strata.notebook.cell_paths import batch
from strata.notebook.executor import CellExecutor
from strata.notebook.file_clone import clone_file
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX cell execution")


def _session(tmp_path: Path, *cells: tuple[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path / "nb", "Staging")
    after = None
    for cell_id, source in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


def _blobs(session: NotebookSession):
    return session.get_artifact_manager().artifact_store.blob_store


def _id(session: NotebookSession, cell_id: str, name: str, variant: str | None = None) -> str:
    return session.get_artifact_manager().cell_artifact_id(cell_id, name, variant=variant)


class _NoFileno:
    """A reader that is only ``read(size)``, as a blob store's contract promises."""

    def __init__(self, data: bytes):
        self.raw = io.BytesIO(data)

    def read(self, size: int = -1) -> bytes:
        return self.raw.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


class _Failing:
    """Hands out *good* bytes, then fails, whatever size it is asked for."""

    def __init__(self, good: bytes):
        self.good = good

    def read(self, size: int = -1) -> bytes:
        if self.good and size >= 0:
            handed, self.good = self.good[:size], self.good[size:]
            return handed
        raise OSError("the disk went away")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _gone(session: NotebookSession, monkeypatch, artifact_id: str) -> None:
    """*artifact_id*'s blob reads as absent."""
    blobs = _blobs(session)
    original = blobs.open_blob_reader
    monkeypatch.setattr(
        blobs,
        "open_blob_reader",
        lambda a, v: None if a == artifact_id else original(a, v),
    )


class _Strict:
    """Serves *reader*, or *size* zeros, and refuses any read that is not a bounded chunk."""

    def __init__(self, reader=None, size: int = 0):
        self.reader = reader
        self.left = size

    def read(self, size: int = -1) -> bytes:
        assert 0 <= size <= BLOB_COPY_CHUNK_BYTES, f"an unbounded read: {size}"
        if self.reader is not None:
            return self.reader.read(size)
        handed = min(size, self.left)
        self.left -= handed
        return bytes(handed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.reader is not None:
            self.reader.close()


class TestClone:
    @staticmethod
    def _latest(session: NotebookSession) -> tuple[str, int]:
        manager = session.get_artifact_manager()
        artifact_id = manager.cell_artifact_id("a", "x")
        return artifact_id, manager.artifact_store.get_latest_version(artifact_id).version

    async def test_a_cell_writing_to_its_input_never_reaches_the_store(self, tmp_path):
        session = _session(tmp_path, ("a", "x = 'stored bytes'\n"), ("b", "print(x)\n"))
        await _run(session, "a")
        artifact_id, version = self._latest(session)
        manager = session.get_artifact_manager()
        stored = manager.load_artifact_data(artifact_id, version)
        path = tmp_path / "input"

        assert manager.copy_blob(artifact_id, version, path)

        assert path.read_bytes() == stored
        assert path.stat().st_nlink == 1
        with path.open("r+b") as written:
            written.write(b"cell was here")
        assert manager.load_artifact_data(artifact_id, version) == stored

    async def test_an_input_has_the_mode_of_a_new_file(self, tmp_path):
        session = _session(tmp_path, ("a", "x = 'stored bytes'\n"), ("b", "print(x)\n"))
        await _run(session, "a")
        new_file = tmp_path / "new"
        new_file.touch()
        path = tmp_path / "input"

        assert session.get_artifact_manager().copy_blob(*self._latest(session), path)

        assert path.stat().st_mode == new_file.stat().st_mode

    @staticmethod
    def _needs_clones(tmp_path: Path) -> None:
        source = tmp_path / "probe-source"
        source.write_bytes(b"bytes")
        with source.open("rb") as opened:
            if not clone_file(opened.fileno(), tmp_path / "probe"):
                pytest.skip("this filesystem has no clones")

    def test_a_clone_has_the_times_of_a_new_file(self, tmp_path):
        self._needs_clones(tmp_path)
        source = tmp_path / "source"
        source.write_bytes(b"old")
        os.utime(source, (1_000_000_000, 1_000_000_000))
        path = tmp_path / "clone"

        with source.open("rb") as opened:
            assert clone_file(opened.fileno(), path)

        assert path.stat().st_mtime > source.stat().st_mtime

    async def test_an_output_folders_files_clone_too(self, tmp_path, monkeypatch):
        self._needs_clones(tmp_path)
        session = _session(
            tmp_path,
            ("a", "# @output_dir out\n(out / 'f.txt').write_text('in a folder')\n"),
            ("b", "print((out / 'f.txt').read_text())\n"),
        )
        await _run(session, "a")

        def no_copy(*_args):
            raise AssertionError("copied instead of cloning")

        monkeypatch.setattr("strata.notebook.output_dirs.shutil.copyfileobj", no_copy)

        result = await _run(session, "b")

        assert "in a folder" in result.stdout

    async def test_an_output_folders_files_copy_where_they_cannot_clone(
        self, tmp_path, monkeypatch
    ):
        session = _session(
            tmp_path,
            ("a", "# @output_dir out\n(out / 'f.txt').write_text('in a folder')\n"),
            ("b", "print((out / 'f.txt').read_text())\n"),
        )
        await _run(session, "a")
        monkeypatch.setattr("strata.notebook.output_dirs.clone_reader", lambda *_args: False)

        result = await _run(session, "b")

        assert "in a folder" in result.stdout

    async def test_a_filesystem_with_clones_reads_nothing(self, tmp_path, monkeypatch):
        self._needs_clones(tmp_path)
        session = _session(tmp_path, ("a", "x = 'stored bytes'\n"), ("b", "print(x)\n"))
        await _run(session, "a")
        manager = session.get_artifact_manager()
        artifact_id, version = self._latest(session)
        stored = manager.load_artifact_data(artifact_id, version)

        def no_copy(*_args):
            raise AssertionError("copied instead of cloning")

        monkeypatch.setattr("strata.notebook.artifact_integration.shutil.copyfileobj", no_copy)
        path = tmp_path / "input"
        assert manager.copy_blob(artifact_id, version, path)
        assert path.read_bytes() == stored

    async def test_a_clone_that_fails_is_copied_instead(self, tmp_path, monkeypatch):
        session = _session(tmp_path, ("a", "x = 'stored bytes'\n"), ("b", "print(x)\n"))
        await _run(session, "a")
        manager = session.get_artifact_manager()
        artifact_id, version = self._latest(session)
        monkeypatch.setattr(
            "strata.notebook.artifact_integration.clone_reader", lambda *_args: False
        )
        path = tmp_path / "input"

        assert manager.copy_blob(artifact_id, version, path)

        assert path.read_bytes() == manager.load_artifact_data(artifact_id, version)

    async def test_a_target_already_there_is_overwritten_as_before(self, tmp_path):
        session = _session(tmp_path, ("a", "x = 'stored bytes'\n"), ("b", "print(x)\n"))
        await _run(session, "a")
        manager = session.get_artifact_manager()
        artifact_id, version = self._latest(session)
        path = tmp_path / "input"
        path.write_bytes(b"old" * 1000)

        assert manager.copy_blob(artifact_id, version, path)

        assert path.read_bytes() == manager.load_artifact_data(artifact_id, version)

    @pytest.mark.parametrize(
        "reader",
        [
            lambda: io.BytesIO(b"from afar"),
            lambda: _NoFileno(b"from afar"),
            lambda: io.BufferedReader(io.BytesIO(b"from afar")),
        ],
        ids=["bytes", "read-only", "buffered-over-bytes"],
    )
    def test_a_reader_with_no_file_is_copied(self, tmp_path, monkeypatch, reader):
        session = _session(tmp_path, ("a", "x = 1\n"))
        manager = session.get_artifact_manager()
        monkeypatch.setattr(manager.artifact_store, "open_blob_reader", lambda *_args: reader())
        path = tmp_path / "input"

        assert manager.copy_blob("a", 1, path)

        assert path.read_bytes() == b"from afar"

    def test_a_reader_over_a_file_is_not_the_files_bytes(self, tmp_path, monkeypatch):
        """A gzip reader's ``fileno`` is the compressed file's: cloning that corrupts the input."""
        packed = tmp_path / "packed.gz"
        with gzip.open(packed, "wb") as writing:
            writing.write(b"what it reads")
        session = _session(tmp_path, ("a", "x = 1\n"))
        manager = session.get_artifact_manager()
        monkeypatch.setattr(
            manager.artifact_store, "open_blob_reader", lambda *_args: gzip.open(packed, "rb")
        )
        path = tmp_path / "input"

        assert manager.copy_blob("a", 1, path)

        assert path.read_bytes() == b"what it reads"

    def test_a_refused_clone_leaves_no_file(self, tmp_path):
        path = tmp_path / "clone"
        # Not a closed descriptor: the new file would take its number and clone itself.
        read_end, write_end = os.pipe()
        try:
            assert clone_file(read_end, path) is False
        finally:
            os.close(read_end)
            os.close(write_end)

        assert not path.exists()

    def test_a_refused_clone_leaves_a_file_already_there(self, tmp_path):
        source = tmp_path / "source"
        source.write_bytes(b"source")
        path = tmp_path / "clone"
        path.write_bytes(b"already there")

        with source.open("rb") as opened:
            assert clone_file(opened.fileno(), path) is False

        assert path.read_bytes() == b"already there"


class TestBoundedMemory:
    @pytest.mark.parametrize("size", [10 * 2**20, 2**30], ids=["10MB", "1GB"])
    def test_a_copy_holds_a_few_chunks_whatever_the_size(self, tmp_path, monkeypatch, size):
        session = _session(tmp_path, ("a", "x = 1\n"))
        manager = session.get_artifact_manager()
        monkeypatch.setattr(
            manager.artifact_store, "open_blob_reader", lambda a, v: _Strict(size=size)
        )

        tracemalloc.start()
        try:
            copied = manager.copy_blob("a", 1, Path(os.devnull))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert copied
        assert peak < 4 * BLOB_COPY_CHUNK_BYTES

    def _strict(self, session, monkeypatch):
        """Reading a blob whole fails; every read of a stream is held to a chunk."""
        blobs = _blobs(session)
        original = blobs.open_blob_reader
        monkeypatch.setattr(blobs, "open_blob_reader", lambda a, v: _Strict(reader=original(a, v)))

    async def test_an_upstream_value_streams_into_the_run(self, tmp_path, monkeypatch):
        session = _session(tmp_path, ("a", "x = 'a value'\n"), ("c", "y = x\n"))
        await _run(session, "a")
        self._strict(session, monkeypatch)
        out = tmp_path / "out"
        out.mkdir()

        specs = CellExecutor(session)._load_input_blobs("c", out)

        assert json.loads((out / specs["x"]["file"]).read_bytes()) == "a value"

    async def test_a_batch_input_streams_into_the_run(self, tmp_path, monkeypatch):
        session = _session(tmp_path, ("a", "x = 1\n"), ("c", "y = x\n"))
        await _run(session, "a")
        uri = f"strata://artifact/{_id(session, 'a', 'x')}@v=1"
        self._strict(session, monkeypatch)
        out = tmp_path / "out"
        out.mkdir()

        spec = batch.materialize_artifact_to_dir(
            session, CellExecutor(session)._batch_context(), uri, out, "x"
        )

        assert json.loads((out / spec["file"]).read_bytes()) == 1

    async def test_a_batch_input_whose_blob_is_gone_is_none(self, tmp_path, monkeypatch):
        session = _session(tmp_path, ("a", "x = 1\n"), ("c", "y = x\n"))
        await _run(session, "a")
        uri = f"strata://artifact/{_id(session, 'a', 'x')}@v=1"
        _gone(session, monkeypatch, _id(session, "a", "x"))

        context = CellExecutor(session)._batch_context()
        assert batch.materialize_artifact_to_dir(session, context, uri, tmp_path, "x") is None


class TestFailures:
    """A value that cannot be staged: the run goes on without it, the others staged."""

    CELLS = (("a", "x = 1\n"), ("b", "y = 2\n"), ("c", "z = (x, y)\n"))

    async def _ready(self, tmp_path):
        session = _session(tmp_path, *self.CELLS)
        await _run(session, "a")
        await _run(session, "b")
        return session

    def _stage(self, session, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        return out, CellExecutor(session)._load_input_blobs("c", out)

    async def test_a_blob_that_is_gone_drops_that_input(self, tmp_path, monkeypatch, caplog):
        session = await self._ready(tmp_path)
        _gone(session, monkeypatch, _id(session, "a", "x"))

        _, specs = self._stage(session, tmp_path)

        assert sorted(specs) == ["y"]
        assert "Failed to load input x" in caplog.text

    async def test_a_read_that_fails_part_way_drops_it_and_leaves_no_file(
        self, tmp_path, monkeypatch, caplog
    ):
        session = await self._ready(tmp_path)
        blobs = _blobs(session)
        original = blobs.open_blob_reader
        target = _id(session, "a", "x")
        monkeypatch.setattr(
            blobs,
            "open_blob_reader",
            lambda a, v: _Failing(b"partial") if a == target else original(a, v),
        )

        out, specs = self._stage(session, tmp_path)

        assert sorted(specs) == ["y"]
        assert "Failed to load input x" in caplog.text
        assert sorted(path.name for path in out.iterdir()) == ["y.json"]

    async def test_a_version_that_is_not_ready_drops_that_input(self, tmp_path, monkeypatch):
        session = await self._ready(tmp_path)
        store = session.get_artifact_manager().artifact_store
        original = store.get_artifact
        target = _id(session, "a", "x")

        def failed(artifact_id, version):
            found = original(artifact_id, version)
            return found.model_copy(update={"state": "failed"}) if artifact_id == target else found

        monkeypatch.setattr(store, "get_artifact", failed)

        _, specs = self._stage(session, tmp_path)

        assert sorted(specs) == ["y"]

    async def test_a_run_goes_on_without_the_input(self, tmp_path, monkeypatch):
        session = await self._ready(tmp_path)
        _gone(session, monkeypatch, _id(session, "a", "x"))
        source = session.notebook_state.get_cell("c").source

        result = await CellExecutor(session).execute_cell_force("c", source)

        assert not result.success
        assert "NameError" in result.error


class TestEachItems:
    SOURCE = "# @each item in items\nout = item\n"

    async def _ready(self, tmp_path):
        session = _session(
            tmp_path,
            ("a", 'items = {"p": 1}\n'),
            ("e", self.SOURCE),
            ("c", "seen = out\n"),
        )
        await _run(session, "a")
        await _run(session, "e")
        return session

    def _instance(self, session) -> str:
        return _id(session, "e", "out", each.instance_name("p"))

    def _stage(self, session, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        return out, CellExecutor(session)._load_input_blobs("c", out)

    async def test_a_consumer_reads_the_version_the_manifest_pins(self, tmp_path):
        session = await self._ready(tmp_path)
        manager = session.get_artifact_manager()
        manager.store_cell_output(
            cell_id="e",
            variable_name="out",
            blob_data=json.dumps(2).encode(),
            content_type="json/object",
            provenance_hash="a later run",
            variant=each.instance_name("p"),
        )
        assert manager.artifact_store.get_latest_version(self._instance(session)).version == 2

        out, specs = self._stage(session, tmp_path)

        variant = specs["out"]["variants"]["p"]
        assert json.loads((out / variant["file"]).read_bytes()) == 1

    async def test_an_item_with_no_row_is_raised(self, tmp_path, monkeypatch):
        session = await self._ready(tmp_path)
        store = session.get_artifact_manager().artifact_store
        original = store.get_artifact
        target = self._instance(session)
        monkeypatch.setattr(
            store, "get_artifact", lambda a, v: None if a == target else original(a, v)
        )

        with pytest.raises(each.EachError, match="which is gone"):
            self._stage(session, tmp_path)

    async def test_an_item_whose_blob_is_gone_drops_the_input(self, tmp_path, monkeypatch):
        session = await self._ready(tmp_path)
        _gone(session, monkeypatch, self._instance(session))

        _, specs = self._stage(session, tmp_path)

        assert specs == {}


class TestManifests:
    """Parsed whole: their bytes are not a value to stream."""

    async def test_an_each_manifest_is_read_whole(self, tmp_path, monkeypatch):
        session = await TestEachItems()._ready(tmp_path)
        manager = session.get_artifact_manager()
        whole: list[str] = []
        original = manager.load_artifact_data

        def recording(artifact_id, version):
            whole.append(artifact_id)
            return original(artifact_id, version)

        monkeypatch.setattr(manager, "load_artifact_data", recording)
        out = tmp_path / "out"
        out.mkdir()

        CellExecutor(session)._load_input_blobs("c", out)

        assert _id(session, "e", "out") in whole

    async def test_an_output_folder_is_rebuilt_beside_its_manifest(self, tmp_path):
        session = _session(
            tmp_path,
            ("a", '# @output_dir model\n(model / "w.txt").write_text("kept")\n'),
            ("c", 'seen = (model / "w.txt").read_text()\n'),
        )
        await _run(session, "a")
        out = tmp_path / "out"
        out.mkdir()

        CellExecutor(session)._load_input_blobs("c", out)

        assert (out / "model.dir" / "w.txt").read_text() == "kept"

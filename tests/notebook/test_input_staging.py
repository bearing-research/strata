"""Staging an upstream value for a run: bounded memory, the pinned version, and its failures.

What a failure does is pinned as it was before staging streamed: ``_load_input_blobs`` logs and
goes on without that input, and only a missing ``@each`` item is raised.
"""

from __future__ import annotations

import json
import os
import sys
import tracemalloc
from pathlib import Path

import pytest

from strata.notebook import each
from strata.notebook.artifact_integration import BLOB_COPY_CHUNK_BYTES
from strata.notebook.executor import CellExecutor
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

        spec = CellExecutor(session)._materialize_artifact_to_dir(uri, out, "x")

        assert json.loads((out / spec["file"]).read_bytes()) == 1

    async def test_a_batch_input_whose_blob_is_gone_is_none(self, tmp_path, monkeypatch):
        session = _session(tmp_path, ("a", "x = 1\n"), ("c", "y = x\n"))
        await _run(session, "a")
        uri = f"strata://artifact/{_id(session, 'a', 'x')}@v=1"
        _gone(session, monkeypatch, _id(session, "a", "x"))

        assert CellExecutor(session)._materialize_artifact_to_dir(uri, tmp_path, "x") is None


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

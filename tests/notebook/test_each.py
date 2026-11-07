"""``# @each``: a cell that runs once per file or row, and caches each item separately.

These run real cells. What a run did is read back from the cell itself: each instance
appends its item to a log file, so an item that did not run left no line.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from vestigo.store import ArtifactStore

from strata.api.publication_bundle import _write_folder_files
from strata.artifact.cli import cmd_pull
from strata.artifact.transfer import copy_chain
from strata.notebook import each, output_dirs
from strata.notebook.datasets import DatasetError, content_type_of
from strata.notebook.executor import CellExecutor, is_cell_batchable
from strata.notebook.models import CellStatus, WorkerBackendType, WorkerSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.snapshot import build_artifact_index, write_committed_files, write_snapshot
from strata.notebook.snapshot_import import import_snapshot
from strata.notebook.table_writes import TableWriteError
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    set_variant_mode,
    write_cell,
)

windows = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local warehouse paths fail on Windows"
)


def _dialect():
    """The ledger's SQL dialect; the Postgres run of this suite replaces it."""
    return None


def _open(path: Path) -> NotebookSession:
    return NotebookSession(parse_notebook(path), path, ledger_dialect=_dialect())


def _cli_store(artifact_dir: Path) -> str | None:
    """The ``--artifact-dir`` a CLI command opens the notebook's store by."""
    return str(artifact_dir)


class Notebook:
    """A notebook of cells ``a``, ``b``, ... in order, and the log its instances append to."""

    def __init__(self, tmp_path: Path, *sources: str) -> None:
        self.tmp_path = tmp_path
        self.log = tmp_path / "ran.log"
        path = create_notebook(tmp_path / "nb", "Each")
        after = None
        self.ids = [chr(ord("a") + i) for i in range(len(sources))]
        for cell_id, source in zip(self.ids, sources, strict=True):
            add_cell_to_notebook(path, cell_id, after)
            write_cell(path, cell_id, source)
            after = cell_id
        self.session = _open(path)
        self.session.refresh_environment_runtime()

    def ran(self) -> list[str]:
        """The items that ran since the last call."""
        if not self.log.exists():
            return []
        lines = sorted(self.log.read_text().splitlines())
        self.log.unlink()
        return lines

    async def run(self, cell_id: str, source: str | None = None):
        cell = self.session.notebook_state.get_cell(cell_id)
        if source is not None and source != cell.source:
            cell.source = source
            self.session.re_analyze_cell(cell_id)
        return await CellExecutor(self.session).execute_cell(cell_id, cell.source)

    async def good(self, cell_id: str, source: str | None = None):
        result = await self.run(cell_id, source)
        assert result.success, result.error
        return result

    def versions(self) -> list:
        manager = self.session.get_artifact_manager()
        return manager.artifact_store.list_versions_by_id_prefix(f"nb_{manager.notebook_id}_cell_")

    def manifest(self, cell_id: str, var: str) -> dict[str, tuple[str, int]]:
        manager = self.session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id(cell_id, var))
        assert latest.params["content_type"] == each.CONTENT_TYPE
        blob = manager.load_artifact_data(latest.id, latest.version)
        return {key: (id_, v) for key, id_, v in each.manifest_refs(latest.id, blob)}


def _logs(log: Path, expression: str) -> str:
    return f"with open({str(log)!r}, 'a') as f:\n    f.write(str({expression}) + '\\n')\n"


def _over_files(tmp_path: Path, extra: str = "", body: str = "out = path.read_text().upper()\n"):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    log = tmp_path / "ran.log"
    return (
        raw,
        f"# @mount raw {raw.as_uri()} ro\n# @each path in raw{extra}\n"
        + _logs(log, "path.name")
        + body,
    )


def _touch(path: Path, text: str, tick: int = 0) -> None:
    path.write_text(text)
    # A digest reads the mtime; a test does not wait for the clock to move.
    stamp = 1_700_000_000 + tick
    os.utime(path, (stamp, stamp))


READER = "seen = dict(out)\n"


class TestFiles:
    async def test_runs_once_per_file_and_a_consumer_sees_each_by_key(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source, READER + "total = len(seen)\nprint(sorted(seen.items()))\n")

        await nb.good("a")
        assert nb.ran() == ["one.txt", "two.txt"]

        result = await nb.good("b")
        assert "[('one.txt', 'A'), ('two.txt', 'B')]" in result.stdout

    async def test_nothing_changed_is_a_cell_level_hit_that_lists_nothing(
        self, tmp_path, monkeypatch
    ):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        await nb.good("b")
        nb.ran()

        def must_not_list(*_args, **_kwargs):
            raise AssertionError("a cell-level hit lists no items")

        monkeypatch.setattr(each, "items_of_mount", must_not_list)
        result = await nb.good("a")

        assert result.cache_hit
        assert nb.ran() == []

    async def test_a_changed_file_runs_alone(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source, READER + "print(sorted(seen.items()))\n")
        await nb.good("a")
        nb.ran()
        before = nb.manifest("a", "out")

        _touch(raw / "two.txt", "changed", tick=1)
        result = await nb.good("a")

        assert nb.ran() == ["two.txt"]
        assert not result.cache_hit
        after = nb.manifest("a", "out")
        assert after["one.txt"] == before["one.txt"]
        assert after["two.txt"][1] > before["two.txt"][1]
        assert "('two.txt', 'CHANGED')" in (await nb.good("b")).stdout

    async def test_the_mount_is_listed_once_a_run_not_once_an_item(self, tmp_path):
        raw, source = _over_files(tmp_path)
        for name in ("one", "two", "three"):
            _touch(raw / f"{name}.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        executor = CellExecutor(nb.session)
        prepared = []
        prepare_mounts = executor._prepare_mounts

        async def counted(specs, cell_id=None, **kwargs):
            prepared.extend(spec.name for spec in specs)
            return await prepare_mounts(specs, cell_id, **kwargs)

        executor._prepare_mounts = counted
        assert (await executor.execute_cell("a", source)).success

        assert len(nb.ran()) == 3
        assert prepared.count("raw") == 1

    async def test_another_read_only_mount_is_listed_once_a_run_too(self, tmp_path):
        raw, source = _over_files(tmp_path)
        for name in ("one", "two", "three"):
            _touch(raw / f"{name}.txt", "a")
        other = tmp_path / "other"
        other.mkdir()
        source = f"# @mount other {other.as_uri()} ro\n{source}"
        nb = Notebook(tmp_path, source, READER)
        executor = CellExecutor(nb.session)
        prepared = []
        prepare_mounts = executor._prepare_mounts

        async def counted(specs, cell_id=None, **kwargs):
            prepared.extend(spec.name for spec in specs)
            return await prepare_mounts(specs, cell_id, **kwargs)

        executor._prepare_mounts = counted
        assert (await executor.execute_cell("a", source)).success

        assert len(nb.ran()) == 3
        assert prepared.count("other") == 1

    @pytest.mark.parametrize("reader", [READER, None])
    async def test_a_cell_ending_in_a_display_still_runs_a_changed_file_alone(
        self, tmp_path, reader
    ):
        raw, source = _over_files(tmp_path, body="out = path.read_text().upper()\nout\n")
        for name in ("one", "two", "three"):
            _touch(raw / f"{name}.txt", name)
        nb = Notebook(tmp_path, source, *([reader] if reader else []))
        await nb.good("a")
        assert len(nb.ran()) == 3

        # The first in order: its new display is what the next item's check tripped on.
        _touch(raw / "one.txt", "changed", tick=1)
        await nb.good("a")

        assert nb.ran() == ["one.txt"]

    async def test_a_link_to_a_file_is_an_item(self, tmp_path):
        raw, source = _over_files(tmp_path)
        target = tmp_path / "blobs" / "abc"
        target.parent.mkdir()
        _touch(target, "linked")
        (raw / "one.txt").symlink_to(target)
        (raw / "gone.txt").symlink_to(tmp_path / "nothing")
        nb = Notebook(tmp_path, source, READER + "print(seen)\n")

        await nb.good("a")

        assert "{'one.txt': 'LINKED'}" in (await nb.good("b")).stdout
        # The link's target moved on, so the fingerprint and the item both did.
        _touch(target, "changed", tick=1)
        await nb.good("a")
        assert "{'one.txt': 'CHANGED'}" in (await nb.good("b")).stdout

    async def test_a_mount_of_one_file_is_refused_rather_than_empty(self, tmp_path):
        target = tmp_path / "one.txt"
        _touch(target, "a")
        source = f"# @mount raw {target.as_uri()} ro\n# @each path in raw\nout = 1\n"
        nb = Notebook(tmp_path, source, READER)

        result = await nb.run("a")

        assert not result.success
        assert "needs a folder" in result.error

    async def test_a_new_file_runs_alone(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        nb.ran()

        _touch(raw / "two.txt", "b")
        await nb.good("a")

        assert nb.ran() == ["two.txt"]
        assert set(nb.manifest("a", "out")) == {"one.txt", "two.txt"}

    async def test_a_removed_file_leaves_the_manifest_and_runs_nothing(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        nb.ran()

        (raw / "two.txt").unlink()
        await nb.good("a")

        assert nb.ran() == []
        assert set(nb.manifest("a", "out")) == {"one.txt"}

    async def test_a_reverted_file_finds_its_old_value(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER + "print(seen)\n")
        await nb.good("a")
        nb.ran()
        _touch(raw / "one.txt", "bb", tick=1)
        await nb.good("a")
        nb.ran()

        _touch(raw / "one.txt", "a")
        await nb.good("a")

        assert nb.ran() == []
        assert "{'one.txt': 'A'}" in (await nb.good("b")).stdout

    async def test_glob_chooses_the_files(self, tmp_path):
        raw, source = _over_files(tmp_path, " glob=*.csv")
        _touch(raw / "one.csv", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source)

        await nb.good("a")

        assert nb.ran() == ["one.csv"]

    async def test_a_nested_file_keeps_its_relative_path(self, tmp_path):
        raw, source = _over_files(tmp_path)
        (raw / "sub").mkdir()
        _touch(raw / "sub" / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)

        await nb.good("a")

        assert set(nb.manifest("a", "out")) == {"sub/one.txt"}

    async def test_the_new_source_recomputes_every_item(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        nb.ran()

        await nb.good("a", source + "other = 1\n")

        assert nb.ran() == ["one.txt", "two.txt"]

    @pytest.mark.parametrize("kind", ["nocache", "rw"])
    async def test_a_cell_that_must_run_runs_every_item(self, tmp_path, kind):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        line = "# @nocache\n" if kind == "nocache" else f"# @mount out_dir {scratch.as_uri()} rw\n"
        nb = Notebook(tmp_path, line + source, READER)
        await nb.good("a")
        assert nb.ran() == ["one.txt"]

        await nb.good("a")

        assert nb.ran() == ["one.txt"]

    async def test_an_empty_folder_stores_an_empty_manifest(self, tmp_path):
        _raw, source = _over_files(tmp_path)
        nb = Notebook(tmp_path, source, READER + "print(seen)\n")

        await nb.good("a")

        assert nb.manifest("a", "out") == {}
        assert "{}" in (await nb.good("b")).stdout


class TestVariables:
    @staticmethod
    def _item_cell(log: Path, name: str = "item") -> str:
        return f"# @each {name} in samples\n" + _logs(log, name) + f"out = {name}\n"

    async def test_a_dict_by_key_only_a_changed_value_runs(self, tmp_path):
        log = tmp_path / "ran.log"
        nb = Notebook(tmp_path, 'samples = {"x": 1, "y": 2}\n', self._item_cell(log), READER)
        await nb.good("a")
        await nb.good("b")
        assert nb.ran() == ["1", "2"]
        before = nb.manifest("b", "out")

        await nb.good("a", 'samples = {"x": 1, "y": 3}\n')
        await nb.good("b")

        assert nb.ran() == ["3"]
        after = nb.manifest("b", "out")
        assert after["x"] == before["x"]
        assert after["y"] != before["y"]

    async def test_a_list_keys_on_content_so_an_insert_moves_nothing(self, tmp_path):
        log = tmp_path / "ran.log"
        nb = Notebook(tmp_path, 'samples = ["p", "q"]\n', self._item_cell(log), READER)
        await nb.good("a")
        await nb.good("b")
        assert nb.ran() == ["p", "q"]

        await nb.good("a", 'samples = ["new", "p", "q"]\n')
        await nb.good("b")

        assert nb.ran() == ["new"]
        assert set(nb.manifest("b", "out")) == {"new", "p", "q"}

    async def test_a_table_by_its_key_column(self, tmp_path):
        log = tmp_path / "ran.log"
        table = 'import pyarrow as pa\nreadings = pa.table({"id": [1, 2], "v": [10, %d]})\n'
        cell = "# @each row in readings key=id\n" + _logs(log, "row['id']") + "out = row['v']\n"
        nb = Notebook(tmp_path, table % 20, cell, READER + "print(sorted(seen.items()))\n")
        await nb.good("a")
        await nb.good("b")
        assert nb.ran() == ["1", "2"]

        await nb.good("a", table % 21)
        await nb.good("b")

        assert nb.ran() == ["2"]
        assert "[('1', 10), ('2', 21)]" in (await nb.good("c")).stdout

    async def test_a_null_reaches_the_cell_as_none_and_is_not_an_empty_string(self, tmp_path):
        log = tmp_path / "ran.log"
        table = 'import pyarrow as pa\nreadings = pa.table({"id": [1], "v": %s})\n'
        nulls = table % "pa.array([None], pa.string())"
        cell = "# @each row in readings key=id\n" + _logs(log, "row['v']") + "out = row['v']\n"
        nb = Notebook(tmp_path, nulls, cell, READER + "print(seen)\n")
        await nb.good("a")
        await nb.good("b")
        assert nb.ran() == ["None"]
        assert "{'1': None}" in (await nb.good("c")).stdout

        await nb.good("a", table % '[""]')
        await nb.good("b")

        assert nb.ran() == [""]

    async def test_another_input_recomputes_every_item(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = "# @each item in samples\n" + _logs(log, "item") + "out = item * scale\n"
        first = 'samples = ["p", "q"]\nscale = 2\n'
        nb = Notebook(tmp_path, first, cell)
        await nb.good("a")
        await nb.good("b")
        nb.ran()

        await nb.good("a", 'samples = ["p", "q"]\nscale = 3\n')
        await nb.good("b")

        assert nb.ran() == ["p", "q"]

    @pytest.mark.parametrize(
        ("producer", "cell", "expected"),
        [
            ("samples = {1, 2}\n", "# @each s in samples\nout = s\n", "expected a dict"),
            ("samples = 3\n", "# @each s in samples\nout = s\n", "expected a dict"),
            (
                'import pyarrow as pa\nsamples = pa.table({"id": [1]})\n',
                "# @each s in samples\nout = s\n",
                "needs key=",
            ),
            (
                'import pyarrow as pa\nsamples = pa.table({"id": [1]})\n',
                "# @each s in samples key=nope\nout = s\n",
                "no column",
            ),
            (
                'import pyarrow as pa\nsamples = pa.table({"id": [1, 1]})\n',
                "# @each s in samples key=id\nout = s\n",
                "two rows",
            ),
            (
                'import pyarrow as pa\nsamples = pa.table({"id": [None, 1]})\n',
                "# @each s in samples key=id\nout = s\n",
                "null",
            ),
            ("samples = [1]\n", "# @each s in samples glob=*.csv\nout = s\n", "not glob="),
        ],
    )
    async def test_a_source_that_cannot_be_iterated_refuses_the_run(
        self, tmp_path, producer, cell, expected
    ):
        nb = Notebook(tmp_path, producer, cell, READER)
        await nb.good("a")

        result = await nb.run("b")

        assert not result.success
        assert expected in result.error


class TestFailure:
    async def test_a_failed_item_stores_no_manifest_and_the_rest_stay_cached(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = (
            "# @each item in samples\n"
            + _logs(log, "item")
            + "if item == 'bad':\n    raise ValueError('no')\nout = item\n"
        )
        nb = Notebook(tmp_path, 'samples = ["good", "bad"]\n', cell, READER)
        await nb.good("a")

        result = await nb.run("b")

        assert not result.success
        assert result.error == "Item 'bad' failed: ValueError: no"
        assert result.traceback.startswith("Item 'bad' failed:\nTraceback")
        assert "ValueError: no" in result.traceback
        assert nb.ran() == ["bad", "good"]
        manager = nb.session.get_artifact_manager()
        artifact_id = manager.cell_artifact_id("b", "out")
        assert manager.artifact_store.get_latest_version(artifact_id) is None

        await nb.good("a", 'samples = ["good", "fine"]\n')
        await nb.good("b")
        assert nb.ran() == ["fine"]
        assert set(nb.manifest("b", "out")) == {"good", "fine"}

    async def test_a_changed_item_that_fails_is_not_kept_at_its_old_value(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = (
            "# @each item in samples\n"
            + _logs(log, "item")
            + "if item == 'bad':\n    raise ValueError('no')\nout = item\n"
        )
        nb = Notebook(tmp_path, 'samples = {"x": "good"}\n', cell, READER)
        await nb.good("a")
        await nb.good("b")
        manager = nb.session.get_artifact_manager()
        manifest_id = manager.cell_artifact_id("b", "out")
        before = manager.artifact_store.get_latest_version(manifest_id)
        nb.ran()

        await nb.good("a", 'samples = {"x": "bad"}\n')
        assert not (await nb.run("b")).success

        after = manager.artifact_store.get_latest_version(manifest_id)
        # Its latest instance is the old value; a manifest here would call that this run's.
        assert (after.id, after.version) == (before.id, before.version)
        assert nb.ran() == ["bad"]
        assert not (await nb.run("b")).success
        assert nb.ran() == ["bad"]

    async def test_a_failed_item_leaves_the_cell_naming_its_manifest(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = (
            "# @each item in samples\n"
            + _logs(log, "item")
            + "if item == 'bad':\n    raise ValueError('no')\nout = item\n"
        )
        nb = Notebook(tmp_path, 'samples = ["good"]\n', cell, READER)
        await nb.good("a")
        await nb.good("b")
        state = nb.session.notebook_state.get_cell("b")
        before = dict(state.artifact_uris)
        assert "@variant=" not in before["out"]

        # The new item stores a value and the next fails: both name themselves meanwhile.
        await nb.good("a", 'samples = ["new", "bad"]\n')
        assert not (await nb.run("b")).success

        assert state.artifact_uris == before
        assert state.artifact_uri == before["out"]

    async def test_a_failed_item_keeps_the_last_manifest(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = (
            "# @each item in samples\n"
            + _logs(log, "item")
            + "if item == 'bad':\n    raise ValueError('no')\nout = item\n"
        )
        nb = Notebook(tmp_path, 'samples = ["good"]\n', cell, READER)
        await nb.good("a")
        await nb.good("b")

        await nb.good("a", 'samples = ["good", "bad"]\n')
        assert not (await nb.run("b")).success

        assert set(nb.manifest("b", "out")) == {"good"}


SHOWS = "out = path.read_text().upper()\nf'shows {out}'\n"


def _shown(outputs) -> list[tuple[str | None, object]]:
    return [
        (o["variant"], o["preview"]) if isinstance(o, dict) else (o.variant, o.preview)
        for o in outputs
    ]


class TestDisplays:
    """An item's displays show under its key, in the order the items ran."""

    async def _two(self, tmp_path, *more: str) -> tuple[Path, Notebook]:
        raw, source = _over_files(tmp_path, body=SHOWS)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        return raw, Notebook(tmp_path, source, *more)

    async def test_a_run_shows_each_items_display_by_its_key(self, tmp_path):
        _, nb = await self._two(tmp_path, READER)

        result = await nb.good("a")

        expected = [("one.txt", "shows A"), ("two.txt", "shows B")]
        assert _shown(result.display_outputs) == expected
        assert _shown(nb.session.notebook_state.get_cell("a").display_outputs) == expected

    async def test_a_hit_that_lists_nothing_shows_them_again(self, tmp_path, monkeypatch):
        _, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        nb.session.notebook_state.get_cell("a").display_outputs = []

        def must_not_list(*_args, **_kwargs):
            raise AssertionError("a cell-level hit lists no items")

        monkeypatch.setattr(each, "items_of_mount", must_not_list)
        again = await nb.good("a")

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one.txt", "shows A"), ("two.txt", "shows B")]
        assert _shown(nb.session.notebook_state.get_cell("a").display_outputs) == [
            ("one.txt", "shows A"),
            ("two.txt", "shows B"),
        ]

    async def test_a_changed_item_shows_beside_the_cached_ones(self, tmp_path):
        raw, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        nb.ran()

        _touch(raw / "one.txt", "c", tick=1)
        again = await nb.good("a")

        assert nb.ran() == ["one.txt"]
        assert _shown(again.display_outputs) == [("one.txt", "shows C"), ("two.txt", "shows B")]

    async def test_a_cell_nothing_reads_shows_them_on_a_hit_too(self, tmp_path):
        _, nb = await self._two(tmp_path)
        await nb.good("a")
        nb.ran()

        again = await nb.good("a")

        assert nb.ran() == []
        assert _shown(again.display_outputs) == [("one.txt", "shows A"), ("two.txt", "shows B")]

    async def test_a_reverted_item_shows_its_earlier_display(self, tmp_path):
        raw, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        _touch(raw / "one.txt", "c", tick=1)
        await nb.good("a")
        _touch(raw / "one.txt", "a")

        again = await nb.good("a")

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one.txt", "shows A"), ("two.txt", "shows B")]

    async def test_a_prune_keeps_what_a_manifest_shows(self, tmp_path, monkeypatch):
        _, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manager.prune(keep_superseded=0, min_idle_seconds=0)
        monkeypatch.setattr(each, "items_of_mount", lambda *_a, **_k: [])

        again = await nb.good("a")

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one.txt", "shows A"), ("two.txt", "shows B")]

    async def test_a_hit_leaves_its_own_for_a_reopen(self, tmp_path):
        raw, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        _touch(raw / "one.txt", "c", tick=1)
        await nb.good("a")
        _touch(raw / "one.txt", "a")
        assert (await nb.good("a")).cache_hit

        path = nb.session.path
        reopened = _open(path)

        assert _shown(reopened.notebook_state.get_cell("a").display_outputs) == [
            ("one.txt", "shows A"),
            ("two.txt", "shows B"),
        ]

    async def test_a_failed_item_shows_nothing(self, tmp_path):
        raw, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        _touch(raw / "two.txt", "", tick=1)
        failing = nb.session.notebook_state.get_cell("a").source + "assert out\n"

        failed = await nb.run("a", failing)

        assert not failed.success
        assert failed.display_outputs == []
        assert nb.session.notebook_state.get_cell("a").display_outputs == []

    async def test_a_copy_to_another_store_names_no_display(self, tmp_path):
        _, nb = await self._two(tmp_path, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "out"))
        blob = manager.load_artifact_data(latest.id, latest.version)
        assert len(each.manifest_displays(blob)) == 2

        moved = each.relocate(blob, latest.id, {})

        assert each.manifest_displays(moved) == []
        assert each.manifest_refs(latest.id, moved) == each.manifest_refs(latest.id, blob)


class TestRetention:
    async def test_prune_sweeps_an_instance_no_manifest_names(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        two = nb.manifest("a", "out")["two.txt"]
        one = nb.manifest("a", "out")["one.txt"]

        (raw / "two.txt").unlink()
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manager.prune(keep_superseded=0, min_idle_seconds=0)

        store = manager.artifact_store
        assert store.get_artifact(*one).state == "ready"
        gone = store.get_artifact(*two)
        assert gone is None or gone.state != "ready"

    async def test_a_consumer_keeps_the_instances_it_read(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER + "print(seen)\n")
        await nb.good("a")
        await nb.good("b")
        old = nb.manifest("a", "out")["one.txt"]

        _touch(raw / "one.txt", "bb", tick=1)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manager.prune(keep_superseded=0, min_idle_seconds=0)

        kept = manager.artifact_store.get_artifact(*old)
        assert kept is not None
        assert kept.state == "ready"


def _table(warehouse: Path) -> list[tuple[str, int]]:
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
    )
    rows = catalog.load_table("lab.rows").scan().to_arrow().sort_by("id")
    return list(zip(rows["id"].to_pylist(), rows["v"].to_pylist(), strict=True))


def _snapshots(warehouse: Path) -> list:
    from pyiceberg.catalog.sql import SqlCatalog

    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
    )
    return list(catalog.load_table("lab.rows").snapshots())


@windows
class TestSync:
    @staticmethod
    def _cell(tmp_path: Path, mode: str = "sync on=id") -> tuple[Path, Path, str]:
        warehouse = tmp_path / "warehouse"
        raw = tmp_path / "raw"
        raw.mkdir()
        source = (
            f"# @mount raw {raw.as_uri()} ro\n"
            "# @each path in raw\n"
            f"# @write_table rows file://{warehouse}#lab.rows {mode}\n"
            + _logs(tmp_path / "ran.log", "path.name")
            + "import pyarrow as pa\n"
            "text = path.read_text()\n"
            "if text == 'bad':\n    raise ValueError('no')\n"
            'rows = pa.table({"id": [path.stem], "v": [len(text)]})\n'
        )
        return warehouse, raw, source

    async def test_every_item_lands_in_one_commit(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source)

        await nb.good("a")

        assert _table(warehouse) == [("one", 1), ("two", 2)]
        # A table's first commit is one snapshot; one per instance would be two, and more.
        assert len(_snapshots(warehouse)) == 1

    async def test_a_removed_file_deletes_its_rows_with_every_instance_cached(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        nb.ran()

        (raw / "two.txt").unlink()
        await nb.good("a")

        assert nb.ran() == []
        assert _table(warehouse) == [("one", 1)]

    async def test_a_changed_file_keeps_the_rows_of_the_cached_ones(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        nb.ran()

        _touch(raw / "two.txt", "bbb", tick=1)
        await nb.good("a")

        assert nb.ran() == ["two.txt"]
        assert _table(warehouse) == [("one", 1), ("two", 3)]

    async def test_nothing_changed_writes_nothing(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        snapshots = len(_snapshots(warehouse))

        assert (await nb.good("a")).cache_hit

        assert len(_snapshots(warehouse)) == snapshots

    async def test_a_failed_item_writes_nothing(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source)
        await nb.good("a")

        # Missing rows would read as deletions.
        _touch(raw / "two.txt", "bad", tick=1)
        assert not (await nb.run("a")).success

        assert _table(warehouse) == [("one", 1), ("two", 2)]

    async def test_no_items_refuses_rather_than_delete_every_row(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source)
        await nb.good("a")

        (raw / "one.txt").unlink()
        result = await nb.run("a")

        assert not result.success
        assert "no items" in result.error
        assert _table(warehouse) == [("one", 1)]

    async def test_reverting_a_removal_writes_again(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        (raw / "two.txt").unlink()
        await nb.good("a")
        assert _table(warehouse) == [("one", 1)]
        nb.ran()

        _touch(raw / "two.txt", "bb")
        await nb.good("a")

        # Every instance is cached and the manifest is an old one; the table is not.
        assert nb.ran() == []
        assert _table(warehouse) == [("one", 1), ("two", 2)]

    async def test_a_reverted_file_finds_its_value_and_writes_again(self, tmp_path):
        warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        _touch(raw / "one.txt", "bb", tick=1)
        await nb.good("a")
        assert _table(warehouse) == [("one", 2)]
        nb.ran()

        _touch(raw / "one.txt", "a")
        await nb.good("a")

        assert nb.ran() == []
        assert _table(warehouse) == [("one", 1)]

    async def test_a_failed_write_keeps_no_manifest(self, tmp_path, monkeypatch):
        _warehouse, raw, source = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source)
        await nb.good("a")
        before = nb.manifest("a", "rows")
        _touch(raw / "two.txt", "bb")

        def refuse(_self, _writes, _outputs):
            raise TableWriteError("the catalog said no")

        monkeypatch.setattr(CellExecutor, "_write_cell_tables", refuse)
        result = await nb.run("a")

        assert not result.success
        assert "the catalog said no" in result.error
        assert nb.manifest("a", "rows") == before

    async def test_an_item_that_is_not_a_table_refuses_the_write(self, tmp_path):
        warehouse, raw, _ = self._cell(tmp_path)
        _touch(raw / "one.txt", "a")
        source = (
            f"# @mount raw {raw.as_uri()} ro\n# @each path in raw\n"
            f"# @write_table rows file://{warehouse}#lab.rows sync on=id\n"
            "rows = {'id': path.stem}\n"
        )
        nb = Notebook(tmp_path, source)

        result = await nb.run("a")

        assert not result.success
        assert "not a table" in result.error

    @pytest.mark.parametrize("mode", ["append", "overwrite", "merge on=id"])
    async def test_the_other_modes_are_refused(self, tmp_path, mode):
        _warehouse, raw, source = self._cell(tmp_path, mode)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source)

        result = await nb.run("a")

        assert not result.success
        assert "each" in result.error
        assert not nb.log.exists()


class TestASnapshot:
    @staticmethod
    def _export(nb: Notebook, out: Path) -> Path:
        with zipfile.ZipFile(out, "w") as archive:
            write_committed_files(nb.session, archive)
            write_snapshot(nb.session, archive, include="all")
        return out

    async def test_an_import_under_a_new_id_still_reads_every_item(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source, READER + "print(sorted(seen.items()))\n")
        await nb.good("a")
        await nb.good("b")
        bundle = self._export(nb, tmp_path / "snap.zip")

        imported = import_snapshot(
            bundle, tmp_path / "dst", taken_ids={nb.session.notebook_state.id}
        )
        assert imported.replaced_id == nb.session.notebook_state.id

        session = NotebookSession(parse_notebook(imported.notebook_dir), imported.notebook_dir)
        session.refresh_environment_runtime()
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "out"))
        refs = each.manifest_refs(latest.id, manager.load_artifact_data(latest.id, latest.version))
        assert all(
            artifact_id.startswith(f"nb_{imported.notebook_id}_") for _, artifact_id, _ in refs
        )
        # What it requires is held under the new id, so a sweep keeps it.
        manager.prune(keep_superseded=0, min_idle_seconds=0)

        executor = CellExecutor(session)
        result = await executor.execute_cell_force("b", session.notebook_state.get_cell("b").source)
        assert result.success, result.error
        assert "[('one.txt', 'A'), ('two.txt', 'BB')]" in result.stdout

    async def test_it_carries_the_version_a_manifest_names_not_the_latest(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = (
            "# @each item in samples\n"
            + _logs(log, "item")
            + "if item == 'bad':\n    raise ValueError('no')\nout = item\n"
        )
        nb = Notebook(tmp_path, 'samples = {"x": "good", "y": "fine"}\n', cell, READER)
        await nb.good("a")
        await nb.good("b")
        named = nb.manifest("b", "out")["x"]

        # x stores a new version, and y's failure keeps the manifest naming the old one.
        await nb.good("a", 'samples = {"x": "new", "y": "bad"}\n')
        assert not (await nb.run("b")).success
        store = nb.session.get_artifact_manager().artifact_store
        assert store.get_latest_version(named[0]).version > named[1]

        index = build_artifact_index(nb.session)

        assert named in {(e["artifact_id"], e["version"]) for e in index["b"]}


class TestAnotherStore:
    async def test_publishing_a_consumer_carries_every_item_with_it(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "bb")
        nb = Notebook(tmp_path, source, READER, "last = seen\n")
        await nb.good("a")
        await nb.good("b")
        manager = nb.session.get_artifact_manager()
        seen = manager.artifact_store.get_latest_version(manager.cell_artifact_id("b", "seen"))
        central = ArtifactStore(tmp_path / "central")

        written, _landed = copy_chain(manager.artifact_store, central, seen, 10)

        manifest_id = manager.cell_artifact_id("a", "out")
        manifest = central.get_latest_version(manifest_id)
        assert manifest is not None
        refs = each.manifest_refs(manifest.id, central.read_blob(manifest.id, manifest.version))
        assert sorted(key for key, _, _ in refs) == ["one.txt", "two.txt"]
        for _key, artifact_id, version in refs:
            assert f"{artifact_id}@v={version}" in written
            assert central.blob_exists(artifact_id, version)
        # What a kept manifest requires, it holds through a sweep.
        central.publish_artifact(seen.id, seen.version)
        central.garbage_collect(max_idle_days=0, collect_latest=True)
        for _key, artifact_id, version in refs:
            assert central.blob_exists(artifact_id, version)

    async def test_a_value_landing_elsewhere_is_named_there(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        store = manager.artifact_store
        manifest = store.get_latest_version(manager.cell_artifact_id("a", "out"))
        _key, instance_id, version = each.manifest_refs(
            manifest.id, store.read_blob(manifest.id, manifest.version)
        )[0]
        central = ArtifactStore(tmp_path / "central")
        # The target holds this computation under another id, so the value lands there.
        record = store.get_artifact(instance_id, version)
        central.import_artifact(
            replace(record, id="elsewhere"), store.read_blob(instance_id, version)
        )

        copy_chain(store, central, manifest, 10)

        landed = central.get_latest_version(manifest.id)
        [(_, landed_id, landed_version)] = each.manifest_refs(
            landed.id, central.read_blob(landed.id, landed.version)
        )
        assert landed_id == "elsewhere"
        assert central.blob_exists(landed_id, landed_version)

    async def test_a_cached_value_lands_with_the_collection_it_read(self, tmp_path):
        log = tmp_path / "ran.log"
        cell = "# @each item in samples\n" + _logs(log, "item") + "out = item\n"
        nb = Notebook(tmp_path, 'samples = {"x": 1, "y": 2}\n', cell, READER)
        await nb.good("a")
        await nb.good("b")
        manager = nb.session.get_artifact_manager()
        store = manager.artifact_store
        collection_id = manager.cell_artifact_id("a", "samples")
        first = store.get_latest_version(collection_id).version
        kept = nb.manifest("b", "out")["x"]

        await nb.good("a", 'samples = {"x": 1, "y": 3}\n')
        await nb.good("b")
        assert nb.manifest("b", "out")["x"] == kept
        central = ArtifactStore(tmp_path / "central")
        manifest = store.get_latest_version(manager.cell_artifact_id("b", "out"))

        copy_chain(store, central, manifest, 10)

        # x kept reading the first collection, and its edge must not dangle.
        assert central.get_artifact(collection_id, first) is not None

    async def test_a_value_holding_a_folder_lands_with_its_files(self, tmp_path):
        raw, _ = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        source = (
            f"# @mount raw {raw.as_uri()} ro\n# @each path in raw\n# @output_dir out\n"
            "(out / 'copy.txt').write_text(path.read_text())\n"
        )
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manifest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "out"))
        central = ArtifactStore(tmp_path / "central")

        copy_chain(manager.artifact_store, central, manifest, 10)

        [(_, folder_id, folder_version)] = each.manifest_refs(
            manifest.id, central.read_blob(manifest.id, manifest.version)
        )
        [(file_id, file_version, _)] = output_dirs.file_refs(
            folder_id, central.read_blob(folder_id, folder_version)
        )
        assert central.blob_exists(file_id, file_version)

    async def test_a_manifest_naming_a_gone_value_fails_its_reader_by_name(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        store = manager.artifact_store
        manifest = store.get_latest_version(manager.cell_artifact_id("a", "out"))
        store.write_blob(
            manifest.id, manifest.version, each.encode_manifest(manifest.id, [("k", "nope", 1)])
        )

        result = await CellExecutor(nb.session).execute_cell_force("b", READER)

        assert not result.success
        assert "nope@v=1, which is gone" in result.error

    async def test_a_pull_refuses_rather_than_write_a_manifest(self, tmp_path, capsys):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manifest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "out"))
        out = tmp_path / "pulled.arrow"

        code = cmd_pull(
            argparse.Namespace(
                ref=f"{manifest.id}@v={manifest.version}",
                artifact_dir=_cli_store(manager.artifact_dir),
                to=str(out),
                tenant=None,
            )
        )

        assert code == 1
        assert "@each" in capsys.readouterr().err
        assert not out.exists()

    def test_a_bundle_refuses_rather_than_archive_a_manifest(self, tmp_path):
        record = SimpleNamespace(id="x", version=1)

        with pytest.raises(ValueError, match="@each"):
            _write_folder_files(SimpleNamespace(), record, each.CONTENT_TYPE, tmp_path)

    async def test_a_manifest_naming_a_gone_value_is_not_sent(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        manager = nb.session.get_artifact_manager()
        manifest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("a", "out"))
        forged = each.encode_manifest(manifest.id, [("one.txt", "nope", 1)])
        manager.artifact_store.write_blob(manifest.id, manifest.version, forged)
        central = ArtifactStore(tmp_path / "central")

        with pytest.raises(ValueError, match="which is gone"):
            copy_chain(manager.artifact_store, central, manifest, 10)

    def test_a_dataset_refuses_a_manifest(self):
        record = SimpleNamespace(id="x", version=1, params={"content_type": each.CONTENT_TYPE})

        with pytest.raises(DatasetError, match="@each"):
            content_type_of(record)


class TestOtherSurfaces:
    def test_a_batch_leaves_it_out(self, tmp_path):
        nb = Notebook(tmp_path, "samples = [1]\n", "# @each s in samples\nout = s\n")
        executor = CellExecutor(nb.session)
        assert not is_cell_batchable(executor, nb.session.notebook_state.get_cell("b"))

    def test_a_batch_leaves_out_a_cell_reading_it(self, tmp_path):
        nb = Notebook(tmp_path, "samples = [1]\n", "# @each s in samples\nout = s\n", READER)
        executor = CellExecutor(nb.session)
        states = nb.session.notebook_state

        assert is_cell_batchable(executor, states.get_cell("a"))
        assert not is_cell_batchable(executor, states.get_cell("c"))

    @pytest.mark.parametrize(
        ("language", "source"),
        [
            ("prompt", "Summarize {{ out }}\n"),
            ("sql", "-- @sql connection=db\nSELECT :out AS x\n"),
            pytest.param(
                "r",
                "x <- length(out)\n",
                marks=pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R"),
            ),
        ],
    )
    async def test_another_language_refuses_to_read_it(self, tmp_path, language, source):
        nb = Notebook(tmp_path, "samples = [1]\n", "# @each s in samples\nout = s\n")
        add_cell_to_notebook(nb.session.path, "z", "b", language=language)
        write_cell(nb.session.path, "z", source)
        nb.session = _open(nb.session.path)
        nb.session.refresh_environment_runtime()
        await nb.good("a")
        await nb.good("b")

        result = await nb.run("z")

        assert not result.success
        assert "only a Python cell" in result.error

    async def test_a_cell_status_goes_ready(self, tmp_path):
        nb = Notebook(tmp_path, "samples = [1]\n", "# @each s in samples\nout = s\n", READER)
        await nb.good("a")
        await nb.good("b")
        assert nb.session.compute_staleness()["b"].status == CellStatus.READY
        assert nb.session.compute_staleness()["c"].status != CellStatus.READY

    async def test_a_changed_file_makes_it_stale_and_says_so(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        assert nb.session.compute_staleness()["a"].status == CellStatus.READY

        _touch(raw / "one.txt", "bb", tick=1)

        staleness = nb.session.compute_staleness()["a"]
        assert staleness.status != CellStatus.READY
        assert "source changed" not in " ".join(r.value for r in staleness.reasons)

    async def test_rerun_runs_every_item_and_force_too(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        _touch(raw / "two.txt", "b")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        nb.ran()

        executor = CellExecutor(nb.session)
        assert (await executor.execute_cell_rerun("a", source)).success
        assert nb.ran() == ["one.txt", "two.txt"]
        assert (await executor.execute_cell_force("a", source)).success
        assert nb.ran() == ["one.txt", "two.txt"]

    async def test_a_consumer_of_nothing_new_is_a_cache_hit(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        await nb.good("b")

        assert (await nb.good("b")).cache_hit

    async def test_a_fresh_session_finds_the_cell_level_hit(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        await nb.good("b")
        nb.ran()

        reopened = _open(nb.session.path)
        reopened.refresh_environment_runtime()
        assert reopened.compute_staleness()["a"].status == CellStatus.READY
        result = await CellExecutor(reopened).execute_cell("a", source)

        assert result.cache_hit
        assert nb.ran() == []

    @pytest.mark.parametrize(
        ("annotation", "expected"),
        [
            ("# @each x in\n", "each_malformed"),
            ("# @loop max_iter=2 carry=out\n# @each x in raw\n", "each_with_loop"),
        ],
    )
    def test_a_bad_annotation_is_a_diagnostic(self, tmp_path, annotation, expected):
        nb = Notebook(tmp_path, annotation + "out = 1\n")
        codes = [d.code for d in nb.session.notebook_state.get_cell("a").annotation_diagnostics]
        assert expected in codes

    async def test_tests_of_it_are_refused(self, tmp_path):
        nb = Notebook(tmp_path, "samples = [1]\n", "# @each s in samples\nout = s\n")
        await nb.good("a")
        executor = CellExecutor(nb.session)
        result = await executor.run_cell_tests("b", "def test_x():\n    assert True\n")
        assert result.errored == 1
        assert "not supported" in str(result)

    async def test_a_branch_is_refused_and_writes_nothing(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        before = nb.manifest("a", "out")
        nb.ran()

        result = await CellExecutor(nb.session).execute_cell("a", source, init="a-release")

        assert not result.success
        assert "Branching" in result.error
        assert nb.ran() == []
        assert nb.manifest("a", "out") == before

    async def test_a_per_variant_cell_ending_in_a_display_runs_a_changed_variant_alone(
        self, tmp_path
    ):
        log = tmp_path / "ran.log"
        nb = Notebook(
            tmp_path,
            "# @variant g one\nval = 1\n",
            "# @variant g two\nval = 2\n",
            "# @per_variant\n" + _logs(log, "val") + "out = val * 2\nout\n",
            "seen = dict(out)\n",
        )
        set_variant_mode(nb.session.path, "g", "sweep")
        nb.session = _open(nb.session.path)
        nb.session.refresh_environment_runtime()
        await nb.good("c")
        assert nb.ran() == ["1", "2"]

        await nb.good("a", "# @variant g one\nval = 10\n")
        await nb.good("c")

        assert nb.ran() == ["10"]

    @pytest.mark.parametrize(
        ("ending", "reader"),
        [
            pytest.param("", (), id="leaf"),
            pytest.param("out = val\n", ("seen = dict(out)\n",), id="consumed"),
        ],
    )
    async def test_a_reverted_per_variant_cell_replays_each_console(self, tmp_path, ending, reader):
        log = tmp_path / "ran.log"
        prints = "# @per_variant\n" + _logs(log, "val") + "print('{}', val)\n" + ending
        nb = Notebook(
            tmp_path,
            "# @variant g one\nval = 1\n",
            "# @variant g two\nval = 2\n",
            prints.format("first"),
            *reader,
        )
        set_variant_mode(nb.session.path, "g", "sweep")
        nb.session = _open(nb.session.path)
        nb.session.refresh_environment_runtime()
        await nb.good("c")
        assert "second" in (await nb.good("c", prints.format("second"))).stdout
        nb.ran()

        result = await nb.good("c", prints.format("first"))

        assert nb.ran() == []
        assert "first 1" in result.stdout and "first 2" in result.stdout
        assert "second" not in result.stdout

    async def test_a_remote_worker_is_refused(self, tmp_path):
        cell = "# @worker gpu\n# @each s in samples\nout = s\n"
        nb = Notebook(tmp_path, "samples = [1]\n", cell, READER)
        nb.session.notebook_state.workers = [
            WorkerSpec(
                name="gpu", backend=WorkerBackendType.EXECUTOR, config={"url": "http://nowhere"}
            )
        ]
        await nb.good("a")

        result = await nb.run("b")

        assert not result.success
        assert "local worker" in result.error

    async def test_an_instance_does_not_load_the_collection(self, tmp_path):
        cell = "# @each s in samples\nout = 'samples' in globals()\n"
        nb = Notebook(tmp_path, "samples = [1]\n", cell, READER + "print(list(seen.values()))\n")
        await nb.good("a")
        await nb.good("b")

        assert "[False]" in (await nb.good("c")).stdout

    async def test_a_mount_takes_no_key_and_a_writable_one_is_refused(self, tmp_path):
        raw, _ = _over_files(tmp_path)
        cell = f"# @mount raw {raw.as_uri()} ro\n# @each p in raw key=id\nout = 1\n"
        nb = Notebook(tmp_path, cell, READER)
        assert "not key=" in (await nb.run("a")).error

        cell = f"# @mount raw {raw.as_uri()} rw\n# @each p in raw\nout = 1\n"
        assert "read-only" in (await nb.run("a", cell)).error


class TestListing:
    def test_a_remote_file_digests_its_content_and_a_local_one_its_state(self, tmp_path):
        path = tmp_path / "one.txt"
        _touch(path, "a")
        local = each.items_of_mount(tmp_path, None, by_content=False)
        remote = each.items_of_mount(tmp_path, None, by_content=True)

        _touch(path, "a", tick=1)

        assert each.items_of_mount(tmp_path, None, by_content=False) != local
        assert each.items_of_mount(tmp_path, None, by_content=True) == remote

    def test_a_remote_file_takes_the_listings_version_and_hashes_what_has_none(
        self, tmp_path, monkeypatch
    ):
        _touch(tmp_path / "one.txt", "a")
        _touch(tmp_path / "two.txt", "b")
        hashed = []
        sha256 = each._sha256

        def counted(handle):
            hashed.append(Path(handle.name).name)
            return sha256(handle)

        monkeypatch.setattr(each, "_sha256", counted)

        items = each.items_of_mount(
            tmp_path, None, by_content=True, versions={"one.txt": "listed-version"}
        )

        assert items[0].digest == "listed-version"
        assert hashed == ["two.txt"]

    def test_a_list_holds_a_repeated_value_once(self):
        items = each.items_of_json(["p", "p", {"x": 1}, {"x": 1}], "samples")

        assert len(items) == 2
        assert items[0].key == "p"

    def test_a_manifest_names_a_value_only_when_it_is_elsewhere(self):
        beside = each.instance_id("nb_x_cell_a_var_out", "k")

        blob = each.encode_manifest("nb_x_cell_a_var_out", [("k", beside, 3)])

        assert b"nb_x" not in blob
        # A renamed manifest finds it beside itself; one suffixed on landing keeps its base.
        assert each.manifest_refs("nb_y_cell_a_var_out", blob) == [
            ("k", each.instance_id("nb_y_cell_a_var_out", "k"), 3)
        ]
        assert each.manifest_refs("nb_x_cell_a_var_out@import=ab", blob) == [("k", beside, 3)]

    def test_a_manifest_sorts_by_key_and_round_trips(self):
        blob = each.encode_manifest("m", [("b", "id-b", 2), ("a", "id-a", 1)])

        assert each.manifest_refs("m", blob) == [("a", "id-a", 1), ("b", "id-b", 2)]

    @pytest.mark.parametrize("blob", [b'{"other": 1}', b"[1]", b'{"items": 3}'])
    def test_other_bytes_under_the_type_are_not_a_manifest(self, blob):
        with pytest.raises(ValueError, match="not an @each manifest"):
            each.manifest_refs("m", blob)

    def test_a_key_of_any_shape_makes_one_id(self):
        assert each.is_instance_id("x_var_y@variant=" + each.instance_name("a/b c" * 100))
        assert "/" not in each.instance_name("a/b")
        assert each.instance_name("a") != each.instance_name("b")


class TestTheSweepDuringARun:
    async def test_it_waits_while_a_run_assembles(self, tmp_path):
        raw, source = _over_files(tmp_path)
        _touch(raw / "one.txt", "a")
        nb = Notebook(tmp_path, source, READER)
        await nb.good("a")
        (raw / "one.txt").unlink()
        await nb.good("a")
        manager = nb.session.get_artifact_manager()

        assert each.collectable_instances(manager)
        with output_dirs.assembling():
            assert each.collectable_instances(manager) == []

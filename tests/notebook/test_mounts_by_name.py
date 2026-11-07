"""A notebook-level mount reaches only the cells that name it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.graph import mounts_of
from strata.notebook.models import (
    CellLanguage,
    CellState,
    CellStatus,
    ConnectionSpec,
    MountMode,
    MountSpec,
    NotebookState,
)
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)
from tests.notebook.conftest import skip_if_no_r

DATA = MountSpec(name="data", uri="file:///d", mode=MountMode.READ_ONLY)
OTHER = MountSpec(name="other", uri="file:///o", mode=MountMode.READ_ONLY)


def _state(*connections: ConnectionSpec) -> NotebookState:
    return NotebookState(id="n", name="n", connections=list(connections))


def _names(source: str, **fields) -> list[str]:
    cell = CellState(id="c", source=source, mounts=[DATA], **fields)
    return [mount.name for mount in mounts_of(_state(), cell, source)]


class TestWhatACellGets:
    @pytest.mark.parametrize(
        "source",
        [
            "x = data / 'f'\n",
            "def f():\n    return data\n",
            "x = f'{data}'\n",
            "# @each path in data\nout = path\n",
        ],
    )
    def test_a_cell_naming_it_gets_it(self, source):
        assert _names(source) == ["data"]

    @pytest.mark.parametrize("source", ["x = 1\n", "data = 1\n", "# data\nx = 'data'\n"])
    def test_a_cell_not_reading_that_name_does_not(self, source):
        assert _names(source) == []

    @pytest.mark.parametrize(
        "lookup",
        ["globals()['data']", "locals()['data']", "vars()['data']", "eval('data')", "dir()"],
    )
    def test_a_lookup_by_string_keeps_every_mount(self, lookup):
        assert _names(f"x = {lookup}\n") == ["data"]

    def test_exec_keeps_every_mount(self):
        assert _names("exec('x = data')\n") == ["data"]

    def test_a_mount_declared_on_the_cell_stays_unnamed(self):
        cell = CellState(id="c", source="x = 1\n", mounts=[DATA, OTHER], mount_overrides=[OTHER])
        assert mounts_of(_state(), cell, cell.source) == [OTHER]

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("x <- 1\n", []),
            ("x <- data\n", ["data"]),
            ("x <- file.path(data, 'f')\n", ["data"]),
            ("f <- function() data\n", ["data"]),
            ("x <- glue('{data}')\n", ["data"]),
            ("# @each path in data\nout <- path\n", ["data"]),
            # A longer R name is another name.
            ("my.data <- 1\ndata.frame(x = 1)\ndata_2 <- 2\n", []),
            # Reads by a name the source does not spell.
            ("x <- get(paste0('da', 'ta'))\n", ["data"]),
            ("x <- base::mget(name)[[1]]\n", ["data"]),
            ("x <- eval(code)\n", ["data"]),
            ("x <- parse(text = code)\n", ["data"]),
            ("x <- as.list(environment())\n", ["data"]),
            ("x <- .GlobalEnv\n", ["data"]),
            ("source('more.R')\n", ["data"]),
            ("x <- ls()\n", ["data"]),
            # Not a read: words that only contain one.
            ("widget <- 1\nforget <- 2\nparsed <- 3\nls.str <- 4\n", []),
        ],
    )
    def test_an_r_cell_gets_what_its_source_names(self, source, expected):
        assert _names(source, language=CellLanguage.R) == expected

    def test_an_r_cell_keeps_its_own_mounts(self):
        cell = CellState(
            id="c",
            source="x <- 1\n",
            language=CellLanguage.R,
            mounts=[DATA, OTHER],
            mount_overrides=[OTHER],
        )
        assert mounts_of(_state(), cell, cell.source) == [OTHER]

    @pytest.mark.parametrize("language", [CellLanguage.PROMPT, CellLanguage.WIDGET])
    def test_a_cell_binding_no_variable_keeps_only_its_own(self, language):
        cell = CellState(
            id="c", source="data", language=language, mounts=[DATA, OTHER], mount_overrides=[OTHER]
        )
        assert mounts_of(_state(), cell, cell.source) == [OTHER]

    @pytest.mark.parametrize(
        ("connection", "expected"), [("reads", [DATA]), ("plain", []), ("absent", [])]
    )
    def test_a_sql_cell_gets_what_its_connection_names(self, connection, expected):
        state = _state(
            ConnectionSpec(name="reads", driver="duckdb", mounts=["data"]),
            ConnectionSpec(name="plain", driver="duckdb"),
        )
        source = f"-- @sql connection={connection}\nSELECT 1\n"
        cell = CellState(id="c", source=source, language=CellLanguage.SQL, mounts=[DATA, OTHER])
        assert mounts_of(state, cell, source) == expected

    def test_the_source_given_decides_not_the_saved_one(self):
        cell = CellState(id="c", source="x = 1\n", mounts=[DATA])
        assert mounts_of(_state(), cell, "x = data\n") == [DATA]


def _session(tmp_path: Path, mode: MountMode, *sources: str) -> tuple[NotebookSession, Path]:
    folder = tmp_path / "folder"
    folder.mkdir()
    _write(folder / "f.txt", "one", tick=0)
    path = create_notebook(tmp_path / "nb", "Mounts")
    update_notebook_mounts(path, [MountSpec(name="data", uri=folder.as_uri(), mode=mode)])
    after = None
    for cell_id, source in zip("abcd", sources, strict=False):
        add_cell_to_notebook(path, cell_id, after)
        write_cell(path, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(path), path)
    session.refresh_environment_runtime()
    return session, folder


def _write(path: Path, text: str, *, tick: int) -> None:
    path.write_text(text)
    # A fingerprint reads the mtime; a test does not wait for the clock to move.
    stamp = 1_700_000_000 + tick
    os.utime(path, (stamp, stamp))


async def _run(session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


READS = "text = (data / 'f.txt').read_text()\n"


class TestARun:
    async def test_a_changed_mount_leaves_a_cell_that_never_names_it_alone(self, tmp_path):
        session, folder = _session(
            tmp_path, MountMode.READ_ONLY, READS, "unrelated = 1\n", "seen = text + 'x'\n"
        )
        for cell_id in "abc":
            await _run(session, cell_id)

        _write(folder / "f.txt", "two", tick=1)

        assert not (await _run(session, "a")).cache_hit
        assert (await _run(session, "b")).cache_hit
        # Through the upstream that read it, not through the mount itself.
        assert not (await _run(session, "c")).cache_hit

    async def test_a_reopened_notebook_keys_a_cell_as_its_run_did(self, tmp_path):
        session, _ = _session(
            tmp_path,
            MountMode.READ_ONLY,
            READS,
            "unrelated = 1\n",
            "seen = text + str(unrelated)\n",
        )
        await _run(session, "a")
        await _run(session, "b")

        reopened = NotebookSession(parse_notebook(session.path), session.path)
        reopened.refresh_environment_runtime()

        status = reopened.compute_staleness()
        assert status["a"].status == CellStatus.READY
        assert status["b"].status == CellStatus.READY

    async def test_a_read_write_mount_only_stops_caching_the_cells_naming_it(self, tmp_path):
        session, _ = _session(
            tmp_path, MountMode.READ_WRITE, "n = len(list(data.iterdir()))\n", "unrelated = 1\n"
        )
        await _run(session, "a")
        await _run(session, "b")

        assert not (await _run(session, "a")).cache_hit
        assert (await _run(session, "b")).cache_hit

    async def test_code_passed_downstream_still_reads_the_mount(self, tmp_path):
        session, _ = _session(
            tmp_path,
            MountMode.READ_ONLY,
            "fns = {'read': lambda: (data / 'f.txt').read_text()}\n",
            "got = fns['read']()\nprint(got)\n",
        )
        await _run(session, "a")

        assert "one" in (await _run(session, "b")).stdout

    async def test_a_cell_test_can_read_a_mount_its_cell_never_names(self, tmp_path):
        session, _ = _session(tmp_path, MountMode.READ_ONLY, "unrelated = 1\n")

        result = await CellExecutor(session).run_cell_tests(
            "a", "def test_reads(cell):\n    assert (cell.data / 'f.txt').read_text() == 'one'\n"
        )

        assert (result.passed, result.failed, result.errored) == (1, 0, 0), result


@skip_if_no_r
class TestAnRRun:
    def _r_session(self, tmp_path: Path, *sources: str) -> tuple[NotebookSession, Path]:
        folder = tmp_path / "folder"
        folder.mkdir()
        _write(folder / "f.txt", "one", tick=0)
        path = create_notebook(tmp_path / "nb", "Mounts")
        update_notebook_mounts(
            path, [MountSpec(name="data", uri=folder.as_uri(), mode=MountMode.READ_ONLY)]
        )
        after = None
        for cell_id, source in zip("abcd", sources, strict=False):
            add_cell_to_notebook(path, cell_id, after, language="r")
            write_cell(path, cell_id, source)
            after = cell_id
        session = NotebookSession(parse_notebook(path), path)
        session.refresh_environment_runtime()
        return session, folder

    async def test_a_changed_mount_leaves_an_r_cell_that_never_names_it_alone(self, tmp_path):
        session, folder = self._r_session(
            tmp_path,
            'text <- readLines(file.path(data, "f.txt"))\n',
            "unrelated <- 1\n",
            "seen <- list(text, unrelated)\n",
        )
        for cell_id in "abc":
            assert not (await _run(session, cell_id)).cache_hit
        assert (await _run(session, "b")).cache_hit

        _write(folder / "f.txt", "two", tick=1)

        assert not (await _run(session, "a")).cache_hit
        assert (await _run(session, "b")).cache_hit

    async def test_a_function_passed_down_still_reads_the_mount(self, tmp_path):
        """A closure carries the upstream cell's environment, and the path in it."""
        session, _ = self._r_session(
            tmp_path,
            # An environment: a function on its own, or in a list, is not stored.
            'box <- new.env()\nbox$read <- function() readLines(file.path(data, "f.txt"))\n',
            "reader <- box$read\nseen <- reader()\n",
            "out <- paste0(seen, '!')\n",
        )
        for cell_id in "abc":
            result = await _run(session, cell_id)

        assert result.outputs["out"]["preview"] == "one!"

"""A hit shows the console of the run it stands for, and is priced by that run.

Edit a cell, run the edit, undo the edit: the store still holds the first run, under the
source the cell now has. Nothing of the edit's run should survive that hit, by any cell path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strata.notebook.execution_result import CachedConsole, CellExecutionResult, hit_console
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, StalenessReason
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    set_variant_mode,
    write_cell,
)
from tests.notebook.conftest import skip_if_no_r

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX cell execution")

READER = "seen = value\n"
PYTHON = ('value = 1\nprint("one")\n', 'value = 2\nprint("two")\n')
R = ('value <- 1\ncat("one\\n")\n', 'value <- 2\ncat("two\\n")\n')


def _session(tmp_path: Path, *cells: tuple[str, str] | tuple[str, str, str]) -> NotebookSession:
    """A notebook of ``(cell id, source[, language])``."""
    nb = create_notebook(tmp_path / "nb", "Console")
    after = None
    for cell_id, source, *language in cells:
        add_cell_to_notebook(nb, cell_id, after, language=language[0] if language else "python")
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str, *, hit: bool | None = None):
    cell = session.notebook_state.get_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    if hit is not None:
        assert result.cache_hit is hit
    return result


def _edit(session: NotebookSession, cell_id: str, source: str) -> None:
    write_cell(session.path, cell_id, source)
    session.notebook_state.get_cell(cell_id).source = source
    session.re_analyze_cell(cell_id)


async def _edit_run_undo(session: NotebookSession, cell_id: str, edit: str) -> None:
    original = session.notebook_state.get_cell(cell_id).source
    _edit(session, cell_id, edit)
    await _run(session, cell_id, hit=False)
    _edit(session, cell_id, original)


def _value(session: NotebookSession, cell_id: str, variable: str) -> bytes:
    manager = session.get_artifact_manager()
    artifact_id, version = session.parse_artifact_uri(
        session.notebook_state.get_cell(cell_id).artifact_uris[variable]
    )
    return manager.load_artifact_data(artifact_id, version)


def _snapshot(session: NotebookSession, cell_id: str) -> Path:
    return session.path / ".strata" / "console" / f"{cell_id}.json"


class TestAnUndoneEdit:
    async def test_the_cell_reads_stale_while_it_shows_the_edits_run(self, tmp_path):
        """A display the edit dropped is still latest, and says nothing of the variables."""
        session = _session(tmp_path, ("a", "x = 1\nx\n"), ("b", "y = x + 1\n"))
        await _run(session, "a")
        await _run(session, "b")
        await _edit_run_undo(session, "a", "x = 2\nprint(x)\n")

        staleness = session.compute_staleness()

        assert staleness["a"].status == CellStatus.STALE
        assert staleness["a"].reasons == [StalenessReason.SELF]
        assert staleness["b"].status != CellStatus.READY
        cell = session.notebook_state.get_cell("a")
        assert cell.console_stdout == "2\n"
        assert cell.display_outputs == []

    async def test_running_it_brings_the_first_run_back_whole(self, tmp_path):
        session = _session(tmp_path, ("a", "x = 1\nx\n"), ("b", "y = x + 1\n"))
        await _run(session, "a")
        first_value = _value(session, "a", "x")
        await _run(session, "b")
        await _edit_run_undo(session, "a", "x = 2\nprint(x)\n")
        session.compute_staleness()

        result = await _run(session, "a", hit=True)

        cell = session.notebook_state.get_cell("a")
        assert result.stdout == ""
        assert cell.console_stdout == ""
        assert len(cell.display_outputs) == 1
        assert _value(session, "a", "x") == first_value
        assert session.compute_staleness()["a"].status == CellStatus.READY

    @pytest.mark.parametrize(
        ("source", "language"),
        [
            pytest.param(PYTHON, "python", id="python"),
            pytest.param(R, "r", id="r", marks=skip_if_no_r),
        ],
    )
    async def test_a_consumed_cell_replays_the_first_runs_console(self, tmp_path, source, language):
        reader = "seen <- value\n" if language == "r" else READER
        session = _session(tmp_path, ("c", source[0], language), ("d", reader, language))
        await _run(session, "c")
        await _run(session, "d")
        await _edit_run_undo(session, "c", source[1])

        result = await _run(session, "c", hit=True)

        assert result.stdout.strip() == "one"
        assert session.notebook_state.get_cell("c").console_stdout.strip() == "one"
        assert "one" in _snapshot(session, "c").read_text()

    async def test_a_silent_first_run_clears_the_edits_console(self, tmp_path):
        session = _session(tmp_path, ("c", "value = 1\n"), ("d", READER))
        await _run(session, "c")
        await _run(session, "d")
        await _edit_run_undo(session, "c", 'value = 2\nprint("two")\n')
        assert _snapshot(session, "c").exists()

        result = await _run(session, "c", hit=True)

        assert result.stdout == ""
        assert session.notebook_state.get_cell("c").console_stdout == ""
        assert not _snapshot(session, "c").exists()

    @pytest.mark.parametrize(
        ("first", "edit", "language"),
        [
            pytest.param(
                'print("one")\ndisplay(Markdown("shown"))\n',
                'print("two")\n',
                "python",
                id="python",
            ),
            pytest.param(
                'cat("one\\n")\nplot(1:3)\n',
                'cat("two\\n")\n',
                "r",
                id="r",
                marks=skip_if_no_r,
            ),
        ],
    )
    async def test_a_leaf_whose_display_outlived_the_edit(self, tmp_path, first, edit, language):
        """Only the console moved: a match on the display must not settle the hit."""
        session = _session(tmp_path, ("c", first, language))
        run = await _run(session, "c", hit=False)
        assert len(session.notebook_state.get_cell("c").display_outputs) == 1
        await _edit_run_undo(session, "c", edit)
        assert session.compute_staleness()["c"].status == CellStatus.STALE

        result = await _run(session, "c", hit=True)

        cell = session.notebook_state.get_cell("c")
        assert result.stdout.strip() == "one"
        assert cell.console_stdout.strip() == "one"
        assert len(cell.display_outputs) == 1
        assert result.hit_run_ms == int(run.duration_ms)
        assert session.compute_staleness()["c"].status == CellStatus.READY

    async def test_a_leaf_whose_edit_stored_no_record(self, tmp_path):
        """The first run's record is still latest, and is not what the cell shows."""
        session = _session(tmp_path, ("c", 'print("one")\n'))
        await _run(session, "c", hit=False)
        await _edit_run_undo(session, "c", '# @nocache\nprint("two")\n')

        staleness = session.compute_staleness()

        assert staleness["c"].status == CellStatus.STALE
        assert session.notebook_state.get_cell("c").console_stdout == "two\n"
        result = await _run(session, "c", hit=True)
        assert result.stdout == "one\n"
        assert session.compute_staleness()["c"].status == CellStatus.READY

    async def test_a_nocache_leaf_with_a_display_is_idle_after_a_reopen(self, tmp_path):
        """Nothing proves it would show the same, display or not."""
        session = _session(tmp_path, ("c", '# @nocache\ndisplay(Markdown("shown"))\n'))
        await _run(session, "c")

        reopened = NotebookSession(parse_notebook(session.path), session.path)

        assert reopened.compute_staleness()["c"].status == CellStatus.IDLE

    async def test_a_leaf_that_gains_a_reader_is_idle(self, tmp_path):
        """Its display matches, and the variable the reader needs was never stored."""
        session = _session(
            tmp_path, ("c", 'value = 1\ndisplay(Markdown("shown"))\n'), ("d", "x = 2\n")
        )
        await _run(session, "c")
        await _run(session, "d")

        _edit(session, "d", READER)

        assert session.compute_staleness()["c"].status == CellStatus.IDLE

    async def test_a_plain_rerun_replays_too(self, tmp_path):
        session = _session(tmp_path, ("c", PYTHON[0]), ("d", READER))
        await _run(session, "c")
        await _run(session, "d")

        result = await _run(session, "c", hit=True)

        assert result.stdout == "one\n"

    async def test_a_failed_run_of_the_same_key_gives_way_to_its_record(self, tmp_path):
        session = _session(tmp_path, ("c", "value = 1\n"), ("d", READER))
        await _run(session, "c")
        await _run(session, "d")
        cell = session.notebook_state.get_cell("c")
        cell.console_stdout = "from a failed rerun\n"

        await _run(session, "c", hit=True)

        assert cell.console_stdout == ""


class TestOtherCellPaths:
    async def test_run_all(self, tmp_path):
        session = _session(tmp_path, ("c", PYTHON[0]), ("d", READER))

        async def batch():
            specs = [
                {
                    "cell_id": cell.id,
                    "source": cell.source,
                    "consumed_vars": sorted(session.dag.consumed_variables.get(cell.id, set())),
                    "env": {},
                    "mount_manifest": {},
                    "source_hash": "",
                    "env_hash": "",
                }
                for cell in session.notebook_state.cells
            ]
            result = await CellExecutor(session).execute_batch(specs)
            assert result.completed
            return next(cell for cell in result.cell_results if cell.cell_id == "c")

        await batch()
        manager = session.get_artifact_manager()
        recorded = manager.artifact_store.get_latest_version(
            manager.cell_artifact_id("c", "__console__")
        ).params["build_duration_ms"]
        _edit(session, "c", PYTHON[1])
        await batch()
        _edit(session, "c", PYTHON[0])

        hit = await batch()

        assert hit.cache_hit
        assert hit.stdout == "one\n"
        assert hit.replaces_console
        assert hit.hit_run_ms == int(recorded)

    async def test_a_loop(self, tmp_path):
        loop = '# @loop max_iter=1 carry=state\nstate = state + 1\nprint("one")\n'
        session = _session(tmp_path, ("s", "state = 0\n"), ("l", loop), ("r", "seen = state\n"))
        await _run(session, "l")
        await _run(session, "r")
        await _edit_run_undo(session, "l", loop.replace("one", "two"))

        result = await _run(session, "l", hit=True)

        assert result.stdout == "one\n"
        assert session.notebook_state.get_cell("l").console_stdout == "one\n"

    async def test_an_each_cell(self, tmp_path):
        source = '# @each item in items\nout = item\nprint("one")\n'
        session = _session(
            tmp_path,
            ("a", 'items = {"x": 1, "y": 2}\n'),
            ("e", source),
            ("w", "seen = out\n"),
        )
        await _run(session, "a")
        first = await _run(session, "e", hit=False)
        await _edit_run_undo(session, "e", source.replace("one", "two"))

        result = await _run(session, "e", hit=True)

        assert result.stdout == first.stdout
        assert "one" in result.stdout

    async def test_a_leaf_each_cell_replays_from_its_instances(self, tmp_path):
        source = '# @each item in items\nout = item\nprint("one")\n'
        session = _session(tmp_path, ("a", 'items = {"x": 1, "y": 2}\n'), ("e", source))
        await _run(session, "a")
        first = await _run(session, "e", hit=False)
        await _edit_run_undo(session, "e", source.replace("one", "two"))

        result = await _run(session, "e", hit=True)

        assert result.stdout == first.stdout
        assert result.replaces_console
        assert result.hit_run_ms > 0

    async def test_a_per_variant_cell(self, tmp_path):
        source = '# @per_variant\nscore = sum(preds)\nprint("one")\n'
        session = _session(
            tmp_path,
            ("v1", "# @variant model double\npreds = [2.0]\n"),
            ("v2", "# @variant model triple\npreds = [3.0]\n"),
            ("ev", source),
            ("r", "current = dict(score)\n"),
        )
        set_variant_mode(session.path, "model", "sweep")
        session = NotebookSession(parse_notebook(session.path), session.path)
        session.refresh_environment_runtime()
        await _run(session, "r")
        first = session.notebook_state.get_cell("ev").console_stdout
        assert "[variant=double]" in first
        await _edit_run_undo(session, "ev", source.replace("one", "two"))

        result = await _run(session, "ev", hit=True)

        assert result.stdout == first
        assert session.notebook_state.get_cell("ev").console_stdout == first
        assert result.hit_run_ms > 0

    async def test_a_leaf_each_cell_reads_stale_when_its_upstream_moves(self, tmp_path):
        """Its record is per instance, and still says what the run read."""
        items = 'items = {"x": 1}\n'
        session = _session(
            tmp_path, ("a", items), ("e", "# @each item in items\nout = item\nprint(item)\n")
        )
        await _run(session, "e")
        _edit(session, "a", 'items = {"x": 2}\n')
        await _run(session, "a", hit=False)

        staleness = session.compute_staleness()

        assert staleness["e"].status == CellStatus.STALE
        assert staleness["e"].reasons == [StalenessReason.UPSTREAM]

    async def test_a_console_is_not_listed_as_an_output(self, tmp_path):
        source = '# @each item in items\nout = item\nprint("one")\n'
        session = _session(
            tmp_path, ("a", 'items = {"x": 1}\n'), ("e", source), ("w", "seen = out\n")
        )
        await _run(session, "a")
        await _run(session, "e")

        names = [row["name"] for row in session.get_artifact_manager().cell_output_digests("e")]

        assert names
        assert not [name for name in names if name.startswith("__console__")]


class TestAHitWithNoRecord:
    """A team pull, or a store from before every cell kept one."""

    @pytest.fixture
    def no_records(self, monkeypatch):
        monkeypatch.setattr(NotebookSession, "resolve_cached_console", lambda *_a, **_k: None)

    async def test_this_keys_console_stays(self, tmp_path, no_records):
        session = _session(tmp_path, ("c", PYTHON[0]), ("d", READER))
        await _run(session, "c")
        await _run(session, "d")

        result = await _run(session, "c", hit=True)

        assert result.replaces_console is False
        assert session.notebook_state.get_cell("c").console_stdout == "one\n"
        assert _snapshot(session, "c").exists()

    async def test_another_keys_console_goes(self, tmp_path, no_records):
        session = _session(tmp_path, ("c", PYTHON[0]), ("d", READER))
        await _run(session, "c")
        await _run(session, "d")
        await _edit_run_undo(session, "c", PYTHON[1])

        result = await _run(session, "c", hit=True)

        assert result.replaces_console is True
        assert session.notebook_state.get_cell("c").console_stdout == ""
        assert not _snapshot(session, "c").exists()

    def test_the_fields(self):
        audit = [{"code": "undeclared_read", "items": ["/a"], "total": 1, "partial": False}]
        console = CachedConsole(stdout="o", stderr="e", run_ms=7, read_audit=audit)
        assert hit_console(console, shows_this_key=True) == {
            "stdout": "o",
            "stderr": "e",
            "replaces_console": True,
            "hit_run_ms": 7,
            "read_audit": audit,
            "mutation_warnings": [],
        }
        assert hit_console(None, shows_this_key=True) == {"replaces_console": False}
        assert hit_console(None, shows_this_key=False) == {"replaces_console": True}


class TestWhatAHitSaved:
    async def test_the_price_is_the_replayed_runs_not_the_edits(self, tmp_path):
        session = _session(tmp_path, ("c", PYTHON[0]), ("d", READER))
        first = await _run(session, "c")
        await _run(session, "d")
        await _edit_run_undo(session, "c", PYTHON[1])

        result = await _run(session, "c", hit=True)

        assert result.hit_run_ms == int(first.duration_ms)

    def test_a_hit_is_priced_by_its_own_run(self, tmp_path):
        session = _session(tmp_path, ("c", "value = 1\n"))
        session.record_execution("c", 1000.0, cache_hit=False)
        session.record_execution("c", 50000.0, cache_hit=False)
        session.record_execution("c", 2.0, cache_hit=True, hit_run_ms=1000)

        assert session.last_hit_saved_ms("c") == 1000
        assert session.get_profiling_summary()["cache_savings_ms"] == 1000

    def test_without_a_record_the_cells_last_run_prices_it(self, tmp_path):
        session = _session(tmp_path, ("c", "value = 1\n"))
        session.record_execution("c", 50000.0, cache_hit=False)
        session.record_execution("c", 2.0, cache_hit=True)

        assert session.last_hit_saved_ms("c") == 50000
        assert session.get_profiling_summary()["cache_savings_ms"] == 50000

    def test_the_price_survives_a_reopen(self, tmp_path):
        session = _session(tmp_path, ("c", "value = 1\n"))
        session.record_execution("c", 50000.0, cache_hit=False)
        session.record_execution("c", 2.0, cache_hit=True, hit_run_ms=1000)

        reopened = NotebookSession(parse_notebook(session.path), session.path)

        assert reopened.last_hit_saved_ms("c") == 1000


def test_replaces_console_defaults_off():
    """Every path with no record of its own (SQL, a prompt) leaves the console alone."""
    assert CellExecutionResult(cell_id="c", success=True, cache_hit=True).replaces_console is False

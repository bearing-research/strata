"""A failed cell's traceback starts at the cell, on every path that runs one."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import pytest

from strata.notebook.cell_traceback import failure_fields
from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.notebook.ws import _execution_result_payload
from tests.notebook.pool_helpers import warm_again


def _run_as_harness(source: str) -> BaseException:
    """Raise out of *source* through a frame standing in for the harness."""
    code = compile(source, "<cell>", "exec")
    try:
        exec(code, {"json": json})
    except Exception as exc:
        return exc
    raise AssertionError("the cell did not fail")


EXPRESSION_LAST = "x = 1\n1 / 0\n"
STATEMENT_LAST = "x = 1\ny = 1 / 0\n"
SYNTAX_ERROR = "x = 1\nx = = 2\n"
THROUGH_A_LIBRARY = "import json\njson.loads('not json')\n"
LOOP = "# @loop max_iter=2 carry=state\nstate = 1 / 0\n"

CELLS = {
    "expr": EXPRESSION_LAST,
    "stmt": STATEMENT_LAST,
    "syntax": SYNTAX_ERROR,
    "lib": THROUGH_A_LIBRARY,
    "seed": "state = 0\n",
    "loop": LOOP,
}


def _session(tmp_path: Path) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "Tracebacks")
    after = None
    for cell_id, source in CELLS.items():
        add_cell_to_notebook(notebook_dir, cell_id, after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _assert_starts_at_the_cell(text: str, last_line: str) -> None:
    lines = text.strip().splitlines()
    assert lines[0] == "Traceback (most recent call last):"
    assert lines[1].startswith('  File "<cell')
    assert ", line 2, in <module>" in lines[1]
    assert lines[-1] == last_line
    assert text.count(last_line) == 1
    assert "harness" not in text and "pool_worker" not in text


class TestTrimming:
    def test_harness_frames_before_the_cell_go(self):
        fields = failure_fields(_run_as_harness(EXPRESSION_LAST))

        _assert_starts_at_the_cell(fields["traceback"], "ZeroDivisionError: division by zero")
        assert fields["error"] == "ZeroDivisionError: division by zero"
        assert "_run_as_harness" not in fields["traceback"]
        assert "_run_as_harness" in fields["traceback_full"]

    def test_library_frames_after_the_cell_stay(self):
        text = failure_fields(_run_as_harness(THROUGH_A_LIBRARY))["traceback"]

        assert text.index('File "<cell>"') < text.index("decoder.py")
        assert "_run_as_harness" not in text

    def test_every_link_of_a_chain_is_trimmed(self):
        source = (
            "try:\n"
            "    1 / 0\n"
            "except ZeroDivisionError as exc:\n"
            "    raise ValueError('wrapped') from exc\n"
        )
        fields = failure_fields(_run_as_harness(source))

        assert "ZeroDivisionError: division by zero" in fields["traceback"]
        assert "The above exception was the direct cause" in fields["traceback"]
        assert fields["traceback"].rstrip().endswith("ValueError: wrapped")
        assert "_run_as_harness" not in fields["traceback"]

    def test_a_group_is_trimmed_all_the_way_down(self):
        source = (
            "def fail():\n"
            "    raise ValueError('inner')\n"
            "try:\n"
            "    fail()\n"
            "except ValueError as exc:\n"
            "    raise ExceptionGroup('many', [exc])\n"
        )
        text = failure_fields(_run_as_harness(source))["traceback"]

        assert "ValueError: inner" in text
        assert "_run_as_harness" not in text

    def test_a_failure_with_no_cell_frame_keeps_every_frame(self):
        def strata_itself():
            raise RuntimeError("a bug of ours")

        try:
            strata_itself()
        except RuntimeError as exc:
            fields = failure_fields(exc)
            untrimmed = "".join(traceback.format_exception(exc))

        assert "strata_itself" in fields["traceback"]
        assert fields["traceback"] == untrimmed

    def test_a_syntax_error_is_just_the_line_and_the_caret(self):
        try:
            compile(SYNTAX_ERROR, "<cell>", "exec")
        except SyntaxError as exc:
            fields = failure_fields(exc)

        assert "Traceback" not in fields["traceback"]
        assert fields["traceback"].startswith('  File "<cell>", line 2')
        assert "Traceback" in fields["traceback_full"]

    def test_a_hint_joins_the_error_line(self):
        fields = failure_fields(ValueError("no"), "try this")

        assert fields["error"] == "ValueError: no\ntry this"


async def _run(executor: CellExecutor, cell_id: str, source: str) -> CellExecutionResult:
    if executor.pool is not None:
        await warm_again(executor.pool)
    result = await executor.execute_cell(cell_id, source)
    assert result.execution_method == ("cold" if executor.pool is None else "warm")
    return result


async def _assert_every_failure_reads_the_same(executor: CellExecutor) -> None:
    expected = "ZeroDivisionError: division by zero"
    for cell_id in ("expr", "stmt"):
        result = await _run(executor, cell_id, CELLS[cell_id])
        assert not result.success
        assert result.error == expected
        _assert_starts_at_the_cell(result.traceback, expected)
        assert "harness" in result.traceback_full or "pool_worker" in result.traceback_full
        assert CELLS[cell_id].splitlines()[1] in result.traceback

    result = await _run(executor, "lib", THROUGH_A_LIBRARY)
    assert "decoder.py" in result.traceback
    assert "harness" not in result.traceback and "pool_worker" not in result.traceback

    result = await _run(executor, "syntax", SYNTAX_ERROR)
    assert result.error == "SyntaxError: invalid syntax (<cell syntax>, line 2)"
    assert result.traceback.startswith('  File "<cell syntax>", line 2')
    assert result.traceback.count("SyntaxError") == 1
    assert "During handling" not in result.traceback_full


class TestEveryPathAgrees:
    async def test_a_cold_run(self, tmp_path):
        await _assert_every_failure_reads_the_same(CellExecutor(_session(tmp_path)))

    @pytest.mark.integration
    @pytest.mark.warm_pool
    async def test_a_warm_run(self, tmp_path, warm_pool):
        session = _session(tmp_path)
        await _assert_every_failure_reads_the_same(CellExecutor(session, await warm_pool(session)))

    async def test_a_run_all_batch(self, tmp_path):
        session = _session(tmp_path)
        spec = {
            "cell_id": "stmt",
            "source": STATEMENT_LAST,
            "consumed_vars": [],
            "env": {},
            "mount_manifest": {},
            "source_hash": "",
            "env_hash": "",
        }

        result = await CellExecutor(session).execute_batch([spec])

        (failed,) = result.cell_results
        assert failed.error == "ZeroDivisionError: division by zero"
        _assert_starts_at_the_cell(failed.traceback, failed.error)
        assert "harness" in failed.traceback_full
        assert "y = 1 / 0" in failed.traceback


def _spec(cell_id: str, source: str, consumed: list[str] | None = None) -> dict:
    return {
        "cell_id": cell_id,
        "source": source,
        "consumed_vars": consumed or [],
        "env": {},
        "mount_manifest": {},
        "source_hash": "",
        "env_hash": "",
    }


class TestAFrameQuotesItsOwnCell:
    """Line numbers are per cell, so a frame must never quote the running cell's line."""

    # Line 4 of each is what a shared name would quote for the other's frame.
    PICKLED = (
        "def make(divisor):\n\n    def divide(x):\n        return x / divisor\n"
        "    return divide\n\ndivide = make(0)\n"
    )
    CALLER = "a = 1\nb = 2\nc = 3\nd = 4\nresult = divide(1)\n"

    def _two_cells(self, tmp_path: Path, first: str, second: str) -> NotebookSession:
        notebook_dir = create_notebook(tmp_path, "Frames")
        add_cell_to_notebook(notebook_dir, "up", None)
        write_cell(notebook_dir, "up", first)
        add_cell_to_notebook(notebook_dir, "down", "up")
        write_cell(notebook_dir, "down", second)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        return session

    async def test_a_function_pickled_from_another_cell_quotes_nothing(self, tmp_path):
        """Its source isn't in this process, and the running cell's line 4 is another line."""
        executor = CellExecutor(self._two_cells(tmp_path, self.PICKLED, self.CALLER))

        assert (await executor.execute_cell("up", self.PICKLED)).success
        result = await executor.execute_cell("down", self.CALLER)

        assert not result.success
        assert 'File "<cell down>", line 5' in result.traceback
        assert "result = divide(1)" in result.traceback
        assert 'File "<cell up>", line 4, in divide' in result.traceback
        assert "d = 4" not in result.traceback

    async def test_a_function_exported_from_another_cell_quotes_its_line(self, tmp_path):
        first = "# a note\n\n\ndef f(x):\n    return 1 / x\n"
        second = "a = 1\nresult = f(0)\n"
        executor = CellExecutor(self._two_cells(tmp_path, first, second))

        assert (await executor.execute_cell("up", first)).success
        result = await executor.execute_cell("down", second)

        assert not result.success
        assert "result = f(0)" in result.traceback
        assert "return 1 / x" in result.traceback

    async def test_cells_sharing_a_batch_quote_their_own_lines(self, tmp_path):
        session = self._two_cells(tmp_path, self.PICKLED, self.CALLER)

        result = await CellExecutor(session).execute_batch(
            [_spec("up", self.PICKLED, ["divide"]), _spec("down", self.CALLER)]
        )

        failed = next(r for r in result.cell_results if r.cell_id == "down")
        assert "result = divide(1)" in failed.traceback
        assert "return x / divisor" in failed.traceback
        assert "d = 4" not in failed.traceback


async def test_a_loop_failure_says_which_iteration(tmp_path):
    executor = CellExecutor(_session(tmp_path))
    await executor.execute_cell("seed", CELLS["seed"])

    result = await executor.execute_cell("loop", LOOP)

    assert result.error == "Loop cell iter 0 failed: ZeroDivisionError: division by zero"
    assert result.traceback.startswith("Loop cell iter 0 failed:\nTraceback")
    assert "harness" in result.traceback_full


def test_the_failure_frame_carries_both_forms():
    result = CellExecutionResult(
        cell_id="c",
        success=False,
        error="ValueError: no",
        traceback="trimmed",
        traceback_full="full",
    )

    payload = _execution_result_payload("c", result)

    assert payload["error"] == "ValueError: no"
    assert payload["traceback"] == "trimmed"
    assert payload["traceback_full"] == "full"

"""Tests for the ``strata run`` headless notebook runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from strata.notebook.cli import _sync_environment, run_main
from strata.notebook.executor import CellExecutionResult
from tests.notebook.conftest import skip_if_no_r


def _build_notebook(
    tmp_path: Path,
    *,
    cells: list[tuple[str, str, str | None]],
    language: str = "python",
) -> Path:
    """Create a notebook with the given cells.

    ``cells`` is a list of ``(cell_id, source, after_id)`` tuples in the
    order they should be added. Pass ``None`` for ``after_id`` to add
    the first cell. ``language`` applies to every cell (Python by
    default; pass ``"r"`` for an R notebook). The notebook is created
    with ``initialize_environment=False`` so ``.venv/`` only exists when
    a test explicitly asks for it (via ``_mk_fake_venv``).
    """
    import shutil

    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(
        tmp_path,
        "CliTest",
        initialize_environment=False,
    )
    # Defensive: if a prior test or the creator left .venv behind, wipe it
    # so tests that rely on its absence are deterministic.
    stale_venv = notebook_dir / ".venv"
    if stale_venv.exists():
        shutil.rmtree(stale_venv)
    for cell_id, source, after_id in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_id, language=language)
        write_cell(notebook_dir, cell_id, source)
    return notebook_dir


def _mk_fake_venv(notebook_dir: Path) -> None:
    """Create a placeholder ``.venv`` directory so ``--no-sync`` passes."""
    (notebook_dir / ".venv").mkdir(exist_ok=True)


def _make_result(
    cell_id: str,
    *,
    success: bool = True,
    cache_hit: bool = False,
    duration_ms: float = 10.0,
    error: str | None = None,
) -> CellExecutionResult:
    return CellExecutionResult(
        cell_id=cell_id,
        success=success,
        duration_ms=duration_ms,
        cache_hit=cache_hit,
        error=error,
    )


class TestArgumentHandling:
    def test_missing_path_exits_2(self, tmp_path):
        bogus = tmp_path / "does-not-exist"
        assert run_main([str(bogus)]) == 2

    def test_not_a_notebook_exits_2(self, tmp_path):
        plain_dir = tmp_path / "plain"
        plain_dir.mkdir()
        assert run_main([str(plain_dir)]) == 2

    def test_no_sync_without_venv_exits_2(self, tmp_path):
        notebook_dir = _build_notebook(tmp_path, cells=[("c1", "x = 1", None)])
        # Intentionally do NOT create .venv.
        assert run_main([str(notebook_dir), "--no-sync"]) == 2


class TestExecutionFlow:
    """Tests that mock the executor so we don't pay for a real uv sync."""

    @pytest.fixture
    def notebook_with_chain(self, tmp_path):
        # c1 defines x, c2 uses x and defines y
        notebook_dir = _build_notebook(
            tmp_path,
            cells=[
                ("c1", "x = 1", None),
                ("c2", "y = x + 1", "c1"),
            ],
        )
        _mk_fake_venv(notebook_dir)
        return notebook_dir

    def test_all_cells_succeed_returns_0(self, notebook_with_chain, capsys):
        async def fake_execute_cell(self, cell_id, source, timeout_seconds=30):
            return _make_result(cell_id, success=True, duration_ms=50)

        with patch(
            "strata.notebook.executor.CellExecutor.execute_cell",
            new=fake_execute_cell,
        ):
            exit_code = run_main([str(notebook_with_chain), "--no-sync"])

        assert exit_code == 0
        captured = capsys.readouterr()
        # Both cell IDs (or their short forms) should appear in output
        assert "c1" in captured.out
        assert "c2" in captured.out
        assert "2 ran" in captured.out or "ran" in captured.out

    def test_json_output_shape(self, notebook_with_chain, capsys):
        async def fake_execute_cell(self, cell_id, source, timeout_seconds=30):
            return _make_result(cell_id, success=True, cache_hit=(cell_id == "c2"))

        with patch(
            "strata.notebook.executor.CellExecutor.execute_cell",
            new=fake_execute_cell,
        ):
            exit_code = run_main([str(notebook_with_chain), "--no-sync", "--format", "json"])

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["notebook"] == str(notebook_with_chain)
        assert {c["id"] for c in payload["cells"]} == {"c1", "c2"}
        # label must NOT leak into json output
        assert all("label" not in c for c in payload["cells"])
        c2 = next(c for c in payload["cells"] if c["id"] == "c2")
        assert c2["cache_hit"] is True
        assert c2["status"] == "ok"

    def test_timeout_flag_threads_to_executor(self, notebook_with_chain):
        seen: dict[str, float] = {}

        async def fake_execute_cell(self, cell_id, source, timeout_seconds=300.0):
            seen[cell_id] = timeout_seconds
            return _make_result(cell_id, success=True)

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute_cell):
            run_main([str(notebook_with_chain), "--no-sync", "--timeout", "999"])

        assert seen == {"c1": 999.0, "c2": 999.0}  # --timeout reaches every cell

    def test_mutation_warnings_surface_in_run(self, notebook_with_chain, capsys):
        from strata.notebook.executor import CellExecutionResult

        async def fake_execute_cell(self, cell_id, source, timeout_seconds=300.0):
            warns = (
                [
                    {
                        "var_name": "df",
                        "message": "'df' was mutated in place (no reassignment)",
                        "suggestion": "(e.g. x = x.copy())",
                    }
                ]
                if cell_id == "c2"
                else []
            )
            return CellExecutionResult(
                cell_id=cell_id, success=True, duration_ms=5, mutation_warnings=warns
            )

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute_cell):
            rc = run_main([str(notebook_with_chain), "--no-sync", "--format", "json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        c2 = next(c for c in payload["cells"] if c["id"] == "c2")
        assert c2["mutation_warnings"][0]["var_name"] == "df"
        c1 = next(c for c in payload["cells"] if c["id"] == "c1")
        assert "mutation_warnings" not in c1  # only emitted when present

    def test_mutation_warnings_print_in_human_output(self, notebook_with_chain, capsys):
        from strata.notebook.executor import CellExecutionResult

        async def fake_execute_cell(self, cell_id, source, timeout_seconds=300.0):
            warns = (
                [{"var_name": "df", "message": "'df' was mutated in place", "suggestion": None}]
                if cell_id == "c2"
                else []
            )
            return CellExecutionResult(
                cell_id=cell_id, success=True, duration_ms=5, mutation_warnings=warns
            )

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute_cell):
            run_main([str(notebook_with_chain), "--no-sync"])
        assert "mutated in place" in capsys.readouterr().out

    def test_cell_failure_returns_1_and_skips_downstream(self, notebook_with_chain, capsys):
        async def fake_execute_cell(self, cell_id, source, timeout_seconds=30):
            if cell_id == "c1":
                return _make_result(
                    cell_id,
                    success=False,
                    error="ValueError: boom",
                    duration_ms=15,
                )
            # c2 should never be invoked because its upstream failed.
            pytest.fail(f"execute_cell should not run for {cell_id}")

        with patch(
            "strata.notebook.executor.CellExecutor.execute_cell",
            new=fake_execute_cell,
        ):
            exit_code = run_main([str(notebook_with_chain), "--no-sync", "--format", "json"])

        assert exit_code == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is False
        c1 = next(c for c in payload["cells"] if c["id"] == "c1")
        c2 = next(c for c in payload["cells"] if c["id"] == "c2")
        assert c1["status"] == "error"
        assert c1["error"] == "ValueError: boom"
        assert c2["status"] == "skipped"
        assert c2["reason"] == "upstream failed"

    def test_force_flag_routes_to_execute_cell_force(self, notebook_with_chain, capsys):
        force_calls: list[str] = []
        honor_calls: list[str] = []

        async def fake_force(self, cell_id, source, timeout_seconds=30):
            force_calls.append(cell_id)
            return _make_result(cell_id, success=True)

        async def fake_honor(self, cell_id, source, timeout_seconds=30):
            honor_calls.append(cell_id)
            return _make_result(cell_id, success=True)

        with (
            patch(
                "strata.notebook.executor.CellExecutor.execute_cell_force",
                new=fake_force,
            ),
            patch(
                "strata.notebook.executor.CellExecutor.execute_cell",
                new=fake_honor,
            ),
        ):
            exit_code = run_main([str(notebook_with_chain), "--no-sync", "--force"])

        assert exit_code == 0
        assert set(force_calls) == {"c1", "c2"}
        assert honor_calls == []

    def test_default_routes_to_cache_honoring_execute(self, notebook_with_chain, capsys):
        honor_calls: list[str] = []

        async def fake_honor(self, cell_id, source, timeout_seconds=30):
            honor_calls.append(cell_id)
            return _make_result(cell_id, success=True)

        with patch(
            "strata.notebook.executor.CellExecutor.execute_cell",
            new=fake_honor,
        ):
            exit_code = run_main([str(notebook_with_chain), "--no-sync"])

        assert exit_code == 0
        assert set(honor_calls) == {"c1", "c2"}


class TestRCellsHeadless:
    """`strata run` executes R cells instead of skipping them (#98).

    Real Rscript harness — no mock — so this is the end-to-end headless
    R path that was previously a no-op. Gated on Rscript being present.
    """

    @skip_if_no_r
    def test_r_cell_runs_not_skipped(self, tmp_path, capsys):
        notebook_dir = _build_notebook(
            tmp_path,
            cells=[("c1", "answer <- 6L * 7L\n", None)],
            language="r",
        )
        _mk_fake_venv(notebook_dir)

        exit_code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 0, payload
        c1 = next(c for c in payload["cells"] if c["id"] == "c1")
        assert c1["status"] == "ok", c1
        # The old behaviour skipped R as an unsupported language.
        assert "unsupported language" not in (c1.get("reason") or "")

    @skip_if_no_r
    def test_r_cell_failure_returns_1_and_skips_downstream(self, tmp_path, capsys):
        """A genuine R error fails the cell (not a silent skip) and the
        downstream R cell is skipped as upstream-failed."""
        # c1 statically defines `answer` (so the DAG links c2 -> c1) but
        # errors at runtime before the binding is made.
        notebook_dir = _build_notebook(
            tmp_path,
            cells=[
                ("c1", "answer <- stop('boom from R')\n", None),
                ("c2", "y <- answer + 1\n", "c1"),
            ],
            language="r",
        )
        _mk_fake_venv(notebook_dir)

        exit_code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        payload = json.loads(capsys.readouterr().out)
        assert exit_code == 1, payload
        by_id = {c["id"]: c for c in payload["cells"]}
        assert by_id["c1"]["status"] == "error"
        assert by_id["c2"]["status"] == "skipped"
        assert by_id["c2"]["reason"] == "upstream failed"


class _FakeJob:
    def __init__(self) -> None:
        self.status = "running"
        self.error: str | None = None


class _FakeSyncSession:
    """Reproduces the session's env-job lifecycle for ``_sync_environment``.

    The real ``_run_environment_job`` mutates the returned job in place to
    its terminal status and then resets ``environment_job`` to None. This
    fake does the same — the None reset is exactly the condition that used
    to trip the false "env sync finished without a status snapshot" error
    (#99) when ``_sync_environment`` read the session attribute instead of
    the returned job.
    """

    def __init__(self, *, final_status: str, error: str | None = None) -> None:
        self._job = _FakeJob()
        self._final_status = final_status
        self._error = error
        self.environment_job = None

    async def submit_environment_job(self, *, action: str):
        assert action == "sync"
        self.environment_job = self._job  # the "currently running" slot
        return self._job

    async def wait_for_environment_job(self) -> None:
        self._job.status = self._final_status
        self._job.error = self._error
        self.environment_job = None  # cleared on completion — the #99 trigger


class TestSyncEnvironment:
    def test_completed_sync_reports_success(self):
        ok, err = asyncio.run(_sync_environment(_FakeSyncSession(final_status="completed")))
        assert ok is True
        assert err is None

    def test_failed_sync_surfaces_error(self):
        session = _FakeSyncSession(final_status="failed", error="uv lock conflict")
        ok, err = asyncio.run(_sync_environment(session))
        assert ok is False
        assert "uv lock conflict" in (err or "")


# ---------------------------------------------------------------------------
# strata validate (issue #114 — agent feedback loop)
# ---------------------------------------------------------------------------


def _validate(path, fmt="human"):
    import argparse

    from strata.notebook.cli import validate_main

    return validate_main(argparse.Namespace(path=str(path), format=fmt))


class TestValidate:
    def test_valid_notebook_exits_zero(self, tmp_path, capsys):
        notebook_dir = _build_notebook(
            tmp_path, cells=[("c1", "x = 1", None), ("c2", "y = x + 1", "c1")]
        )
        assert _validate(notebook_dir) == 0
        assert "valid" in capsys.readouterr().out

    def test_json_payload_shape(self, tmp_path, capsys):
        notebook_dir = _build_notebook(
            tmp_path, cells=[("c1", "x = 1", None), ("c2", "y = x + 1", "c1")]
        )
        assert _validate(notebook_dir, fmt="json") == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is True
        assert payload["errors"] == []
        assert payload["summary"] == {"cells": 2, "errors": 0, "warnings": 0}
        by_id = {c["id"]: c for c in payload["cells"]}
        assert by_id["c1"]["defines"] == ["x"]
        assert "x" in by_id["c2"]["references"]
        assert by_id["c2"]["diagnostics"] == []

    def test_malformed_toml_reports_parse_failed(self, tmp_path, capsys):
        notebook_dir = _build_notebook(tmp_path, cells=[("c1", "x = 1", None)])
        toml_path = notebook_dir / "notebook.toml"
        toml_path.write_text(toml_path.read_text() + "\nname = 'duplicate'\n")

        assert _validate(notebook_dir, fmt="json") == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is False
        assert payload["errors"][0]["code"] == "parse_failed"
        # The message must carry the underlying TOML diagnostic so an
        # agent can act on it without opening the UI.
        assert "line" in payload["errors"][0]["message"]

    def test_error_diagnostic_fails_validation(self, tmp_path, capsys):
        """An error-severity annotation (loop without carry) → exit 1."""
        notebook_dir = _build_notebook(
            tmp_path, cells=[("c1", "# @loop max_iter=5\nstate = 1", None)]
        )
        assert _validate(notebook_dir, fmt="json") == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is False
        codes = [d["code"] for c in payload["cells"] for d in c["diagnostics"]]
        assert "loop_missing_carry" in codes

    def test_warning_only_still_valid(self, tmp_path, capsys):
        """Warnings (unknown worker) surface but don't fail validation."""
        notebook_dir = _build_notebook(
            tmp_path, cells=[("c1", "# @worker nonexistent-worker\nx = 1", None)]
        )
        assert _validate(notebook_dir, fmt="json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is True
        assert payload["summary"]["warnings"] >= 1
        codes = [d["code"] for c in payload["cells"] for d in c["diagnostics"]]
        assert "worker_unknown" in codes

    def test_non_notebook_dir_is_usage_error(self, tmp_path):
        assert _validate(tmp_path) == 2

    def test_missing_dir_is_usage_error(self, tmp_path):
        assert _validate(tmp_path / "nope") == 2


# ---------------------------------------------------------------------------
# strata new
# ---------------------------------------------------------------------------


def _new(name, parent, fmt="human", no_env=True):
    import argparse

    from strata.notebook.cli import new_main

    return new_main(
        argparse.Namespace(
            name=name,
            parent=str(parent),
            python_version=None,
            no_env=no_env,
            format=fmt,
        )
    )


class TestNew:
    def test_scaffolds_notebook_dir(self, tmp_path, capsys):
        assert _new("My Analysis", tmp_path) == 0
        out = capsys.readouterr().out
        notebook_dir = tmp_path / "my_analysis"
        assert str(notebook_dir) in out
        assert (notebook_dir / "notebook.toml").is_file()
        assert (notebook_dir / "pyproject.toml").is_file()
        assert (notebook_dir / "cells").is_dir()
        # The scaffold must immediately pass its own validator.
        assert _validate(notebook_dir) == 0

    def test_json_payload(self, tmp_path, capsys):
        assert _new("Agent NB", tmp_path, fmt="json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["notebook_dir"].endswith("agent_nb")
        assert payload["environment_initialized"] is False

    def test_idempotent_preserves_notebook_id(self, tmp_path, capsys):
        import tomllib

        assert _new("Stable", tmp_path) == 0
        toml_path = tmp_path / "stable" / "notebook.toml"
        with open(toml_path, "rb") as f:
            first_id = tomllib.load(f)["notebook_id"]

        assert _new("Stable", tmp_path) == 0
        with open(toml_path, "rb") as f:
            second_id = tomllib.load(f)["notebook_id"]
        assert first_id == second_id

    def test_invalid_name_is_usage_error(self, tmp_path, capsys):
        assert _new("../escape", tmp_path) == 2


# ---------------------------------------------------------------------------
# Round-trip contract (issue #114): a notebook hand-written from the docs
# alone — no writer helpers, no server — parses, validates, and runs.
# ---------------------------------------------------------------------------


class TestHandWrittenNotebookContract:
    """Pins the external-authoring contract: notebook.toml + cells/*.py
    written byte-by-byte the way docs/reference/notebook-toml.md and
    AGENTS.md describe must be a fully working notebook. If a writer
    or parser change breaks this, agents building notebooks from the
    docs break with it."""

    @staticmethod
    def _hand_write(tmp_path: Path) -> Path:
        notebook_dir = tmp_path / "handwritten"
        (notebook_dir / "cells").mkdir(parents=True)
        (notebook_dir / "notebook.toml").write_text(
            "\n".join(
                [
                    'notebook_id = "agent-handwritten-001"',
                    'name = "Hand-written by an agent"',
                    "cells = [",
                    '  { id = "load", file = "load.py", language = "python", order = 0 },',
                    '  { id = "doc", file = "doc.md", language = "markdown", order = 1 },',
                    '  { id = "stats", file = "stats.py", language = "python", order = 2 },',
                    "]",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (notebook_dir / "cells" / "load.py").write_text(
            "# @name load\nnumbers = [1, 2, 3, 4]\n", encoding="utf-8"
        )
        (notebook_dir / "cells" / "doc.md").write_text(
            "# Analysis\n\nplain prose cell\n", encoding="utf-8"
        )
        (notebook_dir / "cells" / "stats.py").write_text(
            "total = sum(numbers)\nmean = total / len(numbers)\n", encoding="utf-8"
        )
        return notebook_dir

    def test_hand_written_notebook_validates(self, tmp_path, capsys):
        notebook_dir = self._hand_write(tmp_path)
        assert _validate(notebook_dir, fmt="json") == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["valid"] is True
        by_id = {c["id"]: c for c in payload["cells"]}
        # The DAG must connect the hand-written cells.
        assert "numbers" in by_id["stats"]["references"]

    def test_hand_written_notebook_runs(self, tmp_path, capsys):
        """`strata run` executes the hand-written notebook end to end
        (mocked executor — the real-venv path is covered by the examples
        CI job)."""
        notebook_dir = self._hand_write(tmp_path)
        _mk_fake_venv(notebook_dir)

        async def fake_execute(self, cell_id, source, **kwargs):
            return _make_result(cell_id)

        with patch(
            "strata.notebook.executor.CellExecutor.execute_cell",
            new=fake_execute,
        ):
            code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        statuses = {c["id"]: c["status"] for c in payload["cells"]}
        assert statuses == {"load": "ok", "doc": "ok", "stats": "ok"}


class TestRunJsonConsoleOutput:
    """`run --format json` carries per-cell stdout/stderr so external
    authors verify computed values from the payload instead of reaching
    into .strata/ (#114 litmus finding)."""

    def test_stdout_lands_in_json_payload(self, tmp_path, capsys):
        notebook_dir = _build_notebook(tmp_path, cells=[("c1", "print('total=42')", None)])
        _mk_fake_venv(notebook_dir)

        async def fake_execute(self, cell_id, source, **kwargs):
            result = _make_result(cell_id)
            result.stdout = "total=42\n"
            return result

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute):
            code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["cells"][0]["stdout"] == "total=42\n"

    def test_long_stdout_is_truncated(self, tmp_path, capsys):
        notebook_dir = _build_notebook(tmp_path, cells=[("c1", "print('x')", None)])
        _mk_fake_venv(notebook_dir)

        async def fake_execute(self, cell_id, source, **kwargs):
            result = _make_result(cell_id)
            result.stdout = "x" * 20_000
            return result

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute):
            code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        stdout = payload["cells"][0]["stdout"]
        assert len(stdout) < 20_000
        assert "truncated" in stdout

    def test_empty_console_keys_absent(self, tmp_path, capsys):
        notebook_dir = _build_notebook(tmp_path, cells=[("c1", "x = 1", None)])
        _mk_fake_venv(notebook_dir)

        async def fake_execute(self, cell_id, source, **kwargs):
            return _make_result(cell_id)

        with patch("strata.notebook.executor.CellExecutor.execute_cell", new=fake_execute):
            code = run_main([str(notebook_dir), "--no-sync", "--format", "json"])

        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        assert "stdout" not in payload["cells"][0]
        assert "stderr" not in payload["cells"][0]


def test_cell_timeout_message_names_the_remedy():
    from strata.notebook.executor import cell_timeout_message

    msg = cell_timeout_message(300.0)
    assert "300.0s" in msg
    # names all three levers so the limit is discoverable from the error alone
    assert "@timeout" in msg
    assert "notebook.toml" in msg
    assert "--timeout" in msg

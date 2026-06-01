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

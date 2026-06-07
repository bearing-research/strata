"""Tests for the warm R process pool (#81).

The pool machinery is the shared ``WarmProcessPool`` with an R worker
command; these tests cover the R-specific pieces — worker readiness,
end-to-end manifest execution over the frame protocol, and the cold
fallback — plus session gating (R pool only for notebooks with R cells).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from strata.notebook.pool import PooledCellExecutor, WarmProcessPool
from strata.notebook.writer import create_notebook
from tests.notebook.conftest import skip_if_no_r

pytestmark = [pytest.mark.integration, pytest.mark.warm_pool]

_POOL_WORKER = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "strata"
    / "notebook"
    / "languages"
    / "r"
    / "pool_worker.R"
)


def _r_pool(notebook_dir: Path, pool_size: int = 1) -> WarmProcessPool:
    rscript = shutil.which("Rscript")
    assert rscript is not None
    return WarmProcessPool(
        notebook_dir=notebook_dir,
        pool_size=pool_size,
        worker_command=[rscript, str(_POOL_WORKER), str(notebook_dir)],
        ready_timeout_seconds=60.0,
    )


@pytest.fixture
def notebook_dir(tmp_path):
    return create_notebook(tmp_path, "R Pool Test")


@skip_if_no_r
class TestRWarmPool:
    @pytest.mark.asyncio
    async def test_pool_starts_and_workers_ready(self, notebook_dir):
        pool = _r_pool(notebook_dir, pool_size=2)
        await pool.start()
        assert pool._available.qsize() == 2
        await pool.drain()

    @pytest.mark.asyncio
    async def test_executes_manifest_end_to_end(self, notebook_dir, tmp_path):
        """A warm R worker runs a manifest and relays the result line."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source": "x <- 40 + 2\ncat('computed\\n')",
                    "inputs": {},
                    "output_dir": str(output_dir),
                    "mounts": {},
                    "env": {},
                    "mutation_defines": [],
                    "consumed_vars": ["x"],
                }
            )
        )

        pool = _r_pool(notebook_dir)
        await pool.start()
        result = await PooledCellExecutor.execute_with_pool(
            pool, manifest_path, notebook_dir, timeout_seconds=60
        )
        await pool.drain()

        assert result is not None
        assert result["success"] is True
        assert "x" in result["variables"]
        assert "computed" in result["stdout"]

    @pytest.mark.asyncio
    async def test_cell_error_relayed_as_failure_result(self, notebook_dir, tmp_path):
        """Errors come back as success=false over the protocol, not a dead pipe."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "source": "stop('boom')",
                    "inputs": {},
                    "output_dir": str(output_dir),
                    "mounts": {},
                    "env": {},
                    "mutation_defines": [],
                    "consumed_vars": [],
                }
            )
        )

        pool = _r_pool(notebook_dir)
        await pool.start()
        result = await PooledCellExecutor.execute_with_pool(
            pool, manifest_path, notebook_dir, timeout_seconds=60
        )
        await pool.drain()

        assert result is not None
        assert result["success"] is False
        assert "boom" in (result.get("error") or "")

    @pytest.mark.asyncio
    async def test_invalidate_respawns(self, notebook_dir):
        pool = _r_pool(notebook_dir)
        await pool.start()
        assert pool._available.qsize() == 1
        await pool.invalidate()
        assert pool._available.qsize() == 1
        await pool.drain()

    @pytest.mark.asyncio
    async def test_acquire_returns_none_when_not_started(self, notebook_dir):
        pool = _r_pool(notebook_dir)
        assert await pool.acquire() is None


class TestRPoolSessionGating:
    """start_r_pool_background only fires for notebooks with R cells."""

    def _session(self, tmp_path, source: str, language):
        from strata.notebook.models import CellState, NotebookState
        from strata.notebook.session import NotebookSession

        notebook_path = create_notebook(tmp_path, "Gating Test")
        session = NotebookSession.__new__(NotebookSession)
        session.path = notebook_path
        session.r_warm_pool = None
        session.notebook_state = NotebookState(
            id="gate-test",
            name="Gating Test",
            cells=[CellState(id="c1", source=source, language=language)],
        )
        session.environment_sync_state = "ready"
        session.has_active_environment_mutation = lambda: False
        return session

    def test_python_only_notebook_skips_r_pool(self, tmp_path):
        from strata.notebook.models import CellLanguage

        session = self._session(tmp_path, "x = 1", CellLanguage.PYTHON)
        session.start_r_pool_background()
        assert session.r_warm_pool is None

    @skip_if_no_r
    def test_r_notebook_creates_pool(self, tmp_path):
        from strata.notebook.models import CellLanguage

        session = self._session(tmp_path, "x <- 1", CellLanguage.R)
        session.start_r_pool_background()
        assert session.r_warm_pool is not None
        assert session.r_warm_pool.worker_command is not None
        assert session.r_warm_pool.worker_command[1].endswith("pool_worker.R")
        # No event loop here: pool object exists but stays cold
        session.r_warm_pool.shutdown_nowait()

    def test_no_rscript_skips_pool(self, tmp_path, monkeypatch):
        import shutil as shutil_mod

        from strata.notebook.models import CellLanguage

        monkeypatch.setattr(shutil_mod, "which", lambda _: None)
        session = self._session(tmp_path, "x <- 1", CellLanguage.R)
        session.start_r_pool_background()
        assert session.r_warm_pool is None


@skip_if_no_r
class TestRPoolExecutorDispatch:
    """The executor uses a warm R worker when the session holds a pool."""

    @pytest.mark.asyncio
    async def test_r_cell_executes_warm(self, tmp_path):
        from strata.notebook.executor import CellExecutor
        from tests.notebook.test_language_r_executor import _make_r_notebook

        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 1 + 2")])
        session.r_warm_pool = _r_pool(session.path)
        await session.r_warm_pool.start()

        executor = CellExecutor(session)
        result = await executor.execute_cell("c1", "x <- 1 + 2")
        await session.r_warm_pool.drain()

        assert result.success is True, result.error
        assert result.execution_method == "warm"
        assert result.outputs["x"]["content_type"] == "json/object"

    @pytest.mark.asyncio
    async def test_cold_fallback_when_pool_empty(self, tmp_path):
        from strata.notebook.executor import CellExecutor
        from tests.notebook.test_language_r_executor import _make_r_notebook

        _, session = _make_r_notebook(tmp_path, cells=[("c1", None, "x <- 5")])
        session.r_warm_pool = _r_pool(session.path)
        # Never started: acquire() returns None and the cold harness runs.

        executor = CellExecutor(session)
        result = await executor.execute_cell("c1", "x <- 5")

        assert result.success is True, result.error
        assert result.execution_method == "cold"

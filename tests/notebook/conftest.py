"""Notebook-local pytest fixtures for fast default test runs."""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import uvicorn

from strata.config import StrataConfig
from tests.conftest import find_free_port, wait_for_server

# ---------------------------------------------------------------------------
# R availability — central skip markers for integration tests
# ---------------------------------------------------------------------------
#
# R-based tests live in a few spots (analyzer integration, executor
# integration, the cross-language capstone in #59). Each one used to
# roll its own ``shutil.which("Rscript")`` skipif; consolidate them
# here so the skip reason wording stays consistent and the arrow-
# package probe doesn't get duplicated across files.


def _r_package_available(package: str) -> bool:
    """Probe ``requireNamespace(package)`` once at conftest import.

    Returning True requires Rscript on PATH *and* the named R package
    loadable. The 30s timeout is generous — a healthy R install
    resolves the namespace in well under a second; the only reason
    this could hang is a stale RPROFILE doing network I/O, which we'd
    rather skip with a clear reason than block CI on.

    Runs at module load: when Rscript is absent (most dev machines,
    Windows CI) the probe short-circuits without spawning — so the
    one-shot Rscript cost is paid only when R is actually installed
    *and* this conftest is loaded, which is also the only time it
    could matter.
    """
    if shutil.which("Rscript") is None:
        return False
    try:
        proc = subprocess.run(
            [
                "Rscript",
                "-e",
                f'q(status = if (requireNamespace("{package}", quietly = TRUE)) 0 else 1)',
            ],
            capture_output=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


_RSCRIPT_AVAILABLE = shutil.which("Rscript") is not None
_R_ARROW_AVAILABLE = _r_package_available("arrow")
_R_GGPLOT2_AVAILABLE = _r_package_available("ggplot2")

skip_if_no_r = pytest.mark.skipif(
    not _RSCRIPT_AVAILABLE,
    reason="Rscript not on PATH; install R (https://cran.r-project.org/) to run",
)

skip_if_no_r_arrow = pytest.mark.skipif(
    not _R_ARROW_AVAILABLE,
    reason=(
        "R 'arrow' package not loadable; install with "
        "`install.packages('arrow')` to run cross-language tests"
    ),
)

skip_if_no_r_ggplot2 = pytest.mark.skipif(
    not _R_GGPLOT2_AVAILABLE,
    reason=(
        "R 'ggplot2' package not loadable; install with "
        "`install.packages('ggplot2')` to run the ggplot display test"
    ),
)


def _check_notebook_extra() -> None:
    """Fail fast when the dev env is missing the [notebook] extra.

    The harness fixtures here point the per-notebook venv at the dev
    interpreter, so any cell-execution test imports `orjson` / `cloudpickle`
    from this venv. Plain `uv sync` skips those — running tests then fails
    deep inside a harness subprocess with a cryptic
    "Harness did not produce harness-result.json" error that doesn't point
    at the real fix. Surface it at collection time instead.
    """
    missing: list[str] = []
    try:
        import orjson  # noqa: F401
    except ImportError:
        missing.append("orjson")
    try:
        import cloudpickle  # noqa: F401
    except ImportError:
        missing.append("cloudpickle")
    if missing:
        pytest.exit(
            f"Notebook test prereqs missing: {', '.join(missing)}. "
            "Run `uv sync --all-extras` (matches CI). "
            'See CLAUDE.md "Build & Development" for the rationale.',
            returncode=2,
        )


_check_notebook_extra()


def _fake_uv_sync(
    notebook_dir: Path,
    *,
    timeout: int = 60,
    python_version: str | None = None,
) -> bool:
    """Create a minimal local env scaffold without invoking ``uv``."""
    del timeout, python_version
    notebook_dir = Path(notebook_dir)

    lockfile = notebook_dir / "uv.lock"
    if not lockfile.exists():
        lockfile.write_text(
            "\n".join(
                [
                    "version = 1",
                    'requires-python = ">=3.12"',
                    "",
                    "[[package]]",
                    'name = "pyarrow"',
                    'version = "0.0.0"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

    venv_python = notebook_dir / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    if not venv_python.exists():
        venv_python.write_text(
            (f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n'),
            encoding="utf-8",
        )
        venv_python.chmod(0o755)

    return True


@pytest.fixture(autouse=True)
def fast_notebook_env(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Stub notebook env setup unless a test explicitly opts into integration."""

    async def _noop_start(self):
        self._started = True

    if not request.node.get_closest_marker("warm_pool"):
        monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)

    # The production WS handler holds onto execution + inspect state for
    # 60s after the last disconnect so a reconnecting client doesn't lose
    # a running cell. In tests we want the teardown to fire immediately
    # on context exit unless a specific test exercises the grace window.
    monkeypatch.setattr("strata.notebook.ws._GRACE_CANCEL_SECONDS", 0.0)

    if request.node.get_closest_marker("integration"):
        return

    monkeypatch.setattr("strata.notebook.writer._uv_sync", _fake_uv_sync)
    monkeypatch.setattr("strata.notebook.session._uv_sync", _fake_uv_sync)

    async def _run_harness_direct(
        self,
        manifest_path: Path,
        venv_python: Path,
        timeout_seconds: float,
    ) -> dict[str, object]:
        """Run the harness directly with Python instead of ``uv run``."""
        cmd = [str(venv_python), str(self.harness_path), str(manifest_path)]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.session.path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout_seconds,
            )
        except asyncio.CancelledError:
            proc.kill()
            await asyncio.shield(proc.wait())
            raise
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        result_path = manifest_path.parent / "harness-result.json"
        if not result_path.exists():
            raise RuntimeError(f"Harness did not produce harness-result.json: {stderr.decode()}")

        with open(result_path) as f:
            return json.load(f)

    monkeypatch.setattr("strata.notebook.executor.CellExecutor._run_harness", _run_harness_direct)


@pytest.fixture
def notebook_executor_server(monkeypatch):
    """Run the notebook HTTP executor in a background thread.

    The build server we point at is on 127.0.0.1, which the production
    SSRF guard refuses; set STRATA_WORKER_ALLOW_LOCAL_HOSTS so the
    manifest URLs validate without disabling the scheme allowlist
    that the SSRF tests still want exercised.
    """
    from strata.notebook.remote_executor import create_notebook_executor_app

    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    port = find_free_port()
    app = create_notebook_executor_app()
    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not wait_for_server(port):
        raise RuntimeError(f"Notebook executor server failed to start on port {port}")

    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "execute_url": f"http://127.0.0.1:{port}/v1/execute",
            "notebook_execute_url": f"http://127.0.0.1:{port}/v1/notebook-execute",
            "manifest_execute_url": f"http://127.0.0.1:{port}/v1/execute-manifest",
        }
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)


@pytest.fixture
def notebook_build_server(tmp_path: Path):
    """Run a real service-mode Strata server for signed notebook build tests."""
    import strata.server as server_module
    from strata.artifact_store import get_artifact_store, reset_artifact_store
    from strata.server import ServerState, app
    from strata.transforms.build_store import get_build_store, reset_build_store

    port = find_free_port()
    artifact_dir = tmp_path / "service-artifacts"
    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=tmp_path / "service-cache",
        artifact_dir=artifact_dir,
        notebook_storage_dir=tmp_path,
        deployment_mode="service",
        transforms_config={"enabled": True},
    )

    reset_artifact_store()
    reset_build_store()
    server_module._state = ServerState(config)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not wait_for_server(port):
        raise RuntimeError(f"Signed notebook build server failed to start on port {port}")

    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "config": config,
            "artifact_store": get_artifact_store(artifact_dir),
            "build_store": get_build_store(artifact_dir / "artifacts.sqlite"),
        }
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)
        server_module._state = None
        reset_artifact_store()
        reset_build_store()


@pytest.fixture
def notebook_personal_server(tmp_path: Path):
    """Run a real personal-mode Strata server for signed notebook transport tests."""
    import strata.server as server_module
    from strata.artifact_store import get_artifact_store, reset_artifact_store
    from strata.server import ServerState, app
    from strata.transforms.build_store import get_build_store, reset_build_store

    port = find_free_port()
    artifact_dir = tmp_path / "personal-artifacts"
    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=tmp_path / "personal-cache",
        artifact_dir=artifact_dir,
        deployment_mode="personal",
    )

    reset_artifact_store()
    reset_build_store()
    server_module._state = ServerState(config)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(server_config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    if not wait_for_server(port):
        raise RuntimeError(f"Personal notebook server failed to start on port {port}")

    try:
        yield {
            "base_url": f"http://127.0.0.1:{port}",
            "config": config,
            "artifact_store": get_artifact_store(artifact_dir),
            "build_store": get_build_store(artifact_dir / "artifacts.sqlite"),
        }
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)
        server_module._state = None
        reset_artifact_store()
        reset_build_store()


# ---------------------------------------------------------------------------
# R-enabled notebook factory
# ---------------------------------------------------------------------------


@pytest.fixture
def r_notebook(tmp_path: Path):
    """Factory: build a notebook with mixed Python + R cells.

    Sibling of the Iceberg ``temp_warehouse`` for R (issue #59 capstone).
    Returns a callable

        make(cells=[(cell_id, after_id, source, language), ...])
            → (notebook_dir, NotebookSession)

    that builds the on-disk notebook, parses it, forces per-cell
    language to match ``language`` (parser defaults to Python; the
    forced override mirrors ``_make_r_notebook`` in
    ``test_language_r_executor.py`` so dispatch picks the right
    executor).

    The session's ``__init__`` runs ``_analyze_and_build_dag`` — for
    R cells that spawns ``Rscript`` to drive ``analyze_cell.R``, so
    callers must guard with ``skip_if_no_r``.

    No renv restore here. Tests that need the R ``arrow`` package
    rely on the system R install (gated with ``skip_if_no_r_arrow``);
    pre-restored renv libraries land in a follow-up PR once the renv
    bootstrap helper is wired into the fixture.
    """
    from strata.notebook.models import CellLanguage
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import (
        add_cell_to_notebook,
        create_notebook,
        write_cell,
    )

    _language_map = {
        "python": CellLanguage.PYTHON,
        "r": CellLanguage.R,
    }

    def _make(cells: list[tuple[str, str | None, str, str]]):
        notebook_dir = create_notebook(
            tmp_path / "r_notebook",
            "R Notebook",
            initialize_environment=False,
        )
        for cell_id, after_id, source, language in cells:
            add_cell_to_notebook(notebook_dir, cell_id, after_id, language=language)
            write_cell(notebook_dir, cell_id, source)

        notebook_state = parse_notebook(notebook_dir)
        by_id = {cid: lang for cid, _after, _src, lang in cells}
        for cell in notebook_state.cells:
            cell.language = _language_map[by_id[cell.id]]
        session = NotebookSession(notebook_state, notebook_dir)
        return notebook_dir, session

    return _make


# Canonical minimal renv project committed under tests/notebook/fixtures/.
# Pins only ``jsonlite`` (binary, restores in seconds from CRAN/RSPM or
# renv's global cache) so a real ``renv::restore`` stays fast in CI. The
# built ``renv/library`` is deliberately absent — restoring it is the
# point of the integration test.
_RENV_JSONLITE_FIXTURE = Path(__file__).parent / "fixtures" / "renv_jsonlite"


@pytest.fixture
def r_notebook_renv(r_notebook):
    """Like ``r_notebook`` but with a real renv project scaffold attached.

    Copies the committed jsonlite renv scaffold (``renv.lock`` +
    ``.Rprofile`` + ``renv/activate.R`` + ``renv/settings.json``)
    alongside the notebook, so a test can drive an actual
    ``renv::restore`` end-to-end — the gap left open in #59, where the
    plain ``r_notebook`` fixture runs against the system R library and
    ``test_renv_sync.py`` only mocks ``subprocess.run``.

    Deliberately does NOT run the restore itself: the test calls
    ``_renv_sync`` (or opens the session) and asserts, keeping the
    real-restore exercise visible in the test body. The ``renv/library``
    is not committed — restoring it from the lockfile is what's under
    test.
    """
    import shutil as _shutil

    def _make(cells: list[tuple[str, str | None, str, str]]):
        notebook_dir, session = r_notebook(cells)
        _shutil.copy(_RENV_JSONLITE_FIXTURE / "renv.lock", notebook_dir / "renv.lock")
        _shutil.copy(_RENV_JSONLITE_FIXTURE / ".Rprofile", notebook_dir / ".Rprofile")
        _shutil.copytree(_RENV_JSONLITE_FIXTURE / "renv", notebook_dir / "renv")
        return notebook_dir, session

    return _make

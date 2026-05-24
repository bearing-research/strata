"""Notebook-local pytest fixtures for fast default test runs."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
import threading
from pathlib import Path

import pytest
import uvicorn

from strata.config import StrataConfig
from tests.conftest import find_free_port, wait_for_server


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

"""Tests for environment lifecycle: venv creation, sync, and lockfile hashing.

Validates that:
- create_notebook() produces pyproject.toml and runs uv sync
- ensure_venv_synced() sets venv_python on the session
- Lockfile hash changes when dependencies change
- _uv_sync is best-effort (graceful failure)
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import tomli_w

from strata.notebook.dependencies import EnvironmentOperationLog, RequirementsImportResult
from strata.notebook.env import compute_lockfile_hash
from strata.notebook.python_versions import current_python_minor, format_requires_python
from strata.notebook.session import NotebookSession
from strata.notebook.writer import _uv_sync, create_notebook, update_environment_metadata

pytestmark = pytest.mark.integration


class TestCreateNotebookVenv:
    """create_notebook() should scaffold pyproject.toml and run uv sync."""

    def test_pyproject_toml_created(self, tmp_path: Path):
        """Notebook creation writes pyproject.toml."""
        nb_dir = create_notebook(tmp_path, "my_nb")
        assert (nb_dir / "pyproject.toml").exists()
        content = (nb_dir / "pyproject.toml").read_text()
        assert 'name = "my_nb"' in content
        assert f'requires-python = "{format_requires_python(current_python_minor())}"' in content

    def test_pyproject_toml_respects_requested_python_version(self, tmp_path: Path):
        """Notebook creation should persist the requested Python minor version."""
        nb_dir = create_notebook(tmp_path, "py312_nb", python_version="3.12")
        content = (nb_dir / "pyproject.toml").read_text()
        expected = format_requires_python("3.12")
        assert f'requires-python = "{expected}"' in content

    def test_uv_lock_created(self, tmp_path: Path):
        """uv sync produces uv.lock."""
        nb_dir = create_notebook(tmp_path, "test_lock")
        # uv sync should have run and produced uv.lock
        assert (nb_dir / "uv.lock").exists()

    def test_venv_created(self, tmp_path: Path):
        """uv sync produces .venv/ directory."""
        nb_dir = create_notebook(tmp_path, "test_venv")
        assert (nb_dir / ".venv").is_dir()

    def test_uv_sync_failure_does_not_raise(self, tmp_path: Path):
        """If uv is not on PATH, create_notebook still succeeds."""
        with patch("strata.notebook.writer.subprocess.run", side_effect=FileNotFoundError):
            nb_dir = create_notebook(tmp_path, "no_uv")
            # Notebook dir is created, just no venv
            assert (nb_dir / "notebook.toml").exists()
            assert (nb_dir / "pyproject.toml").exists()

    def test_can_skip_initial_environment_creation(self, tmp_path: Path):
        """Notebook scaffolding can skip the initial uv sync when requested."""
        with (
            patch("strata.notebook.writer._uv_sync") as mock_sync,
            patch("strata.notebook.writer._update_environment_metadata") as mock_update,
        ):
            nb_dir = create_notebook(
                tmp_path,
                "deferred_env",
                initialize_environment=False,
            )

        assert (nb_dir / "notebook.toml").exists()
        assert (nb_dir / "pyproject.toml").exists()
        assert not (nb_dir / "uv.lock").exists()
        assert not (nb_dir / ".venv").exists()
        mock_sync.assert_not_called()
        mock_update.assert_not_called()


class TestUvSyncHelper:
    """_uv_sync() helper function."""

    def test_returns_true_on_success(self, tmp_path: Path):
        """Successful sync returns True."""
        nb_dir = create_notebook(tmp_path, "sync_ok")
        # Already synced during creation, but calling again is idempotent
        assert _uv_sync(nb_dir) is True

    def test_sync_uses_requested_python_when_provided(self, tmp_path: Path):
        """Requested Python should be forwarded to uv sync."""
        with patch("strata.notebook.writer.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=["uv", "sync"],
                returncode=0,
                stdout=b"",
                stderr=b"",
            )
            assert _uv_sync(tmp_path, python_version="3.12") is True

        command = mock_run.call_args.args[0]
        assert command == ["uv", "sync", "--python", "3.12"]

    def test_returns_false_when_uv_missing(self, tmp_path: Path):
        """Returns False when uv is not available."""
        with patch("strata.notebook.writer.subprocess.run", side_effect=FileNotFoundError):
            assert _uv_sync(tmp_path) is False

    def test_returns_false_on_timeout(self, tmp_path: Path):
        """Returns False on timeout."""
        with patch(
            "strata.notebook.writer.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="uv sync", timeout=60),
        ):
            assert _uv_sync(tmp_path) is False

    def test_returns_false_on_failure(self, tmp_path: Path):
        """Returns False when uv sync exits non-zero."""
        with patch(
            "strata.notebook.writer.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                returncode=1, cmd="uv sync", stderr=b"some error"
            ),
        ):
            assert _uv_sync(tmp_path) is False


class TestSessionVenvPython:
    """ensure_venv_synced() sets session.venv_python."""

    def test_venv_python_set_after_sync(self, tmp_path: Path):
        """After opening a notebook, session.venv_python points to .venv/bin/python."""
        nb_dir = create_notebook(tmp_path, "venv_session")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)
        session.ensure_venv_synced()

        assert session.venv_python is not None
        assert "python" in str(session.venv_python)
        # Should point to the notebook's venv
        assert session.venv_python.exists()
        assert ".venv" in str(session.venv_python)
        assert session.environment_sync_state == "ready"
        assert session.environment_sync_error is None
        assert session.environment_sync_notice is None
        assert session.environment_last_synced_at is not None
        assert session.environment_last_sync_duration_ms is not None
        assert session.environment_interpreter_source == "venv"

    def test_venv_python_fallback_when_uv_missing(self, tmp_path: Path):
        """When uv is missing but .venv exists, keep using the notebook venv."""
        nb_dir = create_notebook(tmp_path, "no_uv_session")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch("strata.notebook.writer.subprocess.run", side_effect=FileNotFoundError):
            session.ensure_venv_synced()

        assert session.venv_python is not None
        assert session.venv_python.exists()
        assert ".venv" in str(session.venv_python)
        assert session.environment_sync_state == "ready"
        assert session.environment_sync_error is None
        assert session.environment_sync_notice is not None
        assert session.environment_interpreter_source == "venv"

    def test_path_fallback_when_uv_missing_and_no_venv(self, tmp_path: Path):
        """Without uv and without .venv, the session falls back to PATH python."""
        import shutil

        nb_dir = create_notebook(tmp_path, "no_uv_no_venv")
        shutil.rmtree(nb_dir / ".venv")

        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch("strata.notebook.writer.subprocess.run", side_effect=FileNotFoundError):
            session.ensure_venv_synced()

        assert session.venv_python == Path("python")
        assert session.environment_sync_state == "failed"
        assert session.environment_sync_error is not None
        assert session.environment_sync_notice is None
        assert session.environment_interpreter_source == "path"

    def test_refresh_environment_runtime_reuses_existing_venv(self, tmp_path: Path):
        """Dependency changes should refresh runtime state without a second uv sync."""
        nb_dir = create_notebook(tmp_path, "refresh_runtime")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch("strata.notebook.session._uv_sync") as mock_uv_sync:
            session.refresh_environment_runtime()

        mock_uv_sync.assert_not_called()
        assert session.venv_python is not None
        assert session.venv_python.exists()
        assert ".venv" in str(session.venv_python)
        assert session.environment_sync_state == "ready"
        assert session.environment_sync_error is None
        assert session.environment_sync_notice is None
        assert session.environment_interpreter_source == "venv"

    def test_refresh_environment_runtime_reuses_persisted_python_metadata(self, tmp_path: Path):
        """Refreshing runtime should trust persisted notebook metadata before probing."""
        nb_dir = create_notebook(tmp_path, "refresh_runtime_metadata")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch.object(
            session,
            "_probe_python_version",
            side_effect=AssertionError("python probe should not run"),
        ):
            session.refresh_environment_runtime()

        assert session.environment_python_version
        assert session.environment_sync_state == "ready"
        assert session.environment_interpreter_source == "venv"

    def test_refresh_environment_runtime_uses_pyvenv_cfg_when_metadata_missing(
        self, tmp_path: Path
    ):
        """Refreshing runtime should avoid spawning Python when pyvenv.cfg is present."""
        nb_dir = create_notebook(tmp_path, "refresh_runtime_pyvenv")

        notebook_toml = nb_dir / "notebook.toml"
        with open(notebook_toml, "rb") as f:
            data = tomllib.load(f)
        data["environment"] = {}
        with open(notebook_toml, "wb") as f:
            tomli_w.dump(data, f)

        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch(
            "strata.notebook.session.subprocess.run",
            side_effect=AssertionError("python probe should not run"),
        ):
            session.refresh_environment_runtime()

        assert session.environment_python_version
        assert session.environment_sync_state == "ready"
        assert session.environment_interpreter_source == "venv"

    def test_refresh_environment_runtime_falls_back_when_venv_missing(self, tmp_path: Path):
        """If the notebook venv is missing, refresh should fall back to uv sync."""
        import shutil

        nb_dir = create_notebook(tmp_path, "refresh_runtime_fallback")
        shutil.rmtree(nb_dir / ".venv")

        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch.object(session, "ensure_venv_synced") as mock_sync:
            session.refresh_environment_runtime()

        mock_sync.assert_called_once()


class TestDependencyChangeRefresh:
    """Dependency mutations should reuse the synced environment when possible."""

    @pytest.mark.asyncio
    async def test_on_dependencies_changed_uses_runtime_refresh(self, tmp_path: Path):
        """Post-mutation refresh should not trigger a second uv sync."""
        nb_dir = create_notebook(tmp_path, "dependency_refresh")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        with patch.object(session, "refresh_environment_runtime") as mock_refresh:
            with patch.object(session, "ensure_venv_synced") as mock_sync:
                with patch.object(
                    session,
                    "_invalidate_warm_pool_for_environment_change",
                ) as mock_invalidate:
                    with patch(
                        "strata.notebook.session.update_environment_metadata"
                    ) as mock_update_metadata:
                        await session.on_dependencies_changed()

        mock_refresh.assert_called_once()
        mock_sync.assert_not_called()
        mock_invalidate.assert_awaited_once()
        mock_update_metadata.assert_called_once_with(nb_dir)

    @pytest.mark.asyncio
    async def test_submit_environment_job_runs_to_completion(self, tmp_path: Path, monkeypatch):
        """Background dependency jobs should stream, finalize, and clear active state."""
        nb_dir = create_notebook(tmp_path, "dependency_job")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        async def fake_run_uv_command_streaming(
            notebook_dir: Path,
            args: list[str],
            *,
            timeout: int,
            display_name: str,
            on_update=None,
        ):
            del notebook_dir
            del timeout
            del display_name
            if on_update is not None:
                await on_update("stdout", "resolving\n", False)
            return SimpleNamespace(
                success=True,
                error=None,
                operation_log=EnvironmentOperationLog(
                    command=" ".join(["uv", *args]),
                    duration_ms=17,
                    stdout="resolving\n",
                    stderr="",
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
            )

        from types import SimpleNamespace

        monkeypatch.setattr(
            "strata.notebook.session.run_uv_command_streaming",
            fake_run_uv_command_streaming,
        )

        async def _noop_invalidate() -> None:
            return None

        monkeypatch.setattr(
            session,
            "_invalidate_warm_pool_for_environment_change",
            _noop_invalidate,
        )
        monkeypatch.setattr(
            "strata.notebook.session.update_environment_metadata",
            lambda path: None,
        )

        job = await session.submit_environment_job(action="add", package="six")
        assert job.status == "running"
        await session.wait_for_environment_job()
        assert session.environment_job is None
        assert session.wait_for_environment_job_task() is None
        history_path = nb_dir / ".strata" / "environment_jobs.json"
        assert history_path.exists()
        history_payload = json.loads(history_path.read_text())
        assert history_payload[0]["action"] == "add"
        assert history_payload[0]["status"] == "completed"
        current_job = session.serialize_environment_job_state()
        assert current_job is not None
        assert current_job["status"] == "completed"

    @pytest.mark.asyncio
    async def test_submit_environment_import_job_emits_warnings(self, tmp_path: Path, monkeypatch):
        """Async environment imports should surface warnings on the finished job payload."""
        nb_dir = create_notebook(tmp_path, "dependency_import_job")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)
        finished_payloads: list[tuple[str, dict[str, Any]]] = []
        expected_warning = (
            "Ignored conda channels from environment.yaml; notebook environments "
            "use pip/uv resolution."
        )

        async def _noop_event(*args, **kwargs):
            del args
            del kwargs
            return None

        async def _capture_message(event_type: str, payload: dict[str, object]) -> None:
            finished_payloads.append((event_type, payload))

        async def _fake_import_environment_yaml_text_streaming(
            notebook_dir: Path,
            environment_yaml_text: str,
            *,
            timeout: int = 180,
            on_update=None,
        ):
            del notebook_dir
            del environment_yaml_text
            del timeout
            if on_update is not None:
                await on_update("stderr", "Resolving translated environment\n", False)
            return RequirementsImportResult(
                success=True,
                lockfile_changed=True,
                dependencies=list_dependencies(session.path),
                imported_count=2,
                warnings=[expected_warning],
                operation_log=EnvironmentOperationLog(
                    command="uv sync",
                    duration_ms=17,
                    stdout="",
                    stderr="Resolving translated environment\n",
                    stdout_truncated=False,
                    stderr_truncated=False,
                ),
            )

        async def _fake_finalize_environment_job(
            job: EnvironmentJobSnapshot,
            *,
            lockfile_changed: bool,
            refresh_runtime: bool = True,
        ) -> list[str]:
            del refresh_runtime
            job.lockfile_changed = lockfile_changed
            job.stale_cell_count = 1
            job.stale_cell_ids = ["cell-1"]
            return ["cell-1"]

        from strata.notebook.dependencies import list_dependencies
        from strata.notebook.session import EnvironmentJobSnapshot

        monkeypatch.setattr(session, "_broadcast_environment_job_event", _noop_event)
        monkeypatch.setattr(session, "_broadcast_environment_job_message", _capture_message)
        monkeypatch.setattr(
            "strata.notebook.session.import_environment_yaml_text_streaming",
            _fake_import_environment_yaml_text_streaming,
        )
        monkeypatch.setattr(session, "_finalize_environment_job", _fake_finalize_environment_job)

        job = await session.submit_environment_job(
            action="import",
            environment_yaml_text="dependencies:\n  - six=1.17.0\n",
        )
        assert job.status == "running"
        await session.wait_for_environment_job()

        finished = next(
            payload
            for event_type, payload in finished_payloads
            if event_type == "environment_job_finished"
        )
        assert finished["warnings"] == [expected_warning]
        assert finished["imported_count"] == 2
        environment_job = finished["environment_job"]
        assert isinstance(environment_job, dict)
        assert environment_job["action"] == "import"

    def test_session_loads_environment_job_history_from_disk(self, tmp_path: Path):
        """Notebook sessions should rehydrate finished environment jobs on reopen."""
        nb_dir = create_notebook(tmp_path, "job_history_reload")
        history_path = nb_dir / ".strata" / "environment_jobs.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                [
                    {
                        "id": "job-123",
                        "action": "sync",
                        "command": "uv sync",
                        "status": "completed",
                        "phase": "completed",
                        "started_at": 1234567890,
                        "finished_at": 1234567990,
                        "duration_ms": 100,
                        "stdout": "Resolved 3 packages\n",
                        "stderr": "",
                        "stdout_truncated": False,
                        "stderr_truncated": False,
                        "lockfile_changed": True,
                        "stale_cell_count": 2,
                        "stale_cell_ids": ["cell-1", "cell-2"],
                        "error": None,
                    }
                ]
            )
        )

        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)

        assert len(session.serialize_environment_job_history()) == 1
        current_job = session.serialize_environment_job_state()
        assert current_job is not None
        assert current_job["command"] == "uv sync"
        assert session.serialize_environment_job_history()[0]["stale_cell_count"] == 2


class TestEnvironmentMetadata:
    """Environment metadata persisted to notebook.toml."""

    def test_update_environment_metadata_records_runtime_fields(self, tmp_path: Path):
        """Environment metadata is persisted to ``.strata/runtime.json`` —
        the values change on every sync and are not user-authored, so
        they do not belong in the committed ``notebook.toml``."""
        from strata.notebook.runtime_state import load_runtime_state

        nb_dir = create_notebook(tmp_path, "env_metadata")
        update_environment_metadata(nb_dir)

        environment = load_runtime_state(nb_dir).environment
        # has_lockfile / *_package_count are populated even when zero; the
        # truthy-leaning fields (timestamps, hashes) guard against the
        # writer forgetting to compute a field.
        assert environment.last_synced_at > 0
        assert environment.lockfile_hash

        with open(nb_dir / "notebook.toml", "rb") as f:
            data = tomllib.load(f)
        assert "environment" not in data

    def test_serialize_environment_state_includes_runtime_details(self, tmp_path: Path):
        """Live environment state should expose runtime source and sync metadata."""
        nb_dir = create_notebook(tmp_path, "env_state")
        from strata.notebook.parser import parse_notebook

        state = parse_notebook(nb_dir)
        session = NotebookSession(state, nb_dir)
        session.ensure_venv_synced()

        environment = session.serialize_environment_state()
        assert "requested_python_version" in environment
        assert "runtime_python_version" in environment
        assert environment["interpreter_source"] == "venv"
        assert "sync_notice" in environment
        assert "last_sync_duration_ms" in environment


class TestLockfileHash:
    """compute_lockfile_hash() produces consistent hashes."""

    def test_hash_with_lockfile(self, tmp_path: Path):
        """Hash is deterministic for a given lockfile."""
        nb_dir = create_notebook(tmp_path, "hash_test")
        h1 = compute_lockfile_hash(nb_dir)
        h2 = compute_lockfile_hash(nb_dir)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_sentinel_hash_without_lockfile(self, tmp_path: Path):
        """Without uv.lock, returns hash of empty string."""
        import hashlib

        expected = hashlib.sha256(b"").hexdigest()
        assert compute_lockfile_hash(tmp_path) == expected

    def test_hash_changes_on_lockfile_modification(self, tmp_path: Path):
        """Modifying uv.lock changes the hash."""
        nb_dir = create_notebook(tmp_path, "hash_change")
        h1 = compute_lockfile_hash(nb_dir)

        # Modify the lockfile
        lockfile = nb_dir / "uv.lock"
        lockfile.write_text(lockfile.read_text() + "\n# extra\n")

        h2 = compute_lockfile_hash(nb_dir)
        assert h1 != h2

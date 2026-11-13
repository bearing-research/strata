"""A notebook's background environment jobs: one at a time, streamed, kept in history.

``uv add`` / ``uv remove`` / ``uv sync``, a requirements or ``environment.yaml``
import, a change of Python, and ``renv`` init and install run as jobs a client
watches (``environment_job_*`` frames). This module runs them; what a job needs from
its notebook's session is :class:`EnvironmentJobHost`, so the session's other
concerns stay out of it. The session keeps the job state (``environment_job``,
``environment_job_history``, ``environment_job_task``) because its other readers
(the run gate, the staleness pass) take the same lock.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import threading
import time as _time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from vestigo.file_modes import private_file

from strata.notebook.dependencies import (
    EnvironmentOperationLog,
    RequirementsImportResult,
    _get_notebook_lock,
    import_environment_yaml_text_streaming,
    import_requirements_text_streaming,
    list_dependencies,
)
from strata.notebook.env import compute_lockfile_hash
from strata.notebook.models import CellStatus
from strata.notebook.protocol import MessageType
from strata.notebook.python_versions import read_requested_python_minor
from strata.notebook.staleness_explanations import skip_none
from strata.notebook.writer import update_environment_metadata
from strata.notebook.ws_payloads import (
    cell_status_payload,
    environment_job_event_payload,
    environment_job_finished_payload,
)

if TYPE_CHECKING:
    from strata.notebook.env_backend import EnvironmentBackend
    from strata.notebook.models import CellStaleness, NotebookState
    from strata.notebook.staleness_explanations import CausalityChain

logger = logging.getLogger(__name__)

# How many finished jobs a notebook keeps, in memory and in ``environment_jobs.json``.
_ENVIRONMENT_JOB_HISTORY_LIMIT = 8


@dataclass
class EnvironmentJobSnapshot:
    """One notebook-scoped background environment operation."""

    id: str
    action: str
    command: str
    status: str
    started_at: int
    package: str | None = None
    phase: str | None = None
    duration_ms: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    finished_at: int | None = None
    lockfile_changed: bool = False
    stale_cell_count: int = 0
    stale_cell_ids: list[str] = field(default_factory=list)
    error: str | None = None


class EnvironmentJobHost(Protocol):
    """What an environment job uses of its notebook's session, and nothing else."""

    id: str
    path: Path
    notebook_state: NotebookState
    backend: EnvironmentBackend
    causality_map: dict[str, CausalityChain]
    environment_job: EnvironmentJobSnapshot | None
    environment_job_history: list[EnvironmentJobSnapshot]
    environment_job_task: asyncio.Task[None] | None
    _environment_state_lock: threading.RLock

    def _assert_environment_job_can_start(self, action_label: str) -> None: ...
    def _apply_uv_sync_result(self, ok: bool, *, duration_ms: int) -> None: ...
    @property
    def _cached_system_r_version(self) -> str | None: ...
    async def _invalidate_warm_pool_for_environment_change(self) -> None: ...
    def _persist_r_runtime_success(
        self, *, lock_hash: str, last_synced_at: int, r_version: str | None
    ) -> None: ...
    def _record_dependency_install(self, duration_ms: int | None) -> None: ...
    async def compute_staleness_async(
        self, executing: str | None = None, *, hash_mounts: bool = True
    ) -> dict[str, CellStaleness]: ...
    def ensure_renv_synced(self) -> None: ...
    def refresh_environment_runtime(self) -> None: ...
    def serialize_cells(self) -> list[dict[str, Any]]: ...
    def serialize_environment_state(self) -> dict[str, Any]: ...
    def serialize_r_environment_state(
        self, *, include_packages: bool = False
    ) -> dict[str, Any]: ...
    async def start_missing_warm_pools(
        self, owner: EnvironmentJobSnapshot | str | None = None
    ) -> None: ...


class EnvironmentJobs:
    """Runs one notebook's environment jobs over its session."""

    def __init__(self, host: EnvironmentJobHost) -> None:
        self.host = host

    def serialize_state(self) -> dict[str, Any] | None:
        """Serialize the current or most recent environment job when present."""
        with self.host._environment_state_lock:
            if (
                self.host.environment_job is not None
                and self.host.environment_job.status == "running"
            ):
                return asdict(self.host.environment_job)
            if self.host.environment_job_history:
                return asdict(self.host.environment_job_history[0])
            return None

    def serialize_history(self) -> list[dict[str, Any]]:
        """Serialize recent finished environment jobs, newest first."""
        with self.host._environment_state_lock:
            return [asdict(job) for job in self.host.environment_job_history]

    def _history_path(self) -> Path:
        """Return the persisted recent-job history path for this notebook."""
        return self.host.path / ".strata" / "environment_jobs.json"

    def load_history(self) -> None:
        """Load recent finished environment jobs from notebook runtime state."""
        history_path = self._history_path()
        if not history_path.exists():
            return
        try:
            raw = json.loads(history_path.read_text())
        except Exception:
            logger.warning(
                "Failed to read environment job history for %s", self.host.path, exc_info=True
            )
            return
        history = [EnvironmentJobSnapshot(**item) for item in raw]
        self.host.environment_job_history = [
            job for job in history if job.status in {"completed", "failed"}
        ][:_ENVIRONMENT_JOB_HISTORY_LIMIT]

    def persist_history(self) -> None:
        """Persist recent finished environment jobs to notebook runtime state."""
        history_path = self._history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        private_file(history_path)
        history_path.write_text(
            json.dumps(
                [
                    asdict(job)
                    for job in self.host.environment_job_history[:_ENVIRONMENT_JOB_HISTORY_LIMIT]
                ],
                indent=2,
                sort_keys=True,
            )
        )

    def _record_finished(self, job: EnvironmentJobSnapshot) -> None:
        """Add a finished job to recent history and persist it."""
        with self.host._environment_state_lock:
            remaining = [
                existing for existing in self.host.environment_job_history if existing.id != job.id
            ]
            self.host.environment_job_history = [job, *remaining][:_ENVIRONMENT_JOB_HISTORY_LIMIT]
            try:
                self.persist_history()
            except Exception:
                logger.warning(
                    "Failed to persist environment job history for %s",
                    self.host.path,
                    exc_info=True,
                )

    def current_task(self) -> asyncio.Task[None] | None:
        """Return the current environment job task, if any."""
        with self.host._environment_state_lock:
            return self.host.environment_job_task

    async def wait(self) -> None:
        """Wait for the currently active environment job to finish."""
        task = self.current_task()
        if task is not None:
            await task

    async def submit(
        self,
        *,
        action: str,
        package: str | None = None,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
        python_version: str | None = None,
    ) -> EnvironmentJobSnapshot:
        """Start an asynchronous notebook environment job."""
        # R actions reuse the env-job machinery; ``_run_environment_job`` dispatches on ``action``.
        valid_actions = {
            "add",
            "remove",
            "sync",
            "import",
            "change_python",
            "r_init",
            "r_add",
        }
        if action not in valid_actions:
            raise ValueError(f"Unsupported environment job action: {action}")

        if action == "import":
            if (requirements_text is None) == (environment_yaml_text is None):
                raise ValueError(
                    "Import environment jobs require exactly one of requirements_text "
                    "or environment_yaml_text"
                )
        if action == "change_python" and not python_version:
            raise ValueError("change_python jobs require python_version")

        # Catch obvious user errors at submission rather than mid-job.
        if action in {"r_init", "r_add"} and not shutil.which("Rscript"):
            raise ValueError(
                "Rscript not found on PATH. Install R "
                "(https://cran.r-project.org/) before initialising renv."
            )
        if action == "r_add":
            from strata.notebook.dependencies import is_valid_r_package_name

            if not package or not is_valid_r_package_name(package):
                raise ValueError(
                    "r_add requires a valid R package name (match ``[A-Za-z][A-Za-z0-9.]*``)."
                )
            if not (self.host.path / "renv.lock").exists():
                raise ValueError(
                    "renv not initialised in this notebook. Click "
                    "'Initialize renv' before adding R packages."
                )

        if action == "import":
            action_label = (
                "requirements import"
                if requirements_text is not None
                else "environment.yaml import"
            )
        elif action == "change_python":
            action_label = f"change Python to {python_version}"
        elif action == "r_init":
            action_label = "renv::init"
        elif action == "r_add":
            action_label = f"renv::install {package}"
        else:
            action_label = f"{action} {package}".strip()
        with self.host._environment_state_lock:
            self.host._assert_environment_job_can_start(action_label)
            requested_python = read_requested_python_minor(self.host.path)
            command = "uv sync"
            if action == "add" and package:
                command = f"uv add {package}"
            elif action == "remove" and package:
                command = f"uv remove {package}"
            elif action == "sync" and requested_python:
                command = f"uv sync --python {requested_python}"
            elif action == "change_python":
                command = f"uv sync (python {python_version})"
            elif action == "r_init":
                command = "Rscript -e 'renv::init(bare = TRUE)'"
            elif action == "r_add":
                command = (
                    f'Rscript -e \'renv::install("{package}"); '
                    'renv::snapshot(type = "all", prompt = FALSE)\''
                )

            job = EnvironmentJobSnapshot(
                id=str(uuid.uuid4()),
                action=action,
                package=package,
                command=command,
                status="running",
                phase="uv_running",
                started_at=int(_time.time() * 1000),
            )
            self.host.environment_job = job

        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_STARTED, job)
        task = asyncio.create_task(
            self._run(
                job,
                requirements_text=requirements_text,
                environment_yaml_text=environment_yaml_text,
                python_version=python_version,
            )
        )
        with self.host._environment_state_lock:
            self.host.environment_job_task = task
        return job

    async def _run(
        self,
        job: EnvironmentJobSnapshot,
        *,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
        python_version: str | None = None,
    ) -> None:
        """Execute a background environment job and publish updates."""
        stale_cell_ids: list[str] = []
        import_result: RequirementsImportResult | None = None
        try:
            if job.action == "sync":
                stale_cell_ids = await self._run_sync(job)
            elif job.action == "import":
                stale_cell_ids, import_result = await self._run_import(
                    job,
                    requirements_text=requirements_text,
                    environment_yaml_text=environment_yaml_text,
                )
            elif job.action == "change_python":
                assert python_version is not None
                stale_cell_ids = await self._run_change_python(job, new_minor=python_version)
            elif job.action in {"r_init", "r_add"}:
                stale_cell_ids = await self._run_r(job)
            else:
                assert job.package is not None
                stale_cell_ids = await self._run_dependency(
                    job,
                    action=job.action,
                    package=job.package,
                )
            job.status = "completed"
            job.phase = "completed"
        except Exception as exc:
            logger.exception("Environment job %s failed for %s", job.action, self.host.path)
            job.status = "failed"
            job.phase = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = int(_time.time() * 1000)
            job.duration_ms = job.finished_at - job.started_at
            self._record_finished(job)
            payload: dict[str, Any] = {
                "environment_job": asdict(job),
                "environment_job_history": self.serialize_history(),
                "cells": self.host.serialize_cells(),
                **{
                    "lockfile_changed": job.lockfile_changed,
                    "stale_cell_count": job.stale_cell_count,
                    "stale_cell_ids": stale_cell_ids,
                },
            }
            if import_result is not None:
                payload["warnings"] = list(import_result.warnings)
                payload["imported_count"] = import_result.imported_count
            if job.status == "completed":
                payload.update(
                    {
                        "environment": self.host.serialize_environment_state(),
                        # So successful r_init / r_add jobs flip the R panel to synced
                        # without a reopen.
                        "r_environment": self.host.serialize_r_environment_state(),
                        "dependencies": [
                            {
                                "name": dep.name,
                                "version": str(dep.version) if dep.version else None,
                                "specifier": str(dep.specifier) if dep.specifier else None,
                            }
                            for dep in list_dependencies(self.host.path)
                        ],
                    }
                )
                from strata.notebook.dependencies import list_resolved_dependencies

                payload["resolved_dependencies"] = [
                    {
                        "name": dep.name,
                        "version": str(dep.version) if dep.version else None,
                        "specifier": str(dep.specifier) if dep.specifier else None,
                    }
                    for dep in list_resolved_dependencies(self.host.path)
                ]
            await self._broadcast_message(
                MessageType.ENVIRONMENT_JOB_FINISHED,
                environment_job_finished_payload(payload),
            )
            if job.action in {"add", "remove"}:
                await self.broadcast_staleness_updates(job.stale_cell_ids)
            with self.host._environment_state_lock:
                if self.host.environment_job is job:
                    self.host.environment_job = None
                current_task = asyncio.current_task()
                if self.host.environment_job_task is current_task:
                    self.host.environment_job_task = None

    async def _run_r(self, job: EnvironmentJobSnapshot) -> list[str]:
        """Run ``r_init`` / ``r_add`` as a background job.

        ``renv_init`` / ``renv_add`` stream, so ``environment_job_progress`` frames go
        out live during a long compile. A changed ``renv.lock`` changes the shared
        lockfile hash, so Python cells go stale along with R cells (over-stale).
        """
        from strata.notebook.dependencies import renv_add, renv_init

        old_lockfile_hash = compute_lockfile_hash(self.host.path)
        on_update = lambda stream, text, truncated: self._update_stream(  # noqa: E731
            job,
            stream=stream,
            text=text,
            truncated=truncated,
        )

        if job.action == "r_init":
            result = await renv_init(self.host.path, on_update=on_update)
        elif job.action == "r_add":
            assert job.package is not None
            result = await renv_add(self.host.path, job.package, on_update=on_update)
        else:  # pragma: no cover (submit_environment_job guards this)
            raise RuntimeError(f"Unsupported R job action: {job.action!r}")

        if result.operation_log is not None:
            self._apply_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or f"{job.action} failed")

        # The library now matches the new ``renv.lock``: record it as the last good
        # sync so ``sync_state`` reads ``ok`` and the next open doesn't re-restore.
        # The R version is the cached system probe, which just populated the library.
        renv_lock = self.host.path / "renv.lock"
        if renv_lock.exists():
            new_lockfile_hash = hashlib.sha256(renv_lock.read_bytes()).hexdigest()
            self.host._persist_r_runtime_success(
                lock_hash=new_lockfile_hash,
                last_synced_at=int(_time.time() * 1000),
                r_version=self.host._cached_system_r_version,
            )

        job.lockfile_changed = compute_lockfile_hash(self.host.path) != old_lockfile_hash
        return await self._finalize(job, lockfile_changed=job.lockfile_changed)

    async def _run_dependency(
        self,
        job: EnvironmentJobSnapshot,
        *,
        action: str,
        package: str,
    ) -> list[str]:
        """Run ``uv add`` / ``uv remove`` as a background job."""
        timeout = 120
        display_name = f"uv {action}"
        old_lockfile_hash = compute_lockfile_hash(self.host.path)
        lock = _get_notebook_lock(self.host.path)
        await asyncio.to_thread(lock.acquire)
        try:
            on_update = lambda stream, text, truncated: self._update_stream(  # noqa: E731
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            )
            if action == "add":
                result = await self.host.backend.add_streaming(
                    package, timeout=timeout, on_update=on_update
                )
            elif action == "remove":
                result = await self.host.backend.remove_streaming(
                    package, timeout=timeout, on_update=on_update
                )
            else:
                raise RuntimeError(f"Unsupported dependency action: {action!r}")
        finally:
            lock.release()

        self._apply_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or f"{display_name} failed")

        job.lockfile_changed = compute_lockfile_hash(self.host.path) != old_lockfile_hash
        return await self._finalize(job, lockfile_changed=job.lockfile_changed, installed=True)

    async def _run_import(
        self,
        job: EnvironmentJobSnapshot,
        *,
        requirements_text: str | None,
        environment_yaml_text: str | None,
    ) -> tuple[list[str], RequirementsImportResult]:
        """Run a requirements/environment.yaml import as a background job."""
        job.phase = "preparing_import"
        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)

        if requirements_text is not None:
            result = await import_requirements_text_streaming(
                self.host.path,
                requirements_text,
                on_update=lambda stream, text, truncated: self._update_stream(
                    job,
                    stream=stream,
                    text=text,
                    truncated=truncated,
                ),
            )
        else:
            assert environment_yaml_text is not None
            result = await import_environment_yaml_text_streaming(
                self.host.path,
                environment_yaml_text,
                on_update=lambda stream, text, truncated: self._update_stream(
                    job,
                    stream=stream,
                    text=text,
                    truncated=truncated,
                ),
            )

        self._apply_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or "Environment import failed")

        stale_cell_ids = await self._finalize(
            job,
            lockfile_changed=result.lockfile_changed,
            installed=True,
        )
        return stale_cell_ids, result

    async def _run_change_python(
        self,
        job: EnvironmentJobSnapshot,
        *,
        new_minor: str,
    ) -> list[str]:
        """Change ``requires-python`` and rebuild the venv on the new minor.

        If ``uv sync`` fails, restores the previous ``requires-python`` and re-syncs.
        If that also fails the notebook is left with the old pyproject and no venv,
        and the error goes to the job's operation log.
        """
        from strata.notebook.writer import update_requires_python

        old_minor = read_requested_python_minor(self.host.path)
        await asyncio.to_thread(update_requires_python, self.host.path, new_minor)

        # Otherwise uv sync reports an "interpreter mismatch" and refuses.
        venv_dir = self.host.path / ".venv"
        if venv_dir.exists():
            await asyncio.to_thread(shutil.rmtree, venv_dir, ignore_errors=True)

        old_lockfile_hash = compute_lockfile_hash(self.host.path)
        lock = _get_notebook_lock(self.host.path)
        await asyncio.to_thread(lock.acquire)
        try:
            on_update = lambda stream, text, truncated: self._update_stream(  # noqa: E731
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            )
            result = await self.host.backend.sync_streaming(
                python_version=None,
                timeout=180,
                on_update=on_update,
            )
        finally:
            lock.release()

        self._apply_operation_log(job, result.operation_log)
        if not result.success:
            # Rollback: restore previous requires-python and re-sync.
            if old_minor:
                try:
                    await asyncio.to_thread(update_requires_python, self.host.path, old_minor)
                    rollback = await self.host.backend.sync_streaming(
                        python_version=None,
                        timeout=180,
                        on_update=on_update,
                    )
                    self._apply_operation_log(job, rollback.operation_log)
                except Exception:
                    logger.exception(
                        "Failed to rollback python-version change for %s", self.host.path
                    )
            raise RuntimeError(result.error or f"uv sync failed for Python {new_minor}")

        job.lockfile_changed = compute_lockfile_hash(self.host.path) != old_lockfile_hash
        return await self._finalize(
            job,
            lockfile_changed=True,
            installed=True,
        )

    async def _run_sync(
        self,
        job: EnvironmentJobSnapshot,
    ) -> list[str]:
        """Run ``uv sync`` as a background job."""
        old_lockfile_hash = compute_lockfile_hash(self.host.path)
        requested_python = read_requested_python_minor(self.host.path)
        result = await self.host.backend.sync_streaming(
            python_version=requested_python,
            timeout=60,
            on_update=lambda stream, text, truncated: self._update_stream(
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            ),
        )
        self._apply_operation_log(job, result.operation_log)
        self.host._apply_uv_sync_result(
            result.success,
            duration_ms=result.operation_log.duration_ms or 0,
        )
        # No-op without ``renv.lock``; R restores whatever uv did, as an open did inline.
        try:
            await asyncio.to_thread(self.host.ensure_renv_synced)
        except Exception as exc:
            logger.warning("Failed to sync renv: %s", exc)
        if not result.success:
            raise RuntimeError(result.error or "uv sync failed")

        return await self._finalize(
            job,
            lockfile_changed=compute_lockfile_hash(self.host.path) != old_lockfile_hash,
            refresh_runtime=False,
        )

    async def _finalize(
        self,
        job: EnvironmentJobSnapshot,
        *,
        lockfile_changed: bool,
        refresh_runtime: bool = True,
        installed: bool = False,
    ) -> list[str]:
        """Refresh runtime metadata and staleness after a successful env mutation.

        ``installed`` means the job installed the current lockfile into the venv.
        """
        if refresh_runtime:
            job.phase = "refreshing_runtime"
            await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
            await asyncio.to_thread(self.host.refresh_environment_runtime)
        if installed:
            await asyncio.to_thread(self.host._record_dependency_install, job.duration_ms)

        job.phase = "invalidating_warm_pool"
        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        await self.host._invalidate_warm_pool_for_environment_change()

        job.phase = "recomputing_staleness"
        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        try:
            await asyncio.to_thread(update_environment_metadata, self.host.path)
        except Exception:
            logger.exception("Failed to update environment metadata")

        staleness_map = await self.host.compute_staleness_async()
        stale_cell_ids = [
            cell_id
            for cell_id, staleness in staleness_map.items()
            if staleness.status != CellStatus.READY
        ]
        job.phase = "starting_warm_pool"
        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        await self.host.start_missing_warm_pools(job)
        job.lockfile_changed = lockfile_changed
        job.stale_cell_count = len(stale_cell_ids)
        job.stale_cell_ids = stale_cell_ids
        return stale_cell_ids

    def _apply_operation_log(
        self,
        job: EnvironmentJobSnapshot,
        operation_log: EnvironmentOperationLog | None,
    ) -> None:
        """Copy final command log details onto a job snapshot."""
        if operation_log is None:
            return
        job.command = operation_log.command or job.command
        job.duration_ms = operation_log.duration_ms
        job.stdout = operation_log.stdout
        job.stderr = operation_log.stderr
        job.stdout_truncated = operation_log.stdout_truncated
        job.stderr_truncated = operation_log.stderr_truncated

    async def _update_stream(
        self,
        job: EnvironmentJobSnapshot,
        *,
        stream: str,
        text: str,
        truncated: bool,
    ) -> None:
        """Update a running job's live stdout/stderr snapshot and broadcast it."""
        if stream == "stdout":
            job.stdout = text
            job.stdout_truncated = truncated
        else:
            job.stderr = text
            job.stderr_truncated = truncated
        await self._broadcast_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)

    async def _broadcast_event(
        self,
        event_type: MessageType,
        job: EnvironmentJobSnapshot,
    ) -> None:
        """Broadcast one environment-job state snapshot (started / progress) over the notebook WS.

        The payload is validated through ``EnvironmentJobModel``, the documented wire shape.
        """
        await self._broadcast_message(
            event_type,
            environment_job_event_payload(asdict(job)),
        )

    async def _broadcast_message(
        self,
        event_type: MessageType,
        payload: dict[str, Any],
    ) -> None:
        """Send a structured notebook environment-job message to WS clients."""
        try:
            from strata.notebook.ws import broadcast_notebook_message, next_notebook_sequence
        except Exception:
            return

        await broadcast_notebook_message(
            self.host.id,
            {
                "type": event_type,
                "seq": next_notebook_sequence(self.host.id),
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "payload": payload,
            },
        )

    async def broadcast_staleness_updates(self, cell_ids: list[str]) -> None:
        """Broadcast current stale/idle statuses after an environment mutation."""
        if not cell_ids:
            return
        try:
            from strata.notebook.ws import broadcast_notebook_message, next_notebook_sequence
        except Exception:
            return

        for cell_id in cell_ids:
            cell = next(
                (
                    candidate
                    for candidate in self.host.notebook_state.cells
                    if candidate.id == cell_id
                ),
                None,
            )
            if cell is None:
                continue
            status = cell.status.value if isinstance(cell.status, CellStatus) else str(cell.status)
            causality = self.host.causality_map.get(cell.id)
            payload = cell_status_payload(
                cell.id,
                status,
                staleness_reasons=[
                    reason.value for reason in (cell.staleness.reasons if cell.staleness else [])
                ],
                causality=asdict(causality, dict_factory=skip_none)
                if causality is not None
                else None,
                changed_reads=(cell.staleness.changed_reads if cell.staleness else None) or None,
            )

            await broadcast_notebook_message(
                self.host.id,
                {
                    "type": MessageType.CELL_STATUS,
                    "seq": next_notebook_sequence(self.host.id),
                    "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    "payload": payload,
                },
            )

"""FastAPI router for notebook endpoints."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import tempfile
import time
import tomllib
import uuid
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from strata.notebook.authorship import MAX_AUTHOR_LENGTH, resolve_author
from strata.notebook.dependencies import (
    export_requirements_text,
    list_dependencies,
    list_resolved_dependencies,
    preview_environment_yaml_text,
    preview_requirements_text,
)
from strata.notebook.executor import CellExecutor
from strata.notebook.models import (
    CellLanguage,
    CellStatus,
    ConnectionSpec,
    MountSpec,
    WorkerSpec,
)
from strata.notebook.python_versions import (
    current_python_minor,
    normalize_python_minor,
    read_requested_python_minor,
)
from strata.notebook.quiesce import NotebookQuiesced
from strata.notebook.scopes import required_scope_for_route
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.timing import NotebookTimingRecorder
from strata.notebook.workers import (
    build_worker_catalog_with_health,
    notebook_worker_definitions_editable,
    validate_worker_assignment,
)
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    delete_notebook_directory,
    rename_notebook,
    reorder_cells,
    update_notebook_connections,
    update_notebook_env,
    update_notebook_mounts,
    update_notebook_timeout,
    update_notebook_worker,
    update_notebook_workers,
    write_cell,
)

if TYPE_CHECKING:
    from strata.notebook.remote_worker_supervisor import RemoteWorkerSupervisor

logger = logging.getLogger(__name__)

# Shared with the WebSocket handler.
_session_manager = SessionManager()

# Owns `ssh -L` forwards that in-server dispatch must reach; torn down in the lifespan.
_worker_supervisor: RemoteWorkerSupervisor | None = None


def _require_notebook_scope(request: Request) -> None:
    """Router-level gate: the caller must hold the scope this route needs.

    Uses the same ``notebook:*`` table as the WebSocket frames, so REST cannot do
    what the socket forbids. Keyed on the route's path template, so a new route is
    covered without a per-route decorator. No principal auth means nothing to check.
    """
    from strata.auth import get_principal
    from strata.server import get_state

    try:
        config = get_state().config
    except RuntimeError:
        return
    if not getattr(config, "principal_auth_enabled", False):
        return
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    required = required_scope_for_route(request.method, path)
    principal = get_principal()
    if principal is None or not principal.has_scope(required):
        raise HTTPException(status_code=403, detail=f"This route requires the {required} scope")


router = APIRouter(
    prefix="/v1/notebooks",
    tags=["notebooks"],
    dependencies=[Depends(_require_notebook_scope)],
)


def get_session_manager() -> SessionManager:
    """Export session manager for WebSocket handler."""
    return _session_manager


def get_worker_supervisor() -> RemoteWorkerSupervisor:
    """Return the process-wide SSH-worker supervisor, creating it on first use."""
    global _worker_supervisor
    if _worker_supervisor is None:
        from strata.notebook.remote_worker_supervisor import RemoteWorkerSupervisor

        _worker_supervisor = RemoteWorkerSupervisor()
    return _worker_supervisor


def shutdown_worker_supervisor() -> None:
    """Tear down all SSH tunnels on server shutdown (called from the lifespan)."""
    global _worker_supervisor
    if _worker_supervisor is not None:
        _worker_supervisor.shutdown()
        _worker_supervisor = None


def get_notebook_session(notebook_id: str, request: Request) -> NotebookSession:
    """FastAPI dependency: resolve ``notebook_id`` to an open session.

    Raises 404 when the session is unknown or the caller does not own it (404, not
    403, so probes cannot enumerate owners), matching the WS upgrade gate. Unowned
    notebooks and single-user deployments pass through.
    """
    session = _session_manager.get_session(notebook_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    _require_owner(session.notebook_state.owner, _caller_identity(request))
    return session


SessionDep = Annotated[NotebookSession, Depends(get_notebook_session)]


def _require_personal_mode_session_api() -> None:
    """Restrict session discovery/reconnect APIs to personal mode.

    They expose in-memory session IDs and notebook paths: a local reconnect helper,
    not a multi-user surface.
    """
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        # Route unit tests may use the router without full server state.
        return

    if state.config.deployment_mode != "personal":
        raise HTTPException(
            status_code=403,
            detail="Notebook session APIs are only available in personal mode",
        )


def _reuse_open_session_by_path() -> bool:
    """Enable path-based session reuse only in personal mode."""
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return True

    return state.config.deployment_mode == "personal"


def _get_notebook_storage_root() -> Path | None:
    """Return the configured notebook storage root when server state is available."""
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return None

    configured_path = getattr(state.config, "notebook_storage_dir", None)
    if configured_path is None:
        return None
    return Path(configured_path).resolve()


# Email-shaped names survive; anything else becomes ``_`` so a hostile header
# can't escape the storage root with ``../`` or null bytes.
_USER_DIR_SAFE_RE = re.compile(r"[^A-Za-z0-9._@\-]")


def _sanitize_user_dir_name(identity: str) -> str | None:
    """Map a caller identity (typically email) to a filesystem-safe dir name.

    Deterministic, so a user's notebooks stay under one path. ``None`` when the
    input is empty or sanitizes to empty (the caller falls back to no scoping).
    """
    if not identity:
        return None
    cleaned = _USER_DIR_SAFE_RE.sub("_", identity.strip())
    cleaned = cleaned.strip("._-") or ""
    return cleaned or None


def _get_user_storage_root(request: Request | None) -> Path | None:
    """Return the storage root scoped to the calling user.

    ``None`` without server state. The base root when there is no request, no
    header configured, or no identity. Otherwise ``<base>/<sanitized_identity>/``,
    created on first call.
    """
    base = _get_notebook_storage_root()
    if base is None:
        return None
    if request is None:
        return base

    identity = _caller_identity(request)
    if identity is None:
        return base

    safe = _sanitize_user_dir_name(identity)
    if safe is None:
        return base

    user_root = (base / safe).resolve()
    # Defense-in-depth: sanitization should already make escape impossible.
    if user_root != base and base not in user_root.parents:
        return base
    user_root.mkdir(parents=True, exist_ok=True)
    return user_root


def _require_personal_mode_notebook_delete() -> None:
    """Restrict destructive notebook deletion to personal mode for now."""
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return

    if state.config.deployment_mode != "personal":
        raise HTTPException(
            status_code=403,
            detail="Notebook deletion is only available in personal mode",
        )


def _user_scoping_enabled() -> bool:
    """Return whether a per-request identity header is configured.

    Separates single-user mode from per-user mode, where a ``None`` from
    ``_caller_identity`` must deny access to owned notebooks rather than allow it.
    """
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return False
    return bool(getattr(state.config, "personal_mode_user_header", None))


def _caller_identity(request: Request) -> str | None:
    """Resolve the calling user's identity for personal-mode scoping.

    Read from the header named by ``personal_mode_user_header`` (set by an
    authenticating proxy). ``None`` both when the header is not configured and when
    the request lacks it; owner checks must also consult ``_user_scoping_enabled``,
    or omitting the header would bypass them.
    """
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return None

    header_name = getattr(state.config, "personal_mode_user_header", None)
    if not header_name:
        return None

    value = request.headers.get(header_name)
    if not value:
        return None
    value = value.strip()
    return value or None


def _require_owner(notebook_owner: str | None, caller: str | None) -> None:
    """Reject the request if a non-owner is touching an owned notebook.

    Unowned notebooks, and every notebook when ``personal_mode_user_header`` is
    unset, are open to any caller. With the header configured, a request that omits
    it is denied, so a leaked ``session_id`` cannot drive an owned notebook. Denials
    are a generic 404 so probes cannot enumerate owners.
    """
    if notebook_owner is None:
        return
    if caller is None:
        if _user_scoping_enabled():
            raise HTTPException(status_code=404, detail="Notebook not found")
        return
    if notebook_owner != caller:
        raise HTTPException(status_code=404, detail="Notebook not found")


def _timed_json_response(
    data: dict,
    *,
    timing: NotebookTimingRecorder,
    route_name: str,
    log_context: str,
) -> JSONResponse:
    timings_ms = timing.as_dict()
    logger.info(
        "%s timing %s",
        route_name,
        {
            "context": log_context,
            "timings_ms": {name: round(duration, 1) for name, duration in timings_ms.items()},
        },
    )
    return JSONResponse(
        content=jsonable_encoder(data),
        headers={"Server-Timing": timing.server_timing_header()},
    )


def validate_package_name(package: str) -> str:
    """Validate and sanitize a package specifier; rejects shell metacharacters."""
    if len(package) > 200:
        raise ValueError("Package specifier too long")
    if any(c in package for c in ";&|`$(){}!<>\"'\n\r\t"):
        raise ValueError("Package specifier contains invalid characters")
    return package.strip()


def _validate_notebook_path(
    user_path: str,
    label: str = "path",
    request: Request | None = None,
) -> Path:
    """Validate that a notebook path is safe and confined to the storage root.

    With ``request`` and per-user scoping, the path must lie inside the caller's
    subdir. This is the security boundary that stops one user passing another
    user's path in ``parent_path`` / ``notebook_path``.
    """
    path = Path(user_path)
    if ".." in path.parts:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: path traversal not allowed")

    user_root = _get_user_storage_root(request)
    base_root = _get_notebook_storage_root()
    resolution_root = user_root if user_root is not None else base_root
    resolved = (
        (resolution_root / path).resolve()
        if resolution_root is not None and not path.is_absolute()
        else path.resolve()
    )

    # Confine to the per-user root, or the base root in single-user mode.
    boundary = user_root if user_root is not None else base_root
    if boundary is not None and resolved != boundary and boundary not in resolved.parents:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}: must be inside configured notebook storage",
        )

    return resolved


def _safe_filename(name: str) -> str:
    """Sanitize a string for use in Content-Disposition."""
    safe = re.sub(r"[^\w\s.-]", "", name)
    safe = re.sub(r"\s+", "_", safe).strip("_") or "notebook"
    return safe


def validate_env_vars(env: dict[str, str]) -> dict[str, str]:
    """Validate notebook env var keys and values."""
    validated: dict[str, str] = {}
    for key, value in env.items():
        normalized_key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized_key):
            raise ValueError(f"Invalid env var name: {key}")
        if any(c in value for c in "\0\r\n"):
            raise ValueError(f"Invalid env var value for {key}")
        validated[normalized_key] = value
    return validated


async def _serialize_worker_catalog(
    session: NotebookSession,
    *,
    force_refresh: bool = False,
) -> dict:
    return {
        "workers": await build_worker_catalog_with_health(
            session.notebook_state,
            force_refresh=force_refresh,
        ),
        "definitions_editable": notebook_worker_definitions_editable(session.notebook_state),
        "health_checked_at": int(time.time() * 1000),
    }


def _serialize_environment_change(session: NotebookSession, staleness_map: dict) -> dict:
    """Summarize environment change impact for sidebar UX."""
    stale_cell_ids = [
        cell_id
        for cell_id, staleness in staleness_map.items()
        if staleness.status != CellStatus.READY
    ]
    return {
        "stale_cell_count": len(stale_cell_ids),
        "stale_cell_ids": stale_cell_ids,
        "warm_pool_reset": session.warm_pool is not None,
    }


def _serialize_notebook_runtime_config(request: Request | None = None) -> dict:
    """Serialize frontend-relevant notebook runtime defaults.

    With ``request`` and per-user scoping, ``default_parent_path`` is the caller's
    subdir, so new notebooks land under the right user.
    """
    deployment_mode = "service"
    default_parent_path = Path.home() / ".strata" / "notebooks"
    available_python_versions = [current_python_minor()]
    team_store_configured = False

    try:
        from strata.server import get_state

        state = get_state()
        deployment_mode = getattr(state.config, "deployment_mode", deployment_mode)
        team_store_configured = bool(getattr(state.config, "notebook_remote_store_url", None))
        user_root = _get_user_storage_root(request)
        if user_root is not None:
            default_parent_path = user_root
        else:
            configured_path = getattr(state.config, "notebook_storage_dir", None)
            if configured_path is not None:
                default_parent_path = Path(configured_path)
        configured_versions = getattr(state.config, "notebook_python_versions", None)
        if isinstance(configured_versions, list) and configured_versions:
            available_python_versions = [
                normalize_python_minor(str(version)) for version in configured_versions
            ]
    except RuntimeError:
        pass

    return {
        "deployment_mode": deployment_mode,
        "default_parent_path": str(default_parent_path),
        "available_python_versions": available_python_versions,
        "default_python_version": available_python_versions[0],
        "python_selection_fixed": len(available_python_versions) <= 1,
        # Registry routes read through the tenant-scoped gate (a team store answers
        # them when configured), so a registry is always available to show.
        "registry_enabled": True,
        # Without a team store the promote route answers 409, so the strip hides it.
        "team_store_configured": team_store_configured,
    }


def _serialize_dependency_info_list(dependencies: list) -> list[dict]:
    """Serialize dependency metadata for API responses."""
    return [
        {
            "name": dep.name,
            "version": str(dep.version) if dep.version else None,
            "specifier": str(dep.specifier) if dep.specifier else None,
        }
        for dep in dependencies
    ]


def _serialize_environment_payload(session: NotebookSession) -> dict:
    """Serialize the current environment plus direct and resolved dependencies.

    ``r_environment`` is always present (Python-only notebooks get
    ``has_lockfile: false``, ``sync_state: "absent"``): the frontend refreshes R UI
    state from every env-related payload.
    """
    return {
        "environment": session.serialize_environment_state(),
        "environment_job": session.serialize_environment_job_state(),
        "environment_job_history": session.serialize_environment_job_history(),
        "dependencies": _serialize_dependency_info_list(list_dependencies(session.path)),
        "resolved_dependencies": _serialize_dependency_info_list(
            list_resolved_dependencies(session.path)
        ),
        "r_environment": session.serialize_r_environment_state(),
    }


def _serialize_import_preview(result) -> dict:
    """Serialize an import preview response."""
    return {
        "preview_dependencies": _serialize_dependency_info_list(result.dependencies),
        "normalized_requirements": result.normalized_requirements,
        "imported_count": result.imported_count,
        "warnings": list(result.warnings),
        "additions": _serialize_dependency_info_list(result.additions),
        "removals": _serialize_dependency_info_list(result.removals),
        "unchanged": _serialize_dependency_info_list(result.unchanged),
    }


def _serialize_environment_operation_log(raw: object | None) -> dict | None:
    """Serialize structured uv command details for the UI."""
    if raw is None:
        return None

    return {
        "command": getattr(raw, "command", ""),
        "duration_ms": getattr(raw, "duration_ms", None),
        "stdout": getattr(raw, "stdout", ""),
        "stderr": getattr(raw, "stderr", ""),
        "stdout_truncated": getattr(raw, "stdout_truncated", False),
        "stderr_truncated": getattr(raw, "stderr_truncated", False),
    }


def _serialize_result_operation_log(result: object) -> dict:
    """Serialize operation log details from a dependency/import result."""
    operation_log = _serialize_environment_operation_log(getattr(result, "operation_log", None))
    if operation_log is None:
        return {}
    return {"operation_log": operation_log}


def _serialize_operation_error_detail(message: str, result: object) -> dict:
    """Build a structured HTTP error detail payload with optional command logs."""
    detail: dict[str, object] = {"message": message}
    operation_log = _serialize_environment_operation_log(getattr(result, "operation_log", None))
    if operation_log is not None:
        detail["operation_log"] = operation_log
    return detail


def _quiesced_conflict(exc: NotebookQuiesced) -> HTTPException:
    """The same 409 the app-level handler gives, for routes that wrap errors.

    Those routes turn anything unexpected into a 500, which would report a
    notebook held for a copy as a server fault.
    """
    return HTTPException(status_code=409, detail={"message": str(exc), "code": exc.code})


def _raise_environment_busy(session: NotebookSession, message: str) -> None:
    """Raise a structured 409 conflict for competing env/runtime operations."""
    raise HTTPException(
        status_code=409,
        detail={
            "message": message,
            "code": "ENVIRONMENT_BUSY",
            "environment_job": session.serialize_environment_job_state(),
        },
    )


# --- Request/Response Models ---


class OpenNotebookRequest(BaseModel):
    """Request to open a notebook."""

    path: str = "..."


class CreateNotebookRequest(BaseModel):
    """Request to create a new notebook."""

    parent_path: str
    name: str
    python_version: str | None = Field(default=None, max_length=16)
    starter_cell: bool = False

    @field_validator("python_version")
    @classmethod
    def validate_python_version_field(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return normalize_python_minor(value)


class UpdateCellSourceRequest(BaseModel):
    """Request to update cell source."""

    source: str = Field(..., max_length=1_000_000)  # ~1-4 MB UTF-8
    author: str | None = Field(default=None, max_length=MAX_AUTHOR_LENGTH)
    # Overwrites a cell someone else changed moments ago (see ``cell_locked``).
    force: bool = False


class UpdateCellTestsRequest(BaseModel):
    """Request to set a cell's unit-test source."""

    source: str = Field(..., max_length=1_000_000)


class AddCellRequest(BaseModel):
    """Request to add a new cell."""

    after_cell_id: str | None = None
    language: CellLanguage = CellLanguage.PYTHON
    # Credit for an unauthenticated caller. Ignored in service mode, where the
    # principal answers and a self-declared name would be a claim.
    author: str | None = Field(default=None, max_length=MAX_AUTHOR_LENGTH)


class MountConfigRequest(BaseModel):
    """Request to replace a mount list."""

    mounts: list[MountSpec] = Field(default_factory=list)


class ConnectionConfigRequest(BaseModel):
    """Request to replace the full ``[connections.<name>]`` set.

    The list is canonical: an empty list deletes every connection. Auth literals
    are scrubbed on write (placeholders on disk); the running session keeps the
    values in memory until reload.
    """

    connections: list[ConnectionSpec] = Field(default_factory=list)


class WorkerConfigRequest(BaseModel):
    """Request to replace a worker setting."""

    worker: str | None = Field(default=None, max_length=200)

    @field_validator("worker")
    @classmethod
    def normalize_worker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class WorkersConfigRequest(BaseModel):
    """Request to replace notebook-scoped worker definitions."""

    workers: list[WorkerSpec] = Field(default_factory=list)


class SshWorkerRequest(BaseModel):
    """Request to provision + tunnel a worker over SSH and register it."""

    ssh_target: str = Field(..., min_length=1, max_length=255)
    name: str | None = Field(default=None, max_length=64)
    remote_port: int | None = Field(default=None, ge=1, le=65535)
    local_port: int | None = Field(default=None, ge=1, le=65535)
    extras: str = Field(default="notebook", max_length=200)
    pin: str | None = Field(default=None, max_length=100)
    install: bool = True
    set_default: bool = False


class TimeoutConfigRequest(BaseModel):
    """Request to replace a timeout setting."""

    timeout: float | None = Field(default=None, gt=0, le=86_400)


class VariantActiveRequest(BaseModel):
    """Request to switch the active variant and/or the mode of a group.

    Both fields are optional; at least one must be present.
    """

    active: str | None = Field(None, pattern=r"^([a-zA-Z_][a-zA-Z0-9_]*)?$")
    mode: str | None = Field(None, pattern=r"^(switch|sweep)$")


class EnvConfigRequest(BaseModel):
    """Request to replace env vars."""

    env: dict[str, str] = Field(default_factory=dict)

    @field_validator("env")
    @classmethod
    def validate_env_field(cls, value: dict[str, str]) -> dict[str, str]:
        return validate_env_vars(value)


class ReorderCellsRequest(BaseModel):
    """Request to reorder cells."""

    cell_ids: list[str]


class RenameNotebookRequest(BaseModel):
    """Request to rename notebook."""

    name: str = Field(..., min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Notebook name cannot be empty")
        return normalized


class AddDependencyRequest(BaseModel):
    """Request to add a dependency."""

    package: str = Field(..., max_length=200)  # e.g. "requests" or "pandas>=2.0"

    @field_validator("package")
    @classmethod
    def validate_package_field(cls, v: str) -> str:
        return validate_package_name(v)


class RemoveDependencyRequest(BaseModel):
    """Request to remove a dependency."""

    package: str = Field(..., max_length=200)

    @field_validator("package")
    @classmethod
    def validate_package_field(cls, v: str) -> str:
        return validate_package_name(v)


class EnvironmentJobRequest(BaseModel):
    """Request to submit a background environment job."""

    action: str = Field(..., max_length=32)
    package: str | None = Field(default=None, max_length=200)
    requirements: str | None = Field(default=None, max_length=500_000)
    environment_yaml: str | None = Field(default=None, max_length=500_000)

    @field_validator("action")
    @classmethod
    def validate_action_field(cls, value: str) -> str:
        normalized = value.strip().lower()
        # ``r_init`` / ``r_add`` reuse the env-job machinery (Rscript subprocess,
        # job tracking, WS frames).
        if normalized not in {
            "add",
            "remove",
            "sync",
            "import",
            "change_python",
            "r_init",
            "r_add",
        }:
            raise ValueError("Unsupported environment job action")
        return normalized

    @field_validator("package")
    @classmethod
    def validate_package_field(cls, value: str | None) -> str | None:
        # Rejects shell metacharacters for any action; the R name-shape check runs
        # again in ``submit_environment_job`` before Rscript sees it.
        if value is None:
            return None
        return validate_package_name(value)


class ImportRequirementsRequest(BaseModel):
    """Request to import direct dependencies from requirements text."""

    requirements: str = Field(..., max_length=500_000)


class ImportEnvironmentYamlRequest(BaseModel):
    """Request to import dependencies from ``environment.yaml`` text."""

    environment_yaml: str = Field(..., max_length=500_000)


class PreviewRequirementsRequest(BaseModel):
    """Request to preview direct dependency import from requirements text."""

    requirements: str = Field(..., max_length=500_000)


class PreviewEnvironmentYamlRequest(BaseModel):
    """Request to preview dependency import from ``environment.yaml`` text."""

    environment_yaml: str = Field(..., max_length=500_000)


class PromoteArtifactRequest(BaseModel):
    """Send a cell's result to the team store, under a name or only its chain."""

    name: str | None = Field(default=None, min_length=1, max_length=512)
    alias: str | None = Field(default=None, max_length=128)
    tags: dict[str, str] = Field(default_factory=dict)
    # Also write it into this Iceberg table, in the team store's catalog.
    table: str | None = Field(default=None, max_length=1024)


# --- Endpoints ---


@router.post("/open")
async def open_notebook(req: OpenNotebookRequest, request: Request) -> JSONResponse:
    """Open a notebook directory and return its state, session ID and DAG.

    With per-user scoping, the path must lie in the caller's storage subdir.
    """
    timing = NotebookTimingRecorder()

    try:
        with timing.phase("validate"):
            notebook_path = _validate_notebook_path(req.path, "notebook path", request)
            if not notebook_path.exists():
                raise HTTPException(status_code=404, detail="Notebook directory not found")

        with timing.phase("session_open"):
            session = _session_manager.open_notebook(
                notebook_path,
                reuse_existing=_reuse_open_session_by_path(),
                timing=timing,
            )

        with timing.phase("serialize"):
            data = session.serialize_notebook_state()
            data["session_id"] = session.id
            data["path"] = str(session.path)
            data["dag"] = _format_dag(session)
            data.update(_serialize_notebook_runtime_config(request))
        return _timed_json_response(
            data,
            timing=timing,
            route_name="notebook_open",
            log_context=str(notebook_path),
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/create")
async def create_new_notebook(req: CreateNotebookRequest, request: Request) -> JSONResponse:
    """Create a new notebook and return its state.

    With per-user scoping, ``parent_path`` must lie in the caller's storage subdir.
    """
    timing = NotebookTimingRecorder()
    try:
        with timing.phase("validate"):
            parent_path = _validate_notebook_path(req.parent_path, "parent path", request)
            runtime_config = _serialize_notebook_runtime_config(request)
            selected_python_version = req.python_version or runtime_config["default_python_version"]
            allowed_python_versions = runtime_config["available_python_versions"]
            if selected_python_version not in allowed_python_versions:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Python {selected_python_version} is not available for notebook creation"
                    ),
                )

        expected_dir = parent_path / req.name.lower().replace(" ", "_")
        if (expected_dir / "notebook.toml").exists():
            raise HTTPException(
                status_code=409,
                detail=f"A notebook already exists at {expected_dir}. Use Open to open it.",
            )

        with timing.phase("create_notebook"):
            notebook_dir = create_notebook(
                parent_path,
                req.name,
                python_version=selected_python_version,
                initialize_environment=False,
                owner=_caller_identity(request),
            )
        if req.starter_cell:
            with timing.phase("create_starter_cell"):
                add_cell_to_notebook(notebook_dir, str(uuid.uuid4()))
        with timing.phase("session_open"):
            session = _session_manager.open_notebook(
                notebook_dir,
                defer_initial_venv_sync=True,
                timing=timing,
            )
        with timing.phase("environment_job_submit"):
            try:
                await session.submit_environment_job(action="sync")
            except Exception as exc:
                logger.exception(
                    "Failed to start initial environment bootstrap for %s",
                    notebook_dir,
                )
                session.environment_sync_state = "failed"
                session.environment_sync_error = (
                    f"Failed to start notebook environment initialization: {exc}"
                )
                session.environment_sync_notice = None

        with timing.phase("serialize"):
            data = session.serialize_notebook_state()
            data["session_id"] = session.id
            data["path"] = str(session.path)
            data.update(runtime_config)
        return _timed_json_response(
            data,
            timing=timing,
            route_name="notebook_create",
            log_context=str(notebook_dir),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


# Bigger notebooks are almost always embedded image outputs; cap conservatively.
_MAX_IPYNB_UPLOAD_BYTES = 50 * 1024 * 1024


def _resolve_import_target(
    request: Request,
    *,
    name: str | None,
    parent_path: str | None,
    default_stem: str,
) -> tuple[Path, str, Path]:
    """Where an imported notebook lands: ``(parent, name, directory)``.

    Shared by every import route so the name, which becomes a path, passes one set
    of traversal checks whatever the upload format.

    Raises:
        HTTPException: 400 for an unconfigured root or a name that escapes it,
            409 when a notebook already exists there.
    """
    if parent_path:
        target_parent = _validate_notebook_path(parent_path, "parent path", request)
    else:
        user_root = _get_user_storage_root(request)
        target_parent = user_root or _get_notebook_storage_root()
        if target_parent is None:
            raise HTTPException(
                status_code=400,
                detail="Notebook storage root is not configured on this server",
            )
        target_parent.mkdir(parents=True, exist_ok=True)

    # Same slugify rule as `create_notebook`, so layouts match UI-created notebooks.
    raw_name = name or default_stem or "imported"

    # The name flows into a filesystem path; `target_parent / "../x"` would escape.
    if "/" in raw_name or "\\" in raw_name or "\0" in raw_name or raw_name in ("..", "."):
        raise HTTPException(
            status_code=400,
            detail="Invalid notebook name: must not contain path separators or traversal segments",
        )
    if ".." in Path(raw_name).parts:
        raise HTTPException(
            status_code=400,
            detail="Invalid notebook name: must not contain '..' segments",
        )

    notebook_dir_name = raw_name.lower().replace(" ", "_")
    candidate_dir = target_parent / notebook_dir_name
    # Belt-and-braces against escapes the textual checks miss.
    try:
        candidate_resolved = candidate_dir.resolve()
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid notebook name: {exc}")
    target_resolved = target_parent.resolve()
    if candidate_resolved != target_resolved and target_resolved not in candidate_resolved.parents:
        raise HTTPException(
            status_code=400,
            detail="Invalid notebook name: resolves outside the configured storage root",
        )

    if (candidate_dir / "notebook.toml").exists():
        raise HTTPException(
            status_code=409,
            detail=(
                f"A notebook already exists at {candidate_dir}. "
                "Use Open to open it, or import with a different --name."
            ),
        )
    return target_parent, raw_name, candidate_dir


@router.post("/import")
async def import_jupyter_notebook(
    request: Request,
    file: UploadFile = File(..., description="The .ipynb file to import."),
    name: str | None = Form(
        default=None,
        description="Target notebook name. Defaults to the uploaded file's stem.",
    ),
    parent_path: str | None = Form(
        default=None,
        description=(
            "Where the new notebook directory lands. Must be inside the "
            "configured storage root. Defaults to the user's storage root."
        ),
    ),
) -> JSONResponse:
    """Convert an uploaded ``.ipynb`` into a Strata notebook and open it.

    The import report (sources, magic translation, captured deps, warnings) is
    returned inline.
    """
    from strata.notebook.jupyter_import import import_notebook

    timing = NotebookTimingRecorder()

    # Read once into memory (bounded by the cap): it must parse as JSON before
    # anything touches the storage tree.
    with timing.phase("read_upload"):
        try:
            payload = await file.read()
        finally:
            await file.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty .ipynb upload")
    if len(payload) > _MAX_IPYNB_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(f".ipynb upload exceeds the {_MAX_IPYNB_UPLOAD_BYTES // (1024 * 1024)} MB cap"),
        )

    # Malformed JSON becomes a clean 400, not a 500 from inside the converter.
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .ipynb JSON: {exc}")

    with timing.phase("validate"):
        target_parent, raw_name, _candidate_dir = _resolve_import_target(
            request,
            name=name,
            parent_path=parent_path,
            default_stem=Path(file.filename or "imported.ipynb").stem,
        )
    source_filename = file.filename or "imported.ipynb"

    # Keeps the original filename for the import report and converter logs.
    upload_basename = Path(source_filename).name or "imported.ipynb"
    if not upload_basename.endswith(".ipynb"):
        upload_basename = f"{Path(upload_basename).stem or 'imported'}.ipynb"

    with tempfile.TemporaryDirectory(prefix="strata-import-") as tmp_dir:
        with timing.phase("write_temp"):
            tmp_path = Path(tmp_dir) / upload_basename
            tmp_path.write_bytes(payload)

        with timing.phase("convert"):
            try:
                result = import_notebook(
                    tmp_path,
                    out_dir=target_parent / raw_name,
                    owner=_caller_identity(request),
                )
            except (ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=f"Import failed: {exc}")

    # So the frontend can navigate to it immediately, as `create_new_notebook` does.
    with timing.phase("session_open"):
        session = _session_manager.open_notebook(
            result.notebook_dir,
            defer_initial_venv_sync=True,
            timing=timing,
        )

    with timing.phase("environment_job_submit"):
        try:
            await session.submit_environment_job(action="sync")
        except Exception as exc:
            logger.exception(
                "Failed to start initial environment bootstrap for imported %s",
                result.notebook_dir,
            )
            session.environment_sync_state = "failed"
            session.environment_sync_error = (
                f"Failed to start notebook environment initialization: {exc}"
            )
            session.environment_sync_notice = None

    with timing.phase("serialize"):
        data = session.serialize_notebook_state()
        data["session_id"] = session.id
        data["path"] = str(session.path)
        data.update(_serialize_notebook_runtime_config(request))
        data["import_report"] = {
            "markdown_cells": result.markdown_cells,
            "code_cells": result.code_cells,
            "suppressed_outputs": result.suppressed_outputs,
            "skipped_cells": result.skipped_cells,
            "translated_magics": result.translated_magics,
            "dropped_magics": result.dropped_magics,
            "dropped_shells": result.dropped_shells,
            "captured_deps": result.captured_deps,
            "warnings": result.warnings,
            "report_path": str(result.report_path) if result.report_path else None,
            "report_text": result.report_text,
        }

    return _timed_json_response(
        data,
        timing=timing,
        route_name="notebook_import",
        log_context=str(result.notebook_dir),
    )


# A snapshot with every artifact's bytes (`include=all`) is as large as the
# store, so it streams to disk under a much higher cap than .ipynb.
_MAX_SNAPSHOT_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@router.post("/import-snapshot")
async def import_snapshot_bundle(
    request: Request,
    file: UploadFile = File(..., description="A snapshot .zip exported with fmt=snapshot."),
    name: str | None = Form(
        default=None,
        description="Target notebook name. Defaults to the uploaded file's stem.",
    ),
    parent_path: str | None = Form(
        default=None,
        description=(
            "Where the new notebook directory lands. Must be inside the "
            "configured storage root. Defaults to the user's storage root."
        ),
    ),
) -> JSONResponse:
    """Turn an uploaded snapshot bundle into a notebook, and open it.

    The carried cells are cache hits before anything runs; cells whose
    artifacts the bundle described but did not carry open idle and are listed.
    A notebook id already in use under the caller's storage root is replaced,
    with every artifact id and lineage edge that embeds it rewritten to match.
    """
    from strata.notebook.snapshot_import import NotASnapshotError, import_snapshot

    timing = NotebookTimingRecorder()
    upload_name = file.filename or "snapshot.zip"
    stem = Path(upload_name).name.removesuffix(".zip").removesuffix(".snapshot")

    with timing.phase("validate"):
        _parent, _raw_name, candidate_dir = _resolve_import_target(
            request, name=name, parent_path=parent_path, default_stem=stem
        )

    with tempfile.TemporaryDirectory(prefix="strata-snapshot-") as tmp_dir:
        bundle_path = Path(tmp_dir) / "bundle.zip"
        with timing.phase("read_upload"):
            written = 0
            try:
                with open(bundle_path, "wb") as out:
                    while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                        written += len(chunk)
                        if written > _MAX_SNAPSHOT_UPLOAD_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    "Snapshot upload exceeds the "
                                    f"{_MAX_SNAPSHOT_UPLOAD_BYTES // (1024**3)} GiB cap"
                                ),
                            )
                        out.write(chunk)
            finally:
                await file.close()
        if written == 0:
            raise HTTPException(status_code=400, detail="Empty snapshot upload")

        with timing.phase("import"):
            # A copy sharing an id with this caller's notebooks collides once both
            # publish to a shared store. Scan the storage root, never the request's
            # parent_path: that is not where this caller's notebooks are listed.
            scan_root = _get_user_storage_root(request)
            taken = (
                {
                    entry["notebook_id"]
                    for entry in _discover_notebooks(scan_root)
                    if entry.get("notebook_id")
                }
                if scan_root is not None
                else set()
            )
            try:
                result = await asyncio.to_thread(
                    import_snapshot,
                    bundle_path,
                    candidate_dir,
                    taken_ids=taken,
                    owner=_caller_identity(request),
                )
            except zipfile.BadZipFile:
                raise HTTPException(status_code=400, detail="Upload is not a zip file")
            except NotASnapshotError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc))

    with timing.phase("session_open"):
        session = _session_manager.open_notebook(
            result.notebook_dir,
            defer_initial_venv_sync=True,
            timing=timing,
        )

    with timing.phase("environment_job_submit"):
        try:
            await session.submit_environment_job(action="sync")
        except Exception as exc:
            logger.exception(
                "Failed to start initial environment bootstrap for imported %s",
                result.notebook_dir,
            )
            session.environment_sync_state = "failed"
            session.environment_sync_error = (
                f"Failed to start notebook environment initialization: {exc}"
            )
            session.environment_sync_notice = None

    with timing.phase("serialize"):
        data = session.serialize_notebook_state()
        data["session_id"] = session.id
        data["path"] = str(session.path)
        data.update(_serialize_notebook_runtime_config(request))
        data["import_report"] = {
            "imported_artifacts": result.imported_artifacts,
            "replaced_notebook_id": result.replaced_id,
            "by_reference_cells": result.by_reference_cells,
        }

    return _timed_json_response(
        data,
        timing=timing,
        route_name="notebook_import_snapshot",
        log_context=str(result.notebook_dir),
    )


class QuiesceRequest(BaseModel):
    """Hold a notebook, or a project directory of them, still for a copy."""

    timeout_seconds: float = Field(
        default=30.0,
        ge=0,
        le=3600,
        description="How long to wait for running cells before cancelling them.",
    )
    max_hold_seconds: float = Field(
        default=600.0,
        gt=0,
        le=86400,
        description="The hold ends on its own after this, so a caller that dies cannot "
        "freeze the notebook.",
    )


def _require_notebook_admin() -> None:
    """Quiescing freezes other people's work, so it is an admin action.

    Under principal auth it needs ``admin:notebooks`` (``admin:*`` satisfies
    it). Personal mode has one operator, who is allowed.
    """
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        return
    if not state.config.principal_auth_enabled:
        return
    from strata.auth import get_principal

    principal = get_principal()
    if principal is None or not principal.has_scope("admin:notebooks"):
        raise HTTPException(
            status_code=403, detail="Insufficient scope: admin:notebooks required to quiesce"
        )


async def _quiesce(root: Path, sessions: list[NotebookSession], req: QuiesceRequest) -> dict:
    """Drain *sessions*, cancel what outlives the timeout, then hold *root*."""
    from strata.notebook import quiesce
    from strata.notebook.ws import cancel_notebook_execution

    try:
        hold = quiesce.begin(root, req.max_hold_seconds)
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc

    deadline = time.monotonic() + req.timeout_seconds
    while any(s._has_active_execution() for s in sessions) and time.monotonic() < deadline:
        await asyncio.sleep(0.05)
    cancelled: dict[str, list[str]] = {}
    for session in sessions:
        if session._has_active_execution():
            cancelled[str(session.path)] = await cancel_notebook_execution(session.id)
    # Runtime writes are synchronous, so once idle there is nothing to flush.
    quiesce.settle(hold)
    return {
        "path": str(hold.root),
        "notebooks": sorted(str(s.path) for s in sessions),
        "cancelled_cells": cancelled,
        "hold_expires_in_seconds": hold.seconds_left(),
    }


@router.post("/{notebook_id}/quiesce")
async def quiesce_notebook(
    notebook_id: str, session: SessionDep, req: QuiesceRequest | None = None
) -> dict:
    """Hold a notebook still until released, so a copy of it is consistent.

    Waits for running cells to finish (cancelling any still running at
    ``timeout_seconds`` and naming them), then refuses runs and edits with a
    409 until ``POST .../release`` or ``max_hold_seconds``.
    """
    _require_notebook_admin()
    return await _quiesce(session.path.resolve(), [session], req or QuiesceRequest())


@router.post("/{notebook_id}/release")
async def release_notebook(notebook_id: str, session: SessionDep) -> dict:
    """End a hold. ``released: false`` when there was none to end."""
    from strata.notebook import quiesce

    _require_notebook_admin()
    root = session.path.resolve()
    return {"path": str(root), "released": quiesce.release(root)}


projects_router = APIRouter(
    prefix="/v1/projects",
    tags=["notebooks"],
    dependencies=[Depends(_require_notebook_scope)],
)


def _project_root(path: str, request: Request) -> tuple[Path, list[NotebookSession]]:
    root = _validate_notebook_path(path, "project path", request)
    sessions = [
        session
        for session_id in _session_manager.list_sessions()
        if (session := _session_manager.get_session(session_id)) is not None
        and session.path.resolve().is_relative_to(root)
    ]
    return root, sessions


@projects_router.post("/{path:path}/quiesce")
async def quiesce_project(path: str, request: Request, req: QuiesceRequest | None = None) -> dict:
    """Hold every notebook under a directory still, for a project copy or move.

    Covers notebooks that are not open too: the hold is on the directory, so a
    notebook opened during it cannot run or be edited either.
    """
    _require_notebook_admin()
    root, sessions = _project_root(path, request)
    return await _quiesce(root, sessions, req or QuiesceRequest())


@projects_router.post("/{path:path}/release")
async def release_project(path: str, request: Request) -> dict:
    from strata.notebook import quiesce

    _require_notebook_admin()
    root, _sessions = _project_root(path, request)
    return {"path": str(root), "released": quiesce.release(root)}


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str, session: SessionDep) -> dict:
    """Delete a notebook directory and all notebook-owned runtime state."""
    _require_personal_mode_notebook_delete()

    if session.has_active_environment_mutation():
        _raise_environment_busy(
            session,
            "Notebook deletion is blocked while an environment update is in progress.",
        )

    if session._has_active_execution():
        raise HTTPException(
            status_code=409,
            detail="Notebook deletion is blocked while notebook execution is running.",
        )

    notebook_path = session.path.resolve()
    notebook_name = session.notebook_state.name

    _session_manager.close_session(session.id)

    try:
        delete_notebook_directory(notebook_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except Exception:
        logger.exception("Failed to delete notebook %s", notebook_path)
        raise HTTPException(status_code=500, detail="Failed to delete notebook")

    return {
        "deleted": True,
        "session_id": notebook_id,
        "name": notebook_name,
        "path": str(notebook_path),
    }


# Large dirs (node_modules, .venv) are noise. Hidden dirs are skipped
# wholesale; the notebook's own .strata is handled by the ignore below.
_DISCOVER_SKIP_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "dist",
        "build",
        "target",
        ".strata",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
    }
)


def _read_notebook_metadata(notebook_toml_path: Path) -> dict[str, Any] | None:
    """Cheaply read a notebook.toml's summary fields (name, id, updated_at, owner).

    Returns None if unreadable. Skips cells so discovery stays fast on large trees.
    """
    try:
        raw = tomllib.loads(notebook_toml_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    name = raw.get("name")
    notebook_id = raw.get("notebook_id")
    updated_at = raw.get("updated_at")
    owner = raw.get("owner")
    return {
        "name": str(name) if isinstance(name, str) and name.strip() else None,
        "notebook_id": str(notebook_id) if isinstance(notebook_id, str) else None,
        "updated_at": str(updated_at) if updated_at is not None else None,
        "owner": str(owner) if isinstance(owner, str) and owner.strip() else None,
    }


def _discover_notebooks(
    root: Path, *, max_depth: int = 4, max_results: int = 500
) -> list[dict[str, Any]]:
    """Walk ``root`` for directories containing ``notebook.toml``.

    Does not descend into a match (notebooks don't nest) or ``_DISCOVER_SKIP_DIRS``.
    ``max_depth`` and ``max_results`` keep a misconfigured root from stalling the
    server.
    """
    results: list[dict[str, Any]] = []
    if not root.exists() or not root.is_dir():
        return results

    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack and len(results) < max_results:
        current, depth = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue

        notebook_toml = current / "notebook.toml"
        if notebook_toml.is_file():
            metadata = _read_notebook_metadata(notebook_toml)
            if metadata is not None:
                results.append({"path": str(current.resolve()), **metadata})
            # Nested notebooks aren't supported and would create duplicate hits.
            continue

        if depth >= max_depth:
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith(".") or name in _DISCOVER_SKIP_DIRS:
                continue
            stack.append((entry, depth + 1))

    # Newest first; path sort keeps order stable for missing or equal timestamps.
    def sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        ts = entry.get("updated_at") or ""
        return (0 if ts else 1, ts or entry["path"])

    results.sort(key=sort_key, reverse=True)
    return results


@router.get("/discover")
async def discover_notebooks(request: Request) -> dict:
    """List notebook directories under the caller's storage root, newest first.

    Returns ``{"root", "notebooks"}``; each notebook is
    ``{path, name, notebook_id, updated_at, owner}``. With per-user scoping the scan
    root is the caller's subdir, and notebooks owned by others are filtered out too.
    """
    root = _get_user_storage_root(request)
    if root is None:
        return {"root": None, "notebooks": []}

    notebooks = _discover_notebooks(root)
    caller = _caller_identity(request)
    if caller is not None:
        notebooks = [n for n in notebooks if n.get("owner") in (None, caller)]
    return {"root": str(root), "notebooks": notebooks}


class ValidateRecentsRequest(BaseModel):
    """Request body for the recents-validation endpoint.

    The frontend's recents list lives in localStorage and outlives deleted notebooks.
    """

    paths: list[str] = Field(
        default_factory=list,
        description="Filesystem paths the client believes are notebooks",
        max_length=100,
    )


@router.post("/recents/validate")
async def validate_recent_notebooks(req: ValidateRecentsRequest) -> dict:
    """Return the subset of supplied paths that still contain a notebook.

    Existence check only (``<path>/notebook.toml`` is a file). No ownership or root
    check: the list is per-browser, and any follow-up open or delete runs its own.
    """
    valid: list[str] = []
    for raw_path in req.paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        try:
            if (Path(raw_path) / "notebook.toml").is_file():
                valid.append(raw_path)
        except OSError:
            # Permission errors, broken symlinks, etc.: treat as invalid.
            continue
    return {"valid": valid}


class DeleteNotebookByPathRequest(BaseModel):
    """Request for path-based notebook deletion (no session required)."""

    path: str = Field(..., description="Filesystem path of the notebook directory to delete")


@router.post("/delete-by-path")
async def delete_notebook_by_path(req: DeleteNotebookByPathRequest, request: Request) -> dict:
    """Delete a notebook directory by path, without an open session.

    An open session on the same directory is closed first.
    """
    _require_personal_mode_notebook_delete()

    notebook_path = _validate_notebook_path(req.path, "notebook path", request)
    notebook_toml_path = notebook_path / "notebook.toml"
    if not notebook_toml_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"No notebook found at {notebook_path}",
        )

    metadata = _read_notebook_metadata(notebook_toml_path)
    if metadata is not None:
        _require_owner(metadata.get("owner"), _caller_identity(request))

    existing = _session_manager._find_session_by_path(notebook_path)
    if existing is not None:
        if existing.has_active_environment_mutation():
            _raise_environment_busy(
                existing,
                "Notebook deletion is blocked while an environment update is in progress.",
            )
        if existing._has_active_execution():
            raise HTTPException(
                status_code=409,
                detail="Notebook deletion is blocked while notebook execution is running.",
            )
        _session_manager.close_session(existing.id)

    try:
        delete_notebook_directory(notebook_path.resolve())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except Exception:
        logger.exception("Failed to delete notebook by path %s", notebook_path)
        raise HTTPException(status_code=500, detail="Failed to delete notebook")

    return {"deleted": True, "path": str(notebook_path)}


@router.get("/{notebook_id}/environment")
async def get_environment_status(notebook_id: str, session: SessionDep) -> dict:
    """Get the live notebook environment status."""

    return _serialize_environment_payload(session)


@router.get("/{notebook_id}/r-packages")
async def get_r_packages(notebook_id: str, session: SessionDep) -> dict:
    """Return the R packages installed in the notebook's renv project library.

    Separate from ``GET /environment`` because the Rscript call takes ~1-2s.
    Returns ``{"packages": [{name, version}], "packages_status", "packages_error"}``;
    ``packages_status`` is ``ok``, ``absent`` (no ``renv.lock``), ``rscript_missing``,
    ``renv_not_active``, or ``failed`` (see ``packages_error``).
    """
    r_state = session.serialize_r_environment_state(include_packages=True)
    return {
        "packages": r_state["packages"],
        "packages_status": r_state["packages_status"],
        "packages_error": r_state["packages_error"],
    }


@router.get("/config")
async def get_notebook_runtime_config(request: Request) -> dict:
    """Return frontend runtime defaults for notebook creation/open flows."""
    return _serialize_notebook_runtime_config(request)


@router.post("/{notebook_id}/environment/sync")
async def sync_environment(notebook_id: str, session: SessionDep) -> dict:
    """Re-sync the notebook environment and invalidate stale runtimes."""

    try:
        session._begin_synchronous_environment_mutation("environment sync")
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    try:
        old_hash = session.serialize_environment_state()["lockfile_hash"]
        staleness_map = await session.sync_environment()
        new_hash = session.serialize_environment_state()["lockfile_hash"]
    finally:
        session._end_synchronous_environment_mutation()

    return {
        **_serialize_environment_payload(session),
        "lockfile_changed": old_hash != new_hash,
        **_serialize_environment_change(session, staleness_map),
        "operation_log": {
            "command": "uv sync",
            "duration_ms": session.environment_last_sync_duration_ms,
            "stdout": "",
            "stderr": session.environment_sync_error or "",
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
        "cells": session.serialize_cells(),
    }


@router.get("/{notebook_id}/environment/jobs/current")
async def get_current_environment_job(notebook_id: str, session: SessionDep) -> dict:
    """Return the currently active background environment job, if any."""
    return {
        **_serialize_environment_payload(session),
        "cells": session.serialize_cells(),
    }


@router.post("/{notebook_id}/environment/jobs")
async def submit_environment_job(
    notebook_id: str, session: SessionDep, req: EnvironmentJobRequest
) -> JSONResponse:
    """Submit a background notebook environment job."""

    if req.action in {"add", "remove"} and not req.package:
        raise HTTPException(
            status_code=400,
            detail=f"Package is required for {req.action} environment jobs",
        )
    if req.action == "sync" and (
        req.package is not None or req.requirements is not None or req.environment_yaml is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="Sync environment jobs do not accept package or import content",
        )
    if req.action in {"add", "remove"} and (
        req.requirements is not None or req.environment_yaml is not None
    ):
        raise HTTPException(
            status_code=400,
            detail=f"{req.action.title()} environment jobs do not accept import content",
        )
    if req.action == "import":
        provided_inputs = sum(
            value is not None for value in (req.requirements, req.environment_yaml)
        )
        if provided_inputs != 1 or req.package is not None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Import environment jobs require exactly one of requirements or "
                    "environment_yaml and do not accept a package"
                ),
            )
    if req.action == "r_init" and (
        req.package is not None or req.requirements is not None or req.environment_yaml is not None
    ):
        raise HTTPException(
            status_code=400,
            detail="r_init does not accept a package or import content",
        )
    if req.action == "r_add":
        if not req.package:
            raise HTTPException(
                status_code=400,
                detail="r_add requires a package name",
            )
        if req.requirements is not None or req.environment_yaml is not None:
            raise HTTPException(
                status_code=400,
                detail="r_add does not accept import content",
            )

    try:
        await session.submit_environment_job(
            action=req.action,
            package=req.package,
            requirements_text=req.requirements,
            environment_yaml_text=req.environment_yaml,
        )
    except ValueError as exc:
        # Rscript missing, renv not initialised, bad R package name: a 400 lets the
        # frontend show a targeted error.
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            **_serialize_environment_payload(session),
            "cells": session.serialize_cells(),
        },
    )


@router.get("/{notebook_id}/environment/requirements.txt")
async def export_environment_requirements(
    notebook_id: str, session: SessionDep
) -> PlainTextResponse:
    """Export direct notebook dependencies as ``requirements.txt`` text."""

    filename = f"{_safe_filename(session.notebook_state.name)}-requirements.txt"
    return PlainTextResponse(
        export_requirements_text(session.path),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{notebook_id}/environment/requirements.txt")
async def import_environment_requirements(
    notebook_id: str, session: SessionDep, req: ImportRequirementsRequest
) -> dict:
    """Replace direct notebook dependencies from ``requirements.txt`` text."""

    try:
        session._begin_synchronous_environment_mutation("requirements import")
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    try:
        try:
            outcome = await session.import_requirements(req.requirements)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    finally:
        session._end_synchronous_environment_mutation()

    result = outcome.result
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=_serialize_operation_error_detail(
                result.error or "Failed to import requirements.txt",
                result,
            ),
        )

    return {
        "success": True,
        "imported_count": result.imported_count,
        "lockfile_changed": result.lockfile_changed,
        **_serialize_result_operation_log(result),
        **_serialize_environment_payload(session),
        **_serialize_environment_change(session, outcome.staleness_map),
        "cells": session.serialize_cells(),
    }


@router.post("/{notebook_id}/environment/requirements.txt/preview")
async def preview_environment_requirements(
    notebook_id: str, session: SessionDep, req: PreviewRequirementsRequest
) -> dict:
    """Preview replacing direct notebook dependencies from ``requirements.txt`` text."""

    try:
        result = preview_requirements_text(session.path, req.requirements)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        **_serialize_import_preview(result),
        **_serialize_environment_payload(session),
    }


@router.post("/{notebook_id}/environment/environment.yaml")
async def import_environment_yaml(
    notebook_id: str, session: SessionDep, req: ImportEnvironmentYamlRequest
) -> dict:
    """Best-effort import of Conda-style ``environment.yaml`` text."""

    try:
        session._begin_synchronous_environment_mutation("environment.yaml import")
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    try:
        try:
            outcome = await session.import_environment_yaml(req.environment_yaml)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    finally:
        session._end_synchronous_environment_mutation()

    result = outcome.result
    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=_serialize_operation_error_detail(
                result.error or "Failed to import environment.yaml",
                result,
            ),
        )

    return {
        "success": True,
        "imported_count": result.imported_count,
        "warnings": result.warnings,
        "lockfile_changed": result.lockfile_changed,
        **_serialize_result_operation_log(result),
        **_serialize_environment_payload(session),
        **_serialize_environment_change(session, outcome.staleness_map),
        "cells": session.serialize_cells(),
    }


@router.post("/{notebook_id}/environment/environment.yaml/preview")
async def preview_environment_yaml(
    notebook_id: str, session: SessionDep, req: PreviewEnvironmentYamlRequest
) -> dict:
    """Preview best-effort import of Conda-style ``environment.yaml`` text."""

    try:
        result = preview_environment_yaml_text(session.path, req.environment_yaml)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        **_serialize_import_preview(result),
        **_serialize_environment_payload(session),
    }


@router.get("/sessions")
async def list_sessions(request: Request) -> dict:
    """List active notebook sessions visible to the calling user.

    With per-user scoping, only sessions under the caller's storage subdir are
    returned. Each entry has session_id, name, path and timestamps.
    """
    _require_personal_mode_session_api()
    user_root = _get_user_storage_root(request)
    base_root = _get_notebook_storage_root()
    # With scoping off, ``user_root`` equals ``base_root`` and this is a no-op.
    boundary = user_root if user_root is not None else base_root

    sessions = []
    for sid in _session_manager.list_sessions():
        session = _session_manager.get_session(sid)
        if session is None:
            continue
        if boundary is not None:
            try:
                resolved = session.path.resolve()
            except Exception:
                continue
            if resolved != boundary and boundary not in resolved.parents:
                continue
        sessions.append(
            {
                "session_id": session.id,
                "name": session.notebook_state.name,
                "path": str(session.path),
                "notebook_id": session.notebook_state.id,
                "created_at": session.notebook_state.created_at
                if hasattr(session.notebook_state, "created_at")
                else None,
                "updated_at": session.notebook_state.updated_at
                if hasattr(session.notebook_state, "updated_at")
                else None,
            }
        )
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, request: Request) -> JSONResponse:
    """Get full state for an existing session, to reconnect after a page refresh.

    Same shape as the ``open`` response.
    """
    timing = NotebookTimingRecorder()
    _require_personal_mode_session_api()
    with timing.phase("lookup"):
        session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # A leaked session_id must not let another user read this notebook; 404
    # hides that the session exists.
    _require_owner(session.notebook_state.owner, _caller_identity(request))

    with timing.phase("serialize"):
        data = session.serialize_notebook_state()
        data["session_id"] = session.id
        data["path"] = str(session.path)
        data["dag"] = _format_dag(session)
        data.update(_serialize_notebook_runtime_config(request))
    return _timed_json_response(
        data,
        timing=timing,
        route_name="notebook_get_session",
        log_context=session_id,
    )


async def _broadcast_state(notebook_id: str, session: Any) -> None:
    """Push a full ``notebook_state`` to WS spectators after a REST mutation.

    So an agent editing over the CLI / MCP shows up live in the TUI, not only on
    the next resync poll.
    """
    from strata.notebook.ws import broadcast_notebook_sync

    await broadcast_notebook_sync(notebook_id, session)


@router.put("/{notebook_id}/cells/reorder")
async def reorder_notebook_cells(
    notebook_id: str, session: SessionDep, req: ReorderCellsRequest
) -> dict:
    """Reorder cells in the notebook."""

    try:
        reorder_cells(session.path, req.cell_ids)
        session.reload()
        await _broadcast_state(notebook_id, session)
        return {
            "notebook_id": session.notebook_state.id,
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/cells")
async def list_cells(notebook_id: str, session: SessionDep) -> dict:
    """List cells in a notebook, with source."""

    return {
        "notebook_id": session.notebook_state.id,
        "cells": session.serialize_cells(),
    }


@router.put("/{notebook_id}/cells/{cell_id}")
async def update_cell_source(
    notebook_id: str, session: SessionDep, cell_id: str, req: UpdateCellSourceRequest
) -> dict:
    """Update cell source code; returns the updated cell state and DAG."""

    from strata.notebook.presence import lock_window_seconds
    from strata.notebook.ws import broadcast_presence

    author = resolve_author(req.author)
    held_by = session.presence.holder(cell_id, author, lock_window_seconds())
    if held_by is not None and not req.force:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "cell_locked",
                "cell_id": cell_id,
                "held_by": held_by,
                "message": f"{held_by} changed cell {cell_id} moments ago; retry in a "
                "few seconds, or send force to take it over",
            },
        )

    try:
        write_cell(session.path, cell_id, req.source, author=author)
        session.presence.record_edit(cell_id, author)
        session.presence.api_edit(author, cell_id)

        cell_in_session = session.notebook_state.get_cell(cell_id)
        if cell_in_session:
            cell_in_session.source = req.source
            # Or every read until the next reload reports the previous author.
            cell_in_session.updated_by = author

        session.re_analyze_cell(cell_id)

        # Without this, cells keep "ready" and the cascade planner won't trigger.
        await session.compute_staleness_async()

        cell = session.notebook_state.get_cell(cell_id)
        if not cell:
            raise HTTPException(status_code=404, detail="Cell not found")

        # All cells, so the frontend can sync staleness/status changes.
        await _broadcast_state(notebook_id, session)
        await broadcast_presence(notebook_id, session)
        return {
            "cell": session.serialize_cell(cell),
            "dag": _format_dag(session),
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/mounts")
async def update_notebook_mounts_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: MountConfigRequest,
) -> dict:
    """Replace notebook-level mount defaults."""

    try:
        update_notebook_mounts(session.path, req.mounts)
        session.reload()
        return {
            "mounts": [mount.model_dump() for mount in session.notebook_state.mounts],
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/connections")
async def list_notebook_connections(notebook_id: str, session: SessionDep) -> dict:
    """List the notebook's declared connections, shaped as in ``serialize_notebook_state``."""
    return {
        "connections": [conn.model_dump() for conn in session.notebook_state.connections],
    }


@router.put("/{notebook_id}/connections")
async def update_notebook_connections_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: ConnectionConfigRequest,
) -> dict:
    """Replace the notebook's ``[connections.<name>]`` blocks.

    The list is canonical: ``connections=[]`` deletes every connection. Auth
    literals are scrubbed on write, and the response reflects disk, so the UI sees
    the blanked secrets.
    """

    seen: set[str] = set()
    for conn in req.connections:
        if conn.name in seen:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate connection name {conn.name!r}",
            )
        seen.add(conn.name)

    try:
        # Keep [connections.<name>] blocks that failed to parse, so a typo in one
        # isn't erased by an unrelated edit.
        update_notebook_connections(
            session.path,
            req.connections,
            session.notebook_state.malformed_connections,
        )
        session.reload()
        return {
            "connections": [conn.model_dump() for conn in session.notebook_state.connections],
            "malformed_connections": [
                m.model_dump() for m in session.notebook_state.malformed_connections
            ],
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/connections/{name}/schema")
async def get_connection_schema(notebook_id: str, session: SessionDep, name: str) -> dict:
    """Enumerate tables (and columns) on a connection, opened read-only.

    Open and enumeration failures return 502 with the driver's message.
    """
    from strata.notebook.sql.cell_executor import (
        _resolve_runtime_spec,
        _safely_close,
    )
    from strata.notebook.sql.registry import get_adapter

    spec = next(
        (c for c in session.notebook_state.connections if c.name == name),
        None,
    )
    if spec is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown connection {name!r}",
        )
    try:
        adapter = get_adapter(spec.driver)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    runtime_spec = _resolve_runtime_spec(spec, session.path)
    try:
        conn = adapter.open(runtime_spec, read_only=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"failed to open connection {name!r}: {exc}",
        ) from exc

    try:
        tables = adapter.list_schema(conn)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"schema enumeration failed for {name!r}: {exc}",
        ) from exc
    finally:
        _safely_close(conn)

    return {
        "connection": name,
        "driver": spec.driver,
        "tables": [
            {
                "catalog": t.catalog,
                "schema": t.schema,
                "name": t.name,
                "columns": [
                    {"name": c.name, "type": c.type, "nullable": c.nullable} for c in t.columns
                ],
            }
            for t in tables
        ],
    }


@router.get("/{notebook_id}/workers")
async def list_notebook_workers(
    notebook_id: str, session: SessionDep, refresh: bool = False
) -> dict:
    """List the worker catalog visible to a notebook."""

    return await _serialize_worker_catalog(session, force_refresh=refresh)


@router.put("/{notebook_id}/workers")
async def update_notebook_workers_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: WorkersConfigRequest,
) -> dict:
    """Replace notebook-scoped worker definitions."""
    if not notebook_worker_definitions_editable(session.notebook_state):
        raise HTTPException(
            status_code=403,
            detail="Notebook worker definitions are managed by the server in service mode",
        )

    try:
        update_notebook_workers(session.path, req.workers)
        session.reload()
        return {
            "configured_workers": [
                worker.model_dump() for worker in session.notebook_state.workers
            ],
            **await _serialize_worker_catalog(session),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/workers/ssh")
async def list_ssh_workers(notebook_id: str, session: SessionDep) -> dict:
    """List the live SSH-tunnel status for this server's remote workers."""
    from dataclasses import asdict

    records = get_worker_supervisor().status()
    return {"tunnels": [asdict(record) for record in records]}


@router.post("/{notebook_id}/workers/ssh")
async def establish_ssh_worker_endpoint(
    notebook_id: str, session: SessionDep, req: SshWorkerRequest
) -> dict:
    """Provision a strata-worker over SSH, tunnel to it, and register it.

    Personal-mode only (worker definitions must be editable). The provisioning +
    tunnel run in a worker thread since SSH blocks.
    """
    import asyncio
    from dataclasses import asdict

    from strata.notebook.ops import NotebookOpsError
    from strata.notebook.ssh_worker import SshWorkerError
    from strata.notebook.ssh_worker_service import establish_ssh_worker

    supervisor = get_worker_supervisor()
    try:
        record = await asyncio.to_thread(
            establish_ssh_worker,
            session,
            supervisor,
            ssh_target=req.ssh_target,
            name=req.name,
            remote_port=req.remote_port,
            local_port=req.local_port,
            extras=req.extras,
            pin=req.pin,
            install=req.install,
            set_default=req.set_default,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except (SshWorkerError, NotebookOpsError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")
    return {"worker": asdict(record), **await _serialize_worker_catalog(session)}


@router.delete("/{notebook_id}/workers/ssh/{worker_name}")
async def teardown_ssh_worker_endpoint(
    notebook_id: str, session: SessionDep, worker_name: str, stop_remote: bool = False
) -> dict:
    """Close a worker's SSH tunnel and remove its notebook registration."""
    import asyncio

    from strata.notebook.ssh_worker_service import teardown_ssh_worker

    supervisor = get_worker_supervisor()
    try:
        existed = await asyncio.to_thread(
            teardown_ssh_worker, session, supervisor, worker_name, stop_remote=stop_remote
        )
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")
    return {"torn_down": existed, **await _serialize_worker_catalog(session)}


@router.put("/{notebook_id}/worker")
async def update_notebook_worker_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: WorkerConfigRequest,
) -> dict:
    """Replace the notebook-level default worker."""
    policy_error = validate_worker_assignment(session.notebook_state, req.worker)
    if policy_error is not None:
        raise HTTPException(status_code=403, detail=policy_error)

    try:
        update_notebook_worker(session.path, req.worker)
        session.reload()
        return {
            "worker": session.notebook_state.worker,
            **await _serialize_worker_catalog(session),
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


class UpdatePythonVersionRequest(BaseModel):
    """Body for ``PUT /v1/notebooks/{id}/python-version``."""

    python_version: str = Field(..., max_length=16)

    @field_validator("python_version")
    @classmethod
    def _validate_python_version_field(cls, value: str) -> str:
        try:
            return normalize_python_minor(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


@router.put("/{notebook_id}/python-version")
async def update_notebook_python_version(
    notebook_id: str,
    session: SessionDep,
    req: UpdatePythonVersionRequest,
    request: Request,
) -> JSONResponse:
    """Change the notebook's requested Python minor.

    Rewrites ``requires-python``, wipes ``.venv/`` and starts a background
    ``uv sync`` (rolled back if it fails); returns 202, with progress on the
    ``environment_job_progress`` WS stream. Returns 200 without touching disk when
    the version is unchanged.
    """

    runtime_config = _serialize_notebook_runtime_config(request)
    allowed = runtime_config["available_python_versions"]
    if req.python_version not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Python {req.python_version} is not available for this deployment "
                f"(allowed: {', '.join(allowed)})"
            ),
        )

    current = read_requested_python_minor(session.path)
    if current == req.python_version:
        # Idempotent: skips the disk write + uv sync.
        return JSONResponse(
            status_code=200,
            content={
                "accepted": False,
                "reason": "already_at_requested_version",
                **_serialize_environment_payload(session),
                "cells": session.serialize_cells(),
            },
        )

    try:
        await session.submit_environment_job(
            action="change_python",
            python_version=req.python_version,
        )
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            **_serialize_environment_payload(session),
            "cells": session.serialize_cells(),
        },
    )


@router.put("/{notebook_id}/timeout")
async def update_notebook_timeout_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: TimeoutConfigRequest,
) -> dict:
    """Replace the notebook-level default timeout."""

    try:
        update_notebook_timeout(session.path, req.timeout)
        session.reload()
        return {
            "timeout": session.notebook_state.timeout,
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/variant-groups/{group_id}/variants")
async def add_variant_endpoint(
    notebook_id: str,
    session: SessionDep,
    group_id: str,
) -> dict:
    """Add a sibling variant to an existing group.

    Clones the active variant's body under a fresh ``# @variant`` name and makes it
    active. Rename by editing the annotation.
    """

    try:
        new_name, new_cell_id = session.add_variant(group_id, author=resolve_author())
        return {
            "new_variant_name": new_name,
            "new_cell_id": new_cell_id,
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/variant-groups/{group_id}")
async def set_variant_active_endpoint(
    notebook_id: str,
    session: SessionDep,
    group_id: str,
    req: VariantActiveRequest,
) -> dict:
    """Switch the active variant for a group (the pointer in notebook.toml).

    An unknown name is accepted: validation reports ``variant_active_unknown`` and
    the DAG falls back to the first variant in source order.
    """

    if req.mode is None and not req.active:
        raise HTTPException(status_code=400, detail="Provide `active` and/or `mode`.")
    try:
        # Mode first: ``active`` is ignored in sweep mode anyway.
        if req.mode is not None:
            session.set_variant_mode(group_id, req.mode)
        if req.active:
            session.set_variant_active(group_id, req.active)
        return {
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/env")
async def update_notebook_env_endpoint(
    notebook_id: str,
    session: SessionDep,
    req: EnvConfigRequest,
) -> dict:
    """Replace the notebook-level default env vars."""
    from strata.notebook.secret_manager.session_integration import MANUAL_SOURCE

    try:
        # The panel sends every row back. An unchanged provider-fetched value is not
        # an edit: keep it out of the committed notebook.toml and keep its source.
        state = session.notebook_state
        previous_sources = dict(state.env_sources)
        fetched = {
            key
            for key, value in req.env.items()
            if previous_sources.get(key, MANUAL_SOURCE) != MANUAL_SOURCE
            and state.env.get(key) == value
        }
        # A fetched key already declared on disk stays declared, blank, so the file still
        # records which variables the notebook expects.
        with open(session.path / "notebook.toml", "rb") as f:
            declared = tomllib.load(f).get("env", {})
        to_write = {key: value for key, value in req.env.items() if key not in fetched}
        to_write.update({key: "" for key in fetched if key in declared})
        update_notebook_env(session.path, to_write)
        session.reload()
        # The disk writer blanks sensitive values to keep them out of git; restore
        # them in memory for the LLM config and Runtime panel. Edits are manual
        # overrides (for the UI badge).
        state = session.notebook_state
        for key, value in req.env.items():
            if key in fetched and key in state.env:
                continue  # the reload refetched it, possibly rotated
            state.env[key] = value
            state.env_sources[key] = previous_sources[key] if key in fetched else MANUAL_SOURCE
        # Rebuild each cell's env too: the executor reads cell.env, which still has
        # blanked values.
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


def _serialize_env_response(session) -> dict:
    """Env-endpoint response shape shared with the secret-manager refresh path."""
    return {
        "env": session.notebook_state.env,
        "env_sources": session.notebook_state.env_sources,
        "env_fetch_error": session.notebook_state.env_fetch_error,
        "env_fetched_at": session.notebook_state.env_fetched_at,
        "secret_manager_config": dict(session.notebook_state.secret_manager_config),
        "cells": session.serialize_cells(),
    }


class SecretManagerConfigRequest(BaseModel):
    """Payload for the secret-manager config PUT endpoint.

    Fields match what ``update_notebook_secret_manager`` accepts, so arbitrary
    runtime state cannot reach the committed TOML.
    """

    provider: str | None = None
    project_id: str | None = None
    environment: str | None = None
    path: str | None = None
    base_url: str | None = None


@router.put("/{notebook_id}/secret-manager/config")
async def update_notebook_secret_manager_config(
    notebook_id: str,
    session: SessionDep,
    req: SecretManagerConfigRequest,
) -> dict:
    """Persist the [secret_manager] block to notebook.toml and refetch.

    An empty payload removes the block (disconnects the secret manager).
    """
    from strata.notebook.writer import update_notebook_secret_manager

    try:
        config = req.model_dump(exclude_none=True)
        update_notebook_secret_manager(session.path, config)
        session.reload()
        # Runtime-panel keys are blanked on disk; restore in-memory values for keys
        # the fetch isn't replacing.
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/secret-manager/refresh")
async def refresh_notebook_secret_manager(notebook_id: str, session: SessionDep) -> dict:
    """Re-fetch secrets from the configured manager and merge into env.

    Same shape as the env endpoint. Never 500s on a fetch error; the message is in
    ``env_fetch_error``.
    """
    try:
        session.refresh_secrets()
        # So the executor picks up rotated values immediately.
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/cells")
async def add_cell(notebook_id: str, session: SessionDep, req: AddCellRequest) -> dict:
    """Add a new cell to the notebook and return its state."""

    # Before the try, so the 400 isn't masked as a 500 by the catch-all
    # (as in LocalNotebookOps.add_cell).
    if req.after_cell_id is not None and not any(
        c.id == req.after_cell_id for c in session.notebook_state.cells
    ):
        raise HTTPException(
            status_code=400, detail=f"after_cell_id {req.after_cell_id!r} not found"
        )

    try:
        cell_id = str(uuid.uuid4())[:8]

        add_cell_to_notebook(
            session.path,
            cell_id,
            req.after_cell_id,
            language=req.language,
            author=resolve_author(req.author),
        )

        session.reload()

        cell = session.notebook_state.get_cell(cell_id)
        if not cell:
            raise HTTPException(status_code=500, detail="Failed to create cell")

        await _broadcast_state(notebook_id, session)
        return session.serialize_cell(cell)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{notebook_id}/cells/{cell_id}")
async def delete_cell(notebook_id: str, session: SessionDep, cell_id: str) -> dict:
    """Delete a cell from the notebook."""

    try:
        if not any(c.id == cell_id for c in session.notebook_state.cells):
            raise HTTPException(status_code=404, detail="Cell not found")

        # For a variant member, remove_cell promotes the next variant (or dissolves
        # the group) so the toml pointer stays consistent.
        session.remove_cell(cell_id)
        await _broadcast_state(notebook_id, session)

        # Include ``dag``: deleting a variant member can promote a sibling and rewire
        # producers, and without it the DAG view shows stale edges.
        return {
            "message": "Cell deleted",
            "cell_id": cell_id,
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
            "dag": {
                "edges": session.dag.serialize_edges() if session.dag else [],
                "roots": list(session.dag.roots) if session.dag else [],
                "leaves": list(session.dag.leaves) if session.dag else [],
                "topological_order": session.dag.topological_order if session.dag else [],
                "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            },
        }
    except HTTPException:
        raise  # the 404 above is intentional — don't let the catch-all mask it as 500
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/name")
async def rename_notebook_endpoint(
    notebook_id: str, session: SessionDep, req: RenameNotebookRequest
) -> dict:
    """Rename the notebook and return its updated state."""

    try:
        rename_notebook(session.path, req.name)

        session.reload()

        return {
            "notebook_id": session.notebook_state.id,
            "name": session.notebook_state.name,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc) or "Forbidden")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except NotebookQuiesced as exc:
        raise _quiesced_conflict(exc) from exc
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/dag")
async def get_notebook_dag(notebook_id: str, session: SessionDep) -> dict:
    """Get the notebook's DAG: edges, topological order, leaves, roots, per-cell metadata."""

    return _format_dag(session)


@router.get("/{notebook_id}/artifacts")
async def list_notebook_published_artifacts(notebook_id: str, session: SessionDep) -> dict:
    """Per cell, the registry artifacts it published via the ambient ``strata`` client.

    These are ``put``/``materialize`` calls with ``name=``, stamped ``nb_cell=<id>``.
    Read from whichever store the cells write to: the team store when
    ``notebook_remote_store_url`` is set.
    """
    from strata.api.remote_registry import forward, remote_registry
    from strata.services.registry import registry_service

    # One lookup for the notebook: per cell would be a round trip each once
    # the store is remote.
    target = remote_registry()
    if target is not None:
        body = await forward(target, "GET", "/v1/registry/artifacts", params={"tag_key": "nb_cell"})
        published = body.get("artifacts", [])
    else:
        from strata.server import _get_artifact_store

        try:
            # A tenant-scoped read, so available in service mode too.
            store = _get_artifact_store(allow_read=True)
        except HTTPException:
            return {"cells": {}}
        published = registry_service.artifacts_by_tag(store, "nb_cell", tenant=None)

    by_cell: dict[str, list[dict]] = {}
    for item in published:
        by_cell.setdefault(str(item.pop("tag_value", "")), []).append(item)

    known = {cell.id for cell in session.notebook_state.cells}
    # A stamp from a cell no longer in the notebook has nowhere to show.
    return {"cells": {cell_id: items for cell_id, items in by_cell.items() if cell_id in known}}


@router.post("/{notebook_id}/artifacts/{artifact_id}/v/{version}/promote")
async def promote_notebook_artifact(
    notebook_id: str,
    session: SessionDep,
    artifact_id: str,
    version: int,
    request: PromoteArtifactRequest,
) -> dict:
    """Send one of this notebook's results to the team store, under a name.

    Copies the artifact and its whole chain from the notebook's private store into
    the shared one, so colleagues can fetch it by name and get team-cache hits on
    every step behind it. Under the ``promoted`` publish policy, this is the only
    way a result reaches the team.
    """
    from strata.artifact_transfer import RemoteStore, promote_artifact
    from strata.server import get_state

    config = get_state().config
    base_url = getattr(config, "notebook_remote_store_url", None)
    if not base_url:
        # A configuration answer, not a failure: name the missing setting rather
        # than a bare 500.
        raise HTTPException(
            status_code=409,
            detail=(
                "No team store is configured; set notebook_remote_store_url "
                "to the store colleagues read from."
            ),
        )

    manager = session.get_artifact_manager()
    store = manager.artifact_store
    artifact = store.get_artifact(artifact_id, version)
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"{artifact_id}@v={version} is not in this notebook's store"
        )

    from strata.auth import remote_store_headers

    target = RemoteStore(str(base_url), remote_store_headers(config))
    try:
        # Blocking HTTP calls to another machine; off the event loop so the
        # WebSocket keeps broadcasting.
        promotion = await asyncio.to_thread(
            promote_artifact,
            store,
            target,
            artifact,
            name=request.name,
            alias=request.alias,
            tags=dict(request.tags),
            table=request.table,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except (RuntimeError, httpx.HTTPError) as exc:
        # 502: the team store refused or was unreachable. What already copied stays;
        # it is keyed by provenance, so it is a usable cache entry.
        raise HTTPException(status_code=502, detail=f"The team store at {base_url}: {exc}")

    return {
        "name": promotion.name,
        "artifact_uri": f"strata://artifact/{promotion.ref}",
        "copied": promotion.copied,
        "alias": promotion.alias,
        "alias_pending": promotion.alias_pending,
        "table": promotion.table,
        "table_snapshot": promotion.table_snapshot,
        "store": str(base_url),
    }


@router.get("/{notebook_id}/cells/{cell_id}/iterations")
async def get_cell_iterations(
    notebook_id: str,
    session: SessionDep,
    cell_id: str,
    variable: str | None = None,
) -> dict:
    """List stored iteration artifacts for a loop cell.

    ``variable`` defaults to the ``@loop`` carry. Non-loop cells and loops with no
    completed iterations return an empty list.
    """
    from strata.notebook.annotations import parse_annotations

    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise HTTPException(status_code=404, detail="Cell not found")

    variable_name = variable
    if variable_name is None:
        loop_annotation = parse_annotations(cell.source).loop
        variable_name = loop_annotation.carry if loop_annotation else None
    if not variable_name:
        return {"cell_id": cell_id, "variable": None, "iterations": []}

    artifact_mgr = session.get_artifact_manager()
    iterations_payload: list[dict] = []
    for iteration, artifact in artifact_mgr.list_iterations(cell_id, variable_name):
        content_type = "unknown"
        if artifact.transform_spec:
            try:
                spec = json.loads(artifact.transform_spec)
                content_type = spec.get("params", {}).get("content_type") or "unknown"
            except (ValueError, KeyError):
                pass
        iterations_payload.append(
            {
                "iteration": iteration,
                "artifact_uri": (f"strata://artifact/{artifact.id}@v={artifact.version}"),
                "artifact_id": artifact.id,
                "version": artifact.version,
                "content_type": content_type,
                "byte_size": artifact.byte_size or 0,
                "row_count": artifact.row_count,
                "created_at": artifact.created_at,
            }
        )

    return {
        "cell_id": cell_id,
        "variable": variable_name,
        "iterations": iterations_payload,
    }


def _load_cell_data_blob(session, cell_id: str, artifact_uri: str) -> bytes:
    """Resolve ``strata://artifact/{id}@v={version}`` to its blob for a cell.

    Raises ``HTTPException`` for an unknown cell, a malformed URI, or a missing
    artifact.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise HTTPException(status_code=404, detail="Cell not found")
    prefix = "strata://artifact/"
    if not artifact_uri.startswith(prefix) or "@v=" not in artifact_uri:
        raise HTTPException(status_code=400, detail="Malformed artifact_uri")
    artifact_id, _, version_str = artifact_uri[len(prefix) :].rpartition("@v=")
    try:
        version = int(version_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed artifact_uri version") from None
    try:
        return session.get_artifact_manager().load_artifact_data(artifact_id, version)
    except ValueError:
        raise HTTPException(status_code=404, detail="Artifact not found") from None


def _parse_data_filters(filters: str | None) -> list[dict] | None:
    """Parse the ``filters`` query param (a JSON array of predicates)."""
    if not filters:
        return None
    try:
        parsed = json.loads(filters)
    except ValueError:
        raise HTTPException(status_code=400, detail="filters must be valid JSON") from None
    if not isinstance(parsed, list):
        raise HTTPException(status_code=400, detail="filters must be a JSON array")
    return parsed


@router.get("/{notebook_id}/cells/{cell_id}/data")
async def get_cell_data_page(
    notebook_id: str,
    session: SessionDep,
    cell_id: str,
    artifact_uri: str,
    offset: int = 0,
    limit: int = 100,
    sort_by: str | None = None,
    sort_dir: str = "asc",
    search: str | None = None,
    filters: str | None = None,
) -> dict:
    """Return a page of a cell output's cached DataFrame, with search, filters and sort.

    Reads the full Arrow artifact at ``artifact_uri`` (the inline preview stops at
    20 rows). ``filters`` is a JSON array of ``{col, op, value, value2}``. Non-table
    outputs return ``pageable: False``.
    """
    from strata.notebook.serializer import read_table_page

    if sort_dir not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="sort_dir must be 'asc' or 'desc'")
    parsed_filters = _parse_data_filters(filters)
    limit = max(1, min(limit, 500))
    offset = max(0, offset)

    blob = _load_cell_data_blob(session, cell_id, artifact_uri)
    page = read_table_page(
        blob,
        offset=offset,
        limit=limit,
        sort_by=sort_by,
        sort_dir=sort_dir,
        search=search,
        filters=parsed_filters,
    )
    if page is None:
        return {
            "cell_id": cell_id,
            "artifact_uri": artifact_uri,
            "pageable": False,
            "offset": offset,
            "limit": limit,
        }

    return {
        "cell_id": cell_id,
        "artifact_uri": artifact_uri,
        "pageable": True,
        "columns": page["columns"],
        "rows": page["rows"],
        "total": page["total"],
        "offset": offset,
        "limit": limit,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
    }


@router.get("/{notebook_id}/cells/{cell_id}/data/summary")
async def get_cell_data_summary(
    notebook_id: str,
    session: SessionDep,
    cell_id: str,
    artifact_uri: str,
) -> dict:
    """Per-column summary (dtype, nulls, distinct, min/max) of a cell's cached DataFrame."""
    from strata.notebook.serializer import read_table_summary

    blob = _load_cell_data_blob(session, cell_id, artifact_uri)
    summary = read_table_summary(blob)
    if summary is None:
        return {"cell_id": cell_id, "artifact_uri": artifact_uri, "pageable": False}
    return {
        "cell_id": cell_id,
        "artifact_uri": artifact_uri,
        "pageable": True,
        "columns": summary["columns"],
        "total": summary["total"],
    }


@router.get("/{notebook_id}/cells/{cell_id}/data/export")
async def export_cell_data(
    notebook_id: str,
    session: SessionDep,
    cell_id: str,
    artifact_uri: str,
    fmt: str = "csv",
    sort_by: str | None = None,
    sort_dir: str = "asc",
    search: str | None = None,
    filters: str | None = None,
):
    """Download a cell's cached DataFrame as CSV or Parquet.

    The same filters/search/sort as the on-screen view are applied so the
    download matches what the user sees.
    """
    from fastapi import Response

    from strata.notebook.serializer import write_table_export

    if fmt not in ("csv", "parquet"):
        raise HTTPException(status_code=400, detail="fmt must be 'csv' or 'parquet'")
    if sort_dir not in ("asc", "desc"):
        raise HTTPException(status_code=400, detail="sort_dir must be 'asc' or 'desc'")
    parsed_filters = _parse_data_filters(filters)

    blob = _load_cell_data_blob(session, cell_id, artifact_uri)
    data = write_table_export(
        blob,
        fmt,
        sort_by=sort_by,
        sort_dir=sort_dir,
        search=search,
        filters=parsed_filters,
    )
    if data is None:
        raise HTTPException(status_code=400, detail="Output is not an exportable table")

    media_type = "text/csv" if fmt == "csv" else "application/vnd.apache.parquet"
    filename = f"{cell_id}.{fmt}"
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{notebook_id}/cells/{cell_id}/outputs/{index}/blob")
async def get_cell_output_blob(
    notebook_id: str,
    session: SessionDep,
    cell_id: str,
    index: int,
):
    """Return one display output's stored bytes, under its content type when Strata renders it.

    The frontend payload carries a base64 data URL; this serves the raw file for
    clients that want one.
    """
    from fastapi import Response

    from strata.api.served_bytes import served_media_type
    from strata.notebook.ops import NotebookOpsError, display_output_at

    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise HTTPException(status_code=404, detail=f"no cell with id {cell_id!r}")
    try:
        output, resolved = display_output_at(cell, index)
    except NotebookOpsError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    try:
        blob = session.read_display_blob(output)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    media_type, headers = served_media_type(output.content_type or "")
    return Response(
        content=blob,
        media_type=media_type,
        headers={**headers, "X-Strata-Output-Index": str(resolved)},
    )


@router.get("/{notebook_id}/dependencies")
async def get_dependencies(notebook_id: str, session: SessionDep) -> dict:
    """List current dependencies from the notebook's pyproject.toml."""

    return _serialize_environment_payload(session)


@router.post("/{notebook_id}/dependencies")
async def add_notebook_dependency(
    notebook_id: str, session: SessionDep, req: AddDependencyRequest
) -> dict:
    """Add a dependency to the notebook.

    Runs ``uv add``, updates pyproject.toml + uv.lock, syncs venv.
    If the lockfile changes, the session's venv_python is re-synced.
    """

    try:
        session._begin_synchronous_environment_mutation(f"add {req.package}")
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    try:
        outcome = await session.mutate_dependency(req.package, action="add")
    finally:
        session._end_synchronous_environment_mutation()
    result = outcome.result

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=_serialize_operation_error_detail(
                result.error or "Failed to add dependency",
                result,
            ),
        )

    return {
        "success": True,
        "package": result.package,
        "lockfile_changed": result.lockfile_changed,
        **_serialize_result_operation_log(result),
        **_serialize_environment_payload(session),
        **_serialize_environment_change(session, outcome.staleness_map),
        "cells": session.serialize_cells(),
    }


@router.delete("/{notebook_id}/dependencies/{package_name}")
async def remove_notebook_dependency(
    notebook_id: str, session: SessionDep, package_name: str
) -> dict:
    """Remove a dependency from the notebook.

    Runs ``uv remove``, updates pyproject.toml + uv.lock, syncs venv.
    """

    try:
        package_name = validate_package_name(package_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        session._begin_synchronous_environment_mutation(f"remove {package_name}")
    except RuntimeError as exc:
        _raise_environment_busy(session, str(exc))

    try:
        outcome = await session.mutate_dependency(package_name, action="remove")
    finally:
        session._end_synchronous_environment_mutation()
    result = outcome.result

    if not result.success:
        raise HTTPException(
            status_code=400,
            detail=_serialize_operation_error_detail(
                result.error or "Failed to remove dependency",
                result,
            ),
        )

    return {
        "success": True,
        "package": result.package,
        "lockfile_changed": result.lockfile_changed,
        **_serialize_result_operation_log(result),
        **_serialize_environment_payload(session),
        **_serialize_environment_change(session, outcome.staleness_map),
        "cells": session.serialize_cells(),
    }


def _format_dag(session) -> dict:
    """Format the DAG for an API response."""
    from strata.notebook.dag import producer_cell_label

    if not session.dag:
        # An unbuildable graph must not be reported as an empty one.
        return {
            "edges": [],
            "topological_order": [],
            "leaves": [],
            "roots": [],
            "variable_producer": {},
            "error": getattr(session, "dag_error", None) or "notebook DAG could not be built",
        }

    return {
        "edges": [
            {
                "from_cell_id": edge.from_cell_id,
                "to_cell_id": edge.to_cell_id,
                "variable": edge.variable,
            }
            for edge in session.dag.edges
        ],
        "topological_order": session.dag.topological_order,
        "leaves": list(session.dag.leaves),
        "roots": list(session.dag.roots),
        "variable_producer": {
            v: producer_cell_label(p) for v, p in session.dag.variable_producer.items()
        },
    }


@router.post("/{notebook_id}/cells/{cell_id}/execute")
async def execute_cell(
    notebook_id: str, session: SessionDep, cell_id: str, mode: str = "normal"
) -> dict:
    """Execute a cell and return its outputs and stdout/stderr.

    ``mode`` mirrors the WS run modes: ``normal`` (cache on, cascade stale
    upstreams), ``rerun`` (bypass the target's cache, still cascade), or ``force``
    (run against existing upstream artifacts).
    """

    if mode not in ("normal", "rerun", "force"):
        raise HTTPException(status_code=400, detail=f"unknown run mode {mode!r}")

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        raise HTTPException(
            status_code=409,
            detail={
                "message": environment_block_reason,
                "code": "ENVIRONMENT_BUSY",
                "environment_job": session.serialize_environment_job_state(),
            },
        )

    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")

    # The shared execute path, so REST/CLI/MCP runs broadcast the same frames to
    # WS spectators. The exclusive wrapper takes the WS handlers' reservation, so
    # a REST run can't race a browser Run click on the same session.
    from strata.notebook.ws import NotebookBusyError, execute_cell_exclusive

    try:
        result = await execute_cell_exclusive(
            session,
            cell_id,
            notebook_id,
            mode=mode,  # narrowed by the validation above
        )
    except NotebookBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        session.mark_cell_error(cell_id)
        logger.exception("Cell execution failed")
        raise HTTPException(status_code=500, detail="Execution failed")
    if result is None:
        raise HTTPException(status_code=500, detail="Execution failed")
    return result.to_dict()


@router.put("/{notebook_id}/cells/{cell_id}/tests")
async def set_cell_tests_endpoint(
    notebook_id: str, session: SessionDep, cell_id: str, req: UpdateCellTestsRequest
) -> dict:
    """Set a Python cell's unit-test source (the committed ``cells/{id}.test.py``).

    Writes the file, updates the in-memory session so the next run sees it, and
    mirrors the change to WS spectators.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")
    if cell.language != CellLanguage.PYTHON:
        raise HTTPException(
            status_code=400, detail="Cell tests are only supported for Python cells"
        )

    from strata.notebook.writer import write_cell_tests

    try:
        write_cell_tests(session.path, cell_id, req.source)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Cell not found")
    cell.test_source = req.source
    await _broadcast_state(notebook_id, session)
    return session.serialize_cell(cell)


@router.post("/{notebook_id}/cells/{cell_id}/tests")
async def run_cell_tests_endpoint(notebook_id: str, session: SessionDep, cell_id: str) -> dict:
    """Run a Python cell's unit tests and return the per-test outcomes.

    REST twin of the WS ``cell_run_tests`` message: pytest against a re-executed
    copy of the cell; the result is persisted so WS clients see it on next sync.
    """
    cell = session.notebook_state.get_cell(cell_id)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")
    if cell.language != CellLanguage.PYTHON:
        raise HTTPException(
            status_code=400, detail="Cell tests are only supported for Python cells"
        )
    if not cell.test_source.strip():
        raise HTTPException(status_code=400, detail=f"Cell {cell_id} has no tests")

    environment_block_reason = session.environment_execution_block_message()
    if environment_block_reason:
        raise HTTPException(
            status_code=409,
            detail={
                "message": environment_block_reason,
                "code": "ENVIRONMENT_BUSY",
                "environment_job": session.serialize_environment_job_state(),
            },
        )

    try:
        executor = CellExecutor(session, session.warm_pool)
        result = await executor.run_cell_tests(cell_id, cell.test_source)
    except Exception:
        logger.exception("Cell test run failed")
        raise HTTPException(status_code=500, detail="Test run failed")

    return {
        "cell_id": cell_id,
        "passed": result.passed,
        "failed": result.failed,
        "errored": result.errored,
        "skipped": result.skipped,
        "pytest_unavailable": result.pytest_unavailable,
        "ran_at": result.ran_at,
        "tests": [
            {
                "name": case.name,
                "nodeid": case.nodeid,
                "outcome": case.outcome,
                "message": case.message,
            }
            for case in result.tests
        ],
    }


def _render_notebook_export(
    session,
    *,
    fmt: str,
    include_inactive_variants: bool,
    app_view: bool = False,
):
    """Render a notebook to one markdown/HTML file, as ``strata export`` does.

    Served as ``Content-Disposition: attachment`` so the browser downloads it.
    """
    from fastapi.responses import Response

    from strata.notebook.export import ExportFormat, ExportOptions, export_notebook

    try:
        options = ExportOptions(
            output_format=ExportFormat(fmt),
            include_inactive_variants=bool(include_inactive_variants),
            app_view=bool(app_view),
        )
        body = export_notebook(session.path, options)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc) or "Not found")
    except Exception:
        logger.exception("render export failed for notebook %s", session.id)
        raise HTTPException(status_code=500, detail="Export failed")

    safe_name = session.path.name or "notebook"
    if app_view:
        safe_name = f"{safe_name}-app"
    extension = "html" if fmt == "html" else "md"
    media_type = "text/html; charset=utf-8" if fmt == "html" else "text/markdown; charset=utf-8"
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.{extension}"',
        },
    )


@router.get("/{notebook_id}/export")
async def export_notebook(
    notebook_id: str,
    session: SessionDep,
    fmt: str = "zip",
    include_inactive_variants: bool = False,
    app_view: bool = False,
    include: str = "selected",
    cells: str | None = None,
):
    """Export the notebook as ``zip`` (default), ``snapshot``, ``markdown`` or ``html``.

    ``zip``: notebook.toml, pyproject.toml, uv.lock (if present), cell sources and
    ``provenance.json`` (DAG and per-cell provenance hashes).

    ``snapshot``: the zip plus ``outputs/<cell id>/`` (display outputs and
    ``console.json``), per-cell provenance and timings, ``artifacts.json`` naming
    every ready cell's artifacts with digests, and ``artifacts/<id>@v=<n>`` for the
    bytes chosen by ``include``: ``all`` (a move between servers), ``selected``
    (cells named in ``cells``, the rest by reference), or ``none``.

    ``markdown`` / ``html``: a rendered single file without prompt cell responses;
    ``include_inactive_variants`` stacks every variant of each group.
    """
    if fmt not in {"zip", "markdown", "html", "snapshot"}:
        raise HTTPException(
            status_code=400,
            detail="fmt must be one of 'zip', 'markdown', 'html', 'snapshot'",
        )
    if include not in {"all", "selected", "none"}:
        raise HTTPException(
            status_code=400,
            detail="include must be one of 'all', 'selected', 'none'",
        )

    if fmt in {"markdown", "html"}:
        return _render_notebook_export(
            session,
            fmt=fmt,
            include_inactive_variants=bool(include_inactive_variants),
            app_view=bool(app_view),
        )

    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Same helper `strata export` uses, so both agree on the bundle contents.
        from strata.notebook.snapshot import write_committed_files

        write_committed_files(session, zf)

        selected_cells: list[str] = [c for c in (cells or "").split(",") if c]
        if fmt == "snapshot":
            # What the ZIP doesn't already carry: output files, per-cell provenance and
            # timings, the artifact index, and the requested bytes.
            from strata.notebook.snapshot import (
                unknown_selection,
                write_snapshot,
            )

            unknown = unknown_selection(session, selected_cells)
            if unknown:
                raise HTTPException(
                    status_code=400,
                    detail=f"No such cell(s) in this notebook: {', '.join(unknown)}",
                )
            write_snapshot(
                session,
                zf,
                # Narrowed by the check above, which names the allowed values; a Literal
                # query parameter would answer 422 in pydantic's phrasing.
                include=include,
                selected_cells=selected_cells,
            )

    buf.seek(0)
    suffix = "snapshot.zip" if fmt == "snapshot" else "zip"
    filename = f"{_safe_filename(session.notebook_state.name or 'notebook')}.{suffix}"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

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
from typing import TYPE_CHECKING, Any, Literal

import httpx
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from strata.notebook.dependencies import (
    export_requirements_text,
    list_dependencies,
    list_resolved_dependencies,
    preview_environment_yaml_text,
    preview_requirements_text,
)
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, ConnectionSpec, MountSpec, WorkerSpec
from strata.notebook.python_versions import current_python_minor, normalize_python_minor
from strata.notebook.session import SessionManager
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
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# Global session manager (shared with WebSocket handler)
_session_manager = SessionManager()

router = APIRouter(prefix="/v1/notebooks", tags=["notebooks"])


def get_session_manager() -> SessionManager:
    """Export session manager for WebSocket handler."""
    return _session_manager


def _require_personal_mode_session_api() -> None:
    """Restrict session discovery/reconnect APIs to personal mode.

    These endpoints expose in-memory session IDs and notebook filesystem paths.
    They are intended as a local UX helper for page refresh/reconnect, not as a
    multi-user service-mode surface.
    """
    try:
        from strata.server import get_state

        state = get_state()
    except RuntimeError:
        # Route unit tests may use the notebook router without a full server state.
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


# Allowed characters in a per-user storage directory name. Email-shaped
# inputs map cleanly: alphanumerics, ``.``, ``_``, ``-``, and ``@`` all
# survive. Anything else is collapsed to ``_`` so a hostile header value
# can't escape the storage root with ``../`` or null bytes.
_USER_DIR_SAFE_RE = re.compile(r"[^A-Za-z0-9._@\-]")


def _sanitize_user_dir_name(identity: str) -> str | None:
    """Map a caller identity (typically email) to a filesystem-safe dir name.

    Returns ``None`` for empty inputs or values that sanitize to empty,
    so the caller can fall back to "no per-user scoping" behavior. The
    transform is deterministic — same identity always produces the same
    directory — which keeps each user's notebooks pinned to one path.
    """
    if not identity:
        return None
    cleaned = _USER_DIR_SAFE_RE.sub("_", identity.strip())
    cleaned = cleaned.strip("._-") or ""
    return cleaned or None


def _get_user_storage_root(request: Request | None) -> Path | None:
    """Return the storage root scoped to the calling user.

    * No server state → ``None`` (caller should treat as no scoping).
    * No request, or no header configured, or no caller identity →
      base storage root, single-user behavior preserved.
    * Otherwise → ``<base>/<sanitized_identity>/``, auto-created on
      first call so the rest of the request flow doesn't need to
      worry about whether the user dir exists yet.

    The auto-creation also gives us a natural hook for future
    onboarding: seeding a template notebook into the new user's dir
    would happen here, immediately after ``mkdir``.
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
    # Defense-in-depth: even though sanitization should make escape
    # impossible, double-check the resolved path stays under the base.
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


def _caller_identity(request: Request) -> str | None:
    """Resolve the calling user's identity for personal-mode scoping.

    Personal-mode deployments fronted by an authenticating proxy
    (Cloudflare Access, Pomerium, etc.) inject the user's identity in a
    request header. ``personal_mode_user_header`` names which header to
    read; when it's unset the deployment behaves as single-user and this
    function returns ``None`` for every request — no scoping kicks in.

    Returning ``None`` is also the correct behavior when the header is
    configured but missing from a particular request (e.g. health
    checks, internal services). Endpoints that depend on identity for
    ACL decisions must treat ``None`` as "no identity available" — they
    can either fall back to global behavior or refuse the request,
    whichever matches the operation's safety profile.
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

    - Unowned notebooks (``notebook_owner is None``) remain accessible to any
      caller — they're either legacy notebooks created before scoping was
      enabled, or notebooks created by services that don't carry a header.
    - When ``personal_mode_user_header`` is unset (and so ``caller`` is None),
      every notebook acts as if unowned and access is allowed. This preserves
      single-user behavior on local deployments.
    - When the notebook is owned and the caller's identity differs, raise
      403. Use a generic "not found" body so probes can't enumerate owners.
    """
    if notebook_owner is None or caller is None:
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
    """Validate and sanitize a package specifier.

    Rejects shell metacharacters. Used by both REST and WS handlers.
    """
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

    When ``request`` is provided and per-user scoping is configured, the
    path must lie inside the **caller's** subdir, not just the base
    storage root. This is the security-critical boundary that prevents
    user A from passing user B's path in ``parent_path`` / ``notebook_path``
    fields and reaching B's notebooks.
    """
    path = Path(user_path)
    if ".." in path.parts:
        raise HTTPException(status_code=400, detail=f"Invalid {label}: path traversal not allowed")

    user_root = _get_user_storage_root(request)
    base_root = _get_notebook_storage_root()
    # Resolution base: per-user root when scoping is active, else base.
    resolution_root = user_root if user_root is not None else base_root
    resolved = (
        (resolution_root / path).resolve()
        if resolution_root is not None and not path.is_absolute()
        else path.resolve()
    )

    # Confine to the user's root (per-user scoping) or to the base root
    # (single-user mode). Returning the resolved path is safe because
    # we've verified it's contained.
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

    When ``request`` is supplied and per-user scoping is configured,
    ``default_parent_path`` is the caller's subdir so the frontend's
    "Create notebook" flow lands the new notebook under the right user.
    """
    deployment_mode = "service"
    default_parent_path = Path("/tmp/strata-notebooks")
    available_python_versions = [current_python_minor()]

    try:
        from strata.server import get_state

        state = get_state()
        deployment_mode = getattr(state.config, "deployment_mode", deployment_mode)
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
    }


def _serialize_dependency_info_list(dependencies: list) -> list[dict]:
    """Serialize dependency metadata for API responses."""
    return [
        {"name": dep.name, "version": dep.version, "specifier": dep.specifier}
        for dep in dependencies
    ]


def _serialize_environment_payload(session: NotebookSession) -> dict:
    """Serialize the current environment plus direct and resolved dependencies."""
    return {
        "environment": session.serialize_environment_state(),
        "environment_job": session.serialize_environment_job_state(),
        "environment_job_history": session.serialize_environment_job_history(),
        "dependencies": _serialize_dependency_info_list(list_dependencies(session.path)),
        "resolved_dependencies": _serialize_dependency_info_list(
            list_resolved_dependencies(session.path)
        ),
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


# ============================================================================
# Request/Response Models
# ============================================================================


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

    source: str = Field(..., max_length=1_000_000)  # 1MB limit


class AddCellRequest(BaseModel):
    """Request to add a new cell."""

    after_cell_id: str | None = None
    language: Literal["python", "prompt", "markdown", "sql"] = "python"


class MountConfigRequest(BaseModel):
    """Request to replace a mount list."""

    mounts: list[MountSpec] = Field(default_factory=list)


class ConnectionConfigRequest(BaseModel):
    """Request to replace the full ``[connections.<name>]`` set.

    The list is the canonical state — sending an empty list deletes
    every connection. The route handler scrubs auth literals at the
    serializer boundary, same as a hand-edited notebook.toml save,
    so a UI form that includes a literal secret keeps the key as a
    placeholder on disk while the running session keeps the value
    in memory until reload.
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


class TimeoutConfigRequest(BaseModel):
    """Request to replace a timeout setting."""

    timeout: float | None = Field(default=None, gt=0, le=86_400)


class VariantActiveRequest(BaseModel):
    """Request to switch the active variant in a group."""

    active: str = Field(..., pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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
        if normalized not in {"add", "remove", "sync", "import"}:
            raise ValueError("Unsupported environment job action")
        return normalized

    @field_validator("package")
    @classmethod
    def validate_package_field(cls, value: str | None) -> str | None:
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


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/open")
async def open_notebook(req: OpenNotebookRequest, request: Request) -> JSONResponse:
    """Open a notebook directory.

    Args:
        req: OpenNotebookRequest with path
        request: FastAPI request — used to resolve the calling identity
            for per-user path validation when scoping is enabled.

    Returns:
        Notebook state, session ID, and DAG as JSON
    """
    timing = NotebookTimingRecorder()

    try:
        with timing.phase("validate"):
            notebook_path = _validate_notebook_path(req.path, "notebook path", request)
            if not notebook_path.exists():
                raise HTTPException(status_code=404, detail="Notebook directory not found")

        # Create session (parses notebook and triggers DAG analysis)
        with timing.phase("session_open"):
            session = _session_manager.open_notebook(
                notebook_path,
                reuse_existing=_reuse_open_session_by_path(),
                timing=timing,
            )

        # Return notebook state with session ID and DAG
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
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/create")
async def create_new_notebook(req: CreateNotebookRequest, request: Request) -> JSONResponse:
    """Create a new notebook.

    Args:
        req: CreateNotebookRequest with parent_path and name
        request: FastAPI request — used to resolve the calling identity when
            personal-mode user-scoping is enabled.

    Returns:
        Notebook state as JSON
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

        # Check if a notebook already exists at this path
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
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


# Bounded upload size. Notebooks bigger than this are almost always
# packed with embedded image outputs; importing them would put real
# pressure on the server-side temp file. Limit conservatively.
_MAX_IPYNB_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


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
    """Convert an uploaded ``.ipynb`` into a Strata notebook directory.

    Wraps :func:`strata.notebook.jupyter_import.import_notebook` and
    opens a session on the result so the frontend can navigate to it
    immediately. The import report (sources, magic translation,
    captured deps, warnings) is returned inline so the caller doesn't
    need a second round-trip to fetch it.
    """
    from strata.notebook.jupyter_import import import_notebook

    timing = NotebookTimingRecorder()

    # Read the upload, bounded. Reading once into memory is safer than
    # streaming to disk and re-reading: we need to verify it's parseable
    # JSON before we touch the storage tree, and the size cap means this
    # never grows past ~50 MB.
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

    # Defensive parse before we materialize anything on disk — surfaces
    # malformed JSON as a clean 400, not a 500 from deep inside the
    # converter.
    try:
        json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid .ipynb JSON: {exc}")

    # Resolve target storage root. parent_path override goes through the
    # same path-traversal guard as the other endpoints.
    with timing.phase("validate"):
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

    # Pick the notebook name. Use the upload's filename stem unless the
    # caller overrode it — same slugify rule as `create_notebook` so the
    # resulting directory layout matches anything the user creates
    # through the regular UI.
    source_filename = file.filename or "imported.ipynb"
    raw_name = name or Path(source_filename).stem or "imported"

    # Reject path separators / traversal segments / NUL bytes — the
    # name flows into a filesystem path, and `target_parent / "../x"`
    # would otherwise escape the storage root entirely.
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
    # Belt-and-braces: after building the path, verify it still resolves
    # under target_parent. Catches any escape route the textual checks
    # above might miss (e.g. platform-specific weirdness).
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

    # Persist the upload to a tempdir with the user's original filename
    # — preserves the filename in the generated import report and in
    # any path logging the converter does. The tempdir + its contents
    # are unlinked once the import finishes.
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

    # Open a session on the newly-imported notebook so the frontend can
    # navigate to it immediately. Mirrors `create_new_notebook`'s flow.
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


@router.delete("/{notebook_id}")
async def delete_notebook(notebook_id: str, request: Request) -> dict:
    """Delete a notebook directory and all notebook-owned runtime state."""
    _require_personal_mode_notebook_delete()

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    _require_owner(session.notebook_state.owner, _caller_identity(request))

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
    except Exception:
        logger.exception("Failed to delete notebook %s", notebook_path)
        raise HTTPException(status_code=500, detail="Failed to delete notebook")

    return {
        "deleted": True,
        "session_id": notebook_id,
        "name": notebook_name,
        "path": str(notebook_path),
    }


# Names to skip while recursing into notebook_storage_dir. Large dirs
# (node_modules, .venv) are noise; hidden dirs (starting with .) are
# skipped wholesale except for the notebook's own .strata directory
# which is handled by the ignore below rather than a name match.
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

    Returns None if the file is unreadable; intentionally does not parse
    cells — discovery should stay fast even on a directory with hundreds
    of notebooks.
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
    """Walk ``root`` looking for directories containing ``notebook.toml``.

    Stops descending into any matched directory (notebooks don't nest) or
    any name in ``_DISCOVER_SKIP_DIRS``. Bounded by ``max_depth`` and
    ``max_results`` so a misconfigured storage root can't stall the
    server scanning a huge tree.
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
            # Don't descend into a notebook directory — nested notebooks
            # aren't a supported layout and would create duplicate hits.
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

    # Newest first when updated_at is present; fall back to path sort so
    # ordering is stable when timestamps are missing or equal.
    def sort_key(entry: dict[str, Any]) -> tuple[int, str]:
        ts = entry.get("updated_at") or ""
        return (0 if ts else 1, ts or entry["path"])

    results.sort(key=sort_key, reverse=True)
    return results


@router.get("/discover")
async def discover_notebooks(request: Request) -> dict:
    """List notebook directories found under the caller's storage root.

    Used by the "Open existing" UI so users pick from a list instead of
    typing a filesystem path. Returns ``{"root", "notebooks"}`` where
    ``root`` is the scan root (for display) and ``notebooks`` is a
    ``[{path, name, notebook_id, updated_at, owner}]`` list sorted newest first.

    When ``personal_mode_user_header`` is configured the scan root is the
    caller's per-user subdir, so users physically cannot see each other's
    notebooks. The owner-field filter is kept as defense-in-depth in case
    a notebook ever lands outside the user dir (it shouldn't — the path
    validator rejects that case at the boundary).
    """
    root = _get_user_storage_root(request)
    if root is None:
        return {"root": None, "notebooks": []}

    notebooks = _discover_notebooks(root)
    caller = _caller_identity(request)
    if caller is not None:
        notebooks = [n for n in notebooks if n.get("owner") in (None, caller)]
    return {"root": str(root), "notebooks": notebooks}


class DeleteNotebookByPathRequest(BaseModel):
    """Request for path-based notebook deletion (no session required)."""

    path: str = Field(..., description="Filesystem path of the notebook directory to delete")


@router.post("/delete-by-path")
async def delete_notebook_by_path(req: DeleteNotebookByPathRequest, request: Request) -> dict:
    """Delete a notebook directory identified by path.

    Unlike ``DELETE /{notebook_id}`` this does not require the notebook to
    be open in a session — it's for deleting notebooks from the home page
    list. If a session happens to be open against the same directory it
    is closed first so the subsequent ``rmtree`` is safe.
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

    # If a session happens to be open for this path, close it first.
    # _find_session_by_path takes care of path canonicalization.
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
    except Exception:
        logger.exception("Failed to delete notebook by path %s", notebook_path)
        raise HTTPException(status_code=500, detail="Failed to delete notebook")

    return {"deleted": True, "path": str(notebook_path)}


@router.get("/{notebook_id}/environment")
async def get_environment_status(notebook_id: str) -> dict:
    """Get the live notebook environment status."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return _serialize_environment_payload(session)


@router.get("/config")
async def get_notebook_runtime_config(request: Request) -> dict:
    """Return frontend runtime defaults for notebook creation/open flows."""
    return _serialize_notebook_runtime_config(request)


@router.post("/{notebook_id}/environment/sync")
async def sync_environment(notebook_id: str) -> dict:
    """Re-sync the notebook environment and invalidate stale runtimes."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
async def get_current_environment_job(notebook_id: str) -> dict:
    """Return the currently active background environment job, if any."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {
        **_serialize_environment_payload(session),
        "cells": session.serialize_cells(),
    }


@router.post("/{notebook_id}/environment/jobs")
async def submit_environment_job(notebook_id: str, req: EnvironmentJobRequest) -> JSONResponse:
    """Submit a background notebook environment job."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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

    try:
        await session.submit_environment_job(
            action=req.action,
            package=req.package,
            requirements_text=req.requirements,
            environment_yaml_text=req.environment_yaml,
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


@router.get("/{notebook_id}/environment/requirements.txt")
async def export_environment_requirements(notebook_id: str) -> PlainTextResponse:
    """Export direct notebook dependencies as ``requirements.txt`` text."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    filename = f"{_safe_filename(session.notebook_state.name)}-requirements.txt"
    return PlainTextResponse(
        export_requirements_text(session.path),
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{notebook_id}/environment/requirements.txt")
async def import_environment_requirements(notebook_id: str, req: ImportRequirementsRequest) -> dict:
    """Replace direct notebook dependencies from ``requirements.txt`` text."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
    notebook_id: str, req: PreviewRequirementsRequest
) -> dict:
    """Preview replacing direct notebook dependencies from ``requirements.txt`` text."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        result = preview_requirements_text(session.path, req.requirements)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        **_serialize_import_preview(result),
        **_serialize_environment_payload(session),
    }


@router.post("/{notebook_id}/environment/environment.yaml")
async def import_environment_yaml(notebook_id: str, req: ImportEnvironmentYamlRequest) -> dict:
    """Best-effort import of Conda-style ``environment.yaml`` text."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
async def preview_environment_yaml(notebook_id: str, req: PreviewEnvironmentYamlRequest) -> dict:
    """Preview best-effort import of Conda-style ``environment.yaml`` text."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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

    With per-user scoping enabled, only sessions whose notebook lives
    under the caller's storage subdir are returned — open sessions for
    other users stay invisible. Without scoping, all sessions are
    listed (single-user behavior).

    Returns:
        Dictionary with a ``sessions`` array, each entry containing
        session_id, notebook name, filesystem path, and timestamps.
    """
    _require_personal_mode_session_api()
    user_root = _get_user_storage_root(request)
    base_root = _get_notebook_storage_root()
    # When scoping is on, ``user_root`` is the per-user subdir. When
    # scoping is off, ``user_root`` equals ``base_root`` and the
    # containment check is a no-op (every session passes).
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
    """Get full state for an existing session.

    This allows the frontend to reconnect to a session after a page
    refresh without re-opening the notebook from disk.

    Args:
        session_id: UUID session identifier
        request: FastAPI request — used for per-user runtime config.

    Returns:
        Notebook state, session ID, and DAG as JSON (same shape as
        the ``open`` endpoint response).
    """
    timing = NotebookTimingRecorder()
    _require_personal_mode_session_api()
    with timing.phase("lookup"):
        session = _session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Reject cross-user reconnects: a session_id leaked to user B should
    # not let them read user A's notebook state. We hide the existence
    # of the session entirely with a 404.
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


@router.put("/{notebook_id}/cells/reorder")
async def reorder_notebook_cells(notebook_id: str, req: ReorderCellsRequest) -> dict:
    """Reorder cells in the notebook."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        reorder_cells(session.path, req.cell_ids)
        session.reload()
        return {
            "notebook_id": session.notebook_state.id,
            "cells": session.serialize_cells(),
        }
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/cells")
async def list_cells(notebook_id: str) -> dict:
    """List cells in a notebook.

    Args:
        notebook_id: Notebook/session ID

    Returns:
        List of cells with source
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return {
        "notebook_id": session.notebook_state.id,
        "cells": session.serialize_cells(),
    }


@router.put("/{notebook_id}/cells/{cell_id}")
async def update_cell_source(notebook_id: str, cell_id: str, req: UpdateCellSourceRequest) -> dict:
    """Update cell source code.

    Args:
        notebook_id: Notebook/session ID
        cell_id: Cell ID
        req: UpdateCellSourceRequest with source

    Returns:
        Updated cell state and DAG
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        # Write to disk
        write_cell(session.path, cell_id, req.source)

        # Update source in session
        cell_in_session = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
        if cell_in_session:
            cell_in_session.source = req.source

        # Re-analyze just this cell and rebuild DAG
        session.re_analyze_cell(cell_id)

        # Recompute staleness so cell statuses reflect the edit.
        # Without this, cells keep their old "ready" status and the
        # cascade planner won't trigger when the user runs a
        # downstream cell.
        session.compute_staleness()

        # Find and return the updated cell with DAG info
        cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
        if not cell:
            raise HTTPException(status_code=404, detail="Cell not found")

        # Return cell and updated DAG — include all cells so the
        # frontend can sync staleness/status changes.
        return {
            "cell": session.serialize_cell(cell),
            "dag": _format_dag(session),
            "cells": session.serialize_cells(),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/mounts")
async def update_notebook_mounts_endpoint(
    notebook_id: str,
    req: MountConfigRequest,
) -> dict:
    """Replace notebook-level mount defaults."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        update_notebook_mounts(session.path, req.mounts)
        session.reload()
        return {
            "mounts": [mount.model_dump() for mount in session.notebook_state.mounts],
            "cells": session.serialize_cells(),
        }
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/connections")
async def list_notebook_connections(notebook_id: str) -> dict:
    """List the notebook's declared connections.

    Returns the same shape used by ``serialize_notebook_state`` so the
    UI can stay in sync with what the parser already exposes.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return {
        "connections": [conn.model_dump() for conn in session.notebook_state.connections],
    }


@router.put("/{notebook_id}/connections")
async def update_notebook_connections_endpoint(
    notebook_id: str,
    req: ConnectionConfigRequest,
) -> dict:
    """Replace the notebook's ``[connections.<name>]`` blocks.

    The whole list is canonical — sending ``connections=[]`` deletes
    every connection. Auth literals are scrubbed at write time; the
    response reflects the on-disk state after the round-trip so the
    UI immediately sees blanked secrets and can prompt the user to
    move them to ``${VAR}`` indirections.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    seen: set[str] = set()
    for conn in req.connections:
        if conn.name in seen:
            raise HTTPException(
                status_code=400,
                detail=f"duplicate connection name {conn.name!r}",
            )
        seen.add(conn.name)

    try:
        # Preserve any [connections.<name>] blocks that previously
        # failed to parse — a transient typo in one entry shouldn't
        # be silently erased by an unrelated edit elsewhere. The
        # writer's malformed-passthrough is the durable surface for
        # that contract; the route just has to wire it through.
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
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/connections/{name}/schema")
async def get_connection_schema(notebook_id: str, name: str) -> dict:
    """Enumerate tables (and columns) on a connection.

    Powers the schema-discovery sidebar so users can see what's
    available before writing SQL. Opens the connection in
    enforced read-only mode, runs the adapter's ``list_schema``,
    closes. Errors surface verbatim so a missing-driver / auth
    failure / unreachable host is visible to the user instead of
    a confusing 500.
    """
    from strata.notebook.sql.cell_executor import (
        _resolve_runtime_spec,  # type: ignore[attr-defined]
        _safely_close,  # type: ignore[attr-defined]
    )
    from strata.notebook.sql.registry import get_adapter

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
async def list_notebook_workers(notebook_id: str, refresh: bool = False) -> dict:
    """List the worker catalog visible to a notebook."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return await _serialize_worker_catalog(session, force_refresh=refresh)


@router.put("/{notebook_id}/workers")
async def update_notebook_workers_endpoint(
    notebook_id: str,
    req: WorkersConfigRequest,
) -> dict:
    """Replace notebook-scoped worker definitions."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
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
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/worker")
async def update_notebook_worker_endpoint(
    notebook_id: str,
    req: WorkerConfigRequest,
) -> dict:
    """Replace the notebook-level default worker."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
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
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/timeout")
async def update_notebook_timeout_endpoint(
    notebook_id: str,
    req: TimeoutConfigRequest,
) -> dict:
    """Replace the notebook-level default timeout."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        update_notebook_timeout(session.path, req.timeout)
        session.reload()
        return {
            "timeout": session.notebook_state.timeout,
            "cells": session.serialize_cells(),
        }
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/variant-groups/{group_id}/variants")
async def add_variant_endpoint(
    notebook_id: str,
    group_id: str,
) -> dict:
    """Add a sibling variant to an existing group.

    Clones the active variant's body and rewrites the ``# @variant``
    annotation to a fresh auto-generated name. The new variant becomes
    active on creation so the user can immediately edit it; renaming
    happens by editing the annotation line in source.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        new_name, new_cell_id = session.add_variant(group_id)
        return {
            "new_variant_name": new_name,
            "new_cell_id": new_cell_id,
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
        }
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/variant-groups/{group_id}")
async def set_variant_active_endpoint(
    notebook_id: str,
    group_id: str,
    req: VariantActiveRequest,
) -> dict:
    """Switch the active variant for a group.

    The group itself must be declared by ``# @variant`` annotations in
    cell source — this endpoint only updates the persisted active-variant
    pointer in notebook.toml. Picking a name that no cell provides is
    accepted (write succeeds), but ``annotation_validation`` will surface
    a ``variant_active_unknown`` diagnostic and the DAG falls back to
    the first variant in source order.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        session.set_variant_active(group_id, req.active)
        return {
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
        }
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/env")
async def update_notebook_env_endpoint(
    notebook_id: str,
    req: EnvConfigRequest,
) -> dict:
    """Replace the notebook-level default env vars."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        update_notebook_env(session.path, req.env)
        session.reload()
        # The disk writer strips sensitive values (API keys, tokens) so
        # they don't leak into git. Restore the full values in the
        # in-memory session so the LLM config and Runtime panel work
        # for the duration of this session.
        session.notebook_state.env.update(req.env)
        # Also rebuild each cell's resolved env from the fresh
        # notebook-level env plus the preserved per-cell overrides —
        # otherwise the executor (which reads cell.env) still sees the
        # blanked values and API-key-dependent cells fail with "key not
        # set" even though the Runtime panel was updated.
        # Anything the user just typed in the Runtime panel is a
        # manual override — mark it as such in env_sources so the UI
        # shows the correct badge.
        for key in req.env:
            session.notebook_state.env_sources[key] = "manual"
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
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

    Field set matches exactly what update_notebook_secret_manager accepts —
    anything else gets dropped on the writer side so arbitrary runtime
    state can't sneak into the committed TOML via this path.
    """

    provider: str | None = None
    project_id: str | None = None
    environment: str | None = None
    path: str | None = None
    base_url: str | None = None


@router.put("/{notebook_id}/secret-manager/config")
async def update_notebook_secret_manager_config(
    notebook_id: str,
    req: SecretManagerConfigRequest,
) -> dict:
    """Persist the [secret_manager] block to notebook.toml and refetch.

    Empty payload (all fields None / empty) removes the block —
    "disconnect from secret manager". A refresh is triggered after the
    write so the Runtime panel immediately reflects the new values.
    """
    from strata.notebook.writer import update_notebook_secret_manager

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
    try:
        config = req.model_dump(exclude_none=True)
        update_notebook_secret_manager(session.path, config)
        session.reload()
        # After reload, env keys set via the Runtime panel are blanked
        # on disk — restore whatever the in-memory state had for keys
        # we're not replacing from the fresh fetch.
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/secret-manager/refresh")
async def refresh_notebook_secret_manager(notebook_id: str) -> dict:
    """Re-fetch secrets from the configured manager and merge into env.

    Returns the same shape as the env endpoint so the frontend can
    swap the panel state in place. Never 500s on fetch error — the
    error message comes back in ``env_fetch_error`` so the UI can
    display it next to the Refresh button.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
    try:
        session.refresh_secrets()
        # Propagate the refreshed env back to each cell's resolved view
        # so the executor picks up rotated values immediately.
        for cell in session.notebook_state.cells:
            resolved = dict(session.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved
        return _serialize_env_response(session)
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{notebook_id}/cells")
async def add_cell(notebook_id: str, req: AddCellRequest) -> dict:
    """Add a new cell to the notebook.

    Args:
        notebook_id: Notebook/session ID
        req: AddCellRequest with optional after_cell_id

    Returns:
        New cell state
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        # Generate cell ID
        cell_id = str(uuid.uuid4())[:8]

        # Add to notebook
        add_cell_to_notebook(
            session.path,
            cell_id,
            req.after_cell_id,
            language=req.language,
        )

        # Reload notebook state
        session.reload()

        # Find and return the new cell
        cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
        if not cell:
            raise HTTPException(status_code=500, detail="Failed to create cell")

        return session.serialize_cell(cell)
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{notebook_id}/cells/{cell_id}")
async def delete_cell(notebook_id: str, cell_id: str) -> dict:
    """Delete a cell from the notebook.

    Args:
        notebook_id: Notebook/session ID
        cell_id: Cell ID to delete

    Returns:
        Success message
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        # Check if cell exists before deleting
        if not any(c.id == cell_id for c in session.notebook_state.cells):
            raise HTTPException(status_code=404, detail="Cell not found")

        # Variant-aware delete: when the cell is part of a variant group,
        # session.remove_cell promotes the next variant to active (or
        # dissolves the group on the last delete) so the toml pointer
        # stays consistent. Non-variant cells take the same code path
        # with no extra work.
        session.remove_cell(cell_id)

        # Return the refreshed variant_groups + serialized cells *and*
        # the DAG so the client can resync without a separate round-trip.
        # Deleting a variant member can promote a sibling to active and
        # re-wire downstream producers; without ``dag`` in the response
        # the DAG view ends up rendering stale edges and the group
        # appears orphaned until the next reload.
        from strata.notebook.ws import _serialize_dag_edges

        return {
            "message": "Cell deleted",
            "cell_id": cell_id,
            "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            "cells": session.serialize_cells(),
            "dag": {
                "edges": _serialize_dag_edges(session),
                "roots": list(session.dag.roots) if session.dag else [],
                "leaves": list(session.dag.leaves) if session.dag else [],
                "topological_order": session.dag.topological_order if session.dag else [],
                "variant_groups": [vg.model_dump() for vg in session.notebook_state.variant_groups],
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/{notebook_id}/name")
async def rename_notebook_endpoint(notebook_id: str, req: RenameNotebookRequest) -> dict:
    """Rename the notebook.

    Args:
        notebook_id: Notebook/session ID
        req: RenameNotebookRequest with name

    Returns:
        Updated notebook state
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    try:
        rename_notebook(session.path, req.name)

        # Reload notebook state
        session.reload()

        return {
            "notebook_id": session.notebook_state.id,
            "name": session.notebook_state.name,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Internal server error")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{notebook_id}/dag")
async def get_notebook_dag(notebook_id: str) -> dict:
    """Get the DAG for a notebook.

    Args:
        notebook_id: Notebook/session ID

    Returns:
        DAG edges, topological order, leaves, roots, and per-cell metadata
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return _format_dag(session)


@router.get("/{notebook_id}/cells/{cell_id}/iterations")
async def get_cell_iterations(
    notebook_id: str,
    cell_id: str,
    variable: str | None = None,
) -> dict:
    """List stored iteration artifacts for a loop cell.

    The carry variable is read from the cell's ``@loop`` annotation when
    the ``variable`` query parameter is not supplied. Non-loop cells and
    loops with no completed iterations return an empty list — the
    endpoint is a safe poll target for the inspect panel.
    """
    from strata.notebook.annotations import parse_annotations

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
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


@router.get("/{notebook_id}/dependencies")
async def get_dependencies(notebook_id: str) -> dict:
    """List current dependencies for a notebook.

    Returns:
        List of dependencies from pyproject.toml
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    return _serialize_environment_payload(session)


@router.post("/{notebook_id}/dependencies")
async def add_notebook_dependency(notebook_id: str, req: AddDependencyRequest) -> dict:
    """Add a dependency to the notebook.

    Runs ``uv add``, updates pyproject.toml + uv.lock, syncs venv.
    If the lockfile changes, the session's venv_python is re-synced.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
async def remove_notebook_dependency(notebook_id: str, package_name: str) -> dict:
    """Remove a dependency from the notebook.

    Runs ``uv remove``, updates pyproject.toml + uv.lock, syncs venv.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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
    """Format the DAG for API response.

    Args:
        session: NotebookSession

    Returns:
        DAG data as dict
    """
    if not session.dag:
        return {
            "edges": [],
            "topological_order": [],
            "leaves": [],
            "roots": [],
            "variable_producer": {},
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
        "variable_producer": session.dag.variable_producer,
    }


@router.post("/{notebook_id}/cells/{cell_id}/execute")
async def execute_cell(notebook_id: str, cell_id: str) -> dict:
    """Execute a cell and return results.

    Args:
        notebook_id: Notebook/session ID
        cell_id: Cell ID to execute

    Returns:
        Execution result with outputs and stdout/stderr
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

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

    # Find the cell
    cell = next((c for c in session.notebook_state.cells if c.id == cell_id), None)
    if not cell:
        raise HTTPException(status_code=404, detail="Cell not found")

    try:
        # Execute the cell
        cell.status = CellStatus.RUNNING
        executor = CellExecutor(session, session.warm_pool)
        result = await executor.execute_cell(cell_id, cell.source)
        session.record_execution(cell_id, result.duration_ms, result.cache_hit)
        session.apply_execution_result_metadata(cell_id, result)
        if result.success:
            session.compute_staleness()
            session.mark_executed_ready(cell_id)
        else:
            cell.status = CellStatus.ERROR

        return result.to_dict()
    except Exception:
        cell.status = CellStatus.ERROR
        logger.exception("Cell execution failed")
        raise HTTPException(status_code=500, detail="Execution failed")


def _render_notebook_export(
    session,
    *,
    fmt: str,
    include_inactive_variants: bool,
):
    """Render a notebook to a single shareable markdown/HTML file.

    Mirrors ``strata export`` (the CLI). Response is the rendered body
    plus a ``Content-Disposition: attachment`` header so the browser
    triggers a download rather than displaying inline.
    """
    from fastapi.responses import Response

    from strata.notebook.export import ExportOptions, export_notebook

    try:
        options = ExportOptions(
            output_format=fmt,  # type: ignore[arg-type]
            include_inactive_variants=bool(include_inactive_variants),
        )
        body = export_notebook(session.path, options)
    except Exception:
        logger.exception("render export failed for notebook %s", session.id)
        raise HTTPException(status_code=500, detail="Export failed")

    safe_name = session.path.name or "notebook"
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
    fmt: str = "zip",
    include_inactive_variants: bool = False,
):
    """Export the notebook in the requested format.

    Default (``fmt=zip``): a reproducible bundle containing
    ``notebook.toml``, ``pyproject.toml``, ``uv.lock`` (if present),
    every cell source file, and ``provenance.json`` (DAG + per-cell
    provenance hashes). This is the original endpoint behavior — kept
    as the default so callers and tooling that didn't specify a format
    continue to receive a ZIP.

    ``fmt=markdown`` and ``fmt=html`` return a rendered single-file
    export of the notebook produced by
    :func:`strata.notebook.export.export_notebook`. Prompt cell
    responses are intentionally excluded; see
    ``docs/internal/design-notebook-export.md``.

    Query parameters:
        fmt: one of ``zip`` (default), ``markdown``, ``html``.
        include_inactive_variants: only relevant for markdown/html
            renderings; stacks all variants of every group when true.
    """
    if fmt not in {"zip", "markdown", "html"}:
        raise HTTPException(
            status_code=400,
            detail="fmt must be one of 'zip', 'markdown', 'html'",
        )

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    if fmt in {"markdown", "html"}:
        return _render_notebook_export(
            session,
            fmt=fmt,
            include_inactive_variants=bool(include_inactive_variants),
        )

    # Default: ZIP bundle.
    nb_dir = session.path
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # notebook.toml
        toml_path = nb_dir / "notebook.toml"
        if toml_path.exists():
            zf.write(toml_path, "notebook.toml")

        # pyproject.toml
        pyproject_path = nb_dir / "pyproject.toml"
        if pyproject_path.exists():
            zf.write(pyproject_path, "pyproject.toml")

        # uv.lock
        lock_path = nb_dir / "uv.lock"
        if lock_path.exists():
            zf.write(lock_path, "uv.lock")

        # Cell source files
        cells_dir = nb_dir / "cells"
        if cells_dir.is_dir():
            for cell_file in sorted(cells_dir.glob("*.py")):
                zf.write(cell_file, f"cells/{cell_file.name}")

        # provenance.json — DAG + per-cell hashes for reproducibility
        from strata.notebook.env import compute_lockfile_hash
        from strata.notebook.provenance import compute_source_hash

        provenance: dict = {
            "notebook_id": session.notebook_state.id,
            "lockfile_hash": compute_lockfile_hash(nb_dir),
            "dag": _format_dag(session),
            "cells": {},
        }
        for cell in session.notebook_state.cells:
            provenance["cells"][cell.id] = {
                "source_hash": compute_source_hash(cell.source),
                "defines": cell.defines,
                "references": cell.references,
                "status": cell.status,
                "artifact_uri": cell.artifact_uri,
            }
        zf.writestr("provenance.json", json.dumps(provenance, indent=2))

    buf.seek(0)
    filename = f"{_safe_filename(session.notebook_state.name or 'notebook')}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# LLM Assistant
# ============================================================================


class LlmChatTurn(BaseModel):
    """One prior chat turn passed back to the LLM for context."""

    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=20_000)


class LlmCompleteRequest(BaseModel):
    """Request for LLM assistant completion."""

    message: str = Field(..., max_length=10_000, description="User message")
    cell_id: str | None = Field(default=None, description="Target cell ID for context")
    history: list[LlmChatTurn] = Field(
        default_factory=list,
        description="Prior chat turns in this session (capped to most recent)",
        max_length=20,
    )


def _read_notebook_ai_config(session) -> dict | None:
    """Read [ai] section from notebook.toml if present."""
    notebook_toml = session.path / "notebook.toml"
    if not notebook_toml.exists():
        return None
    try:
        with open(notebook_toml, "rb") as f:
            data = tomllib.load(f)
        ai_section = data.get("ai")
        return ai_section if isinstance(ai_section, dict) else None
    except Exception:
        return None


def _get_llm_config(session):
    """Resolve LLM config for a notebook session."""
    from strata.notebook.llm import resolve_llm_config

    notebook_ai = _read_notebook_ai_config(session)

    server_config = None
    try:
        from strata.server import get_state

        server_config = get_state().config
    except RuntimeError:
        pass

    # Notebook-level env vars (set via Runtime panel)
    notebook_env = getattr(session.notebook_state, "env", None) or {}

    return resolve_llm_config(notebook_ai, server_config, notebook_env)


@router.get("/{notebook_id}/ai/status")
async def llm_status(notebook_id: str) -> dict:
    """Check if the LLM assistant is configured and available."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    from strata.notebook.llm import infer_provider_name

    config = _get_llm_config(session)
    if config is None:
        return {"available": False, "model": None, "provider": None}

    return {
        "available": True,
        "model": config.model,
        "provider": infer_provider_name(config.base_url),
    }


@router.get("/{notebook_id}/ai/models")
async def llm_models(notebook_id: str) -> dict:
    """Fetch available models from the configured LLM provider.

    Queries the provider's ``/models`` endpoint (OpenAI-compatible) and
    returns a list of model IDs sorted alphabetically.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    config = _get_llm_config(session)
    if config is None:
        return {"models": [], "current": None}

    from strata.notebook.llm import infer_provider_name

    models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{config.base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {config.api_key}"},
            )
            if resp.status_code == 200:
                data = resp.json()
                raw = data.get("data") or data.get("models") or []
                for item in raw:
                    model_id = item.get("id") if isinstance(item, dict) else str(item)
                    if model_id:
                        models.append(model_id)
                models.sort()
    except Exception:
        logger.debug("Failed to fetch models from %s", config.base_url, exc_info=True)

    return {
        "models": models,
        "current": config.model,
        "provider": infer_provider_name(config.base_url),
    }


class UpdateAiModelRequest(BaseModel):
    """Request to update the notebook's default LLM model."""

    model: str = Field(..., max_length=200, description="Model ID")


@router.put("/{notebook_id}/ai/model")
async def update_ai_model(notebook_id: str, req: UpdateAiModelRequest) -> dict:
    """Set the notebook's default LLM model.

    Writes to the ``[ai]`` section of notebook.toml.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    from strata.notebook.writer import update_notebook_ai_model

    update_notebook_ai_model(session.path, req.model)
    session.reload()

    config = _get_llm_config(session)
    return {
        "model": config.model if config else req.model,
    }


def _prepare_chat_request(session, req: LlmCompleteRequest):
    """Resolve config + build messages for a chat request.

    Returns ``(config, messages)``. Raises ``HTTPException`` if LLM is not
    configured.
    """
    from strata.notebook.llm import build_messages, build_notebook_context

    config = _get_llm_config(session)
    if config is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM assistant not configured for this notebook. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or STRATA_AI_API_KEY "
                "in the Runtime panel, or add an [ai] section to notebook.toml."
            ),
        )

    notebook_context = build_notebook_context(session, max_tokens=config.max_context_tokens // 4)

    cell_source = None
    if req.cell_id:
        cell = next(
            (c for c in session.notebook_state.cells if c.id == req.cell_id),
            None,
        )
        if cell:
            cell_source = cell.source

    history = [{"role": t.role, "content": t.content} for t in req.history]
    messages = build_messages(
        req.message,
        notebook_context,
        history=history,
        cell_source=cell_source,
    )
    return config, messages


@router.post("/{notebook_id}/ai/complete")
async def llm_complete(notebook_id: str, req: LlmCompleteRequest) -> dict:
    """Run a (blocking) LLM chat completion. Prefer ``/ai/stream`` for UI."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    from strata.notebook.llm import chat_completion

    config, messages = _prepare_chat_request(session, req)

    try:
        result = await chat_completion(config, messages)
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 429:
            raise HTTPException(status_code=429, detail="LLM provider rate limited")
        raise HTTPException(status_code=502, detail=f"LLM provider error: {status}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM provider timed out")
    except Exception:
        logger.exception("LLM completion failed")
        raise HTTPException(status_code=502, detail="LLM completion failed")

    return {
        "content": result.content,
        "model": result.model,
        "tokens": {
            "input": result.input_tokens,
            "output": result.output_tokens,
        },
    }


@router.post("/{notebook_id}/ai/stream")
async def llm_stream(notebook_id: str, req: LlmCompleteRequest):
    """Stream an LLM chat completion as Server-Sent Events.

    Event types:
    - ``delta``: ``{"text": "..."}`` — incremental content chunk
    - ``done``: ``{"model": "...", "tokens": {"input": N, "output": N}}``
    - ``error``: ``{"message": "..."}``
    """
    from fastapi.responses import StreamingResponse

    from strata.notebook.llm import chat_completion_stream

    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    config, messages = _prepare_chat_request(session, req)

    async def event_stream():
        def _sse(event: str, payload: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(payload)}\n\n"

        try:
            async for chunk in chat_completion_stream(config, messages):
                if chunk["type"] == "delta":
                    yield _sse("delta", {"text": chunk["text"]})
                elif chunk["type"] == "done":
                    yield _sse(
                        "done",
                        {
                            "model": chunk.get("model"),
                            "tokens": {
                                "input": chunk.get("input_tokens", 0),
                                "output": chunk.get("output_tokens", 0),
                            },
                        },
                    )
        except httpx.HTTPStatusError as e:
            logger.warning("LLM stream HTTP error: %s", e.response.status_code)
            yield _sse("error", {"message": f"LLM provider error: {e.response.status_code}"})
        except httpx.TimeoutException:
            yield _sse("error", {"message": "LLM provider timed out"})
        except Exception as e:
            logger.exception("LLM stream failed")
            yield _sse("error", {"message": f"LLM stream failed: {e}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================================
# LLM Agent
# ============================================================================


class AgentRequest(BaseModel):
    """Request to start an agent loop."""

    message: str = Field(..., max_length=10_000, description="User instruction")
    auto_approve: bool = Field(
        default=False,
        description=(
            "Skip the user-approval gate for destructive tools (``delete_cell``, ``add_package``)."
        ),
    )


# Per-notebook agent cancellation events
_agent_cancel_events: dict[str, asyncio.Event] = {}


def _short_payload_detail(payload: dict[str, Any]) -> str:
    """Compact stringification of a structured progress payload.

    The progress log in the panel shows ``event + detail`` lines; we
    keep that surface stable by collapsing the dict into a short summary
    while still emitting the full payload alongside.
    """
    if "detail" in payload and isinstance(payload["detail"], str):
        return payload["detail"][:200]
    if "summary" in payload and isinstance(payload["summary"], str):
        return payload["summary"][:200]
    if "tool" in payload:
        if "result" in payload:
            return f"{payload['tool']} → {str(payload['result'])[:160]}"
        if "arguments" in payload:
            args = payload["arguments"]
            arg_str = ", ".join(f"{k}={str(v)[:50]}" for k, v in args.items())
            return f"{payload['tool']}({arg_str})"
    return ""


@router.post("/{notebook_id}/ai/agent")
async def run_agent(notebook_id: str, req: AgentRequest) -> dict:
    """Run an LLM agent loop with tool use, streaming, and approvals.

    Progress is streamed via WebSocket: ``agent_progress`` for log
    events, ``agent_text_delta`` for live narrative chunks, and
    ``agent_confirm_request`` when the agent wants to run a destructive
    tool. Completion arrives as ``agent_done``. Only one agent runs per
    notebook at a time.
    """
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")

    from strata.notebook.llm import run_agent_loop

    config = _get_llm_config(session)
    if config is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM not configured for this notebook. "
                "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or STRATA_AI_API_KEY "
                "in the Runtime panel, or add an [ai] section to notebook.toml."
            ),
        )

    # Only one agent per notebook
    if notebook_id in _agent_cancel_events:
        raise HTTPException(
            status_code=409,
            detail="An agent is already running for this notebook.",
        )

    cancel_event = asyncio.Event()
    _agent_cancel_events[notebook_id] = cancel_event

    job_id = str(uuid.uuid4())[:8]

    async def _progress(event_type: str, payload: dict[str, Any]) -> None:
        """Translate a structured agent event into the right WS message."""
        from strata.notebook.ws import _broadcast_message

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            if event_type == "text_delta":
                await _broadcast_message(
                    notebook_id,
                    {
                        "type": "agent_text_delta",
                        "seq": 0,
                        "ts": ts,
                        "payload": {"job_id": job_id, "text": payload.get("text", "")},
                    },
                )
                return
            if event_type == "confirm_request":
                await _broadcast_message(
                    notebook_id,
                    {
                        "type": "agent_confirm_request",
                        "seq": 0,
                        "ts": ts,
                        "payload": {**payload, "job_id": job_id},
                    },
                )
                return
            await _broadcast_message(
                notebook_id,
                {
                    "type": "agent_progress",
                    "seq": 0,
                    "ts": ts,
                    "payload": {
                        "event": event_type,
                        "detail": _short_payload_detail(payload),
                        "data": payload,
                        "job_id": job_id,
                    },
                },
            )
        except Exception:
            pass

    async def _run_agent_task() -> None:
        from strata.notebook.ws import _broadcast_message, broadcast_notebook_sync

        try:
            result = await run_agent_loop(
                config,
                session,
                req.message,
                notebook_id=notebook_id,
                max_iterations=10,
                cancel_event=cancel_event,
                progress_callback=_progress,
                auto_approve=req.auto_approve,
            )

            # Broadcast completion via WS
            await _broadcast_message(
                notebook_id,
                {
                    "type": "agent_done",
                    "seq": 0,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "payload": {
                        "job_id": job_id,
                        "content": result.content,
                        "model": result.model,
                        "tokens": {
                            "input": result.total_input_tokens,
                            "output": result.total_output_tokens,
                        },
                        "iterations": result.iterations,
                        "tool_calls": [
                            {
                                "tool": tc.tool_name,
                                "args": tc.arguments,
                                "result": tc.result,
                                "duration_ms": int(tc.duration_ms),
                            }
                            for tc in result.tool_calls
                        ],
                        "cancelled": result.cancelled,
                        "error": result.error,
                    },
                },
            )
        except Exception as e:
            logger.exception("Agent loop failed")
            try:
                await _broadcast_message(
                    notebook_id,
                    {
                        "type": "agent_done",
                        "seq": 0,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "payload": {
                            "job_id": job_id,
                            "content": "",
                            "error": str(e),
                            "cancelled": False,
                        },
                    },
                )
            except Exception:
                pass
        finally:
            _agent_cancel_events.pop(notebook_id, None)
            try:
                await broadcast_notebook_sync(notebook_id, session)
            except Exception:
                pass

    asyncio.create_task(_run_agent_task())

    return {"job_id": job_id, "status": "started"}


def cancel_agent(notebook_id: str) -> bool:
    """Cancel a running agent loop. Called from WebSocket handler."""
    event = _agent_cancel_events.get(notebook_id)
    if event is not None:
        event.set()
        return True
    return False


@router.post("/{notebook_id}/ai/agent/reset")
async def reset_agent(notebook_id: str) -> dict:
    """Clear the persistent conversation history for a notebook."""
    session = _session_manager.get_session(notebook_id)
    if not session:
        raise HTTPException(status_code=404, detail="Notebook not found")
    from strata.notebook.llm import reset_history

    reset_history(notebook_id)
    return {"status": "ok"}

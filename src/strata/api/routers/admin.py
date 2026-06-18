"""Admin routes: the service-mode notebook worker registry and per-tenant
observability.

Moved verbatim from ``server.py`` (P3, router split). The notebook-worker routes
are gated by the ``require_notebook_worker_admin`` dependency (service-mode + the
``admin:notebook-workers`` scope); its gate body
(``_require_notebook_worker_admin_access``) stays in ``server.py``, where the
dependency delegates to it. The tenant routes are gated by
``require_scope("admin:tenants")``. The admin-only request models and the two
serialize/validate helpers move here with the routes; the worker-registry
mutators come from ``strata.notebook.workers`` (already imported at server load).
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from strata.api.dependencies import require_notebook_worker_admin, require_scope
from strata.notebook.models import WorkerBackendType, WorkerConfig, WorkerSpec
from strata.notebook.workers import (
    ManagedWorkerRecord,
    build_server_worker_catalog_with_health,
    create_server_managed_worker_record,
    delete_server_managed_worker_record,
    get_server_managed_worker_records,
    replace_server_managed_worker_records,
    set_server_managed_worker_enabled,
    update_server_managed_worker_record,
)
from strata.tenant_registry import get_tenant_registry

router = APIRouter(tags=["admin"])


class AdminNotebookWorkerEntryRequest(BaseModel):
    """Service-managed notebook worker config entry."""

    name: str = Field(..., pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    backend: WorkerBackendType = Field(default=WorkerBackendType.LOCAL)
    runtime_id: str | None = Field(default=None)
    config: WorkerConfig = Field(default_factory=WorkerConfig)
    enabled: bool = True

    def to_worker_spec(self) -> WorkerSpec:
        return WorkerSpec(
            name=self.name,
            backend=self.backend,
            runtime_id=self.runtime_id,
            config=self.config,
        )


class AdminNotebookWorkersRequest(BaseModel):
    """Request payload for replacing the server-managed notebook worker registry."""

    workers: list[AdminNotebookWorkerEntryRequest] = Field(default_factory=list)


class AdminNotebookWorkerPatchRequest(BaseModel):
    """Request payload for patching one service-managed worker."""

    enabled: bool


async def _serialize_admin_notebook_workers(
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    return {
        "configured_workers": [
            {
                **record.worker.model_dump(mode="json"),
                "enabled": record.enabled,
            }
            for record in get_server_managed_worker_records()
        ],
        "workers": await build_server_worker_catalog_with_health(force_refresh=force_refresh),
        "definitions_editable": False,
        "health_checked_at": int(time.time() * 1000),
    }


def _validate_admin_notebook_worker_names(
    workers: list[AdminNotebookWorkerEntryRequest],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for worker in workers:
        if worker.name in seen:
            duplicates.add(worker.name)
        seen.add(worker.name)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate notebook worker names are not allowed: {duplicate_list}",
        )


@router.get(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def list_admin_notebook_workers(refresh: bool = False):
    """List the server-managed notebook worker registry."""
    return await _serialize_admin_notebook_workers(force_refresh=refresh)


@router.put(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def update_admin_notebook_workers(request: AdminNotebookWorkersRequest):
    """Replace the server-managed notebook worker registry."""
    _validate_admin_notebook_worker_names(request.workers)
    replace_server_managed_worker_records(
        [
            ManagedWorkerRecord(
                worker=worker.to_worker_spec(),
                enabled=worker.enabled,
            )
            for worker in request.workers
        ]
    )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.post(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def create_admin_notebook_worker(request: AdminNotebookWorkerEntryRequest):
    """Create one service-managed notebook worker."""
    try:
        create_server_managed_worker_record(
            ManagedWorkerRecord(
                worker=request.to_worker_spec(),
                enabled=request.enabled,
            )
        )
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Notebook worker already exists: {request.name}",
        )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.put(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def replace_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerEntryRequest,
):
    """Replace one service-managed notebook worker definition."""
    try:
        update_server_managed_worker_record(
            worker_name,
            ManagedWorkerRecord(
                worker=request.to_worker_spec(),
                enabled=request.enabled,
            ),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Notebook worker already exists: {request.name}",
        )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.patch(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def patch_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerPatchRequest,
):
    """Patch one service-managed notebook worker."""
    try:
        set_server_managed_worker_enabled(worker_name, request.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.delete(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def delete_admin_notebook_worker(worker_name: str):
    """Delete one service-managed notebook worker."""
    try:
        delete_server_managed_worker_record(worker_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.post(
    "/v1/admin/notebook-workers/{worker_name}/refresh",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def refresh_admin_notebook_worker(worker_name: str):
    """Force-refresh health for one service-managed notebook worker."""
    known_workers = {record.worker.name for record in get_server_managed_worker_records()}
    if worker_name not in known_workers:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@router.get("/v1/admin/tenants", dependencies=[require_scope("admin:tenants")])
async def list_tenants():
    """List all tracked tenants with their metrics.

    Admin endpoint for multi-tenant observability.
    Returns metrics for all tenants that have made requests.
    """
    registry = get_tenant_registry()
    return {"tenants": registry.get_all_tenant_metrics()}


@router.get("/v1/admin/tenants/{tenant_id}", dependencies=[require_scope("admin:tenants")])
async def get_tenant_info(tenant_id: str):
    """Get configuration and metrics for a specific tenant.

    Path params:
    - tenant_id: The tenant identifier
    """
    registry = get_tenant_registry()
    config = registry.get_config(tenant_id)
    metrics = registry.get_tenant_metrics(tenant_id)

    if config is None and metrics is None:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    return {
        "tenant_id": tenant_id,
        "registered": config is not None,
        "enabled": config.enabled if config else True,
        "metrics": metrics,
        "config": {
            "interactive_slots": config.interactive_slots if config else None,
            "bulk_slots": config.bulk_slots if config else None,
            "per_client_interactive": config.per_client_interactive if config else None,
            "per_client_bulk": config.per_client_bulk if config else None,
        }
        if config
        else None,
    }

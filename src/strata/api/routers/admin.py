"""Admin routes: the server-managed notebook worker registry and per-tenant observability.

Worker routes need the ``admin:notebook-workers`` scope under principal auth, in either mode:
a personal server's caller already runs cells as its owner. Tenant routes need ``admin:tenants``.
"""

from __future__ import annotations

import time

import anyio.to_thread
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from strata.api.dependencies import require_scope
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

    @field_validator("name")
    @classmethod
    def _not_local(cls, name: str) -> str:
        # Resolution returns the built-in first, so a "local" entry would never run.
        if name == "local":
            raise ValueError("'local' is reserved for the built-in worker")
        return name

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
    records: list[ManagedWorkerRecord],
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    return {
        "configured_workers": [
            {
                **record.worker.model_dump(mode="json"),
                "enabled": record.enabled,
            }
            for record in records
        ],
        "workers": await build_server_worker_catalog_with_health(
            force_refresh=force_refresh, records=records
        ),
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
    dependencies=[require_scope("admin:notebook-workers")],
)
async def list_admin_notebook_workers(refresh: bool = False):
    """List the server-managed notebook worker registry."""
    records = await anyio.to_thread.run_sync(get_server_managed_worker_records)
    return await _serialize_admin_notebook_workers(records, force_refresh=refresh)


@router.put(
    "/v1/admin/notebook-workers",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def update_admin_notebook_workers(request: AdminNotebookWorkersRequest):
    """Replace the server-managed notebook worker registry."""
    _validate_admin_notebook_worker_names(request.workers)
    records = await anyio.to_thread.run_sync(
        replace_server_managed_worker_records,
        [
            ManagedWorkerRecord(
                worker=worker.to_worker_spec(),
                enabled=worker.enabled,
            )
            for worker in request.workers
        ],
    )
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.post(
    "/v1/admin/notebook-workers",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def create_admin_notebook_worker(request: AdminNotebookWorkerEntryRequest):
    """Create one service-managed notebook worker."""
    try:
        records = await anyio.to_thread.run_sync(
            create_server_managed_worker_record,
            ManagedWorkerRecord(
                worker=request.to_worker_spec(),
                enabled=request.enabled,
            ),
        )
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Notebook worker already exists: {request.name}",
        )
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.put(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def replace_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerEntryRequest,
):
    """Replace one service-managed notebook worker definition."""
    try:
        records = await anyio.to_thread.run_sync(
            update_server_managed_worker_record,
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
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.patch(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def patch_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerPatchRequest,
):
    """Patch one service-managed notebook worker."""
    try:
        records = await anyio.to_thread.run_sync(
            set_server_managed_worker_enabled, worker_name, request.enabled
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.delete(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def delete_admin_notebook_worker(worker_name: str):
    """Delete one service-managed notebook worker."""
    try:
        records = await anyio.to_thread.run_sync(delete_server_managed_worker_record, worker_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.post(
    "/v1/admin/notebook-workers/{worker_name}/refresh",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def refresh_admin_notebook_worker(worker_name: str):
    """Force-refresh health for one service-managed notebook worker."""
    records = await anyio.to_thread.run_sync(get_server_managed_worker_records)
    if worker_name not in {record.worker.name for record in records}:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.post(
    "/v1/admin/notebook-workers/reload",
    dependencies=[require_scope("admin:notebook-workers")],
)
async def reload_admin_notebook_workers():
    """Refresh every worker's health and drop cached health for workers no longer listed.

    The registry itself is read from the metadata store on every request, so a change
    made through another node needs no reload.
    """
    from strata.notebook.workers import prune_worker_health_cache

    records = await anyio.to_thread.run_sync(get_server_managed_worker_records)
    await prune_worker_health_cache(records)
    return await _serialize_admin_notebook_workers(records, force_refresh=True)


@router.get("/v1/admin/tenants", dependencies=[require_scope("admin:tenants")])
async def list_tenants():
    """List every tenant that has made requests, with its metrics."""
    registry = get_tenant_registry()
    return {"tenants": registry.get_all_tenant_metrics()}


@router.get("/v1/admin/tenants/{tenant_id}", dependencies=[require_scope("admin:tenants")])
async def get_tenant_info(tenant_id: str):
    """Get configuration and metrics for one tenant; 404 if neither exists."""
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

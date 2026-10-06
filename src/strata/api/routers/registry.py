"""Registry routes: audit, events, dashboard summary, and the protected-alias approval queue.

Reads are tenant-scoped; personal mode (no principal) and ``admin:*`` see the whole
store, except the approval queue, which stays in the caller's tenant because approval
does, and the by-tag lookup, which finds the caller's own stamps. With
``notebook_remote_store_url`` set, every route answers from that store, where the
notebook's names actually live.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strata.api.dependencies import CurrentPrincipal, ReadStore, RegistryDecisionContext
from strata.api.remote_registry import forward, remote_registry
from strata.services.registry import registry_service

router = APIRouter(tags=["registry"])


@router.get("/v1/registry/audit")
async def registry_audit(
    store: ReadStore,
    principal: CurrentPrincipal,
    name: str | None = None,
    artifact_id: str | None = None,
    limit: int = 100,
):
    """Read the append-only registry audit, newest first, scoped to the caller's tenant."""
    target = remote_registry()
    if target is not None:
        return await forward(
            target,
            "GET",
            "/v1/registry/audit",
            params={"name": name, "artifact_id": artifact_id, "limit": limit},
        )

    if principal is None or principal.has_scope("admin:*"):
        entries = store.read_audit(name=name, artifact_id=artifact_id, limit=limit)
    else:
        entries = store.read_audit(
            name=name, artifact_id=artifact_id, limit=limit, tenant=principal.tenant
        )
    return {"entries": entries}


@router.get("/v1/events")
async def store_events(
    store: ReadStore,
    principal: CurrentPrincipal,
    since: int = 0,
    limit: int = 100,
):
    """List store events after ``since``, oldest first.

    Registry moves, alias requests and outcomes, publications and withdrawals.

    Pass back ``next`` as ``since`` to follow without missing or repeating an event.
    """
    limit = max(1, min(limit, 1000))
    target = remote_registry()
    if target is not None:
        return await forward(target, "GET", "/v1/events", params={"since": since, "limit": limit})

    if principal is None or principal.has_scope("admin:*"):
        events = store.read_events(since=since, limit=limit)
    else:
        events = store.read_events(since=since, limit=limit, tenant=principal.tenant)
    return {"events": events, "next": events[-1]["seq"] if events else since}


@router.get("/v1/registry/summary")
async def registry_summary(store: ReadStore, principal: CurrentPrincipal):
    """List each name with its aliases (``alias -> version``), current version and its tags."""
    target = remote_registry()
    if target is not None:
        return await forward(target, "GET", "/v1/registry/summary")

    if principal is None or principal.has_scope("admin:*"):
        return {"names": registry_service.summary(store, tenant=None, all_tenants=True)}
    return {"names": registry_service.summary(store, tenant=principal.tenant)}


@router.get("/v1/registry/artifacts")
async def registry_artifacts_by_tag(
    store: ReadStore,
    principal: CurrentPrincipal,
    tag_key: str,
    tag_value: str | None = None,
):
    """List ready artifacts carrying one tag, with their names and tags.

    Without ``tag_value``, every artifact carrying the key comes back. Lets the
    notebook find a cell's published artifacts (``nb_cell=<id>``) on any store.
    """
    target = remote_registry()
    if target is not None:
        params = {"tag_key": tag_key}
        if tag_value is not None:
            params["tag_value"] = tag_value
        return await forward(target, "GET", "/v1/registry/artifacts", params=params)

    tenant = principal.tenant if principal is not None else None
    return {
        "artifacts": registry_service.artifacts_by_tag(store, tag_key, tag_value, tenant=tenant)
    }


class PendingDecisionRequest(BaseModel):
    name: str
    alias: str


@router.get("/v1/registry/pending")
async def registry_pending(store: ReadStore, principal: CurrentPrincipal):
    """List protected-alias changes awaiting approval."""
    target = remote_registry()
    if target is not None:
        return await forward(target, "GET", "/v1/registry/pending")

    tenant_id = principal.tenant if principal else None
    return {"pending": store.list_pending_changes(tenant=tenant_id)}


@router.post("/v1/registry/pending/approve")
async def approve_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Apply a pending alias change; the approver becomes the audit actor.

    Requires ``admin:registry`` under trusted-proxy auth. The requester cannot
    self-approve (403) without ``admin:*``. With a team store, the decision is
    forwarded under this server's remote-store identity.
    """
    target = remote_registry()
    if target is not None:
        return await forward(
            target,
            "POST",
            "/v1/registry/pending/approve",
            json_body={"name": request.name, "alias": request.alias},
        )

    principal, store = decision
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None
    is_superadmin = principal is not None and principal.has_scope("admin:*")

    try:
        applied = store.approve_alias_change(
            request.name,
            request.alias,
            tenant=tenant_id,
            actor=actor,
            require_distinct_approver=not is_superadmin,
        )
    except ValueError as e:
        msg = str(e)
        status = 403 if msg.startswith("Separation of duty") else 404
        raise HTTPException(status_code=status, detail=msg)
    if applied.get("action") == "set":
        from strata.api.routers.names import _follow_alias_in_table

        await _follow_alias_in_table(
            store, applied["artifact_id"], applied["version"], request.alias, tenant_id
        )
    return {"status": "approved", "applied": applied}


@router.post("/v1/registry/pending/reject")
async def reject_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Discard a pending alias change (audited).

    Requires ``admin:registry`` under trusted-proxy auth.
    """
    target = remote_registry()
    if target is not None:
        return await forward(
            target,
            "POST",
            "/v1/registry/pending/reject",
            json_body={"name": request.name, "alias": request.alias},
        )

    principal, store = decision
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    try:
        rejected = store.reject_alias_change(
            request.name, request.alias, tenant=tenant_id, actor=actor
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "rejected", "rejected": rejected}

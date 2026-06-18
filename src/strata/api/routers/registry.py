"""Registry routes: audit, dashboard summary, and the protected-alias
approval queue (pending / approve / reject).

Moved verbatim from ``server.py`` (P3, router split). Unlike the other routers
this one needs no ``server`` import at all — every handler takes its store and
principal through the typed dependencies in ``strata.api.dependencies`` and
delegates aggregation to ``registry_service``. The governance gate body
(``_require_registry_approver``) stays in ``server.py``; the
``RegistryDecisionContext`` dependency delegates to it.

All registry reads are tenant-scoped: a principal sees only its own tenant;
personal mode (no principal) and ``admin:*`` see the whole store.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from strata.api.dependencies import CurrentPrincipal, ReadStore, RegistryDecisionContext
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
    """Read the append-only registry audit, newest first.

    Scoped to the caller's tenant: a principal sees only its own tenant's
    history. ``admin:*`` (and personal mode, where there is no principal)
    sees the whole store — every other registry route scopes the same way.
    """
    if principal is None or principal.has_scope("admin:*"):
        entries = store.read_audit(name=name, artifact_id=artifact_id, limit=limit)
    else:
        entries = store.read_audit(
            name=name, artifact_id=artifact_id, limit=limit, tenant=principal.tenant
        )
    return {"entries": entries}


@router.get("/v1/registry/summary")
async def registry_summary(store: ReadStore, principal: CurrentPrincipal):
    """Registry state for the dashboard names table: each name with its
    aliases (``alias -> version``), current version, and that version's tags.
    One call instead of ``/v1/names`` + a per-name alias fetch. Tenant-scoped
    like the other registry reads (personal mode / ``admin:*`` see all)."""
    tenant = None if (principal is None or principal.has_scope("admin:*")) else principal.tenant
    return {"names": registry_service.summary(store, tenant=tenant)}


class PendingDecisionRequest(BaseModel):
    name: str
    alias: str


@router.get("/v1/registry/pending")
async def registry_pending(store: ReadStore, principal: CurrentPrincipal):
    """List protected-alias changes awaiting approval."""
    tenant_id = principal.tenant if principal else None
    return {"pending": store.list_pending_changes(tenant=tenant_id)}


@router.post("/v1/registry/pending/approve")
async def approve_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Apply a pending alias change; the approver becomes the audit actor.

    Requires the ``admin:registry`` scope under trusted-proxy auth, and
    enforces separation of duty — the requester cannot self-approve unless
    they hold the ``admin:*`` break-glass scope.
    """
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
    return {"status": "approved", "applied": applied}


@router.post("/v1/registry/pending/reject")
async def reject_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Discard a pending alias change (audited).

    Requires the ``admin:registry`` scope under trusted-proxy auth so a
    tenant member cannot quietly drop a colleague's pending promotion.
    """
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

"""Name registry routes: names (resolve/set/delete/list/status), aliases, and tags.

Moved verbatim from ``server.py`` (P3 / A1b, router split). Thin handlers over
the P1 typed dependencies. ``get_state`` (the protected-alias config on the
alias set/delete paths) and the governance gate ``_require_registry_approver``
are reached via in-body lazy import and stay in ``server.py``. The shared
table-ACL resolver used by name-status staleness is ``resolve_input_version``,
imported from ``strata.api.dependencies`` (#295) — the same enforced unit
materialize and explain call.

Route order matters: the greedy ``/v1/names/{name:path}`` resolver MUST stay
registered AFTER the more specific ``/v1/names/{name:path}/aliases/...`` routes,
or ``.../aliases/x`` URLs get swallowed as part of the name — so the alias
handlers are defined first in this module.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from strata.api.dependencies import (
    CurrentPrincipal,
    PersonalModeStore,
    ReadStore,
    WriteStore,
)
from strata.types import (
    InputChangeInfo,
    NameResolveResponse,
    NameSetRequest,
    NameSetResponse,
    NameStatusResponse,
)

router = APIRouter(tags=["names"])


class AliasSetRequest(BaseModel):
    artifact_id: str
    version: int


@router.put("/v1/names/{name:path}/aliases/{alias}")
async def set_alias(
    name: str,
    alias: str,
    request: AliasSetRequest,
    store: WriteStore,
    principal: CurrentPrincipal,
):
    """Point ``name @ alias`` (e.g. champion) at an artifact version.

    Protected aliases (``registry_protected_aliases`` config) do not apply
    immediately: the change lands in the pending queue (202) and an
    explicit approve applies it.
    """
    from strata.server import get_state

    state = get_state()
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    if alias in state.config.registry_protected_aliases:
        try:
            queued = store.request_alias_change(
                name,
                alias,
                "set",
                artifact_id=request.artifact_id,
                version=request.version,
                tenant=tenant_id,
                actor=actor,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not queued:
            # Alias already points at exactly this version — nothing to
            # approve. Idempotent promote cells re-run without refiling.
            return {
                "status": "unchanged",
                "name": name,
                "alias": alias,
                "artifact_uri": f"strata://artifact/{request.artifact_id}@v={request.version}",
            }
        return JSONResponse(
            status_code=202,
            content={
                "status": "pending",
                "name": name,
                "alias": alias,
                "detail": f"Alias '{alias}' is protected — the change awaits approval "
                "(POST /v1/registry/pending/approve).",
            },
        )

    try:
        changed = store.set_alias(
            name, alias, request.artifact_id, request.version, tenant=tenant_id, actor=actor
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "status": "applied" if changed else "unchanged",
        "name": name,
        "alias": alias,
        "artifact_uri": f"strata://artifact/{request.artifact_id}@v={request.version}",
    }


@router.get("/v1/names/{name:path}/aliases/{alias}")
async def resolve_alias(name: str, alias: str, store: ReadStore, principal: CurrentPrincipal):
    """Resolve ``name @ alias`` to its artifact version."""
    tenant_id = principal.tenant if principal else None

    artifact = store.resolve_alias(name, alias, tenant=tenant_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Alias '{name}@{alias}' not found")
    return {
        "name": name,
        "alias": alias,
        "artifact_uri": f"strata://artifact/{artifact.id}@v={artifact.version}",
        "artifact_id": artifact.id,
        "version": artifact.version,
        "state": artifact.state,
    }


@router.delete("/v1/names/{name:path}/aliases/{alias}")
async def delete_alias(
    name: str, alias: str, store: PersonalModeStore, principal: CurrentPrincipal
):
    """Delete ``name @ alias`` (pending queue for protected aliases)."""
    from strata.server import get_state

    state = get_state()
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    if alias in state.config.registry_protected_aliases:
        if store.resolve_alias(name, alias, tenant=tenant_id) is None:
            raise HTTPException(status_code=404, detail=f"Alias '{name}@{alias}' not found")
        store.request_alias_change(name, alias, "delete", tenant=tenant_id, actor=actor)
        return JSONResponse(
            status_code=202,
            content={"status": "pending", "name": name, "alias": alias},
        )

    if not store.delete_alias(name, alias, tenant=tenant_id, actor=actor):
        raise HTTPException(status_code=404, detail=f"Alias '{name}@{alias}' not found")
    return {"status": "deleted", "name": name, "alias": alias}


@router.get("/v1/names/{name:path}/aliases")
async def list_aliases(name: str, store: ReadStore, principal: CurrentPrincipal):
    """List the aliases held by a name."""
    tenant_id = principal.tenant if principal else None

    aliases = store.list_aliases(name, tenant=tenant_id)
    return {
        "name": name,
        "aliases": [
            {
                "alias": a.alias,
                "artifact_id": a.artifact_id,
                "version": a.version,
                "updated_at": a.updated_at,
            }
            for a in aliases
        ],
    }


class TagSetRequest(BaseModel):
    key: str
    value: str


@router.put("/v1/artifacts/{artifact_id}/v/{version}/tags")
async def set_tag(
    artifact_id: str,
    version: int,
    request: TagSetRequest,
    store: WriteStore,
    principal: CurrentPrincipal,
):
    """Set a key/value tag on an artifact version."""
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    try:
        store.set_tag(
            artifact_id, version, request.key, request.value, tenant=tenant_id, actor=actor
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"artifact_id": artifact_id, "version": version, request.key: request.value}


@router.get("/v1/artifacts/{artifact_id}/v/{version}/tags")
async def get_tags(artifact_id: str, version: int, store: ReadStore, principal: CurrentPrincipal):
    """Get the tags on an artifact version."""
    tenant_id = principal.tenant if principal else None

    return {
        "artifact_id": artifact_id,
        "version": version,
        "tags": store.get_tags(artifact_id, version, tenant=tenant_id),
    }


@router.delete("/v1/artifacts/{artifact_id}/v/{version}/tags/{key}")
async def delete_tag(
    artifact_id: str, version: int, key: str, store: WriteStore, principal: CurrentPrincipal
):
    """Delete one tag from an artifact version."""
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    if not store.delete_tag(artifact_id, version, key, tenant=tenant_id, actor=actor):
        raise HTTPException(status_code=404, detail=f"Tag '{key}' not found")
    return {"status": "deleted", "artifact_id": artifact_id, "version": version, "key": key}


# NOTE: this greedy {name:path} route must stay registered AFTER the more
# specific "/v1/names/{name:path}/aliases/..." routes above, or ".../aliases/x"
# URLs would be swallowed as part of the name. The "/aliases" suffix is reserved.
@router.get("/v1/names/{name:path}", response_model=NameResolveResponse)
async def resolve_name(name: str, store: ReadStore, principal: CurrentPrincipal):
    """Resolve a name to its artifact.

    Args:
        name: Name to resolve (without strata://name/ prefix)

    Returns:
        NameResolveResponse with resolved artifact URI
    """
    # Get tenant from auth context for name isolation
    tenant_id = principal.tenant if principal else None

    name_info = store.get_name(name, tenant=tenant_id)
    if name_info is None:
        raise HTTPException(status_code=404, detail=f"Name '{name}' not found")

    artifact_uri = f"strata://artifact/{name_info.artifact_id}@v={name_info.version}"

    return NameResolveResponse(
        artifact_uri=artifact_uri,
        version=name_info.version,
        updated_at=name_info.updated_at,
    )


@router.post("/v1/names", response_model=NameSetResponse)
async def set_name(request: NameSetRequest, store: WriteStore, principal: CurrentPrincipal):
    """Set or update a name pointer.

    Args:
        request: NameSetRequest with name, artifact_id, and version

    Returns:
        NameSetResponse with name and artifact URIs
    """
    # Get tenant + actor from auth context for name isolation and audit
    # attribution (who published this name).
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    try:
        store.set_name(
            request.name,
            request.artifact_id,
            request.version,
            tenant=tenant_id,
            actor=actor,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name_uri = f"strata://name/{request.name}"
    artifact_uri = f"strata://artifact/{request.artifact_id}@v={request.version}"

    return NameSetResponse(
        name_uri=name_uri,
        artifact_uri=artifact_uri,
    )


@router.delete("/v1/names/{name:path}")
async def delete_name(name: str, store: PersonalModeStore, principal: CurrentPrincipal):
    """Delete a name pointer.

    Args:
        name: Name to delete

    Returns:
        Success status
    """
    # Get tenant from auth context for name isolation
    tenant_id = principal.tenant if principal else None

    if not store.delete_name(name, tenant=tenant_id):
        raise HTTPException(status_code=404, detail=f"Name '{name}' not found")

    return {"status": "deleted", "name": name}


@router.get("/v1/names")
async def list_names(store: ReadStore, principal: CurrentPrincipal):
    """List all name pointers.

    Returns:
        List of name entries with their artifact mappings
    """
    # Get tenant from auth context for name isolation
    tenant_id = principal.tenant if principal else None

    names = store.list_names(tenant=tenant_id)
    return {
        "names": [
            {
                "name": n.name,
                "artifact_uri": f"strata://artifact/{n.artifact_id}@v={n.version}",
                "updated_at": n.updated_at,
            }
            for n in names
        ]
    }


@router.get("/v1/artifacts/names/{name:path}/status", response_model=NameStatusResponse)
async def get_name_status(name: str, store: ReadStore, principal: CurrentPrincipal):
    """Get status of a named artifact including staleness info.

    Returns the current state of a named artifact and checks whether any of its
    input dependencies have newer versions available. This is useful for:
    - Determining if an artifact needs to be rebuilt
    - Understanding which specific inputs have changed
    - Debugging dependency chains

    Args:
        name: Name to check status for

    Returns:
        NameStatusResponse with staleness information
    """
    from strata.api.dependencies import resolve_input_version

    # Get tenant from auth context for name isolation
    tenant_id = principal.tenant if principal else None

    # Get name status from store (includes input_versions)
    status = store.get_name_status(name, tenant=tenant_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Name '{name}' not found")

    # Check for staleness by comparing stored vs current input versions
    changed_inputs: list[InputChangeInfo] = []
    for input_uri, old_version in status.input_versions.items():
        try:
            current_version = resolve_input_version(input_uri, tenant=tenant_id)
            if current_version != old_version:
                changed_inputs.append(
                    InputChangeInfo(
                        input_uri=input_uri,
                        old_version=old_version,
                        new_version=current_version,
                    )
                )
        except HTTPException:
            # Input no longer exists or is inaccessible - treat as changed
            changed_inputs.append(
                InputChangeInfo(
                    input_uri=input_uri,
                    old_version=old_version,
                    new_version="<unavailable>",
                )
            )

    # Build staleness reason
    is_stale = len(changed_inputs) > 0
    stale_reason = None
    if is_stale:
        changes = [f"{c.input_uri}: {c.old_version} → {c.new_version}" for c in changed_inputs]
        stale_reason = f"Rebuild needed: {', '.join(changes)}"

    return NameStatusResponse(
        name=status.name,
        artifact_uri=status.artifact_uri,
        artifact_id=status.artifact_id,
        version=status.version,
        state=status.state,
        updated_at=status.updated_at,
        input_versions=status.input_versions,
        is_stale=is_stale,
        stale_reason=stale_reason,
        changed_inputs=changed_inputs if changed_inputs else None,
    )

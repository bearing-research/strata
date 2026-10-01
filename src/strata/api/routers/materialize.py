"""Materialize-plane routes whose gates live entirely in the dependency/service layer.

The stateful materialize/streams handlers stay in ``server.py`` until the
stream-state, QoS and build runtime is extracted.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from strata.api.dependencies import CurrentPrincipal, ReadStore, resolve_input_version
from strata.types import ExplainMaterializeRequest, ExplainMaterializeResponse

router = APIRouter(tags=["materialize"])


@router.post("/v1/artifacts/explain-materialize", response_model=ExplainMaterializeResponse)
async def explain_materialize(
    request: ExplainMaterializeRequest, store: ReadStore, principal: CurrentPrincipal
):
    """Dry-run materialize: report hit or miss and why a rebuild would be needed."""
    from strata.services.materialize import materialize_service

    tenant_id = principal.tenant if principal else None

    # A per-input 400/404 becomes an error marker so the dry run still returns
    # a full picture.
    resolved_versions: dict[str, str] = {}
    for input_uri in request.inputs:
        try:
            resolved_versions[input_uri] = resolve_input_version(input_uri, tenant=tenant_id)
        except HTTPException as e:
            resolved_versions[input_uri] = f"<error: {e.detail}>"

    return materialize_service.explain(
        store, request=request, tenant=tenant_id, resolved_versions=resolved_versions
    )

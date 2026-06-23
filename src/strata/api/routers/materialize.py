"""Materialize-plane routes.

Holds the routes whose gates are fully lifted into the dependency/service layer
(#295). ``explain-materialize`` is pure dependency + service today; the stateful
materialize/streams handlers (``unified_materialize`` / ``materialize_artifact``
/ ``get_stream``) stay in ``server.py`` for now — they are entangled with the
live stream-state registry, two-tier QoS admission, and the background build /
prefetch runtime, which is a separate extraction (see
``docs/internal/design-server-decomposition.md`` phase 3). They join this router
once that runtime is lifted.
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
    """Explain what materialize would do without actually doing it (dry run).

    This endpoint is useful for:
    - Checking if a computation would be a cache hit or miss
    - Understanding why a rebuild is needed
    - Debugging provenance and staleness issues
    - Scripts that want to print "Rebuild needed: raw_q1 moved from v12 → v13"

    Args:
        request: ExplainMaterializeRequest with inputs, transform, and optional name

    Returns:
        ExplainMaterializeResponse explaining what would happen
    """
    from strata.services.materialize import materialize_service

    # Get tenant from auth context for artifact isolation.
    tenant_id = principal.tenant if principal else None

    # Resolve current input versions (may 400/404 per input — captured as an
    # error marker so the dry run still returns a full picture). The pure
    # provenance / cache-hit / staleness logic lives in MaterializeService.
    resolved_versions: dict[str, str] = {}
    for input_uri in request.inputs:
        try:
            resolved_versions[input_uri] = resolve_input_version(input_uri, tenant=tenant_id)
        except HTTPException as e:
            resolved_versions[input_uri] = f"<error: {e.detail}>"

    return materialize_service.explain(
        store, request=request, tenant=tenant_id, resolved_versions=resolved_versions
    )

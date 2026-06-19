"""Build status + pull-model manifest routes.

Moved verbatim from ``server.py`` (P3 / A3a, router split). The build handlers
are thin shells over build-store lookup + post-fetch authz; the gate/transport
helpers stay centralized in ``server.py`` and are reached via in-body lazy
import (``get_state``, ``StreamState``, ``_authorize_build_access``,
``_identity_build_status``, ``_build_transport_available``,
``_get_runtime_build_store``, ``_get_artifact_store``, ``_ACTIVE_BUILD_STATES``).
The pure manifest assembly already lives in ``BuildService.assemble_manifest``.
The signed download/upload/finalize transport routes are a separate slice.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from strata.services.build import build_service
from strata.types import BuildStatusResponse

router = APIRouter(tags=["builds"])


@router.get("/v1/artifacts/builds/{build_id}", response_model=BuildStatusResponse)
async def get_build_status(build_id: str):
    """Get async build status.

    Use this endpoint to poll the status of a build that was started
    asynchronously via materialize, including scan@v1 artifact-mode builds.

    Args:
        build_id: Build ID from materialize response

    Returns:
        BuildStatusResponse with current build state
    """
    from strata.server import (
        StreamState,
        _authorize_build_access,
        _build_transport_available,
        _get_runtime_build_store,
        _identity_build_status,
        get_state,
    )

    state = get_state()

    # Identity materialize artifact-mode builds are tracked in the stream registry
    # even when server-mode transforms are disabled.
    stream_state = state._streams.get(build_id)
    if isinstance(stream_state, StreamState):
        _authorize_build_access(
            owner_principal=stream_state.plan.owner_principal,
            owner_tenant=stream_state.plan.owner_tenant,
        )

        return _identity_build_status(stream_state)

    if not _build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build polling is only available when personal-mode writes "
                "or server-mode transforms are enabled"
            ),
        )

    # Get build store
    build_store = _get_runtime_build_store()
    if build_store is None:
        raise HTTPException(
            status_code=500,
            detail="Build store not initialized",
        )

    # Look up build
    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    # Check access control if auth is enabled
    _authorize_build_access(
        owner_principal=build.principal_id,
        owner_tenant=build.tenant_id,
    )

    return BuildStatusResponse(
        build_id=build.build_id,
        artifact_id=build.artifact_id,
        version=build.version,
        state=build.state,
        artifact_uri=f"strata://artifact/{build.artifact_id}@v={build.version}",
        executor_ref=build.executor_ref,
        created_at=build.created_at,
        started_at=build.started_at,
        completed_at=build.completed_at,
        error_message=build.error_message,
        error_code=build.error_code,
    )


@router.get("/v1/builds/{build_id}", response_model=BuildStatusResponse, include_in_schema=False)
async def get_build_status_compat(build_id: str):
    """Compatibility alias for older clients polling build status."""
    return await get_build_status(build_id)


@router.get("/v1/builds/{build_id}/manifest")
async def get_build_manifest(build_id: str, request: Request):
    """Get build manifest with signed URLs for pull-model execution.

    This endpoint returns a manifest containing:
    - Signed download URLs for each input artifact
    - Signed upload URL for the output
    - Signed finalize URL to call after upload completes

    Executors use this manifest to:
    1. Pull inputs directly from Strata storage
    2. Execute the transform
    3. Push output directly to Strata storage
    4. Call finalize to mark the build complete

    Args:
        build_id: Build ID from materialize response

    Returns:
        BuildManifest with all signed URLs
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _authorize_build_access,
        _build_transport_available,
        _get_artifact_store,
        _get_runtime_build_store,
        get_state,
    )

    state = get_state()

    if not _build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build manifest is only available when personal-mode writes "
                "or server-mode transforms are enabled"
            ),
        )

    build_store = _get_runtime_build_store()
    if build_store is None:
        raise HTTPException(status_code=500, detail="Build store not initialized")

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    # Only allow manifest retrieval for pending/running builds
    if build.state not in _ACTIVE_BUILD_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Build is not in pending or building state (state={build.state})",
        )

    # Access control
    _authorize_build_access(
        owner_principal=build.principal_id,
        owner_tenant=build.tenant_id,
    )

    # The server-mode store gate here never 403s — _build_transport_available()
    # above already guaranteed it — so this is just the store handle.
    store = _get_artifact_store(allow_server_mode=True)
    base_url = str(request.base_url).rstrip("/")

    try:
        return build_service.assemble_manifest(
            store,
            build=build,
            base_url=base_url,
            max_output_bytes=state.config.max_transform_output_bytes,
            url_expiry_seconds=state.config.signed_url_expiry_seconds,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

"""Build status + pull-model manifest routes.

Moved verbatim from ``server.py`` (P3 / A3a, router split). The build handlers
are thin shells over build-store lookup + post-fetch authz. The build-store /
transport gate lives in ``strata.api.dependencies`` (#295): the manifest +
finalize routes take the ``BuildTransportStore`` param dependency (404 if
transport is off, else the resolved store); ``get_build_status`` and the
signature-authed upload route resolve it in-body via ``build_transport_available``
/ ``runtime_build_store`` to preserve their exact gate ordering. The remaining
server-owned collaborators (``get_state``, ``_authorize_build_access``,
``_identity_build_status``, ``_get_artifact_store``, ``_ACTIVE_BUILD_STATES``)
are still reached via in-body lazy import; ``StreamState`` comes from
``strata.streaming`` and ``record_build_output_bytes`` from
``strata.transforms.build_qos`` (#302). The pure manifest assembly lives in
``BuildService.assemble_manifest``.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from strata.api.dependencies import (
    BuildTransportStore,
    build_transport_available,
    runtime_build_store,
)
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
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
        _authorize_build_access,
        _identity_build_status,
        get_state,
    )
    from strata.streaming import StreamState

    state = get_state()

    # Identity materialize artifact-mode builds are tracked in the stream registry
    # even when server-mode transforms are disabled.
    stream_state = state.streams.get(build_id)
    if isinstance(stream_state, StreamState):
        _authorize_build_access(
            owner_principal=stream_state.plan.owner_principal,
            owner_tenant=stream_state.plan.owner_tenant,
        )

        return _identity_build_status(stream_state)

    # In-body gate (not the BuildTransportStore param dependency): the identity
    # StreamState path above must answer even when transport is off, so the 404
    # can't run as a blanket param.
    if not build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build polling is only available when personal-mode writes "
                "or server-mode transforms are enabled"
            ),
        )

    # Get build store
    build_store = runtime_build_store()
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
async def get_build_manifest(build_id: str, request: Request, build_store: BuildTransportStore):
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
        _get_artifact_store,
        get_state,
    )

    state = get_state()

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
            signer=state.url_signer,
            build=build,
            base_url=base_url,
            max_output_bytes=state.config.max_transform_output_bytes,
            url_expiry_seconds=state.config.signed_url_expiry_seconds,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/v1/artifacts/download")
async def download_artifact_signed(
    artifact_id: str,
    version: str,
    build_id: str,
    expires_at: str,
    signature: str,
):
    """Download artifact blob using a signed URL.

    This endpoint is called by executors to pull input artifacts.
    The URL must be signed by Strata and not expired.

    Query Parameters:
        artifact_id: Artifact ID to download
        version: Version number
        build_id: Build ID this download is for (audit trail)
        expires_at: URL expiry timestamp (Unix epoch)
        signature: HMAC-SHA256 signature

    Returns:
        Arrow IPC stream bytes
    """
    from strata.server import _get_artifact_store, get_state

    # Parse and verify signature
    try:
        version_int = int(version)
        expires_at_float = float(expires_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter format")

    if not get_state().url_signer.verify_download_signature(
        artifact_id=artifact_id,
        version=version_int,
        build_id=build_id,
        expires_at=expires_at_float,
        signature=signature,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Get artifact blob
    store = _get_artifact_store(allow_server_mode=True)
    artifact = store.get_artifact(artifact_id, version_int)

    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if artifact.state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not ready (state={artifact.state})",
        )

    reader_cm = await asyncio.to_thread(store.open_blob_reader, artifact_id, version_int)
    if reader_cm is None:
        raise HTTPException(status_code=404, detail="Artifact blob not found")

    byte_size = artifact.byte_size or 0

    def _iter_signed_blob():
        with reader_cm as f:
            while True:
                chunk = f.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk

    headers = {
        "Content-Disposition": f'attachment; filename="{artifact_id}_v{version_int}.arrow"',
    }
    if byte_size:
        headers["Content-Length"] = str(byte_size)

    return StreamingResponse(
        _iter_signed_blob(),
        media_type="application/vnd.apache.arrow.stream",
        headers=headers,
    )


@router.post("/v1/artifacts/upload")
async def upload_artifact_signed(
    build_id: str,
    max_bytes: str,
    expires_at: str,
    signature: str,
    request: Request,
):
    """Upload artifact blob using a signed URL.

    This endpoint is called by executors to push output artifacts.
    The URL must be signed by Strata and not expired.
    The upload size must not exceed max_bytes.

    Query Parameters:
        build_id: Build ID this upload is for
        max_bytes: Maximum allowed upload size
        expires_at: URL expiry timestamp (Unix epoch)
        signature: HMAC-SHA256 signature

    Body:
        Raw Arrow IPC stream bytes

    Returns:
        Upload status
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _get_artifact_store,
        get_state,
    )

    # Parse and verify signature
    try:
        max_bytes_int = int(max_bytes)
        expires_at_float = float(expires_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter format")

    if not get_state().url_signer.verify_upload_signature(
        build_id=build_id,
        max_bytes=max_bytes_int,
        expires_at=expires_at_float,
        signature=signature,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Check build exists and is in correct state. Resolved in-body (not via the
    # RequiredBuildStore param) so the signature check above runs first — the
    # signature is the authorization, it must precede the store-500.
    build_store = runtime_build_store()
    if build_store is None:
        raise HTTPException(status_code=500, detail="Build store not initialized")

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    if build.state not in _ACTIVE_BUILD_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Build is not in pending or building state (state={build.state})",
        )

    store = _get_artifact_store(allow_server_mode=True)
    byte_size = 0
    fd, tmp_name = tempfile.mkstemp(prefix="strata_upload_", suffix=".tmp")
    staged = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as dst:
            async for chunk in request.stream():
                if not chunk:
                    continue
                byte_size += len(chunk)
                if byte_size > max_bytes_int:
                    raise HTTPException(
                        status_code=413,
                        detail=(f"Upload exceeds maximum size: {byte_size} > {max_bytes_int}"),
                    )
                dst.write(chunk)
        if byte_size == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        await asyncio.to_thread(
            store.publish_blob_from_path, build.artifact_id, build.version, staged
        )
    finally:
        staged.unlink(missing_ok=True)

    return {"status": "uploaded", "build_id": build_id, "byte_size": byte_size}


@router.post("/v1/builds/{build_id}/finalize")
async def finalize_build(
    build_id: str,
    request: Request,
    build_store: BuildTransportStore,
    expires_at: str | None = None,
    signature: str | None = None,
):
    """Finalize a build after upload (pull-model execution).

    Called by executors after uploading the output artifact.
    This endpoint:
    1. Verifies the blob was uploaded
    2. Reads Arrow metadata (schema, row count)
    3. Finalizes the artifact
    4. Marks the build as complete
    5. Optionally sets the name pointer

    Args:
        build_id: Build ID to finalize

    Body (JSON):
        Optional fields for metadata the executor provides

    Returns:
        Finalize status with artifact URI
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _authorize_build_access,
        _get_artifact_store,
        get_state,
    )
    from strata.transforms.build_qos import record_build_output_bytes

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    if signature is not None or expires_at is not None:
        if signature is None or expires_at is None:
            raise HTTPException(status_code=400, detail="Missing finalize signature parameters")
        try:
            expires_at_float = float(expires_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid parameter format")

        if not get_state().url_signer.verify_finalize_signature(
            build_id=build_id,
            expires_at=expires_at_float,
            signature=signature,
        ):
            raise HTTPException(status_code=403, detail="Invalid or expired signature")
    else:
        _authorize_build_access(
            owner_principal=build.principal_id,
            owner_tenant=build.tenant_id,
        )

    # Check build state
    if build.state not in _ACTIVE_BUILD_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Build is not in pending or building state (state={build.state})",
        )

    try:
        finalize_payload = await request.json()
        if not isinstance(finalize_payload, dict):
            finalize_payload = {}
    except Exception:
        finalize_payload = {}

    output_format = str(finalize_payload.get("output_format", "")).strip()

    # Verify blob was uploaded
    store = _get_artifact_store(allow_server_mode=True)
    if not store.blob_exists(build.artifact_id, build.version):
        raise HTTPException(
            status_code=400,
            detail="Blob not uploaded. Upload using the signed URL first.",
        )

    byte_size = store.blob_size(build.artifact_id, build.version) or 0
    if byte_size == 0:
        raise HTTPException(status_code=500, detail="Failed to read uploaded blob")

    if output_format == "notebook-output-bundle@v1":

        def _validate_notebook_bundle() -> tuple[str, int]:
            from strata.notebook.remote_bundle import (
                read_notebook_output_bundle_manifest_path,
            )

            reader_cm = store.open_blob_reader(build.artifact_id, build.version)
            if reader_cm is None:
                raise RuntimeError("Uploaded blob disappeared before validation")
            fd, staged_name = tempfile.mkstemp(prefix="strata_bundle_validate_", suffix=".tar")
            staged_path = Path(staged_name)
            try:
                with os.fdopen(fd, "wb") as dst, reader_cm as src:
                    while True:
                        chunk = src.read(BLOB_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        dst.write(chunk)
                read_notebook_output_bundle_manifest_path(staged_path)
            finally:
                staged_path.unlink(missing_ok=True)
            return "", 0

        try:
            schema_json, row_count = await asyncio.to_thread(_validate_notebook_bundle)
        except Exception as e:
            build_store.fail_build(build_id, str(e), "INVALID_NOTEBOOK_BUNDLE")
            store.fail_artifact(build.artifact_id, build.version)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid notebook output bundle: {e}",
            )
    else:

        def _parse_arrow_stream() -> tuple[str, int]:
            import pyarrow.ipc as arrow_ipc

            reader_cm = store.open_blob_reader(build.artifact_id, build.version)
            if reader_cm is None:
                raise RuntimeError("Uploaded blob disappeared before validation")
            row_count_inner = 0
            with reader_cm as blob_handle:
                ipc_reader = arrow_ipc.open_stream(blob_handle)
                schema = ipc_reader.schema
                for batch in ipc_reader:
                    row_count_inner += batch.num_rows
            return schema.to_string(), row_count_inner

        try:
            schema_json, row_count = await asyncio.to_thread(_parse_arrow_stream)
        except Exception as e:
            build_store.fail_build(build_id, str(e), "INVALID_ARROW_FORMAT")
            store.fail_artifact(build.artifact_id, build.version)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Arrow IPC format: {e}",
            )

    # Finalize the artifact atomically with name if provided
    try:
        finalized_artifact = store.finalize_and_set_name(
            artifact_id=build.artifact_id,
            version=build.version,
            schema_json=schema_json,
            row_count=row_count,
            byte_size=byte_size,
            name=build.name,
            tenant=build.tenant,
        )
    except ValueError as e:
        build_store.fail_build(build_id, str(e), "FINALIZE_FAILED")
        store.fail_artifact(build.artifact_id, build.version)
        raise HTTPException(status_code=400, detail=str(e))
    if finalized_artifact is None:
        build_store.fail_build(build_id, "Failed to finalize artifact", "FINALIZE_FAILED")
        store.fail_artifact(build.artifact_id, build.version)
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    # Mark build as complete
    # First start the build if it's still pending (pull model may finalize directly)
    if build.state == "pending":
        build_store.start_build(build_id)
    if finalized_artifact.id != build.artifact_id or finalized_artifact.version != build.version:
        build_store.update_build_output(
            build_id,
            finalized_artifact.id,
            finalized_artifact.version,
        )
    build_store.complete_build(build_id)
    await record_build_output_bytes(build.tenant_id, byte_size)

    artifact_uri = f"strata://artifact/{finalized_artifact.id}@v={finalized_artifact.version}"
    name_uri = f"strata://name/{build.name}" if build.name else None

    return {
        "status": "finalized",
        "build_id": build_id,
        "artifact_uri": artifact_uri,
        "name_uri": name_uri,
        "byte_size": byte_size,
        "row_count": row_count,
    }

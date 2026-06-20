"""Artifact CRUD routes: put, info, data, list, delete, gc, stats, usage.

Moved verbatim from ``server.py`` (P3 / A1, router split). These handlers are
thin: they take an already-gated store + tenant filter via the typed
dependencies and shape the response. The post-fetch ACL helpers
(``_ensure_artifact_access``, ``_authorize_artifact_read``) stay in ``server.py``
— ``_authorize_artifact_read`` re-checks the shared table ACL and both are used
by other (still-resident) routes — so the handlers lazy-import them in-body.
Names / aliases / tags and the materialize/build/transport routes are separate
slices and stay put.
"""

from __future__ import annotations

import asyncio
import re
import uuid

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    PersonalModeStore,
    ReadStore,
    WriteStore,
)
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.logging import get_logger
from strata.services.artifact import artifact_service
from strata.types import (
    ArtifactDependentsResponse,
    ArtifactInfoResponse,
    ArtifactLineageResponse,
    PutArtifactResponse,
)

logger = get_logger(__name__)

router = APIRouter(tags=["artifacts"])


@router.put("/v1/artifacts", response_model=PutArtifactResponse)
async def put_artifact(request: Request, store: WriteStore, principal: CurrentPrincipal):
    """Directly upload and persist an artifact with provenance tracking.

    This is a simplified API for clients that execute transforms locally
    and want to persist the result with full provenance tracking and deduplication.

    Accepts two content types:
    1. application/json: JSON body with inputs, transform, data, name
    2. multipart/form-data: metadata (JSON) + data (Arrow IPC bytes)

    The multipart format is more efficient for large data or pre-serialized Arrow.

    Deduplication: If an artifact with the same provenance hash already exists,
    returns the existing artifact (hit=True) without storing duplicate data.

    Returns:
        PutArtifactResponse with artifact URI and cache hit status
    """
    import json as json_module

    content_type = request.headers.get("content-type", "")

    # Parse request based on content type
    if "multipart/form-data" in content_type:
        # Multipart: metadata JSON + Arrow IPC data
        form = await request.form()

        # Get metadata
        metadata_file = form.get("metadata")
        if metadata_file is None:
            raise HTTPException(
                status_code=400,
                detail="Missing 'metadata' field in multipart request",
            )
        if isinstance(metadata_file, str):
            raise HTTPException(
                status_code=400,
                detail="'metadata' field must be a file, not form data",
            )
        metadata_content = await metadata_file.read()
        try:
            metadata = json_module.loads(metadata_content)
        except json_module.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {e}")

        inputs = metadata.get("inputs", [])
        transform_dict = metadata.get("transform", {})
        artifact_name = metadata.get("name")

        # Get Arrow data
        data_file = form.get("data")
        if data_file is None:
            raise HTTPException(status_code=400, detail="Missing 'data' field in multipart request")
        if isinstance(data_file, str):
            raise HTTPException(
                status_code=400,
                detail="'data' field must be a file, not form data",
            )
        arrow_bytes = await data_file.read()

        # Parse Arrow to get schema and row count. The buffer must be exactly
        # one IPC stream — trailing bytes (concatenated streams) would be
        # silently dropped by every standard reader downstream (#123).
        try:
            buf = pa.BufferReader(arrow_bytes)
            reader = ipc.open_stream(buf)
            table = reader.read_all()
            if buf.tell() != len(arrow_bytes):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid Arrow IPC data: {len(arrow_bytes) - buf.tell()} trailing "
                        "bytes after stream end (concatenated streams?)"
                    ),
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Arrow IPC data: {e}")

    else:
        # JSON body (legacy format)
        try:
            body = await request.json()
        except json_module.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        inputs = body.get("inputs", [])
        transform_dict = body.get("transform", {})
        artifact_name = body.get("name")
        data = body.get("data")

        if data is None:
            raise HTTPException(status_code=400, detail="Missing 'data' field")

        # Convert JSON data to Arrow
        try:
            if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                # Columnar data - convert directly
                table = pa.Table.from_pydict(data)
            else:
                # Non-columnar - store as single JSON column
                json_str = json_module.dumps(data)
                table = pa.Table.from_pydict({"data": [json_str]})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to convert data to Arrow: {e}")

        # Serialize to Arrow IPC
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        arrow_bytes = sink.getvalue().to_pybytes()

    # Validate transform
    executor = transform_dict.get("executor")
    if not executor:
        raise HTTPException(status_code=400, detail="Missing 'executor' in transform")
    params = transform_dict.get("params", {})

    # Resolve tenant the same way materialize does (principal-based) —
    # get_tenant_id() defaults to "_default", which stranded put-created
    # artifacts in a tenant the name routes (None) could never address.
    tenant_id = principal.tenant if principal else None

    # Resolve input versions for provenance
    input_versions: dict[str, str] = {}
    for input_uri in inputs:
        try:
            # Try to resolve artifact URIs
            if input_uri.startswith("strata://artifact/"):
                match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
                if match:
                    input_versions[input_uri] = f"{match.group(1)}@v={match.group(2)}"
                else:
                    input_versions[input_uri] = input_uri
            elif input_uri.startswith("strata://name/"):
                name = input_uri[len("strata://name/") :]
                resolved = store.resolve_name(name, tenant=tenant_id)
                if resolved:
                    input_versions[input_uri] = f"{resolved.id}@v={resolved.version}"
                else:
                    input_versions[input_uri] = input_uri
            else:
                # Table URI or unknown - use as-is
                input_versions[input_uri] = input_uri
        except Exception:
            # Fallback: use URI as version
            input_versions[input_uri] = input_uri

    # Compute provenance hash
    from strata.artifact_store import TransformSpec as ArtifactTransformSpec
    from strata.artifact_store import compute_provenance_hash

    # Convert to internal TransformSpec
    artifact_transform = ArtifactTransformSpec(
        executor=executor,
        params=params,
        inputs=inputs,
    )

    input_hashes = [f"{uri}:{ver}" for uri, ver in sorted(input_versions.items())]
    provenance_hash = compute_provenance_hash(input_hashes, artifact_transform)

    # Check for existing artifact with same provenance
    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None and existing.state == "ready":
        artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"
        name_uri = None

        # Set name if requested
        if artifact_name:
            try:
                store.set_name(artifact_name, existing.id, existing.version, tenant=tenant_id)
                name_uri = f"strata://name/{artifact_name}"
            except ValueError:
                pass

        return PutArtifactResponse(
            artifact_uri=artifact_uri,
            hit=True,
            byte_size=existing.byte_size or 0,
            name_uri=name_uri,
        )

    # Create artifact
    artifact_id = str(uuid.uuid4())
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=artifact_transform,
        input_versions=input_versions,
        tenant=tenant_id,
    )

    # Write blob
    store.write_blob(artifact_id, version, arrow_bytes)

    # Finalize
    schema_json = table.schema.to_string()
    finalized_artifact = store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json=schema_json,
        row_count=table.num_rows,
        byte_size=len(arrow_bytes),
    )
    if finalized_artifact is None:
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    artifact_uri = f"strata://artifact/{finalized_artifact.id}@v={finalized_artifact.version}"
    name_uri = None

    # Set name if requested
    if artifact_name:
        try:
            store.set_name(
                artifact_name,
                finalized_artifact.id,
                finalized_artifact.version,
                tenant=tenant_id,
            )
            name_uri = f"strata://name/{artifact_name}"
        except Exception as e:
            logger.warning(f"Failed to set name {artifact_name}: {e}")

    return PutArtifactResponse(
        artifact_uri=artifact_uri,
        hit=finalized_artifact.id != artifact_id or finalized_artifact.version != version,
        byte_size=finalized_artifact.byte_size or len(arrow_bytes),
        name_uri=name_uri,
    )


@router.get("/v1/artifacts/{artifact_id}/v/{version}", response_model=ArtifactInfoResponse)
async def get_artifact_info(
    artifact_id: str, version: int, store: ReadStore, tenant_filter: CurrentTenant
):
    """Get artifact metadata.

    Available in service mode (a client needs to poll state/schema of a result),
    gated by tenant + the table ACL of the artifact's inputs.

    Args:
        artifact_id: Artifact ID
        version: Version number

    Returns:
        ArtifactInfoResponse with artifact metadata
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    _authorize_artifact_read(artifact)

    return ArtifactInfoResponse(
        artifact_id=artifact.id,
        version=artifact.version,
        state=artifact.state,
        arrow_schema=artifact.schema_json,
        row_count=artifact.row_count,
        byte_size=artifact.byte_size,
        created_at=artifact.created_at or 0,
    )


@router.get("/v1/artifacts/stats")
async def get_artifact_stats(store: PersonalModeStore, tenant_filter: CurrentTenant):
    """Get artifact store statistics (personal mode only).

    Returns:
        Artifact store statistics
    """
    return store.stats(tenant=tenant_filter)


@router.get("/v1/artifacts/usage")
async def get_artifact_usage(store: PersonalModeStore, tenant_filter: CurrentTenant):
    """Get artifact store usage metrics (personal mode only).

    Returns comprehensive usage statistics including:
    - Total bytes used
    - Number of artifacts and versions
    - Unreferenced artifact count (candidates for GC)

    Returns:
        Usage metrics dictionary
    """
    return store.get_usage(tenant=tenant_filter)


@router.get("/v1/artifacts")
async def list_artifacts(
    store: PersonalModeStore,
    tenant_filter: CurrentTenant,
    limit: int = 100,
    offset: int = 0,
    state: str | None = None,
    name_prefix: str | None = None,
):
    """List artifacts with optional filtering (personal mode only).

    Args:
        limit: Maximum number of artifacts to return (default 100)
        offset: Number of artifacts to skip for pagination
        state: Filter by state ("ready", "building", "failed")
        name_prefix: Filter by artifacts with names starting with prefix

    Returns:
        List of artifact versions with their metadata
    """
    if state is not None and state not in ("ready", "building", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state filter: {state}. Must be 'ready', 'building', or 'failed'",
        )

    artifacts = store.list_artifacts(
        limit=limit,
        offset=offset,
        state=state,
        name_prefix=name_prefix,
        tenant=tenant_filter,
    )

    return {
        "artifacts": [
            {
                "artifact_uri": f"strata://artifact/{a.id}@v={a.version}",
                "artifact_id": a.id,
                "version": a.version,
                "state": a.state,
                "row_count": a.row_count,
                "byte_size": a.byte_size,
                "created_at": a.created_at,
            }
            for a in artifacts
        ],
        "limit": limit,
        "offset": offset,
    }


@router.delete("/v1/artifacts/{artifact_id}/v/{version}")
async def delete_artifact(
    artifact_id: str, version: int, store: PersonalModeStore, tenant_filter: CurrentTenant
):
    """Delete an artifact version (personal mode only).

    Deletes the artifact blob and metadata. Also removes any name pointers
    that reference this specific version.

    Args:
        artifact_id: Artifact ID
        version: Version number

    Returns:
        Success status
    """
    from strata.server import _ensure_artifact_access

    _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )

    deleted = store.delete_artifact(artifact_id, version, tenant=tenant_filter)
    if not deleted:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return {"deleted": True, "artifact_uri": f"strata://artifact/{artifact_id}@v={version}"}


@router.post("/v1/artifacts/gc")
async def garbage_collect_artifacts(
    store: PersonalModeStore,
    tenant_filter: CurrentTenant,
    max_age_days: float = 7.0,
):
    """Garbage collect unreferenced artifacts (personal mode only).

    Deletes artifacts that:
    1. Have no name pointer referencing them
    2. Are older than max_age_days
    3. Are in "ready" or "failed" state

    This is safe to run periodically to clean up temporary artifacts
    that were never named or whose names were deleted.

    Args:
        max_age_days: Maximum age in days for unreferenced artifacts (default 7)

    Returns:
        GC statistics including deleted count and bytes freed
    """
    if max_age_days < 0:
        raise HTTPException(status_code=400, detail="max_age_days must be non-negative")

    result = store.garbage_collect(
        max_age_days=max_age_days,
        tenant=tenant_filter,
    )
    return result


@router.get("/v1/artifacts/{artifact_id}/v/{version}/data")
async def get_artifact_data(
    artifact_id: str, version: int, store: ReadStore, tenant_filter: CurrentTenant
):
    """Stream artifact data as Arrow IPC.

    Returns the raw Arrow IPC stream bytes for the artifact, so an identity-scan
    cache hit (or any materialized result) can be read back. Available in service
    mode, gated by tenant + the table ACL of the artifact's inputs.

    Args:
        artifact_id: Artifact ID
        version: Version number

    Returns:
        StreamingResponse with Arrow IPC data
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # Verify artifact exists and is ready (tenant scoping)
    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    # Result retrieval is ACL-gated: re-check the table ACL of the inputs.
    _authorize_artifact_read(artifact)
    if artifact.state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not ready (state={artifact.state})",
        )

    reader_cm = await asyncio.to_thread(store.open_blob_reader, artifact_id, version)
    if reader_cm is None:
        raise HTTPException(status_code=404, detail="Artifact data not found")

    def _iter_blob():
        with reader_cm as f:
            while True:
                chunk = f.read(BLOB_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                yield chunk

    # Note: We don't include schema in headers since it may contain newlines
    # Clients should read the schema from the Arrow IPC stream itself
    return StreamingResponse(
        _iter_blob(),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-Arrow-Row-Count": str(artifact.row_count or 0),
        },
    )


# ---------------------------------------------------------------------------
# Lineage and Dependency Introspection Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/v1/artifacts/{artifact_id}/v/{version}/lineage",
    response_model=ArtifactLineageResponse,
)
async def get_artifact_lineage(
    artifact_id: str,
    version: int,
    store: ReadStore,
    tenant_filter: CurrentTenant,
    max_depth: int = Query(default=10, ge=1, le=100),
):
    """Get the lineage (input dependency graph) for an artifact.

    Returns the full input dependency tree, showing all artifacts and tables
    that this artifact depends on, including transitive dependencies.

    This is useful for:
    - Understanding data provenance (what data went into this artifact)
    - Debugging computation graphs
    - Auditing data lineage for compliance

    Args:
        artifact_id: Artifact ID to get lineage for
        version: Version number
        max_depth: Maximum depth to traverse (default: 10, max: 100)

    Returns:
        ArtifactLineageResponse with nodes and edges representing the lineage graph
    """
    from strata.server import _ensure_artifact_access

    # Get the root artifact
    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )

    if artifact.state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not ready (state={artifact.state})",
        )

    return artifact_service.build_lineage(
        store,
        artifact=artifact,
        artifact_id=artifact_id,
        version=version,
        tenant_filter=tenant_filter,
        max_depth=max_depth,
    )


@router.get(
    "/v1/artifacts/{artifact_id}/v/{version}/dependents",
    response_model=ArtifactDependentsResponse,
)
async def get_artifact_dependents(
    artifact_id: str,
    version: int,
    store: ReadStore,
    tenant_filter: CurrentTenant,
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Get artifacts that depend on this artifact (reverse dependencies).

    Returns all artifacts that use this artifact as an input. This is useful for:
    - Impact analysis before modifying or deleting an artifact
    - Understanding downstream consumers
    - Planning cascading rebuilds

    Note: Only searches for direct dependents, not transitive dependents.
    Only returns ready artifacts.

    Args:
        artifact_id: Artifact ID to find dependents of
        version: Version number
        limit: Maximum number of dependents to return (default: 100, max: 1000)

    Returns:
        ArtifactDependentsResponse with list of dependent artifacts
    """
    from strata.server import _ensure_artifact_access

    # Verify the artifact exists
    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )

    if artifact.state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not ready (state={artifact.state})",
        )

    return artifact_service.build_dependents(
        store,
        artifact_id=artifact_id,
        version=version,
        tenant_filter=tenant_filter,
        limit=limit,
    )

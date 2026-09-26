"""Artifact CRUD + personal-mode upload/finalize routes.

Moved verbatim from ``server.py`` (P3 / A1, router split; upload/finalize added
in #295). These handlers are thin: they take an already-gated store + tenant
filter via the typed dependencies and shape the response. The personal-mode
upload/finalize pair takes ``PersonalModeStore`` (the bare write-mode gate); the
signature-authed signed-transport upload/finalize live in ``builds.py``. The
post-fetch ACL helpers (``_ensure_artifact_access``, ``_authorize_artifact_read``)
stay in ``server.py`` — ``_authorize_artifact_read`` re-checks the shared table
ACL and both are used by other (still-resident) routes — so the handlers
lazy-import them in-body. Names / aliases / tags and the stateful
materialize/streams routes are separate slices and stay put.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Annotated, NamedTuple

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as FastPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    PersonalModeStore,
    ReadStore,
    WriteStore,
    store_for_scope,
)
from strata.api.remote_registry import quoted, relay, remote_registry
from strata.artifact_store import ArtifactStore, reject_unsafe_artifact_id
from strata.artifact_transfer import PROMOTION_TAG
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.logging import get_logger
from strata.services.artifact import artifact_service
from strata.types import (
    PROVENANCE_MISS_HEADER,
    ArtifactDependentsResponse,
    ArtifactInfoResponse,
    ArtifactLineageResponse,
    ArtifactProvenanceMatchResponse,
    Principal,
    PutArtifactResponse,
    UploadFinalizeRequest,
    UploadFinalizeResponse,
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
        # A shared store hands you results you did not compute, so who did is
        # part of the result. The column and the store parameter have both
        # existed since the tenancy migration; this write site never passed it,
        # which left every published artifact anonymous — including under
        # `service_writes_enabled`, whose whole premise is attributed publish.
        principal=principal.id if principal else None,
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
        content_sha256=hashlib.sha256(arrow_bytes).hexdigest(),
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
        content_sha256=artifact.content_sha256,
        provenance_hash=artifact.provenance_hash,
        transform_spec=artifact.transform_spec,
    )


@router.put("/v1/artifacts/import/blobs/{content_sha256}", status_code=201)
async def stage_import_blob_route(
    request: Request,
    store: WriteStore,
    principal: CurrentPrincipal,
    content_sha256: str = FastPath(pattern="^[0-9a-f]{64}$"),
):
    """Upload the bytes of an artifact about to be imported, ahead of its record.

    The first of the import's two steps: the bytes here, then ``POST
    /v1/artifacts/import`` with the record, whose ``content_sha256`` names
    them. The body is streamed to disk and checked against the digest in the
    path before anything is kept, so a 201 means these are exactly the bytes
    that digest describes. Only the caller's tenant can import them.
    """
    tenant_id = principal.tenant if principal else None
    with tempfile.TemporaryDirectory(prefix="strata_import_") as workdir:
        staged = Path(workdir) / "blob"
        hasher = hashlib.sha256()
        byte_size = 0
        with open(staged, "wb") as out:
            async for chunk in request.stream():
                hasher.update(chunk)
                byte_size += len(chunk)
                out.write(chunk)
        received = hasher.hexdigest()
        if received != content_sha256:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded bytes do not match the digest in the path "
                    f"(path {content_sha256[:12]}…, received {received[:12]}…)"
                ),
            )
        await asyncio.to_thread(
            store.stage_import_blob, tenant_id, content_sha256, staged, byte_size
        )
    return {"content_sha256": content_sha256, "byte_size": byte_size}


def _copy_staged_blob(store: ArtifactStore, tenant: str | None, digest: str, dest: Path) -> bool:
    """Copy bytes the caller staged under *digest* to *dest*; False if none."""
    reader_cm = store.open_staged_import(tenant, digest)
    if reader_cm is None:
        return False
    with reader_cm as reader, open(dest, "wb") as out:
        while chunk := reader.read(BLOB_STREAM_CHUNK_BYTES):
            out.write(chunk)
    return True


@router.post("/v1/artifacts/import")
async def import_artifact_route(
    request: Request,
    store: WriteStore,
    principal: CurrentPrincipal,
    remap: bool = False,
):
    """Copy one artifact version into this store, keeping its id and version.

    The route publishing needs. A chain lives in the store its cells wrote to;
    a link resolves from the store the server serves. Those are different
    machines, so the chain has to travel, and it has to arrive with its
    versions intact — lineage edges are recorded as ``id@v=N``, so a copy that
    let this store assign fresh versions would land ancestors under numbers the
    descendants' edges do not name.

    Ancestors first is the caller's job. This route answers where each record
    landed, which is not always where the caller asked:

    * the same computation may already be here under another id, in which case
      it resolves onto that row (one ready row per tenant and provenance hash
      is what the store's uniqueness index permits);
    * with ``remap``, a version already taken by *another tenant* is minted
      under a fresh id here.

    Either way the caller rewrites the edges of everything it imports next from
    the ref this returns, or the chain resolves to nothing.

    Idempotency is by completeness rather than existence: a record that is
    already here but missing its bytes, or missing the lineage this caller
    knows, is completed rather than skipped.

    The caller's tenant is stamped on the row whatever the record says, so an
    import cannot place an artifact in someone else's namespace.

    Two forms. A JSON body is the record alone, and its bytes are the ones the
    caller uploaded with ``PUT /v1/artifacts/import/blobs/{content_sha256}``,
    which is how a large artifact travels without being held in memory. A
    multipart body carries a ``metadata`` file and the bytes as ``data``.
    """
    with tempfile.TemporaryDirectory(prefix="strata_import_") as workdir:
        return await _import_artifact(request, store, principal, remap, Path(workdir))


async def _import_artifact(
    request: Request,
    store: ArtifactStore,
    principal: Principal | None,
    remap: bool,
    workdir: Path,
) -> dict:
    """The import route's body, with a directory to stage a copied blob in."""
    import json as json_module

    from strata.artifact_store import ArtifactVersion

    tenant_id = principal.tenant if principal else None
    staged = request.headers.get("content-type", "").startswith("application/json")
    blob: bytes | Path | None = None
    if staged:
        try:
            metadata = await request.json()
        except (json_module.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="The body must be a JSON object")
    else:
        form = await request.form()
        metadata_file = form.get("metadata")
        data_file = form.get("data")
        if metadata_file is None or isinstance(metadata_file, str):
            raise HTTPException(status_code=400, detail="Missing 'metadata' file field")

        try:
            metadata = json_module.loads(await metadata_file.read())
        except json_module.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="Metadata must be a JSON object")
        if data_file is not None and not isinstance(data_file, str):
            blob = await data_file.read()

    artifact_id = str(metadata.get("id") or "").strip()
    if not artifact_id:
        raise HTTPException(status_code=400, detail="Metadata is missing 'id'")
    try:
        # The id is the caller's, and it becomes a blob key.
        reject_unsafe_artifact_id(artifact_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        version = int(metadata.get("version"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Metadata 'version' must be an integer")
    if version < 1:
        # Versions start at 1, and staged uploads wait under version 0.
        raise HTTPException(status_code=400, detail="Metadata 'version' must be 1 or more")
    provenance_hash = str(metadata.get("provenance_hash") or "").strip()
    if not provenance_hash:
        raise HTTPException(status_code=400, detail="Metadata is missing 'provenance_hash'")
    # The row keeps the source's creation time (see import_artifact), and the
    # column is NOT NULL: a record without it reached the database and came
    # back as a 500 from the constraint.
    try:
        created_at = float(metadata["created_at"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Metadata 'created_at' must be the source's creation time, in epoch seconds",
        )

    declared_digest = str(metadata.get("content_sha256") or "").strip()
    if staged:
        if not re.fullmatch(r"[0-9a-f]{64}", declared_digest):
            raise HTTPException(
                status_code=400,
                detail="A JSON import names its bytes by 'content_sha256' (64 hex digits)",
            )
        copied = workdir / "blob"
        if await asyncio.to_thread(_copy_staged_blob, store, tenant_id, declared_digest, copied):
            # Checked against this digest when it was uploaded.
            blob = copied
    elif declared_digest and isinstance(blob, bytes):
        # Verified before anything is written. The digest is the caller's claim
        # about its own bytes, and an import that stored bytes contradicting it
        # would publish a page whose verify step fails against a record this
        # store vouched for.
        actual = hashlib.sha256(blob).hexdigest()
        if actual != declared_digest:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded bytes do not match the declared digest "
                    f"(declared {declared_digest[:12]}…, received {actual[:12]}…)"
                ),
            )

    record = ArtifactVersion(
        id=artifact_id,
        version=version,
        state=str(metadata.get("state") or "ready"),
        provenance_hash=provenance_hash,
        schema_json=metadata.get("schema_json"),
        row_count=metadata.get("row_count"),
        byte_size=metadata.get("byte_size"),
        created_at=created_at,
        transform_spec=metadata.get("transform_spec"),
        input_versions=metadata.get("input_versions"),
        tenant=tenant_id,
        principal=metadata.get("principal"),
        content_sha256=declared_digest or None,
    )

    existing = store.get_artifact(artifact_id, version)
    # Two ways the id is already taken: by another tenant, and by another
    # computation. Ids are not globally unique -- a notebook's are built from
    # its own id and its cells' -- so two people working from one repository
    # send the same id for cells they have each edited differently.
    taken_by_another_tenant = existing is not None and (existing.tenant or "") != (tenant_id or "")
    taken_by_another_computation = (
        existing is not None
        and not taken_by_another_tenant
        and existing.provenance_hash != record.provenance_hash
    )
    if taken_by_another_tenant or taken_by_another_computation:
        if not remap:
            held = "another tenant" if taken_by_another_tenant else "a different computation"
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{artifact_id}@v={version} already exists, holding {held}. "
                    f"Retry with remap=true to import it under a fresh id."
                ),
            )
        record = replace(record, id=f"{artifact_id}@import={uuid.uuid4().hex[:8]}")

    if staged and blob is None:
        # Nothing uploaded, which is right only for a record this store already
        # holds with its bytes: a retry after an import that went through.
        already = store.get_artifact(record.id, record.version)
        same = store.find_by_provenance(record.provenance_hash, tenant_id)
        complete = already is not None and store.blob_exists(already.id, already.version)
        if not complete and same is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No bytes uploaded for content_sha256 {declared_digest[:12]}…; "
                    f"PUT them to /v1/artifacts/import/blobs/{declared_digest} first"
                ),
            )

    landed = store.import_artifact(record, blob)
    if isinstance(blob, Path):
        await asyncio.to_thread(store.release_staged_import, tenant_id, declared_digest)
    return {
        "artifact_uri": f"strata://artifact/{landed.ref}",
        "id": landed.id,
        "version": landed.version,
        "written": landed.written,
        "remapped": landed.ref != f"{artifact_id}@v={version}",
    }


@router.put(
    "/v1/artifacts/by-provenance/{provenance_hash}",
    response_model=PutArtifactResponse,
)
async def put_artifact_by_provenance(
    request: Request,
    store: WriteStore,
    principal: CurrentPrincipal,
    provenance_hash: str = FastPath(pattern="^[0-9a-f]{64}$"),
):
    """Store a result under a provenance key the caller computed.

    The write half of the team cache, and the one place the two materialize
    pipelines have to meet above the artifact store. ``PUT /v1/artifacts``
    derives the provenance hash itself from inputs + transform; a notebook cell
    derives its own from ``sorted_input_hashes + source_hash + env_hash``, over
    material — the cell's source, the environment lockfile — that the server
    never sees. There is therefore no way for the store to *check* this key,
    and no way for a notebook to publish under it without a route that accepts
    one.

    So this is a deliberate trust delegation, and it is bounded three ways: the
    caller must be authenticated with ``artifacts:write``, the row is stamped
    with their tenant so the reach is exactly their own team, and **an existing
    hash is never overwritten** — a second write of the same key returns the
    first one. That last part matters most: it turns "poison the shared cache"
    into "race to be first", and a team already runs each other's code by
    sharing a cache at all.

    The body is opaque bytes, not Arrow. A cell variable can be Arrow, JSON, or
    a pickle, and only the notebook's serializer knows which — parsing here
    would reject two of the three. ``content_type`` travels in the metadata so
    the puller can decode without a second round trip.
    """
    import json as json_module

    from strata.artifact_store import TransformSpec as ArtifactTransformSpec

    form = await request.form()
    metadata_file = form.get("metadata")
    data_file = form.get("data")
    if metadata_file is None or isinstance(metadata_file, str):
        raise HTTPException(status_code=400, detail="Missing 'metadata' file field")
    if data_file is None or isinstance(data_file, str):
        raise HTTPException(status_code=400, detail="Missing 'data' file field")

    try:
        metadata = json_module.loads(await metadata_file.read())
    except json_module.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid metadata JSON: {exc}")
    if not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="Metadata must be a JSON object")

    content_type = metadata.get("content_type")
    if not content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing 'content_type' — a reader cannot decode the blob without it",
        )

    blob = await data_file.read()
    tenant_id = principal.tenant if principal else None

    # First writer wins. Returning the incumbent rather than superseding it is
    # what keeps a shared cache key from being reassignable by anyone who can
    # write to the tenant.
    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None and existing.state == "ready":
        return PutArtifactResponse(
            artifact_uri=f"strata://artifact/{existing.id}@v={existing.version}",
            hit=True,
            byte_size=existing.byte_size or 0,
        )

    params: dict[str, str] = {"content_type": str(content_type)}
    variable_name = metadata.get("variable_name")
    if variable_name:
        params["variable_name"] = str(variable_name)
    build_env = metadata.get("build_env")
    if build_env:
        params["build_env"] = str(build_env)
    build_duration_ms = metadata.get("build_duration_ms")
    if build_duration_ms:
        params["build_duration_ms"] = str(build_duration_ms)
    env_hash = metadata.get("env_hash")
    if env_hash:
        params["env_hash"] = str(env_hash)

    artifact_id = str(metadata.get("artifact_id") or uuid.uuid4())
    # The caller names the id, so it can name somebody else's — and a version
    # appended there becomes that artifact's latest, which is what a notebook
    # reads and what GC protects. The import route refuses the same case.
    existing = store.get_latest_version(artifact_id)
    if existing is not None and (existing.tenant or "") != (tenant_id or ""):
        raise HTTPException(
            status_code=409,
            detail=f"{artifact_id} already exists under another tenant",
        )
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=ArtifactTransformSpec(
            executor="notebook/cell@v1",
            params=params,
            inputs=[],
        ),
        tenant=tenant_id,
        principal=principal.id if principal else None,
    )
    store.write_blob(artifact_id, version, blob)
    finalized = store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json=str(metadata.get("schema_json") or ""),
        row_count=int(metadata.get("row_count") or 0),
        byte_size=len(blob),
        content_sha256=hashlib.sha256(blob).hexdigest(),
    )
    if finalized is None:
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    return PutArtifactResponse(
        artifact_uri=f"strata://artifact/{finalized.id}@v={finalized.version}",
        hit=finalized.id != artifact_id or finalized.version != version,
        byte_size=finalized.byte_size or len(blob),
    )


@router.get(
    "/v1/artifacts/by-provenance/{provenance_hash}",
    response_model=ArtifactProvenanceMatchResponse,
)
async def find_artifact_by_provenance(
    store: ReadStore,
    tenant_filter: CurrentTenant,
    principal: CurrentPrincipal,
    provenance_hash: str = FastPath(pattern="^[0-9a-f]{64}$"),
):
    """Has anyone already computed this?

    The primitive a *shared* store exists to answer. Every other artifact read
    starts from an id someone already holds; this one starts from a provenance
    hash, which is the only identifier two people arrive at independently —
    notebook artifact ids embed the notebook, so a colleague's copy of the same
    computation is never at an id you could have guessed.

    Answering it over HTTP is what turns the local dedup that already happens
    into a team cache hit: look up by hash, and on a match fetch the bytes from
    the sibling ``/data`` route.

    404 is the ordinary answer, not an error — it means "nobody has, go
    compute it". A caller cannot tell that apart from the *other* 404s it might
    get (an old server that lacks this route; a gateway with no artifact store
    configured), and treating those as a miss means recomputing forever while
    everything looks healthy. So a genuine miss carries
    ``X-Strata-Provenance-Miss``, and clients key off the header rather than
    the status alone.

    The hash is constrained to a sha256 digest at the route, because every
    hash the store issues is one (including the ``derive_subkey`` per-variable
    hashes) and a lookup key that reaches the database should not be free-form.

    Scoping: the lookup is filtered to the caller's own tenant in SQL, so a
    team only ever hits its own results, and the ACL re-check below applies the
    same table-level deny rules the by-id reads apply — a shared cache does not
    become a way around a table someone is denied.
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # Two different Nones meet here. `CurrentTenant` yields None for "do not
    # filter" (an `admin:*` caller, or auth switched off), while
    # `find_by_provenance(tenant=None)` means the *tenantless* namespace
    # specifically — never "any tenant", by design, so a tenantless build
    # cannot dedup against another tenant's artifact. Passing one straight into
    # the other made an admin miss on results they had just published.
    #
    # The lookup therefore scopes to the principal's own tenant, exactly as the
    # publish path does. Admin does not widen it: "has anyone computed this?"
    # is a question about one team's cache, and a match drawn from whichever
    # tenant happened to write last is not an answer to it. Reading another
    # tenant's artifact stays available by id, where admin does widen.
    lookup_tenant = principal.tenant if principal else None
    artifact = store.find_by_provenance(provenance_hash, tenant=lookup_tenant)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="No artifact has been computed for that provenance hash",
            headers={PROVENANCE_MISS_HEADER: "1"},
        )
    artifact = _ensure_artifact_access(artifact, tenant_filter)
    _authorize_artifact_read(artifact)

    return ArtifactProvenanceMatchResponse(
        artifact_id=artifact.id,
        version=artifact.version,
        provenance_hash=artifact.provenance_hash,
        content_type=_spec_param(artifact, "content_type"),
        build_env=_spec_param(artifact, "build_env"),
        build_duration_ms=_spec_int(artifact, "build_duration_ms"),
        env_hash=_spec_param(artifact, "env_hash"),
        state=artifact.state,
        arrow_schema=artifact.schema_json,
        row_count=artifact.row_count,
        byte_size=artifact.byte_size,
        created_at=artifact.created_at or 0.0,
        principal=artifact.principal,
        content_sha256=artifact.content_sha256,
        promotion=store.get_tags(artifact.id, artifact.version, tenant=artifact.tenant).get(
            PROMOTION_TAG
        ),
    )


def _spec_param(artifact, key: str) -> str:
    """One value out of the stored transform spec's params.

    Carries ``content_type`` (how the blob was serialized) and ``build_env``
    (which interpreter on which machine produced it). Both come back "" rather
    than a guess when unrecorded: core transforms record neither, and every
    artifact written before those fields existed has neither. A caller that has
    to deserialize should see "unstated" and decide, not receive a
    plausible-looking default that is wrong for pickled values.
    """
    if not artifact.transform_spec:
        return ""
    try:
        spec = json.loads(artifact.transform_spec)
    except ValueError:
        return ""
    if not isinstance(spec, dict):
        return ""
    params = spec.get("params")
    if not isinstance(params, dict):
        return ""
    return str(params.get(key) or "")


def _spec_int(artifact, key: str) -> int:
    """The same, for a param that is a whole number of milliseconds.

    Params are stored as strings, and a stored value that will not parse means
    a producer wrote something unexpected. Zero, matching "unrecorded", is the
    honest reading — the alternative is a 500 on a read path over metadata
    nothing depends on.
    """
    raw = _spec_param(artifact, key)
    try:
        return int(raw)
    except ValueError:
        return 0


class UsageScope(NamedTuple):
    """Whose usage a stats/usage call reports, and from which store."""

    store: ArtifactStore
    tenant: str | None
    include_tenantless: bool


def usage_scope(
    tenant: str | None = Query(
        default=None,
        description="Tenant to report on. Service mode, admin:* only; others get their own.",
    ),
) -> UsageScope:
    """Resolve the scope of a usage read.

    Personal mode reports the whole store, as it always has. Service mode
    reports one tenant's holdings and never another's: the caller's own, from
    the trusted-proxy identity, or for ``admin:*`` a tenant named in the query
    (the whole store when none is named). A tenant's figure there excludes
    legacy tenantless rows, which belong to no one and would otherwise be
    charged to every tenant at once.
    """
    from strata.auth import get_principal
    from strata.server import _get_artifact_request_tenant, _get_artifact_store, get_state

    config = get_state().config
    if config.writes_enabled:
        return UsageScope(_get_artifact_store(), _get_artifact_request_tenant(), True)

    if not config.principal_auth_enabled:
        # Without an authenticated caller there is no tenant to scope to, and
        # an unscoped answer would be every tenant's usage at once.
        raise HTTPException(
            status_code=403,
            detail="Usage in service mode is per tenant and needs trusted-proxy auth",
        )
    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    store = _get_artifact_store(allow_read=True)
    if principal.has_scope("admin:*"):
        return UsageScope(store, tenant, tenant is None)
    if principal.tenant is None:
        raise HTTPException(status_code=400, detail="Tenant header required for artifact usage")
    if tenant is not None and tenant != principal.tenant:
        raise HTTPException(status_code=403, detail="Usage of another tenant requires admin:*")
    return UsageScope(store, principal.tenant, False)


UsageScopeDep = Annotated[UsageScope, Depends(usage_scope)]


@router.get("/v1/artifacts/stats")
async def get_artifact_stats(scope: UsageScopeDep):
    """Get artifact store statistics.

    Whole store in personal mode; one tenant's in service mode (see
    :func:`usage_scope`).

    Returns:
        Artifact store statistics
    """
    return scope.store.stats(tenant=scope.tenant, include_tenantless=scope.include_tenantless)


@router.get("/v1/artifacts/usage")
async def get_artifact_usage(scope: UsageScopeDep):
    """Get artifact store usage metrics.

    Returns comprehensive usage statistics including:
    - Total bytes used
    - Number of artifacts and versions
    - Unreferenced artifact count (candidates for GC)

    Whole store in personal mode; one tenant's in service mode, which is what
    metering reads (see :func:`usage_scope`).

    Returns:
        Usage metrics dictionary
    """
    return scope.store.get_usage(tenant=scope.tenant, include_tenantless=scope.include_tenantless)


@router.get("/v1/artifacts")
async def list_artifacts(
    store: PersonalModeStore,
    tenant_filter: CurrentTenant,
    # Bounded: these flow straight into "LIMIT ? OFFSET ?", and SQLite treats a
    # NEGATIVE limit as unbounded — so ?limit=-1 materialized every
    # artifact_versions row into one response. The sibling lineage/dependents
    # params already carry ge/le.
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    state: str | None = None,
    name_prefix: str | None = None,
    since: float | None = None,
    sort: str = "created_at",
    order: str = "desc",
):
    """List artifacts with optional filtering (personal mode only).

    Args:
        limit: Maximum number of artifacts to return (default 100)
        offset: Number of artifacts to skip for pagination
        state: Filter by state ("ready", "building", "failed")
        name_prefix: Filter by artifacts with names starting with prefix
        since: Only artifacts created at or after this epoch timestamp
        sort: Sort column — "created_at", "byte_size", or "row_count"
        order: "asc" or "desc" (default "desc")

    Returns:
        List of artifact versions with their metadata
    """
    if state is not None and state not in ("ready", "building", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state filter: {state}. Must be 'ready', 'building', or 'failed'",
        )
    if sort not in ("created_at", "byte_size", "row_count"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort: {sort}. Must be 'created_at', 'byte_size', or 'row_count'",
        )
    if order not in ("asc", "desc"):
        raise HTTPException(
            status_code=400, detail=f"Invalid order: {order}. Must be 'asc' or 'desc'"
        )

    artifacts = store.list_artifacts(
        limit=limit,
        offset=offset,
        state=state,
        name_prefix=name_prefix,
        tenant=tenant_filter,
        since=since,
        sort=sort,
        order=order,
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

    try:
        deleted = store.delete_artifact(artifact_id, version, tenant=tenant_filter)
    except ValueError as exc:
        # Published: withdrawing the citation is a separate, deliberate act.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return {"deleted": True, "artifact_uri": f"strata://artifact/{artifact_id}@v={version}"}


class PinRequest(BaseModel):
    reason: str


@router.post("/v1/artifacts/{artifact_id}/v/{version}/pin")
async def pin_artifact(
    artifact_id: str,
    version: int,
    request: PinRequest,
    tenant_filter: CurrentTenant,
    principal: CurrentPrincipal,
    store: ArtifactStore = store_for_scope("artifacts:pin"),
):
    """Hold a version and every ancestor against garbage collection.

    For a chain the store has no other reason to keep — a snapshot that must
    stay restorable, a review still open. One pin per reason; pinning again
    under the same reason refreshes it.
    """
    from strata.server import _ensure_artifact_access

    _ensure_artifact_access(store.get_artifact(artifact_id, version), tenant_filter)
    try:
        return store.pin_artifact(
            artifact_id,
            version,
            request.reason,
            tenant=tenant_filter,
            pinned_by=principal.id if principal is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/v1/artifacts/{artifact_id}/v/{version}/pin")
async def unpin_artifact(
    artifact_id: str,
    version: int,
    reason: str,
    tenant_filter: CurrentTenant,
    store: ArtifactStore = store_for_scope("artifacts:pin"),
):
    """Release the pin placed under ``reason``."""
    if not store.unpin_artifact(artifact_id, version, reason, tenant=tenant_filter):
        raise HTTPException(status_code=404, detail="No pin with that reason on this version")
    return {"unpinned": True, "artifact_id": artifact_id, "version": version, "reason": reason}


class ExportTableRequest(BaseModel):
    table: str
    alias: str | None = None


@router.post("/v1/artifacts/{artifact_id}/v/{version}/export")
async def export_artifact_to_table(
    artifact_id: str,
    version: int,
    request: ExportTableRequest,
    tenant_filter: CurrentTenant,
    principal: CurrentPrincipal,
    store: WriteStore,
):
    """Write a tabular artifact into an Iceberg table as its current snapshot.

    For a platform that exports a dataset once it has been promoted. The
    snapshot's summary names this version, and ``alias`` becomes a tag on it.
    The catalog is this server's: ``table`` is a ``<warehouse>#ns.table`` URI or
    a ``ns.table`` in the configured catalog, as ``@table`` reads it.
    """
    from strata.api.dependencies import authorize_table_access
    from strata.iceberg import table_identity_for
    from strata.server import _ensure_artifact_access, get_state
    from strata.table_export import export_artifact

    config = get_state().config
    # The table this writes is authorized like any table a scan reads, so a
    # principal denied a table cannot write it either.
    try:
        identity = table_identity_for(request.table, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    authorize_table_access(request.table, identity)
    artifact = _ensure_artifact_access(store.get_artifact(artifact_id, version), tenant_filter)
    try:
        written = await asyncio.to_thread(
            export_artifact,
            store,
            artifact,
            request.table,
            config=config,
            promoted_by=principal.id if principal is not None else None,
            alias=request.alias,
            tenant=tenant_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "table": written.table,
        "snapshot_id": written.snapshot_id,
        "created": written.created,
        "artifact_uri": f"strata://artifact/{artifact_id}@v={version}",
    }


@router.post("/v1/artifacts/gc")
async def garbage_collect_artifacts(
    tenant_filter: CurrentTenant,
    max_age_days: float = 7.0,
    collect_latest: bool = False,
    store: ArtifactStore = store_for_scope("admin:*"),
):
    """Garbage collect unreachable artifacts.

    Personal mode, or service mode for a principal holding ``admin:*``, scoped
    to the caller's tenant.

    Deletes artifact versions that:
    1. Have no name **or alias** pointing at them
    2. Are not the latest version of their id (unless ``collect_latest``)
    3. Are older than ``max_age_days``
    4. Are in "ready", "superseded" or "failed" state
    5. Are not published or pinned, and nothing published or pinned depends
       on them

    The latest version of an id is spared because that is the artifact's
    *current value*: ``get_latest_version(id)`` is how the store resolves it,
    and for some producers it is the only handle that exists — notebook cell
    outputs are stored under a canonical id and never named, so the previous
    rule treated every live notebook variable as garbage.

    Args:
        max_age_days: Maximum age in days for unreachable artifacts (default 7)
        collect_latest: Also reclaim current values (see above). Off by
            default because it deletes live state.

    Returns:
        GC statistics including deleted count and bytes freed
    """
    if max_age_days < 0:
        raise HTTPException(status_code=400, detail="max_age_days must be non-negative")

    result = store.garbage_collect(
        max_age_days=max_age_days,
        tenant=tenant_filter,
        collect_latest=collect_latest,
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
    # Answered by the team store when one is configured. The dashboard opens
    # lineage from the Registry tab and the cell strip, both of which list what
    # the team's registry holds — so asking the local store here would 404 on
    # exactly the artifacts the reader just clicked.
    target = remote_registry()
    if target is not None:
        return await relay(
            target,
            "GET",
            f"/v1/artifacts/{quoted(artifact_id)}/v/{version}/lineage",
            params={"max_depth": max_depth},
        )

    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # Get the root artifact
    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    # Same table-ACL re-check the sibling read endpoints run. Without it a
    # principal denied a table could still read the lineage graph naming it —
    # the response carries every upstream table URI, the snapshot pinned in
    # input_version, and each transform ref, which is most of what the deny
    # rule exists to withhold.
    _authorize_artifact_read(artifact)

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
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # Verify the artifact exists
    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    _authorize_artifact_read(artifact)

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


@router.post("/v1/artifacts/upload/{artifact_id}/v/{version}")
async def upload_artifact_blob(
    artifact_id: str, version: int, request: Request, store: PersonalModeStore
):
    """Upload artifact blob data (personal mode only).

    The client POSTs raw Arrow IPC stream bytes to this endpoint.
    After upload, call /v1/artifacts/finalize to complete the artifact.

    Args:
        artifact_id: Artifact ID from materialize response
        version: Version number from materialize response
        request: Raw request body containing Arrow IPC bytes
    """
    # Verify artifact exists and is in building state
    artifact = store.get_artifact(artifact_id, version)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")
    if artifact.state != "building":
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not in building state (state={artifact.state})",
        )

    byte_size = 0
    fd, tmp_name = tempfile.mkstemp(prefix="strata_upload_", suffix=".tmp")
    staged = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as dst:
            async for chunk in request.stream():
                if not chunk:
                    continue
                byte_size += len(chunk)
                dst.write(chunk)
        if byte_size == 0:
            raise HTTPException(status_code=400, detail="Empty request body")
        await asyncio.to_thread(store.publish_blob_from_path, artifact_id, version, staged)
    finally:
        staged.unlink(missing_ok=True)

    return {"status": "uploaded", "byte_size": byte_size}


@router.post("/v1/artifacts/finalize", response_model=UploadFinalizeResponse)
async def finalize_artifact(request: UploadFinalizeRequest, store: PersonalModeStore):
    """Finalize an artifact after upload (personal mode only).

    After uploading the blob, call this to transition the artifact to ready state.
    Optionally sets a name pointer to the artifact.

    Returns:
        UploadFinalizeResponse with artifact URI and optional name URI
    """
    # Verify blob exists
    if not store.blob_exists(request.artifact_id, request.version):
        raise HTTPException(
            status_code=400,
            detail="Blob not uploaded. Call upload endpoint first.",
        )

    # Get blob size without materializing the payload.
    #
    # ``blob_size`` returns None both when the object is absent and when the
    # backend call simply failed, so ``or 0`` could stamp byte_size=0 onto an
    # artifact this call is about to mark READY — a ready row claiming an
    # empty blob. The sibling build-finalize route already refuses that; do
    # the same here rather than record a figure we know is wrong.
    byte_size = store.blob_size(request.artifact_id, request.version) or 0
    if byte_size == 0:
        raise HTTPException(status_code=500, detail="Failed to read uploaded blob")

    # Finalize artifact
    try:
        finalized_artifact = store.finalize_artifact(
            artifact_id=request.artifact_id,
            version=request.version,
            schema_json=request.arrow_schema,
            row_count=request.row_count,
            byte_size=byte_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if finalized_artifact is None:
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    artifact_uri = f"strata://artifact/{finalized_artifact.id}@v={finalized_artifact.version}"
    name_uri = None

    # Set name if requested
    if request.name:
        try:
            store.set_name(request.name, finalized_artifact.id, finalized_artifact.version)
            name_uri = f"strata://name/{request.name}"
        except ValueError as e:
            # Don't fail the whole request if name setting fails
            logger.warning(f"Failed to set name {request.name}: {e}")

    return UploadFinalizeResponse(
        artifact_uri=artifact_uri,
        byte_size=finalized_artifact.byte_size or byte_size,
        name_uri=name_uri,
    )

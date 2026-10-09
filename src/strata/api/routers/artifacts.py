"""Artifact CRUD plus the personal-mode upload/finalize routes.

Handlers take an already-gated store and tenant filter from the typed
dependencies. The ACL helpers stay in ``server.py`` and are lazy-imported
in-body; the signed-transport upload/finalize routes live in ``builds.py``.
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
from typing import Annotated, BinaryIO, NamedTuple

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import Path as FastPath
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.types import Message

from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    PersonalModeStore,
    ReadStore,
    WriteStore,
    store_for_scope,
)
from strata.api.remote_registry import quoted, relay, remote_registry
from strata.api.served_bytes import data_headers
from strata.artifact_store import ArtifactImportConflict, ArtifactStore, reject_unsafe_artifact_id
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


def _upload_too_large(limit: int) -> HTTPException:
    return HTTPException(
        status_code=413, detail=f"Request body exceeds max_upload_bytes ({limit} bytes)"
    )


def _capped(request: Request) -> Request:
    """*request* with its body refused (413) past ``max_upload_bytes``.

    The count runs as the body arrives, so a chunked body with no Content-Length is held to it.
    """
    from strata.server import get_state

    limit = get_state().config.max_upload_bytes
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _upload_too_large(limit)
    received = 0

    async def receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > limit:
                raise _upload_too_large(limit)
        return message

    return Request(request.scope, receive)


def _spool_upload(source: BinaryIO, dest: Path) -> tuple[str, int]:
    """Copy an uploaded part to *dest* in chunks, returning its sha256 and size."""
    hasher = hashlib.sha256()
    size = 0
    source.seek(0)
    with open(dest, "wb") as out:
        while chunk := source.read(BLOB_STREAM_CHUNK_BYTES):
            hasher.update(chunk)
            size += len(chunk)
            out.write(chunk)
    return hasher.hexdigest(), size


def _arrow_stream_shape(path: Path) -> tuple[str, int]:
    """The schema and row count of the one Arrow IPC stream in *path*, read through a memory map.

    Trailing bytes (concatenated streams) are refused: every standard reader downstream would
    silently drop them.
    """
    with pa.memory_map(str(path)) as source:
        reader = ipc.open_stream(source)
        row_count = sum(batch.num_rows for batch in reader)
        trailing = source.size() - source.tell()
        if trailing:
            raise ValueError(f"{trailing} trailing bytes after stream end (concatenated streams?)")
        return reader.schema.to_string(), row_count


@router.put("/v1/artifacts", response_model=PutArtifactResponse)
async def put_artifact(request: Request, store: WriteStore, principal: CurrentPrincipal):
    """Upload and persist a locally computed artifact with provenance tracking.

    Accepts a JSON body (inputs, transform, data, name) or multipart (``metadata``
    JSON plus ``data`` Arrow IPC bytes). An existing artifact with the same
    provenance hash is returned with ``hit=True`` and nothing is stored.
    """
    with tempfile.TemporaryDirectory(prefix="strata_put_") as workdir:
        return await _put_artifact(request, store, principal, Path(workdir) / "data")


async def _put_artifact(
    request: Request, store: ArtifactStore, principal: Principal | None, data_path: Path
) -> PutArtifactResponse:
    """The PUT route's body, with a path to hold the uploaded bytes."""
    import json as json_module

    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # Multipart: metadata JSON + Arrow IPC data
        form = await _capped(request).form()

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

        data_file = form.get("data")
        if data_file is None:
            raise HTTPException(status_code=400, detail="Missing 'data' field in multipart request")
        if isinstance(data_file, str):
            raise HTTPException(
                status_code=400,
                detail="'data' field must be a file, not form data",
            )
        content_sha256, byte_size = await asyncio.to_thread(
            _spool_upload, data_file.file, data_path
        )
        try:
            schema_json, row_count = await asyncio.to_thread(_arrow_stream_shape, data_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Arrow IPC data: {e}")

    else:
        # JSON body (legacy format)
        try:
            body = await _capped(request).json()
        except json_module.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        inputs = body.get("inputs", [])
        transform_dict = body.get("transform", {})
        artifact_name = body.get("name")
        data = body.get("data")

        if data is None:
            raise HTTPException(status_code=400, detail="Missing 'data' field")

        try:
            if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
                table = pa.Table.from_pydict(data)
            else:
                # Non-columnar: store as a single JSON column.
                json_str = json_module.dumps(data)
                table = pa.Table.from_pydict({"data": [json_str]})
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to convert data to Arrow: {e}")

        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        arrow_bytes = sink.getvalue().to_pybytes()
        data_path.write_bytes(arrow_bytes)
        content_sha256 = hashlib.sha256(arrow_bytes).hexdigest()
        byte_size = len(arrow_bytes)
        schema_json, row_count = table.schema.to_string(), table.num_rows

    executor = transform_dict.get("executor")
    if not executor:
        raise HTTPException(status_code=400, detail="Missing 'executor' in transform")
    params = transform_dict.get("params", {})

    # Resolve the tenant from the principal, as materialize does: get_tenant_id() defaults to
    # "_default", which the name routes (None) can never address.
    tenant_id = principal.tenant if principal else None

    input_versions: dict[str, str] = {}
    for input_uri in inputs:
        try:
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
                # Table or unknown URI: the URI stands in for its version.
                input_versions[input_uri] = input_uri
        except Exception:
            input_versions[input_uri] = input_uri

    from strata.artifact_store import TransformSpec as ArtifactTransformSpec
    from strata.artifact_store import compute_provenance_hash

    artifact_transform = ArtifactTransformSpec(
        executor=executor,
        params=params,
        inputs=inputs,
    )

    input_hashes = [f"{uri}:{ver}" for uri, ver in sorted(input_versions.items())]
    provenance_hash = compute_provenance_hash(input_hashes, artifact_transform)

    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None and existing.state == "ready":
        artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"
        name_uri = None

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

    artifact_id = str(uuid.uuid4())
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=artifact_transform,
        input_versions=input_versions,
        tenant=tenant_id,
        # A shared store hands out results you did not compute, so who did is part of the result.
        principal=principal.id if principal else None,
        minted=True,
    )

    await asyncio.to_thread(store.publish_blob_from_path, artifact_id, version, data_path)

    finalized_artifact = store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json=schema_json,
        row_count=row_count,
        byte_size=byte_size,
        content_sha256=content_sha256,
    )
    if finalized_artifact is None:
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    artifact_uri = f"strata://artifact/{finalized_artifact.id}@v={finalized_artifact.version}"
    name_uri = None

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
        byte_size=finalized_artifact.byte_size or byte_size,
        name_uri=name_uri,
    )


@router.get("/v1/artifacts/{artifact_id}/v/{version}", response_model=ArtifactInfoResponse)
async def get_artifact_info(
    artifact_id: str, version: int, store: ReadStore, tenant_filter: CurrentTenant
):
    """Get artifact metadata.

    Available in service mode, gated by tenant and the table ACL of the artifact's inputs.
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    _authorize_artifact_read(artifact, store)

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
        input_versions=artifact.input_versions,
    )


@router.put("/v1/artifacts/import/blobs/{content_sha256}", status_code=201)
async def stage_import_blob_route(
    request: Request,
    store: WriteStore,
    principal: CurrentPrincipal,
    content_sha256: str = FastPath(pattern="^[0-9a-f]{64}$"),
):
    """Upload the bytes of an artifact about to be imported, ahead of its record.

    The body is streamed to disk and checked against the digest in the path, so a
    201 means these are exactly those bytes. Only the caller's tenant can import them.
    """
    tenant_id = principal.tenant if principal else None
    with tempfile.TemporaryDirectory(prefix="strata_import_") as workdir:
        staged = Path(workdir) / "blob"
        hasher = hashlib.sha256()
        byte_size = 0
        with open(staged, "wb") as out:
            async for chunk in _capped(request).stream():
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


def _is_json_object(value: object) -> bool:
    """Whether *value* is a string holding a JSON object, as ``transform_spec`` and edges are."""
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value), dict)
    except json.JSONDecodeError:
        return False


@router.post("/v1/artifacts/import")
async def import_artifact_route(
    request: Request,
    store: WriteStore,
    principal: CurrentPrincipal,
    remap: bool = False,
):
    """Copy one artifact version into this store, keeping its id and version.

    Versions must survive because lineage edges name ``id@v=N``. Importing
    ancestors first is the caller's job, and the caller must rewrite later edges
    from the returned ref: a record may resolve onto an existing row with the same
    provenance hash, or, with ``remap``, land under a fresh id when another tenant
    holds that version.

    Idempotent by completeness: a record already here but missing bytes or lineage
    is completed. The caller's tenant is always stamped on the row. A JSON body is
    the record alone (bytes staged via ``PUT /v1/artifacts/import/blobs/{content_sha256}``);
    a multipart body carries ``metadata`` and ``data``.
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
    blob: Path | None = None
    received_digest: str | None = None
    if staged:
        try:
            metadata = await _capped(request).json()
        except (json_module.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}")
        if not isinstance(metadata, dict):
            raise HTTPException(status_code=400, detail="The body must be a JSON object")
    else:
        form = await _capped(request).form()
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
            blob = workdir / "blob"
            received_digest, _ = await asyncio.to_thread(_spool_upload, data_file.file, blob)

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
    # The record is stored as sent, and a string byte_size broke every later sweep.
    for field in ("row_count", "byte_size"):
        value = metadata.get(field)
        if value is not None and (type(value) is not int or value < 0):
            raise HTTPException(
                status_code=400, detail=f"Metadata '{field}' must be a non-negative integer"
            )
    for field in ("schema_json", "principal"):
        if not isinstance(metadata.get(field), str | None):
            raise HTTPException(status_code=400, detail=f"Metadata '{field}' must be a string")
    for field in ("transform_spec", "input_versions"):
        if metadata.get(field) is not None and not _is_json_object(metadata[field]):
            raise HTTPException(
                status_code=400,
                detail=f"Metadata '{field}' must be a JSON object encoded as a string",
            )
    state = metadata.get("state") or "ready"
    if state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400, detail="Metadata 'state' must be 'ready' or 'superseded'"
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
    elif declared_digest and received_digest is not None:
        # Verified before anything is written. The digest is the caller's claim
        # about its own bytes, and an import that stored bytes contradicting it
        # would publish a page whose verify step fails against a record this
        # store vouched for.
        if received_digest != declared_digest:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Uploaded bytes do not match the declared digest "
                    f"(declared {declared_digest[:12]}…, received {received_digest[:12]}…)"
                ),
            )

    record = ArtifactVersion(
        id=artifact_id,
        version=version,
        state=state,
        provenance_hash=provenance_hash,
        schema_json=metadata.get("schema_json"),
        row_count=metadata.get("row_count"),
        byte_size=metadata.get("byte_size"),
        created_at=created_at,
        transform_spec=metadata.get("transform_spec"),
        input_versions=metadata.get("input_versions"),
        tenant=tenant_id,
        principal=metadata.get("principal"),
        content_sha256=declared_digest or received_digest,
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

    try:
        landed = await asyncio.to_thread(store.import_artifact, record, blob)
    except ArtifactImportConflict as exc:
        # Another import of this id@v=N with a different computation committed
        # after the check above.
        raise HTTPException(
            status_code=409, detail=f"{exc} Retry with remap=true to import it under a fresh id."
        ) from exc
    if staged and blob is not None:
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

    The server cannot verify the key (a notebook cell's hash covers source and
    lockfile it never sees), so this is a bounded trust delegation: it requires
    ``artifacts:write``, stamps the caller's tenant, and never overwrites an
    existing hash, so a second write of the same key returns the first. The body is
    opaque bytes (Arrow, JSON or pickle); ``content_type`` travels in the metadata.
    """
    with tempfile.TemporaryDirectory(prefix="strata_put_") as workdir:
        return await _put_artifact_by_provenance(
            request, store, principal, provenance_hash, Path(workdir) / "data"
        )


async def _put_artifact_by_provenance(
    request: Request,
    store: ArtifactStore,
    principal: Principal | None,
    provenance_hash: str,
    data_path: Path,
) -> PutArtifactResponse:
    """The by-provenance PUT route's body, with a path to hold the uploaded bytes."""
    import json as json_module

    from strata.artifact_store import TransformSpec as ArtifactTransformSpec

    form = await _capped(request).form()
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
            detail="Missing 'content_type': a reader cannot decode the blob without it",
        )
    # Unknown stays None: a recorded 0 makes verify report a mismatch for every Arrow result.
    row_count = metadata.get("row_count")
    if row_count is not None and (type(row_count) is not int or row_count < 0):
        raise HTTPException(
            status_code=400, detail="Metadata 'row_count' must be a non-negative integer"
        )

    content_sha256, byte_size = await asyncio.to_thread(_spool_upload, data_file.file, data_path)
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

    named_id = metadata.get("artifact_id")
    artifact_id = str(named_id or uuid.uuid4())
    # The caller names the id, so it can name somebody else's; a version appended there becomes that
    # artifact's latest, which a notebook reads and GC protects. The import route refuses the same
    # case.
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
        # A notebook storing a cell output names its id and reads it back as
        # "latest"; an upload that named nothing got an id nobody resolves.
        minted=not named_id,
    )
    await asyncio.to_thread(store.publish_blob_from_path, artifact_id, version, data_path)
    finalized = store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json=str(metadata.get("schema_json") or ""),
        row_count=row_count,
        byte_size=byte_size,
        content_sha256=content_sha256,
    )
    if finalized is None:
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    return PutArtifactResponse(
        artifact_uri=f"strata://artifact/{finalized.id}@v={finalized.version}",
        hit=finalized.id != artifact_id or finalized.version != version,
        byte_size=finalized.byte_size or byte_size,
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
    """Look up a ready artifact by provenance hash, the team-cache read.

    A genuine miss is a 404 carrying ``X-Strata-Provenance-Miss``; clients key off
    the header, because other 404s (an old server, no artifact store) must not read
    as a miss. The hash must be a sha256 digest. Results are filtered to the
    caller's tenant and re-checked against table-level ACL deny rules.
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # Two different Nones meet here. `CurrentTenant` yields None for "do not filter" (an `admin:*`
    # caller, or auth off), while `find_by_provenance(tenant=None)` means the tenantless namespace,
    # never "any tenant".
    #
    # So scope to the principal's own tenant, as the publish path does. Admin does not widen it:
    # "has anyone computed this?" is a question about one team's cache. Reading another tenant's
    # artifact by id is where admin widens.
    lookup_tenant = principal.tenant if principal else None
    artifact = store.find_by_provenance(provenance_hash, tenant=lookup_tenant)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="No artifact has been computed for that provenance hash",
            headers={PROVENANCE_MISS_HEADER: "1"},
        )
    artifact = _ensure_artifact_access(artifact, tenant_filter)
    _authorize_artifact_read(artifact, store)

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
    """Return one value from the stored transform spec's params, or "" when unrecorded.

    Used for ``content_type`` and ``build_env``; "" rather than a guessed default,
    since a wrong default would mis-decode pickled values.
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
    """Return an integer millisecond param, or 0 when unrecorded or unparseable.

    0 rather than a 500 on a read path over metadata nothing depends on.
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

    Personal mode: the whole store. Service mode: the caller's tenant, or for
    ``admin:*`` a tenant named in the query (whole store when none). A tenant's
    figure excludes legacy tenantless rows, which would otherwise be charged to every tenant.
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
    """Get artifact store statistics: whole store in personal mode, one tenant's in service mode."""
    return scope.store.stats(tenant=scope.tenant, include_tenantless=scope.include_tenantless)


@router.get("/v1/artifacts/usage")
async def get_artifact_usage(scope: UsageScopeDep):
    """Get artifact store usage: bytes, artifact and version counts, unreferenced count.

    Whole store in personal mode; one tenant's in service mode, which is what metering reads.
    """
    return scope.store.get_usage(tenant=scope.tenant, include_tenantless=scope.include_tenantless)


@router.get("/v1/artifacts")
async def list_artifacts(
    store: PersonalModeStore,
    tenant_filter: CurrentTenant,
    # Bounded: these feed "LIMIT ? OFFSET ?", and SQLite treats a negative limit as unbounded.
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    state: str | None = None,
    name_prefix: str | None = None,
    since: float | None = None,
    sort: str = "created_at",
    order: str = "desc",
):
    """List artifact versions with optional filtering and sorting (personal mode only).

    ``since`` is an epoch timestamp; ``sort`` is ``created_at``, ``byte_size`` or ``row_count``.
    """
    if state is not None and state not in ("ready", "building", "failed", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid state filter: {state}. "
                "Must be 'ready', 'building', 'failed', or 'superseded'"
            ),
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
    """Delete an artifact version, its blob and its name pointers (personal mode only)."""
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

    One pin per reason; pinning again under the same reason refreshes it.
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

    The snapshot summary names this version and ``alias`` becomes a tag on it.
    ``table`` is a ``<warehouse>#ns.table`` URI or ``ns.table`` in the configured catalog.
    """
    from strata.api.dependencies import authorize_table_access
    from strata.iceberg import TableOfAnotherTenant, table_identity_for
    from strata.server import _authorize_artifact_read, _ensure_artifact_access, get_state
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
    # Exporting copies the bytes into a table the caller can scan, so it is a read of them.
    _authorize_artifact_read(artifact, store)
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
    except TableOfAnotherTenant as exc:
        if config.hide_forbidden_as_not_found:
            raise HTTPException(status_code=404, detail="Table not found") from exc
        raise HTTPException(status_code=403, detail="Access denied") from exc
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
    max_idle_days: float | None = None,
    max_bytes: int | None = None,
    min_idle_seconds: float | None = None,
    collect_latest: bool = False,
    dry_run: bool = False,
    store: ArtifactStore = store_for_scope("admin:*"),
):
    """Collect unneeded artifact versions, least recently used first.

    Personal mode, or service mode with ``admin:*``. Neither has a tenant filter
    (``admin:*`` is unscoped), so the sweep covers the whole store, every tenant.
    A version is kept if it has a name, alias, pin or publication, if something
    pinned, published or building depends on it, or if it is the latest value of a
    caller-named id (see ``ArtifactStore.garbage_collect``). Omitted parameters take
    the configured ``STRATA_ARTIFACT_GC_*`` retention.

    Args:
        max_idle_days: Collect what has not been used for this long.
        max_bytes: Collect until the store is at 80% of this.
        min_idle_seconds: Never collect anything used more recently.
        collect_latest: Also collect the latest value of caller-named ids (deletes live state).
        dry_run: Report what would go; delete nothing.

    Returns:
        ``deleted_count``, ``deleted_bytes``, ``store_bytes`` and, with ``dry_run``, ``collected``.
    """
    for name, value in (
        ("max_idle_days", max_idle_days),
        ("max_bytes", max_bytes),
        ("min_idle_seconds", min_idle_seconds),
    ):
        if value is not None and value < 0:
            raise HTTPException(status_code=400, detail=f"{name} must be non-negative")
    from strata.server import get_state

    policy = get_state().config.artifact_gc_policy()
    if max_idle_days is not None:
        policy["max_idle_days"] = max_idle_days
    if max_bytes is not None:
        policy["max_bytes"] = max_bytes
    if min_idle_seconds is not None:
        policy["min_idle_seconds"] = min_idle_seconds
    # A sweep unlinks a blob per version; on the loop it would stall every route.
    return await asyncio.to_thread(
        store.garbage_collect,
        **policy,
        tenant=tenant_filter,
        collect_latest=collect_latest,
        dry_run=dry_run,
    )


@router.get("/v1/artifacts/{artifact_id}/v/{version}/data")
async def get_artifact_data(
    artifact_id: str, version: int, store: ReadStore, tenant_filter: CurrentTenant
):
    """Stream an artifact's bytes as Arrow IPC.

    Available in service mode, gated by tenant and the table ACL of the artifact's inputs.
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    # Result retrieval is ACL-gated: re-check the table ACL of the inputs.
    _authorize_artifact_read(artifact, store)
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

    # Schema stays out of the headers (it may contain newlines); clients read it from the IPC
    # stream.
    return StreamingResponse(
        _iter_blob(),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-Arrow-Row-Count": str(artifact.row_count or 0),
            **data_headers(),
        },
    )


# --- Lineage and dependency introspection ---


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
    """Get the transitive input graph (artifacts and tables) of an artifact, up to ``max_depth``."""
    # Answered by the team store when one is configured: the dashboard opens lineage from views that
    # list the team's registry, so the local store would 404 on exactly what the reader clicked.
    target = remote_registry()
    if target is not None:
        return await relay(
            target,
            "GET",
            f"/v1/artifacts/{quoted(artifact_id)}/v/{version}/lineage",
            params={"max_depth": max_depth},
        )

    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    # Same table-ACL re-check as the sibling read endpoints: the graph carries every upstream table
    # URI, pinned snapshot and transform ref, most of what a deny rule withholds.
    _authorize_artifact_read(artifact, store)

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
    """List ready artifacts that use this artifact as a direct input (not transitive)."""
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    artifact = _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )
    _authorize_artifact_read(artifact, store)

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
    """Upload an artifact's Arrow IPC bytes (personal mode only).

    Call ``/v1/artifacts/finalize`` afterwards to complete the artifact.
    """
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
            async for chunk in _capped(request).stream():
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
    """Mark an uploaded artifact ready, optionally setting a name (personal mode only)."""
    if not store.blob_exists(request.artifact_id, request.version):
        raise HTTPException(
            status_code=400,
            detail="Blob not uploaded. Call upload endpoint first.",
        )

    # ``blob_size`` returns None both for an absent object and a failed backend call. Refuse rather
    # than mark READY a row claiming an empty blob, as build-finalize does.
    byte_size = store.blob_size(request.artifact_id, request.version) or 0
    if byte_size == 0:
        raise HTTPException(status_code=500, detail="Failed to read uploaded blob")

    try:
        # Without a digest, finalize reads the whole blob to hash it.
        finalized_artifact = await asyncio.to_thread(
            store.finalize_artifact,
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

    if request.name:
        try:
            store.set_name(request.name, finalized_artifact.id, finalized_artifact.version)
            name_uri = f"strata://name/{request.name}"
        except ValueError as e:
            # A name failure does not fail the whole request.
            logger.warning(f"Failed to set name {request.name}: {e}")

    return UploadFinalizeResponse(
        artifact_uri=artifact_uri,
        byte_size=finalized_artifact.byte_size or byte_size,
        name_uri=name_uri,
    )

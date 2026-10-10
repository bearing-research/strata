"""Build status and pull-model manifest routes.

The manifest and finalize routes take ``BuildTransportStore`` (404 when signed
transport is off). ``get_build_status`` and the signed upload route resolve the
store in-body to keep their gate ordering. Server-owned helpers are lazy-imported.
"""

from __future__ import annotations

import functools
import os
import tempfile
import time
from hmac import compare_digest
from pathlib import Path
from typing import Any

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from strata.api.dependencies import (
    BuildTransportStore,
    build_transport_available,
    runtime_build_store,
)
from strata.artifact_store import BuildLeaseLost
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.services.build import build_service
from strata.transforms.signed_urls import lease_attempt, lease_token
from strata.types import BuildStatusResponse

router = APIRouter(tags=["builds"])

# Bounded so a runaway cell cannot move server memory through a display-only route.
_MAX_LOG_CHUNK_BYTES = 256 * 1024

# Marks a build handed to a pull-model executor, so BuildRunner won't also run it.
_EXTERNAL_LEASE_OWNER = "external:manifest"

_LEASE_LOST_UNPUBLISHED = "Build lease is no longer held by this caller; nothing was published"

# The lease must outlive every signed URL handed to the executor, or BuildRunner
# can reclaim the build while those URLs are still usable.
_LEASE_MARGIN_SECONDS = 60.0


@router.get("/v1/artifacts/builds/{build_id}", response_model=BuildStatusResponse)
def get_build_status(build_id: str):
    """Poll the state of an asynchronous build started by materialize."""
    from strata.server import (
        _authorize_build_access,
        _identity_build_status,
        get_state,
    )
    from strata.streaming import StreamState

    state = get_state()

    # Identity artifact-mode builds live in the stream registry even when
    # server-mode transforms are disabled.
    stream_state = state.streams.get(build_id)
    if isinstance(stream_state, StreamState):
        _authorize_build_access(
            owner_principal=stream_state.plan.owner_principal,
            owner_tenant=stream_state.plan.owner_tenant,
        )

        return _identity_build_status(stream_state)

    # Checked in-body, not as a param dependency: the identity path above must
    # answer even when transport is off.
    if not build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build polling is only available when personal-mode writes "
                "or server-mode transforms are enabled"
            ),
        )

    build_store = runtime_build_store()
    if build_store is None:
        raise HTTPException(
            status_code=500,
            detail="Build store not initialized",
        )

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

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
def get_build_status_compat(build_id: str):
    """Compatibility alias for older clients polling build status."""
    return get_build_status(build_id)


@router.get("/v1/builds/{build_id}/manifest")
def get_build_manifest(build_id: str, request: Request, build_store: BuildTransportStore):
    """Get the pull-model manifest of signed URLs for a build.

    It carries download URLs for each input, an upload URL for the output, and the
    finalize URL to call once the upload completes. A sync route: the claim, the presign
    (which can call the cloud for role credentials, IAM signBlob or a delegation key) and
    the attempt record all run in a worker thread.
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _authorize_build_access,
        _get_artifact_store,
        get_state,
    )

    state = get_state()

    # A manifest mints signed upload and finalize URLs, a write capability, so it follows the
    # write rules: open where writes_enabled (personal mode, as require_writes_enabled), and in
    # service mode only under trusted-proxy auth (as service_writes_enabled). Nothing binds the
    # uploaded bytes to the executor, so an unchecked caller could upload forged bytes that later
    # identical materializes serve as a dedup hit. Redeeming needs only the signature, since a
    # pull-model worker holds a capability, not a principal.
    if not state.config.writes_enabled and state.config.auth_mode != "trusted_proxy":
        raise HTTPException(
            status_code=404,
            detail=(
                "Build manifests are only issued with trusted-proxy auth in "
                "service mode (they carry upload/finalize capabilities)"
            ),
        )

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    if build.state not in _ACTIVE_BUILD_STATES:
        raise HTTPException(
            status_code=400,
            detail=f"Build is not in pending or building state (state={build.state})",
        )

    _authorize_build_access(
        owner_principal=build.principal_id,
        owner_tenant=build.tenant_id,
    )

    # Claim before handing out capabilities, or BuildRunner's poll loop also picks the
    # build up and two writers publish onto the same (artifact_id, version).
    # claim_build is atomic (UPDATE ... WHERE state = 'pending'), so exactly one of
    # manifest-issue and runner-claim wins; lease expiry reclaims an abandoned build.
    # The margin keeps the lease alive past every URL minted below, by construction.
    lease_seconds = state.config.signed_url_expiry_seconds + _LEASE_MARGIN_SECONDS
    if build.state == "pending":
        claimed = build_store.claim_build(
            build_id,
            lease_owner=_EXTERNAL_LEASE_OWNER,
            lease_duration_seconds=lease_seconds,
        )
        if not claimed:
            # The runner claimed it in between; don't hand out a second writer's capabilities.
            raise HTTPException(
                status_code=409,
                detail="Build was claimed by the local runner; retry is not safe",
            )
    elif build.lease_owner == _EXTERNAL_LEASE_OWNER:
        # Re-fetch of a manifest issued here: it mints fresh URLs, so push the lease out
        # to cover them, or the sweep could reclaim while the executor can still finalize.
        build_store.renew_lease(
            build_id,
            _EXTERNAL_LEASE_OWNER,
            lease_duration_seconds=lease_seconds,
        )
    elif build.lease_owner is not None:
        # BuildRunner holds the lease (it claims with its own runner id); capabilities now
        # would put a second writer on the same blob. A lease_owner of None is not the
        # runner: it is the deprecated start_build() path (the notebook's in-process
        # manifest) and legacy rows, which keep working.
        raise HTTPException(
            status_code=409,
            detail="Build is already being executed by the local runner",
        )

    # Never 403s: build_transport_available() above already passed.
    store = _get_artifact_store(allow_server_mode=True)
    base_url = str(request.base_url).rstrip("/")

    # Re-read: the claim or renewal above moved the lease deadline, and URLs signed
    # against the pre-claim row would all be rejected on arrival.
    leased = build_store.get_build(build_id) or build

    try:
        manifest = build_service.assemble_manifest(
            store,
            signer=state.url_signer,
            build=build,
            base_url=base_url,
            max_output_bytes=state.config.max_transform_output_bytes,
            url_expiry_seconds=state.config.signed_url_expiry_seconds,
            lease_owner=leased.lease_owner,
            lease_expires_at=leased.lease_expires_at,
            presign=state.config.artifact_presigned_urls,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A presigned upload URL never passes through here again, so record its attempt
    # key and write deadline before it leaves; an unfinalized upload is then findable.
    attempt = lease_attempt(lease_token(leased.lease_owner, leased.lease_expires_at))
    if attempt is not None:
        build_store.record_attempt(
            build_id,
            build.artifact_id,
            build.version,
            attempt,
            writable_until=time.time() + state.config.signed_url_expiry_seconds,
        )
    return manifest


@router.get("/v1/artifacts/download")
def download_artifact_signed(
    artifact_id: str,
    version: str,
    build_id: str,
    expires_at: str,
    signature: str,
):
    """Download an input artifact's Arrow IPC bytes via a Strata-signed, unexpired URL.

    A sync route, so its store and blob reads run in a worker thread.
    """
    from strata.server import _get_artifact_store, get_state

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

    store = _get_artifact_store(allow_server_mode=True)
    artifact = store.get_artifact(artifact_id, version_int)

    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    if artifact.state not in ("ready", "superseded"):
        raise HTTPException(
            status_code=400,
            detail=f"Artifact is not ready (state={artifact.state})",
        )

    reader_cm = store.open_blob_reader(artifact_id, version_int)
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
    attempt: str = "",
):
    """Upload a build's output Arrow IPC bytes via a Strata-signed, unexpired URL.

    The body may not exceed the signed ``max_bytes``. With a leased manifest, the
    signed ``attempt`` puts the bytes under that attempt's key, so an earlier
    manifest's URL cannot reach what finalize reads.
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _get_artifact_store,
        get_state,
    )

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
        attempt=attempt,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    def load_build():
        # Resolved in-body, not via RequiredBuildStore, so the signature check (the
        # authorization) runs before any store-500.
        build_store = runtime_build_store()
        if build_store is None:
            raise HTTPException(status_code=500, detail="Build store not initialized")
        return build_store.get_build(build_id)

    build = await anyio.to_thread.run_sync(load_build)
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
        await anyio.to_thread.run_sync(
            store.publish_blob_from_path,
            build.artifact_id,
            build.version,
            staged,
            attempt or None,
        )
    finally:
        staged.unlink(missing_ok=True)

    return {"status": "uploaded", "build_id": build_id, "byte_size": byte_size}


@router.post("/v1/builds/{build_id}/log")
async def append_build_log(
    build_id: str,
    expires_at: str,
    signature: str,
    request: Request,
    seq: int,
    stream: str = "stdout",
):
    """Append console chunk ``seq`` (from 0 per stream) of a build that is still running.

    Advisory, hence 202. In a multi-node deployment a chunk for a build another node
    dispatched is left in the shared build store for that node; otherwise one for a
    build this process is not running is accepted and dropped, since failing the
    worker's cell over a console chunk would be worse than a missing line.
    """
    from strata.notebook import console_relay
    from strata.server import get_state
    from strata.transforms.build_store import get_build_store

    try:
        expires_float = float(expires_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter format")

    if not get_state().url_signer.verify_log_signature(build_id, expires_float, signature):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Keep only the tail as it arrives; a runaway cell must not grow server memory
    # through a display route, so the whole body is never held.
    tail = bytearray()
    async for chunk in request.stream():
        tail += chunk
        del tail[:-_MAX_LOG_CHUNK_BYTES]

    text = tail.decode("utf-8", errors="replace")
    delivered = await console_relay.deliver(build_id, stream, seq, text)
    store = get_build_store()
    if not delivered and get_state().config.node_advertised_url and store is not None:
        delivered = await anyio.to_thread.run_sync(
            store.append_console_chunk,
            build_id,
            "stderr" if stream == "stderr" else "stdout",
            seq,
            text,
        )
    return JSONResponse(status_code=202, content={"delivered": delivered})


@router.post("/v1/builds/{build_id}/finalize")
async def finalize_build(
    build_id: str,
    request: Request,
    build_store: BuildTransportStore,
    expires_at: str | None = None,
    signature: str | None = None,
    lease: str = "",
):
    """Finalize a pull-model build after its output was uploaded.

    Verifies the blob, reads its Arrow schema and row count, finalizes the
    artifact, completes the build and optionally sets the name pointer.
    """
    from strata.server import (
        _ACTIVE_BUILD_STATES,
        _authorize_build_access,
        _get_artifact_store,
        get_state,
    )
    from strata.transforms.build_qos import record_build_output_bytes

    def admit() -> tuple[Any, Any]:
        """The build, and the claim this request's lease token was verified against, if any."""
        build = build_store.get_build(build_id)
        if build is None:
            raise HTTPException(status_code=404, detail="Build not found")

        current = None
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
                lease=lease,
            ):
                raise HTTPException(status_code=403, detail="Invalid or expired signature")

            # Ownership before publication: everything below writes, and the final fence can
            # only refuse the build row, not un-publish bytes. The signature proves this
            # server minted the capability; the lease token proves it was minted for the
            # claim current now, which a stale executor cannot satisfy. Read the row fresh.
            if lease:
                current = build_store.get_build(build_id)
                expected = lease_token(
                    current.lease_owner if current else None,
                    current.lease_expires_at if current else None,
                )
                if not compare_digest(lease, expected):
                    raise HTTPException(status_code=409, detail=_LEASE_LOST_UNPUBLISHED)
        else:
            _authorize_build_access(
                owner_principal=build.principal_id,
                owner_tenant=build.tenant_id,
            )

        if build.state not in _ACTIVE_BUILD_STATES:
            raise HTTPException(
                status_code=400,
                detail=f"Build is not in pending or building state (state={build.state})",
            )
        return build, current

    build, current = await anyio.to_thread.run_sync(admit)

    try:
        finalize_payload = await request.json()
        if not isinstance(finalize_payload, dict):
            finalize_payload = {}
    except Exception:
        finalize_payload = {}

    output_format = str(finalize_payload.get("output_format", "")).strip()

    # Under a lease, finalize reads only that claim's attempt key; an executor holding
    # an earlier manifest wrote somewhere else.
    attempt = lease_attempt(lease) if current is not None else None
    store = _get_artifact_store(allow_server_mode=True)
    claim = (
        (current.lease_owner, current.lease_expires_at)
        if current is not None and current.lease_owner is not None
        else None
    )

    def fail(message: str, code: str) -> None:
        build_store.fail_build(build_id, message, code)
        store.fail_artifact(build.artifact_id, build.version)

    # On S3, GCS or Azure each blob probe is a network round trip.
    if not await anyio.to_thread.run_sync(
        store.blob_exists, build.artifact_id, build.version, attempt
    ):
        raise HTTPException(
            status_code=400,
            detail="Blob not uploaded. Upload using the signed URL first.",
        )

    byte_size = (
        await anyio.to_thread.run_sync(store.blob_size, build.artifact_id, build.version, attempt)
        or 0
    )
    if byte_size == 0:
        raise HTTPException(status_code=500, detail="Failed to read uploaded blob")

    # A presigned upload bypasses the upload route's size check, so enforce it here.
    max_output_bytes = get_state().config.max_transform_output_bytes
    if byte_size > max_output_bytes:
        await anyio.to_thread.run_sync(
            fail,
            f"Output is {byte_size} bytes, over the {max_output_bytes}-byte limit",
            "OUTPUT_TOO_LARGE",
        )
        raise HTTPException(
            status_code=413,
            detail=f"Output exceeds maximum size: {byte_size} > {max_output_bytes}",
        )

    if output_format == "notebook-output-bundle@v1":

        def _validate_notebook_bundle() -> tuple[str, int]:
            from strata.notebook.remote_bundle import (
                read_notebook_output_bundle_manifest_path,
            )

            reader_cm = store.open_blob_reader(build.artifact_id, build.version, attempt)
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
            schema_json, row_count = await anyio.to_thread.run_sync(_validate_notebook_bundle)
        except Exception as e:
            await anyio.to_thread.run_sync(fail, str(e), "INVALID_NOTEBOOK_BUNDLE")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid notebook output bundle: {e}",
            )
    else:

        def _parse_arrow_stream() -> tuple[str, int]:
            import pyarrow.ipc as arrow_ipc

            reader_cm = store.open_blob_reader(build.artifact_id, build.version, attempt)
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
            schema_json, row_count = await anyio.to_thread.run_sync(_parse_arrow_stream)
        except Exception as e:
            await anyio.to_thread.run_sync(fail, str(e), "INVALID_ARROW_FORMAT")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Arrow IPC format: {e}",
            )

    # Under a lease, publishing and completing the build are one transaction fenced
    # on this request's claim; the check at the top still left a window.
    fence = None
    if claim is not None:
        claim_owner, claim_deadline = claim

        def fence(conn, artifact_id: str, version: int) -> bool:
            return build_store.complete_within(
                conn,
                build_id,
                artifact_id=artifact_id,
                version=version,
                lease_owner=claim_owner,
                lease_expires_at=claim_deadline,
                output_byte_count=byte_size,
            )

    try:
        # Finalize reads the blob back to hash it.
        finalized_artifact = await anyio.to_thread.run_sync(
            functools.partial(
                store.finalize_and_set_name,
                artifact_id=build.artifact_id,
                version=build.version,
                schema_json=schema_json,
                row_count=row_count,
                byte_size=byte_size,
                name=build.name,
                tenant=build.tenant,
                blob_attempt=attempt,
                fence=fence,
            )
        )
    except BuildLeaseLost:
        if attempt is not None:
            await anyio.to_thread.run_sync(
                store.delete_attempt_blob, build.artifact_id, build.version, attempt
            )
        raise HTTPException(status_code=409, detail=_LEASE_LOST_UNPUBLISHED)
    except ValueError as e:
        await anyio.to_thread.run_sync(fail, str(e), "FINALIZE_FAILED")
        raise HTTPException(status_code=400, detail=str(e))
    if finalized_artifact is None:
        await anyio.to_thread.run_sync(fail, "Failed to finalize artifact", "FINALIZE_FAILED")
        raise HTTPException(status_code=500, detail="Failed to finalize artifact")

    if fence is not None:
        # The fence completed the build in finalize's own transaction.
        await record_build_output_bytes(build.tenant_id, byte_size)
        return _finalized(build, finalized_artifact, byte_size, row_count)

    def record_completion() -> bool:
        # The pull model may finalize a build that is still pending.
        if build.state == "pending":
            build_store.start_build(build_id)
        if (finalized_artifact.id, finalized_artifact.version) != (
            build.artifact_id,
            build.version,
        ):
            build_store.update_build_output(
                build_id,
                finalized_artifact.id,
                finalized_artifact.version,
            )
        # Fenced on the owner seen at request start, so an executor whose lease was
        # reclaimed mid-flight cannot record over its successor. A ``lease_owner`` of None
        # stays unfenced for the deprecated ``start_build`` path and legacy rows.
        return build_store.complete_build(build_id, lease_owner=build.lease_owner)

    if not await anyio.to_thread.run_sync(record_completion):
        raise HTTPException(
            status_code=409,
            detail="Build lease is no longer held by this caller; result not recorded",
        )
    await record_build_output_bytes(build.tenant_id, byte_size)
    return _finalized(build, finalized_artifact, byte_size, row_count)


def _finalized(build: Any, artifact: Any, byte_size: int, row_count: int) -> dict:
    """The finalize route's response, for the build and what it produced."""
    return {
        "status": "finalized",
        "build_id": build.build_id,
        "artifact_uri": f"strata://artifact/{artifact.id}@v={artifact.version}",
        "name_uri": f"strata://name/{build.name}" if build.name else None,
        "byte_size": byte_size,
        "row_count": row_count,
    }

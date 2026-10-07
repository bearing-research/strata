"""Stream route: ``GET /v1/streams/{stream_id}`` serves a ``scan@v1`` materialize's Arrow IPC.

Runtime state (stream registry, scan builds, QoS admission) is reached through a lazy
``from strata.server import get_state``, so this module stays a leaf.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.fast_io import IncrementalIpcMerger
from strata.logging import get_logger
from strata.pool_metrics import get_pool_tracker
from strata.streaming import QoSRejected

if TYPE_CHECKING:
    from strata.server import ServerState

logger = get_logger(__name__)

router = APIRouter(tags=["streams"])


def _resolve_stream_owner(state: ServerState, stream_id: str) -> str | None:
    """URL of another node serving ``stream_id``, or None.

    None when single-node, unclaimed, expired, claimed by this node, or on a lookup failure,
    which degrades to 404 rather than 500.
    """
    node_url = state.config.node_advertised_url
    if not node_url:
        return None

    from strata.streaming.ownership import get_stream_ownership_store

    store = get_stream_ownership_store()
    if store is None:
        return None
    try:
        return store.resolve(stream_id, exclude_node_url=node_url)
    except Exception:
        logger.warning("stream_owner_lookup_failed", stream_id=stream_id, exc_info=True)
        return None


@router.get("/v1/streams/{stream_id}")
async def get_stream(stream_id: str, request: Request):
    """Stream Arrow IPC data for a materialize request while the artifact builds.

    404 when the stream is not found; 429 when the server is at capacity.
    """
    from strata.server import get_state

    state = get_state()

    stream_state = state.streams.get(stream_id)
    if stream_state is None:
        # A stream's plan and task are in-process and cannot move, so in a
        # multi-node deployment redirect to the sibling that holds it.
        owner_url = _resolve_stream_owner(state, stream_id)
        if owner_url is not None:
            logger.info("stream_redirected", stream_id=stream_id, owner=owner_url)
            # stream_id comes from the request path; a raw '?' or '#' would
            # turn the rest into a query or fragment. Keep it one path segment.
            target = f"{owner_url.rstrip('/')}/v1/streams/{quote(stream_id, safe='')}"
            return RedirectResponse(
                url=target,
                # 307 rather than 302: the method must survive the redirect.
                status_code=307,
            )
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    plan = stream_state.plan
    scan_id = plan.scan_id

    if scan_id not in state.scan_builds:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    if state.config.principal_auth_enabled:
        from strata.auth import get_principal

        principal = get_principal()
        if principal is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        # Principal ids are only unique within a tenant.
        if plan.owner_principal != principal.id or plan.owner_tenant != principal.tenant:
            if not principal.has_scope("admin:*"):
                if state.config.hide_forbidden_as_not_found:
                    raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
                raise HTTPException(status_code=403, detail="Access denied")

    state.streams.cancel_cleanup(stream_id)
    stream_state.started = True
    stream_state.started_at = time.time()

    # Every exit path below must release the admission token, and one that leaves the stream
    # unserved must re-arm the cleanup cancelled above, or the stream and its plan stay for good.
    try:
        admission = await state.qos.admit(plan, request, scan_id)
    except asyncio.CancelledError:
        state.streams.schedule_cleanup(stream_id, scan_id)
        raise
    except QoSRejected as exc:
        state.streams.schedule_cleanup(stream_id, scan_id)
        return JSONResponse(
            status_code=429,
            content={"error": exc.error, "tier": exc.tier},
            headers={"Retry-After": str(exc.retry_after)},
        )

    from strata.artifact_store import get_artifact_store

    try:
        store = get_artifact_store(state.config.artifact_dir)
    except BaseException:
        await admission.release()
        state.streams.schedule_cleanup(stream_id, scan_id)
        raise

    # No artifact store: stream a bounded pass-through from the fetcher. With
    # nothing to finalize, a client disconnect just ends the generator.
    if store is None:

        async def serve_passthrough():
            start_time = time.perf_counter()
            try:
                if not plan.tasks:
                    if plan.schema is not None:
                        sink = pa.BufferOutputStream()
                        writer = ipc.new_stream(sink, plan.schema)
                        writer.close()
                        yield sink.getvalue().to_pybytes()
                else:
                    merger = IncrementalIpcMerger() if len(plan.tasks) > 1 else None
                    for task in plan.tasks:
                        if time.perf_counter() - start_time > state.config.scan_timeout_seconds:
                            state.metrics.record_stream_abort_timeout()
                            raise RuntimeError(
                                f"Scan timed out after {state.config.scan_timeout_seconds}s"
                            )
                        with get_pool_tracker().track("fetch"):
                            chunk = await asyncio.get_running_loop().run_in_executor(
                                state._fetch_executor,
                                state.fetcher.fetch_as_stream_bytes,
                                task,
                            )
                        out = merger.feed(chunk) if merger is not None else chunk
                        if out:
                            yield out
                    if merger is not None:
                        tail = merger.finish()
                        if tail:
                            yield tail
                stream_state.completed = True
            finally:
                await admission.release()
                stream_state.completed_at = time.time()
                state.streams.schedule_cleanup(stream_id, scan_id)

        return StreamingResponse(
            serve_passthrough(),
            media_type="application/vnd.apache.arrow.stream",
        )

    # Shielded so a client disconnect never cancels the build. A handler cancel
    # (e.g. shutdown) frees the slot but leaves the build running. The slot gated the
    # scan, which is done: release it on every exit here, a store error included, not in
    # the generator's finally, so a client gone before iteration can't strand it.
    try:
        # Decouple the build from this client's read so a slow or dropped reader
        # cannot poison the cache entry: the background build writes row groups
        # straight to the blob and finalizes on its own, then we serve the blob.
        if stream_state.background_task is None:
            stream_state.background_task = asyncio.create_task(
                state.scan_builds.build_identity_artifact(state, stream_state)
            )
        await asyncio.shield(stream_state.background_task)
        artifact = store.get_artifact(stream_state.artifact_id, stream_state.artifact_version)
    finally:
        await admission.release()
        stream_state.completed_at = time.time()
        # serve_blob re-arms it when it ends.
        state.streams.schedule_cleanup(stream_id, scan_id)
    stream_state.completed = True

    if artifact is None or artifact.state not in ("ready", "superseded"):
        return JSONResponse(
            status_code=500,
            content={
                "error": "scan_build_failed",
                "detail": stream_state.error_message or "scan build failed",
            },
        )

    # A reader that drops mid-send surfaces as a cancel/close; the artifact is
    # already finalized, so only count it.
    reader_cm = await asyncio.to_thread(
        store.open_blob_reader, stream_state.artifact_id, stream_state.artifact_version
    )

    async def serve_blob():
        bytes_out = 0
        try:
            if reader_cm is not None:
                with reader_cm as blob:
                    while True:
                        chunk = await asyncio.to_thread(blob.read, BLOB_STREAM_CHUNK_BYTES)
                        if not chunk:
                            break
                        bytes_out += len(chunk)
                        yield chunk
            stream_state.bytes_streamed = bytes_out
        except (asyncio.CancelledError, GeneratorExit):
            state.metrics.record_client_disconnect()
            raise
        finally:
            state.streams.schedule_cleanup(stream_id, scan_id)

    return StreamingResponse(
        serve_blob(),
        media_type="application/vnd.apache.arrow.stream",
        headers={
            "X-Arrow-Row-Count": str(artifact.row_count or 0),
            # The canonical artifact: when two misses for one scan were in flight, finalize
            # superseded this stream's own artifact and the materialize response's URI is not it.
            "X-Strata-Artifact-Uri": f"strata://artifact/{artifact.id}@v={artifact.version}",
        },
    )

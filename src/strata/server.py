"""FastAPI server for Strata."""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

if TYPE_CHECKING:
    from strata.adaptive_concurrency import AdaptiveConcurrencyController
    from strata.transforms.build_qos import BuildSlot

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from strata.adaptive_concurrency import ResizableLimiter
from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    PersonalModeStore,
    ReadStore,
    RegistryDecisionContext,
    WriteStore,
    require_notebook_worker_admin,
    require_scope,
)
from strata.auth import (
    AuthError,
    get_principal,
    parse_principal,
    set_principal,
    verify_proxy_token,
)
from strata.blob_store import BLOB_STREAM_CHUNK_BYTES
from strata.cache import CachedFetcher
from strata.cache_metrics import get_eviction_tracker
from strata.cache_stats import get_cache_histogram
from strata.cache_warmer import CacheWarmer
from strata.config import StrataConfig
from strata.fast_io import (
    IncrementalIpcMerger,
    validate_ipc_stream_reader,
)
from strata.gc_tracker import get_gc_stats, get_recent_gc_pauses, install_gc_tracker
from strata.health import HealthStatus, run_health_checks
from strata.json_types import JsonValue
from strata.logging import (
    configure_logging,
    get_logger,
    request_context_middleware,
)
from strata.memory_profiler import get_detailed_memory_report, get_memory_snapshot
from strata.metrics import MetricsCollector
from strata.notebook.models import WorkerBackendType, WorkerConfig, WorkerSpec
from strata.notebook.workers import (
    ManagedWorkerRecord,
    build_server_worker_catalog_with_health,
    create_server_managed_worker_record,
    delete_server_managed_worker_record,
    get_server_managed_worker_records,
    replace_server_managed_worker_records,
    set_server_managed_worker_enabled,
    update_server_managed_worker_record,
)
from strata.planner import ReadPlanner
from strata.pool_metrics import get_connection_metrics, get_pool_tracker
from strata.rate_limiter import (
    RateLimitConfig,
    get_rate_limiter,
    init_rate_limiter,
)
from strata.services.artifact import artifact_service
from strata.services.build import build_service
from strata.services.registry import registry_service
from strata.slow_ops import get_latency_stats
from strata.tenant import (
    DEFAULT_TENANT_ID,
    clear_tenant_context,
    get_tenant_id,
    set_tenant_id,
    validate_tenant_id,
)
from strata.tenant_registry import get_tenant_registry
from strata.tracing import init_tracing, instrument_fastapi, trace_span
from strata.types import (
    ArtifactDependentsResponse,
    ArtifactInfoResponse,
    ArtifactLineageResponse,
    BuildSpec,
    BuildStatusResponse,
    ExplainMaterializeRequest,
    ExplainMaterializeResponse,
    IdentityParams,
    InputChangeInfo,
    MaterializeRequest,
    MaterializeResponse,
    NameResolveResponse,
    NameSetRequest,
    NameSetResponse,
    NameStatusResponse,
    PutArtifactResponse,
    ReadPlan,
    TableRef,
    Task,
    UploadFinalizeRequest,
    UploadFinalizeResponse,
    WarmAsyncRequest,
    WarmAsyncResponse,
    WarmJobProgress,
    WarmJobStatus,
    WarmRequest,
    WarmResponse,
)

logger = get_logger(__name__)

# Graceful shutdown configuration
DRAIN_TIMEOUT_SECONDS = 30  # Max time to wait for active scans to complete

# Readiness probe thresholds
SATURATION_THRESHOLD_SECONDS = 30.0  # Fail readiness if saturated for this long
STUCK_SCAN_THRESHOLD_SECONDS = 60.0  # Fail readiness if scan makes no progress for this long


class AdminNotebookWorkerEntryRequest(BaseModel):
    """Service-managed notebook worker config entry."""

    name: str = Field(..., pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")
    backend: WorkerBackendType = Field(default=WorkerBackendType.LOCAL)
    runtime_id: str | None = Field(default=None)
    config: WorkerConfig = Field(default_factory=WorkerConfig)
    enabled: bool = True

    def to_worker_spec(self) -> WorkerSpec:
        return WorkerSpec(
            name=self.name,
            backend=self.backend,
            runtime_id=self.runtime_id,
            config=self.config,
        )


class AdminNotebookWorkersRequest(BaseModel):
    """Request payload for replacing the server-managed notebook worker registry."""

    workers: list[AdminNotebookWorkerEntryRequest] = Field(default_factory=list)


class AdminNotebookWorkerPatchRequest(BaseModel):
    """Request payload for patching one service-managed worker."""

    enabled: bool


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""

    pass


def _eager_warmup(config: StrataConfig) -> dict:
    """Eagerly warm up expensive resources at startup.

    This eliminates cold-start latency by pre-initializing:
    1. GC pause tracking (must be first to catch early GC events)
    2. Arrow memory pool configuration (must be done before any Arrow ops)
    3. Heavy module imports (pyiceberg, pyarrow)
    4. SQLite metadata store (connection + schema validation)
    5. Memory-resident caches

    Returns timing information for observability.
    """
    warmup_times = {}
    total_start = time.perf_counter()

    # 0. Install GC pause tracker FIRST (to catch all GC events including during warmup)
    # This gives us precise pause duration measurements for diagnosing latency stalls.
    install_gc_tracker()
    warmup_times["gc_tracker"] = True

    # 1. Configure Arrow memory pool BEFORE any Arrow allocations
    # This must happen before importing pyarrow.parquet which triggers allocations
    try:
        pool_name = config.configure_arrow_memory_pool()
        if pool_name:
            warmup_times["arrow_memory_pool"] = pool_name
    except ValueError as e:
        warmup_times["arrow_memory_pool_error"] = str(e)

    # 1. Pre-import heavy modules (these are already imported at module level,
    #    but we explicitly touch them to ensure all submodules are loaded)
    import_start = time.perf_counter()
    import pyarrow.parquet  # noqa: F401 - heavy, loads libparquet
    from pyiceberg.catalog.sql import SqlCatalog  # noqa: F401 - loads SQLAlchemy
    from pyiceberg.table import Table  # noqa: F401 - loads table machinery

    warmup_times["imports_ms"] = (time.perf_counter() - import_start) * 1000

    # 2. Initialize and warm up SQLite metadata store
    sqlite_start = time.perf_counter()
    try:
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store(config.cache_dir)
        # Run a simple query to warm up SQLite (page cache, WAL)
        stats = store.stats()
        warmup_times["sqlite_ms"] = (time.perf_counter() - sqlite_start) * 1000
        warmup_times["sqlite_entries"] = stats.get("parquet_meta_entries", 0)
    except Exception:
        warmup_times["sqlite_ms"] = (time.perf_counter() - sqlite_start) * 1000
        warmup_times["sqlite_error"] = True

    # 3. Pre-create in-memory caches (backed by SQLite)
    cache_start = time.perf_counter()
    from strata.metadata_cache import get_manifest_cache, get_parquet_cache

    get_parquet_cache(cache_dir=config.cache_dir)
    get_manifest_cache(cache_dir=config.cache_dir)
    warmup_times["caches_ms"] = (time.perf_counter() - cache_start) * 1000

    warmup_times["total_ms"] = (time.perf_counter() - total_start) * 1000
    return warmup_times


class ServerState:
    """Shared server state."""

    def __init__(self, config: StrataConfig) -> None:
        import os
        from concurrent.futures import ThreadPoolExecutor

        self.config = config

        # Dedicated thread pool for planning operations.
        # The default executor has only 8-16 workers (min(32, cpu_count + 4)),
        # which becomes a bottleneck under high concurrency (50+ users).
        # Planning involves reading Parquet metadata from disk/cache, so we use
        # a larger pool to handle concurrent planning requests without queueing.
        self._planning_executor = ThreadPoolExecutor(
            max_workers=64,
            thread_name_prefix="strata-planner",
        )

        # Dedicated thread pool for row group fetch operations.
        # Capped by max_fetch_workers to bound total I/O concurrency.
        # Default 32 workers is tuned for typical 8-16 core boxes.
        # Increase to 64 for high-core-count servers with fast storage.
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=config.max_fetch_workers,
            thread_name_prefix="strata-fetch",
        )
        # Check if metrics logging is disabled via environment
        metrics_enabled = os.environ.get("STRATA_METRICS_ENABLED", "true").lower() != "false"
        self.metrics = MetricsCollector(enabled=metrics_enabled)
        self.planner = ReadPlanner(config)
        self.fetcher = CachedFetcher(config, metrics=self.metrics)

        # Active scans (scan_id -> ReadPlan)
        self.scans: dict[str, ReadPlan] = {}

        # QoS: Two-tier admission control with ResizableLimiters
        # This prevents bulk queries from starving interactive (dashboard) queries.
        # Interactive: small, fast queries (dashboards) - get dedicated slots
        # Bulk: large, slow queries (ETL, exports) - separate pool
        # Using ResizableLimiter instead of Semaphore for correct dynamic resizing
        from strata.adaptive_concurrency import ResizableLimiter

        self._interactive_limiter = ResizableLimiter(config.interactive_slots)
        self._bulk_limiter = ResizableLimiter(config.bulk_slots)

        # Track which tier each scan is using for proper cleanup
        self._scan_tier: dict[str, str] = {}  # scan_id -> "interactive" | "bulk"
        # Track per-client semaphore association for cleanup
        # scan_id -> (client_id, semaphore_acquired)
        self._scan_client: dict[str, tuple[str, bool]] = {}

        # Legacy semaphore kept for backwards compatibility in metrics
        # but no longer used for admission control
        self._scan_semaphore = asyncio.Semaphore(config.max_concurrent_scans)

        # Approximate active scan counter for observability only.
        # Note: This is not thread-safe in async context (+=/-= are not atomic).
        # It's accurate enough for metrics/logging but should NOT be used for
        # control flow decisions. For authoritative count, derive from semaphores.
        self._active_scans = 0
        self._active_interactive = 0
        self._active_bulk = 0

        # Graceful shutdown state
        self._draining = False  # True when server is shutting down
        self._shutdown_event = asyncio.Event()  # Signaled when shutdown begins

        # QoS rejection counters (when queue deadline exceeded)
        self._interactive_rejected = 0  # Interactive queries rejected (429)
        self._bulk_rejected = 0  # Bulk queries rejected (429)

        # QoS queue wait tracking (for observability)
        self._interactive_queue_wait_total_ms = 0.0  # Cumulative wait time
        self._interactive_queue_wait_count = 0  # Number of requests that waited
        self._bulk_queue_wait_total_ms = 0.0
        self._bulk_queue_wait_count = 0

        # Per-client fairness: prevent one client from monopolizing capacity
        # Uses LRU dict of client_id -> Semaphore for each tier
        # Clients must acquire their per-client semaphore BEFORE global semaphore
        self._client_interactive_semaphores: dict[str, asyncio.Semaphore] = {}
        self._client_bulk_semaphores: dict[str, asyncio.Semaphore] = {}
        self._client_semaphore_max_entries = 10000  # LRU eviction threshold
        self._client_rejected = 0  # Rejections due to per-client cap

        # Prefetch management: limit concurrent prefetches to avoid resource exhaustion
        # when clients spam POST /scan without consuming the streams.
        # Max 4 concurrent prefetches (independent of streaming concurrency).
        self._prefetch_semaphore = asyncio.Semaphore(4)
        # Track prefetch tasks by scan_id for cancellation on stream cleanup.
        self._prefetch_futures: dict[str, asyncio.Task[None]] = {}
        # Prefetch metrics for observability
        self._prefetch_started = 0  # Total prefetches started
        self._prefetch_used = 0  # Prefetches consumed by streaming
        self._prefetch_wasted = 0  # Prefetches discarded (scan deleted/abandoned)
        self._prefetch_skipped = 0  # Prefetches skipped (server busy)
        self._prefetch_in_flight = 0  # Prefetches actively fetching

        # Readiness tracking for capacity-based health checks
        # Track when each tier became saturated (no slots available)
        self._interactive_saturated_since: float | None = None
        self._bulk_saturated_since: float | None = None
        # Track scan progress: scan_id -> (start_time, last_bytes_streamed)
        self._scan_progress: dict[str, tuple[float, int]] = {}

        # Register thread pools for metrics tracking
        pool_tracker = get_pool_tracker()
        pool_tracker.register_pool("planning", self._planning_executor)
        pool_tracker.register_pool("fetch", self._fetch_executor)

        # Cache warmer for background warming jobs (initialized async in lifespan)
        self._cache_warmer: CacheWarmer | None = None

        # Adaptive concurrency controller (initialized async in lifespan)
        self._adaptive_controller: AdaptiveConcurrencyController | None = None

        # Unified materialize streaming state
        # Maps stream_id -> StreamState for active streams
        self._streams: dict[str, StreamState] = {}
        self._stream_cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        # TTL for stream entries (default 5 minutes after creation, then cleaned up)
        self._stream_ttl_seconds = config.stream_state_ttl_seconds


@dataclass
class StreamState:
    """State for a streaming materialize operation.

    Tracks the read plan, streaming progress, and artifact metadata
    for a unified materialize request in stream mode.
    """

    stream_id: str
    plan: ReadPlan  # The underlying scan plan
    artifact_id: str  # Artifact being built
    artifact_version: int
    created_at: float  # Unix timestamp
    mode: str = "stream"  # "stream" for client streaming, "artifact" for background build
    name: str | None = None
    tenant: str | None = None
    executor_ref: str = "scan@v1"
    started: bool = False  # True once streaming has begun
    completed: bool = False  # True once streaming finished
    bytes_streamed: int = 0  # Bytes streamed to client so far
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    background_task: asyncio.Task[None] | None = None
    build_slot: BuildSlot | None = None
    qos_tenant_id: str | None = None


# Global state (initialized in lifespan)
_state: ServerState | None = None


def get_state() -> ServerState:
    if _state is None:
        raise RuntimeError("Server not initialized")
    return _state


def _is_signed_finalize_request(request: Request) -> bool:
    """Return True when the request is using the signed finalize contract."""
    return bool(
        re.fullmatch(r"/v1/builds/[^/]+/finalize", request.url.path)
        and "signature" in request.query_params
        and "expires_at" in request.query_params
    )


def _is_signed_data_plane_request(request: Request) -> bool:
    """Return True for pull-model data-plane requests that self-authenticate."""
    return request.url.path in (
        "/v1/artifacts/download",
        "/v1/artifacts/upload",
    ) or _is_signed_finalize_request(request)


def _deny_build_access() -> None:
    """Raise the configured build access error."""
    state = get_state()
    if state.config.hide_forbidden_as_not_found:
        raise HTTPException(status_code=404, detail="Build not found")
    raise HTTPException(status_code=403, detail="Access denied")


def _authorize_build_access(
    *,
    owner_principal: str | None,
    owner_tenant: str | None,
) -> None:
    """Authorize access to a build or identity stream under trusted proxy auth."""
    state = get_state()
    if state.config.auth_mode != "trusted_proxy":
        return

    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if principal.has_scope("admin:*"):
        return
    if owner_principal is None and owner_tenant is None:
        _deny_build_access()
    if owner_principal is not None and owner_principal != principal.id:
        _deny_build_access()
    if owner_tenant is not None and principal.tenant != owner_tenant:
        _deny_build_access()


def _cancel_stream_cleanup(state: ServerState, stream_id: str) -> None:
    """Cancel any pending cleanup task for a stream."""
    task = state._stream_cleanup_tasks.pop(stream_id, None)
    if task is not None:
        task.cancel()


def _schedule_stream_cleanup(
    state: ServerState,
    stream_id: str,
    scan_id: str | None = None,
) -> None:
    """Remove completed or abandoned stream state after the configured TTL."""
    _cancel_stream_cleanup(state, stream_id)

    async def _cleanup() -> None:
        try:
            await asyncio.sleep(state._stream_ttl_seconds)
        except asyncio.CancelledError:
            return
        finally:
            if state._stream_cleanup_tasks.get(stream_id) is asyncio.current_task():
                state._stream_cleanup_tasks.pop(stream_id, None)

        if scan_id is not None:
            _discard_prefetch(state, scan_id, count_wasted=True)
        state._streams.pop(stream_id, None)
        if scan_id is not None:
            state.scans.pop(scan_id, None)

    state._stream_cleanup_tasks[stream_id] = asyncio.create_task(_cleanup())


def _mark_stream_artifact_failed(stream_state: StreamState) -> None:
    """Best-effort transition a stream-backed artifact to failed state."""
    from strata.artifact_store import get_artifact_store

    state = get_state()
    store = get_artifact_store(state.config.artifact_dir)
    if store is None:
        return

    try:
        store.fail_artifact(stream_state.artifact_id, stream_state.artifact_version)
    except Exception:
        pass


def _discard_prefetch(state: ServerState, scan_id: str, *, count_wasted: bool) -> None:
    """Cancel or discard any prefetched first chunk for a scan."""
    plan = state.scans.get(scan_id)
    prefetched_ready = plan is not None and plan.prefetched_first is not None
    task = state._prefetch_futures.pop(scan_id, None)

    if task is not None and not task.done():
        task.cancel()
        if count_wasted:
            state._prefetch_wasted += 1
    elif prefetched_ready and count_wasted:
        state._prefetch_wasted += 1

    if plan is not None:
        plan.prefetched_first = None


def _start_prefetch(state: ServerState, plan: ReadPlan) -> None:
    """Best-effort prefetch of the first row group for stream-mode reads."""
    if (
        not plan.tasks
        or plan.scan_id in state._prefetch_futures
        or plan.prefetched_first is not None
    ):
        return

    async def _prefetch() -> None:
        if state._draining:
            return

        # Prefetch is opportunistic; skip instead of queueing behind existing work.
        if getattr(state._prefetch_semaphore, "_value", 0) <= 0:
            state._prefetch_skipped += 1
            return

        await state._prefetch_semaphore.acquire()
        state._prefetch_started += 1
        state._prefetch_in_flight += 1
        try:
            loop = asyncio.get_running_loop()
            plan.prefetched_first = await loop.run_in_executor(
                state._fetch_executor,
                state.fetcher.fetch_as_stream_bytes,
                plan.tasks[0],
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "prefetch_failed",
                scan_id=plan.scan_id,
                error=str(e),
            )
        finally:
            state._prefetch_in_flight -= 1
            state._prefetch_semaphore.release()

    task = asyncio.create_task(_prefetch())
    state._prefetch_futures[plan.scan_id] = task

    def _cleanup_prefetch(done_task: asyncio.Task[None]) -> None:
        if state._prefetch_futures.get(plan.scan_id) is done_task:
            state._prefetch_futures.pop(plan.scan_id, None)

    task.add_done_callback(_cleanup_prefetch)


def _normalized_build_qos_tenant_id(tenant_id: str | None) -> str:
    """Normalize tenant IDs for build QoS accounting."""
    return tenant_id or "__default__"


def _estimate_transform_output_bytes(
    state: ServerState,
    max_output_bytes: int | None,
) -> int:
    """Choose the best available output-size estimate for build admission."""
    if max_output_bytes is not None and max_output_bytes > 0:
        return max_output_bytes
    return state.config.build_runner_default_max_output


async def _record_build_output_bytes(tenant_id: str | None, bytes_produced: int) -> None:
    """Best-effort quota accounting for completed builds."""
    from strata.transforms.build_qos import get_build_qos

    build_qos = get_build_qos()
    if build_qos is None:
        return

    await build_qos.record_bytes(
        _normalized_build_qos_tenant_id(tenant_id),
        max(0, bytes_produced),
    )


def _identity_build_status(stream_state: StreamState) -> BuildStatusResponse:
    """Project an identity stream/background build onto the build-status contract."""
    from strata.artifact_store import get_artifact_store

    state = get_state()
    artifact_uri = f"strata://artifact/{stream_state.artifact_id}@v={stream_state.artifact_version}"

    artifact_state = None
    store = get_artifact_store(state.config.artifact_dir)
    if store is not None:
        artifact = store.get_artifact(stream_state.artifact_id, stream_state.artifact_version)
        if artifact is not None:
            artifact_state = artifact.state

    build_state = build_service.derive_build_state(
        error_message=stream_state.error_message,
        completed=stream_state.completed,
        started=stream_state.started,
        artifact_state=artifact_state,
    )

    return BuildStatusResponse(
        build_id=stream_state.stream_id,
        artifact_id=stream_state.artifact_id,
        version=stream_state.artifact_version,
        state=build_state,
        artifact_uri=artifact_uri,
        executor_ref=stream_state.executor_ref,
        created_at=stream_state.created_at,
        started_at=stream_state.started_at,
        completed_at=stream_state.completed_at,
        error_message=stream_state.error_message,
    )


async def _build_identity_artifact(stream_state: StreamState) -> None:
    """Build a scan@v1 artifact in the background for artifact-mode requests."""
    state = get_state()
    plan = stream_state.plan

    stream_state.started = True
    stream_state.started_at = time.time()

    try:
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store(state.config.artifact_dir)
        if store is None:
            return  # No artifact store in service mode

        if not plan.tasks:
            if plan.schema is not None:
                sink = pa.BufferOutputStream()
                writer = ipc.new_stream(sink, plan.schema)
                writer.close()
                empty_stream = sink.getvalue().to_pybytes()
            else:
                empty_stream = b""

            store.write_blob(stream_state.artifact_id, stream_state.artifact_version, empty_stream)
            await _finalize_written_blob(stream_state, 0, len(empty_stream))
            stream_state.bytes_streamed = len(empty_stream)
            stream_state.completed = True
            return

        loop = asyncio.get_running_loop()
        scan_id = plan.scan_id
        row_count = 0
        byte_size = 0
        start_time = time.perf_counter()

        # Write each merged row-group chunk straight to the blob (write-through,
        # bounded memory — no full-result buffer). The IncrementalIpcMerger emits
        # one valid IPC stream across the row groups so standard readers see every
        # row. The blob commits atomically when the writer context exits.
        merger = IncrementalIpcMerger() if len(plan.tasks) > 1 else None
        with store.open_blob_writer(
            stream_state.artifact_id, stream_state.artifact_version
        ) as blob:
            for index, task in enumerate(plan.tasks):
                if state._draining:
                    raise RuntimeError("Server is shutting down")

                # Bound the build's wall-clock the same way the old streaming
                # generator did — a runaway scan marks the artifact failed
                # rather than holding resources indefinitely.
                if time.perf_counter() - start_time > state.config.scan_timeout_seconds:
                    state.metrics.record_stream_abort_timeout()
                    raise RuntimeError(f"Scan timed out after {state.config.scan_timeout_seconds}s")

                # Consume the eagerly-prefetched first row group when stream mode
                # warmed one (no-op in artifact mode, which never prefetches).
                chunk: bytes | None = None
                if index == 0 and plan.prefetched_first is not None:
                    chunk = plan.prefetched_first
                    plan.prefetched_first = None
                    state._prefetch_used += 1
                elif index == 0:
                    prefetch_task = state._prefetch_futures.get(scan_id)
                    if prefetch_task is not None:
                        try:
                            await asyncio.wait_for(asyncio.shield(prefetch_task), timeout=0.05)
                        except TimeoutError:
                            pass
                        if plan.prefetched_first is not None:
                            chunk = plan.prefetched_first
                            plan.prefetched_first = None
                            state._prefetch_used += 1
                        else:
                            _discard_prefetch(state, scan_id, count_wasted=True)

                if chunk is None:
                    chunk = await loop.run_in_executor(
                        state._fetch_executor,
                        state.fetcher.fetch_as_stream_bytes,
                        task,
                    )
                out = merger.feed(chunk) if merger is not None else chunk
                if out:
                    blob.write(out)
                    byte_size += len(out)
                row_count += task.num_rows

            if merger is not None:
                tail = merger.finish()
                if tail:
                    blob.write(tail)
                    byte_size += len(tail)

        await _finalize_written_blob(stream_state, row_count, byte_size)
        stream_state.bytes_streamed = byte_size
        stream_state.completed = True
    except asyncio.CancelledError:
        stream_state.error_message = "Build cancelled"
        _mark_stream_artifact_failed(stream_state)
        raise
    except Exception as e:
        stream_state.error_message = str(e)
        logger.error(
            "identity_artifact_build_error",
            artifact_id=stream_state.artifact_id,
            error=str(e),
        )
        _mark_stream_artifact_failed(stream_state)
    finally:
        from strata.artifact_store import get_artifact_store

        artifact_ready = False
        store = get_artifact_store(state.config.artifact_dir)
        if store is not None:
            artifact = store.get_artifact(stream_state.artifact_id, stream_state.artifact_version)
            artifact_ready = artifact is not None and artifact.state == "ready"

        if stream_state.completed and stream_state.error_message is None and artifact_ready:
            await _record_build_output_bytes(
                stream_state.qos_tenant_id,
                stream_state.bytes_streamed,
            )
        if stream_state.build_slot is not None:
            await stream_state.build_slot.release()
        stream_state.completed_at = time.time()
        _schedule_stream_cleanup(state, stream_state.stream_id)


def require_writes_enabled() -> None:
    """FastAPI dependency that requires write endpoints to be enabled.

    In service mode (default), write endpoints return 403 with writes_disabled error.
    In personal mode, write endpoints are enabled.

    Raises:
        HTTPException: 403 if writes are disabled (service mode)
    """
    state = get_state()
    if not state.config.writes_enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "writes_disabled",
                "message": (
                    "Write endpoints are disabled in service mode. "
                    "Set deployment_mode='personal' for local development."
                ),
            },
        )


def _get_active_scan_count() -> int:
    """Get authoritative active scan count from the admission limiters.

    Stream admission acquires the per-tenant limiters from the tenant registry
    — the global ServerState limiters are never acquired — so the registry is
    the source of truth for in-flight scans (single-tenant deployments have
    exactly one, the ``_default`` tenant). Counting the global limiters here
    would always report 0 and let graceful shutdown drain past live streams.
    """
    i_in_use, _, b_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
    return i_in_use + b_in_use


def _classify_query(plan) -> str:
    """Classify a query as 'interactive' or 'bulk' based on its characteristics.

    Interactive queries are small, fast dashboard-style queries:
    - Estimated response size <= interactive_max_bytes (default 10MB)
    - Number of columns <= interactive_max_columns (default 10)

    Everything else is bulk (ETL, exports, analyst queries).
    """
    state = get_state()
    config = state.config

    # Check estimated response size
    if plan.estimated_bytes > config.interactive_max_bytes:
        return "bulk"

    # Check column count (None means all columns = likely bulk)
    if plan.columns is None:
        return "bulk"
    if len(plan.columns) > config.interactive_max_columns:
        return "bulk"

    return "interactive"


def _get_qos_metrics(state: ServerState) -> dict:
    """Get QoS tier metrics including queue wait times and per-tenant stats."""
    # Get global limiter stats (for backward compatibility and aggregate metrics)
    interactive_stats = state._interactive_limiter.get_stats()
    bulk_stats = state._bulk_limiter.get_stats()

    # Calculate average queue wait times
    interactive_avg_wait_ms = (
        state._interactive_queue_wait_total_ms / state._interactive_queue_wait_count
        if state._interactive_queue_wait_count > 0
        else 0.0
    )
    bulk_avg_wait_ms = (
        state._bulk_queue_wait_total_ms / state._bulk_queue_wait_count
        if state._bulk_queue_wait_count > 0
        else 0.0
    )

    # Per-tenant QoS metrics (only for tenants with active limiters)
    tenant_registry = get_tenant_registry()
    per_tenant_qos = {}
    with tenant_registry._lock:
        for tenant_id, quotas in tenant_registry._quotas.items():
            interactive = quotas.interactive_limiter
            bulk = quotas.bulk_limiter
            if isinstance(interactive, ResizableLimiter) and isinstance(bulk, ResizableLimiter):
                per_tenant_qos[tenant_id] = {
                    "interactive_capacity": interactive.capacity,
                    "interactive_in_use": interactive.in_use,
                    "bulk_capacity": bulk.capacity,
                    "bulk_in_use": bulk.in_use,
                }

    return {
        "interactive_slots": interactive_stats["capacity"],
        "interactive_active": interactive_stats["in_use"],
        "interactive_available": interactive_stats["available"],
        "interactive_rejected": state._interactive_rejected,
        "interactive_queue_timeout_seconds": state.config.interactive_queue_timeout,
        "interactive_queue_wait_avg_ms": round(interactive_avg_wait_ms, 2),
        "interactive_queue_wait_total_ms": round(state._interactive_queue_wait_total_ms, 2),
        "interactive_queue_wait_count": state._interactive_queue_wait_count,
        "bulk_slots": bulk_stats["capacity"],
        "bulk_active": bulk_stats["in_use"],
        "bulk_available": bulk_stats["available"],
        "bulk_rejected": state._bulk_rejected,
        "bulk_queue_timeout_seconds": state.config.bulk_queue_timeout,
        "bulk_queue_wait_avg_ms": round(bulk_avg_wait_ms, 2),
        "bulk_queue_wait_total_ms": round(state._bulk_queue_wait_total_ms, 2),
        "bulk_queue_wait_count": state._bulk_queue_wait_count,
        # Per-client fairness metrics
        "per_client_interactive": state.config.per_client_interactive,
        "per_client_bulk": state.config.per_client_bulk,
        "client_rejected": state._client_rejected,
        "tracked_clients": len(state._client_interactive_semaphores),
        # Per-tenant QoS metrics
        "per_tenant": per_tenant_qos,
    }


def _get_client_semaphore(
    state: ServerState, client_id: str, tier: str
) -> asyncio.Semaphore | None:
    """Get or create a per-client semaphore for the given tier.

    Returns None if per-client caps are disabled (set to 0).
    Uses simple LRU eviction when cache exceeds max entries.
    """
    if tier == "interactive":
        max_concurrent = state.config.per_client_interactive
        client_semaphores = state._client_interactive_semaphores
    else:
        max_concurrent = state.config.per_client_bulk
        client_semaphores = state._client_bulk_semaphores

    # 0 = disabled
    if max_concurrent <= 0:
        return None

    # Get existing or create new semaphore
    if client_id in client_semaphores:
        # Move to end for LRU (dict maintains insertion order in Python 3.7+)
        sem = client_semaphores.pop(client_id)
        client_semaphores[client_id] = sem
        return sem

    # Create new semaphore
    sem = asyncio.Semaphore(max_concurrent)
    client_semaphores[client_id] = sem

    # LRU eviction if too many clients tracked
    while len(client_semaphores) > state._client_semaphore_max_entries:
        # Remove oldest (first) entry
        oldest_client = next(iter(client_semaphores))
        del client_semaphores[oldest_client]

    return sem


def _get_cache_size_bytes(state: ServerState) -> int:
    """Get current cache size in bytes."""
    from strata.cache import DiskCache

    cache = state.fetcher.cache
    if isinstance(cache, DiskCache):
        return cache.get_size_bytes()
    return 0


def _get_cache_entry_count(state: ServerState) -> int:
    """Get current number of cache entries."""
    from strata.cache import DiskCache

    cache = state.fetcher.cache
    if isinstance(cache, DiskCache):
        return len(cache.list_entries())
    return 0


def _update_saturation_tracking(state: ServerState) -> None:
    """Update saturation tracking based on current semaphore state.

    Called periodically (e.g., from health checks) to track how long
    the server has been at capacity. This is used by /health/ready to
    detect unhealthy saturation conditions.
    """
    now = time.time()

    # Saturation is measured on the per-tenant admission limiters (what stream
    # admission actually acquires), not the never-acquired global limiters. A
    # server with no live limiters yet — no requests served — is idle, not
    # saturated, so require some in-use slots before flagging it.
    i_in_use, i_avail, b_in_use, b_avail = get_tenant_registry().aggregate_limiter_usage()

    # Check interactive tier saturation
    if i_in_use > 0 and i_avail == 0:
        if state._interactive_saturated_since is None:
            state._interactive_saturated_since = now
    else:
        state._interactive_saturated_since = None

    # Check bulk tier saturation
    if b_in_use > 0 and b_avail == 0:
        if state._bulk_saturated_since is None:
            state._bulk_saturated_since = now
    else:
        state._bulk_saturated_since = None


def _check_readiness(state: ServerState) -> tuple[bool, dict]:
    """Check if server is ready to accept new requests.

    Returns (is_ready, details) where details contains diagnostic info.

    Checks:
    1. Server not draining (shutting down)
    2. Has some capacity (not all slots exhausted for too long)
    3. No stuck scans (scans making no progress for too long)
    4. Logger queue not jammed (metrics can be written)
    """
    now = time.time()
    checks = {}
    issues = []

    # Update saturation tracking
    _update_saturation_tracking(state)

    # Check 1: Not draining
    if state._draining:
        checks["draining"] = True
        issues.append("server is draining (shutting down)")
    else:
        checks["draining"] = False

    # Check 2: Capacity - fail if BOTH tiers saturated for too long
    interactive_saturated_duration = (
        now - state._interactive_saturated_since if state._interactive_saturated_since else 0.0
    )
    bulk_saturated_duration = (
        now - state._bulk_saturated_since if state._bulk_saturated_since else 0.0
    )

    checks["interactive_saturated_seconds"] = round(interactive_saturated_duration, 1)
    checks["bulk_saturated_seconds"] = round(bulk_saturated_duration, 1)

    # Only fail if BOTH tiers are saturated beyond threshold
    # (if one tier has capacity, we can still serve some queries)
    both_saturated = (
        interactive_saturated_duration > SATURATION_THRESHOLD_SECONDS
        and bulk_saturated_duration > SATURATION_THRESHOLD_SECONDS
    )
    if both_saturated:
        checks["capacity_exhausted"] = True
        issues.append(
            f"both tiers saturated for >{SATURATION_THRESHOLD_SECONDS}s "
            f"(interactive={interactive_saturated_duration:.1f}s, "
            f"bulk={bulk_saturated_duration:.1f}s)"
        )
    else:
        checks["capacity_exhausted"] = False

    # Check 3: Stuck scans - find scans with no progress for too long
    stuck_scans = []
    for scan_id, (start_time, last_bytes) in list(state._scan_progress.items()):
        age = now - start_time
        if age > STUCK_SCAN_THRESHOLD_SECONDS:
            stuck_scans.append({"scan_id": scan_id, "age_seconds": round(age, 1)})

    checks["stuck_scans"] = len(stuck_scans)
    if stuck_scans:
        checks["stuck_scan_details"] = stuck_scans[:5]  # Limit to first 5
        issues.append(f"{len(stuck_scans)} scan(s) stuck with no progress")

    # Check 4: Logger queue health
    # If dropped_logs is increasing rapidly, the logger is overwhelmed
    dropped_logs = state.metrics.dropped_logs
    checks["dropped_logs"] = dropped_logs
    # Note: We don't fail on dropped_logs alone since it's a soft limit,
    # but we report it for observability

    # Overall readiness
    is_ready = len(issues) == 0
    checks["ready"] = is_ready
    if issues:
        checks["issues"] = issues

    return is_ready, checks


async def _graceful_shutdown(state: ServerState) -> None:
    """Wait for active scans to complete during shutdown."""
    state._draining = True
    state._shutdown_event.set()

    active = _get_active_scan_count()
    if active > 0:
        state.metrics.log_event(
            "shutdown_draining",
            active_scans=active,
            timeout_seconds=DRAIN_TIMEOUT_SECONDS,
        )

        # Wait for active scans to complete (with timeout)
        start = time.perf_counter()
        while _get_active_scan_count() > 0:
            elapsed = time.perf_counter() - start
            if elapsed > DRAIN_TIMEOUT_SECONDS:
                state.metrics.log_event(
                    "shutdown_timeout",
                    remaining_scans=_get_active_scan_count(),
                )
                break
            await asyncio.sleep(0.1)

        if _get_active_scan_count() == 0:
            state.metrics.log_event("shutdown_drained")

    for cleanup_task in list(state._stream_cleanup_tasks.values()):
        cleanup_task.cancel()
    state._stream_cleanup_tasks.clear()

    for stream_state in list(state._streams.values()):
        if stream_state.background_task is not None and not stream_state.background_task.done():
            stream_state.background_task.cancel()

    # Shutdown the executors
    state._planning_executor.shutdown(wait=False)
    state._fetch_executor.shutdown(wait=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server state on startup, graceful shutdown on exit."""
    global _state

    # Allow tests to pre-configure state before uvicorn starts
    # If state is already set, use its config instead of loading fresh
    if _state is not None:
        config = _state.config
    else:
        config = StrataConfig.load()

    # Validate personal mode binding safety before starting
    # This prevents accidental exposure of write endpoints to the network
    config.validate_personal_mode_binding()

    # Initialize transform registry from config
    from strata.transforms.registry import TransformRegistry, set_transform_registry

    transform_registry = TransformRegistry.from_config(config.transforms_config)
    set_transform_registry(transform_registry)

    # Pin the pull-model signing secret from config so signed build URLs survive
    # restarts and match across replicas. Without it the secret is a random
    # per-process value (set lazily on first use) — in-flight signed URLs die on
    # restart. Warn only when the pull model is actually in use.
    from strata.transforms.signed_urls import set_signing_secret

    if config.transform_signing_secret:
        set_signing_secret(config.transform_signing_secret.encode("utf-8"))
    elif config.pull_model_enabled:
        logger.warning(
            "pull_model_signing_secret_unset",
            detail=(
                "pull_model_enabled but transform_signing_secret is not set; using "
                "a random per-process secret. Signed build URLs will break on "
                "restart and differ across replicas. Set "
                "STRATA_TRANSFORM_SIGNING_SECRET for a stable deployment."
            ),
        )

    # Authenticated write-back is a preview feature in this release — re-opening
    # writes in service mode is security-sensitive. Surface it at startup so an
    # operator knows the surface is new and may change.
    if config.service_writes_enabled:
        logger.warning(
            "service_writes_enabled_preview",
            detail=(
                "service_writes_enabled is a PREVIEW feature: authenticated clients "
                "can write to this service-mode store (scope: artifacts:write, "
                "tenant-scoped). The surface is new and may change; evaluate before "
                "relying on it in production."
            ),
        )

    # Configure structured logging first
    configure_logging()

    # Initialize OpenTelemetry tracing (no-op if not installed/configured)
    tracing_enabled = init_tracing()

    # Eager warmup: pre-initialize expensive resources before accepting requests
    # This makes the first request as fast as subsequent "warm" requests
    warmup_times = _eager_warmup(config)

    # Create state only if not pre-configured (allows tests to inject custom state)
    if _state is None:
        _state = ServerState(config)

    # Initialize rate limiter
    rate_limit_config = RateLimitConfig(
        enabled=config.rate_limit_enabled,
        global_requests_per_second=config.rate_limit_global_rps,
        global_burst=config.rate_limit_global_burst,
        client_requests_per_second=config.rate_limit_client_rps,
        client_burst=config.rate_limit_client_burst,
        scan_requests_per_second=config.rate_limit_scan_rps,
        warm_requests_per_second=config.rate_limit_warm_rps,
    )
    init_rate_limiter(rate_limit_config)

    # Initialize cache warmer for background warming jobs
    _state._cache_warmer = CacheWarmer(
        planner=_state.planner,
        fetcher=_state.fetcher,
        metrics=_state.metrics,
    )
    await _state._cache_warmer.start()

    # Initialize adaptive concurrency controller (if enabled)
    from strata.adaptive_concurrency import AdaptiveConcurrencyController, AdaptiveConfig

    adaptive_config = AdaptiveConfig(
        enabled=config.adaptive_enabled,
        adjustment_interval_seconds=config.adaptive_interval_seconds,
        latency_target_p95_ms=config.adaptive_target_p95_ms,
        min_slots_interactive=config.adaptive_min_interactive,
        max_slots_interactive=config.adaptive_max_interactive,
        min_slots_bulk=config.adaptive_min_bulk,
        max_slots_bulk=config.adaptive_max_bulk,
        hysteresis_count=config.adaptive_hysteresis,
    )
    _state._adaptive_controller = AdaptiveConcurrencyController(
        config=adaptive_config,
        interactive_limiter=_state._interactive_limiter,
        bulk_limiter=_state._bulk_limiter,
    )
    await _state._adaptive_controller.start()

    # Cleanup stale metadata entries on startup
    stale_removed = 0
    try:
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store(config.cache_dir)
        stale_removed = store.cleanup_stale_parquet_meta()
    except Exception:
        pass  # Don't fail startup if cleanup fails

    # Sweep zombie builds (#123): artifacts stuck in 'building' — queued into
    # a mode with no executor, or left by a crashed builder — are demoted to
    # failed so they surface as failures instead of lingering forever.
    if config.writes_enabled or config.server_transforms_enabled:
        try:
            from strata.artifact_store import get_artifact_store as _get_store_for_sweep

            sweep_store = _get_store_for_sweep(config.artifact_dir)
            if sweep_store is not None:
                zombie_count = sweep_store.sweep_zombie_builds(
                    config.artifact_zombie_build_timeout_seconds
                )
                if zombie_count:
                    logger.warning("zombie_builds_swept", count=zombie_count)
        except Exception:
            pass  # Don't fail startup if sweep fails

    # Initialize build QoS for server-mode transforms (quotas + backpressure)
    build_qos = None
    if config.server_transforms_enabled:
        from strata.transforms.build_qos import BuildQoS, set_build_qos

        build_qos = BuildQoS(config.get_build_qos_config())
        set_build_qos(build_qos)

    # Initialize build metrics collector for observability
    if config.transforms_runtime_enabled:
        from strata.transforms.build_metrics import init_build_metrics

        init_build_metrics()

    # Initialize build runner. Service mode requires the explicit
    # transforms config; personal mode always runs embedded transforms
    # in-process so the artifact workflow works out of the box.
    build_runner = None
    if config.transforms_runtime_enabled:
        from strata.artifact_store import get_artifact_store
        from strata.transforms.build_store import get_build_store
        from strata.transforms.registry import get_transform_registry
        from strata.transforms.runner import (
            BuildRunner,
            RunnerConfig,
            set_build_runner,
        )

        # Determine artifact_dir for server-mode transforms
        artifact_dir = config.artifact_dir
        if artifact_dir is None:
            artifact_dir = Path.home() / ".strata" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Initialize stores
        artifact_store = get_artifact_store(artifact_dir)
        build_store = get_build_store(artifact_dir / "artifacts.sqlite")

        if artifact_store and build_store:
            runner_config = RunnerConfig(
                poll_interval_ms=config.build_runner_poll_interval_ms,
                max_concurrent_builds=config.build_runner_max_concurrent,
                max_builds_per_tenant=config.build_runner_max_per_tenant,
                default_timeout_seconds=config.build_runner_default_timeout,
                default_max_output_bytes=config.build_runner_default_max_output,
            )

            build_runner = BuildRunner(
                config=runner_config,
                artifact_store=artifact_store,
                build_store=build_store,
                transform_registry=get_transform_registry(),
                artifact_dir=artifact_dir,
                runtime_config=config,
                scan_planner=_state.planner,
                scan_fetcher=_state.fetcher,
            )
            set_build_runner(build_runner)
            await build_runner.start()

    # Log startup with warmup timing info
    _state.metrics.log_event(
        "server_started",
        host=config.host,
        port=config.port,
        notebook_storage_dir=str(config.notebook_storage_dir),
        warmup_ms=warmup_times.get("total_ms", 0),
        warmup_imports_ms=warmup_times.get("imports_ms", 0),
        warmup_sqlite_ms=warmup_times.get("sqlite_ms", 0),
        warmup_caches_ms=warmup_times.get("caches_ms", 0),
        sqlite_entries=warmup_times.get("sqlite_entries", 0),
        stale_entries_removed=stale_removed,
        tracing_enabled=tracing_enabled,
        arrow_memory_pool=warmup_times.get("arrow_memory_pool"),
        build_runner_enabled=build_runner is not None,
        build_qos_enabled=build_qos is not None,
    )

    yield

    # Reset build QoS
    from strata.transforms.build_qos import reset_build_qos

    reset_build_qos()

    # Stop build runner (cancel pending builds)
    from strata.transforms.runner import get_build_runner, reset_build_runner

    build_runner = get_build_runner()
    if build_runner:
        await build_runner.stop()
        reset_build_runner()

    # Stop adaptive controller (cancel background loop)
    if _state._adaptive_controller:
        await _state._adaptive_controller.stop()

    # Stop cache warmer (cancel background jobs)
    if _state._cache_warmer:
        await _state._cache_warmer.stop()

    # Graceful shutdown: wait for active scans to complete
    await _graceful_shutdown(_state)

    _state.metrics.log_event("server_stopped")
    _state = None

    # Reset transform registry
    from strata.transforms.registry import reset_transform_registry

    reset_transform_registry()


app = FastAPI(
    title="Strata",
    description="Snapshot-aware serving layer for Iceberg tables",
    version="0.2.0",
    lifespan=lifespan,
)

# Allow cross-origin requests (needed for dev and when frontend served separately)
app.add_middleware(
    cast(type, CORSMiddleware),
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request context middleware (sets request_id, adds to response headers)
app.middleware("http")(request_context_middleware)


# Connection tracking middleware
@app.middleware("http")
async def connection_tracking_middleware(request: Request, call_next):
    """Track HTTP connection metrics."""
    connection_metrics = get_connection_metrics()

    # Check for Connection: keep-alive header
    connection_header = request.headers.get("connection", "").lower()
    has_keepalive = connection_header != "close"

    connection_metrics.request_started(has_keepalive=has_keepalive)
    try:
        response = await call_next(request)
        return response
    finally:
        connection_metrics.request_completed()


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to incoming requests."""
    rate_limiter = get_rate_limiter()

    # Skip rate limiting if not initialized or for health/metrics endpoints
    if rate_limiter is None:
        return await call_next(request)

    path = request.url.path
    if path in ("/health", "/ready", "/metrics", "/v1/debug/pools", "/v1/debug/memory"):
        return await call_next(request)

    # Use client IP as identifier (X-Forwarded-For if behind proxy)
    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"

    result = rate_limiter.check(client_id=client_ip, endpoint=path)

    if not result.allowed:
        retry_after = int(result.retry_after_seconds or 1)
        return Response(
            content=f"Rate limit exceeded ({result.limit_type}). Retry after {retry_after}s.",
            status_code=429,
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit-Type": result.limit_type or "unknown",
            },
        )

    response = await call_next(request)

    # Add rate limit headers to response
    if result.tokens_remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(int(result.tokens_remaining))

    return response


# Tenant context middleware - sets tenant_id for multi-tenancy
@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    """Extract tenant ID and set up tenant context for multi-tenancy.

    Tenant identification priority:
    1. X-Tenant-ID header (simple header-based auth for MVP)
    2. Default tenant "_default" (backward compatibility)

    Returns 403 if tenant is disabled.
    Skips tenant setup for health/metrics endpoints.
    """
    # Skip tenant setup for health/metrics endpoints and signed data-plane routes.
    path = request.url.path
    if path in (
        "/health",
        "/health/ready",
        "/health/dependencies",
        "/metrics",
        "/metrics/prometheus",
    ) or _is_signed_data_plane_request(request):
        return await call_next(request)

    state = get_state()
    config = state.config

    # Only apply tenant context if multi-tenancy is enabled
    if not getattr(config, "multi_tenant_enabled", False):
        # Single-tenant mode: use default tenant
        set_tenant_id(DEFAULT_TENANT_ID)
        try:
            response = await call_next(request)
            return response
        finally:
            clear_tenant_context()

    # Multi-tenant mode: extract tenant from header
    tenant_header = getattr(config, "tenant_header", "X-Tenant-ID")
    tenant_id = request.headers.get(tenant_header)

    if not tenant_id:
        # Check if tenant header is required
        if getattr(config, "require_tenant_header", False):
            return Response(
                content=f"Missing required header: {tenant_header}",
                status_code=400,
            )
        # Use default tenant for backward compatibility
        tenant_id = DEFAULT_TENANT_ID
    else:
        # Validate tenant ID format (only for explicitly provided headers)
        is_valid, error_msg = validate_tenant_id(tenant_id)
        if not is_valid:
            return Response(
                content=f"Invalid tenant ID: {error_msg}",
                status_code=400,
            )

    # Validate tenant is enabled
    registry = get_tenant_registry()
    if not registry.is_tenant_enabled(tenant_id):
        return Response(
            content=f"Tenant '{tenant_id}' is not enabled",
            status_code=403,
        )

    # Set tenant context for this request
    set_tenant_id(tenant_id)
    try:
        response = await call_next(request)
        # Echo tenant ID back in response header for debugging
        response.headers["X-Tenant-ID"] = tenant_id
        return response
    finally:
        clear_tenant_context()


# Trusted proxy authentication middleware
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Verify trusted proxy and parse principal for authorization.

    When auth_mode="trusted_proxy":
    1. Verify X-Strata-Proxy-Token matches configured secret
    2. Parse X-Strata-Principal, X-Strata-Tenant, X-Strata-Scopes headers
    3. Set principal context for downstream use

    Skips auth for health/metrics endpoints.
    """
    state = get_state()
    config = state.config

    # Skip if auth is disabled
    if config.auth_mode == "none":
        return await call_next(request)

    # Skip auth for health/metrics endpoints and signed data-plane routes.
    path = request.url.path
    if path in (
        "/health",
        "/health/ready",
        "/health/dependencies",
        "/metrics",
        "/metrics/prometheus",
    ) or _is_signed_data_plane_request(request):
        return await call_next(request)

    # Verify proxy token
    proxy_token = request.headers.get(config.proxy_token_header)
    if not verify_proxy_token(proxy_token, config.proxy_token):
        logger.warning(
            "auth_failed",
            reason="invalid_proxy_token",
            path=path,
        )
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"},
        )

    # Parse principal from headers
    try:
        principal = parse_principal(dict(request.headers), config)
        set_principal(principal)
        # list[str] would be invariant against the structured-logger's
        # list[JsonValue] kwargs; cast widens the declared element type
        # without changing runtime behavior.
        scopes_json: list[JsonValue] = cast(list[JsonValue], list(principal.scopes))
        logger.debug(
            "auth_success",
            principal=principal.id,
            tenant=principal.tenant,
            scopes=scopes_json,
        )
    except AuthError as e:
        logger.warning(
            "auth_failed",
            reason="missing_principal",
            path=path,
        )
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.message},
        )

    try:
        response = await call_next(request)
        return response
    finally:
        set_principal(None)  # Clear principal context


# Instrument FastAPI with OpenTelemetry (no-op if OTel not installed)
instrument_fastapi(app)

# Register notebook routes
from strata.notebook import router as notebook_router  # noqa: E402
from strata.notebook.ws import router as notebook_ws_router  # noqa: E402

app.include_router(notebook_router)
app.include_router(notebook_ws_router)


@app.get("/health")
async def health():
    """Basic health check endpoint (liveness probe).

    Returns 200 if the server process is running.
    Use /health/ready for readiness checks that verify dependencies.
    """
    return {"status": "ok"}


@app.get("/health/dependencies")
async def health_dependencies():
    """Comprehensive health check for all dependencies.

    Checks the health of:
    - disk_cache: Cache directory accessibility and disk space
    - metadata_store: SQLite connectivity and entry counts
    - arrow_memory: PyArrow memory pool usage
    - thread_pools: Planning and fetch executor utilization
    - rate_limiter: Rate limiting status and rejection rate
    - cache_evictions: Cache eviction pressure level

    Each check returns:
    - status: healthy, degraded, or unhealthy
    - latency_ms: Time taken to perform the check
    - details: Check-specific information

    Overall status is the worst of all individual checks.

    Returns:
    - 200 if all dependencies are healthy
    - 200 with degraded status if some checks show degraded state
    - 503 if any dependency is unhealthy
    """
    state = get_state()

    report = run_health_checks(
        cache_dir=state.config.cache_dir,
        max_cache_size_bytes=state.config.max_cache_size_bytes,
        planning_executor=state._planning_executor,
        fetch_executor=state._fetch_executor,
    )

    status_code = 503 if report.status == HealthStatus.UNHEALTHY else 200

    return Response(
        content=__import__("json").dumps(report.to_dict()),
        status_code=status_code,
        media_type="application/json",
    )


@app.get("/health/ready")
async def health_ready():
    """Readiness probe - checks if server can handle requests.

    This is the Kubernetes readiness probe endpoint. Returns 503 when:
    - Server is draining (shutting down)
    - Both QoS tiers saturated for >30 seconds (no capacity)
    - Scans stuck with no progress for >60 seconds
    - Metadata store inaccessible

    Returns 200 if ready, 503 if not ready.
    Use this as your Kubernetes readiness probe.
    """
    import json

    from strata.metadata_cache import get_metadata_store

    # Check server initialized
    try:
        state = get_state()
    except RuntimeError:
        return Response(
            content='{"status": "not_ready", "checks": {"server_initialized": false}}',
            status_code=503,
            media_type="application/json",
        )

    # Run comprehensive readiness checks
    is_ready, checks = _check_readiness(state)
    checks["server_initialized"] = True

    # Also check metadata store accessibility
    try:
        store = get_metadata_store()
        store.stats()  # Quick sanity check
        checks["metadata_store"] = True
    except Exception as e:
        checks["metadata_store"] = False
        checks["metadata_store_error"] = str(e)
        is_ready = False
        if "issues" not in checks:
            checks["issues"] = []
        checks["issues"].append(f"metadata store error: {e}")

    # Add QoS capacity info for observability
    qos = _get_qos_metrics(state)
    checks["interactive_available"] = qos["interactive_available"]
    checks["bulk_available"] = qos["bulk_available"]
    checks["active_scans"] = _get_active_scan_count()

    status = "ready" if is_ready else "not_ready"
    status_code = 200 if is_ready else 503

    return Response(
        content=json.dumps({"status": status, "checks": checks}),
        status_code=status_code,
        media_type="application/json",
    )


@app.get("/metrics")
async def metrics():
    """Get aggregate metrics including resource utilization."""
    import asyncio
    import gc

    state = get_state()
    stats = state.metrics.get_aggregate_stats()

    # Add Arrow memory pool info
    pool = pa.default_memory_pool()
    stats["arrow_memory"] = {
        "pool_backend": pool.backend_name,
        "bytes_allocated": pool.bytes_allocated(),
        "max_memory": pool.max_memory(),
    }

    # Add GC stats for diagnosing periodic stalls
    # Include both gc.get_stats() (collection counts) and gc_tracker (pause durations)
    gc_builtin = gc.get_stats()
    stats["gc"] = {
        # Built-in GC stats (counts only)
        "gen0_collections": gc_builtin[0]["collections"],
        "gen1_collections": gc_builtin[1]["collections"],
        "gen2_collections": gc_builtin[2]["collections"],
        "gen0_collected": gc_builtin[0]["collected"],
        "gen1_collected": gc_builtin[1]["collected"],
        "gen2_collected": gc_builtin[2]["collected"],
        "gen0_uncollectable": gc_builtin[0]["uncollectable"],
        "gen1_uncollectable": gc_builtin[1]["uncollectable"],
        "gen2_uncollectable": gc_builtin[2]["uncollectable"],
    }

    # Add GC pause duration tracking (from gc.callbacks)
    gc_pause_stats = get_gc_stats()
    if gc_pause_stats:
        stats["gc_pauses"] = gc_pause_stats
    # Add resource utilization info
    stats["resource_limits"] = {
        "max_concurrent_scans": state.config.max_concurrent_scans,
        "active_scans": state._active_scans,
        "max_tasks_per_scan": state.config.max_tasks_per_scan,
        "plan_timeout_seconds": state.config.plan_timeout_seconds,
        "scan_timeout_seconds": state.config.scan_timeout_seconds,
        "max_response_bytes": state.config.max_response_bytes,
    }
    # Add prefetch metrics for observability
    stats["prefetch"] = {
        "started": state._prefetch_started,
        "used": state._prefetch_used,
        "wasted": state._prefetch_wasted,
        "skipped": state._prefetch_skipped,
        "in_flight": state._prefetch_in_flight,
    }
    # Add QoS tier metrics
    stats["qos"] = _get_qos_metrics(state)
    # Get cache size and entry count in thread pool to avoid blocking (involves filesystem ops)
    loop = asyncio.get_event_loop()
    cache_bytes, cache_entries = await asyncio.gather(
        loop.run_in_executor(None, _get_cache_size_bytes, state),
        loop.run_in_executor(None, _get_cache_entry_count, state),
    )
    # Add disk cache metrics
    stats["disk_cache"] = {
        "bytes_current": cache_bytes,
        "entries_current": cache_entries,
        "bytes_max": state.config.max_cache_size_bytes,
        "evictions_count": stats.get("cache_evictions_count", 0),
        "evicted_bytes": stats.get("cache_evicted_bytes", 0),
    }

    # Add thread pool metrics
    pool_tracker = get_pool_tracker()
    stats["thread_pools"] = {name: s.to_dict() for name, s in pool_tracker.get_all_stats().items()}

    # Add connection metrics
    connection_metrics = get_connection_metrics()
    stats["connections"] = connection_metrics.get_stats()

    # Add adaptive concurrency control metrics
    if state._adaptive_controller is not None:
        stats["adaptive_concurrency"] = state._adaptive_controller.get_metrics()

    # Add build QoS metrics (server-mode transforms)
    if state.config.server_transforms_enabled:
        from strata.transforms.build_qos import get_build_qos

        build_qos = get_build_qos()
        if build_qos is not None:
            stats["build_qos"] = build_qos.get_metrics()

    return stats


@app.get("/metrics/tables")
async def metrics_tables(limit: int = 10):
    """Get per-table metrics for the most accessed tables.

    Returns metrics aggregated by table including:
    - scan_count: Number of scans for this table
    - avg_latency_ms: Average scan latency
    - p50_ms, p95_ms, p99_ms: Latency percentiles
    - cache_hit_rate: Cache hit ratio for this table
    - bytes_from_cache/storage: Data transfer breakdown
    - rows_returned: Total rows returned
    - row_groups_pruned: Total row groups skipped by filters

    Query params:
    - limit: Max number of tables to return (default 10)
    """
    state = get_state()
    return {"tables": state.metrics.get_top_tables(limit)}


@app.get("/metrics/tables/{table_id:path}")
async def metrics_table(table_id: str):
    """Get metrics for a specific table.

    Path params:
    - table_id: The canonical table identity (e.g., "catalog.namespace.table")
    """
    state = get_state()
    table_metrics = state.metrics.get_table_metrics(table_id)

    if table_metrics is None:
        raise HTTPException(status_code=404, detail=f"No metrics found for table: {table_id}")

    return table_metrics.to_dict()


def _require_notebook_worker_admin_access() -> ServerState:
    """Authorize access to the service-mode notebook worker registry."""
    state = get_state()

    if state.config.deployment_mode != "service":
        raise HTTPException(
            status_code=409,
            detail="Server-managed notebook workers are only available in service mode",
        )

    if state.config.auth_mode == "trusted_proxy":
        principal = get_principal()
        if principal is None or not principal.has_scope("admin:notebook-workers"):
            raise HTTPException(status_code=403, detail="Insufficient scope")

    return state


async def _serialize_admin_notebook_workers(
    *,
    force_refresh: bool = False,
) -> dict[str, object]:
    return {
        "configured_workers": [
            {
                **record.worker.model_dump(mode="json"),
                "enabled": record.enabled,
            }
            for record in get_server_managed_worker_records()
        ],
        "workers": await build_server_worker_catalog_with_health(force_refresh=force_refresh),
        "definitions_editable": False,
        "health_checked_at": int(time.time() * 1000),
    }


def _validate_admin_notebook_worker_names(
    workers: list[AdminNotebookWorkerEntryRequest],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for worker in workers:
        if worker.name in seen:
            duplicates.add(worker.name)
        seen.add(worker.name)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise HTTPException(
            status_code=400,
            detail=f"Duplicate notebook worker names are not allowed: {duplicate_list}",
        )


@app.get(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def list_admin_notebook_workers(refresh: bool = False):
    """List the server-managed notebook worker registry."""
    return await _serialize_admin_notebook_workers(force_refresh=refresh)


@app.put(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def update_admin_notebook_workers(request: AdminNotebookWorkersRequest):
    """Replace the server-managed notebook worker registry."""
    _validate_admin_notebook_worker_names(request.workers)
    replace_server_managed_worker_records(
        [
            ManagedWorkerRecord(
                worker=worker.to_worker_spec(),
                enabled=worker.enabled,
            )
            for worker in request.workers
        ]
    )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.post(
    "/v1/admin/notebook-workers",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def create_admin_notebook_worker(request: AdminNotebookWorkerEntryRequest):
    """Create one service-managed notebook worker."""
    try:
        create_server_managed_worker_record(
            ManagedWorkerRecord(
                worker=request.to_worker_spec(),
                enabled=request.enabled,
            )
        )
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Notebook worker already exists: {request.name}",
        )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.put(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def replace_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerEntryRequest,
):
    """Replace one service-managed notebook worker definition."""
    try:
        update_server_managed_worker_record(
            worker_name,
            ManagedWorkerRecord(
                worker=request.to_worker_spec(),
                enabled=request.enabled,
            ),
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    except ValueError:
        raise HTTPException(
            status_code=409,
            detail=f"Notebook worker already exists: {request.name}",
        )
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.patch(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def patch_admin_notebook_worker(
    worker_name: str,
    request: AdminNotebookWorkerPatchRequest,
):
    """Patch one service-managed notebook worker."""
    try:
        set_server_managed_worker_enabled(worker_name, request.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.delete(
    "/v1/admin/notebook-workers/{worker_name}",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def delete_admin_notebook_worker(worker_name: str):
    """Delete one service-managed notebook worker."""
    try:
        delete_server_managed_worker_record(worker_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.post(
    "/v1/admin/notebook-workers/{worker_name}/refresh",
    dependencies=[Depends(require_notebook_worker_admin)],
)
async def refresh_admin_notebook_worker(worker_name: str):
    """Force-refresh health for one service-managed notebook worker."""
    known_workers = {record.worker.name for record in get_server_managed_worker_records()}
    if worker_name not in known_workers:
        raise HTTPException(status_code=404, detail=f"Notebook worker not found: {worker_name}")
    return await _serialize_admin_notebook_workers(force_refresh=True)


@app.get("/v1/admin/tenants", dependencies=[require_scope("admin:tenants")])
async def list_tenants():
    """List all tracked tenants with their metrics.

    Admin endpoint for multi-tenant observability.
    Returns metrics for all tenants that have made requests.
    """
    registry = get_tenant_registry()
    return {"tenants": registry.get_all_tenant_metrics()}


@app.get("/v1/admin/tenants/{tenant_id}", dependencies=[require_scope("admin:tenants")])
async def get_tenant_info(tenant_id: str):
    """Get configuration and metrics for a specific tenant.

    Path params:
    - tenant_id: The tenant identifier
    """
    registry = get_tenant_registry()
    config = registry.get_config(tenant_id)
    metrics = registry.get_tenant_metrics(tenant_id)

    if config is None and metrics is None:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")

    return {
        "tenant_id": tenant_id,
        "registered": config is not None,
        "enabled": config.enabled if config else True,
        "metrics": metrics,
        "config": {
            "interactive_slots": config.interactive_slots if config else None,
            "bulk_slots": config.bulk_slots if config else None,
            "per_client_interactive": config.per_client_interactive if config else None,
            "per_client_bulk": config.per_client_bulk if config else None,
        }
        if config
        else None,
    }


@app.get("/metrics/prometheus")
async def metrics_prometheus():
    """Prometheus-format metrics endpoint.

    Returns metrics in Prometheus text exposition format for scraping.
    Includes:
    - Cache hit/miss counters
    - Active scan gauge
    - Request latency histograms (TODO: requires histogram support)
    - Resource utilization gauges
    """
    from strata.metadata_cache import get_metadata_store

    state = get_state()
    stats = state.metrics.get_aggregate_stats()

    lines = [
        "# HELP strata_cache_hits_total Total number of cache hits",
        "# TYPE strata_cache_hits_total counter",
        f"strata_cache_hits_total {stats.get('cache_hits', 0)}",
        "",
        "# HELP strata_cache_misses_total Total number of cache misses",
        "# TYPE strata_cache_misses_total counter",
        f"strata_cache_misses_total {stats.get('cache_misses', 0)}",
        "",
        "# HELP strata_scans_total Total number of completed scans",
        "# TYPE strata_scans_total counter",
        f"strata_scans_total {stats.get('scan_count', 0)}",
        "",
        "# HELP strata_active_scans Current number of active scans",
        "# TYPE strata_active_scans gauge",
        f"strata_active_scans {state._active_scans}",
        "",
        "# HELP strata_max_concurrent_scans Maximum allowed concurrent scans",
        "# TYPE strata_max_concurrent_scans gauge",
        f"strata_max_concurrent_scans {state.config.max_concurrent_scans}",
        "",
        "# HELP strata_bytes_from_cache_total Total bytes served from cache",
        "# TYPE strata_bytes_from_cache_total counter",
        f"strata_bytes_from_cache_total {stats.get('bytes_from_cache', 0)}",
        "",
        "# HELP strata_bytes_from_storage_total Total bytes read from storage",
        "# TYPE strata_bytes_from_storage_total counter",
        f"strata_bytes_from_storage_total {stats.get('bytes_from_storage', 0)}",
        "",
        "# HELP strata_rows_returned_total Total rows returned across all scans",
        "# TYPE strata_rows_returned_total counter",
        f"strata_rows_returned_total {stats.get('rows_returned', 0)}",
        "",
        "# HELP strata_row_groups_pruned_total Total row groups pruned by filters",
        "# TYPE strata_row_groups_pruned_total counter",
        f"strata_row_groups_pruned_total {stats.get('row_groups_pruned', 0)}",
        "",
        "# HELP strata_draining Server is draining (shutting down)",
        "# TYPE strata_draining gauge",
        f"strata_draining {1 if state._draining else 0}",
        "",
        "# HELP strata_stream_aborts_timeout_total Streams aborted due to timeout",
        "# TYPE strata_stream_aborts_timeout_total counter",
        f"strata_stream_aborts_timeout_total {stats.get('stream_aborts_timeout', 0)}",
        "",
        "# HELP strata_stream_aborts_size_total Streams aborted due to size limit",
        "# TYPE strata_stream_aborts_size_total counter",
        f"strata_stream_aborts_size_total {stats.get('stream_aborts_size', 0)}",
        "",
        "# HELP strata_client_disconnects_total Client disconnects during streaming",
        "# TYPE strata_client_disconnects_total counter",
        f"strata_client_disconnects_total {stats.get('client_disconnects', 0)}",
        "",
        "# HELP strata_cache_evictions_total Total cache entries evicted",
        "# TYPE strata_cache_evictions_total counter",
        f"strata_cache_evictions_total {stats.get('cache_evictions_count', 0)}",
        "",
        "# HELP strata_cache_evicted_bytes_total Total bytes evicted from cache",
        "# TYPE strata_cache_evicted_bytes_total counter",
        f"strata_cache_evicted_bytes_total {stats.get('cache_evicted_bytes', 0)}",
        "",
        "# HELP strata_cache_bytes_written_total Total bytes written to cache",
        "# TYPE strata_cache_bytes_written_total counter",
        f"strata_cache_bytes_written_total {stats.get('bytes_written_to_cache', 0)}",
        "",
        "# HELP strata_cache_bytes_current Current cache size in bytes",
        "# TYPE strata_cache_bytes_current gauge",
        f"strata_cache_bytes_current {_get_cache_size_bytes(state)}",
        "",
        "# HELP strata_cache_entries_current Current number of cache entries",
        "# TYPE strata_cache_entries_current gauge",
        f"strata_cache_entries_current {_get_cache_entry_count(state)}",
        "",
        "# HELP strata_cache_max_bytes Maximum cache size limit in bytes",
        "# TYPE strata_cache_max_bytes gauge",
        f"strata_cache_max_bytes {state.config.max_cache_size_bytes}",
        "",
        "# HELP strata_prefetch_started_total Total prefetches started",
        "# TYPE strata_prefetch_started_total counter",
        f"strata_prefetch_started_total {state._prefetch_started}",
        "",
        "# HELP strata_prefetch_used_total Prefetches successfully used by streaming",
        "# TYPE strata_prefetch_used_total counter",
        f"strata_prefetch_used_total {state._prefetch_used}",
        "",
        "# HELP strata_prefetch_wasted_total Prefetches wasted (scan deleted/abandoned)",
        "# TYPE strata_prefetch_wasted_total counter",
        f"strata_prefetch_wasted_total {state._prefetch_wasted}",
        "",
        "# HELP strata_prefetch_skipped_total Prefetches skipped (server busy)",
        "# TYPE strata_prefetch_skipped_total counter",
        f"strata_prefetch_skipped_total {state._prefetch_skipped}",
        "",
        "# HELP strata_prefetch_in_flight Current prefetches in flight",
        "# TYPE strata_prefetch_in_flight gauge",
        f"strata_prefetch_in_flight {state._prefetch_in_flight}",
    ]

    # Add GC stats for diagnosing periodic stalls
    import gc

    gc_stats = gc.get_stats()
    lines.extend(
        [
            "",
            "# HELP strata_gc_collections_total GC collections by generation",
            "# TYPE strata_gc_collections_total counter",
            f'strata_gc_collections_total{{generation="0"}} {gc_stats[0]["collections"]}',
            f'strata_gc_collections_total{{generation="1"}} {gc_stats[1]["collections"]}',
            f'strata_gc_collections_total{{generation="2"}} {gc_stats[2]["collections"]}',
            "",
            "# HELP strata_gc_collected_total Objects collected by generation",
            "# TYPE strata_gc_collected_total counter",
            f'strata_gc_collected_total{{generation="0"}} {gc_stats[0]["collected"]}',
            f'strata_gc_collected_total{{generation="1"}} {gc_stats[1]["collected"]}',
            f'strata_gc_collected_total{{generation="2"}} {gc_stats[2]["collected"]}',
        ]
    )

    # Add GC pause duration metrics (from gc.callbacks tracker)
    gc_pause_stats = get_gc_stats()
    if gc_pause_stats:
        lines.extend(
            [
                "",
                "# HELP strata_gc_pause_total_ms Total GC pause time in milliseconds",
                "# TYPE strata_gc_pause_total_ms counter",
                f"strata_gc_pause_total_ms {gc_pause_stats.get('total_pause_ms', 0)}",
                "",
                "# HELP strata_gc_pause_max_ms Maximum single GC pause in milliseconds",
                "# TYPE strata_gc_pause_max_ms gauge",
                f"strata_gc_pause_max_ms {gc_pause_stats.get('max_pause_ms', 0)}",
                "",
                "# HELP strata_gc_pauses_total Total number of GC pauses",
                "# TYPE strata_gc_pauses_total counter",
                f"strata_gc_pauses_total {gc_pause_stats.get('total_pauses', 0)}",
            ]
        )
        # Per-generation pause stats
        for gen in ["gen0", "gen1", "gen2"]:
            gen_stats = gc_pause_stats.get(gen, {})
            gen_num = gen[-1]  # "0", "1", or "2"
            lines.extend(
                [
                    "",
                    "# HELP strata_gc_pause_count GC pause count by generation",
                    "# TYPE strata_gc_pause_count counter",
                    f'strata_gc_pause_count{{generation="{gen_num}"}} {gen_stats.get("count", 0)}',
                    "# HELP strata_gc_pause_total_ms_by_gen Total pause time by generation",
                    "# TYPE strata_gc_pause_total_ms_by_gen counter",
                    f'strata_gc_pause_total_ms_by_gen{{generation="{gen_num}"}} '
                    f"{gen_stats.get('total_ms', 0)}",
                    "# HELP strata_gc_pause_max_ms_by_gen Max pause time by generation",
                    "# TYPE strata_gc_pause_max_ms_by_gen gauge",
                    f'strata_gc_pause_max_ms_by_gen{{generation="{gen_num}"}} '
                    f"{gen_stats.get('max_ms', 0)}",
                ]
            )

    # Add metadata store stats if available
    try:
        store = get_metadata_store()
        store_stats = store.stats()
        lines.extend(
            [
                "",
                "# HELP strata_metadata_manifest_hits_total Manifest cache hits in metadata store",
                "# TYPE strata_metadata_manifest_hits_total counter",
                f"strata_metadata_manifest_hits_total {store_stats.get('manifest_hits', 0)}",
                "",
                "# HELP strata_metadata_manifest_misses_total Manifest cache misses",
                "# TYPE strata_metadata_manifest_misses_total counter",
                f"strata_metadata_manifest_misses_total {store_stats.get('manifest_misses', 0)}",
                "",
                "# HELP strata_metadata_parquet_hits_total Parquet metadata cache hits",
                "# TYPE strata_metadata_parquet_hits_total counter",
                f"strata_metadata_parquet_hits_total {store_stats.get('parquet_meta_hits', 0)}",
                "",
                "# HELP strata_metadata_parquet_misses_total Parquet metadata cache misses",
                "# TYPE strata_metadata_parquet_misses_total counter",
                f"strata_metadata_parquet_misses_total {store_stats.get('parquet_meta_misses', 0)}",
                "",
                "# HELP strata_metadata_stale_invalidations_total Stale entries invalidated",
                "# TYPE strata_metadata_stale_invalidations_total counter",
                f"strata_metadata_stale_invalidations_total "
                f"{store_stats.get('stale_invalidations', 0)}",
            ]
        )
    except Exception:
        pass  # Metadata store not available

    # Add in-memory cache stats
    pq_cache_stats = state.planner.parquet_cache.stats()
    manifest_cache_stats = state.planner.manifest_cache.stats()

    lines.extend(
        [
            "",
            "# HELP strata_parquet_cache_hits_total In-memory parquet cache hits",
            "# TYPE strata_parquet_cache_hits_total counter",
            f"strata_parquet_cache_hits_total {pq_cache_stats.get('hits', 0)}",
            "",
            "# HELP strata_parquet_cache_misses_total In-memory parquet cache misses",
            "# TYPE strata_parquet_cache_misses_total counter",
            f"strata_parquet_cache_misses_total {pq_cache_stats.get('misses', 0)}",
            "",
            "# HELP strata_parquet_cache_size Current entries in parquet cache",
            "# TYPE strata_parquet_cache_size gauge",
            f"strata_parquet_cache_size {pq_cache_stats.get('size', 0)}",
            "",
            "# HELP strata_manifest_cache_hits_total In-memory manifest cache hits",
            "# TYPE strata_manifest_cache_hits_total counter",
            f"strata_manifest_cache_hits_total {manifest_cache_stats.get('hits', 0)}",
            "",
            "# HELP strata_manifest_cache_misses_total In-memory manifest cache misses",
            "# TYPE strata_manifest_cache_misses_total counter",
            f"strata_manifest_cache_misses_total {manifest_cache_stats.get('misses', 0)}",
            "",
            "# HELP strata_manifest_cache_size Current entries in manifest cache",
            "# TYPE strata_manifest_cache_size gauge",
            f"strata_manifest_cache_size {manifest_cache_stats.get('size', 0)}",
        ]
    )

    # Add QoS tier metrics
    qos = _get_qos_metrics(state)
    lines.extend(
        [
            "",
            "# HELP strata_qos_interactive_slots Max interactive query slots",
            "# TYPE strata_qos_interactive_slots gauge",
            f"strata_qos_interactive_slots {qos['interactive_slots']}",
            "",
            "# HELP strata_qos_interactive_active Current interactive queries running",
            "# TYPE strata_qos_interactive_active gauge",
            f"strata_qos_interactive_active {qos['interactive_active']}",
            "",
            "# HELP strata_qos_interactive_rejected_total Interactive queries rejected (429)",
            "# TYPE strata_qos_interactive_rejected_total counter",
            f"strata_qos_interactive_rejected_total {qos['interactive_rejected']}",
            "",
            "# HELP strata_qos_interactive_queue_wait_avg_ms Average queue wait time (ms)",
            "# TYPE strata_qos_interactive_queue_wait_avg_ms gauge",
            f"strata_qos_interactive_queue_wait_avg_ms {qos['interactive_queue_wait_avg_ms']}",
            "",
            "# HELP strata_qos_bulk_slots Max bulk query slots",
            "# TYPE strata_qos_bulk_slots gauge",
            f"strata_qos_bulk_slots {qos['bulk_slots']}",
            "",
            "# HELP strata_qos_bulk_active Current bulk queries running",
            "# TYPE strata_qos_bulk_active gauge",
            f"strata_qos_bulk_active {qos['bulk_active']}",
            "",
            "# HELP strata_qos_bulk_rejected_total Bulk queries rejected (429)",
            "# TYPE strata_qos_bulk_rejected_total counter",
            f"strata_qos_bulk_rejected_total {qos['bulk_rejected']}",
            "",
            "# HELP strata_qos_bulk_queue_wait_avg_ms Average queue wait time (ms)",
            "# TYPE strata_qos_bulk_queue_wait_avg_ms gauge",
            f"strata_qos_bulk_queue_wait_avg_ms {qos['bulk_queue_wait_avg_ms']}",
            "",
            "# HELP strata_qos_per_client_limit Per-client concurrent query limit",
            "# TYPE strata_qos_per_client_limit gauge",
            f'strata_qos_per_client_limit{{tier="interactive"}} {qos["per_client_interactive"]}',
            f'strata_qos_per_client_limit{{tier="bulk"}} {qos["per_client_bulk"]}',
            "",
            "# HELP strata_qos_client_rejected_total Queries rejected due to per-client limit",
            "# TYPE strata_qos_client_rejected_total counter",
            f"strata_qos_client_rejected_total {qos['client_rejected']}",
            "",
            "# HELP strata_qos_tracked_clients Number of clients with active semaphores",
            "# TYPE strata_qos_tracked_clients gauge",
            f"strata_qos_tracked_clients {qos['tracked_clients']}",
        ]
    )

    # Add fetch parallelism metrics
    lines.extend(
        [
            "",
            "# HELP strata_fetch_parallelism Max concurrent row group fetches per scan",
            "# TYPE strata_fetch_parallelism gauge",
            f"strata_fetch_parallelism {state.config.fetch_parallelism}",
            "",
            "# HELP strata_fetch_executor_workers Number of workers in fetch thread pool",
            "# TYPE strata_fetch_executor_workers gauge",
            f"strata_fetch_executor_workers {state._fetch_executor._max_workers}",
        ]
    )

    # Add timeout configuration metrics
    lines.extend(
        [
            "",
            "# HELP strata_timeout_plan_seconds Planning timeout in seconds",
            "# TYPE strata_timeout_plan_seconds gauge",
            f"strata_timeout_plan_seconds {state.config.plan_timeout_seconds}",
            "",
            "# HELP strata_timeout_scan_seconds Scan timeout in seconds",
            "# TYPE strata_timeout_scan_seconds gauge",
            f"strata_timeout_scan_seconds {state.config.scan_timeout_seconds}",
            "",
            "# HELP strata_timeout_fetch_seconds Fetch timeout in seconds",
            "# TYPE strata_timeout_fetch_seconds gauge",
            f"strata_timeout_fetch_seconds {state.config.fetch_timeout_seconds}",
            "",
            "# HELP strata_timeout_queue_seconds Queue wait timeout by tier",
            "# TYPE strata_timeout_queue_seconds gauge",
            f'strata_timeout_queue_seconds{{tier="interactive"}} '
            f"{state.config.interactive_queue_timeout}",
            f'strata_timeout_queue_seconds{{tier="bulk"}} {state.config.bulk_queue_timeout}',
            "",
            "# HELP strata_timeout_s3_seconds S3 timeout by type",
            "# TYPE strata_timeout_s3_seconds gauge",
            f'strata_timeout_s3_seconds{{type="connect"}} '
            f"{state.config.s3_connect_timeout_seconds}",
            f'strata_timeout_s3_seconds{{type="request"}} '
            f"{state.config.s3_request_timeout_seconds}",
        ]
    )

    # Add rate limiter metrics
    rate_limiter = get_rate_limiter()
    if rate_limiter is not None:
        rl_stats = rate_limiter.get_stats()
        rl_rejected_global = rl_stats.get("rejected_global", 0)
        rl_rejected_client = rl_stats.get("rejected_client", 0)
        rl_rejected_endpoint = rl_stats.get("rejected_endpoint", 0)
        lines.extend(
            [
                "",
                "# HELP strata_rate_limit_requests_total Total requests processed",
                "# TYPE strata_rate_limit_requests_total counter",
                f"strata_rate_limit_requests_total {rl_stats.get('total_requests', 0)}",
                "",
                "# HELP strata_rate_limit_allowed_total Requests allowed",
                "# TYPE strata_rate_limit_allowed_total counter",
                f"strata_rate_limit_allowed_total {rl_stats.get('allowed_requests', 0)}",
                "",
                "# HELP strata_rate_limit_rejected_total Requests rejected by reason",
                "# TYPE strata_rate_limit_rejected_total counter",
                f'strata_rate_limit_rejected_total{{reason="global"}} {rl_rejected_global}',
                f'strata_rate_limit_rejected_total{{reason="client"}} {rl_rejected_client}',
                f'strata_rate_limit_rejected_total{{reason="endpoint"}} {rl_rejected_endpoint}',
                "",
                "# HELP strata_rate_limit_active_clients Tracked clients",
                "# TYPE strata_rate_limit_active_clients gauge",
                f"strata_rate_limit_active_clients {rl_stats.get('active_clients', 0)}",
            ]
        )

    # Add cache eviction metrics
    eviction_tracker = get_eviction_tracker()
    eviction_stats = eviction_tracker.get_stats()
    pressure_map = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    pressure_value = pressure_map.get(eviction_stats.pressure_level, 0)
    lines.extend(
        [
            "",
            "# HELP strata_cache_eviction_events_total Total eviction events",
            "# TYPE strata_cache_eviction_events_total counter",
            f"strata_cache_eviction_events_total {eviction_stats.total_evictions}",
            "",
            "# HELP strata_cache_files_evicted_total Total files evicted",
            "# TYPE strata_cache_files_evicted_total counter",
            f"strata_cache_files_evicted_total {eviction_stats.total_files_evicted}",
            "",
            "# HELP strata_cache_eviction_bytes_total Total bytes evicted",
            "# TYPE strata_cache_eviction_bytes_total counter",
            f"strata_cache_eviction_bytes_total {eviction_stats.total_bytes_evicted}",
            "",
            "# HELP strata_cache_eviction_rate Evictions per minute",
            "# TYPE strata_cache_eviction_rate gauge",
            f"strata_cache_eviction_rate {eviction_stats.eviction_rate_per_minute}",
            "",
            "# HELP strata_cache_eviction_pressure Pressure level (0-3)",
            "# TYPE strata_cache_eviction_pressure gauge",
            f"strata_cache_eviction_pressure {pressure_value}",
        ]
    )

    # Add thread pool metrics
    pool_tracker = get_pool_tracker()
    for pool_name, pool_stats in pool_tracker.get_all_stats().items():
        active = pool_stats.active_workers
        max_w = pool_stats.max_workers
        util = pool_stats.utilization_pct / 100.0  # Convert percentage to ratio
        lines.extend(
            [
                "",
                "# HELP strata_thread_pool_active_workers Active workers",
                "# TYPE strata_thread_pool_active_workers gauge",
                f'strata_thread_pool_active_workers{{pool="{pool_name}"}} {active}',
                "# HELP strata_thread_pool_max_workers Max workers",
                "# TYPE strata_thread_pool_max_workers gauge",
                f'strata_thread_pool_max_workers{{pool="{pool_name}"}} {max_w}',
                "# HELP strata_thread_pool_utilization Utilization ratio",
                "# TYPE strata_thread_pool_utilization gauge",
                f'strata_thread_pool_utilization{{pool="{pool_name}"}} {util}',
            ]
        )

    # Add connection metrics
    conn_metrics = get_connection_metrics()
    conn_stats = conn_metrics.get_stats()
    lines.extend(
        [
            "",
            "# HELP strata_http_requests_total Total HTTP requests",
            "# TYPE strata_http_requests_total counter",
            f"strata_http_requests_total {conn_stats.get('total_requests', 0)}",
            "",
            "# HELP strata_http_connections_active Current active HTTP connections",
            "# TYPE strata_http_connections_active gauge",
            f"strata_http_connections_active {conn_stats.get('concurrent_requests', 0)}",
            "",
            "# HELP strata_http_connections_keepalive Requests with keep-alive",
            "# TYPE strata_http_connections_keepalive counter",
            f"strata_http_connections_keepalive {conn_stats.get('keepalive_requests', 0)}",
        ]
    )

    # Add Arrow memory metrics
    pool = pa.default_memory_pool()
    lines.extend(
        [
            "",
            "# HELP strata_arrow_memory_bytes_allocated Current Arrow memory allocated",
            "# TYPE strata_arrow_memory_bytes_allocated gauge",
            f"strata_arrow_memory_bytes_allocated {pool.bytes_allocated()}",
            "",
            "# HELP strata_arrow_memory_max_bytes Maximum Arrow memory ever allocated",
            "# TYPE strata_arrow_memory_max_bytes gauge",
            f"strata_arrow_memory_max_bytes {pool.max_memory()}",
        ]
    )

    # Add circuit breaker metrics
    from strata.circuit_breaker import get_circuit_breaker_registry

    cb_registry = get_circuit_breaker_registry()
    cb_all_stats = cb_registry.get_all_stats()
    if cb_all_stats:
        lines.extend(
            [
                "",
                # Circuit breaker state: 0=closed, 1=open, 2=half_open
                "# HELP strata_circuit_breaker_state Circuit breaker state",
                "# TYPE strata_circuit_breaker_state gauge",
            ]
        )
        state_map = {"closed": 0, "open": 1, "half_open": 2}
        for cb_name, cb_stats in cb_all_stats.items():
            cb_state_val = state_map.get(cb_stats.get("state", "closed"), 0)
            lines.append(f'strata_circuit_breaker_state{{name="{cb_name}"}} {cb_state_val}')

        lines.extend(
            [
                "",
                "# HELP strata_circuit_breaker_calls_total Total calls by circuit breaker",
                "# TYPE strata_circuit_breaker_calls_total counter",
            ]
        )
        for cb_name, cb_stats in cb_all_stats.items():
            lines.append(
                f'strata_circuit_breaker_calls_total{{name="{cb_name}"}} '
                f"{cb_stats.get('total_calls', 0)}"
            )

        lines.extend(
            [
                "",
                "# HELP strata_circuit_breaker_failures_total Total failures by circuit breaker",
                "# TYPE strata_circuit_breaker_failures_total counter",
            ]
        )
        for cb_name, cb_stats in cb_all_stats.items():
            lines.append(
                f'strata_circuit_breaker_failures_total{{name="{cb_name}"}} '
                f"{cb_stats.get('total_failures', 0)}"
            )

        lines.extend(
            [
                "",
                "# HELP strata_circuit_breaker_rejections_total Rejected calls by circuit breaker",
                "# TYPE strata_circuit_breaker_rejections_total counter",
            ]
        )
        for cb_name, cb_stats in cb_all_stats.items():
            lines.append(
                f'strata_circuit_breaker_rejections_total{{name="{cb_name}"}} '
                f"{cb_stats.get('total_rejections', 0)}"
            )

    # Add per-table metrics (top 20 most accessed tables)
    table_metrics = state.metrics.get_top_tables(20)
    if table_metrics:
        lines.extend(
            [
                "",
                "# HELP strata_table_scans_total Total scans by table",
                "# TYPE strata_table_scans_total counter",
            ]
        )
        for tm in table_metrics:
            # Escape table_id for Prometheus label (replace dots with underscores for label)
            table_id = tm["table_id"]
            lines.append(f'strata_table_scans_total{{table="{table_id}"}} {tm["scan_count"]}')

        lines.extend(
            [
                "",
                "# HELP strata_table_latency_p95_ms P95 latency by table (ms)",
                "# TYPE strata_table_latency_p95_ms gauge",
            ]
        )
        for tm in table_metrics:
            table_id = tm["table_id"]
            lines.append(f'strata_table_latency_p95_ms{{table="{table_id}"}} {tm["p95_ms"]}')

        lines.extend(
            [
                "",
                "# HELP strata_table_cache_hit_rate Cache hit rate by table",
                "# TYPE strata_table_cache_hit_rate gauge",
            ]
        )
        for tm in table_metrics:
            table_id = tm["table_id"]
            lines.append(
                f'strata_table_cache_hit_rate{{table="{table_id}"}} {tm["cache_hit_rate"]}'
            )

    # Add per-tenant metrics (multi-tenancy support)
    tenant_registry = get_tenant_registry()
    tenant_metrics = tenant_registry.get_all_tenant_metrics()
    if tenant_metrics:
        lines.extend(
            [
                "",
                "# HELP strata_tenant_scans_total Total scans by tenant",
                "# TYPE strata_tenant_scans_total counter",
            ]
        )
        for tm in tenant_metrics:
            tenant_id = tm["tenant_id"]
            lines.append(f'strata_tenant_scans_total{{tenant="{tenant_id}"}} {tm["total_scans"]}')

        lines.extend(
            [
                "",
                "# HELP strata_tenant_cache_hit_rate Cache hit rate by tenant",
                "# TYPE strata_tenant_cache_hit_rate gauge",
            ]
        )
        for tm in tenant_metrics:
            tenant_id = tm["tenant_id"]
            lines.append(
                f'strata_tenant_cache_hit_rate{{tenant="{tenant_id}"}} {tm["cache_hit_rate"]}'
            )

        lines.extend(
            [
                "",
                "# HELP strata_tenant_bytes_total Total bytes processed by tenant",
                "# TYPE strata_tenant_bytes_total counter",
            ]
        )
        for tm in tenant_metrics:
            tenant_id = tm["tenant_id"]
            total_bytes = tm["bytes_from_cache"] + tm["bytes_from_storage"]
            lines.append(f'strata_tenant_bytes_total{{tenant="{tenant_id}"}} {total_bytes}')

    # Add build metrics (if server transforms are enabled)
    try:
        from strata.transforms.build_metrics import get_build_metrics

        build_metrics = get_build_metrics()
        if build_metrics is not None:
            # Append build-specific metrics
            build_prom = build_metrics.get_prometheus_metrics()
            if build_prom:
                lines.append("")
                lines.append(build_prom)
    except Exception:
        pass  # Build metrics not available

    return Response(
        content="\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# =============================================================================
# API v1 Endpoints (stable contracts)
#
# v1 API Guarantees:
# - Response types are stable (MaterializeResponse, error format)
# - Arrow IPC streams via /v1/streams/{stream_id}
# - Error codes: 400 (bad request), 404 (not found), 413 (too large),
#                429 (rate limited, includes Retry-After header),
#                503 (draining/unhealthy), 504 (timeout)
# - Cache key format is versioned (CACHE_VERSION in cache.py)
#
# Note: /v1/scan endpoints were removed. Use /v1/materialize instead.
# =============================================================================


@app.get("/v1/cache/stats")
async def get_cache_stats_v1():
    """Get cache statistics.

    Returns information about what's in the cache and why.
    Operators can use this to understand cache behavior and debug issues.
    """
    from strata.cache import DiskCache

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    stats = cache.get_stats()
    return stats.to_dict()


@app.get("/v1/cache/evictions")
async def get_cache_evictions_v1(
    include_events: Annotated[
        bool,
        Query(description="Include recent eviction events"),
    ] = False,
    limit: Annotated[
        int,
        Query(description="Max number of recent events to include", ge=1, le=100),
    ] = 10,
):
    """Get cache eviction metrics and monitoring data.

    Returns eviction statistics including:
    - Total evictions and bytes evicted (lifetime)
    - Evictions in last minute/hour
    - Eviction rate (per minute)
    - Pressure level indicator (low/medium/high/critical)

    Use include_events=true to get recent eviction events for debugging.

    Pressure levels:
    - low: < 1 eviction per minute (healthy)
    - medium: 1-5 evictions per minute (monitor)
    - high: 5-10 evictions per minute (consider increasing cache size)
    - critical: 10+ evictions per minute (cache is thrashing)
    """
    tracker = get_eviction_tracker()
    stats = tracker.get_stats()
    result = stats.to_dict()

    if include_events:
        result["recent_events"] = tracker.get_recent_events(limit)

    return result


@app.get("/v1/cache/histogram")
async def get_cache_histogram_v1():
    """Get cache hit/miss statistics over time windows.

    Returns hit rate trends for understanding cache effectiveness:
    - lifetime: Total hits, misses, hit rate, bytes served
    - windows: Statistics for 1 minute, 5 minutes, and 1 hour windows
    - top_tables: Top 5 tables by cache access count

    Each window includes:
    - hits/misses: Access counts
    - hit_rate: Hits / total (0.0 to 1.0)
    - bytes_from_cache/bytes_from_storage: Data served from each source

    Use this to:
    - Track cache warm-up progress (watch hit rate climb)
    - Identify cache thrashing (sudden hit rate drops)
    - Find hot tables that dominate cache usage
    """
    histogram = get_cache_histogram()
    return histogram.get_summary()


@app.get("/v1/metadata/stats")
async def get_metadata_stats_v1():
    """Get metadata store and cache statistics.

    Returns hit/miss counters and entry counts for:
    - SQLite metadata store (manifest cache, parquet metadata)
    - In-memory LRU caches (parquet metadata, manifest resolution)

    Useful for:
    - Proving cache value (hit rates)
    - Debugging performance issues
    - Capacity planning
    """
    from strata.metadata_cache import get_metadata_store

    state = get_state()

    result = {
        "parquet_cache": state.planner.parquet_cache.stats(),
        "manifest_cache": state.planner.manifest_cache.stats(),
    }

    # Add SQLite store stats if available
    try:
        store = get_metadata_store()
        result["metadata_store"] = store.stats()
    except Exception:
        result["metadata_store"] = None

    return result


@app.get("/v1/debug/latency")
async def get_latency_histograms_v1():
    """Get latency histograms for each operation stage.

    Returns latency distribution data for:
    - plan: Table planning (catalog + metadata)
    - ttfb: Time to first byte
    - fetch: Individual row group fetch
    - total_request: End-to-end request time

    Each stage includes:
    - Histogram buckets with counts
    - Estimated percentiles (p50, p95, p99)
    - Count, sum, avg, max

    This is useful for:
    - Identifying which stage dominates tail latency
    - Understanding latency distribution over time
    - Detecting bimodal latency patterns
    """
    stats = get_latency_stats()

    # Add percentile estimates for key stages
    from strata.slow_ops import get_latency_percentiles

    result = {"histograms": stats}

    for stage in ["plan", "ttfb", "fetch", "total_request"]:
        if stage in stats:
            result["histograms"][stage]["percentiles"] = get_latency_percentiles(stage)

    return result


@app.get("/v1/debug/gc/pauses")
async def get_gc_pauses_v1(
    limit: Annotated[int, Query(description="Maximum pauses to return", ge=1, le=1000)] = 100,
):
    """Get recent GC pause events for debugging.

    Returns detailed timing information about recent garbage collection pauses.
    This is useful for:
    - Correlating latency spikes with GC activity
    - Understanding GC pause duration distribution
    - Diagnosing periodic latency stalls

    Returns:
    - pauses: List of recent GC pauses (most recent first)
      - timestamp: Unix timestamp when GC completed
      - generation: GC generation (0, 1, or 2)
      - duration_ms: Pause duration in milliseconds
    - stats: Aggregate statistics (p50, p95, p99 if enough data)
    """
    pauses = get_recent_gc_pauses(limit=limit)
    stats = get_gc_stats()

    return {
        "pauses": pauses,
        "stats": stats,
    }


@app.get("/v1/debug/pools")
async def get_pool_metrics_v1():
    """Get thread pool metrics for debugging.

    Returns utilization and queue depth for server thread pools:
    - planning: Thread pool for Iceberg catalog/metadata operations
    - fetch: Thread pool for Parquet row group I/O

    Each pool includes:
    - max_workers: Pool capacity
    - active_workers: Currently executing workers
    - queue_depth: Tasks waiting for a worker
    - utilization_pct: (active_workers / max_workers) * 100

    High queue_depth indicates pool saturation (bottleneck).
    """
    pool_tracker = get_pool_tracker()
    return pool_tracker.get_summary()


@app.get("/v1/debug/connections")
async def get_connection_metrics_v1():
    """Get HTTP connection metrics for debugging.

    Returns:
    - active_requests: Currently in-flight requests
    - total_requests: Total requests since server start
    - max_concurrent_requests: Peak concurrency observed
    - request_rate_per_sec: Average request rate
    - keepalive_pct: Percentage of requests using keep-alive

    High active_requests with low throughput may indicate connection issues.
    """
    connection_metrics = get_connection_metrics()
    return connection_metrics.get_stats()


@app.get("/v1/debug/memory")
async def get_memory_debug_v1(
    detailed: Annotated[
        bool,
        Query(description="Include detailed breakdown (slower, includes object type counts)"),
    ] = False,
):
    """Get memory profiling information for debugging.

    Returns memory statistics across multiple levels:
    - Arrow: Memory pool allocations (bytes_allocated, max_memory, pool_backend)
    - Python: GC tracked objects, objects by generation
    - Process: RSS and VMS memory (if available)

    Use detailed=true for comprehensive analysis including:
    - Top object types by count
    - GC thresholds and collection stats
    - Memory recommendations

    Note: detailed=true is more expensive and enumerates all GC objects.
    """
    if detailed:
        return get_detailed_memory_report()
    else:
        snapshot = get_memory_snapshot()
        return snapshot.to_dict()


@app.get("/v1/debug/rate-limits")
async def get_rate_limits_debug_v1():
    """Get rate limiter statistics for debugging.

    Returns:
    - total_requests: Total requests processed
    - allowed_requests: Requests that passed rate limiting
    - rejected_global: Requests rejected by global limit
    - rejected_client: Requests rejected by per-client limit
    - rejected_endpoint: Requests rejected by per-endpoint limit
    - active_clients: Number of tracked client buckets
    - global_tokens_available: Current global bucket tokens
    - enabled: Whether rate limiting is enabled
    """
    rate_limiter = get_rate_limiter()
    if rate_limiter is None:
        return {"error": "Rate limiter not initialized", "enabled": False}
    return rate_limiter.get_stats()


@app.get("/v1/config/timeouts")
async def get_timeout_config_v1():
    """Get all timeout configuration settings.

    Returns timeout configuration organized by category:
    - planning: Plan timeout settings
    - scanning: Scan timeout settings
    - qos_queue: QoS queue wait timeouts
    - fetching: Row group fetch timeouts
    - s3: S3 connection and request timeouts
    """
    state = get_state()
    return state.config.get_timeout_config()


@app.get("/v1/debug/circuit-breakers")
async def get_circuit_breakers_v1():
    """Get circuit breaker status for all dependencies.

    Returns status for each registered circuit breaker:
    - state: Current state (closed, open, half_open)
    - failure_count: Current consecutive failures
    - success_count: Current consecutive successes (in half_open)
    - total_calls: Lifetime call count
    - total_failures: Lifetime failure count
    - total_successes: Lifetime success count
    - total_rejections: Requests rejected when open
    """
    from strata.circuit_breaker import get_circuit_breaker_registry

    registry = get_circuit_breaker_registry()
    return {"breakers": registry.get_all_stats()}


@app.post("/v1/metadata/cleanup")
async def cleanup_metadata_v1():
    """Remove stale metadata entries from the SQLite store.

    Scans all cached parquet metadata entries and removes those where:
    - The file no longer exists on disk
    - The file has been modified (different mtime or size)

    This is automatically run on server startup, but can be triggered
    manually if needed (e.g., after bulk file operations).

    Returns the number of stale entries removed.
    """
    from strata.metadata_cache import get_metadata_store

    try:
        store = get_metadata_store()
        removed = store.cleanup_stale_parquet_meta()
        return {
            "status": "completed",
            "stale_entries_removed": removed,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/cache/entries")
async def list_cache_entries_v1():
    """List all cache entries with metadata.

    Returns detailed information about each cached entry.
    """
    from strata.cache import DiskCache

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    entries = cache.list_entries()
    return {"entries": [e.to_dict() for e in entries]}


@app.get("/v1/debug/cache/inspect")
async def inspect_cache_v1(
    prefix: Annotated[
        str | None, Query(description="Hash prefix to filter entries (hex, e.g., 'a1b2')")
    ] = None,
    table_id: Annotated[str | None, Query(description="Filter by table identifier")] = None,
    snapshot_id: Annotated[int | None, Query(description="Filter by snapshot ID")] = None,
    limit: Annotated[int, Query(description="Maximum entries to return", ge=1, le=1000)] = 100,
):
    """Inspect cache entries with detailed diagnostics (admin endpoint).

    This endpoint provides low-level cache inspection for debugging and
    operational troubleshooting. Use it to:
    - Verify specific entries are cached
    - Debug cache key hashing issues
    - Understand cache distribution by prefix
    - Inspect metadata for specific tables/snapshots

    Query parameters:
    - prefix: Filter by cache key hash prefix (hex string)
    - table_id: Filter by table identifier
    - snapshot_id: Filter by snapshot ID
    - limit: Max entries to return (default 100, max 1000)

    Returns detailed information including:
    - Cache key hash (for debugging key generation)
    - File path on disk
    - Metadata (table, snapshot, row group, columns)
    - File size and creation time
    """
    import json as json_module

    from strata.cache import CACHE_FILE_EXTENSION, CACHE_META_EXTENSION, CACHE_VERSION, DiskCache

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")

    results = []
    versioned_dir = cache.cache_dir / f"v{CACHE_VERSION}"

    if not versioned_dir.exists():
        return {
            "cache_version": CACHE_VERSION,
            "cache_dir": str(cache.cache_dir),
            "entries": [],
            "total_matched": 0,
            "truncated": False,
        }

    # If prefix is provided, narrow the search
    if prefix:
        # Normalize to lowercase
        prefix = prefix.lower()
        # Build search paths based on prefix length
        if len(prefix) >= 4:
            # Can go directly to specific subdirectory
            search_dir = versioned_dir / prefix[:2] / prefix[2:4]
            if not search_dir.exists():
                return {
                    "cache_version": CACHE_VERSION,
                    "cache_dir": str(cache.cache_dir),
                    "prefix_filter": prefix,
                    "entries": [],
                    "total_matched": 0,
                    "truncated": False,
                }
            search_paths = [search_dir]
        elif len(prefix) >= 2:
            # Search within first-level subdirectory
            search_dir = versioned_dir / prefix[:2]
            if not search_dir.exists():
                return {
                    "cache_version": CACHE_VERSION,
                    "cache_dir": str(cache.cache_dir),
                    "prefix_filter": prefix,
                    "entries": [],
                    "total_matched": 0,
                    "truncated": False,
                }
            search_paths = [search_dir]
        else:
            # Search everything but filter by prefix
            search_paths = [versioned_dir]
    else:
        search_paths = [versioned_dir]

    matched_count = 0
    truncated = False

    for search_path in search_paths:
        for meta_path in search_path.rglob(f"*{CACHE_META_EXTENSION}"):
            # Extract hash from filename
            cache_hash = meta_path.stem.replace(".meta", "")

            # Apply prefix filter
            if prefix and not cache_hash.startswith(prefix):
                continue

            try:
                meta_data = json_module.loads(meta_path.read_text())

                # Apply table_id filter
                if table_id and meta_data.get("table_id") != table_id:
                    continue

                # Apply snapshot_id filter
                if snapshot_id is not None and meta_data.get("snapshot_id") != snapshot_id:
                    continue

                matched_count += 1

                if len(results) >= limit:
                    truncated = True
                    continue  # Keep counting but don't add more

                # Get data file info
                data_path = meta_path.with_suffix(CACHE_FILE_EXTENSION)
                file_size = data_path.stat().st_size if data_path.exists() else None
                file_exists = data_path.exists()

                results.append(
                    {
                        "hash": cache_hash,
                        "hash_prefix": cache_hash[:8],
                        "file_path": str(data_path.relative_to(cache.cache_dir)),
                        "file_exists": file_exists,
                        "file_size_bytes": file_size,
                        "metadata": meta_data,
                    }
                )

            except Exception as e:
                # Include corrupted entries for debugging
                results.append(
                    {
                        "hash": cache_hash,
                        "file_path": str(meta_path.relative_to(cache.cache_dir)),
                        "error": str(e),
                        "corrupted": True,
                    }
                )
                matched_count += 1

    # Sort by hash for consistent output
    results.sort(key=lambda x: x.get("hash", ""))

    response = {
        "cache_version": CACHE_VERSION,
        "cache_dir": str(cache.cache_dir),
        "entries": results,
        "total_matched": matched_count,
        "truncated": truncated,
    }

    if prefix:
        response["prefix_filter"] = prefix
    if table_id:
        response["table_id_filter"] = table_id
    if snapshot_id is not None:
        response["snapshot_id_filter"] = snapshot_id

    return response


@app.post("/v1/cache/clear", dependencies=[require_scope("admin:cache")])
async def clear_cache_v1():
    """Clear the disk cache.

    Requires admin:cache scope when auth_mode=trusted_proxy.
    """
    state = get_state()

    try:
        state.fetcher.cache.clear()
        state.metrics.reset()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cache/warm", response_model=WarmResponse)
async def warm_cache_v1(request: WarmRequest):
    """Warm the cache for specified tables.

    Preloads row group data into the cache so subsequent queries are fast.
    This is useful for:
    - Warming cache after server restart
    - Preloading data before a batch of dashboards query it
    - Ensuring low latency for critical tables

    The operation runs synchronously and returns when all row groups
    have been fetched and cached (or skipped if already cached).

    Request body:
    - tables: List of table URIs to warm (e.g., "file:///warehouse#ns.table")
    - columns: Optional column projection (None = all columns)
    - max_row_groups: Optional limit per table (None = all row groups)
    - concurrent: Max concurrent fetches (default 4)

    Returns:
    - tables_warmed: Number of tables processed
    - row_groups_cached: Total row groups written to cache
    - row_groups_skipped: Already in cache (cache hits)
    - bytes_written: Total bytes written to cache
    - elapsed_ms: Total time taken
    - errors: Any errors encountered (list of error messages)
    """
    state = get_state()

    start_time = time.perf_counter()
    tables_warmed = 0
    row_groups_cached = 0
    row_groups_skipped = 0
    bytes_written = 0
    errors: list[str] = []

    # Limit concurrency for cache warming
    warming_semaphore = asyncio.Semaphore(request.concurrent)

    async def fetch_task(task: Task) -> tuple[bool, int]:
        """Fetch a single task, return (was_cached, bytes_written)."""
        async with warming_semaphore:
            try:
                # Run fetch in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state.fetcher.fetch_as_stream_bytes, task)
                if task.cached:
                    return (True, 0)  # Already cached
                else:
                    return (False, task.bytes_read)
            except Exception:
                return (False, 0)

    for table_uri in request.tables:
        try:
            # Plan the table
            plan = state.planner.plan(
                table_uri=table_uri,
                snapshot_id=None,  # Current snapshot
                columns=request.columns,
                filters=[],
            )

            # Limit row groups if specified
            tasks = plan.tasks
            if request.max_row_groups is not None:
                tasks = tasks[: request.max_row_groups]

            if not tasks:
                tables_warmed += 1
                continue

            # Fetch all tasks concurrently (bounded by semaphore)
            results = await asyncio.gather(
                *[fetch_task(task) for task in tasks],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, BaseException):
                    continue
                was_cached, written = result
                if was_cached:
                    row_groups_skipped += 1
                else:
                    row_groups_cached += 1
                    bytes_written += written

            tables_warmed += 1

        except Exception as e:
            errors.append(f"{table_uri}: {e!s}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Log the warming operation
    state.metrics.log_event(
        "cache_warm",
        tables_warmed=tables_warmed,
        row_groups_cached=row_groups_cached,
        row_groups_skipped=row_groups_skipped,
        bytes_written=bytes_written,
        elapsed_ms=elapsed_ms,
        errors_count=len(errors),
    )

    return WarmResponse(
        tables_warmed=tables_warmed,
        row_groups_cached=row_groups_cached,
        row_groups_skipped=row_groups_skipped,
        bytes_written=bytes_written,
        elapsed_ms=elapsed_ms,
        errors=errors,
    )


@app.post("/v1/cache/warm/async", response_model=WarmAsyncResponse)
async def warm_cache_async_v1(request: WarmAsyncRequest):
    """Start an async/background cache warming job.

    Unlike POST /v1/cache/warm (which blocks until complete), this endpoint
    starts a background job and returns immediately with a job ID for tracking.

    This is useful for:
    - Warming large tables without blocking the request
    - Scheduling warmup before batch operations
    - Warming specific snapshots (not just current)

    Request body:
    - tables: List of table URIs to warm
    - columns: Optional column projection (None = all columns)
    - snapshot_id: Optional specific snapshot (None = current)
    - max_row_groups: Optional limit per table (None = all)
    - concurrent: Max concurrent fetches within job (default 4)
    - priority: Job priority (higher = more urgent, default 0)

    Returns:
    - job_id: Unique ID for tracking progress via GET /v1/cache/warm/jobs/{id}
    - status: Initial job status (pending or running)
    - tables_count: Number of tables in the job
    - message: Human-readable status message
    """
    state = get_state()

    if state._cache_warmer is None:
        raise HTTPException(status_code=503, detail="Cache warmer not initialized")

    job_id = await state._cache_warmer.start_job(request)

    return WarmAsyncResponse(
        job_id=job_id,
        status=WarmJobStatus.PENDING,
        tables_count=len(request.tables),
        message=f"Warming job started with {len(request.tables)} tables",
    )


@app.get("/v1/cache/warm/jobs")
async def list_warm_jobs_v1(
    include_completed: Annotated[bool, Query(description="Include completed/failed jobs")] = False,
):
    """List all cache warming jobs.

    Returns a list of all warming jobs with their current status and progress.
    By default only shows pending and running jobs.

    Query params:
    - include_completed: Include completed/failed/cancelled jobs (default false)

    Returns:
    - jobs: List of job progress objects
    """
    state = get_state()

    if state._cache_warmer is None:
        return {"jobs": []}

    jobs = state._cache_warmer.list_jobs(include_completed=include_completed)
    return {"jobs": [j.model_dump() for j in jobs]}


@app.get("/v1/cache/warm/jobs/{job_id}", response_model=WarmJobProgress)
async def get_warm_job_v1(job_id: str):
    """Get progress for a specific warming job.

    Returns detailed progress information for a warming job including:
    - Current status (pending, running, completed, failed, cancelled)
    - Tables completed vs total
    - Row groups cached vs skipped
    - Bytes written
    - Elapsed time
    - Current table being warmed
    - Any errors encountered

    Path params:
    - job_id: Job ID returned from POST /v1/cache/warm/async
    """
    state = get_state()

    if state._cache_warmer is None:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = state._cache_warmer.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return progress


@app.delete("/v1/cache/warm/jobs/{job_id}")
async def cancel_warm_job_v1(job_id: str):
    """Cancel a running warming job.

    Cancels the job and stops any in-progress warming operations.
    Already-cached data is not removed.

    Path params:
    - job_id: Job ID to cancel

    Returns:
    - cancelled: True if job was cancelled
    - message: Human-readable result message
    """
    state = get_state()

    if state._cache_warmer is None:
        raise HTTPException(status_code=404, detail="Job not found")

    cancelled = await state._cache_warmer.cancel_job(job_id)

    if cancelled:
        return {"cancelled": True, "message": f"Job {job_id} cancelled"}
    else:
        raise HTTPException(
            status_code=404,
            detail="Job not found or already completed",
        )


# ---------------------------------------------------------------------------
# Artifact Endpoints (Personal Mode Only)
# ---------------------------------------------------------------------------


def _get_artifact_store(
    allow_server_mode: bool = False,
    allow_read: bool = False,
    allow_write: bool = False,
):
    """Get the artifact store, raising 403 if not in appropriate mode.

    Args:
        allow_server_mode: If True, also allow access when server-mode
            transforms are enabled (for materialize endpoint).
        allow_read: If True, permit access in service mode for read-only
            endpoints. The caller MUST then gate by tenant
            (``_ensure_artifact_access``) and table ACL
            (``_authorize_artifact_read``) — reads are shared-cache results, and
            "result retrieval is ACL-gated" (not blanket-blocked by mode).
        allow_write: If True, permit access in service mode for write endpoints
            when ``service_writes_enabled`` (authenticated write-back). The caller
            MUST then gate by ``_authorize_artifact_write`` (trusted-proxy auth +
            ``artifacts:write`` scope); writes stamp the caller's tenant/principal.
    """
    from strata.artifact_store import get_artifact_store

    state = get_state()

    # Check if access is allowed
    writes_ok = state.config.writes_enabled  # personal mode
    server_transforms_ok = allow_server_mode and state.config.server_transforms_enabled
    service_write_ok = allow_write and state.config.service_writes_enabled

    if not (writes_ok or server_transforms_ok or allow_read or service_write_ok):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "writes_disabled",
                "message": (
                    "Artifact endpoints are disabled in service mode. "
                    "Set deployment_mode='personal' for local development, "
                    "or enable server-mode transforms."
                ),
            },
        )

    store = get_artifact_store(state.config.artifact_dir)
    if store is None:
        # A service-mode read gateway without an artifact_dir simply has no
        # persisted artifacts to read.
        raise HTTPException(
            status_code=404 if allow_read else 500,
            detail="Artifact store not available in this deployment",
        )
    return store


def _get_artifact_request_tenant() -> str | None:
    """Return the tenant filter for direct artifact endpoints.

    When trusted-proxy auth is enabled, direct artifact reads/writes should be
    scoped to the caller's tenant unless the caller is an admin. Tenantless
    artifacts remain visible for backwards compatibility with legacy data.
    """
    state = get_state()
    if state.config.auth_mode != "trusted_proxy":
        return None

    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if principal.has_scope("admin:*"):
        return None
    if principal.tenant is None:
        raise HTTPException(
            status_code=400,
            detail="Tenant header required for artifact operations",
        )

    return principal.tenant


def _ensure_artifact_access(
    artifact,
    tenant_filter: str | None,
    resource_type: str = "artifact",
):
    """Authorize access to a concrete artifact record for direct endpoints."""
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"{resource_type.capitalize()} not found")

    if tenant_filter is None or artifact.tenant is None or artifact.tenant == tenant_filter:
        return artifact

    if get_state().config.hide_forbidden_as_not_found:
        raise HTTPException(status_code=404, detail=f"{resource_type.capitalize()} not found")
    raise HTTPException(status_code=403, detail=f"Access denied to {resource_type}")


def _validate_transform_allowed(executor_ref: str, principal=None):
    """Validate transform is allowed in server mode and return its definition.

    In personal mode, all transforms are allowed (returns None).
    In server mode with transforms enabled, validates against registry.

    Args:
        executor_ref: Executor reference (e.g., "local://duckdb_sql@v1")

    Returns:
        TransformDefinition if in server mode and found, None in personal mode

    Raises:
        HTTPException: 403 if transform is not allowed in server mode
    """
    from strata.transforms.registry import get_transform_registry

    state = get_state()

    # Personal mode: the embedded runner executes builds in-process, so an
    # executor the registry can't resolve would otherwise sit in 'building'
    # forever. Fail fast instead (#123-adjacent: no silent dead-ends).
    if state.config.writes_enabled:
        registry = get_transform_registry()
        defn = registry.get(executor_ref)
        if defn is None:
            available = sorted(d.ref for d in registry.definitions)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "transform_unknown",
                    "message": f"Transform '{executor_ref}' is not registered on "
                    "this server — nothing can execute it. "
                    f"Available transforms: {available}.",
                    "executor": executor_ref,
                },
            )
        return defn

    # Server mode with transforms enabled: validate against registry
    if state.config.server_transforms_enabled:
        registry = get_transform_registry()
        defn = registry.get(executor_ref)

        if defn is None:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "transform_not_allowed",
                    "message": f"Transform '{executor_ref}' is not registered. "
                    "Contact your administrator to add it to the allowlist.",
                    "executor": executor_ref,
                },
            )

        if (
            defn.requires_scope
            and state.config.auth_mode == "trusted_proxy"
            and not (principal and principal.has_scope(defn.requires_scope))
        ):
            if principal is None:
                raise HTTPException(status_code=401, detail="Unauthorized")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "insufficient_scope",
                    "message": (
                        f"Transform '{executor_ref}' requires scope '{defn.requires_scope}'."
                    ),
                    "required_scope": defn.requires_scope,
                    "executor": executor_ref,
                },
            )

        return defn

    # Shouldn't reach here if called correctly
    raise HTTPException(
        status_code=403,
        detail={
            "error": "writes_disabled",
            "message": "Artifact endpoints are disabled.",
        },
    )


def _authorize_table_access(table_uri: str, table_identity) -> None:
    """Enforce table-level ACL under trusted-proxy auth; no-op otherwise.

    The direct scan path (``scan@v1``) and every path that resolves a *table*
    URI as a transform input share this one gate, so a table input can't be used
    to read a table the caller is denied on the direct scan path (the ACL was
    previously only checked on ``scan@v1``, letting transforms bypass it).

    Raises:
        HTTPException: 401 if no principal; 403/404 (per
        ``hide_forbidden_as_not_found``) if the ACL denies the table.
    """
    state = get_state()
    if state.config.auth_mode != "trusted_proxy":
        return

    from strata.auth import AclEvaluator

    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    table_ref = TableRef.from_table_identity(table_identity, table_uri=table_uri)
    if not AclEvaluator(state.config.acl_config).authorize(principal, table_ref):
        if state.config.hide_forbidden_as_not_found:
            raise HTTPException(status_code=404, detail="Table not found")
        raise HTTPException(status_code=403, detail="Access denied")


def _authorize_artifact_read(artifact) -> None:
    """ACL-gate reading an artifact's bytes/metadata under trusted-proxy auth.

    The artifact cache is shared across principals, and "result retrieval is
    ACL-gated": a principal denied a table must not read it back via a cached
    scan result. Re-checks the table ACL for each *table* input in the
    artifact's provenance — the table identity is parsed straight from the
    stored transform_spec (no Iceberg I/O), mirroring the table-input check the
    build path applies in `_resolve_input_version`. No-op without trusted-proxy
    auth (tenant scoping via `_ensure_artifact_access` still applies).
    """
    state = get_state()
    if state.config.auth_mode != "trusted_proxy":
        return
    if not getattr(artifact, "transform_spec", None):
        return

    from strata.artifact_store import TransformSpec
    from strata.iceberg import PyIcebergCatalog
    from strata.types import TableIdentity

    try:
        spec = TransformSpec.from_json(artifact.transform_spec)
    except Exception:
        return  # unparseable spec → no table inputs to gate

    for input_uri in spec.inputs:
        if not (input_uri.startswith("file://") or input_uri.startswith("s3://")):
            continue
        if "#" not in input_uri:
            continue  # not a `…#namespace.table` reference
        _, table_id = PyIcebergCatalog.parse_table_uri(input_uri)
        try:
            identity = TableIdentity.from_table_id(table_id)
        except ValueError:
            continue
        _authorize_table_access(input_uri, identity)


def _authorize_artifact_write() -> None:
    """Gate a write endpoint (put / set_name / set_alias / tags).

    Personal mode: unrestricted (existing behavior). Service mode (only reachable
    with ``service_writes_enabled``, enforced by ``_get_artifact_store(allow_write
    =True)``): require trusted-proxy auth and the ``artifacts:write`` scope. The
    endpoint stamps the caller's tenant + principal, so the write lands in the
    caller's team namespace and can't target another tenant.
    """
    state = get_state()
    if state.config.writes_enabled:
        return  # personal mode — unrestricted

    # Service-mode authenticated write-back.
    if state.config.auth_mode != "trusted_proxy":
        # Also prevented by validate_mode_coherence; belt and suspenders.
        raise HTTPException(status_code=403, detail="Writes require trusted-proxy auth")
    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if not principal.has_scope("artifacts:write"):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "missing_scope",
                "message": "Publishing to the store requires the 'artifacts:write' scope.",
            },
        )


def _resolve_input_version(input_uri: str, tenant: str | None = None) -> str:
    """Resolve an input URI to its current version string.

    For table URIs (file:// or s3://): returns the current snapshot ID
    For artifact URIs (strata://artifact/...): returns the artifact version
    For artifact names (strata://name/...): resolves to artifact version

    Args:
        input_uri: Input URI to resolve

    Returns:
        Version string (snapshot ID for tables, "artifact_id@v=N" for artifacts)

    Raises:
        HTTPException: If input cannot be resolved
    """
    import re

    store = _get_artifact_store(allow_server_mode=True)
    state = get_state()

    # Artifact URI: strata://artifact/{id}@v={version}
    if input_uri.startswith("strata://artifact/"):
        match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
        if match:
            artifact_id = match.group(1)
            version = int(match.group(2))
            return f"{artifact_id}@v={version}"
        raise HTTPException(status_code=400, detail=f"Invalid artifact URI: {input_uri}")

    # Name URI: strata://name/{name}
    if input_uri.startswith("strata://name/"):
        name = input_uri.replace("strata://name/", "")
        artifact = store.resolve_name(name, tenant=tenant)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"Name not found: {name}")
        return f"{artifact.id}@v={artifact.version}"

    # Table URI: file:// or s3://
    if input_uri.startswith("file://") or input_uri.startswith("s3://"):
        try:
            # Get current snapshot ID from planner
            plan = state.planner.plan(
                table_uri=input_uri,
                snapshot_id=None,  # Current snapshot
                columns=None,
                filters=None,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not resolve table {input_uri}: {str(e)}",
            ) from e
        # Gate table inputs through the same ACL as the direct scan path — a
        # transform input must not bypass it. Raised outside the try so a
        # 403/404 isn't rewritten into a 400.
        _authorize_table_access(input_uri, plan.table_identity)
        return str(plan.snapshot_id)

    # Unknown URI type
    raise HTTPException(status_code=400, detail=f"Unknown input URI type: {input_uri}")


@app.post("/v1/artifacts/materialize", response_model=MaterializeResponse)
async def materialize_artifact(request: MaterializeRequest):
    """Materialize a computed artifact.

    This endpoint supports two modes:
    1. Personal mode: Returns build_spec for client-side execution
    2. Server mode (transforms enabled): Validates transform against allowlist
       and returns build_spec for server-orchestrated execution

    Flow:
    1. Validate transform is allowed (server mode only)
    2. Resolve input versions (snapshot IDs for tables, artifact versions for artifacts)
    3. Compute provenance hash from resolved versions + transform
    4. If cached, return artifact URI (hit=True)
    5. If not cached, create building artifact with input_versions and return build spec

    Returns:
        MaterializeResponse with hit status and artifact URI or build spec
    """
    import uuid

    from strata.artifact_store import TransformSpec, compute_provenance_hash
    from strata.auth import get_principal

    # Get tenant and principal from auth context early for artifact isolation
    principal = get_principal()
    tenant_id = principal.tenant if principal else None
    principal_id = principal.id if principal else None

    # Parse transform spec early so we can validate it
    transform = request.transform
    executor_ref = transform.executor

    # Validate transform is allowed (raises 403 if not)
    # In personal mode this returns None (no validation needed)
    # In server mode this returns the TransformDefinition
    transform_defn = _validate_transform_allowed(executor_ref, principal=principal)

    # Get artifact store (allows server mode when transforms are enabled)
    store = _get_artifact_store(allow_server_mode=True)

    transform_spec = TransformSpec(
        executor=executor_ref,
        params=transform.params,
        inputs=request.inputs,
    )

    # Resolve input versions for both hashing and staleness tracking
    # If resolution fails, fall back to using the URI as the version (legacy behavior)
    input_versions: dict[str, str] = {}
    for input_uri in request.inputs:
        try:
            input_versions[input_uri] = _resolve_input_version(input_uri, tenant=tenant_id)
        except HTTPException as e:
            # Authz / not-found failures must propagate: a denied table input
            # (403 from the table ACL) or a missing name (404) must never fall
            # back to building anyway. Only the "unresolvable URI" 400 — fake or
            # legacy URIs used in tests — uses the raw URI as its version.
            if e.status_code in (401, 403, 404):
                raise
            input_versions[input_uri] = input_uri

    # Use resolved versions as hashes for provenance calculation
    input_hashes = [f"{uri}:{version}" for uri, version in sorted(input_versions.items())]

    provenance_hash = compute_provenance_hash(input_hashes, transform_spec)

    # Check for existing artifact with same provenance (tenant-scoped)
    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None and not request.refresh:
        artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"

        # Optionally set name (tenant-scoped)
        if request.name:
            store.set_name(request.name, existing.id, existing.version, tenant=tenant_id)

        return MaterializeResponse(
            hit=True,
            artifact_uri=artifact_uri,
            build_spec=None,
            state="ready",
        )

    # Cache miss - create new artifact in building state (tenant-scoped).
    # A refresh rebuild reuses the existing artifact id so it becomes a new
    # version of the same artifact; finalize supersedes the old ready version
    # and provenance lookups resolve to the rebuild (#123).
    if request.refresh and existing is not None:
        artifact_id = existing.id
    else:
        artifact_id = str(uuid.uuid4())
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=transform_spec,
        input_versions=input_versions,  # Track for staleness detection
        tenant=tenant_id,
        principal=principal_id,
    )

    artifact_uri = f"strata://artifact/{artifact_id}@v={version}"
    state = get_state()

    # Create build record for async execution by the build runner
    # (service mode with transforms enabled, or personal mode embedded).
    if state.config.transforms_runtime_enabled:
        from strata.transforms.build_qos import (
            BuildQoSError,
            get_build_qos,
        )
        from strata.transforms.build_store import get_build_store

        build_id = str(uuid.uuid4())

        # Get build store (uses same directory as artifact store)
        if state.config.artifact_dir is None:
            raise HTTPException(status_code=500, detail="Artifact directory not configured")
        build_store = get_build_store(state.config.artifact_dir / "artifacts.sqlite")
        if build_store is None:
            raise HTTPException(
                status_code=500,
                detail="Build store not initialized",
            )

        # Use tenant for build QoS (fallback to __default__ for QoS tracking)
        build_tenant_id = _normalized_build_qos_tenant_id(tenant_id)
        estimated_output_bytes = _estimate_transform_output_bytes(
            state,
            transform_defn.max_output_bytes if transform_defn is not None else None,
        )

        # Build QoS admission control
        # Classify build and check quotas before creating the build
        build_qos = get_build_qos()
        build_slot = None

        if build_qos is not None:
            # Classify and admit the build using the best available size estimate.
            priority = build_qos.classify_build(
                estimated_output_bytes=estimated_output_bytes,
                input_count=len(request.inputs),
            )

            try:
                await build_qos.check_quota(build_tenant_id, estimated_output_bytes)
            except BuildQoSError as e:
                # Quota exceeded - clean up artifact and return 429
                store.fail_artifact(artifact_id, version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": str(int(e.retry_after or 5))},
                )

            # Acquire build slot (early rejection if at capacity)
            try:
                build_slot = await build_qos.acquire(build_tenant_id, priority)
            except BuildQoSError as e:
                # At capacity - clean up artifact and return 429
                store.fail_artifact(artifact_id, version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": str(int(e.retry_after or 5))},
                )

        try:
            # Create build record
            build_store.create_build(
                build_id=build_id,
                artifact_id=artifact_id,
                version=version,
                executor_ref=executor_ref,
                executor_url=transform_defn.executor_url if transform_defn else None,
                tenant_id=tenant_id,
                principal_id=principal_id,
                input_uris=request.inputs,
                params=transform.params,
                name=request.name,
            )

            # Build is now queued - release the admission slot
            # The runner has its own concurrency control for execution
            if build_slot:
                await build_slot.release()

            return MaterializeResponse(
                hit=False,
                artifact_uri=artifact_uri,
                build_id=build_id,
                state="pending",
            )
        except Exception:
            # Release slot on failure
            if build_slot:
                await build_slot.release()
            raise

    # Personal mode: return build spec for client-side execution
    build_spec = BuildSpec(
        artifact_id=artifact_id,
        version=version,
        executor=transform_spec.executor,
        params=transform_spec.params,
        input_uris=request.inputs,
    )

    return MaterializeResponse(
        hit=False,
        artifact_uri=artifact_uri,
        build_spec=build_spec.model_dump(),
        state="building",
    )


@app.post("/v1/artifacts/upload/{artifact_id}/v/{version}")
async def upload_artifact_blob(artifact_id: str, version: int, request: Request):
    """Upload artifact blob data (personal mode only).

    The client POSTs raw Arrow IPC stream bytes to this endpoint.
    After upload, call /v1/artifacts/finalize to complete the artifact.

    Args:
        artifact_id: Artifact ID from materialize response
        version: Version number from materialize response
        request: Raw request body containing Arrow IPC bytes
    """
    store = _get_artifact_store()

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
        try:
            staged.unlink()
        except FileNotFoundError:
            pass

    return {"status": "uploaded", "byte_size": byte_size}


@app.post("/v1/artifacts/finalize", response_model=UploadFinalizeResponse)
async def finalize_artifact(request: UploadFinalizeRequest):
    """Finalize an artifact after upload (personal mode only).

    After uploading the blob, call this to transition the artifact to ready state.
    Optionally sets a name pointer to the artifact.

    Returns:
        UploadFinalizeResponse with artifact URI and optional name URI
    """
    store = _get_artifact_store()

    # Verify blob exists
    if not store.blob_exists(request.artifact_id, request.version):
        raise HTTPException(
            status_code=400,
            detail="Blob not uploaded. Call upload endpoint first.",
        )

    # Get blob size without materializing the payload
    byte_size = store.blob_size(request.artifact_id, request.version) or 0

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


@app.put("/v1/artifacts", response_model=PutArtifactResponse)
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


@app.get("/v1/artifacts/{artifact_id}/v/{version}", response_model=ArtifactInfoResponse)
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


# ---------------------------------------------------------------------------
# Registry: aliases, tags, audit (#129)
# ---------------------------------------------------------------------------


class AliasSetRequest(BaseModel):
    artifact_id: str
    version: int


@app.put("/v1/names/{name:path}/aliases/{alias}")
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


@app.get("/v1/names/{name:path}/aliases/{alias}")
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


@app.delete("/v1/names/{name:path}/aliases/{alias}")
async def delete_alias(
    name: str, alias: str, store: PersonalModeStore, principal: CurrentPrincipal
):
    """Delete ``name @ alias`` (pending queue for protected aliases)."""
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


@app.get("/v1/names/{name:path}/aliases")
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


@app.put("/v1/artifacts/{artifact_id}/v/{version}/tags")
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


@app.get("/v1/artifacts/{artifact_id}/v/{version}/tags")
async def get_tags(artifact_id: str, version: int, store: ReadStore, principal: CurrentPrincipal):
    """Get the tags on an artifact version."""
    tenant_id = principal.tenant if principal else None

    return {
        "artifact_id": artifact_id,
        "version": version,
        "tags": store.get_tags(artifact_id, version, tenant=tenant_id),
    }


@app.delete("/v1/artifacts/{artifact_id}/v/{version}/tags/{key}")
async def delete_tag(
    artifact_id: str, version: int, key: str, store: WriteStore, principal: CurrentPrincipal
):
    """Delete one tag from an artifact version."""
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    if not store.delete_tag(artifact_id, version, key, tenant=tenant_id, actor=actor):
        raise HTTPException(status_code=404, detail=f"Tag '{key}' not found")
    return {"status": "deleted", "artifact_id": artifact_id, "version": version, "key": key}


@app.get("/v1/registry/audit")
async def registry_audit(
    store: ReadStore,
    principal: CurrentPrincipal,
    name: str | None = None,
    artifact_id: str | None = None,
    limit: int = 100,
):
    """Read the append-only registry audit, newest first.

    Scoped to the caller's tenant: a principal sees only its own tenant's
    history. ``admin:*`` (and personal mode, where there is no principal)
    sees the whole store — every other registry route scopes the same way.
    """
    if principal is None or principal.has_scope("admin:*"):
        entries = store.read_audit(name=name, artifact_id=artifact_id, limit=limit)
    else:
        entries = store.read_audit(
            name=name, artifact_id=artifact_id, limit=limit, tenant=principal.tenant
        )
    return {"entries": entries}


@app.get("/v1/registry/summary")
async def registry_summary(store: ReadStore, principal: CurrentPrincipal):
    """Registry state for the dashboard names table: each name with its
    aliases (``alias -> version``), current version, and that version's tags.
    One call instead of ``/v1/names`` + a per-name alias fetch. Tenant-scoped
    like the other registry reads (personal mode / ``admin:*`` see all)."""
    tenant = None if (principal is None or principal.has_scope("admin:*")) else principal.tenant
    return {"names": registry_service.summary(store, tenant=tenant)}


class PendingDecisionRequest(BaseModel):
    name: str
    alias: str


@app.get("/v1/registry/pending")
async def registry_pending(store: ReadStore, principal: CurrentPrincipal):
    """List protected-alias changes awaiting approval."""
    tenant_id = principal.tenant if principal else None
    return {"pending": store.list_pending_changes(tenant=tenant_id)}


def _require_registry_approver():
    """Authorize a registry approval decision; return the principal.

    Approving/rejecting a protected-alias change is a governance action.
    Under trusted-proxy auth it requires the ``admin:registry`` scope
    (``admin:*`` satisfies it). Personal mode (no principal) is the single
    operator and is allowed.
    """
    from strata.auth import get_principal

    state = get_state()
    principal = get_principal()
    if state.config.auth_mode == "trusted_proxy":
        if principal is None or not principal.has_scope("admin:registry"):
            raise HTTPException(
                status_code=403,
                detail="Insufficient scope: admin:registry required to decide "
                "protected-alias changes",
            )
    return principal


@app.post("/v1/registry/pending/approve")
async def approve_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Apply a pending alias change; the approver becomes the audit actor.

    Requires the ``admin:registry`` scope under trusted-proxy auth, and
    enforces separation of duty — the requester cannot self-approve unless
    they hold the ``admin:*`` break-glass scope.
    """
    principal, store = decision
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None
    is_superadmin = principal is not None and principal.has_scope("admin:*")

    try:
        applied = store.approve_alias_change(
            request.name,
            request.alias,
            tenant=tenant_id,
            actor=actor,
            require_distinct_approver=not is_superadmin,
        )
    except ValueError as e:
        msg = str(e)
        status = 403 if msg.startswith("Separation of duty") else 404
        raise HTTPException(status_code=status, detail=msg)
    return {"status": "approved", "applied": applied}


@app.post("/v1/registry/pending/reject")
async def reject_pending(request: PendingDecisionRequest, decision: RegistryDecisionContext):
    """Discard a pending alias change (audited).

    Requires the ``admin:registry`` scope under trusted-proxy auth so a
    tenant member cannot quietly drop a colleague's pending promotion.
    """
    principal, store = decision
    tenant_id = principal.tenant if principal else None
    actor = principal.id if principal else None

    try:
        rejected = store.reject_alias_change(
            request.name, request.alias, tenant=tenant_id, actor=actor
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"status": "rejected", "rejected": rejected}


# NOTE: the routes below use a greedy {name:path} converter; the registry
# routes above must stay registered FIRST or ".../aliases/x" URLs would be
# swallowed as part of the name. The "/aliases" suffix is reserved.
@app.get("/v1/names/{name:path}", response_model=NameResolveResponse)
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


@app.post("/v1/names", response_model=NameSetResponse)
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


@app.delete("/v1/names/{name:path}")
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


@app.get("/v1/names")
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


@app.get("/v1/artifacts/names/{name:path}/status", response_model=NameStatusResponse)
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
            current_version = _resolve_input_version(input_uri, tenant=tenant_id)
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


# ---------------------------------------------------------------------------
# Lineage and Dependency Introspection Endpoints
# ---------------------------------------------------------------------------


@app.get(
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


@app.get(
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


_ACTIVE_BUILD_STATES = ("pending", "building", "running")


def _build_transport_available() -> bool:
    """Whether signed build transport APIs are available in the current mode."""
    state = get_state()
    return state.config.writes_enabled or state.config.server_transforms_enabled


def _get_runtime_build_store():
    """Get or initialize the runtime build store for signed build transport APIs."""
    from strata.transforms.build_store import get_build_store

    state = get_state()
    artifact_dir = state.config.artifact_dir
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return get_build_store(artifact_dir / "artifacts.sqlite")


@app.get("/v1/artifacts/builds/{build_id}", response_model=BuildStatusResponse)
async def get_build_status(build_id: str):
    """Get async build status.

    Use this endpoint to poll the status of a build that was started
    asynchronously via materialize, including scan@v1 artifact-mode builds.

    Args:
        build_id: Build ID from materialize response

    Returns:
        BuildStatusResponse with current build state
    """
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


@app.get("/v1/builds/{build_id}", response_model=BuildStatusResponse, include_in_schema=False)
async def get_build_status_compat(build_id: str):
    """Compatibility alias for older clients polling build status."""
    return await get_build_status(build_id)


# ---------------------------------------------------------------------------
# Pull Model Endpoints (Stage 2) - Signed URL based execution
# ---------------------------------------------------------------------------


@app.get("/v1/builds/{build_id}/manifest")
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


@app.get("/v1/artifacts/download")
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
    from strata.transforms.signed_urls import verify_download_signature

    # Parse and verify signature
    try:
        version_int = int(version)
        expires_at_float = float(expires_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter format")

    if not verify_download_signature(
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


@app.post("/v1/artifacts/upload")
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
    from strata.transforms.signed_urls import verify_upload_signature

    # Parse and verify signature
    try:
        max_bytes_int = int(max_bytes)
        expires_at_float = float(expires_at)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid parameter format")

    if not verify_upload_signature(
        build_id=build_id,
        max_bytes=max_bytes_int,
        expires_at=expires_at_float,
        signature=signature,
    ):
        raise HTTPException(status_code=403, detail="Invalid or expired signature")

    # Check build exists and is in correct state
    build_store = _get_runtime_build_store()
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
        try:
            staged.unlink()
        except FileNotFoundError:
            pass

    return {"status": "uploaded", "build_id": build_id, "byte_size": byte_size}


@app.post("/v1/builds/{build_id}/finalize")
async def finalize_build(
    build_id: str,
    request: Request,
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
    if not _build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Build finalize is only available when personal-mode writes "
                "or server-mode transforms are enabled"
            ),
        )

    build_store = _get_runtime_build_store()
    if build_store is None:
        raise HTTPException(status_code=500, detail="Build store not initialized")

    build = build_store.get_build(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail="Build not found")

    if signature is not None or expires_at is not None:
        from strata.transforms.signed_urls import verify_finalize_signature

        if signature is None or expires_at is None:
            raise HTTPException(status_code=400, detail="Missing finalize signature parameters")
        try:
            expires_at_float = float(expires_at)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid parameter format")

        if not verify_finalize_signature(
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
                try:
                    staged_path.unlink()
                except FileNotFoundError:
                    pass
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
    await _record_build_output_bytes(build.tenant_id, byte_size)

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


@app.post("/v1/artifacts/explain-materialize", response_model=ExplainMaterializeResponse)
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
    from strata.artifact_store import TransformSpec, compute_provenance_hash

    # Get tenant from auth context for artifact isolation
    tenant_id = principal.tenant if principal else None

    # Parse transform spec
    transform = request.transform
    transform_spec = TransformSpec(
        executor=transform.executor,
        params=transform.params,
        inputs=request.inputs,
    )

    # Resolve current input versions
    resolved_versions: dict[str, str] = {}
    for input_uri in request.inputs:
        try:
            resolved_versions[input_uri] = _resolve_input_version(input_uri, tenant=tenant_id)
        except HTTPException as e:
            resolved_versions[input_uri] = f"<error: {e.detail}>"

    # Compute provenance hash with current versions
    input_hashes = [f"{uri}:{version}" for uri, version in sorted(resolved_versions.items())]
    provenance_hash = compute_provenance_hash(input_hashes, transform_spec)

    # Check for existing artifact with same provenance (tenant-scoped)
    existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
    if existing is not None:
        return ExplainMaterializeResponse(
            would_hit=True,
            artifact_uri=f"strata://artifact/{existing.id}@v={existing.version}",
            would_build=False,
            resolved_input_versions=resolved_versions,
        )

    # Cache miss - check if there's an existing named artifact that would be stale
    changed_inputs: list[InputChangeInfo] = []
    is_stale = False
    stale_reason = None
    existing_artifact_uri = None

    if request.name:
        name_status = store.get_name_status(request.name, tenant=tenant_id)
        if name_status is not None:
            existing_artifact_uri = name_status.artifact_uri
            # Compare stored versions vs current
            for input_uri, old_version in name_status.input_versions.items():
                current_version = resolved_versions.get(input_uri)
                if current_version and current_version != old_version:
                    changed_inputs.append(
                        InputChangeInfo(
                            input_uri=input_uri,
                            old_version=old_version,
                            new_version=current_version,
                        )
                    )

            is_stale = len(changed_inputs) > 0
            if is_stale:
                changes = [
                    f"{c.input_uri}: {c.old_version} → {c.new_version}" for c in changed_inputs
                ]
                stale_reason = f"Rebuild needed: {', '.join(changes)}"

    return ExplainMaterializeResponse(
        would_hit=False,
        artifact_uri=existing_artifact_uri,
        would_build=True,
        is_stale=is_stale,
        stale_reason=stale_reason,
        changed_inputs=changed_inputs if changed_inputs else None,
        resolved_input_versions=resolved_versions,
    )


@app.get("/v1/artifacts/stats")
async def get_artifact_stats(store: PersonalModeStore, tenant_filter: CurrentTenant):
    """Get artifact store statistics (personal mode only).

    Returns:
        Artifact store statistics
    """
    return store.stats(tenant=tenant_filter)


@app.get("/v1/artifacts/usage")
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


@app.get("/v1/artifacts")
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


@app.delete("/v1/artifacts/{artifact_id}/v/{version}")
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
    _ensure_artifact_access(
        store.get_artifact(artifact_id, version),
        tenant_filter,
    )

    deleted = store.delete_artifact(artifact_id, version, tenant=tenant_filter)
    if not deleted:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return {"deleted": True, "artifact_uri": f"strata://artifact/{artifact_id}@v={version}"}


@app.post("/v1/artifacts/gc")
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


@app.get("/v1/artifacts/{artifact_id}/v/{version}/data")
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


def _parse_artifact_uri(uri: str) -> tuple[str, int] | None:
    """Parse artifact URI to (artifact_id, version).

    Formats:
        strata://artifact/{id}@v={version}
        strata://artifact/{id}  (resolves to latest)

    Returns:
        Tuple of (artifact_id, version) or None if not an artifact URI
    """
    import re

    # Match strata://artifact/{id}@v={version}
    match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", uri)
    if match:
        return (match.group(1), int(match.group(2)))

    # Match strata://artifact/{id} (latest version)
    match = re.match(r"^strata://artifact/([^@]+)$", uri)
    if match:
        return (match.group(1), -1)  # -1 indicates "latest"

    return None


def _parse_name_uri(uri: str) -> str | None:
    """Parse name URI to name.

    Format: strata://name/{name}

    Returns:
        Name string or None if not a name URI
    """
    import re

    match = re.match(r"^strata://name/(.+)$", uri)
    if match:
        return match.group(1)
    return None


def _resolve_artifact_uri(uri: str) -> tuple[str, int] | None:
    """Resolve URI to artifact (id, version).

    Handles:
        strata://artifact/{id}@v={version} -> (id, version)
        strata://artifact/{id} -> (id, latest_version)
        strata://name/{name} -> (resolved_id, resolved_version)

    Returns:
        Tuple of (artifact_id, version) or None if not a Strata URI
    """
    from strata.artifact_store import get_artifact_store

    state = get_state()
    if not state.config.writes_enabled:
        return None  # Artifacts only in personal mode

    store = get_artifact_store(state.config.artifact_dir)
    if store is None:
        return None

    # Try artifact URI
    result = _parse_artifact_uri(uri)
    if result is not None:
        artifact_id, version = result
        if version == -1:
            # Resolve to latest
            latest = store.get_latest_version(artifact_id)
            if latest is not None:
                return (artifact_id, latest.version)
            return None
        return result

    # Try name URI
    name = _parse_name_uri(uri)
    if name is not None:
        artifact = store.resolve_name(name, tenant=_get_artifact_request_tenant())
        if artifact is not None:
            return (artifact.id, artifact.version)
        return None

    return None


# =============================================================================
# Unified Materialize API
# =============================================================================
# This implements the unified /v1/materialize endpoint that replaces both
# /v1/scan and /v1/artifacts/materialize. The key insight is that scanning
# an Iceberg table is a materialize with scan@v1 transform.
# =============================================================================


def _compute_identity_provenance(
    table_identity: str,
    snapshot_id: int,
    columns: list[str] | None,
    filters: list,
) -> str:
    """Compute provenance hash for identity transform.

    The hash uniquely identifies a table scan based on:
    - Table identity + snapshot ID
    - Column projection (sorted for determinism)
    - Row filters (normalized)

    This enables query-level deduplication: same query -> same artifact.
    """
    import hashlib

    from strata.types import compute_filter_fingerprint

    hasher = hashlib.sha256()

    # Input: table identity + snapshot
    hasher.update(f"table:{table_identity}@{snapshot_id}".encode())

    # Transform: executor ref
    hasher.update(b"executor:scan@v1")

    # Params: sorted columns
    if columns:
        hasher.update(f"columns:{sorted(columns)}".encode())
    else:
        hasher.update(b"columns:*")

    # Params: normalized filters
    filter_fp = compute_filter_fingerprint(filters)
    hasher.update(f"filters:{filter_fp}".encode())

    return hasher.hexdigest()


@app.post("/v1/materialize", response_model=MaterializeResponse)
async def unified_materialize(request: MaterializeRequest):
    """Unified endpoint for all data access (replaces /v1/scan and /v1/artifacts/materialize).

    This is the single entry point for materializing data in Strata.
    Scanning an Iceberg table is expressed as a materialize with scan@v1.

    Modes:
    - stream (default): Data streams immediately while artifact builds in parallel
    - artifact: Client polls /v1/builds/{build_id} then fetches when ready

    Both modes create and persist artifacts. The mode only affects how
    the client receives data, not whether it's cached.

    For scan@v1 transform:
    - Reads from exactly one Iceberg table input
    - Applies optional column projection and row filtering
    - Executed internally by Strata (no external executor needed)

    For other transforms:
    - Delegates to the existing materialize flow

    Returns:
        MaterializeResponse with hit status and artifact/stream URLs
    """

    state = get_state()
    transform = request.transform

    # Reject new requests during shutdown
    if state._draining:
        raise HTTPException(
            status_code=503,
            detail="Server is shutting down. Not accepting new requests.",
        )

    # Handle scan@v1 transform specially - executed internally
    if transform.executor == "scan@v1":
        return await _handle_identity_materialize(request)

    # For non-identity transforms, delegate to existing materialize flow
    # This reuses the existing /v1/artifacts/materialize logic
    return await _handle_transform_materialize(request)


async def _handle_identity_materialize(
    request: MaterializeRequest,
) -> MaterializeResponse | JSONResponse:
    """Handle scan@v1 transform (internal execution).

    This is the fast path for table scans. The identity transform reads
    from an Iceberg table with optional projection/filtering and returns
    the data unchanged.

    Flow:
    1. Validate exactly one table input
    2. Parse identity params (columns, filters, snapshot_id)
    3. Plan the scan using existing ReadPlanner
    4. Compute provenance hash
    5. Check artifact cache (cache hit -> return immediately)
    6. Cache miss -> create artifact record + stream state
    7. Return stream_url or build_id based on mode
    """
    import uuid

    from strata.artifact_store import TransformSpec as ArtifactTransformSpec
    from strata.artifact_store import get_artifact_store
    from strata.auth import get_principal

    state = get_state()

    # Get tenant and principal from auth context
    principal = get_principal()
    tenant_id = principal.tenant if principal else None
    principal_id = principal.id if principal else None

    # Validate inputs: scan@v1 requires exactly one table input
    if len(request.inputs) != 1:
        raise HTTPException(
            status_code=400,
            detail="scan@v1 transform requires exactly one input",
        )

    table_uri = request.inputs[0]

    # Validate it's a table URI (not an artifact)
    if table_uri.startswith("strata://"):
        raise HTTPException(
            status_code=400,
            detail="scan@v1 transform input must be a table URI, not an artifact",
        )

    # Parse identity params via model_validate so ty doesn't try to
    # narrow each dict[str, object] value against the typed fields.
    try:
        identity_params = IdentityParams.model_validate(request.transform.params)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scan@v1 params: {e}",
        )

    # Convert to internal filter format
    filters = identity_params.to_strata_filters()

    # Plan the scan using existing planner
    plan_timeout = state.config.plan_timeout_seconds

    def do_plan():
        with trace_span(
            "plan_identity_materialize",
            table_uri=table_uri,
            snapshot_id=identity_params.snapshot_id,
            columns_count=len(identity_params.columns) if identity_params.columns else None,
        ) as span:
            plan = state.planner.plan(
                table_uri=table_uri,
                snapshot_id=identity_params.snapshot_id,
                columns=identity_params.columns,
                filters=filters,
            )
            span.set_attribute("scan_id", plan.scan_id)
            span.set_attribute("row_groups_total", plan.total_row_groups)
            span.set_attribute("row_groups_pruned", plan.pruned_row_groups)
            span.set_attribute("estimated_bytes", plan.estimated_bytes)
            return plan

    try:
        plan = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(state._planning_executor, do_plan),
            timeout=plan_timeout,
        )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Planning timed out after {plan_timeout}s.",
        )

    # Enforce task limit
    max_tasks = state.config.max_tasks_per_scan
    if len(plan.tasks) > max_tasks:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Query would read {len(plan.tasks)} row groups, exceeding limit of {max_tasks}."
            ),
        )

    # Pre-flight size check
    max_response = state.config.max_response_bytes
    if plan.estimated_bytes > max_response:
        state.metrics.record_stream_abort_size()
        raise HTTPException(
            status_code=413,
            detail=f"Estimated response size ({plan.estimated_bytes:,} bytes) exceeds limit.",
        )

    # Authorization check (when auth_mode=trusted_proxy)
    if state.config.auth_mode == "trusted_proxy":
        if principal is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        _authorize_table_access(table_uri, plan.table_identity)

        plan.owner_principal = principal.id
        plan.owner_tenant = principal.tenant

    # Compute provenance hash for identity transform
    provenance_hash = _compute_identity_provenance(
        table_identity=str(plan.table_identity),
        snapshot_id=plan.snapshot_id,
        columns=identity_params.columns,
        filters=filters,
    )
    # Get artifact store (allow writes for personal mode)
    store = get_artifact_store(state.config.artifact_dir)

    # Input version for staleness tracking
    input_versions = {table_uri: str(plan.snapshot_id)}

    # Check for existing artifact with same provenance
    existing = None
    if store is not None:
        existing = store.find_by_provenance(provenance_hash, tenant=tenant_id)
        if existing is not None and existing.state == "ready" and not request.refresh:
            artifact_uri = f"strata://artifact/{existing.id}@v={existing.version}"

            # Set name if requested
            if request.name:
                store.set_name(request.name, existing.id, existing.version, tenant=tenant_id)

            logger.info(
                "identity_materialize_cache_hit",
                artifact_id=existing.id,
                table_uri=table_uri,
                snapshot_id=plan.snapshot_id,
            )

            # Provide stream_url for cached data access
            # This allows clients to fetch the data using the same pattern as cache misses
            stream_url = f"/v1/artifacts/{existing.id}/v/{existing.version}/data"

            return MaterializeResponse(
                hit=True,
                artifact_uri=artifact_uri,
                state="ready",
                stream_url=stream_url if request.mode == "stream" else None,
            )

    # Cache miss (or refresh rebuild) - need to build the artifact.
    # A refresh rebuild reuses the existing artifact id so it becomes a new
    # version of the same artifact; finalize supersedes the old ready version
    # and provenance lookups resolve to the rebuild (#123). The stream id
    # stays unique per request — older streams for the same artifact id may
    # still be in the registry.
    if request.refresh and existing is not None:
        artifact_id = existing.id
        stream_id = str(uuid.uuid4())
    else:
        artifact_id = str(uuid.uuid4())
        stream_id = artifact_id  # Use same ID for simplicity

    # Create artifact in building state (if store is available)
    artifact_version = 1
    if store is not None:
        transform_spec = ArtifactTransformSpec(
            executor="scan@v1",
            params=request.transform.params,
            inputs=request.inputs,
        )
        artifact_version = store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions=input_versions,
            tenant=tenant_id,
            principal=principal_id,
        )

    artifact_uri = f"strata://artifact/{artifact_id}@v={artifact_version}"

    # Create stream state for tracking
    stream_state = StreamState(
        stream_id=stream_id,
        plan=plan,
        artifact_id=artifact_id,
        artifact_version=artifact_version,
        created_at=time.time(),
        mode=request.mode,
        name=request.name,
        tenant=tenant_id,
    )

    logger.info(
        "identity_materialize_cache_miss",
        artifact_id=artifact_id,
        stream_id=stream_id,
        table_uri=table_uri,
        snapshot_id=plan.snapshot_id,
        estimated_bytes=plan.estimated_bytes,
        mode=request.mode,
    )

    if request.mode == "stream":
        state._streams[stream_id] = stream_state
        # Register for QoS/accounting only when a client will actually stream.
        state.scans[plan.scan_id] = plan
        _start_prefetch(state, plan)
        _schedule_stream_cleanup(state, stream_id, plan.scan_id)

        return MaterializeResponse(
            hit=False,
            artifact_uri=artifact_uri,
            state="building",
            stream_id=stream_id,
            stream_url=f"/v1/streams/{stream_id}",
        )
    else:
        # Artifact mode persists the result, which requires an artifact store.
        # Without one the background build would no-op and the returned build_id
        # would never resolve — reject up front instead of hanging the client.
        if store is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "mode='artifact' requires an artifact store, but this deployment "
                    "has no artifact_dir configured. Use mode='stream' to scan "
                    "without persistence, or set artifact_dir."
                ),
            )

        # Artifact mode - admit through build QoS, then build in the background.
        from strata.transforms.build_qos import (
            BuildQoSError,
            get_build_qos,
        )

        build_qos = get_build_qos()
        if build_qos is not None:
            qos_tenant_id = _normalized_build_qos_tenant_id(tenant_id)
            estimated_output_bytes = max(0, plan.estimated_bytes)
            priority = build_qos.classify_build(
                estimated_output_bytes=estimated_output_bytes,
                input_count=len(plan.tasks),
            )

            try:
                await build_qos.check_quota(qos_tenant_id, estimated_output_bytes)
                stream_state.build_slot = await build_qos.acquire(qos_tenant_id, priority)
                stream_state.qos_tenant_id = qos_tenant_id
            except BuildQoSError as e:
                if store is not None:
                    store.fail_artifact(artifact_id, artifact_version)
                return JSONResponse(
                    status_code=e.status_code,
                    content=e.to_dict(),
                    headers={"Retry-After": str(int(e.retry_after or 5))},
                )

        state._streams[stream_id] = stream_state
        stream_state.background_task = asyncio.create_task(_build_identity_artifact(stream_state))
        return MaterializeResponse(
            hit=False,
            artifact_uri=artifact_uri,
            state="pending",
            build_id=stream_id,
        )


async def _handle_transform_materialize(request: MaterializeRequest) -> MaterializeResponse:
    """Handle non-identity transforms (external execution).

    Delegates to the existing /v1/artifacts/materialize flow.
    """
    # Reuse the existing materialize_artifact logic
    return await materialize_artifact(request)


@app.get("/v1/streams/{stream_id}")
async def get_stream(stream_id: str, request: Request):
    """Stream Arrow IPC data for a materialize request.

    This endpoint handles streaming mode for unified materialize.
    Data is streamed immediately while the artifact is built in parallel.

    Returns:
        StreamingResponse with Arrow IPC data

    Error codes:
        404: Stream not found
        429: Server at capacity (QoS rate limiting)
    """
    state = get_state()

    # Look up stream state
    if stream_id not in state._streams:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    stream_state = state._streams[stream_id]
    plan = stream_state.plan

    # Reuse the scan-based streaming infrastructure
    # The plan is already registered in state.scans
    scan_id = plan.scan_id

    if scan_id not in state.scans:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")

    # Ownership check (when auth_mode=trusted_proxy)
    if state.config.auth_mode == "trusted_proxy":
        from strata.auth import get_principal

        principal = get_principal()
        if principal is None:
            raise HTTPException(status_code=401, detail="Unauthorized")

        if plan.owner_principal != principal.id:
            if not principal.has_scope("admin:*"):
                if state.config.hide_forbidden_as_not_found:
                    raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
                raise HTTPException(status_code=403, detail="Access denied")

    # Mark stream as started
    _cancel_stream_cleanup(state, stream_id)
    stream_state.started = True
    stream_state.started_at = time.time()

    # QoS: Classify query and acquire appropriate tier limiter
    tier = _classify_query(plan)
    tenant_id = get_tenant_id()
    tenant_registry = get_tenant_registry()
    interactive_limiter, bulk_limiter = tenant_registry.get_or_create_limiters(tenant_id)

    if tier == "interactive":
        limiter = interactive_limiter
        queue_timeout = state.config.interactive_queue_timeout
    else:
        limiter = bulk_limiter
        queue_timeout = state.config.bulk_queue_timeout

    # Per-client fairness
    client_id = request.client.host if request.client else "unknown"
    client_semaphore = _get_client_semaphore(state, client_id, tier)
    client_semaphore_acquired = False

    if client_semaphore is not None:
        try:
            await asyncio.wait_for(client_semaphore.acquire(), timeout=1.0)
            client_semaphore_acquired = True
        except TimeoutError:
            state._client_rejected += 1
            return JSONResponse(
                status_code=429,
                content={"error": "per_client_limit", "tier": tier},
                headers={"Retry-After": "1"},
            )

    # Queue with deadline
    acquired = await limiter.acquire(timeout=queue_timeout)

    if not acquired:
        if client_semaphore_acquired and client_semaphore is not None:
            client_semaphore.release()
        return JSONResponse(
            status_code=429,
            content={"error": "too_many_requests", "tier": tier},
            headers={"Retry-After": str(max(1, int(queue_timeout / 2)))},
        )

    # Track tier for cleanup
    state._scan_tier[scan_id] = tier
    state._scan_client[scan_id] = (client_id, client_semaphore_acquired)

    if tier == "interactive":
        state._active_interactive += 1
    else:
        state._active_bulk += 1

    # Release the QoS resources acquired above. Called exactly once per request,
    # in whichever exit path runs (build cancelled, build failed, or after the
    # blob has finished streaming).
    async def _release_qos() -> None:
        await limiter.release()
        state._scan_tier.pop(scan_id, None)
        scan_client = state._scan_client.pop(scan_id, None)
        if scan_client is not None:
            client_id_cleanup, client_sem_acquired = scan_client
            if client_sem_acquired:
                client_sem = _get_client_semaphore(state, client_id_cleanup, tier)
                if client_sem is not None:
                    client_sem.release()
        if tier == "interactive":
            state._active_interactive -= 1
        else:
            state._active_bulk -= 1
        state._active_scans -= 1

    from strata.artifact_store import get_artifact_store

    store = get_artifact_store(state.config.artifact_dir)
    state._active_scans += 1

    # Service mode (no artifact store): there's nothing to persist, so stream a
    # bounded pass-through straight from the fetcher. With no artifact to
    # finalize, none of the decoupling smells apply — a client disconnect simply
    # ends the generator (the next ``yield`` raises), nothing to poison.
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
                await _release_qos()
                stream_state.completed_at = time.time()
                _schedule_stream_cleanup(state, stream_id, scan_id)

        return StreamingResponse(
            serve_passthrough(),
            media_type="application/vnd.apache.arrow.stream",
        )

    # Personal mode: decouple the artifact build from this client's read. The
    # background build (bounded write-through, PR #165) scans row-group-by-row-
    # group straight to the blob and finalizes ready/failed on its own merits —
    # empty plans included. A slow or dropped reader can no longer poison the
    # cache entry (the #164 IncompleteRead bug): we wait for the build, then
    # serve the persisted blob over the same reliable path as GET .../data.
    if stream_state.background_task is None:
        stream_state.background_task = asyncio.create_task(_build_identity_artifact(stream_state))
    build_task = stream_state.background_task

    # Wait for the build. It runs as a decoupled background task, shielded so a
    # client disconnect never cancels it — the artifact finalizes regardless of
    # who is (or isn't) reading. (A handler-level cancel, e.g. shutdown, frees
    # the slot but leaves the build running.)
    try:
        await asyncio.shield(build_task)
    except asyncio.CancelledError:
        await _release_qos()
        stream_state.completed_at = time.time()
        _schedule_stream_cleanup(state, stream_id, scan_id)
        raise

    # The build is done — the scan is what the QoS slot gated, so release it now,
    # deterministically, in the handler. Serving the already-persisted blob is
    # cheap bounded I/O that needs no scan slot, and releasing here (rather than
    # in the response generator's finally) means a client that vanished before
    # iteration can never strand the slot.
    artifact = store.get_artifact(stream_state.artifact_id, stream_state.artifact_version)
    await _release_qos()
    stream_state.completed = True
    stream_state.completed_at = time.time()

    if artifact is None or artifact.state not in ("ready", "superseded"):
        _schedule_stream_cleanup(state, stream_id, scan_id)
        return JSONResponse(
            status_code=500,
            content={
                "error": "scan_build_failed",
                "detail": stream_state.error_message or "scan build failed",
            },
        )

    # Serve the persisted blob in bounded chunks (no merger, no accumulation, no
    # is_disconnected coupling). A reader that drops mid-send surfaces here as a
    # cancel/close — the artifact is already finalized, so we only count it.
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
            _schedule_stream_cleanup(state, stream_id, scan_id)

    return StreamingResponse(
        serve_blob(),
        media_type="application/vnd.apache.arrow.stream",
        headers={"X-Arrow-Row-Count": str(artifact.row_count or 0)},
    )


async def _finalize_written_blob(stream_state: StreamState, row_count: int, byte_size: int) -> None:
    """Finalize a scan artifact whose blob was already **written through** to the
    blob store (bounded memory) — no full-result buffer.

    The blob is on disk by the time this runs; re-read it in bounded memory (one
    record batch at a time, in a worker thread) for the integrity gate, then flip
    state to ``ready``. The sole finalizer for scan builds — both the stream and
    artifact materialize paths route through the write-through build
    (``_build_identity_artifact``). See docs/internal/design-streaming-decouple.md.
    """
    from strata.artifact_store import get_artifact_store

    state = get_state()
    store = get_artifact_store(state.config.artifact_dir)

    if store is None:
        return  # No artifact store in service mode

    try:
        # Integrity gate (#124): bounded re-read confirms the persisted blob is
        # exactly one readable IPC stream whose row total matches the plan.
        if byte_size == 0:
            readable_rows, schema_json = 0, ""
        else:

            def _read_and_validate() -> tuple[int, str]:
                with store.open_blob_reader(
                    stream_state.artifact_id, stream_state.artifact_version
                ) as blob:
                    return validate_ipc_stream_reader(blob)

            readable_rows, schema_json = await asyncio.to_thread(_read_and_validate)
        if readable_rows != row_count:
            raise ValueError(
                f"Artifact blob integrity check failed: stream yields "
                f"{readable_rows} rows, build reported {row_count}"
            )

        # The blob is already persisted; finalize_and_set_name flips state to
        # ready and records metadata + the requested name pointer atomically.
        finalized_artifact = store.finalize_and_set_name(
            artifact_id=stream_state.artifact_id,
            version=stream_state.artifact_version,
            schema_json=schema_json,
            row_count=row_count,
            byte_size=byte_size,
            name=stream_state.name,
            tenant=stream_state.tenant,
        )
        if finalized_artifact is not None:
            stream_state.artifact_id = finalized_artifact.id
            stream_state.artifact_version = finalized_artifact.version

        logger.info(
            "stream_artifact_finalized",
            artifact_id=stream_state.artifact_id,
            version=stream_state.artifact_version,
            byte_size=byte_size,
            row_count=row_count,
        )
    except Exception as e:
        logger.error(
            "stream_artifact_finalize_error",
            artifact_id=stream_state.artifact_id,
            error=str(e),
        )
        try:
            store.fail_artifact(stream_state.artifact_id, stream_state.artifact_version)
        except Exception:
            pass


def _mount_frontend(application: FastAPI) -> None:
    """Mount the frontend SPA if the dist directory exists.

    Lookup order:
    1. ``src/strata/_frontend/`` — bundled into the wheel by the
       release workflow (copy of ``frontend/dist`` made before
       ``uv build``). This is the only path that works for users
       who installed via ``pip install strata-notebook``.
    2. ``<repo_root>/frontend/dist/`` — editable / source-tree
       installs. ``__file__`` is ``<repo>/src/strata/server.py``,
       so three parents up is the repo root.
    3. ``<cwd>/frontend/dist/`` — last-resort fallback for
       deployments that drop the dist next to the working dir.
    """
    candidates = [
        Path(__file__).resolve().parent / "_frontend",
        Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
        Path.cwd() / "frontend" / "dist",
    ]
    dist_dir = None
    for c in candidates:
        if (c / "index.html").exists():
            dist_dir = c
            break

    if dist_dir is None:
        return

    # Serve static assets (JS, CSS, etc.)
    application.mount(
        "/assets",
        StaticFiles(directory=str(dist_dir / "assets")),
        name="frontend-assets",
    )

    # SPA fallback: any non-API GET returns index.html
    @application.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        # Don't intercept API or health routes
        if full_path.startswith(("v1/", "health", "docs", "openapi")):
            raise HTTPException(status_code=404)
        file_path = dist_dir / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(dist_dir / "index.html"))


_mount_frontend(app)


def _build_server_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="strata-notebook",
        description="Run the Strata notebook server.",
    )
    parser.add_argument(
        "--notebook-dir",
        default=None,
        metavar="DIR",
        help=(
            "Where new notebooks are created and discovered. Default: "
            "~/.strata/notebooks (or $STRATA_NOTEBOOK_STORAGE_DIR). Pass '.' "
            "to use the current directory."
        ),
    )
    return parser


def _apply_server_cli_overrides(args) -> None:
    """Thread CLI flags into the environment so both this process and the app's
    config load (in the lifespan) pick them up. The flag wins over an existing
    ``STRATA_NOTEBOOK_STORAGE_DIR`` — it's the more explicit signal.
    """
    import os
    from pathlib import Path

    if args.notebook_dir is not None:
        os.environ["STRATA_NOTEBOOK_STORAGE_DIR"] = str(
            Path(args.notebook_dir).expanduser().resolve()
        )


def main(argv: list[str] | None = None):
    """Run the server."""
    import uvicorn

    from strata._uv_runtime import assert_uv_managed_runtime

    args = _build_server_arg_parser().parse_args(argv)
    _apply_server_cli_overrides(args)

    assert_uv_managed_runtime()

    config = StrataConfig.load()
    # Make the notebook location obvious — it's a common surprise that new
    # notebooks land in ~/.strata/notebooks, not the current directory.
    print(f"Strata: new notebooks are created in {config.notebook_storage_dir}")
    uvicorn.run(
        "strata.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()

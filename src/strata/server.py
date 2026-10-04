"""FastAPI server for Strata."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import math
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from strata.adaptive_concurrency import AdaptiveConcurrencyController

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose

from strata.api.dependencies import authorize_table_access
from strata.artifact_uris import LATEST_VERSION, parse_artifact_uri, parse_name_uri
from strata.auth import (
    AuthError,
    get_principal,
    parse_api_key_principal,
    parse_principal,
    set_principal,
    verify_proxy_token,
)
from strata.cache import CachedFetcher
from strata.cache_warmer import CacheWarmer
from strata.config import StrataConfig
from strata.gc_tracker import install_gc_tracker
from strata.health import _package_version
from strata.json_types import JsonValue
from strata.logging import (
    configure_logging,
    get_logger,
    request_context_middleware,
)
from strata.metrics import MetricsCollector
from strata.planner import ReadPlanner
from strata.pool_metrics import get_connection_metrics, get_pool_tracker
from strata.rate_limiter import (
    RateLimitConfig,
    get_rate_limiter,
    init_rate_limiter,
)
from strata.services.build import build_service
from strata.streaming import (
    QoSAdmission,
    ScanBuildManager,
    StreamRegistry,
    StreamState,
)
from strata.tenant import (
    DEFAULT_TENANT_ID,
    clear_tenant_context,
    set_tenant_id,
    validate_tenant_id,
)
from strata.tenant_registry import get_tenant_registry, init_tenant_registry
from strata.tracing import init_tracing, instrument_fastapi
from strata.types import (
    BuildStatusResponse,
)
from strata.url_safety import host_is_allowlisted

logger = get_logger(__name__)

DRAIN_TIMEOUT_SECONDS = 30  # Max time to wait for active scans to complete

SATURATION_THRESHOLD_SECONDS = 30.0  # Fail readiness if saturated for this long


class ResourceLimitError(Exception):
    """Raised when a resource limit is exceeded."""

    pass


def _eager_warmup(config: StrataConfig) -> dict:
    """Pre-initialize expensive resources at startup and return per-step timings.

    GC tracking goes first to catch early GC events, and the Arrow memory pool is configured
    before any Arrow allocation.
    """
    warmup_times = {}
    total_start = time.perf_counter()

    # First, so it catches GC events during warmup too.
    install_gc_tracker()
    warmup_times["gc_tracker"] = True

    # Before importing pyarrow.parquet, which allocates.
    try:
        pool_name = config.configure_arrow_memory_pool()
        if pool_name:
            warmup_times["arrow_memory_pool"] = pool_name
    except ValueError as e:
        warmup_times["arrow_memory_pool_error"] = str(e)

    # Load the heavy submodules that module-level imports leave lazy.
    import_start = time.perf_counter()
    import pyarrow.parquet  # noqa: F401 - heavy, loads libparquet
    from pyiceberg.catalog.sql import SqlCatalog  # noqa: F401 - loads SQLAlchemy
    from pyiceberg.table import Table  # noqa: F401 - loads table machinery

    warmup_times["imports_ms"] = (time.perf_counter() - import_start) * 1000

    sqlite_start = time.perf_counter()
    try:
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store(config.cache_dir)
        # A cheap query warms the SQLite page cache and WAL.
        stats = store.stats()
        warmup_times["sqlite_ms"] = (time.perf_counter() - sqlite_start) * 1000
        warmup_times["sqlite_entries"] = stats.get("parquet_meta_entries", 0)
    except Exception:
        warmup_times["sqlite_ms"] = (time.perf_counter() - sqlite_start) * 1000
        warmup_times["sqlite_error"] = True

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
        import secrets
        from concurrent.futures import ThreadPoolExecutor

        from strata.transforms.signed_urls import URLSigner

        self.config = config

        # A configured secret keeps signed URLs valid across restarts and replicas;
        # unset falls back to a per-process secret (lifespan warns in service mode).
        signing_secret = (
            config.transform_signing_secret.encode("utf-8")
            if config.transform_signing_secret
            else secrets.token_bytes(32)
        )
        self.url_signer = URLSigner(signing_secret)

        # The default executor (min(32, cpu_count + 4)) queues planning under
        # high concurrency; planning reads Parquet metadata, so give it more.
        self._planning_executor = ThreadPoolExecutor(
            max_workers=64,
            thread_name_prefix="strata-planner",
        )

        # max_fetch_workers bounds total I/O concurrency.
        self._fetch_executor = ThreadPoolExecutor(
            max_workers=config.max_fetch_workers,
            thread_name_prefix="strata-fetch",
        )
        metrics_enabled = os.environ.get("STRATA_METRICS_ENABLED", "true").lower() != "false"
        self.metrics = MetricsCollector(enabled=metrics_enabled)
        self.planner = ReadPlanner(config)
        self.fetcher = CachedFetcher(config, metrics=self.metrics)

        self.scan_builds = ScanBuildManager()

        # Admission acquires the per-tenant limiters in the tenant registry;
        # QoSAdmission holds the per-scan tables, counters and fairness semaphores.
        self.qos = QoSAdmission(config)

        # Reported in metrics only; not used for admission.
        self._scan_semaphore = asyncio.Semaphore(config.max_concurrent_scans)

        self._draining = False
        self._shutdown_event = asyncio.Event()

        # When each tier became saturated (no slots available), for readiness.
        self._interactive_saturated_since: float | None = None
        self._bulk_saturated_since: float | None = None

        pool_tracker = get_pool_tracker()
        pool_tracker.register_pool("planning", self._planning_executor)
        pool_tracker.register_pool("fetch", self._fetch_executor)

        # Both initialized async in lifespan.
        self._cache_warmer: CacheWarmer | None = None
        self._adaptive_controller: AdaptiveConcurrencyController | None = None

        # on_expire runs the scan-side cleanup (prefetch discard + scan pop)
        # when a stream's TTL elapses.
        self.streams = StreamRegistry(
            config.stream_state_ttl_seconds,
            on_expire=self.scan_builds.expire_scan,
            on_claim=self._claim_stream_ownership,
            on_release=self._release_stream_ownership,
            on_drop=self._fail_unbuilt_stream,
        )

    def _fail_unbuilt_stream(self, stream_state: StreamState) -> None:
        """Fail the artifact of a stream-mode miss that expired unfetched.

        Its build starts only when the stream is fetched, so the row would stay
        ``building`` for good, holding its chain against collection.
        """
        if stream_state.background_task is None:
            self.scan_builds.mark_stream_artifact_failed(self, stream_state)

    def _ownership_store(self):
        """The stream ownership store, or None outside a multi-node setup.

        Unset ``node_advertised_url`` means single node, and nothing here reads or writes.
        """
        if not self.config.node_advertised_url:
            return None

        from strata.streaming.ownership import get_stream_ownership_store

        return get_stream_ownership_store()

    def _claim_stream_ownership(self, stream_id: str, ttl_seconds: float) -> None:
        """Advertise that this node is serving ``stream_id``.

        Best-effort: a failed claim only costs a possible redirect and must not fail the
        materialize request.
        """
        store = self._ownership_store()
        if store is None:
            return
        try:
            store.claim(stream_id, self.config.node_advertised_url or "", ttl_seconds)
        except Exception:
            logger.warning("stream_ownership_claim_failed", stream_id=stream_id, exc_info=True)

    def _release_stream_ownership(self, stream_id: str) -> None:
        """Drop the claim when the stream ends (best-effort).

        A claim that outlives its stream expires on its own; until then it redirects to a node
        that answers 404.
        """
        store = self._ownership_store()
        if store is None:
            return
        try:
            store.release(stream_id)
        except Exception:
            logger.warning("stream_ownership_release_failed", stream_id=stream_id, exc_info=True)


# Initialized in lifespan.
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


def _is_public_publication_request(request: Request) -> bool:
    """Return True for the read routes of an explicitly published artifact.

    The token in the URL is the credential, minted deliberately for one version, so these
    bypass the auth and tenant middleware: a published figure is for a reader with no account
    or tenant. Only GET on the ``/p/`` page tree and the machine-readable record;
    publishing, revoking and listing stay gated.
    """
    if request.method != "GET":
        return False
    path = request.url.path
    if path == "/v1/publications":  # the authenticated listing, not one record
        return False
    # Wikis and CMSs call ``/oembed`` to unfurl a link, unauthenticated; it
    # answers only for published tokens.
    if path == "/oembed":
        return True
    return path.startswith("/p/") or path.startswith("/v1/publications/")


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
    if not state.config.principal_auth_enabled:
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


def require_writes_enabled() -> None:
    """FastAPI dependency: 403 ``writes_disabled`` unless writes are enabled (personal mode)."""
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
    """Get the authoritative active scan count from the per-tenant admission limiters.

    Stream admission only acquires the tenant registry's limiters (single-tenant deployments
    have one, ``_default``); the global ServerState limiters would always read 0 and let
    graceful shutdown drain past live streams.
    """
    i_in_use, _, b_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
    return i_in_use + b_in_use


def _get_qos_metrics(state: ServerState) -> dict:
    """QoS tier metrics from ``state.qos``; the metrics router imports this shim."""
    return state.qos.qos_metrics()


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
    """Record when each QoS tier became saturated, for ``/health/ready``."""
    now = time.time()

    # Measured on the per-tenant limiters admission acquires. A server with no
    # live limiters yet is idle, not saturated, so require in-use slots.
    i_in_use, i_avail, b_in_use, b_avail = get_tenant_registry().aggregate_limiter_usage()

    if i_in_use > 0 and i_avail == 0:
        if state._interactive_saturated_since is None:
            state._interactive_saturated_since = now
    else:
        state._interactive_saturated_since = None

    if b_in_use > 0 and b_avail == 0:
        if state._bulk_saturated_since is None:
            state._bulk_saturated_since = now
    else:
        state._bulk_saturated_since = None


def _check_readiness(state: ServerState) -> tuple[bool, dict]:
    """Check whether the server can accept new requests; returns ``(is_ready, details)``.

    Not ready when draining, or when both QoS tiers stay saturated past the threshold.
    Dropped logs are reported but do not fail readiness.
    """
    now = time.time()
    checks = {}
    issues = []

    _update_saturation_tracking(state)

    if state._draining:
        checks["draining"] = True
        issues.append("server is draining (shutting down)")
    else:
        checks["draining"] = False

    interactive_saturated_duration = (
        now - state._interactive_saturated_since if state._interactive_saturated_since else 0.0
    )
    bulk_saturated_duration = (
        now - state._bulk_saturated_since if state._bulk_saturated_since else 0.0
    )

    checks["interactive_saturated_seconds"] = round(interactive_saturated_duration, 1)
    checks["bulk_saturated_seconds"] = round(bulk_saturated_duration, 1)

    # Fail only when both tiers are saturated; one tier with capacity still serves.
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

    # Reported, not failed on: dropped logs are a soft limit.
    dropped_logs = state.metrics.dropped_logs
    checks["dropped_logs"] = dropped_logs

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

    state.streams.shutdown_cleanups()

    for stream_state in state.streams.active_streams():
        if stream_state.background_task is not None and not stream_state.background_task.done():
            stream_state.background_task.cancel()

    state._planning_executor.shutdown(wait=False)
    state._fetch_executor.shutdown(wait=False)

    # No `ssh -L` children may outlive the server; remote workers stay up for reuse.
    from strata.notebook.routes import shutdown_worker_supervisor

    shutdown_worker_supervisor()


# None when the endpoint is disabled or the [mcp] extra is absent.
# Set at import by ``_mount_mcp_if_enabled``; read by lifespan.
_mcp_app: Starlette | None = None


def _init_configured_artifact_store(config: StrataConfig) -> None:
    """Create the artifact-store singleton using the configured blob and metadata backends.

    ``get_artifact_store`` caches on first call, so later calls get the operator's backend
    rather than the ``LocalBlobStore`` fallback. A backend that cannot be constructed is
    fatal: silently degrading to local disk hides the misconfiguration and loses every
    artifact when the pod is replaced.
    """
    from strata.artifact_store import get_artifact_store

    backend = (config.artifact_blob_backend or "local").lower()
    blob_store = None
    if backend != "local":
        blob_store = config.create_blob_store()
        logger.info("artifact_blob_backend_initialized", backend=backend)

    # None keeps SQLite under artifact_dir.
    dialect = config.create_metadata_dialect()
    if dialect is not None:
        logger.info("artifact_metadata_backend_initialized", backend=dialect.name)

    store = get_artifact_store(config.artifact_dir, blob_store=blob_store, dialect=dialect)

    # API keys share the artifact database and backend. Created here so the
    # per-request auth middleware never creates the schema.
    if config.auth_mode == "api_key" and config.artifact_dir is not None:
        from strata.api_keys import get_api_key_store

        get_api_key_store(
            config.artifact_dir / "artifacts.sqlite",
            dialect=store.dialect if store else None,
        )

    # Multi-node only. Streams stay node-local; recording the owner lets a
    # sibling redirect instead of 404ing.
    if config.node_advertised_url and config.artifact_dir is not None:
        from strata.streaming.ownership import get_stream_ownership_store

        get_stream_ownership_store(
            config.artifact_dir / "artifacts.sqlite",
            dialect=store.dialect if store else None,
        )
        logger.info("stream_ownership_enabled", node_url=config.node_advertised_url)


def _should_warn_unset_signing_secret(config: StrataConfig) -> bool:
    """Whether pull-model URLs are being signed with a throwaway per-process secret.

    Then a callback landing on another replica gets 403, and every in-flight URL dies on
    restart. Fine in personal mode (one loopback process); a hazard in service mode.
    """
    return not config.transform_signing_secret and config.deployment_mode == "service"


# Long enough to stay clear of startup work, short enough that a server
# restarted often still sweeps.
_FIRST_ARTIFACT_GC_DELAY_SECONDS = 60.0


async def _artifact_gc_loop(store, interval_seconds: float, policy: dict[str, Any]) -> None:
    """Run ``garbage_collect`` every ``interval_seconds`` until cancelled.

    The first pass runs shortly after startup, so a server restarted more often than the
    interval still sweeps. A failing pass is logged and the loop continues.
    """
    delay = min(interval_seconds, _FIRST_ARTIFACT_GC_DELAY_SECONDS)
    while True:
        await asyncio.sleep(delay)
        delay = interval_seconds
        try:
            result = await asyncio.to_thread(store.garbage_collect, **policy)
        except Exception:
            logger.exception("artifact_gc_failed")
            continue
        if result.get("deleted_count"):
            logger.info(
                "artifact_gc_collected",
                deleted_count=result["deleted_count"],
                deleted_bytes=result["deleted_bytes"],
            )


# Shared envs expire on a TTL of days, so hourly is plenty.
_SHARED_ENV_GC_INTERVAL_SECONDS = 3600.0


async def _shared_env_gc_loop(root: Path, ttl_days: float) -> None:
    """Remove unlinked shared notebook environments every hour until cancelled."""
    from strata.notebook.shared_env import collect

    while True:
        await asyncio.sleep(_SHARED_ENV_GC_INTERVAL_SECONDS)
        try:
            result = await asyncio.to_thread(collect, root, ttl_days=ttl_days)
        except Exception:
            logger.exception("shared_env_gc_failed")
            continue
        if result.removed:
            logger.info("shared_env_gc_collected", removed=", ".join(result.removed))


_NOTEBOOK_SESSION_SWEEP_SECONDS = 60.0


async def _notebook_session_sweep_loop() -> None:
    """Close idle, over-limit and memory-pressure notebook sessions every minute."""
    from strata.notebook.routes import get_session_manager

    while True:
        await asyncio.sleep(_NOTEBOOK_SESSION_SWEEP_SECONDS)
        try:
            await get_session_manager().sweep()
        except Exception:
            logger.exception("notebook_session_sweep_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize server state on startup, graceful shutdown on exit."""
    global _state

    # Tests may pre-set state before uvicorn starts; reuse its config.
    if _state is not None:
        config = _state.config
    else:
        config = StrataConfig.load()

    # Keeps personal-mode write endpoints off the network.
    config.validate_personal_mode_binding()

    from strata.transforms.registry import TransformRegistry, set_transform_registry

    transform_registry = TransformRegistry.from_config(config.transforms_config)
    set_transform_registry(transform_registry)

    if _should_warn_unset_signing_secret(config):
        logger.warning(
            "transform_signing_secret_unset",
            detail=(
                "transform_signing_secret is not set; using a random per-process "
                "secret. Signed build URLs will break on restart and differ across "
                "replicas, surfacing as 403 'Invalid or expired signature' on "
                "executor callbacks. Set STRATA_TRANSFORM_SIGNING_SECRET for a "
                "stable deployment."
            ),
        )

    # Service-mode writes are security-sensitive and still a preview; say so at boot.
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

    # Without auth headers there is no tenant (one flat namespace for every team)
    # and no principal (no attribution), and nothing looks wrong until a second
    # team joins. A warning, not an error: a single-team private store is valid.
    if config.notebook_team_cache_enabled and not config.notebook_remote_store_headers:
        logger.warning(
            "team_cache_store_unauthenticated",
            detail=(
                "notebook_team_cache_enabled is set but notebook_remote_store_headers "
                "is empty, so this notebook reaches the shared store with no identity. "
                "Results will be shared and published into the tenantless namespace "
                "with no attribution, and no isolation from any other team using the "
                "same store. Set the trusted-proxy headers the store expects."
            ),
        )

    # A warning, not an error: cells on server-managed workers never need a
    # harness user, and which workers notebooks use is unknown until they run.
    if config.deployment_mode == "service" and not config.notebook_harness_user:
        logger.warning(
            "notebook_cells_refused_on_this_host",
            detail=(
                "Service mode with no notebook_harness_user: a notebook cell that "
                "would run on this host is refused, because it could read this "
                "server's credentials. Cells assigned a server-managed worker on "
                "another machine run as normal. Set STRATA_NOTEBOOK_HARNESS_USER "
                "to run cells here as a separate OS user."
            ),
        )

    configure_logging()

    # Backs GET /v1/logs; installed server-side only so CLI and harness skip it.
    from strata.log_buffer import install_ring_buffer

    install_ring_buffer()

    # No-op if OpenTelemetry is not installed or configured.
    tracing_enabled = init_tracing()

    # So the first request is as fast as warm ones.
    warmup_times = _eager_warmup(config)

    # Tests may inject their own state.
    if _state is None:
        _state = ServerState(config)

    # Must run before anything else creates the store singleton, or callers
    # that omit ``blob_store`` pin it to ``LocalBlobStore`` and a configured
    # S3/GCS/Azure backend is silently ignored.
    if config.artifact_dir is not None:
        _init_configured_artifact_store(config)

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

    _state._cache_warmer = CacheWarmer(
        planner=_state.planner,
        fetcher=_state.fetcher,
        metrics=_state.metrics,
    )
    await _state._cache_warmer.start()

    # Configured slots must reach the per-tenant limiters admission acquires.
    # Create the default tenant's limiters now so the adaptive controller and
    # /metrics hold the same handles admission uses.
    init_tenant_registry(
        default_interactive_slots=config.interactive_slots,
        default_bulk_slots=config.bulk_slots,
        default_per_client_interactive=config.per_client_interactive,
        default_per_client_bulk=config.per_client_bulk,
    )
    default_interactive_limiter, default_bulk_limiter = (
        get_tenant_registry().get_or_create_limiters(DEFAULT_TENANT_ID)
    )

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
    # Resizes the limiters admission acquires (the _default tenant's).
    _state._adaptive_controller = AdaptiveConcurrencyController(
        config=adaptive_config,
        interactive_limiter=default_interactive_limiter,
        bulk_limiter=default_bulk_limiter,
    )
    # Attach only when enabled to keep recording off the admission hot path.
    if config.adaptive_enabled:
        _state.qos.attach_controller(_state._adaptive_controller)
    await _state._adaptive_controller.start()

    stale_removed = 0
    try:
        from strata.metadata_cache import get_metadata_store

        store = get_metadata_store(config.cache_dir)
        stale_removed = store.cleanup_stale_parquet_meta()
    except Exception:
        pass  # Don't fail startup if cleanup fails

    # Demote artifacts stuck in 'building' (no executor, or a crashed builder)
    # to failed so they surface instead of lingering forever.
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

    # One pass covers the whole store: garbage_collect treats every tenant's
    # roots as roots.
    gc_task: asyncio.Task | None = None
    if config.artifact_gc_interval_seconds and config.artifact_dir is not None:
        from strata.artifact_store import get_artifact_store as _get_store_for_gc

        gc_store = _get_store_for_gc(config.artifact_dir)
        if gc_store is not None:
            gc_task = asyncio.create_task(
                _artifact_gc_loop(
                    gc_store,
                    config.artifact_gc_interval_seconds,
                    config.artifact_gc_policy(),
                )
            )

    env_gc_task: asyncio.Task | None = None
    if config.notebook_env_backend == "shared":
        from strata.notebook.env_backend import shared_env_root

        env_gc_task = asyncio.create_task(
            _shared_env_gc_loop(shared_env_root(config), config.notebook_shared_env_ttl_days)
        )

    session_sweep_task = asyncio.create_task(_notebook_session_sweep_loop())

    build_qos = None
    if config.server_transforms_enabled:
        from strata.transforms.build_qos import BuildQoS, set_build_qos

        build_qos = BuildQoS(config.get_build_qos_config())
        set_build_qos(build_qos)

    if config.transforms_runtime_enabled:
        from strata.transforms.build_metrics import init_build_metrics

        init_build_metrics()

    # Service mode needs the explicit transforms config; personal mode always
    # runs embedded transforms so artifacts work out of the box.
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

        artifact_dir = config.artifact_dir
        if artifact_dir is None:
            artifact_dir = Path.home() / ".strata" / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        artifact_store = get_artifact_store(artifact_dir)
        # Build rows live in the artifact database, so on Postgres they must
        # land there too, not in a node-local SQLite file.
        artifact_store = get_artifact_store(artifact_dir)
        build_store = get_build_store(
            artifact_dir / "artifacts.sqlite",
            dialect=artifact_store.dialect if artifact_store else None,
        )

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

    # The parent does not start a mounted sub-app's lifespan, and the MCP
    # session manager must run for the life of the server.
    if _mcp_app is not None:
        async with _mcp_app.router.lifespan_context(_mcp_app):
            yield
    else:
        yield

    for task in (gc_task, env_gc_task, session_sweep_task):
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    from strata.transforms.build_qos import reset_build_qos

    reset_build_qos()

    # Stop only the runner this lifespan started: one left by an earlier
    # lifespan belongs to a dead event loop, and stopping it raises.
    from strata.transforms.runner import reset_build_runner

    if build_runner is not None:
        await build_runner.stop()
    reset_build_runner()

    if _state._adaptive_controller:
        await _state._adaptive_controller.stop()

    if _state._cache_warmer:
        await _state._cache_warmer.stop()

    await _graceful_shutdown(_state)

    _state.metrics.log_event("server_stopped")
    _state = None

    from strata.transforms.registry import reset_transform_registry

    reset_transform_registry()


app = FastAPI(
    title="Strata",
    description="Snapshot-aware serving layer for Iceberg tables",
    version=_package_version(),
    lifespan=lifespan,
)

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_CORS_MAX_AGE = "600"


def _configured_cors_origins() -> list[str]:
    """Extra allowed browser origins, or [] before the server is configured."""
    try:
        return list(get_state().config.cors_allow_origins)
    except RuntimeError:
        # No ServerState yet (import-time probes, some tests). Same-origin only.
        return []


def _origin_is_allowed(conn: HTTPConnection, origin: str) -> bool:
    """Whether *origin* may make cross-origin calls (or open a WebSocket) to this server.

    Same-origin is always allowed (bundled frontend); others come from ``cors_allow_origins``.
    """
    host = conn.headers.get("host")
    if host and origin in (f"http://{host}", f"https://{host}"):
        return True
    return origin in _configured_cors_origins()


@app.middleware("http")
async def cors_and_origin_guard(request: Request, call_next):
    """Answer CORS preflights and refuse cross-origin writes.

    Personal mode has no auth on loopback, so any page the user visits could otherwise drive
    the notebook API. Preflights are not enough: ``POST .../cells/{id}/execute`` has no
    body, making it a CORS simple request, so a disallowed Origin fails closed on every
    unsafe method. Requests without ``Origin`` (CLI, MCP, SDK) are untouched.
    """
    origin = request.headers.get("origin")
    if origin is None:
        return await call_next(request)

    allowed = _origin_is_allowed(request, origin)

    if request.method == "OPTIONS" and "access-control-request-method" in request.headers:
        if not allowed:
            # No CORS headers: the browser fails the preflight and never sends
            # the real request.
            return Response(status_code=403)
        requested = request.headers.get("access-control-request-headers", "")
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
                "Access-Control-Allow-Headers": requested or "content-type",
                "Access-Control-Max-Age": _CORS_MAX_AGE,
                "Vary": "Origin",
            },
        )

    if not allowed and request.method not in _SAFE_METHODS:
        return JSONResponse(
            status_code=403,
            content={"detail": f"Cross-origin request from {origin} is not allowed."},
        )

    response = await call_next(request)
    if allowed:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    return response


app.middleware("http")(request_context_middleware)


@app.middleware("http")
async def connection_tracking_middleware(request: Request, call_next):
    """Track HTTP connection metrics."""
    connection_metrics = get_connection_metrics()

    connection_header = request.headers.get("connection", "").lower()
    has_keepalive = connection_header != "close"

    connection_metrics.request_started(has_keepalive=has_keepalive)
    try:
        response = await call_next(request)
        return response
    finally:
        connection_metrics.request_completed()


@app.middleware("http")
async def frame_ancestors_middleware(request: Request, call_next):
    """Set ``Content-Security-Policy: frame-ancestors`` from ``embed_frame_ancestors``.

    Default ``'self'``; listing origins opts into cross-origin embedding, ``*`` allows any.
    A route that sets its own policy keeps it: the publication embed card allows any origin.
    Also sets ``X-Content-Type-Options: nosniff``.
    """
    response = await call_next(request)
    origins = list(getattr(_state.config, "embed_frame_ancestors", [])) if _state else []
    ancestors = "*" if "*" in origins else " ".join(["'self'", *origins])
    # A route's own policy is kept (stored bytes are sandboxed; the publication embed card
    # allows any origin); frame-ancestors is added only where that policy has none.
    existing = response.headers.get("Content-Security-Policy")
    if not existing:
        response.headers["Content-Security-Policy"] = f"frame-ancestors {ancestors}"
    elif "frame-ancestors" not in existing:
        response.headers["Content-Security-Policy"] = f"{existing}; frame-ancestors {ancestors}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _retry_after_header(seconds: float | None, fallback: float) -> str:
    """Render a wait as a ``Retry-After`` value, rounded up to whole seconds.

    Truncation would emit ``Retry-After: 0`` ("retry immediately") for every rate-limit
    rejection, since every wait is sub-second, and clients honoring the header would spin.
    ``fallback`` applies only when the caller has no computed wait.
    """
    return str(max(1, math.ceil(fallback if seconds is None else seconds)))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Apply rate limiting to incoming requests."""
    rate_limiter = get_rate_limiter()

    if rate_limiter is None:
        return await call_next(request)

    path = request.url.path
    # Never rate limited: a spike that drains the global bucket would 429 the
    # readiness probe, pull the pod from the load balancer while it is serving,
    # and amplify the overload across the fleet.
    if path in (
        "/health",
        "/health/ready",
        "/health/dependencies",
        "/ready",
        "/metrics",
        "/metrics/prometheus",
        "/v1/debug/pools",
        "/v1/debug/memory",
    ):
        return await call_next(request)

    client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip:
        client_ip = request.client.host if request.client else "unknown"

    result = rate_limiter.check(client_id=client_ip, endpoint=path)

    if not result.allowed:
        retry_after = _retry_after_header(result.retry_after_seconds, 1.0)
        return Response(
            content=f"Rate limit exceeded ({result.limit_type}). Retry after {retry_after}s.",
            status_code=429,
            headers={
                "Retry-After": retry_after,
                "X-RateLimit-Limit-Type": result.limit_type or "unknown",
            },
        )

    response = await call_next(request)

    if result.tokens_remaining is not None:
        response.headers["X-RateLimit-Remaining"] = str(int(result.tokens_remaining))

    return response


@app.middleware("http")
async def tenant_context_middleware(request: Request, call_next):
    """Set the request's tenant context from the tenant header (default ``X-Tenant-ID``).

    Under principal auth the tenant is the authenticated principal's, and an API key
    caller's header is ignored. Without multi-tenancy every request gets ``_default``.
    Missing header falls back to ``_default`` unless required (400); an invalid id is
    400, a disabled tenant 403. Health, metrics and self-authenticating routes skip this.
    """
    path = request.url.path
    if (
        path
        in (
            "/health",
            "/health/ready",
            "/health/dependencies",
            "/metrics",
            "/metrics/prometheus",
        )
        or _is_signed_data_plane_request(request)
        or _is_public_publication_request(request)
    ):
        return await call_next(request)

    state = get_state()
    config = state.config

    if not getattr(config, "multi_tenant_enabled", False):
        set_tenant_id(DEFAULT_TENANT_ID)
        try:
            response = await call_next(request)
            return response
        finally:
            clear_tenant_context()

    tenant_header = getattr(config, "tenant_header", "X-Tenant-ID")
    principal = get_principal()
    if config.principal_auth_enabled and principal is not None:
        # The authenticated principal's tenant, not the header: under api_key the
        # header is the client's own claim. Under trusted_proxy both are the proxy's.
        tenant_id = principal.tenant
    else:
        tenant_id = request.headers.get(tenant_header)

    if not tenant_id:
        if getattr(config, "require_tenant_header", False):
            return Response(
                content=f"Missing required header: {tenant_header}",
                status_code=400,
            )
        tenant_id = DEFAULT_TENANT_ID
    else:
        is_valid, error_msg = validate_tenant_id(tenant_id)
        if not is_valid:
            return Response(
                content=f"Invalid tenant ID: {error_msg}",
                status_code=400,
            )

    registry = get_tenant_registry()
    if not registry.is_tenant_enabled(tenant_id):
        return Response(
            content=f"Tenant '{tenant_id}' is not enabled",
            status_code=403,
        )

    set_tenant_id(tenant_id)
    try:
        response = await call_next(request)
        response.headers["X-Tenant-ID"] = tenant_id
        return response
    finally:
        clear_tenant_context()


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Verify the trusted proxy and set the principal context.

    Under ``auth_mode="trusted_proxy"``, checks ``X-Strata-Proxy-Token`` and parses the
    principal, tenant (configured ``tenant_header``) and scopes headers. Health, metrics and
    self-authenticating routes skip this.
    """
    state = get_state()
    config = state.config

    if config.auth_mode == "none":
        return await call_next(request)

    path = request.url.path
    if (
        path
        in (
            "/health",
            "/health/ready",
            "/health/dependencies",
            "/metrics",
            "/metrics/prometheus",
        )
        or _is_signed_data_plane_request(request)
        or _is_public_publication_request(request)
    ):
        return await call_next(request)

    # Under api_key the key is the credential and there is no proxy. Under
    # trusted_proxy the token proves the caller is the proxy; without it anyone
    # who reaches Strata could assert any principal.
    if config.auth_mode != "api_key":
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

    try:
        if config.auth_mode == "api_key":
            principal = parse_api_key_principal(dict(request.headers), config)
        else:
            principal = parse_principal(dict(request.headers), config)
        set_principal(principal)
        # list is invariant, so widen list[str] to the logger's list[JsonValue].
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
        set_principal(None)


_LOOPBACK_HOST_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def _host_name(host_header: str) -> str:
    """The name in a Host header, lowercased and without the port: ``[::1]:8765`` is ``::1``."""
    host = host_header.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    return host.rsplit(":", 1)[0]


def _host_is_allowed(host_header: str | None, config: StrataConfig) -> bool:
    """Whether this server answers to *host_header* (see ``StrataConfig.allowed_hosts``)."""
    configured = tuple(config.allowed_hosts)
    if "*" in configured or (config.deployment_mode != "personal" and not configured):
        return True
    if host_header is None:
        # Browsers always send Host, so a request without one is no rebinding page.
        return True
    name = _host_name(host_header)
    if name in _LOOPBACK_HOST_NAMES or name == config.host.lower():
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return host_is_allowlisted(name, configured)
    # Rebinding needs a DNS name the attacker controls; an IP literal is not one.
    return True


class HostAllowlistMiddleware:
    """Refuse a Host the server does not answer to, the DNS-rebinding defence.

    A rebound page is same-origin with its own name, so the origin guard alone admits it.
    Pure ASGI because ``@app.middleware("http")`` never sees WebSocket upgrades.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            host = Headers(scope=scope).get("host")
            if not _host_is_allowed(host, get_state().config):
                logger.warning("host_refused", host=host, path=scope.get("path"))
                if scope["type"] == "http":
                    refusal = PlainTextResponse(f"Host {host!r} is not allowed.", status_code=400)
                    await refusal(scope, receive, send)
                else:
                    await WebSocketClose(code=1008)(scope, receive, send)
                return
        await self.app(scope, receive, send)


# Added after every other middleware so it runs first: the origin guard trusts Host.
app.add_middleware(HostAllowlistMiddleware)

# No-op if OTel is not installed.
instrument_fastapi(app)

from strata.api.routers.admin import router as admin_router  # noqa: E402
from strata.api.routers.artifacts import router as artifacts_router  # noqa: E402
from strata.api.routers.builds import router as builds_router  # noqa: E402
from strata.api.routers.cache import router as cache_router  # noqa: E402
from strata.api.routers.debug import router as debug_router  # noqa: E402
from strata.api.routers.logs import router as logs_router  # noqa: E402
from strata.api.routers.materialize import router as materialize_router  # noqa: E402
from strata.api.routers.metadata import router as metadata_router  # noqa: E402
from strata.api.routers.metrics_health import router as metrics_health_router  # noqa: E402
from strata.api.routers.names import router as names_router  # noqa: E402
from strata.api.routers.publications import router as publications_router  # noqa: E402
from strata.api.routers.registry import router as registry_router  # noqa: E402
from strata.api.routers.streams import router as streams_router  # noqa: E402
from strata.notebook import router as notebook_router  # noqa: E402
from strata.notebook.quiesce import NotebookQuiesced  # noqa: E402
from strata.notebook.routes import projects_router as notebook_projects_router  # noqa: E402
from strata.notebook.ws import router as notebook_ws_router  # noqa: E402

app.include_router(notebook_router)
app.include_router(notebook_projects_router)


@app.exception_handler(NotebookQuiesced)
async def _notebook_quiesced(_request: Request, exc: NotebookQuiesced) -> JSONResponse:
    """Answer a write to a notebook held still for a copy with 409, not a fault.

    The writers raise ``NotebookQuiesced`` themselves, so every editing route answers the same way.
    """
    return JSONResponse(
        status_code=409, content={"detail": {"message": str(exc), "code": exc.code}}
    )


app.include_router(notebook_ws_router)
app.include_router(cache_router)
app.include_router(debug_router)
app.include_router(logs_router)
app.include_router(registry_router)
app.include_router(metadata_router)
app.include_router(metrics_health_router)
app.include_router(admin_router)
app.include_router(artifacts_router)
app.include_router(names_router)
app.include_router(publications_router)
app.include_router(builds_router)
app.include_router(materialize_router)
app.include_router(streams_router)


def _mount_mcp_if_enabled() -> None:
    """Mount the MCP endpoint at ``/mcp`` when configured (decided once, at import).

    Requires ``mcp_enabled``, a deployment that is personal or authenticates its callers
    (defense in depth over ``validate_mode_coherence``), and the ``[mcp]`` extra; without the
    extra the server still boots.
    """
    global _mcp_app

    config = _state.config if _state is not None else StrataConfig.load()
    if not config.mcp_enabled:
        return
    if config.deployment_mode != "personal" and not config.principal_auth_enabled:
        return

    from strata.notebook.mcp_server import build_mcp_app
    from strata.notebook.routes import get_session_manager

    mcp_app = build_mcp_app(get_session_manager())
    if mcp_app is None:
        logger.warning(
            "mcp_enabled_without_extra",
            detail=(
                "mcp_enabled=True but the [mcp] extra is not installed; the /mcp "
                "endpoint was not mounted. Install strata-notebook[mcp]."
            ),
        )
        return

    app.mount("/mcp", mcp_app)
    _mcp_app = mcp_app
    logger.info("mcp_endpoint_mounted", path="/mcp")


_mount_mcp_if_enabled()


def _require_notebook_worker_admin_access() -> ServerState:
    """Authorize access to the service-mode notebook worker registry."""
    state = get_state()

    if state.config.deployment_mode != "service":
        raise HTTPException(
            status_code=409,
            detail="Server-managed notebook workers are only available in service mode",
        )

    if state.config.principal_auth_enabled:
        principal = get_principal()
        if principal is None or not principal.has_scope("admin:notebook-workers"):
            raise HTTPException(status_code=403, detail="Insufficient scope")

    return state


# =============================================================================
# API v1 (stable contracts)
#
# - Response types are stable (MaterializeResponse, error format)
# - Arrow IPC streams via /v1/streams/{stream_id}
# - Error codes: 400, 404, 413 (too large), 429 (with Retry-After),
#   503 (draining/unhealthy), 504 (timeout)
# - Cache key format is versioned (CACHE_VERSION in cache.py)
# =============================================================================


# --- Artifact endpoints ---


def _get_artifact_store(
    allow_server_mode: bool = False,
    allow_read: bool = False,
    allow_write: bool = False,
):
    """Get the artifact store, raising 403 if the deployment mode does not allow this access.

    ``allow_server_mode`` also allows it when server-mode transforms are enabled.
    ``allow_read`` permits service-mode reads; the caller MUST then gate by tenant
    (``_ensure_artifact_access``) and table ACL (``_authorize_artifact_read``).
    ``allow_write`` permits service-mode writes when ``service_writes_enabled``; the caller
    MUST then gate with ``_authorize_artifact_write``.
    """
    from strata.artifact_store import get_artifact_store

    state = get_state()

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
        # A service-mode read gateway with no artifact_dir has nothing to read.
        raise HTTPException(
            status_code=404 if allow_read else 500,
            detail="Artifact store not available in this deployment",
        )
    return store


def _get_artifact_request_tenant() -> str | None:
    """Return the tenant filter for direct artifact endpoints (``None`` means unfiltered).

    Under trusted-proxy auth, scoped to the caller's tenant unless it has ``admin:*``.
    Tenantless artifacts stay visible to everyone (see ``_ensure_artifact_access``).
    """
    state = get_state()
    if not state.config.principal_auth_enabled:
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


# Alias for in-module callers and tests.
_authorize_table_access = authorize_table_access


# How far back a read follows an artifact's lineage to find the tables it came from.
_ACL_MAX_ANCESTRY_DEPTH = 100


def _authorize_artifact_read(artifact, store) -> None:
    """ACL-gate reading an artifact's bytes or metadata under trusted-proxy auth.

    The cache is shared, so a principal denied a table must not read it back through a cached
    result, nor through anything computed from one: re-checks the table ACL for each table input
    in the stored transform spec of the artifact and of every artifact in its lineage (store
    reads only, no Iceberg I/O). No-op without trusted-proxy auth; tenant scoping still applies.
    """
    state = get_state()
    if not state.config.principal_auth_enabled:
        return

    _authorize_spec_tables(getattr(artifact, "transform_spec", None))
    ancestor_specs = store.ancestor_transform_specs(artifact, max_depth=_ACL_MAX_ANCESTRY_DEPTH)
    if ancestor_specs is None:
        # Fails closed: a table beyond the bound could be one the caller is denied.
        raise HTTPException(
            status_code=403,
            detail=f"Artifact lineage is deeper than {_ACL_MAX_ANCESTRY_DEPTH} steps; "
            "its source tables cannot be authorized",
        )
    for spec_json in ancestor_specs:
        _authorize_spec_tables(spec_json)


def _authorize_spec_tables(spec_json: str | None) -> None:
    """Check the table ACL for each table input a stored transform spec names."""
    if not spec_json:
        return

    from strata.artifact_store import TransformSpec

    try:
        spec = TransformSpec.from_json(spec_json)
    except Exception:
        return  # unparseable spec → no table inputs to gate

    for input_uri in spec.inputs:
        identity = _table_identity_from_uri(input_uri)
        if identity is not None:
            _authorize_table_access(input_uri, identity)


def _table_identity_from_uri(table_uri: str):
    """Resolve a table URI to its canonical identity without planning.

    Lets the ACL run before manifest work, so a denied caller learns nothing about the table.
    ``None`` when the URI does not parse; the caller then falls back to the post-plan check.
    """
    from strata.iceberg import table_identity_for

    if "#" not in table_uri and ":" not in table_uri and "." not in table_uri:
        return None
    try:
        # The planner's helper, so an ACL rule names the table the same way
        # before and after planning.
        return table_identity_for(table_uri, get_state().config)
    except ValueError:
        return None


def _authorize_artifact_write() -> None:
    """Gate a write endpoint (put / set_name / set_alias / tags).

    Unrestricted in personal mode. In service mode, requires trusted-proxy auth and the
    ``artifacts:write`` scope; the write is stamped with the caller's tenant and principal, so
    it cannot target another tenant.
    """
    state = get_state()
    if state.config.writes_enabled:
        return  # personal mode: unrestricted

    if state.config.auth_mode != "trusted_proxy":
        # Also prevented by validate_mode_coherence.
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


def _require_registry_approver():
    """Authorize a protected-alias approval decision and return the principal.

    Under trusted-proxy auth requires ``admin:registry`` (``admin:*`` satisfies it); personal
    mode is the single operator and is allowed.
    """
    from strata.auth import get_principal

    state = get_state()
    principal = get_principal()
    if state.config.principal_auth_enabled:
        if principal is None or not principal.has_scope("admin:registry"):
            raise HTTPException(
                status_code=403,
                detail="Insufficient scope: admin:registry required to decide "
                "protected-alias changes",
            )
    return principal


_ACTIVE_BUILD_STATES = ("pending", "building", "running")


def _resolve_artifact_uri(uri: str) -> tuple[str, int] | None:
    """Resolve a Strata URI to ``(artifact_id, version)``, or None if not a Strata URI.

    Accepts ``strata://artifact/{id}@v={version}``, ``strata://artifact/{id}`` (latest) and
    ``strata://name/{name}``.
    """
    from strata.artifact_store import get_artifact_store

    state = get_state()
    if not state.config.writes_enabled:
        return None  # Artifacts only in personal mode

    store = get_artifact_store(state.config.artifact_dir)
    if store is None:
        return None

    result = parse_artifact_uri(uri)
    if result is not None:
        artifact_id, version = result
        if version == LATEST_VERSION:
            latest = store.get_latest_version(artifact_id)
            if latest is not None:
                return (artifact_id, latest.version)
            return None
        return result

    name = parse_name_uri(uri)
    if name is not None:
        artifact = store.resolve_name(name, tenant=_get_artifact_request_tenant())
        if artifact is not None:
            return (artifact.id, artifact.version)
        return None

    return None


def _mount_frontend(application: FastAPI, dist_dir: Path | None = None) -> None:
    """Mount the frontend SPA from ``dist_dir``, or the first dist directory that exists.

    Tries ``src/strata/_frontend/`` (bundled into the wheel at release), then
    ``<repo>/frontend/dist/`` (source installs), then ``<cwd>/frontend/dist/``.
    """
    if dist_dir is None:
        candidates = [
            Path(__file__).resolve().parent / "_frontend",
            Path(__file__).resolve().parent.parent.parent / "frontend" / "dist",
            Path.cwd() / "frontend" / "dist",
        ]
        dist_dir = next((c for c in candidates if (c / "index.html").exists()), None)

    if dist_dir is None:
        return

    application.mount(
        "/assets",
        StaticFiles(directory=str(dist_dir / "assets")),
        name="frontend-assets",
    )

    dist_root = dist_dir.resolve()
    index = dist_root / "index.html"

    # SPA fallback: any non-API GET returns index.html
    @application.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith(("v1/", "health", "docs", "openapi")):
            raise HTTPException(status_code=404)
        # An absolute or dot-segment path would otherwise escape the dist and read any file.
        file_path = (dist_root / full_path).resolve()
        if not (file_path.is_relative_to(dist_root) and file_path.is_file()):
            file_path = index
        # index.html names this build's hashed assets; a cached copy outlives an upgrade.
        headers = {"Cache-Control": "no-cache"} if file_path == index else None
        return FileResponse(str(file_path), headers=headers)


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
    """Export CLI flags as env vars so this process and the lifespan config load see them.

    The flag wins over an existing ``STRATA_NOTEBOOK_STORAGE_DIR``.
    """
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
    # After parsing, so --help prints from any Python; before the overrides,
    # so a refused run changes nothing.
    assert_uv_managed_runtime()
    _apply_server_cli_overrides(args)

    config = StrataConfig.load()
    # Users often expect new notebooks in the current directory.
    print(f"Strata: new notebooks are created in {config.notebook_storage_dir}")
    uvicorn.run(
        "strata.server:app",
        host=config.host,
        port=config.port,
        log_level="info",
        # The default legacy ``websockets`` protocol asserts on asyncio internals
        # that changed in CPython 3.14, killing notebook WebSockets there.
        ws="websockets-sansio",
    )


if __name__ == "__main__":
    main()

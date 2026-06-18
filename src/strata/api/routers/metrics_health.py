"""Health probes and metrics: liveness/readiness, the JSON ``/metrics`` summary,
per-table metrics, and the Prometheus exposition endpoint.

Moved verbatim from ``server.py`` (P3, router split). These handlers are thin
projections of ``ServerState`` internals, so they reach into a handful of
server-private helpers (``_get_qos_metrics``, ``_get_cache_size_bytes``,
``_get_cache_entry_count``, ``_check_readiness``, ``_get_active_scan_count``) via
a lazy ``from strata.server import ...`` inside the body — the same leaf pattern
the other routers use for ``get_state``. Those helpers stay in ``server.py``:
they read limiter/scan internals and ``_get_active_scan_count`` is also used by
the shutdown path.
"""

from __future__ import annotations

import pyarrow as pa
from fastapi import APIRouter, HTTPException, Response

from strata.cache_metrics import get_eviction_tracker
from strata.gc_tracker import get_gc_stats
from strata.health import HealthStatus, run_health_checks
from strata.pool_metrics import get_connection_metrics, get_pool_tracker
from strata.rate_limiter import get_rate_limiter
from strata.tenant_registry import get_tenant_registry

router = APIRouter(tags=["metrics"])


@router.get("/health")
async def health():
    """Basic health check endpoint (liveness probe).

    Returns 200 if the server process is running.
    Use /health/ready for readiness checks that verify dependencies.
    """
    return {"status": "ok"}


@router.get("/health/dependencies")
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
    from strata.server import get_state

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


@router.get("/health/ready")
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
    from strata.server import _check_readiness, _get_active_scan_count, _get_qos_metrics, get_state

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


@router.get("/metrics")
async def metrics():
    """Get aggregate metrics including resource utilization."""
    import asyncio
    import gc

    from strata.server import (
        _get_cache_entry_count,
        _get_cache_size_bytes,
        _get_qos_metrics,
        get_state,
    )

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


@router.get("/metrics/tables")
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
    from strata.server import get_state

    state = get_state()
    return {"tables": state.metrics.get_top_tables(limit)}


@router.get("/metrics/tables/{table_id:path}")
async def metrics_table(table_id: str):
    """Get metrics for a specific table.

    Path params:
    - table_id: The canonical table identity (e.g., "catalog.namespace.table")
    """
    from strata.server import get_state

    state = get_state()
    table_metrics = state.metrics.get_table_metrics(table_id)

    if table_metrics is None:
        raise HTTPException(status_code=404, detail=f"No metrics found for table: {table_id}")

    return table_metrics.to_dict()


@router.get("/metrics/prometheus")
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
    from strata.server import (
        _get_cache_entry_count,
        _get_cache_size_bytes,
        _get_qos_metrics,
        get_state,
    )

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

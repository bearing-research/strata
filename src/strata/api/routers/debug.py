"""Debug / diagnostics routes: latency, GC, pools, connections, memory, rate
limits, circuit breakers, and low-level cache inspection.

Moved verbatim from ``server.py`` (P3, router split). All read-only operator
diagnostics. ``inspect_cache_v1`` reaches server state through a lazy
``from strata.server import get_state`` inside the body, keeping this module a
leaf. ``/v1/config/timeouts`` and ``/v1/metadata/*`` are intentionally left in
``server.py`` — they are config/metadata domains, not debug.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from strata.gc_tracker import get_gc_stats, get_recent_gc_pauses
from strata.memory_profiler import get_detailed_memory_report, get_memory_snapshot
from strata.pool_metrics import get_connection_metrics, get_pool_tracker
from strata.rate_limiter import get_rate_limiter
from strata.slow_ops import get_latency_stats

router = APIRouter(tags=["debug"])


@router.get("/v1/debug/latency")
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


@router.get("/v1/debug/gc/pauses")
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


@router.get("/v1/debug/pools")
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


@router.get("/v1/debug/connections")
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


@router.get("/v1/debug/memory")
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


@router.get("/v1/debug/rate-limits")
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


@router.get("/v1/debug/circuit-breakers")
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


@router.get("/v1/debug/cache/inspect")
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
    from strata.server import get_state

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

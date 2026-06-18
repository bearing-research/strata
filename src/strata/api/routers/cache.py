"""Cache-plane routes: stats, eviction/histogram metrics, entries, clear, warm.

Moved verbatim from ``server.py`` (P3, router split). The handlers reach server
state through a lazy ``from strata.server import get_state`` inside the body, so
this module stays a leaf. ``/v1/debug/cache/inspect`` is intentionally *not*
here — it belongs to the future ``debug`` router.
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from strata.api.dependencies import require_scope
from strata.cache_metrics import get_eviction_tracker
from strata.cache_stats import get_cache_histogram
from strata.types import (
    Task,
    WarmAsyncRequest,
    WarmAsyncResponse,
    WarmJobProgress,
    WarmJobStatus,
    WarmRequest,
    WarmResponse,
)

router = APIRouter(tags=["cache"])


@router.get("/v1/cache/stats")
async def get_cache_stats_v1():
    """Get cache statistics.

    Returns information about what's in the cache and why.
    Operators can use this to understand cache behavior and debug issues.
    """
    from strata.cache import DiskCache
    from strata.server import get_state

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    stats = cache.get_stats()
    return stats.to_dict()


@router.get("/v1/cache/evictions")
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


@router.get("/v1/cache/histogram")
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


@router.get("/v1/cache/entries")
async def list_cache_entries_v1():
    """List all cache entries with metadata.

    Returns detailed information about each cached entry.
    """
    from strata.cache import DiskCache
    from strata.server import get_state

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    entries = cache.list_entries()
    return {"entries": [e.to_dict() for e in entries]}


@router.post("/v1/cache/clear", dependencies=[require_scope("admin:cache")])
async def clear_cache_v1():
    """Clear the disk cache.

    Requires admin:cache scope when auth_mode=trusted_proxy.
    """
    from strata.server import get_state

    state = get_state()

    try:
        state.fetcher.cache.clear()
        state.metrics.reset()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/cache/warm", response_model=WarmResponse)
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
    from strata.server import get_state

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


@router.post("/v1/cache/warm/async", response_model=WarmAsyncResponse)
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
    from strata.server import get_state

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


@router.get("/v1/cache/warm/jobs")
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
    from strata.server import get_state

    state = get_state()

    if state._cache_warmer is None:
        return {"jobs": []}

    jobs = state._cache_warmer.list_jobs(include_completed=include_completed)
    return {"jobs": [j.model_dump() for j in jobs]}


@router.get("/v1/cache/warm/jobs/{job_id}", response_model=WarmJobProgress)
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
    from strata.server import get_state

    state = get_state()

    if state._cache_warmer is None:
        raise HTTPException(status_code=404, detail="Job not found")

    progress = state._cache_warmer.get_progress(job_id)
    if progress is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return progress


@router.delete("/v1/cache/warm/jobs/{job_id}")
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
    from strata.server import get_state

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

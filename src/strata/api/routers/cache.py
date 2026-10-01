"""Cache-plane routes: stats, eviction/histogram metrics, entries, clear, warm.

Handlers reach server state through a lazy ``from strata.server import get_state`` so
this module stays a leaf.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from strata.api.dependencies import authorize_table_access, require_scope
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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["cache"])


@router.get("/v1/cache/stats")
async def get_cache_stats_v1():
    """Get disk cache statistics."""
    from strata.cache import DiskCache
    from strata.server import get_state

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    return asdict(cache.get_stats())


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
    """Get cache eviction metrics and pressure level.

    Pressure bands by evictions per minute: low < 1, medium 1-5, high 5-10, critical 10+
    (cache is thrashing). ``include_events=true`` adds recent eviction events.
    """
    tracker = get_eviction_tracker()
    result = asdict(tracker.get_stats())

    if include_events:
        result["recent_events"] = tracker.get_recent_events(limit)

    return result


@router.get("/v1/cache/histogram")
async def get_cache_histogram_v1():
    """Get cache hit/miss statistics: lifetime, 1m/5m/1h windows, and top tables.

    Counts are exact for each window and recorded per row group, not per request.
    ``covered_seconds`` is how much of the window the counts span.
    """
    histogram = get_cache_histogram()
    return histogram.get_summary()


@router.get("/v1/cache/entries", dependencies=[require_scope("admin:cache")])
async def list_cache_entries_v1():
    """List all cache entries with metadata (requires ``admin:cache``).

    Cache-wide across tenants (table identity, snapshot, projection, on-disk path), so it is
    operator introspection gated like ``/v1/cache/clear``.
    """
    from strata.cache import DiskCache
    from strata.server import get_state

    state = get_state()
    cache = state.fetcher.cache
    if not isinstance(cache, DiskCache):
        raise HTTPException(status_code=501, detail="Operation requires DiskCache")
    entries = cache.list_entries()
    return {"entries": [asdict(e) for e in entries]}


@router.post("/v1/cache/clear", dependencies=[require_scope("admin:cache")])
async def clear_cache_v1():
    """Clear the disk cache (requires ``admin:cache`` under trusted-proxy auth)."""
    from strata.server import get_state

    state = get_state()

    try:
        state.fetcher.cache.clear()
        state.metrics.reset()
        return {"status": "cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _authorize_warm_tables(table_uris: list[str]) -> None:
    """Apply the deny-first table ACL to every table a warm request names.

    Warming is a read of the table, so it takes the same gate as a scan. Runs before planning:
    reporting a planning failure for a denied table would confirm it exists.
    """
    from strata.iceberg import PyIcebergCatalog
    from strata.types import TableIdentity

    for table_uri in table_uris:
        _, table_id = PyIcebergCatalog.parse_table_uri(table_uri)
        try:
            identity = TableIdentity.from_table_id(table_id)
        except ValueError:
            # Not a well-formed ``namespace.table``: leave it to the handler's
            # own error path, which reveals nothing about a table that cannot
            # exist under this id anyway.
            continue
        authorize_table_access(table_uri, identity)


@router.post("/v1/cache/warm", response_model=WarmResponse)
async def warm_cache_v1(request: WarmRequest):
    """Warm the cache for the given tables and block until every row group is fetched.

    Row groups already cached count as skipped; failures are reported in ``errors``.
    """
    from strata.server import get_state

    state = get_state()

    # Warming reads these tables into the shared cache, so it takes the same deny-first gate as the
    # scan path, before any planning or fetching.
    _authorize_warm_tables(request.tables)

    start_time = time.perf_counter()
    tables_warmed = 0
    row_groups_cached = 0
    row_groups_skipped = 0
    bytes_written = 0
    errors: list[str] = []

    warming_semaphore = asyncio.Semaphore(request.concurrent)

    async def fetch_task(task: Task) -> tuple[str, int, str | None]:
        """Fetch one row group; returns ``(outcome, bytes_written, error)``.

        ``outcome`` is ``"cached"``, ``"skipped"`` or ``"failed"``.
        """
        async with warming_semaphore:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, state.fetcher.fetch_as_stream_bytes, task)
                if task.cached:
                    return ("skipped", 0, None)  # Already cached
                return ("cached", task.bytes_read, None)
            except Exception as exc:
                logger.exception(
                    "cache warm failed for %s row group %s",
                    task.file_path,
                    task.row_group_id,
                )
                return ("failed", 0, str(exc))

    for table_uri in request.tables:
        try:
            plan = state.planner.plan(
                table_uri=table_uri,
                snapshot_id=None,  # Current snapshot
                columns=request.columns,
                filters=[],
            )

            tasks = plan.tasks
            if request.max_row_groups is not None:
                tasks = tasks[: request.max_row_groups]

            if not tasks:
                tables_warmed += 1
                continue

            results = await asyncio.gather(
                *[fetch_task(task) for task in tasks],
                return_exceptions=True,
            )

            failures: list[str] = []
            for result in results:
                if isinstance(result, BaseException):
                    failures.append(str(result))
                    continue
                outcome, written, error = result
                if outcome == "skipped":
                    row_groups_skipped += 1
                elif outcome == "cached":
                    row_groups_cached += 1
                    bytes_written += written
                else:
                    failures.append(error or "unknown error")

            if failures:
                # Summarised, not one entry per row group: a wide table can
                # fail thousands of them and the response has to stay bounded.
                distinct = sorted(set(failures))[:3]
                errors.append(
                    f"{table_uri}: {len(failures)} row group(s) failed to warm: "
                    + "; ".join(distinct)
                )

            tables_warmed += 1

        except Exception as e:
            errors.append(f"{table_uri}: {e!s}")

    elapsed_ms = (time.perf_counter() - start_time) * 1000

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
    """Start a background cache warming job and return its ID immediately.

    Unlike ``POST /v1/cache/warm``, this does not block and can target a specific snapshot.
    Track progress via ``GET /v1/cache/warm/jobs/{id}``.
    """
    from strata.server import get_state

    state = get_state()

    # Same gate as the synchronous endpoint: a background job must not be a
    # way around the table ACL.
    _authorize_warm_tables(request.tables)

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
    """List cache warming jobs; pending and running only unless ``include_completed``."""
    from strata.server import get_state

    state = get_state()

    if state._cache_warmer is None:
        return {"jobs": []}

    jobs = state._cache_warmer.list_jobs(include_completed=include_completed)
    return {"jobs": [j.model_dump() for j in jobs]}


@router.get("/v1/cache/warm/jobs/{job_id}", response_model=WarmJobProgress)
async def get_warm_job_v1(job_id: str):
    """Get progress for a warming job."""
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
    """Cancel a warming job; already-cached data is kept."""
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

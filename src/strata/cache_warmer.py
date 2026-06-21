"""Background cache warming for Strata.

This module provides asynchronous cache warming capabilities:
- Background job execution with progress tracking
- Job queue with priority support
- Cancellation and cleanup

Usage:
    warmer = CacheWarmer(planner, fetcher, metrics)

    # Start a warming job
    job_id = await warmer.start_job(tables=["file:///warehouse#ns.table"])

    # Check progress
    progress = warmer.get_progress(job_id)

    # Cancel if needed
    warmer.cancel_job(job_id)
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.logging import get_logger
from strata.types import Task, WarmAsyncRequest, WarmJobProgress, WarmJobStatus

if TYPE_CHECKING:
    from strata.cache import CachedFetcher
    from strata.metrics import MetricsCollector
    from strata.planner import ReadPlanner

logger = get_logger(__name__)


@dataclass
class WarmingJob:
    """Internal state for a warming job."""

    job_id: str
    request: WarmAsyncRequest
    status: WarmJobStatus = WarmJobStatus.PENDING

    # Progress tracking
    tables_total: int = 0
    tables_completed: int = 0
    row_groups_total: int = 0
    row_groups_completed: int = 0
    row_groups_cached: int = 0
    row_groups_skipped: int = 0
    bytes_written: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Current state
    current_table: str | None = None
    errors: list[str] = field(default_factory=list)

    # Control
    cancelled: bool = False
    _task: asyncio.Task | None = field(default=None, repr=False)

    def to_progress(self) -> WarmJobProgress:
        """Convert to progress response."""
        now = time.time()
        if self.started_at:
            elapsed_ms = (self.completed_at or now) - self.started_at
        else:
            elapsed_ms = 0.0

        return WarmJobProgress(
            job_id=self.job_id,
            status=self.status,
            tables_total=self.tables_total,
            tables_completed=self.tables_completed,
            row_groups_total=self.row_groups_total,
            row_groups_completed=self.row_groups_completed,
            row_groups_cached=self.row_groups_cached,
            row_groups_skipped=self.row_groups_skipped,
            bytes_written=self.bytes_written,
            started_at=self.started_at,
            completed_at=self.completed_at,
            elapsed_ms=elapsed_ms * 1000,
            current_table=self.current_table,
            errors=list(self.errors),
        )


class CacheWarmer:
    """Manages background cache warming jobs.

    Features:
    - Async job execution with progress tracking
    - Concurrent job support (with limits)
    - Job cancellation
    - Automatic cleanup of completed jobs
    """

    def __init__(
        self,
        planner: "ReadPlanner",
        fetcher: "CachedFetcher",
        metrics: "MetricsCollector",
        max_concurrent_jobs: int = 3,
        job_retention_seconds: float = 3600.0,  # Keep completed jobs for 1 hour
    ):
        """Initialize the cache warmer.

        Parameters
        ----------
        planner : ReadPlanner
            Planner for table scans.
        fetcher : CachedFetcher
            Fetcher for row groups.
        metrics : MetricsCollector
            Sink for warming event logs.
        max_concurrent_jobs : int, optional
            Maximum jobs running simultaneously (default 3).
        job_retention_seconds : float, optional
            How long to keep completed job info (default 3600).
        """
        self._planner = planner
        self._fetcher = fetcher
        self._metrics = metrics
        self._max_concurrent_jobs = max_concurrent_jobs
        self._job_retention_seconds = job_retention_seconds

        # Job storage
        self._jobs: dict[str, WarmingJob] = {}
        self._lock = asyncio.Lock()

        # Concurrency control
        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        # Cleanup task
        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop and cancel all jobs."""
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Cancel all running jobs
        async with self._lock:
            for job in self._jobs.values():
                if job._task and not job._task.done():
                    job.cancelled = True
                    job._task.cancel()

    async def start_job(self, request: WarmAsyncRequest) -> str:
        """Start a new warming job and return its id.

        Parameters
        ----------
        request : WarmAsyncRequest
            Tables to warm and warming options.

        Returns
        -------
        str
            Job id for tracking progress.
        """
        job_id = str(uuid.uuid4())[:8]

        job = WarmingJob(
            job_id=job_id,
            request=request,
            tables_total=len(request.tables),
        )

        async with self._lock:
            self._jobs[job_id] = job

        # Start the job task
        job._task = asyncio.create_task(self._run_job(job))

        logger.info(
            "Warming job started",
            job_id=job_id,
            tables_count=len(request.tables),
            priority=request.priority,
        )

        return job_id

    def get_progress(self, job_id: str) -> WarmJobProgress | None:
        """Return progress for a job, or ``None`` if unknown.

        Parameters
        ----------
        job_id : str
            Job id to query.

        Returns
        -------
        WarmJobProgress or None
            Progress snapshot, or ``None`` when the job is not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.to_progress()

    def list_jobs(self, include_completed: bool = False) -> list[WarmJobProgress]:
        """List jobs, sorted by priority then start time.

        Parameters
        ----------
        include_completed : bool, optional
            Include completed/failed jobs (default ``False``).

        Returns
        -------
        list of WarmJobProgress
            Progress for each matching job.
        """
        result = []
        for job in self._jobs.values():
            if include_completed or job.status in (
                WarmJobStatus.PENDING,
                WarmJobStatus.RUNNING,
            ):
                result.append(job.to_progress())

        # Sort by priority (descending) then created time
        result.sort(key=lambda p: (-self._jobs[p.job_id].request.priority, p.started_at or 0))
        return result

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job.

        Parameters
        ----------
        job_id : str
            Job id to cancel.

        Returns
        -------
        bool
            ``True`` if cancelled; ``False`` if unknown or already finished.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False

            if job.status not in (WarmJobStatus.PENDING, WarmJobStatus.RUNNING):
                return False

            job.cancelled = True
            job.status = WarmJobStatus.CANCELLED
            job.completed_at = time.time()

            if job._task and not job._task.done():
                job._task.cancel()

        logger.info("Warming job cancelled", job_id=job_id)
        return True

    async def _run_job(self, job: WarmingJob) -> None:
        """Execute a warming job."""
        # Acquire semaphore (may wait if max jobs running)
        async with self._job_semaphore:
            if job.cancelled:
                return

            job.status = WarmJobStatus.RUNNING
            job.started_at = time.time()

            try:
                await self._execute_warming(job)

                if job.cancelled:
                    job.status = WarmJobStatus.CANCELLED
                elif job.errors:
                    job.status = WarmJobStatus.FAILED
                else:
                    job.status = WarmJobStatus.COMPLETED

            except asyncio.CancelledError:
                job.status = WarmJobStatus.CANCELLED
                raise

            except Exception as e:
                job.status = WarmJobStatus.FAILED
                job.errors.append(f"Job failed: {e!s}")
                logger.error("Warming job failed", job_id=job.job_id, error=str(e))

            finally:
                job.completed_at = time.time()
                job.current_table = None

                # Log completion
                self._metrics.log_event(
                    "cache_warm_async",
                    job_id=job.job_id,
                    status=job.status.value,
                    tables_completed=job.tables_completed,
                    row_groups_cached=job.row_groups_cached,
                    row_groups_skipped=job.row_groups_skipped,
                    bytes_written=job.bytes_written,
                    elapsed_ms=(job.completed_at - (job.started_at or job.created_at)) * 1000,
                    errors_count=len(job.errors),
                )

    async def _execute_warming(self, job: WarmingJob) -> None:
        """Execute the warming logic for a job."""
        request = job.request

        # Concurrency control for fetches within job
        fetch_semaphore = asyncio.Semaphore(request.concurrent)

        async def fetch_task(task: Task) -> tuple[bool, int]:
            """Fetch a single row group."""
            async with fetch_semaphore:
                if job.cancelled:
                    return (False, 0)

                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._fetcher.fetch_as_stream_bytes, task)
                    if task.cached:
                        return (True, 0)
                    else:
                        return (False, task.bytes_read)
                except Exception:
                    return (False, 0)

        for table_uri in request.tables:
            if job.cancelled:
                break

            job.current_table = table_uri

            try:
                # Plan the table
                plan = self._planner.plan(
                    table_uri=table_uri,
                    snapshot_id=request.snapshot_id,
                    columns=request.columns,
                    filters=[],
                )

                # Limit row groups if specified
                tasks = plan.tasks
                if request.max_row_groups is not None:
                    tasks = tasks[: request.max_row_groups]

                job.row_groups_total += len(tasks)

                if not tasks:
                    job.tables_completed += 1
                    continue

                # Fetch all row groups concurrently
                results = await asyncio.gather(
                    *[fetch_task(task) for task in tasks],
                    return_exceptions=True,
                )

                for result in results:
                    # Handle exceptions that may be raised during gather
                    if isinstance(result, BaseException) and not isinstance(result, Exception):
                        continue
                    if isinstance(result, Exception):
                        continue
                    # At this point, result is the tuple (bool, int)
                    was_cached, written = result
                    job.row_groups_completed += 1
                    if was_cached:
                        job.row_groups_skipped += 1
                    else:
                        job.row_groups_cached += 1
                        job.bytes_written += written

                job.tables_completed += 1

            except Exception as e:
                job.errors.append(f"{table_uri}: {e!s}")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old completed jobs."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                now = time.time()
                to_remove = []

                async with self._lock:
                    for job_id, job in self._jobs.items():
                        if job.completed_at is not None:
                            age = now - job.completed_at
                            if age > self._job_retention_seconds:
                                to_remove.append(job_id)

                    for job_id in to_remove:
                        del self._jobs[job_id]

                if to_remove:
                    logger.debug(
                        "Cleaned up old warming jobs",
                        removed_count=len(to_remove),
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Cleanup loop error", error=str(e))


# Global instance (initialized with server state)
_warmer: CacheWarmer | None = None


def get_cache_warmer() -> CacheWarmer | None:
    """Get the global cache warmer instance."""
    return _warmer


def set_cache_warmer(warmer: CacheWarmer) -> None:
    """Set the global cache warmer instance."""
    global _warmer
    _warmer = warmer

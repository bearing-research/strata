"""Background cache warming jobs with progress tracking, priorities and cancellation."""

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.logging import get_logger
from strata.types import TableIdentity, Task, WarmAsyncRequest, WarmJobProgress, WarmJobStatus

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

    tables_total: int = 0
    tables_completed: int = 0
    row_groups_total: int = 0
    row_groups_completed: int = 0
    row_groups_cached: int = 0
    row_groups_skipped: int = 0
    bytes_written: int = 0

    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    current_table: str | None = None
    errors: list[str] = field(default_factory=list)

    cancelled: bool = False
    # The tenant that started the job; only it (or an unscoped caller) can see or cancel it.
    tenant: str | None = None
    # Called with each table's planned identity; raises to skip a table the caller may not read.
    authorize: Callable[[str, TableIdentity], None] | None = field(default=None, repr=False)
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
    """Runs bounded concurrent cache-warming jobs in the background and expires finished ones."""

    def __init__(
        self,
        planner: "ReadPlanner",
        fetcher: "CachedFetcher",
        metrics: "MetricsCollector",
        max_concurrent_jobs: int = 3,
        job_retention_seconds: float = 3600.0,  # Keep completed jobs for 1 hour
    ):
        """Initialize the warmer.

        ``job_retention_seconds`` is how long finished jobs stay queryable.
        """
        self._planner = planner
        self._fetcher = fetcher
        self._metrics = metrics
        self._max_concurrent_jobs = max_concurrent_jobs
        self._job_retention_seconds = job_retention_seconds

        self._jobs: dict[str, WarmingJob] = {}
        self._lock = asyncio.Lock()

        self._job_semaphore = asyncio.Semaphore(max_concurrent_jobs)

        self._cleanup_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop and cancel all jobs."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            for job in self._jobs.values():
                if job._task and not job._task.done():
                    job.cancelled = True
                    job._task.cancel()

    async def start_job(
        self,
        request: WarmAsyncRequest,
        authorize: Callable[[str, TableIdentity], None] | None = None,
        tenant: str | None = None,
    ) -> str:
        """Start a new warming job for ``tenant`` and return its id.

        ``authorize`` runs on each table's planned identity before any fetch; a table it
        raises for is recorded as an error and skipped.
        """
        job_id = str(uuid.uuid4())[:8]

        job = WarmingJob(
            job_id=job_id,
            request=request,
            tables_total=len(request.tables),
            authorize=authorize,
            tenant=tenant,
        )

        async with self._lock:
            self._jobs[job_id] = job

        job._task = asyncio.create_task(self._run_job(job))

        logger.info(
            "Warming job started",
            job_id=job_id,
            tables_count=len(request.tables),
            priority=request.priority,
        )

        return job_id

    def _visible_job(self, job_id: str, tenant: str | None) -> WarmingJob | None:
        """The job if ``tenant`` may see it (``None`` sees every tenant's), else ``None``."""
        job = self._jobs.get(job_id)
        if job is None or (tenant is not None and job.tenant != tenant):
            return None
        return job

    def get_progress(self, job_id: str, tenant: str | None = None) -> WarmJobProgress | None:
        """Return a job's progress snapshot, or ``None`` if unknown or another tenant's."""
        job = self._visible_job(job_id, tenant)
        if job is None:
            return None
        return job.to_progress()

    def list_jobs(
        self, include_completed: bool = False, tenant: str | None = None
    ) -> list[WarmJobProgress]:
        """List ``tenant``'s jobs (all with ``None``) by priority then start time.

        Finished ones only with ``include_completed``.
        """
        result = []
        for job in self._jobs.values():
            if tenant is not None and job.tenant != tenant:
                continue
            if include_completed or job.status in (
                WarmJobStatus.PENDING,
                WarmJobStatus.RUNNING,
            ):
                result.append(job.to_progress())

        result.sort(key=lambda p: (-self._jobs[p.job_id].request.priority, p.started_at or 0))
        return result

    async def cancel_job(self, job_id: str, tenant: str | None = None) -> bool:
        """Cancel a pending or running job; ``False`` if unknown, another tenant's or finished."""
        async with self._lock:
            job = self._visible_job(job_id, tenant)
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

        fetch_semaphore = asyncio.Semaphore(request.concurrent)

        async def fetch_task(task: Task) -> tuple[str, int, str | None]:
            """Fetch one row group, returning ``(outcome, bytes_written, error)``.

            The named outcome keeps failed and cancelled fetches from being counted as cached.
            """
            async with fetch_semaphore:
                if job.cancelled:
                    return ("cancelled", 0, None)

                try:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._fetcher.fetch_as_stream_bytes, task)
                    if task.cached:
                        return ("skipped", 0, None)
                    return ("cached", task.bytes_read, None)
                except Exception as exc:
                    logger.exception(
                        "cache warm job %s failed for %s row group %s",
                        job.job_id,
                        task.file_path,
                        task.row_group_id,
                    )
                    return ("failed", 0, str(exc))

        for table_uri in request.tables:
            if job.cancelled:
                break

            job.current_table = table_uri

            try:
                plan = self._planner.plan(
                    table_uri=table_uri,
                    snapshot_id=request.snapshot_id,
                    columns=request.columns,
                    filters=[],
                )
                if job.authorize is not None:
                    job.authorize(table_uri, plan.table_identity)

                tasks = plan.tasks
                if request.max_row_groups is not None:
                    tasks = tasks[: request.max_row_groups]

                job.row_groups_total += len(tasks)

                if not tasks:
                    job.tables_completed += 1
                    continue

                results = await asyncio.gather(
                    *[fetch_task(task) for task in tasks],
                    return_exceptions=True,
                )

                failures: list[str] = []
                for result in results:
                    if isinstance(result, BaseException) and not isinstance(result, Exception):
                        continue
                    if isinstance(result, Exception):
                        failures.append(str(result))
                        continue
                    outcome, written, error = result
                    if outcome == "cancelled":
                        continue
                    job.row_groups_completed += 1
                    if outcome == "skipped":
                        job.row_groups_skipped += 1
                    elif outcome == "cached":
                        job.row_groups_cached += 1
                        job.bytes_written += written
                    else:
                        failures.append(error or "unknown error")

                if failures:
                    # Summarised: a wide table can fail thousands of row groups and the job record
                    # has to stay bounded.
                    distinct = sorted(set(failures))[:3]
                    job.errors.append(
                        f"{table_uri}: {len(failures)} row group(s) failed to warm: "
                        + "; ".join(distinct)
                    )

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


_warmer: CacheWarmer | None = None


def get_cache_warmer() -> CacheWarmer | None:
    """Get the global cache warmer instance."""
    return _warmer


def set_cache_warmer(warmer: CacheWarmer) -> None:
    """Set the global cache warmer instance."""
    global _warmer
    _warmer = warmer

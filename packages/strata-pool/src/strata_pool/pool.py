"""Job dispatch over a fleet of ephemeral workers.

What this slice does: accept a job, hand it to a warm worker of the right
machine type (preferring one that already served the same session), start a
machine when there is none, forward the payload over HTTP, record the result,
and meter the execution.

What it deliberately does not do yet: cool down idle workers, keep a warm
floor, reap stuck jobs, or retry preemptions. Those are the scaler and the
reaper, and they are timer loops — this repo has just spent a release fixing
control loops that shipped before anyone watched them run, so they land as
their own change with their own tests rather than as scaffolding here.

Two consequences of having no timers yet, both intentional and tested:

- A worker stays warm forever once booted. Nothing bills it down.
- If the only booting worker fails to come up, its queued jobs stay queued
  until the next submit for that machine type triggers another start.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine, Iterable
from typing import Any

import httpx

from strata_pool.backend import Backend
from strata_pool.store import PoolStore
from strata_pool.types import (
    TERMINAL_JOB_STATES,
    Job,
    JobState,
    MachineType,
    UsageEvent,
    Worker,
    WorkerState,
    new_id,
)

logger = logging.getLogger(__name__)

_HEALTH_POLL_SECONDS = 0.5


class Pool:
    """Dispatches jobs to workers provisioned by a backend."""

    def __init__(
        self,
        store: PoolStore,
        backend: Backend,
        machine_types: Iterable[MachineType],
        *,
        client: httpx.AsyncClient | None = None,
        wall: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        health_poll_seconds: float = _HEALTH_POLL_SECONDS,
    ):
        """
        Args:
            wall: Wall-clock source, for the timestamps that place a job in a
                billing period.
            monotonic: Monotonic source, for durations. Separate from `wall`
                so a clock step cannot change what a customer is charged.
        """
        self.store = store
        self.backend = backend
        self.machine_types = {mt.name: mt for mt in machine_types}
        self._client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None
        self._wall = wall
        self._monotonic = monotonic
        self._health_poll_seconds = health_poll_seconds
        self._tasks: set[asyncio.Task] = set()

    async def aclose(self) -> None:
        """Cancel in-flight work and release resources.

        The tasks are owned by the loop that created them, so they are
        cancelled here rather than left for whatever loop runs next.
        """
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    # --- submission ---

    async def submit(
        self,
        tenant_id: str,
        machine_type: str,
        payload: bytes,
        *,
        priority: int = 0,
        session_id: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Job:
        """Queue a job and try to place it. Returns as soon as it is durable.

        The caller decides whether the work is needed at all. The pool has no
        idea Strata has a cache; submitting a job whose result already exists
        boots a machine to recompute it.
        """
        if machine_type not in self.machine_types:
            raise ValueError(f"unknown machine type: {machine_type!r}")

        job = Job(
            id=new_id("job"),
            tenant_id=tenant_id,
            machine_type=machine_type,
            payload=payload,
            state=JobState.QUEUED,
            submitted_at=self._wall(),
            priority=priority,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )
        self.store.save_job(job)

        if not await self._try_dispatch(job):
            await self._ensure_capacity(machine_type)
        return job

    async def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Block until a job reaches a terminal state.

        A polling loop rather than a future, because the authoritative state
        is the row, not an in-memory handle — a job dispatched before a
        restart is still waited on correctly after one.
        """
        deadline = self._monotonic() + timeout
        while True:
            job = self.store.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state in TERMINAL_JOB_STATES:
                return job
            if self._monotonic() >= deadline:
                raise TimeoutError(f"job {job_id} did not finish within {timeout}s")
            await asyncio.sleep(0.01)

    # --- dispatch ---

    async def _try_dispatch(self, job: Job) -> bool:
        """Assign the job to a warm worker if one is available."""
        worker = None
        if job.session_id is not None:
            worker = self.store.find_warm_worker(job.machine_type, session_id=job.session_id)
        if worker is None:
            worker = self.store.find_warm_worker(job.machine_type)
        if worker is None:
            return False
        self._assign(worker, job)
        return True

    def _assign(self, worker: Worker, job: Job) -> None:
        worker.state = WorkerState.BUSY
        worker.current_job_id = job.id
        worker.session_id = job.session_id
        job.state = JobState.DISPATCHED
        job.worker_id = worker.id
        self.store.save_worker(worker)
        self.store.save_job(job)
        self._spawn(self._execute(worker, job))

    async def _drain(self, machine_type: str) -> None:
        """Place as many queued jobs as there are warm workers."""
        while True:
            job = self.store.next_queued_job(machine_type)
            if job is None:
                return
            if not await self._try_dispatch(job):
                return

    async def _ensure_capacity(self, machine_type: str) -> None:
        """Start workers for jobs the current fleet cannot absorb."""
        spec = self.machine_types[machine_type]
        queued = self.store.count_queued(machine_type)
        starting = self.store.count_workers(machine_type, [WorkerState.STARTING])
        warm = self.store.count_workers(machine_type, [WorkerState.WARM])
        total = self.store.count_workers(
            machine_type,
            [WorkerState.STARTING, WorkerState.WARM, WorkerState.BUSY],
        )
        needed = min(queued - starting - warm, spec.max_workers - total)
        for _ in range(max(needed, 0)):
            await self._start_worker(spec)

    async def _start_worker(self, spec: MachineType) -> None:
        """Provision a machine and poll it to warm in the background.

        The row is written before the backend call so a crash mid-start
        leaves evidence. It leaves a machine leaked if the crash lands after
        the provider created one — reconciling that needs a backend that can
        list its own machines, which arrives with the first real backend.
        """
        worker = Worker(
            id=new_id("worker"),
            machine_type=spec.name,
            backend=self.backend.name,
            state=WorkerState.STARTING,
            created_at=self._wall(),
        )
        self.store.save_worker(worker)

        try:
            provisioned = await self.backend.start(spec.name, spec.image, spec.env)
        except Exception:
            logger.exception(
                "backend failed to start a worker",
                extra={"machine_type": spec.name, "worker_id": worker.id},
            )
            self.store.delete_worker(worker.id)
            return

        worker.backend_id = provisioned.backend_id
        worker.endpoint = provisioned.endpoint
        worker.region = provisioned.region
        self.store.save_worker(worker)
        self._spawn(self._await_boot(worker, spec, provisioned.endpoint))

    async def _await_boot(self, worker: Worker, spec: MachineType, endpoint: str) -> None:
        deadline = self._monotonic() + spec.boot_timeout_seconds
        while self._monotonic() < deadline:
            if await self.backend.health(endpoint):
                worker.state = WorkerState.WARM
                self.store.save_worker(worker)
                logger.info(
                    "worker is warm",
                    extra={"worker_id": worker.id, "machine_type": spec.name},
                )
                await self._drain(spec.name)
                return
            await asyncio.sleep(self._health_poll_seconds)

        logger.warning(
            "worker did not boot within its timeout; stopping it",
            extra={
                "worker_id": worker.id,
                "machine_type": spec.name,
                "boot_timeout_seconds": spec.boot_timeout_seconds,
            },
        )
        await self._stop_worker(worker)

    # --- execution ---

    async def _execute(self, worker: Worker, job: Job) -> None:
        """Forward the payload to the worker and record what came back."""
        spec = self.machine_types[job.machine_type]
        timeout = (
            job.timeout_seconds if job.timeout_seconds is not None else spec.job_timeout_seconds
        )

        job.state = JobState.RUNNING
        job.started_at = self._wall()
        self.store.save_job(job)
        started_at_mono = self._monotonic()

        worker_died = False
        try:
            response = await self._client.post(
                f"{worker.endpoint}/execute",
                content=job.payload,
                timeout=timeout,
            )
        except httpx.TimeoutException:
            job.state = JobState.TIMED_OUT
            job.error = f"job exceeded its {timeout}s timeout"
        except httpx.TransportError as exc:
            # The connection failed, not the work: the machine is gone.
            job.state = JobState.FAILED
            job.error = f"worker unreachable: {exc}"
            worker_died = True
        else:
            if response.status_code >= 400:
                job.state = JobState.FAILED
                job.error = f"worker returned {response.status_code}: {response.text[:500]}"
            else:
                job.state = JobState.COMPLETED
                job.result = response.content

        duration_ms = (self._monotonic() - started_at_mono) * 1000
        job.completed_at = self._wall()
        self.store.save_job(job)
        self.store.record_usage(
            UsageEvent(
                id=new_id("usage"),
                tenant_id=job.tenant_id,
                job_id=job.id,
                machine_type=job.machine_type,
                duration_ms=duration_ms,
                started_at=job.started_at,
                completed_at=job.completed_at,
                terminal_state=job.state,
                worker_id=worker.id,
            )
        )

        if worker_died:
            await self._stop_worker(worker)
            # The queue may still hold work this worker was going to take.
            await self._ensure_capacity(job.machine_type)
            return

        worker.state = WorkerState.WARM
        worker.current_job_id = None
        worker.last_active_at = self._wall()
        self.store.save_worker(worker)
        await self._drain(job.machine_type)

    async def _stop_worker(self, worker: Worker) -> None:
        """Deallocate a machine and forget it.

        The row goes away rather than becoming a tombstone: nothing in this
        slice would ever bring it back, and jobs keep `worker_id` as plain
        history.
        """
        if worker.backend_id is not None:
            try:
                await self.backend.stop(worker.backend_id)
            except Exception:
                logger.exception(
                    "backend failed to stop a worker; it may still be billing",
                    extra={"worker_id": worker.id, "backend_id": worker.backend_id},
                )
        self.store.delete_worker(worker.id)

    # --- restart ---

    async def recover(self) -> None:
        """Reconcile persisted state with reality after a restart.

        Workers that no longer answer are dropped. A job that was in flight
        is failed rather than silently re-run: its result went to a process
        that is gone, and the caller is owed an answer, not a duplicate
        charge.

        In-flight jobs are not metered. Their monotonic start is gone with the
        process, and the only remaining source is the wall clock — inventing a
        duration from it would bill a customer for our own crash. Undercharging
        is the right direction to be wrong in.
        """
        for worker in self.store.list_workers():
            if worker.endpoint is None or not await self._health_or_false(worker.endpoint):
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.BUSY:
                worker.state = WorkerState.WARM
                worker.current_job_id = None
                self.store.save_worker(worker)

        for job in self.store.list_jobs([JobState.DISPATCHED, JobState.RUNNING]):
            job.state = JobState.FAILED
            job.error = "pool restarted while the job was in flight"
            job.completed_at = self._wall()
            self.store.save_job(job)

        for machine_type in self.machine_types:
            await self._drain(machine_type)
            await self._ensure_capacity(machine_type)

    async def _health_or_false(self, endpoint: str) -> bool:
        try:
            return await self.backend.health(endpoint)
        except Exception:
            logger.exception(
                "health check raised; treating the worker as gone",
                extra={"endpoint": endpoint},
            )
            return False

    # --- task bookkeeping ---

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

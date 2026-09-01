"""Job dispatch over a fleet of ephemeral workers.

What this slice does: accept a job, hand it to a warm worker of the right
machine type (preferring one that already served the same session), start a
machine when there is none, forward the payload over HTTP, record the result,
and meter the execution.

Idle machines are stopped by the scaler (`start_scaler`), which is the only
thing in the pool that ever ends a machine that finished its work. A
deployment that forgets to call it bills for every machine it ever started.

What it deliberately still does not do: keep a warm floor, or retry
preemptions. A floor per tenant means paying for every tenant that ever
existed, and per machine type it means choosing whose latency to subsidise;
neither has a caller. Pre-warm belongs with the proxy, which is the thing
that knows a user just opened a notebook.

One consequence of having no other timer, intentional and tested: if the
only booting worker fails to come up, its queued jobs stay queued until the
next submit for that machine type triggers another start.

A machine is retired whenever the pool cannot vouch for what is running on
it: unreachable, timed out, or holding an orphaned job across a restart.
Nothing here can cancel remote work, so reuse would mean two jobs on
hardware sized for one. That trades a cold start for a correctness
guarantee, which is the right trade until the worker protocol grows a
cancel.
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
    new_auth_token,
    new_id,
)

logger = logging.getLogger(__name__)

_HEALTH_POLL_SECONDS = 0.5

WORKER_TOKEN_ENV = "STRATA_WORKER_TOKEN"
"""Environment variable carrying the worker's own credential.

The worker contract, such as it is: reject any request to /execute that does
not present this value as a bearer token. /health stays open — it is polled
before the machine is trusted with anything, and it reveals nothing.
"""


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
        """Queue a job and place it, without waiting for it to run.

        Returns once the job is durable and any machines it needs have been
        requested — so it does wait on `backend.start()`, which a cloud API
        makes slow. Moving provisioning off the caller's path needs the counts
        it reads to stay consistent, and belongs with the first backend where
        that latency is real rather than as untested indirection now.

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
            await self._ensure_capacity(machine_type, tenant_id)
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
            worker = self.store.find_warm_worker(
                job.machine_type, job.tenant_id, session_id=job.session_id
            )
        if worker is None:
            worker = self.store.find_warm_worker(job.machine_type, job.tenant_id)
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

    async def _drain(self, machine_type: str, tenant_id: str) -> None:
        """Place this tenant's queued jobs onto its warm machines."""
        while True:
            job = self.store.next_queued_job(machine_type, tenant_id)
            if job is None:
                return
            if not await self._try_dispatch(job):
                return

    async def _ensure_capacity(self, machine_type: str, tenant_id: str) -> None:
        """Start machines for the jobs this tenant's fleet cannot absorb.

        Counted per tenant, because a machine belonging to another tenant is
        not capacity this one can use.
        """
        spec = self.machine_types[machine_type]
        queued = self.store.count_queued(machine_type, tenant_id)
        starting = self.store.count_workers(machine_type, tenant_id, [WorkerState.STARTING])
        warm = self.store.count_workers(machine_type, tenant_id, [WorkerState.WARM])
        total = self.store.count_workers(
            machine_type,
            tenant_id,
            # STOPPING is deliberately absent: a machine being torn down is
            # not capacity, and counting it would keep a tenant at its cap
            # from starting the replacement.
            [WorkerState.STARTING, WorkerState.WARM, WorkerState.BUSY],
        )
        needed = min(queued - starting - warm, spec.max_workers - total)
        for _ in range(max(needed, 0)):
            await self._start_worker(spec, tenant_id)

    async def _start_worker(self, spec: MachineType, tenant_id: str) -> None:
        """Provision a machine and poll it to warm in the background.

        The row is written before the backend call so a crash mid-start
        leaves evidence. It leaves a machine leaked if the crash lands after
        the provider created one — reconciling that needs a backend that can
        list its own machines, which arrives with the first real backend.
        """
        token = new_auth_token()
        worker = Worker(
            id=new_id("worker"),
            machine_type=spec.name,
            tenant_id=tenant_id,
            backend=self.backend.name,
            state=WorkerState.STARTING,
            created_at=self._wall(),
            auth_token=token,
        )
        self.store.save_worker(worker)

        # The token has to reach the machine before it can accept anything, so
        # it goes in the environment the backend boots it with. Minted per
        # worker: one machine's credential must not open another's.
        env = {**spec.env, WORKER_TOKEN_ENV: token}
        try:
            provisioned = await self.backend.start(spec, env)
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
        try:
            self.store.save_worker(worker)
        except Exception:
            # The machine exists but we could not write down its ID, so nothing
            # would ever be able to stop it. Stop it now, while the ID is still
            # in hand, rather than leak a billing resource.
            logger.exception(
                "could not record a machine that was just started; stopping it",
                extra={"worker_id": worker.id, "backend_id": provisioned.backend_id},
            )
            await self.backend.stop(provisioned.backend_id)
            self.store.delete_worker(worker.id)
            return
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
                await self._drain(spec.name, worker.tenant_id)
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

        keep_worker = False
        try:
            # asyncio.timeout is the wall-clock bound; httpx's own timeout is
            # per phase (its read timeout is the gap between bytes, so a worker
            # dribbling output could outlive the budget the error text claims).
            async with asyncio.timeout(timeout):
                response = await self._client.post(
                    f"{worker.endpoint}/execute",
                    content=job.payload,
                    headers={"Authorization": f"Bearer {worker.auth_token}"},
                    timeout=timeout,
                )
        except (httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # These are timeouts that never reached the worker. A machine that
            # black-holes packets looks exactly like this, and calling it a slow
            # job would hand the next one to a corpse.
            job.state = JobState.FAILED
            job.error = f"worker unreachable: {exc}"
        except (httpx.TimeoutException, TimeoutError):
            # The work is still running on the machine and there is no way to
            # tell it to stop, so the machine is not reusable: handing it the
            # next job would put two jobs on hardware sized for one.
            job.state = JobState.TIMED_OUT
            job.error = f"job exceeded its {timeout}s timeout"
        except httpx.TransportError as exc:
            job.state = JobState.FAILED
            job.error = f"worker unreachable: {exc}"
        except Exception as exc:
            # Anything else (a corrupt response body, a malformed endpoint) is
            # a machine behaving in a way we cannot reason about. Losing the
            # job is recoverable; leaving the worker BUSY forever is not — it
            # would bill indefinitely and hold a slot against max_workers.
            logger.exception(
                "unexpected failure while running a job",
                extra={"job_id": job.id, "worker_id": worker.id},
            )
            job.state = JobState.FAILED
            job.error = f"pool failed to run the job: {exc!r}"
        else:
            keep_worker = True
            if response.status_code >= 400:
                job.state = JobState.FAILED
                job.error = f"worker returned {response.status_code}: {response.text[:500]}"
            else:
                job.state = JobState.COMPLETED
                job.result = response.content

        duration_ms = (self._monotonic() - started_at_mono) * 1000
        job.completed_at = self._wall()

        try:
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
        finally:
            # Releasing the worker happens even if persistence just failed.
            # A store that rejects a write is a problem; a worker stuck BUSY
            # with nothing left to release it is a permanent one.
            if keep_worker:
                worker.state = WorkerState.WARM
                worker.current_job_id = None
                worker.last_active_at = self._wall()
                self.store.save_worker(worker)
                await self._drain(job.machine_type, job.tenant_id)
            else:
                await self._stop_worker(worker)
                # The queue may still hold work this worker was going to take.
                await self._ensure_capacity(job.machine_type, job.tenant_id)

    async def _stop_worker(self, worker: Worker) -> None:
        """Deallocate a machine and forget it.

        The row goes away rather than becoming a tombstone: nothing in this
        slice would ever bring it back, and jobs keep `worker_id` as plain
        history.

        The state flips to STOPPING first, synchronously. Everything below
        awaits, and in that window the dispatcher could otherwise find this
        machine warm and hand it a job we are about to kill.
        """
        if worker.state is not WorkerState.STOPPING:
            worker.state = WorkerState.STOPPING
            worker.current_job_id = None
            self.store.save_worker(worker)

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
                # Our process died, not the machine's: it may still be running
                # the job we are about to fail, and nothing can tell it to stop.
                # Reusing it would put the next job alongside an orphan.
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.STOPPING:
                # A previous process claimed this machine and died before the
                # backend call landed. Nothing else would ever finish the job:
                # the scaler only looks at warm machines and the dispatcher
                # cannot see this one, so it would bill forever, invisible.
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.STARTING:
                # Answering a health check is exactly the promotion criterion,
                # and no _await_boot task survived the restart to apply it.
                # Left alone this machine bills forever and takes a slot
                # against max_workers without ever accepting work.
                worker.state = WorkerState.WARM
                self.store.save_worker(worker)

        for job in self.store.list_jobs([JobState.DISPATCHED, JobState.RUNNING]):
            job.state = JobState.FAILED
            job.error = "pool restarted while the job was in flight"
            job.completed_at = self._wall()
            self.store.save_job(job)

        for machine_type in self.machine_types:
            for tenant_id in self.store.queued_tenants(machine_type):
                await self._drain(machine_type, tenant_id)
                await self._ensure_capacity(machine_type, tenant_id)

    async def _health_or_false(self, endpoint: str) -> bool:
        try:
            return await self.backend.health(endpoint)
        except Exception:
            logger.exception(
                "health check raised; treating the worker as gone",
                extra={"endpoint": endpoint},
            )
            return False

    # --- scaling down ---

    async def reap_idle_workers(self) -> int:
        """One scaler pass. Returns how many machines were stopped.

        Public and synchronous-to-call so tests drive it directly with an
        injected clock, rather than proving a cost control works by sleeping
        and hoping.
        """
        now = self._wall()
        stopped = 0
        for name, spec in self.machine_types.items():
            for candidate in self.store.list_workers(name, [WorkerState.WARM]):
                # Stopping a machine awaits the backend, and the list was read
                # before that. Re-read: a machine further down it may have
                # taken a job in the meantime, and stopping a busy machine
                # kills the job running on it.
                worker = self.store.get_worker(candidate.id)
                if worker is None or worker.state is not WorkerState.WARM:
                    continue

                # A machine that never ran a job ages from when it booted, so
                # one started for a job that then failed still gets reaped.
                last_active = worker.last_active_at or worker.created_at

                if last_active > now:
                    # The wall clock moved backwards. Left alone this machine
                    # is unreapable until the clock catches up, which is
                    # unbounded idle billing; clamping makes it age from now.
                    logger.warning(
                        "machine was last active in the future; clamping to now",
                        extra={"worker_id": worker.id, "skew_seconds": round(last_active - now, 1)},
                    )
                    worker.last_active_at = now
                    self.store.save_worker(worker)
                    continue

                idle_for = now - last_active
                if idle_for < spec.cool_down_seconds:
                    continue

                logger.info(
                    "stopping an idle machine",
                    extra={
                        "worker_id": worker.id,
                        "machine_type": name,
                        "tenant_id": worker.tenant_id,
                        "idle_seconds": round(idle_for, 1),
                    },
                )
                await self._stop_worker(worker)
                stopped += 1
        return stopped

    def start_scaler(self, interval_seconds: float = 10.0) -> None:
        """Run `reap_idle_workers` on a timer until the pool closes.

        Nothing else stops a machine that finished its work, so a pool
        without this call bills for every machine it ever started, forever.
        """
        self._spawn(self._scaler_loop(interval_seconds))

    async def _scaler_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.reap_idle_workers()
            except Exception:
                # A pass that raises must not take the loop down with it:
                # the loop dying is indistinguishable from having no scaler,
                # and that failure is measured in dollars per hour.
                logger.exception("scaler pass failed")

    # --- task bookkeeping ---

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task) -> None:
        """Drop the reference, and say so when a background task died.

        Without this the only trace of a crashed dispatch is asyncio's
        "Task exception was never retrieved" at collection time, on the root
        logger, with no job or worker to correlate it to.
        """
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("pool background task failed", exc_info=error)

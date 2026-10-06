"""Job dispatch over a fleet of ephemeral workers.

A job goes to a warm worker of its machine type (same session preferred), or a
machine is started for it. Only the scaler (`start_scaler`) stops idle machines;
without it every machine bills forever. There is no warm floor and no retry: if
the only booting worker fails, its jobs wait for the next submit of that type.

A machine is retired whenever the pool cannot vouch for what runs on it
(unreachable, timed out, orphaned job). A cancelled job's machine is told to stop
the build (`cancel`) and goes back to warm once it answers.

Several processes may share one store, each with its own `instance_id`. Every
dispatch and start is a claim in the store, and each process leases what it acts
on; a dead process's leases expire and the others take over.
"""

import asyncio
import contextlib
import logging
import os
import re
import socket
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Iterator
from typing import Any

import httpx

from strata_pool.backend import Backend
from strata_pool.store import Store
from strata_pool.types import (
    TERMINAL_JOB_STATES,
    WORKER_TOKEN_ENV,
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
_LEASE_SECONDS = 30.0
_CANCEL_TIMEOUT_SECONDS = 5.0
# The build id lands in the machine's URL path, so it must not be able to leave it.
_BUILD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")

__all__ = ["Pool", "WORKER_TOKEN_ENV"]


class Pool:
    """Dispatches jobs to workers provisioned by a backend."""

    def __init__(
        self,
        store: Store,
        backend: Backend,
        machine_types: Iterable[MachineType],
        *,
        client: httpx.AsyncClient | None = None,
        wall: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        health_poll_seconds: float = _HEALTH_POLL_SECONDS,
        max_workers_total: int | None = None,
        tracer: Any = None,
        instance_id: str | None = None,
        lease_seconds: float = _LEASE_SECONDS,
    ):
        """Create a pool over a store and a backend.

        Args:
            wall: Wall-clock source, for timestamps that place a job in a billing period.
            monotonic: Monotonic source, for durations, so a clock step cannot change a charge.
            max_workers_total: Machines across every tenant and type. None means no
                ceiling: fine for one tenant, wrong for a hosted deployment, since
                `MachineType.max_workers` caps each tenant separately.
            tracer: OpenTelemetry tracer; None uses the global one if OpenTelemetry is installed.
            instance_id: This process's name in its leases. Required for a shared
                store (raises ValueError otherwise); reusing an old name after a
                restart reclaims its leases at once.
            lease_seconds: Lease length without renewal; renewed at a third of this.
        """
        if store.shared and instance_id is None:
            raise ValueError("a pool over a shared store needs its own instance_id")
        self.store = store
        self.backend = backend
        self.machine_types = {mt.name: mt for mt in machine_types}
        # The stored catalogue's version this process last applied; None until it applies one.
        self._catalogue_version: int | None = None
        # Types dropped from the catalogue while machines of them still run,
        # kept only for the cool-down those machines retire on.
        self._retired_types: dict[str, MachineType] = {}
        self._client = client if client is not None else httpx.AsyncClient()
        self._owns_client = client is None
        self._wall = wall
        self._monotonic = monotonic
        self._health_poll_seconds = health_poll_seconds
        self.max_workers_total = max_workers_total
        self._tracer = tracer
        # Not a constant: two processes sharing a SQLite file under one name would each
        # take the other's machines and jobs as its own. A restart under a new name waits
        # out the old leases instead, which is the safe direction.
        self.instance_id = instance_id or f"{socket.gethostname()}:{os.getpid()}:{new_id('p')}"
        self.lease_seconds = lease_seconds
        self._tasks: set[asyncio.Task] = set()
        # worker id -> consecutive failed probes. In memory: recover() re-checks everything
        # on restart, and a persisted count would let one stale bad reading survive it.
        self._probe_failures: dict[str, int] = {}

    async def aclose(self) -> None:
        """Cancel in-flight background tasks and close the HTTP client if the pool owns it."""
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    # --- catalogue ---

    async def replace_machine_types(self, machine_types: Iterable[MachineType]) -> None:
        """Swap the catalogue without restarting.

        A removed type's queued jobs fail; its machines finish their work and
        retire once idle. Machines on a changed type's old image take no new
        jobs and retire the same way.
        """
        updated = {mt.name: mt for mt in machine_types}
        for name, spec in self.machine_types.items():
            if name not in updated:
                self._retired_types[name] = spec
        for name in updated:
            self._retired_types.pop(name, None)
        self.machine_types = updated
        self._fail_jobs_without_a_type()
        # Stale machines are not capacity, so work queued behind them needs
        # machines on the current image.
        for machine_type in self.machine_types:
            for tenant_id in self.store.queued_tenants(machine_type):
                await self._drain(machine_type, tenant_id)
                await self._ensure_capacity(machine_type, tenant_id)

    async def sync_catalogue(self) -> bool:
        """Apply the stored catalogue if another write set it since this process last looked.

        Once a catalogue is stored it replaces the one the pool was built with.
        Every submit and scaler pass calls this, so a change made through any
        process sharing the store reaches all of them. Returns whether it applied one.
        """
        version = self.store.catalogue_version()
        if version is None or version == self._catalogue_version:
            return False
        machine_types = self.store.load_machine_types()
        if machine_types is None:
            return False
        # Set before the first await, so a concurrent caller does not apply it twice.
        self._catalogue_version = version
        await self.replace_machine_types(machine_types)
        return True

    def _fail_jobs_without_a_type(self) -> None:
        """Queued work for a type the catalogue no longer names can never run."""
        for machine_type in self.store.queued_machine_types():
            if machine_type in self.machine_types:
                continue
            for job in self.store.list_jobs([JobState.QUEUED]):
                if job.machine_type != machine_type:
                    continue
                now = self._wall()
                self.store.fail_job(
                    job.id,
                    f"machine type {machine_type!r} was removed from the catalogue",
                    now,
                    states=[JobState.QUEUED],
                    owner=self.instance_id,
                    now=now,
                )

    def _is_stale(self, worker: Worker) -> bool:
        spec = self.machine_types.get(worker.machine_type)
        return spec is None or (worker.image is not None and worker.image != spec.image)

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
        trace_context: dict[str, str] | None = None,
    ) -> Job:
        """Queue a job and place it, without waiting for it to run.

        Does wait on `backend.start()` for any machine the job needs, which can
        be slow. The pool knows nothing of Strata's cache: every submit runs.
        Raises ValueError for an unknown machine type.
        """
        await self.sync_catalogue()
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
            trace_context=dict(trace_context or {}),
        )
        self.store.save_job(job)

        if not await self._try_dispatch(job):
            await self._ensure_capacity(machine_type, tenant_id)
        return job

    async def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Block until a job reaches a terminal state, polling the store.

        Raises KeyError for an unknown job and TimeoutError after ``timeout`` seconds.
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

    async def cancel(self, job_id: str, build_id: str) -> Job:
        """Cancel a job that has not finished; return it as it now stands.

        A job not yet running never reaches a machine. A running job's machine is
        asked to stop *build_id* (the payload is opaque, so the caller names the
        build) and goes back to warm once its execute call answers; the process
        running the job meters the time it ran. A job that already finished is
        returned unchanged. Raises ValueError for a malformed build id and
        KeyError for an unknown job.
        """
        if not _BUILD_ID.fullmatch(build_id):
            raise ValueError(f"invalid build id: {build_id!r}")
        cancelled = self.store.cancel_job(job_id, "cancelled", self._wall())
        job = self.store.get_job(job_id)
        if job is None:
            raise KeyError(job_id)
        # started_at is written with ``running``, so without it nothing reached the machine.
        if cancelled and job.started_at is not None:
            await self._cancel_on_machine(job, build_id)
        return job

    async def _cancel_on_machine(self, job: Job, build_id: str) -> None:
        """Ask the machine running *job* to stop *build_id*.

        Best effort: a machine that never stops is bounded by the job's timeout,
        which retires it.
        """
        worker = self.store.get_worker(job.worker_id) if job.worker_id is not None else None
        # A machine already released has nothing of this job's left to stop.
        if worker is None or worker.endpoint is None or worker.current_job_id != job.id:
            return
        extra = {"job_id": job.id, "worker_id": worker.id, "build_id": build_id}
        try:
            response = await self._client.post(
                f"{worker.endpoint}/v1/executions/{build_id}/cancel",
                headers={"Authorization": f"Bearer {worker.auth_token}"},
                timeout=_CANCEL_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "could not reach a machine to cancel its job", extra={**extra, "error": str(exc)}
            )
            return
        if response.status_code >= 400:
            logger.warning(
                "a machine refused a cancel", extra={**extra, "status_code": response.status_code}
            )

    # --- dispatch ---

    async def _try_dispatch(self, job: Job) -> bool:
        """Assign the job to a warm worker if one is available.

        True also when the job turns out to be placed already: another pool
        process sharing the store claimed it first.
        """
        spec = self.machine_types.get(job.machine_type)
        if spec is None:
            return False
        while True:
            worker = None
            if job.session_id is not None:
                worker = self.store.find_warm_worker(
                    job.machine_type, job.tenant_id, session_id=job.session_id, image=spec.image
                )
            if worker is None:
                worker = self.store.find_warm_worker(
                    job.machine_type, job.tenant_id, image=spec.image
                )
            if worker is None:
                return False
            if self._assign(worker, job):
                return True
            # Lost the race for this machine or this job; which one decides what is left to do.
            latest = self.store.get_job(job.id)
            if latest is None or latest.state is not JobState.QUEUED:
                return True

    def _assign(self, worker: Worker, job: Job) -> bool:
        now = self._wall()
        expires = now + self.lease_seconds
        if not self.store.claim_dispatch(worker, job, self.instance_id, expires):
            return False
        self._record_span(
            "pool.queue",
            job.trace_context,
            job.submitted_at,
            now,
            {
                "job_id": job.id,
                "machine_type": job.machine_type,
                "tenant_id": job.tenant_id,
                "worker_id": worker.id,
            },
        )
        worker.state = WorkerState.BUSY
        worker.current_job_id = job.id
        worker.session_id = job.session_id
        worker.lease_owner = job.lease_owner = self.instance_id
        worker.lease_expires_at = job.lease_expires_at = expires
        job.state = JobState.DISPATCHED
        job.worker_id = worker.id
        self._spawn(self._execute(worker, job))
        return True

    async def _drain(self, machine_type: str, tenant_id: str) -> None:
        """Place this tenant's queued jobs onto its warm machines."""
        while True:
            job = self.store.next_queued_job(machine_type, tenant_id)
            if job is None:
                return
            if not await self._try_dispatch(job):
                return

    async def _ensure_capacity(self, machine_type: str, tenant_id: str) -> None:
        """Start machines for the jobs this tenant's own fleet cannot absorb."""
        spec = self.machine_types.get(machine_type)
        if spec is None:
            return
        queued = self.store.count_queued(machine_type, tenant_id)

        # One machine per reservation, each checked against the store as it is then: another
        # tenant or process can start one while the backend call awaits. Bounded by the queue,
        # so a backend that fails every start cannot spin here.
        for _ in range(max(queued, 0)):
            worker = Worker(
                id=new_id("worker"),
                machine_type=spec.name,
                tenant_id=tenant_id,
                backend=self.backend.name,
                state=WorkerState.STARTING,
                created_at=self._wall(),
                auth_token=new_auth_token(),
                image=spec.image,
                lease_owner=self.instance_id,
                lease_expires_at=self._wall() + self.lease_seconds,
            )
            outcome = self.store.reserve_worker(
                worker, max_workers=spec.max_workers, max_workers_total=self.max_workers_total
            )
            if outcome == "fleet_cap":
                # Log it: a fleet at its ceiling otherwise looks like a slow queue.
                logger.warning(
                    "fleet is at its global cap; jobs will wait",
                    extra={
                        "machine_type": machine_type,
                        "tenant_id": tenant_id,
                        "max_workers_total": self.max_workers_total,
                        "waiting": queued,
                    },
                )
                return
            if outcome != "reserved":
                return
            # The boot is traced under the job first in line, the one it most likely serves.
            await self._start_worker(
                spec, worker, self.store.next_queued_job(machine_type, tenant_id)
            )

    async def _offer_freed_capacity(self) -> None:
        """Hand freed headroom to any tenant waiting on it.

        A tenant deferred by the fleet cap is otherwise retried only when it submits again.
        """
        for machine_type in self.machine_types:
            for tenant_id in self.store.queued_tenants(machine_type):
                await self._ensure_capacity(machine_type, tenant_id)

    async def _start_worker(self, spec: MachineType, worker: Worker, waiting: Job | None) -> None:
        """Provision the machine *worker* reserved, then poll it to warm in the background.

        The boot's span joins *waiting*'s trace.

        The row exists before the backend call, but a crash after the provider
        creates the machine still leaks it: backends cannot list their machines.
        """
        # The token must reach the machine before it accepts anything, so it goes in the
        # boot environment. Minted per worker: one machine's credential must not open another's.
        assert worker.auth_token is not None
        env = {**spec.env, WORKER_TOKEN_ENV: worker.auth_token}
        try:
            async with self._holding(worker_id=worker.id):
                provisioned = await self.backend.start(spec, env)
        except Exception:
            logger.exception(
                "backend failed to start a worker",
                extra={"machine_type": spec.name, "worker_id": worker.id},
            )
            self.store.delete_worker(worker.id)
            self._boot_span(worker, waiting, error="the backend failed to start the machine")
            return

        worker.backend_id = provisioned.backend_id
        worker.endpoint = provisioned.endpoint
        worker.region = provisioned.region
        try:
            recorded = self.store.record_provisioned(worker, self.instance_id, self._wall())
        except Exception:
            # The machine exists but its ID was not recorded, so nothing could ever stop it.
            # Stop it now rather than leak a billing resource.
            logger.exception(
                "could not record a machine that was just started; stopping it",
                extra={"worker_id": worker.id, "backend_id": provisioned.backend_id},
            )
            await self.backend.stop(provisioned.backend_id)
            self.store.delete_worker(worker.id)
            return
        if not recorded:
            # The start outlasted the lease and another process took the row, with no machine
            # to stop; this one stops its own.
            logger.warning(
                "lost a starting machine's row to another pool process; stopping it",
                extra={"worker_id": worker.id, "backend_id": provisioned.backend_id},
            )
            await self.backend.stop(provisioned.backend_id)
            return
        self._spawn(self._await_boot(worker, spec, provisioned.endpoint, waiting))

    async def _await_boot(
        self, worker: Worker, spec: MachineType, endpoint: str, waiting: Job | None
    ) -> None:
        deadline = self._monotonic() + spec.boot_timeout_seconds
        healthy = warmed = False
        async with self._holding(worker_id=worker.id):
            while self._monotonic() < deadline:
                if await self.backend.health(endpoint):
                    healthy = True
                    warmed = self.store.mark_warm(worker, self.instance_id, self._wall())
                    break
                await asyncio.sleep(self._health_poll_seconds)
        if healthy:
            if not warmed:
                # Boot outlasted the lease and another process is stopping it.
                logger.warning(
                    "another pool process took over a machine while it booted",
                    extra={"worker_id": worker.id, "machine_type": spec.name},
                )
                return
            worker.state = WorkerState.WARM
            worker.lease_owner = None
            worker.lease_expires_at = None
            self._boot_span(worker, waiting)
            logger.info(
                "worker is warm",
                extra={"worker_id": worker.id, "machine_type": spec.name},
            )
            await self._drain(spec.name, worker.tenant_id)
            return

        logger.warning(
            "worker did not boot within its timeout; stopping it",
            extra={
                "worker_id": worker.id,
                "machine_type": spec.name,
                "boot_timeout_seconds": spec.boot_timeout_seconds,
            },
        )
        self._boot_span(worker, waiting, error=f"did not boot within {spec.boot_timeout_seconds}s")
        await self._stop_worker(worker)

    # --- execution ---

    async def _execute(self, worker: Worker, job: Job) -> None:
        """Forward the payload to the worker and record what came back."""
        spec = self.machine_types[job.machine_type]
        # A job may shorten its type's timeout, never extend it past what the operator set.
        timeout = (
            min(job.timeout_seconds, spec.job_timeout_seconds)
            if job.timeout_seconds is not None
            else spec.job_timeout_seconds
        )

        # Another process may have reclaimed this job after this one stalled past its lease,
        # or the job was cancelled. Writing ``running`` over its terminal row would leave it
        # running forever: nothing reclaims a row with no lease owner.
        job.started_at = self._wall()
        if not self.store.start_job(job):
            if self.store.settle_cancelled_job(job.id, self.instance_id):
                # Cancelled before it reached the machine: nothing ran, so nothing is metered.
                await self._return_to_warm(worker, job)
                return
            current = self.store.get_job(job.id)
            logger.info(
                "job is no longer ours to run",
                extra={"job_id": job.id, "state": None if current is None else current.state.value},
            )
            return
        job.state = JobState.RUNNING
        started_at_mono = self._monotonic()

        keep_worker = False
        try:
            # asyncio.timeout bounds wall-clock time; httpx's read timeout is per gap between
            # bytes, so a worker dribbling output could outlive the budget.
            with self._execution_span(job, worker) as trace_headers:
                async with (
                    self._holding(job_id=job.id, worker_id=worker.id),
                    asyncio.timeout(timeout),
                ):
                    response = await self._client.post(
                        f"{worker.endpoint}/execute",
                        content=job.payload,
                        headers={
                            **trace_headers,
                            "Authorization": f"Bearer {worker.auth_token}",
                        },
                        timeout=timeout,
                    )
        except (httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            # Timeouts that never reached the worker. A machine black-holing packets looks like
            # this, and treating it as a slow job would hand the next job to a dead machine.
            job.state = JobState.FAILED
            job.error = f"worker unreachable: {exc}"
        except (httpx.TimeoutException, TimeoutError):
            # The work still runs on the machine and cannot be stopped, so the machine is not
            # reusable: the next job would share hardware sized for one.
            job.state = JobState.TIMED_OUT
            job.error = f"job exceeded its {timeout}s timeout"
        except httpx.TransportError as exc:
            job.state = JobState.FAILED
            job.error = f"worker unreachable: {exc}"
        except Exception as exc:
            # Anything else (a corrupt body, a malformed endpoint) is a machine we cannot reason
            # about. Losing the job is recoverable; a worker left BUSY bills forever and holds a
            # max_workers slot.
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

        taken_over = False
        try:
            if not self.store.finish_job(job, self.instance_id):
                if self.store.settle_cancelled_job(job.id, self.instance_id):
                    # The cancel stands over whatever the machine answered; the time still bills.
                    job.state = JobState.CANCELLED
                else:
                    # This process missed renewals for longer than a lease, and another one
                    # failed the job and is stopping the machine. Its answer stands; ours would
                    # bill twice.
                    taken_over = True
                    logger.warning(
                        "another pool process took over a job before it finished",
                        extra={"job_id": job.id, "worker_id": worker.id},
                    )
                    return
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
            # Release the worker even if persistence just failed: a worker stuck BUSY with
            # nothing to release it is permanent. A machine another process took over is that
            # process's to stop.
            if not taken_over:
                if keep_worker:
                    await self._return_to_warm(worker, job)
                else:
                    await self._stop_worker(worker)
                    # The freed slot belongs to whoever is waiting, not only to this job's tenant.
                    await self._offer_freed_capacity()

    async def _return_to_warm(self, worker: Worker, job: Job) -> None:
        """Hand a machine that is done with *job* back to the warm fleet, then refill it."""
        worker.last_active_at = self._wall()
        if self.store.release_worker(worker, self.instance_id):
            worker.state = WorkerState.WARM
            worker.current_job_id = None
            worker.lease_owner = None
            worker.lease_expires_at = None
        await self._drain(job.machine_type, job.tenant_id)

    @contextlib.contextmanager
    def _execution_span(self, job: Job, worker: Worker) -> Iterator[dict[str, str]]:
        """Span the job's time on the machine; yield headers that make its work a child span.

        Without OpenTelemetry the submitter's trace context is forwarded unchanged.
        """
        try:
            from opentelemetry import trace
            from opentelemetry.propagate import extract, inject
        except ImportError:
            yield dict(job.trace_context)
            return
        tracer = self._tracer or trace.get_tracer("strata_pool")
        with tracer.start_as_current_span(
            "pool.execute",
            context=extract(job.trace_context),
            attributes={
                "job_id": job.id,
                "machine_type": job.machine_type,
                "tenant_id": job.tenant_id,
                "worker_id": worker.id,
            },
        ):
            carrier: dict[str, str] = {}
            inject(carrier)
            yield carrier

    def _boot_span(self, worker: Worker, waiting: Job | None, *, error: str | None = None) -> None:
        attributes = {
            "worker_id": worker.id,
            "machine_type": worker.machine_type,
            "tenant_id": worker.tenant_id,
            "backend": worker.backend,
        }
        if waiting is not None:
            attributes["job_id"] = waiting.id
        self._record_span(
            "pool.boot",
            waiting.trace_context if waiting is not None else {},
            worker.created_at,
            self._wall(),
            attributes,
            error=error,
        )

    def _record_span(
        self,
        name: str,
        trace_context: dict[str, str],
        start: float,
        end: float,
        attributes: dict[str, str],
        *,
        error: str | None = None,
    ) -> None:
        """Record a finished span from wall-clock times, under the submitter's trace.

        After the fact, because its start may have been written by another process.
        """
        try:
            from opentelemetry import trace
            from opentelemetry.propagate import extract
        except ImportError:
            return
        tracer = self._tracer or trace.get_tracer("strata_pool")
        span = tracer.start_span(
            name,
            context=extract(trace_context),
            attributes=attributes,
            start_time=int(start * 1e9),
        )
        if error is not None:
            span.set_status(trace.Status(trace.StatusCode.ERROR, error))
        span.end(end_time=int(end * 1e9))

    async def _stop_worker(self, worker: Worker) -> bool:
        """Deallocate a machine and delete its row; return whether this process stopped it.

        The machine is claimed as STOPPING before any await, so the dispatcher
        cannot hand it a job in the meantime.
        """
        # Every path that ends a machine comes through here, so drop the probe counter here.
        # A machine reaped or stopped after one missed probe is never WARM again, so the
        # probe loop would never pop its entry.
        self._probe_failures.pop(worker.id, None)

        # A claim, not a write: of two processes stopping the same machine, or one stopping
        # it while another hands it a job, one wins. A machine another live process holds is
        # left to it.
        now = self._wall()
        if not self.store.claim_for_stop(
            worker.id, self.instance_id, now, now + self.lease_seconds
        ):
            return False
        worker.state = WorkerState.STOPPING
        worker.current_job_id = None
        worker.lease_owner = self.instance_id

        if worker.backend_id is not None:
            # No lease renewal around the stop: a renewal task would add a suspension between the
            # machine stopping and its row going, exposing a stopped machine still listed. The
            # lease bounds it instead, so a hung provider cannot let another process take the row
            # mid-call.
            try:
                async with asyncio.timeout(self.lease_seconds / 2):
                    await self.backend.stop(worker.backend_id)
            except Exception:
                logger.exception(
                    "backend failed to stop a worker; its row is kept so the stop is tried again",
                    extra={"worker_id": worker.id, "backend_id": worker.backend_id},
                )
                # The row stays in ``stopping``, holding the machine's place against the fleet cap:
                # deleting it would leave a machine billing with nothing naming it.
                return False
        self.store.delete_worker(worker.id)
        return True

    # --- restart ---

    async def recover(self) -> None:
        """Reconcile persisted state with reality after a restart.

        Unreachable workers are dropped. In-flight jobs fail rather than re-run,
        and are not metered: their monotonic start died with the process. Over a
        shared store, only this instance's rows and expired leases are touched.
        """
        now = self._wall()
        for worker in self.store.list_workers():
            if worker.endpoint is None or not await self._health_or_false(worker.endpoint):
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.BUSY:
                # Our process died, not the machine: it may still run the job we are about to fail,
                # and nothing can stop it. Reusing it would put the next job beside an orphan.
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.STOPPING:
                # A previous process claimed this machine and died before the backend call landed.
                # Nothing else would finish it: the scaler only sees warm machines and the
                # dispatcher cannot see this one, so it would bill forever, invisible.
                await self._stop_worker(worker)
                continue
            if worker.state is WorkerState.STARTING:
                # A passing health check is the promotion criterion, and no _await_boot task
                # survived the restart to apply it. Left alone, the machine bills and holds a
                # max_workers slot without ever accepting work.
                self.store.mark_warm(worker, self.instance_id, now)

        in_flight = [JobState.DISPATCHED, JobState.RUNNING]
        for job in self.store.list_jobs(in_flight):
            self.store.fail_job(
                job.id,
                "pool restarted while the job was in flight",
                self._wall(),
                states=in_flight,
                owner=self.instance_id,
                now=now,
            )

        self._fail_jobs_without_a_type()
        for machine_type in self.machine_types:
            for tenant_id in self.store.queued_tenants(machine_type):
                await self._drain(machine_type, tenant_id)
                await self._ensure_capacity(machine_type, tenant_id)

    async def reclaim_expired_leases(self) -> int:
        """Fail the in-flight jobs and stop the machines of a process whose leases expired.

        Returns how many machines were stopped. Run by the scaler.
        """
        now = self._wall()
        in_flight = [JobState.DISPATCHED, JobState.RUNNING]
        # Other processes' rows only: this one's are live here, and a warm machine has no
        # lease. The store decides lease expiry in the same statement that acts on it.
        for job in self.store.list_jobs(in_flight):
            if job.lease_owner is None or job.lease_owner == self.instance_id:
                continue
            if self.store.fail_job(
                job.id,
                f"pool instance {job.lease_owner!r} stopped renewing its lease "
                "while the job was in flight",
                now,
                states=in_flight,
                owner=self.instance_id,
                now=now,
            ):
                logger.warning(
                    "failed a job abandoned by another pool process",
                    extra={"job_id": job.id, "lease_owner": job.lease_owner},
                )

        stopped = 0
        for worker in self.store.list_workers():
            if worker.lease_owner is None or worker.lease_owner == self.instance_id:
                continue
            if await self._stop_worker(worker):
                logger.warning(
                    "stopped a machine abandoned by another pool process",
                    extra={"worker_id": worker.id, "lease_owner": worker.lease_owner},
                )
                stopped += 1
        if stopped:
            await self._offer_freed_capacity()
        return stopped

    @contextlib.asynccontextmanager
    async def _holding(
        self, *, job_id: str | None = None, worker_id: str | None = None
    ) -> AsyncIterator[None]:
        """Renew this process's leases on a job and a machine while the body runs."""

        async def renew() -> None:
            while True:
                await asyncio.sleep(self.lease_seconds / 3)
                try:
                    self.store.renew_lease(
                        self.instance_id,
                        self._wall() + self.lease_seconds,
                        job_id=job_id,
                        worker_id=worker_id,
                    )
                except Exception:
                    logger.exception(
                        "could not renew a lease",
                        extra={"job_id": job_id, "worker_id": worker_id},
                    )

        renewal = asyncio.create_task(renew())
        try:
            yield
        finally:
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await renewal

    async def _health_or_false(self, endpoint: str) -> bool:
        try:
            return await self.backend.health(endpoint)
        except Exception:
            logger.exception(
                "health check raised; treating the worker as gone",
                extra={"endpoint": endpoint},
            )
            return False

    async def probe_warm_workers(self) -> int:
        """Health-check warm machines and retire the dead ones; return how many were stopped.

        Busy machines are skipped: a probe slowed by a long job must not retire it.
        """
        # Carry the endpoint rather than re-read it: a comprehension filter would not narrow
        # the attribute to str for the use below.
        candidates: list[tuple[MachineType, Worker, str]] = []
        for name, spec in self.machine_types.items():
            if spec.health_check_failures <= 0:
                continue
            for candidate in self.store.list_workers(name, [WorkerState.WARM]):
                endpoint = candidate.endpoint
                if endpoint is None:
                    continue
                candidates.append((spec, candidate, endpoint))
        if not candidates:
            return 0

        # Concurrent because this shares the scaler pass with reap_idle_workers. Each health
        # check has its own timeout (10s for RunPod), so a serial pass against a provider
        # black-holing packets would stall reaping, the only thing that stops billing, for
        # minutes.
        results = await asyncio.gather(
            *(self._health_or_false(endpoint) for _, _, endpoint in candidates)
        )

        if len(candidates) > 1 and not any(results):
            # Every machine failed at once: far more likely this process's own network (DNS, a
            # proxy, an exhausted connection pool) than a fleet-wide death, so retire nothing.
            # Counts are kept, so a real outage is caught by the first pass that sees anything
            # healthy, and a dead machine still fails the job dispatched to it. A single
            # candidate is no evidence either way, so it is acted on.
            logger.warning(
                "every warm machine failed its probe; treating it as a "
                "pool-side fault rather than retiring the fleet",
                extra={"candidates": len(candidates)},
            )
            return 0

        stopped = 0
        for (spec, candidate, _endpoint), healthy in zip(candidates, results, strict=True):
            # Re-read after the await, as reap_idle_workers does: the machine may have taken a
            # job during the probe, and stopping a busy machine kills that job.
            worker = self.store.get_worker(candidate.id)
            if worker is None or worker.state is not WorkerState.WARM:
                self._probe_failures.pop(candidate.id, None)
                continue

            if healthy:
                self._probe_failures.pop(worker.id, None)
                continue

            failures = self._probe_failures.get(worker.id, 0) + 1
            self._probe_failures[worker.id] = failures
            if failures < spec.health_check_failures:
                logger.info(
                    "a warm machine missed a health probe",
                    extra={
                        "worker_id": worker.id,
                        "machine_type": spec.name,
                        "consecutive_failures": failures,
                        "retire_at": spec.health_check_failures,
                    },
                )
                continue

            logger.warning(
                "retiring a warm machine that stopped answering",
                extra={
                    "worker_id": worker.id,
                    "machine_type": spec.name,
                    "consecutive_failures": failures,
                },
            )
            self._probe_failures.pop(worker.id, None)
            await self._stop_worker(worker)
            stopped += 1

        if stopped:
            # The freed slot belongs to whoever is waiting, not only the dead machine's tenant.
            await self._offer_freed_capacity()
        return stopped

    # --- scaling down ---

    async def reap_idle_workers(self) -> int:
        """Stop machines idle past their cool-down and retry failed stops; return the count."""
        now = self._wall()
        stopped = 0
        # A machine whose stop failed keeps its row in ``stopping``, holding its place against
        # the fleet cap. Retry it here: nothing else revisits it.
        for stopping in self.store.list_workers(states=[WorkerState.STOPPING]):
            held_elsewhere = (
                stopping.lease_owner not in (None, self.instance_id)
                and (stopping.lease_expires_at or 0) > now
            )
            if held_elsewhere:
                continue
            if await self._stop_worker(stopping):
                stopped += 1
        # Every warm machine, including those of types removed from the catalogue.
        for candidate in self.store.list_workers(states=[WorkerState.WARM]):
            # Its type's cool-down, or the one it had when removed. A type forgotten across a
            # restart has none.
            spec = self.machine_types.get(candidate.machine_type) or self._retired_types.get(
                candidate.machine_type
            )
            cool_down = spec.cool_down_seconds if spec is not None else 0.0
            # Re-read: a machine later in the list may have taken a job while an earlier stop
            # awaited the backend, and stopping a busy machine kills that job.
            worker = self.store.get_worker(candidate.id)
            if worker is None or worker.state is not WorkerState.WARM:
                continue

            # A machine that never ran a job ages from when it booted, so
            # one started for a job that then failed still gets reaped.
            last_active = worker.last_active_at or worker.created_at

            if last_active > now:
                # The wall clock moved backwards. Unclamped, this machine is unreapable until the
                # clock catches up (unbounded idle billing); clamping makes it age from now.
                logger.warning(
                    "machine was last active in the future; clamping to now",
                    extra={"worker_id": worker.id, "skew_seconds": round(last_active - now, 1)},
                )
                self.store.touch_worker(worker.id, now)
                continue

            idle_for = now - last_active
            if idle_for < cool_down:
                continue

            logger.info(
                "stopping an idle machine",
                extra={
                    "worker_id": worker.id,
                    "machine_type": worker.machine_type,
                    "stale": self._is_stale(worker),
                    "tenant_id": worker.tenant_id,
                    "idle_seconds": round(idle_for, 1),
                },
            )
            await self._stop_worker(worker)
            stopped += 1

        if stopped:
            await self._offer_freed_capacity()
        return stopped

    def start_scaler(self, interval_seconds: float = 10.0) -> None:
        """Run the scaler passes on a timer until the pool closes.

        Nothing else stops idle machines; without this every machine bills forever.
        """
        self._spawn(self._scaler_loop(interval_seconds))

    async def _scaler_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.sync_catalogue()
                await self.reclaim_expired_leases()
                await self.reap_idle_workers()
                await self.probe_warm_workers()
            except Exception:
                # A raising pass must not kill the loop: a dead loop looks like having no scaler,
                # and that failure costs dollars per hour.
                logger.exception("scaler pass failed")

    # --- task bookkeeping ---

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_finished)

    def _task_finished(self, task: asyncio.Task) -> None:
        """Drop the task reference, logging the exception if the task died."""
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error("pool background task failed", exc_info=error)

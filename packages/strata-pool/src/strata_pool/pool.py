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

Several pool processes may share one store (`PostgresPoolStore`), each with
its own `instance_id`. They never dispatch the same job or start a machine for
the same demand, because every such change is a claim or a reservation in the
store. Each holds a lease on the machines and jobs it is acting on and renews
it while it works; when a process dies, the others fail its jobs and stop its
machines once the lease runs out (`reclaim_expired_leases`, run by the
scaler).
"""

import asyncio
import contextlib
import logging
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
        """
        Args:
            wall: Wall-clock source, for the timestamps that place a job in a
                billing period.
            monotonic: Monotonic source, for durations. Separate from `wall`
                so a clock step cannot change what a customer is charged.
            max_workers_total: Machines this pool may run at once, across
                every tenant and machine type. `MachineType.max_workers` caps
                one tenant; without this, the fleet is that cap times however
                many tenants show up. None means no ceiling, which is the
                right default for a single-tenant pool and the wrong one for
                a hosted deployment.
            tracer: OpenTelemetry tracer for the span around each execution.
                None uses the global provider's, when OpenTelemetry is
                installed; the pool does not require it.
            instance_id: This process's name in the leases it holds. Required
                when the store is shared: two processes under one name would
                each take the other's machines and jobs for their own. A
                process restarted under its old name takes back what it held
                at once, without waiting for those leases to expire.
            lease_seconds: How long a lease lasts without renewal, and so how
                long a dead process's jobs and machines wait before another
                takes them over. Renewed at a third of this.
        """
        if store.shared and instance_id is None:
            raise ValueError("a pool over a shared store needs its own instance_id")
        self.store = store
        self.backend = backend
        self.machine_types = {mt.name: mt for mt in machine_types}
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
        self.instance_id = instance_id or "pool"
        self.lease_seconds = lease_seconds
        self._tasks: set[asyncio.Task] = set()
        # worker id -> consecutive failed probes. In memory rather than in the
        # store: it is a judgement about right now, and a pool that restarts
        # health-checks everything in recover() anyway, so carrying a stale
        # count across a restart would only let one bad reading survive the
        # thing that would have corrected it.
        self._probe_failures: dict[str, int] = {}

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

    # --- catalogue ---

    async def replace_machine_types(self, machine_types: Iterable[MachineType]) -> None:
        """Swap the catalogue without restarting.

        A new type accepts jobs at once. A removed type accepts no more: its
        queued jobs fail with a reason, and its machines finish what they are
        running and retire once idle past their cool-down. A type whose image
        changed starts new machines with the new image; the machines already
        running the old one take no new jobs and retire the same way. A
        machine on the old image is never handed work, because a caller who
        changed the image wants the next job to run on the new one.
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
            trace_context=dict(trace_context or {}),
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
            # Lost the race for this machine or this job. Which one decides
            # whether there is anything left to do.
            latest = self.store.get_job(job.id)
            if latest is None or latest.state is not JobState.QUEUED:
                return True

    def _assign(self, worker: Worker, job: Job) -> bool:
        expires = self._wall() + self.lease_seconds
        if not self.store.claim_dispatch(worker, job, self.instance_id, expires):
            return False
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
        """Start machines for the jobs this tenant's fleet cannot absorb.

        Counted per tenant, because a machine belonging to another tenant is
        not capacity this one can use.
        """
        spec = self.machine_types.get(machine_type)
        if spec is None:
            return
        queued = self.store.count_queued(machine_type, tenant_id)

        # One machine per reservation, each decided against the store as it is
        # then: starting a machine awaits the backend, and in that window
        # another tenant, or another pool process, can start its own. Bounded
        # by the queue, so a backend that fails every start cannot spin here.
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
                # Saying so matters: a fleet at its ceiling looks exactly like
                # a queue that is simply slow, and a silent stall is the kind
                # of thing that gets debugged at 3am.
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
            await self._start_worker(spec, worker)

    async def _offer_freed_capacity(self) -> None:
        """Hand headroom back to whoever is waiting for it.

        A tenant the fleet cap deferred has no other way back: its jobs sit
        queued, and `_ensure_capacity` otherwise only runs when *that* tenant
        submits again. Freeing a machine is by definition freeing it for
        somebody else, so the release needs a path back to the work waiting
        on it — the same shape as every other bug in this package.
        """
        for machine_type in self.machine_types:
            for tenant_id in self.store.queued_tenants(machine_type):
                await self._ensure_capacity(machine_type, tenant_id)

    async def _start_worker(self, spec: MachineType, worker: Worker) -> None:
        """Provision the machine *worker* reserved, and poll it to warm in the
        background.

        The row is written before the backend call so a crash mid-start
        leaves evidence. It leaves a machine leaked if the crash lands after
        the provider created one — reconciling that needs a backend that can
        list its own machines, which arrives with the first real backend.
        """
        # The token has to reach the machine before it can accept anything, so
        # it goes in the environment the backend boots it with. Minted per
        # worker: one machine's credential must not open another's.
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
            return

        worker.backend_id = provisioned.backend_id
        worker.endpoint = provisioned.endpoint
        worker.region = provisioned.region
        try:
            recorded = self.store.record_provisioned(worker, self.instance_id, self._wall())
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
        if not recorded:
            # The start outlasted the lease and another process took the row
            # over. It had no machine to stop, so this one stops its own.
            logger.warning(
                "lost a starting machine's row to another pool process; stopping it",
                extra={"worker_id": worker.id, "backend_id": provisioned.backend_id},
            )
            await self.backend.stop(provisioned.backend_id)
            return
        self._spawn(self._await_boot(worker, spec, provisioned.endpoint))

    async def _await_boot(self, worker: Worker, spec: MachineType, endpoint: str) -> None:
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

        taken_over = False
        try:
            if not self.store.finish_job(job, self.instance_id):
                # This process stopped renewing for longer than a lease, and
                # another one failed the job and is stopping the machine. Its
                # answer stands; recording ours would bill a job twice.
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
            # Releasing the worker happens even if persistence just failed.
            # A store that rejects a write is a problem; a worker stuck BUSY
            # with nothing left to release it is a permanent one. A machine
            # another process took over is that process's to stop.
            if not taken_over:
                if keep_worker:
                    worker.last_active_at = self._wall()
                    if self.store.release_worker(worker, self.instance_id):
                        worker.state = WorkerState.WARM
                        worker.current_job_id = None
                        worker.lease_owner = None
                        worker.lease_expires_at = None
                    await self._drain(job.machine_type, job.tenant_id)
                else:
                    await self._stop_worker(worker)
                    # The queue may still hold work this machine was going to
                    # take — and the slot it just freed belongs to whoever is
                    # waiting, not only to this job's tenant.
                    await self._offer_freed_capacity()

    @contextlib.contextmanager
    def _execution_span(self, job: Job, worker: Worker) -> Iterator[dict[str, str]]:
        """A span for this job's time on the machine, and the headers that
        make the machine's work its child.

        Without OpenTelemetry the submitter's context is forwarded as it came,
        so the trace still joins up one level higher.
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

    async def _stop_worker(self, worker: Worker) -> bool:
        """Deallocate a machine and forget it.

        The row goes away rather than becoming a tombstone: nothing in this
        slice would ever bring it back, and jobs keep `worker_id` as plain
        history.

        The state flips to STOPPING first, synchronously. Everything below
        awaits, and in that window the dispatcher could otherwise find this
        machine warm and hand it a job we are about to kill.

        Returns whether this process stopped it.
        """
        # Every path that ends a machine comes through here, so this is the
        # one place the probe counter can be dropped without leaking. A
        # machine that missed one probe and was then reaped for idleness, or
        # stopped after a failed job, is never listed WARM again — so the
        # probe loop that would have popped it never sees it, and the entry
        # would outlive the machine for the life of the process.
        self._probe_failures.pop(worker.id, None)

        # A claim, not a write: of two processes deciding to stop the same
        # machine, or one stopping it while another hands it a job, one wins.
        # A machine another live process holds is left to that process.
        now = self._wall()
        if not self.store.claim_for_stop(
            worker.id, self.instance_id, now, now + self.lease_seconds
        ):
            return False
        worker.state = WorkerState.STOPPING
        worker.current_job_id = None
        worker.lease_owner = self.instance_id

        if worker.backend_id is not None:
            # No lease renewal around the stop: it is one provider call, well
            # inside the lease just claimed, and a renewal task would put an
            # extra suspension between the machine stopping and its row going,
            # where a caller could see a stopped machine still listed.
            try:
                await self.backend.stop(worker.backend_id)
            except Exception:
                logger.exception(
                    "backend failed to stop a worker; it may still be billing",
                    extra={"worker_id": worker.id, "backend_id": worker.backend_id},
                )
        self.store.delete_worker(worker.id)
        return True

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

        Over a shared store, only rows this instance may take over change: its
        own from before the restart, and any whose lease has run out. A machine
        or job another live process holds is that process's business, and the
        store's conditional updates are what leave it alone.
        """
        now = self._wall()
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
        """Finish what a pool process that stopped renewing left behind.

        Its in-flight jobs fail, since their results went to a process that is
        gone, and its starting, busy and stopping machines are stopped, since
        nothing can vouch for what is running on them. Returns how many
        machines were stopped. Run by the scaler; a pool alone on its store
        never has an expired lease that is not its own, and `recover` takes
        those back at startup.
        """
        now = self._wall()
        in_flight = [JobState.DISPATCHED, JobState.RUNNING]
        # Another process's rows only: this one's own are live in this process,
        # and a warm machine has no lease to expire. Whether the lease has run
        # out is the store's call, made in the same statement that acts on it.
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
        """Renew this process's lease on a job and a machine while the body
        runs, so another process takes them over only if this one dies."""

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
        """Health-check idle machines and retire the ones that are gone.

        Returns how many were stopped.

        Without this, a machine that dies while warm is discovered by the next
        job being sent to it — and that job fails. The machine was already
        unusable; the only thing the delay bought was a user watching a cell
        fail for reasons that have nothing to do with their code.

        Only warm machines. A busy one is answering a job, and a probe that
        loses a race against a long-running cell must not retire the machine
        running it.
        """
        # The endpoint is carried rather than re-read, so it is a str by
        # construction: a filter in a comprehension does not narrow the
        # attribute for the use below it.
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

        # Concurrently, because this shares the scaler pass with
        # reap_idle_workers. A backend's health check has its own timeout (10s
        # for RunPod), and a provider black-holing packets would make a serial
        # pass take that times the fleet size -- minutes during which nothing
        # is reaped, and reaping is the only thing that stops a machine
        # billing. The stall would be worst exactly when it costs most.
        results = await asyncio.gather(
            *(self._health_or_false(endpoint) for _, _, endpoint in candidates)
        )

        if len(candidates) > 1 and not any(results):
            # Every machine in the fleet failed at once. That is far more
            # likely to be this process's own network -- DNS, a proxy, an
            # exhausted connection pool -- than every machine dying
            # simultaneously, and acting on it would retire the entire warm
            # fleet for a fault that was never on the machines.
            #
            # Counts are left untouched rather than cleared, so a genuine
            # fleet-wide outage is still caught by the first pass that sees
            # anything healthy. And a machine that really is gone is still
            # stopped the way it always was: by the job dispatched to it
            # failing. With a single candidate there is no evidence either
            # way, so it is acted on -- one cold start beats never noticing.
            logger.warning(
                "every warm machine failed its probe; treating it as a "
                "pool-side fault rather than retiring the fleet",
                extra={"candidates": len(candidates)},
            )
            return 0

        stopped = 0
        for (spec, candidate, _endpoint), healthy in zip(candidates, results, strict=True):
            # Re-read after the await, exactly as reap_idle_workers does:
            # the machine may have taken a job while the probe was in
            # flight, and stopping a busy machine kills the job on it. A
            # machine that just started work is also evidence it is alive.
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
            # The slot this freed belongs to whoever is waiting, not only to
            # the tenant whose machine died.
            await self._offer_freed_capacity()
        return stopped

    # --- scaling down ---

    async def reap_idle_workers(self) -> int:
        """One scaler pass. Returns how many machines were stopped.

        Public and synchronous-to-call so tests drive it directly with an
        injected clock, rather than proving a cost control works by sleeping
        and hoping.
        """
        now = self._wall()
        stopped = 0
        # Every warm machine, not only those of types still in the catalogue:
        # a removed type's machines retire here too.
        for candidate in self.store.list_workers(states=[WorkerState.WARM]):
            # Its type's cool-down, or the one it had when the type was
            # removed. A type forgotten across a restart has none to honour.
            spec = self.machine_types.get(candidate.machine_type) or self._retired_types.get(
                candidate.machine_type
            )
            cool_down = spec.cool_down_seconds if spec is not None else 0.0
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
        """Run `reap_idle_workers` on a timer until the pool closes.

        Nothing else stops a machine that finished its work, so a pool
        without this call bills for every machine it ever started, forever.
        """
        self._spawn(self._scaler_loop(interval_seconds))

    async def _scaler_loop(self, interval_seconds: float) -> None:
        while True:
            await asyncio.sleep(interval_seconds)
            try:
                await self.reclaim_expired_leases()
                await self.reap_idle_workers()
                await self.probe_warm_workers()
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

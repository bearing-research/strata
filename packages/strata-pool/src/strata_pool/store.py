"""Persistence for pool state: SQLite by default, Postgres for a shared store.

The pool process holds no authoritative state in memory: workers, jobs, and
usage events live here, so a restart resumes rather than forgets (see
`Pool.recover`). SQLite is the default because self-hosting the pool should
not require standing up a database. `PostgresPoolStore` is for more than one
pool process over one store, so a restart of one does not pause dispatch.

Several processes over one store must not both dispatch a job, both start a
machine for the same demand, or both act on a machine the other is using.
Three things keep them apart, and every method that changes a job or a worker
the pool may share goes through one of them:

- **Claims** are conditional updates (`claim_dispatch`, `claim_for_stop`):
  the row changes only if it is still in the state the caller read, so of two
  processes that saw the same warm machine, one wins and the other learns so.
- **Reservations** (`reserve_worker`) count demand and capacity and insert the
  new machine's row in one serialized transaction, so two processes cannot
  both spend the same slot.
- **Leases** mark the rows a process is acting on: a starting, busy or
  stopping machine, and a dispatched or running job. The process renews them
  while it works. A row whose lease has expired belongs to whoever takes it
  over, which is how a surviving process finishes what a dead one left.

Timestamps are REAL epoch seconds (DOUBLE PRECISION in Postgres), matching
the artifact store. Lease expiry is compared across processes by wall clock,
so their clocks have to agree to well within a lease.
"""

import json
import re
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Protocol, cast

from strata_pool.types import Job, JobState, MachineType, UsageEvent, Worker, WorkerState

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    machine_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    backend_id TEXT,
    state TEXT NOT NULL CHECK (state IN ('starting','warm','busy','stopping')),
    endpoint TEXT,
    region TEXT,
    session_id TEXT,
    current_job_id TEXT,
    created_at REAL NOT NULL,
    last_active_at REAL,
    -- The pool's credential for this machine. Sensitive: it authorises code
    -- execution there. Kept out of Worker's repr for the same reason.
    auth_token TEXT,
    image TEXT,
    lease_owner TEXT,
    lease_expires_at REAL
);

-- Partial, because a worker row exists before backend.start() has told us the
-- resource ID: those rows carry NULL and must not collide with each other.
CREATE UNIQUE INDEX IF NOT EXISTS idx_workers_backend_id
    ON workers (backend, backend_id) WHERE backend_id IS NOT NULL;

-- Placement always filters by tenant as well as type: a machine belongs to
-- one tenant for its life.
CREATE INDEX IF NOT EXISTS idx_workers_type_state
    ON workers (machine_type, tenant_id, state);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL CHECK (state IN
        ('queued','dispatched','running','completed','failed','timed_out')),
    session_id TEXT,
    worker_id TEXT,
    timeout_seconds REAL,
    payload BLOB NOT NULL,
    result BLOB,
    error TEXT,
    submitted_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    trace_context TEXT,
    lease_owner TEXT,
    lease_expires_at REAL
);

-- No foreign key to workers(id): a worker row is deleted when its machine
-- dies, and the job history that names it must outlive the machine.
CREATE INDEX IF NOT EXISTS idx_jobs_queue
    ON jobs (machine_type, priority DESC, submitted_at ASC) WHERE state = 'queued';

CREATE INDEX IF NOT EXISTS idx_jobs_tenant
    ON jobs (tenant_id, submitted_at DESC);

CREATE TABLE IF NOT EXISTS usage_events (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    machine_type TEXT NOT NULL,
    worker_id TEXT,
    duration_ms REAL NOT NULL,
    started_at REAL NOT NULL,
    completed_at REAL NOT NULL,
    terminal_state TEXT NOT NULL
);

-- One event per job. A second insert for the same job is a double bill, and
-- it should fail loudly rather than be quietly deduplicated: the two events
-- would not necessarily agree on duration.
CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_job ON usage_events (job_id);

CREATE INDEX IF NOT EXISTS idx_usage_tenant
    ON usage_events (tenant_id, started_at DESC);

-- The machine-type catalogue last set over the API, so a restart serves the
-- catalogue the operator set rather than the one the process was started with.
-- One row; absent until a catalogue is set.
CREATE TABLE IF NOT EXISTS catalogue (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    machine_types TEXT NOT NULL
);
"""

# Columns added after the first release, so a store written by an older pool
# is upgraded in place.
_ADDED_COLUMNS = (
    ("workers", "image", "TEXT"),
    ("workers", "lease_owner", "TEXT"),
    ("workers", "lease_expires_at", "REAL"),
    ("jobs", "trace_context", "TEXT"),
    ("jobs", "lease_owner", "TEXT"),
    ("jobs", "lease_expires_at", "REAL"),
)

# A row with no lease, a lease held by the caller, or a lease nobody renewed in
# time: the rows a process may act on. Parameters: (owner, now).
_TAKEOVER = "(lease_owner IS NULL OR lease_owner = ? OR lease_expires_at < ?)"

Reservation = Literal["reserved", "no_demand", "tenant_cap", "fleet_cap"]


def _to_worker(row: Any) -> Worker:
    return Worker(
        id=row["id"],
        machine_type=row["machine_type"],
        tenant_id=row["tenant_id"],
        backend=row["backend"],
        state=WorkerState(row["state"]),
        created_at=row["created_at"],
        backend_id=row["backend_id"],
        endpoint=row["endpoint"],
        region=row["region"],
        session_id=row["session_id"],
        current_job_id=row["current_job_id"],
        last_active_at=row["last_active_at"],
        image=row["image"],
        auth_token=row["auth_token"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
    )


def _to_job(row: Any) -> Job:
    return Job(
        id=row["id"],
        tenant_id=row["tenant_id"],
        machine_type=row["machine_type"],
        payload=bytes(row["payload"]),
        state=JobState(row["state"]),
        submitted_at=row["submitted_at"],
        priority=row["priority"],
        session_id=row["session_id"],
        worker_id=row["worker_id"],
        timeout_seconds=row["timeout_seconds"],
        result=bytes(row["result"]) if row["result"] is not None else None,
        error=row["error"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        trace_context=json.loads(row["trace_context"]) if row["trace_context"] else {},
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
    )


def _placeholders(values: list[object]) -> str:
    return ",".join("?" * len(values))


class Store(Protocol):
    """What `Pool` needs from its persistence."""

    shared: bool
    """Whether other pool processes may use this store at the same time."""

    def close(self) -> None: ...

    def get_worker(self, worker_id: str) -> Worker | None: ...

    def delete_worker(self, worker_id: str) -> None: ...

    def list_workers(
        self, machine_type: str | None = None, states: Iterable[WorkerState] | None = None
    ) -> list[Worker]: ...

    def find_warm_worker(
        self,
        machine_type: str,
        tenant_id: str,
        session_id: str | None = None,
        image: str | None = None,
    ) -> Worker | None: ...

    def count_all_workers(self, states: Iterable[WorkerState]) -> int: ...

    def reserve_worker(
        self, worker: Worker, *, max_workers: int, max_workers_total: int | None
    ) -> Reservation: ...

    def record_provisioned(self, worker: Worker, owner: str, now: float) -> bool: ...

    def mark_warm(self, worker: Worker, owner: str, now: float) -> bool: ...

    def claim_dispatch(
        self, worker: Worker, job: Job, owner: str, lease_expires_at: float
    ) -> bool: ...

    def release_worker(self, worker: Worker, owner: str) -> bool: ...

    def claim_for_stop(
        self, worker_id: str, owner: str, now: float, lease_expires_at: float
    ) -> bool: ...

    def touch_worker(self, worker_id: str, last_active_at: float) -> None: ...

    def renew_lease(
        self,
        owner: str,
        lease_expires_at: float,
        *,
        job_id: str | None = None,
        worker_id: str | None = None,
    ) -> None: ...

    def save_job(self, job: Job) -> None: ...

    def get_job(self, job_id: str) -> Job | None: ...

    def list_jobs(self, states: Iterable[JobState] | None = None) -> list[Job]: ...

    def next_queued_job(self, machine_type: str, tenant_id: str) -> Job | None: ...

    def queued_tenants(self, machine_type: str) -> list[str]: ...

    def queued_machine_types(self) -> list[str]: ...

    def count_queued(self, machine_type: str, tenant_id: str) -> int: ...

    def finish_job(self, job: Job, owner: str) -> bool: ...

    def fail_job(
        self,
        job_id: str,
        error: str,
        completed_at: float,
        *,
        states: Iterable[JobState],
        owner: str,
        now: float,
    ) -> bool: ...

    def save_machine_types(self, machine_types: Iterable[MachineType]) -> None: ...

    def load_machine_types(self) -> list[MachineType] | None: ...

    def record_usage(self, event: UsageEvent) -> None: ...

    def list_usage(self, tenant_id: str | None = None) -> list[UsageEvent]: ...


class _SqlStore:
    """The pool's SQL, written once for both engines.

    Statements use `?` placeholders; the Postgres store rewrites them. None of
    this module's SQL has a `?` or a `%` inside a literal, which is what makes
    the plain rewrite safe here.
    """

    shared = False

    def __init__(self) -> None:
        # Re-entrant: a transaction holds it while calling the read helpers,
        # which take it themselves.
        self._lock = threading.RLock()

    # Engine-specific pieces.

    def _run(self, sql: str, params: Iterable[object] = ()) -> Any:
        raise NotImplementedError

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        """A write transaction serialized against every other process."""
        raise NotImplementedError
        yield

    def close(self) -> None:
        raise NotImplementedError

    def _one(self, sql: str, params: Iterable[object] = ()) -> Any:
        with self._lock:
            return self._run(sql, params).fetchone()

    def _all(self, sql: str, params: Iterable[object] = ()) -> list[Any]:
        with self._lock:
            return self._run(sql, params).fetchall()

    def _changed(self, sql: str, params: Iterable[object] = ()) -> bool:
        """Run a conditional update; whether it matched a row."""
        with self._lock:
            return self._run(sql, params).rowcount > 0

    # --- workers ---

    def save_worker(self, worker: Worker) -> None:
        """Write a worker as it stands, whatever the stored row says.

        Not for rows another process may be changing: `Pool` goes through the
        claims below instead. Kept for tests and single-process tooling.
        """
        with self._lock:
            self._run(
                """
                INSERT INTO workers (id, machine_type, tenant_id, backend, backend_id,
                                     state, endpoint, region, session_id, current_job_id,
                                     created_at, last_active_at, auth_token, image,
                                     lease_owner, lease_expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    backend_id = excluded.backend_id,
                    state = excluded.state,
                    endpoint = excluded.endpoint,
                    region = excluded.region,
                    session_id = excluded.session_id,
                    current_job_id = excluded.current_job_id,
                    last_active_at = excluded.last_active_at,
                    auth_token = excluded.auth_token,
                    lease_owner = excluded.lease_owner,
                    lease_expires_at = excluded.lease_expires_at
                """,
                self._worker_values(worker),
            )

    @staticmethod
    def _worker_values(worker: Worker) -> tuple[object, ...]:
        return (
            worker.id,
            worker.machine_type,
            worker.tenant_id,
            worker.backend,
            worker.backend_id,
            worker.state.value,
            worker.endpoint,
            worker.region,
            worker.session_id,
            worker.current_job_id,
            worker.created_at,
            worker.last_active_at,
            worker.auth_token,
            worker.image,
            worker.lease_owner,
            worker.lease_expires_at,
        )

    def get_worker(self, worker_id: str) -> Worker | None:
        row = self._one("SELECT * FROM workers WHERE id = ?", (worker_id,))
        return _to_worker(row) if row else None

    def delete_worker(self, worker_id: str) -> None:
        with self._lock:
            self._run("DELETE FROM workers WHERE id = ?", (worker_id,))

    def list_workers(
        self,
        machine_type: str | None = None,
        states: Iterable[WorkerState] | None = None,
    ) -> list[Worker]:
        sql = "SELECT * FROM workers"
        clauses: list[str] = []
        params: list[object] = []
        if machine_type is not None:
            clauses.append("machine_type = ?")
            params.append(machine_type)
        if states is not None:
            state_values: list[object] = [s.value for s in states]
            clauses.append(f"state IN ({_placeholders(state_values)})")
            params.extend(state_values)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC"
        return [_to_worker(row) for row in self._all(sql, params)]

    def find_warm_worker(
        self,
        machine_type: str,
        tenant_id: str,
        session_id: str | None = None,
        image: str | None = None,
    ) -> Worker | None:
        """The warm worker to hand the next job to.

        Tenant is not optional. A machine that ran one tenant's code is never
        offered to another, so there is no call site that legitimately wants
        "any warm worker of this type".

        With `session_id`, only a worker that already served that session
        matches — the caller falls back to the tenant's other warm workers
        itself, so an affinity miss is a visible decision rather than a silent
        one.

        With `image`, a machine booted with another image is not a match: it
        is stale, and a job sent to it would run on what the catalogue no
        longer names.
        """
        sql = "SELECT * FROM workers WHERE machine_type = ? AND tenant_id = ? AND state = 'warm'"
        params: list[object] = [machine_type, tenant_id]
        if image is not None:
            sql += " AND (image IS NULL OR image = ?)"
            params.append(image)
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        # Most recently active first, never-active last, on both engines.
        sql += (
            " ORDER BY CASE WHEN last_active_at IS NULL THEN 1 ELSE 0 END,"
            " last_active_at DESC, created_at ASC LIMIT 1"
        )
        row = self._one(sql, params)
        return _to_worker(row) if row else None

    def count_workers(
        self,
        machine_type: str,
        tenant_id: str,
        states: Iterable[WorkerState],
        image: str | None = None,
    ) -> int:
        """Machines of this type belonging to this tenant. Capacity is
        counted per tenant because `max_workers` is a per-tenant cap.

        With `image`, stale machines booted with another image are left out:
        they take no new work, so they are not capacity."""
        state_values: list[object] = [s.value for s in states]
        if not state_values:
            return 0
        sql = (
            "SELECT COUNT(*) AS n FROM workers WHERE machine_type = ? AND tenant_id = ? "
            f"AND state IN ({_placeholders(state_values)})"
        )
        params: list[object] = [machine_type, tenant_id, *state_values]
        if image is not None:
            sql += " AND (image IS NULL OR image = ?)"
            params.append(image)
        return self._one(sql, params)["n"]

    def count_all_workers(self, states: Iterable[WorkerState]) -> int:
        """Machines in these states across every tenant and machine type.

        The per-tenant count is what capacity planning uses; this is what a
        global cap needs, and the two must not be confused.
        """
        state_values: list[object] = [s.value for s in states]
        if not state_values:
            return 0
        return self._one(
            f"SELECT COUNT(*) AS n FROM workers WHERE state IN ({_placeholders(state_values)})",
            state_values,
        )["n"]

    def reserve_worker(
        self, worker: Worker, *, max_workers: int, max_workers_total: int | None
    ) -> Reservation:
        """Insert *worker*, a machine about to start, if the queue needs it.

        Decided and written in one serialized transaction, so two processes
        reading the same queue cannot both start a machine for the same job,
        and two tenants cannot both spend the fleet's last slot.
        """
        live = [WorkerState.STARTING, WorkerState.WARM, WorkerState.BUSY]
        # A machine whose stop is in flight is still allocated at the provider,
        # and still billing, until the call comes back — so the fleet counts it
        # even though the tenant's own cap does not (below).
        allocated = [*live, WorkerState.STOPPING]
        with self._transaction():
            queued = self.count_queued(worker.machine_type, worker.tenant_id)
            # Stale machines (another image) are left out: they take no new
            # jobs, so counting them would hold work back from machines that
            # could run it. They still count against the global fleet cap.
            absorbing = self.count_workers(
                worker.machine_type,
                worker.tenant_id,
                [WorkerState.STARTING, WorkerState.WARM],
                image=worker.image,
            )
            if queued - absorbing <= 0:
                return "no_demand"
            # STOPPING is deliberately absent: a machine being torn down is not
            # capacity, and counting it would keep a tenant at its cap from
            # starting the replacement.
            if (
                self.count_workers(worker.machine_type, worker.tenant_id, live, image=worker.image)
                >= max_workers
            ):
                return "tenant_cap"
            if (
                max_workers_total is not None
                and self.count_all_workers(allocated) >= max_workers_total
            ):
                return "fleet_cap"
            self.save_worker(worker)
        return "reserved"

    def record_provisioned(self, worker: Worker, owner: str, now: float) -> bool:
        """Write down the machine the backend started, if the row is still ours."""
        return self._changed(
            f"UPDATE workers SET backend_id = ?, endpoint = ?, region = ? "
            f"WHERE id = ? AND state = 'starting' AND {_TAKEOVER}",
            (worker.backend_id, worker.endpoint, worker.region, worker.id, owner, now),
        )

    def mark_warm(self, worker: Worker, owner: str, now: float) -> bool:
        """A starting machine answered its health check: it takes work now."""
        return self._changed(
            f"UPDATE workers SET state = 'warm', lease_owner = NULL, lease_expires_at = NULL "
            f"WHERE id = ? AND state = 'starting' AND {_TAKEOVER}",
            (worker.id, owner, now),
        )

    def claim_dispatch(self, worker: Worker, job: Job, owner: str, lease_expires_at: float) -> bool:
        """Give *job* to *worker*, if the worker is still warm and the job
        still queued. Either both change or neither does."""
        try:
            with self._transaction():
                if not self._changed(
                    "UPDATE workers SET state = 'busy', current_job_id = ?, session_id = ?, "
                    "lease_owner = ?, lease_expires_at = ? WHERE id = ? AND state = 'warm'",
                    (job.id, job.session_id, owner, lease_expires_at, worker.id),
                ):
                    raise _Rollback
                if not self._changed(
                    "UPDATE jobs SET state = 'dispatched', worker_id = ?, lease_owner = ?, "
                    "lease_expires_at = ? WHERE id = ? AND state = 'queued'",
                    (worker.id, owner, lease_expires_at, job.id),
                ):
                    raise _Rollback
        except _Rollback:
            return False
        return True

    def release_worker(self, worker: Worker, owner: str) -> bool:
        """Hand a machine that finished its job back to the warm fleet."""
        return self._changed(
            "UPDATE workers SET state = 'warm', current_job_id = NULL, last_active_at = ?, "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = ? AND state = 'busy' AND lease_owner = ?",
            (worker.last_active_at, worker.id, owner),
        )

    def claim_for_stop(
        self, worker_id: str, owner: str, now: float, lease_expires_at: float
    ) -> bool:
        """Take a machine to stop it, unless another live process holds it."""
        return self._changed(
            f"UPDATE workers SET state = 'stopping', current_job_id = NULL, lease_owner = ?, "
            f"lease_expires_at = ? WHERE id = ? AND {_TAKEOVER}",
            (owner, lease_expires_at, worker_id, owner, now),
        )

    def touch_worker(self, worker_id: str, last_active_at: float) -> None:
        with self._lock:
            self._run(
                "UPDATE workers SET last_active_at = ? WHERE id = ?", (last_active_at, worker_id)
            )

    def renew_lease(
        self,
        owner: str,
        lease_expires_at: float,
        *,
        job_id: str | None = None,
        worker_id: str | None = None,
    ) -> None:
        with self._lock:
            if job_id is not None:
                self._run(
                    "UPDATE jobs SET lease_expires_at = ? WHERE id = ? AND lease_owner = ?",
                    (lease_expires_at, job_id, owner),
                )
            if worker_id is not None:
                self._run(
                    "UPDATE workers SET lease_expires_at = ? WHERE id = ? AND lease_owner = ?",
                    (lease_expires_at, worker_id, owner),
                )

    # --- jobs ---

    def save_job(self, job: Job) -> None:
        with self._lock:
            self._run(
                """
                INSERT INTO jobs (id, tenant_id, machine_type, priority, state,
                                  session_id, worker_id, timeout_seconds, payload,
                                  result, error, submitted_at, started_at, completed_at,
                                  trace_context, lease_owner, lease_expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state = excluded.state,
                    worker_id = excluded.worker_id,
                    result = excluded.result,
                    error = excluded.error,
                    started_at = excluded.started_at,
                    completed_at = excluded.completed_at
                """,
                (
                    job.id,
                    job.tenant_id,
                    job.machine_type,
                    job.priority,
                    job.state.value,
                    job.session_id,
                    job.worker_id,
                    job.timeout_seconds,
                    job.payload,
                    job.result,
                    job.error,
                    job.submitted_at,
                    job.started_at,
                    job.completed_at,
                    json.dumps(job.trace_context) if job.trace_context else None,
                    job.lease_owner,
                    job.lease_expires_at,
                ),
            )

    def get_job(self, job_id: str) -> Job | None:
        row = self._one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return _to_job(row) if row else None

    def list_jobs(self, states: Iterable[JobState] | None = None) -> list[Job]:
        sql = "SELECT * FROM jobs"
        params: list[object] = []
        if states is not None:
            state_values: list[object] = [s.value for s in states]
            sql += f" WHERE state IN ({_placeholders(state_values)})"
            params.extend(state_values)
        sql += " ORDER BY submitted_at ASC"
        return [_to_job(row) for row in self._all(sql, params)]

    def next_queued_job(self, machine_type: str, tenant_id: str) -> Job | None:
        """Highest priority first, FIFO within a priority, one tenant only.

        Scoped by tenant because a freed machine can only serve the tenant it
        belongs to. Draining globally would stop at the first job the machine
        is not allowed to run and starve everything behind it.
        """
        row = self._one(
            "SELECT * FROM jobs WHERE machine_type = ? AND tenant_id = ? "
            "AND state = 'queued' ORDER BY priority DESC, submitted_at ASC LIMIT 1",
            (machine_type, tenant_id),
        )
        return _to_job(row) if row else None

    def count_queued(self, machine_type: str, tenant_id: str) -> int:
        return self._one(
            "SELECT COUNT(*) AS n FROM jobs WHERE machine_type = ? AND tenant_id = ? "
            "AND state = 'queued'",
            (machine_type, tenant_id),
        )["n"]

    def queued_tenants(self, machine_type: str) -> list[str]:
        """Tenants with work waiting for this machine type.

        Recovery needs it: after a restart there is no submit to drive
        placement, so the pool has to ask who is waiting.
        """
        rows = self._all(
            "SELECT DISTINCT tenant_id FROM jobs WHERE machine_type = ? AND state = 'queued'",
            (machine_type,),
        )
        return [row["tenant_id"] for row in rows]

    def queued_machine_types(self) -> list[str]:
        """Machine types with work waiting, whether or not the catalogue
        still names them."""
        rows = self._all("SELECT DISTINCT machine_type FROM jobs WHERE state = 'queued'")
        return [row["machine_type"] for row in rows]

    def finish_job(self, job: Job, owner: str) -> bool:
        """Record a job's outcome, if the caller still holds it.

        False when another process already failed the job after the caller's
        lease expired: its answer stands, and this one is not metered.
        """
        return self._changed(
            "UPDATE jobs SET state = ?, result = ?, error = ?, completed_at = ?, "
            "lease_owner = NULL, lease_expires_at = NULL "
            "WHERE id = ? AND state IN ('dispatched', 'running') AND lease_owner = ?",
            (job.state.value, job.result, job.error, job.completed_at, job.id, owner),
        )

    def fail_job(
        self,
        job_id: str,
        error: str,
        completed_at: float,
        *,
        states: Iterable[JobState],
        owner: str,
        now: float,
    ) -> bool:
        """Fail a job still in one of *states* that no other live process holds."""
        state_values: list[object] = [s.value for s in states]
        return self._changed(
            f"UPDATE jobs SET state = 'failed', error = ?, completed_at = ?, "
            f"lease_owner = NULL, lease_expires_at = NULL "
            f"WHERE id = ? AND state IN ({_placeholders(state_values)}) AND {_TAKEOVER}",
            (error, completed_at, job_id, *state_values, owner, now),
        )

    # --- catalogue ---

    def save_machine_types(self, machine_types: Iterable[MachineType]) -> None:
        payload = json.dumps([asdict(spec) for spec in machine_types])
        with self._lock:
            self._run(
                "INSERT INTO catalogue (id, machine_types) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET machine_types = excluded.machine_types",
                (payload,),
            )

    def load_machine_types(self) -> list[MachineType] | None:
        """The last catalogue set, or None if none ever was."""
        row = self._one("SELECT machine_types FROM catalogue WHERE id = 1")
        if row is None:
            return None
        return [MachineType(**spec) for spec in json.loads(row["machine_types"])]

    # --- usage ---

    def record_usage(self, event: UsageEvent) -> None:
        """Persist one billable execution.

        Raises the engine's integrity error if this job was already metered.
        """
        with self._lock:
            self._run(
                """
                INSERT INTO usage_events (id, tenant_id, job_id, machine_type, worker_id,
                                          duration_ms, started_at, completed_at,
                                          terminal_state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.tenant_id,
                    event.job_id,
                    event.machine_type,
                    event.worker_id,
                    event.duration_ms,
                    event.started_at,
                    event.completed_at,
                    event.terminal_state.value,
                ),
            )

    def list_usage(self, tenant_id: str | None = None) -> list[UsageEvent]:
        sql = "SELECT * FROM usage_events"
        params: list[object] = []
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            params.append(tenant_id)
        sql += " ORDER BY started_at ASC"
        return [
            UsageEvent(
                id=row["id"],
                tenant_id=row["tenant_id"],
                job_id=row["job_id"],
                machine_type=row["machine_type"],
                duration_ms=row["duration_ms"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                terminal_state=JobState(row["terminal_state"]),
                worker_id=row["worker_id"],
            )
            for row in self._all(sql, params)
        ]


class _Rollback(Exception):
    """Abandons a transaction whose conditional update found the row taken."""


class PoolStore(_SqlStore):
    """Pool state in a SQLite file: the default.

    One connection guarded by a lock. Writes are small and the pool's
    background tasks all run on a single event loop, so the lock only ever
    contends with a caller inspecting the pool from another thread. Several
    processes on one host may share the file; `BEGIN IMMEDIATE` serializes
    their transactions.
    """

    def __init__(self, db_path: Path | str):
        super().__init__()
        self.db_path = db_path
        # Autocommit, with explicit transactions where several statements have
        # to land together.
        self._conn = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level=None, timeout=30.0
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            existing = {
                row[0]
                for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            for table, column, kind in _ADDED_COLUMNS:
                if table not in existing:
                    continue
                columns = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
                if column not in columns:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
            self._conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _run(self, sql: str, params: Iterable[object] = ()) -> Any:
        return self._conn.execute(sql, tuple(params))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")


# One advisory lock for every serialized pool transaction. Placement decisions
# are small and rare next to the machines they start, so one lock is enough.
_POSTGRES_LOCK_KEY = 0x5354_5241_504F_4F4C  # "STRAPOOL"


class PostgresPoolStore(_SqlStore):
    """Pool state in Postgres, for several pool processes over one store.

    Needs `psycopg` (the `postgres` extra). Each process must run its `Pool`
    with its own `instance_id`, since leases are how the processes tell their
    own rows from each other's.
    """

    shared = True

    def __init__(self, dsn: str):
        super().__init__()
        import psycopg
        from psycopg.rows import dict_row

        # Typed loosely: psycopg's overloads want literal SQL, and this store
        # builds its statements (placeholder lists, the `?` rewrite).
        self._conn: Any = cast(Any, psycopg).connect(dsn, autocommit=True, row_factory=dict_row)
        with self._transaction():
            self._conn.execute(
                re.sub(r"\bBLOB\b", "BYTEA", re.sub(r"\bREAL\b", "DOUBLE PRECISION", _SCHEMA_SQL))
            )
            for table, column, kind in _ADDED_COLUMNS:
                kind = "DOUBLE PRECISION" if kind == "REAL" else kind
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {kind}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _run(self, sql: str, params: Iterable[object] = ()) -> Any:
        return self._conn.execute(sql.replace("?", "%s"), tuple(params))

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("SELECT pg_advisory_xact_lock(%s)", (_POSTGRES_LOCK_KEY,))
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

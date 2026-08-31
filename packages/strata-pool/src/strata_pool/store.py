"""SQLite persistence for pool state.

The pool process holds no authoritative state in memory: workers, jobs, and
usage events live here, so a restart resumes rather than forgets (see
`Pool.recover`). SQLite is the default because self-hosting the pool should
not require standing up a database; the schema is plain enough to move to the
artifact store's Postgres dialect when a deployment needs it.

Timestamps are stored as REAL epoch seconds, matching the artifact store.
"""

import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path

from strata_pool.types import Job, JobState, UsageEvent, Worker, WorkerState

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    machine_type TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    backend TEXT NOT NULL,
    backend_id TEXT,
    state TEXT NOT NULL CHECK (state IN ('starting','warm','busy')),
    endpoint TEXT,
    region TEXT,
    session_id TEXT,
    current_job_id TEXT,
    created_at REAL NOT NULL,
    last_active_at REAL,
    -- The pool's credential for this machine. Sensitive: it authorises code
    -- execution there. Kept out of Worker's repr for the same reason.
    auth_token TEXT
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
    completed_at REAL
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
"""


def _to_worker(row: sqlite3.Row) -> Worker:
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
        auth_token=row["auth_token"],
    )


def _to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        tenant_id=row["tenant_id"],
        machine_type=row["machine_type"],
        payload=row["payload"],
        state=JobState(row["state"]),
        submitted_at=row["submitted_at"],
        priority=row["priority"],
        session_id=row["session_id"],
        worker_id=row["worker_id"],
        timeout_seconds=row["timeout_seconds"],
        result=row["result"],
        error=row["error"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


class PoolStore:
    """Pool state on disk.

    One connection guarded by a lock. Writes are small and the pool's
    background tasks all run on a single event loop, so the lock only ever
    contends with a caller inspecting the pool from another thread.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- workers ---

    def save_worker(self, worker: Worker) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO workers (id, machine_type, tenant_id, backend, backend_id,
                                     state, endpoint, region, session_id, current_job_id,
                                     created_at, last_active_at, auth_token)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    backend_id = excluded.backend_id,
                    state = excluded.state,
                    endpoint = excluded.endpoint,
                    region = excluded.region,
                    session_id = excluded.session_id,
                    current_job_id = excluded.current_job_id,
                    last_active_at = excluded.last_active_at,
                    auth_token = excluded.auth_token
                """,
                (
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
                ),
            )
            self._conn.commit()

    def get_worker(self, worker_id: str) -> Worker | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM workers WHERE id = ?", (worker_id,)).fetchone()
        return _to_worker(row) if row else None

    def delete_worker(self, worker_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
            self._conn.commit()

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
            state_values = [s.value for s in states]
            clauses.append(f"state IN ({','.join('?' * len(state_values))})")
            params.extend(state_values)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_worker(row) for row in rows]

    def find_warm_worker(
        self,
        machine_type: str,
        tenant_id: str,
        session_id: str | None = None,
    ) -> Worker | None:
        """The warm worker to hand the next job to.

        Tenant is not optional. A machine that ran one tenant's code is never
        offered to another, so there is no call site that legitimately wants
        "any warm worker of this type".

        With `session_id`, only a worker that already served that session
        matches — the caller falls back to the tenant's other warm workers
        itself, so an affinity miss is a visible decision rather than a silent
        one.
        """
        sql = "SELECT * FROM workers WHERE machine_type = ? AND tenant_id = ? AND state = 'warm'"
        params: list[object] = [machine_type, tenant_id]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        # SQLite already sorts NULLs last under DESC; spelling it out would
        # require SQLite >= 3.30 for no behavioural gain.
        sql += " ORDER BY last_active_at DESC, created_at ASC LIMIT 1"
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return _to_worker(row) if row else None

    def count_workers(
        self,
        machine_type: str,
        tenant_id: str,
        states: Iterable[WorkerState],
    ) -> int:
        """Machines of this type belonging to this tenant. Capacity is
        counted per tenant because `max_workers` is a per-tenant cap."""
        state_values = [s.value for s in states]
        if not state_values:
            return 0
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM workers WHERE machine_type = ? AND tenant_id = ? "
                f"AND state IN ({','.join('?' * len(state_values))})",
                [machine_type, tenant_id, *state_values],
            ).fetchone()
        return row[0]

    # --- jobs ---

    def save_job(self, job: Job) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (id, tenant_id, machine_type, priority, state,
                                  session_id, worker_id, timeout_seconds, payload,
                                  result, error, submitted_at, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_job(row) if row else None

    def list_jobs(self, states: Iterable[JobState] | None = None) -> list[Job]:
        sql = "SELECT * FROM jobs"
        params: list[object] = []
        if states is not None:
            state_values = [s.value for s in states]
            sql += f" WHERE state IN ({','.join('?' * len(state_values))})"
            params.extend(state_values)
        sql += " ORDER BY submitted_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_to_job(row) for row in rows]

    def next_queued_job(self, machine_type: str, tenant_id: str) -> Job | None:
        """Highest priority first, FIFO within a priority, one tenant only.

        Scoped by tenant because a freed machine can only serve the tenant it
        belongs to. Draining globally would stop at the first job the machine
        is not allowed to run and starve everything behind it.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE machine_type = ? AND tenant_id = ? "
                "AND state = 'queued' ORDER BY priority DESC, submitted_at ASC LIMIT 1",
                (machine_type, tenant_id),
            ).fetchone()
        return _to_job(row) if row else None

    def count_queued(self, machine_type: str, tenant_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE machine_type = ? AND tenant_id = ? "
                "AND state = 'queued'",
                (machine_type, tenant_id),
            ).fetchone()
        return row[0]

    def queued_tenants(self, machine_type: str) -> list[str]:
        """Tenants with work waiting for this machine type.

        Recovery needs it: after a restart there is no submit to drive
        placement, so the pool has to ask who is waiting.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT tenant_id FROM jobs WHERE machine_type = ? AND state = 'queued'",
                (machine_type,),
            ).fetchall()
        return [row[0] for row in rows]

    # --- usage ---

    def record_usage(self, event: UsageEvent) -> None:
        """Persist one billable execution.

        Raises `sqlite3.IntegrityError` if this job was already metered.
        """
        with self._lock:
            self._conn.execute(
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
            self._conn.commit()

    def list_usage(self, tenant_id: str | None = None) -> list[UsageEvent]:
        sql = "SELECT * FROM usage_events"
        params: list[object] = []
        if tenant_id is not None:
            sql += " WHERE tenant_id = ?"
            params.append(tenant_id)
        sql += " ORDER BY started_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
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
            for row in rows
        ]

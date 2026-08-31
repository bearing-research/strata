"""Value types for the worker pool.

States are deliberately narrow: only the transitions the pool actually
performs today exist here. `cold` (a stopped-but-provisioned worker) and
`cancelled` (a job killed on request) belong to the scaler and the reaper,
and land with them — an enum member nothing produces is an invariant nobody
is checking.
"""

import enum
import secrets
import uuid
from dataclasses import dataclass, field


class WorkerState(enum.StrEnum):
    """Lifecycle of a machine that runs jobs."""

    STARTING = "starting"
    """Boot in progress. Becomes WARM when the backend health check passes."""

    WARM = "warm"
    """Running, idle, ready to accept a job. Incurring compute cost."""

    BUSY = "busy"
    """Running a job."""


class JobState(enum.StrEnum):
    """Lifecycle of a unit of work."""

    QUEUED = "queued"
    DISPATCHED = "dispatched"
    """Assigned to a worker; the execute request has not been sent yet."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


TERMINAL_JOB_STATES = frozenset(
    {JobState.COMPLETED, JobState.FAILED, JobState.TIMED_OUT},
)


def new_id(prefix: str) -> str:
    """Pool-assigned identifier. Callers never invent these."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def new_auth_token() -> str:
    """A worker's credential. One per machine, minted before it boots.

    `secrets`, not `uuid`: this gates arbitrary code execution on the machine,
    so it has to come from a CSPRNG.
    """
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class MachineType:
    """A named capability class the pool can provision.

    The pool does not know what the label means — it matches a job's declared
    requirement against a worker's capability and hands the rest to the
    backend.
    """

    name: str
    image: str
    max_workers: int = 10
    """Cap on machines of this type *per tenant*, since a machine belongs to
    one tenant for its life. There is no global cap yet; a hosted deployment
    will want one."""

    boot_timeout_seconds: float = 120.0
    job_timeout_seconds: float = 300.0
    env: dict[str, str] = field(default_factory=dict)

    cpus: float | None = None
    """CPU allowance. None means whatever the backend defaults to, which for
    a container is the whole host."""

    memory_mb: int | None = None


@dataclass
class Worker:
    """A machine, as the pool tracks it."""

    id: str
    machine_type: str
    tenant_id: str
    """Who this machine belongs to, for its whole life.

    A machine is never handed across tenants. Even scrubbed of files, a
    process that ran one tenant's code is not a boundary the next tenant
    should have to trust — and for a GPU, memory is not reliably zeroed
    between processes at all.
    """

    backend: str
    state: WorkerState
    created_at: float
    backend_id: str | None = None
    """Cloud-assigned resource ID. None between row insert and backend.start()."""

    endpoint: str | None = None
    region: str | None = None
    session_id: str | None = None
    """Affinity tag of the last session this worker served."""

    current_job_id: str | None = None
    last_active_at: float | None = None

    auth_token: str | None = field(default=None, repr=False)
    """Bearer credential the pool presents to this machine.

    repr=False so it cannot ride along into a log line: every logger call
    that carries a worker would otherwise print a working credential for a
    machine that runs arbitrary code.
    """


@dataclass
class Job:
    """A unit of work submitted to the pool."""

    id: str
    tenant_id: str
    machine_type: str
    payload: bytes
    state: JobState
    submitted_at: float
    priority: int = 0
    """Higher runs first; ties break FIFO by submission time."""

    session_id: str | None = None
    worker_id: str | None = None
    timeout_seconds: float | None = None
    """Overrides the machine type's job timeout when set."""

    result: bytes | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass(frozen=True)
class UsageEvent:
    """One billable execution.

    `duration_ms` is measured on a monotonic clock while `started_at` and
    `completed_at` are wall-clock: a clock step during a job must not be able
    to change what a customer is charged, but the billing period a job falls
    into is a wall-clock question.

    The pool records an event for every terminal job, including failures —
    the machine time was consumed either way. Deciding what is actually
    billable is the metering layer's call, not the pool's.
    """

    id: str
    tenant_id: str
    job_id: str
    machine_type: str
    duration_ms: float
    started_at: float
    completed_at: float
    terminal_state: JobState
    worker_id: str | None = None

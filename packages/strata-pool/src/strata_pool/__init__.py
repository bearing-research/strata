"""Worker pool for dispatching Strata jobs to ephemeral machines."""

from strata_pool.backend import Backend, ProvisionedWorker
from strata_pool.backends import DockerBackend
from strata_pool.pool import Pool
from strata_pool.store import PoolStore
from strata_pool.types import (
    Job,
    JobState,
    MachineType,
    UsageEvent,
    Worker,
    WorkerState,
)

__all__ = [
    "Backend",
    "DockerBackend",
    "Job",
    "JobState",
    "MachineType",
    "Pool",
    "PoolStore",
    "ProvisionedWorker",
    "UsageEvent",
    "Worker",
    "WorkerState",
]

__version__ = "0.1.0"

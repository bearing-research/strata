"""Worker pool for dispatching Strata jobs to ephemeral machines."""

from importlib import metadata as _metadata

from strata_pool.backend import Backend, ProvisionedWorker
from strata_pool.backends import DockerBackend, FlyBackend, RunPodBackend
from strata_pool.pool import Pool
from strata_pool.store import PoolStore, PostgresPoolStore, Store
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
    "FlyBackend",
    "Job",
    "JobState",
    "MachineType",
    "Pool",
    "PoolStore",
    "PostgresPoolStore",
    "ProvisionedWorker",
    "RunPodBackend",
    "Store",
    "UsageEvent",
    "Worker",
    "WorkerState",
]

# From the installed distribution, so it cannot fall behind pyproject.toml the
# way a literal did (it said 0.1.0 through 0.8.0).
__version__ = _metadata.version("strata-pool")

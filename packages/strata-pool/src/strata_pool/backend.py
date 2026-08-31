"""The seam between the pool and whatever actually runs machines."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ProvisionedWorker:
    """What a backend hands back from `start()`."""

    backend_id: str
    """Cloud-specific resource ID, used for stop()."""

    endpoint: str
    """HTTP base URL where the worker will accept jobs.

    Required at start time. A backend that only learns the address later
    should block until it knows it: the pool has no other way to health-check
    a booting machine, and a worker it cannot reach is a worker it cannot
    stop billing for.
    """

    region: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


class Backend(Protocol):
    """A cloud provider or local runtime that can start and stop machines."""

    name: str

    async def start(
        self,
        machine_type: str,
        image: str,
        env: dict[str, str] | None = None,
    ) -> ProvisionedWorker:
        """Provision and start a worker.

        Returns once the machine is booting; it does not need to be ready.
        The pool polls `health()` until it is.
        """
        ...

    async def stop(self, backend_id: str) -> None:
        """Stop and deallocate a worker. Must be idempotent."""
        ...

    async def health(self, endpoint: str) -> bool:
        """Report whether a worker is ready to accept jobs."""
        ...

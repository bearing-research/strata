"""Establish and tear down SSH-tunneled workers for a notebook session.

The logic behind the ``/workers/ssh`` routes, testable without FastAPI or a
network. A worker is provisioned and tunneled via the supervisor, then
registered as a ``[[workers]]`` entry so it routes like any other.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.notebook.remote_worker_supervisor import RemoteWorkerSupervisor, TunnelRecord
    from strata.notebook.session import NotebookSession


def default_worker_name(ssh_target: str) -> str:
    """Derive a worker name from an SSH target's host (slugified)."""
    host = ssh_target.split("@")[-1].split(":")[0]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", host).strip("-.")
    return slug or "remote"


def establish_ssh_worker(
    session: NotebookSession,
    supervisor: RemoteWorkerSupervisor,
    *,
    ssh_target: str,
    name: str | None = None,
    remote_port: int | None = None,
    local_port: int | None = None,
    extras: str = "notebook",
    pin: str | None = None,
    install: bool = True,
    set_default: bool = False,
) -> TunnelRecord:
    """Provision and tunnel a remote worker, register it in ``notebook.toml``; return the tunnel.

    Raises :class:`PermissionError` before provisioning if worker definitions
    are not editable (service mode). A failed registration tears the tunnel down.
    """
    from strata.notebook.ops import LocalNotebookOps, NotebookOpsError
    from strata.notebook.workers import notebook_worker_definitions_editable

    if not notebook_worker_definitions_editable(session.notebook_state):
        raise PermissionError("worker definitions are managed by the server in service mode")

    worker_name = name or default_worker_name(ssh_target)
    record = supervisor.establish(
        worker_name,
        ssh_target,
        remote_port=remote_port,
        local_port=local_port,
        extras=extras,
        pin=pin,
        install=install,
    )
    try:
        LocalNotebookOps.from_session(session).add_worker(
            worker_name,
            url=record.executor_url,
            transport="direct",
            # Per-target runtime id keeps this worker's cached results apart from local runs.
            runtime_id=f"ssh:{record.ssh_target}",
            set_default=set_default,
        )
    except NotebookOpsError:
        supervisor.teardown(worker_name)  # don't leave a tunnel with no registration
        raise
    # add_worker reloaded the session.
    return record


def teardown_ssh_worker(
    session: NotebookSession,
    supervisor: RemoteWorkerSupervisor,
    name: str,
    *,
    stop_remote: bool = False,
) -> bool:
    """Close the tunnel for *name* and remove its ``[[workers]]`` entry.

    Returns whether a tunnel was present; a missing entry is a no-op.
    """
    from strata.notebook.ops import LocalNotebookOps

    existed = supervisor.teardown(name, stop_remote=stop_remote)
    if any(worker.name == name for worker in session.notebook_state.workers):
        # remove_worker reloads the session.
        LocalNotebookOps.from_session(session).remove_worker(name)
    return existed

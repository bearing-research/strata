"""Establish / tear down SSH-tunneled workers for a notebook session.

The request-shaping logic behind the ``/workers/ssh`` routes, factored out so it
is unit-testable with a fake supervisor and a real local session — no FastAPI, no
network. ``establish_ssh_worker`` provisions + tunnels via the supervisor, then
registers the result as a ``[[workers]]`` entry through the shared ops so it
routes like any other worker; ``teardown_ssh_worker`` reverses both.
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
    """Provision + tunnel a remote worker and register it in ``notebook.toml``.

    Returns the tunnel record. Raises :class:`PermissionError` if worker
    definitions aren't editable (service mode) — checked *before* provisioning so
    a doomed request never opens an SSH connection. If provisioning succeeds but
    registration fails, the tunnel is torn back down so nothing dangles.
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
            # A stable per-target runtime id so this worker's cached results are
            # keyed apart from local runs (train/serve provenance parity).
            runtime_id=f"ssh:{record.ssh_target}",
            set_default=set_default,
        )
    except NotebookOpsError:
        supervisor.teardown(worker_name)  # don't leave a tunnel with no registration
        raise
    session.reload()
    return record


def teardown_ssh_worker(
    session: NotebookSession,
    supervisor: RemoteWorkerSupervisor,
    name: str,
    *,
    stop_remote: bool = False,
) -> bool:
    """Close the tunnel for *name* and remove its ``[[workers]]`` entry.

    Returns whether a tunnel was present. Removing the notebook entry is a no-op
    when it isn't registered (a tunnel can outlive its registration if a user
    hand-edited the TOML).
    """
    from strata.notebook.ops import LocalNotebookOps

    existed = supervisor.teardown(name, stop_remote=stop_remote)
    if any(worker.name == name for worker in session.notebook_state.workers):
        LocalNotebookOps.from_session(session).remove_worker(name)
        session.reload()
    return existed

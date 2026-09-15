"""Which notebook scope each operation needs, for every way into a notebook.

``notebook:read`` / ``notebook:write`` / ``notebook:execute`` are the scopes the
service-mode proxy config and the deployment docs advertise. One table serves
the WebSocket frames and the REST routes, so a viewer is a viewer however it
reaches the server; a second table kept beside one of them would drift the way
the REST routes did, which checked nothing while the frames checked all three.

Both default to ``notebook:execute``: an operation nobody classified is
privileged until someone does.
"""

from __future__ import annotations

from strata.notebook.protocol import MessageType

NOTEBOOK_SCOPE_READ = "notebook:read"
NOTEBOOK_SCOPE_WRITE = "notebook:write"
NOTEBOOK_SCOPE_EXECUTE = "notebook:execute"

# --- WebSocket frames --------------------------------------------------------
#
# Each C→S frame is mapped to the least scope that covers what it can actually
# do; anything unlisted defaults to ``notebook:execute``.

# Read-only: observe state, compute previews. No mutation, no code runs.
_READ_FRAMES = frozenset(
    {
        MessageType.NOTEBOOK_SYNC,
        MessageType.IMPACT_PREVIEW_REQUEST,
        MessageType.PROFILING_REQUEST,
    }
)

# Mutate committed notebook content, but don't themselves run code.
_WRITE_FRAMES = frozenset(
    {
        MessageType.CELL_SOURCE_UPDATE,
        MessageType.VARIANT_SET_ACTIVE,
        MessageType.VARIANT_ADD,
    }
)

# Everything else runs code or mutates the environment — cell execution, the
# inspect REPL (evals arbitrary expressions), widget updates (re-run the
# widget cell and cascade), dependency changes (invoke uv), and the agent
# confirm/cancel controls. Listed explicitly for documentation value even
# though the default is already ``notebook:execute``.
_EXECUTE_FRAMES = frozenset(
    {
        MessageType.CELL_EXECUTE,
        MessageType.CELL_EXECUTE_CASCADE,
        MessageType.CELL_EXECUTE_FORCE,
        MessageType.CELL_EXECUTE_RERUN,
        MessageType.CELL_RUN_TESTS,
        MessageType.NOTEBOOK_RUN_ALL,
        MessageType.NOTEBOOK_RERUN_ALL,
        MessageType.CELL_CANCEL,
        MessageType.WIDGET_UPDATE,
        MessageType.INSPECT_OPEN,
        MessageType.INSPECT_EVAL,
        MessageType.INSPECT_CLOSE,
        MessageType.DEPENDENCY_ADD,
        MessageType.DEPENDENCY_REMOVE,
        MessageType.AGENT_CANCEL,
        MessageType.AGENT_CONFIRM_RESPONSE,
    }
)


def required_scope_for_frame(msg_type: str) -> str:
    """Return the notebook scope a C→S frame requires (fail-closed default)."""
    if msg_type in _READ_FRAMES:
        return NOTEBOOK_SCOPE_READ
    if msg_type in _WRITE_FRAMES:
        return NOTEBOOK_SCOPE_WRITE
    return NOTEBOOK_SCOPE_EXECUTE


# --- REST routes -------------------------------------------------------------
#
# A GET only reads, and so do the two previews below. The mutations listed as
# write change the notebook's committed content or configuration without
# running anything. Every other route defaults to ``notebook:execute``, the same
# fail-closed default as the frames: running a cell or its tests, syncing or
# changing dependencies (uv runs build scripts), importing a requirements file,
# changing the Python version, provisioning an SSH worker, the assistant (which
# runs cells), and any route added later until someone classifies it. Keys are
# (method, route path template).
_READ_POST_ROUTES = frozenset(
    {
        ("POST", "/v1/notebooks/{notebook_id}/environment/requirements.txt/preview"),
        ("POST", "/v1/notebooks/{notebook_id}/environment/environment.yaml/preview"),
    }
)

_WRITE_ROUTES = frozenset(
    {
        ("POST", "/v1/notebooks/open"),
        ("POST", "/v1/notebooks/create"),
        ("POST", "/v1/notebooks/import"),
        ("POST", "/v1/notebooks/import-snapshot"),
        ("POST", "/v1/notebooks/{notebook_id}/quiesce"),
        ("POST", "/v1/notebooks/{notebook_id}/release"),
        ("DELETE", "/v1/notebooks/{notebook_id}"),
        ("POST", "/v1/notebooks/recents/validate"),
        ("POST", "/v1/notebooks/delete-by-path"),
        ("PUT", "/v1/notebooks/{notebook_id}/cells/reorder"),
        ("PUT", "/v1/notebooks/{notebook_id}/cells/{cell_id}"),
        ("PUT", "/v1/notebooks/{notebook_id}/mounts"),
        ("PUT", "/v1/notebooks/{notebook_id}/connections"),
        ("PUT", "/v1/notebooks/{notebook_id}/workers"),
        ("DELETE", "/v1/notebooks/{notebook_id}/workers/ssh/{worker_name}"),
        ("PUT", "/v1/notebooks/{notebook_id}/worker"),
        ("PUT", "/v1/notebooks/{notebook_id}/timeout"),
        ("POST", "/v1/notebooks/{notebook_id}/variant-groups/{group_id}/variants"),
        ("PUT", "/v1/notebooks/{notebook_id}/variant-groups/{group_id}"),
        ("PUT", "/v1/notebooks/{notebook_id}/env"),
        ("PUT", "/v1/notebooks/{notebook_id}/secret-manager/config"),
        ("POST", "/v1/notebooks/{notebook_id}/secret-manager/refresh"),
        ("POST", "/v1/notebooks/{notebook_id}/cells"),
        ("DELETE", "/v1/notebooks/{notebook_id}/cells/{cell_id}"),
        ("PUT", "/v1/notebooks/{notebook_id}/name"),
        ("POST", "/v1/notebooks/{notebook_id}/artifacts/{artifact_id}/v/{version}/promote"),
        ("PUT", "/v1/notebooks/{notebook_id}/cells/{cell_id}/tests"),
        ("PUT", "/v1/notebooks/{notebook_id}/ai/model"),
        ("POST", "/v1/notebooks/{notebook_id}/ai/agent/reset"),
        ("POST", "/v1/projects/{path:path}/quiesce"),
        ("POST", "/v1/projects/{path:path}/release"),
    }
)


def required_scope_for_route(method: str, path: str) -> str:
    """Return the notebook scope a REST route requires (fail-closed default).

    ``path`` is the route's template (``/v1/notebooks/{notebook_id}/…``), not
    the request URL, so a notebook id can never match a table entry by accident.
    """
    method = method.upper()
    if method in ("GET", "HEAD") or (method, path) in _READ_POST_ROUTES:
        return NOTEBOOK_SCOPE_READ
    if (method, path) in _WRITE_ROUTES:
        return NOTEBOOK_SCOPE_WRITE
    return NOTEBOOK_SCOPE_EXECUTE

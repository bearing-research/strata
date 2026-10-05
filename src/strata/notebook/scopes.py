"""Which notebook scope each operation needs, for every way into a notebook.

One table serves the WebSocket frames and the REST routes, so a viewer is a
viewer however it reaches the server. Both default to ``notebook:execute``: an
unclassified operation is privileged until someone classifies it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata.auth import get_principal
from strata.notebook.protocol import MessageType
from strata.tenant import DEFAULT_TENANT_ID

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

NOTEBOOK_SCOPE_READ = "notebook:read"
NOTEBOOK_SCOPE_WRITE = "notebook:write"
NOTEBOOK_SCOPE_EXECUTE = "notebook:execute"

# --- WebSocket frames ---
# Each frame maps to the least scope covering what it can do; anything unlisted
# defaults to ``notebook:execute``.

# Read-only: observe state, compute previews. No mutation, no code runs.
_READ_FRAMES = frozenset(
    {
        MessageType.NOTEBOOK_SYNC,
        MessageType.CELL_FOCUS,
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

# Runs code or mutates the environment (the inspect REPL evals expressions,
# widget updates cascade, dependency changes invoke uv). Listed for
# documentation; the default is already ``notebook:execute``.
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
    }
)


def required_scope_for_frame(msg_type: str) -> str:
    """Return the notebook scope a C→S frame requires (fail-closed default)."""
    if msg_type in _READ_FRAMES:
        return NOTEBOOK_SCOPE_READ
    if msg_type in _WRITE_FRAMES:
        return NOTEBOOK_SCOPE_WRITE
    return NOTEBOOK_SCOPE_EXECUTE


# --- REST routes ---
# GETs and the two previews below only read. Write routes change committed
# content or config without running anything. Every other route defaults to
# ``notebook:execute`` (fail closed), including routes nobody has classified yet.
# Keys are (method, route path template).
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
        # The inverse of open, which needs write: a reader cannot end others' sessions.
        ("POST", "/v1/notebooks/{notebook_id}/close"),
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
        ("POST", "/v1/projects/{path:path}/quiesce"),
        ("POST", "/v1/projects/{path:path}/release"),
    }
)


def required_scope_for_route(method: str, path: str) -> str:
    """Return the notebook scope a REST route requires (fail-closed default).

    *path* is the route template (``/v1/notebooks/{notebook_id}/...``), not the
    request URL, so a notebook id can never match a table entry by accident.
    """
    method = method.upper()
    if method in ("GET", "HEAD") or (method, path) in _READ_POST_ROUTES:
        return NOTEBOOK_SCOPE_READ
    if (method, path) in _WRITE_ROUTES:
        return NOTEBOOK_SCOPE_WRITE
    return NOTEBOOK_SCOPE_EXECUTE


# --- MCP tools ---
# Same three scopes per tool; unclassified tools are execute. Publishing mints a
# public link, so it needs ``artifacts:publish`` like the REST publish route.
_READ_TOOLS = frozenset(
    {
        "list_notebooks",
        "get_notebook",
        "get_cell",
        # Writes only into ``.strata/outputs/`` and returns bytes a reader can already
        # see the metadata for, so it is a read.
        "save_cell_output",
        "get_variable",
        "dag",
        "status",
        "list_workers",
        "lineage",
        "publish_preflight",
    }
)

_WRITE_TOOLS = frozenset(
    {
        "add_cell",
        "edit_cell",
        "remove_cell",
        "move_cell",
        "note",
        "add_worker",
        "set_default_worker",
        "set_variant",
        "remove_worker",
        "disconnect_ssh_worker",
        "promote",
    }
)

_EXECUTE_TOOLS = frozenset(
    {
        "run_cell",
        "run_tests",
        "run_snippet",
        # Re-materializes the widget cell, so it is a run, gated like ``run_cell``.
        "set_widget_value",
        "add_dependency",
        "remove_dependency",
        "connect_ssh_worker",
    }
)

_OTHER_TOOL_SCOPES = {"publish": "artifacts:publish"}

CLASSIFIED_TOOLS = _READ_TOOLS | _WRITE_TOOLS | _EXECUTE_TOOLS | set(_OTHER_TOOL_SCOPES)
"""Every tool the table was written against; a test holds the MCP server to it."""


def required_scope_for_tool(name: str) -> str:
    """Return the scope an MCP tool requires (fail-closed default)."""
    if name in _READ_TOOLS:
        return NOTEBOOK_SCOPE_READ
    if name in _WRITE_TOOLS:
        return NOTEBOOK_SCOPE_WRITE
    return _OTHER_TOOL_SCOPES.get(name, NOTEBOOK_SCOPE_EXECUTE)


def session_visible_to_caller(session: NotebookSession) -> bool:
    """Whether the current caller's tenant may use *session*, as artifacts are scoped.

    A session carries the tenant that opened it. A caller or opener the proxy sent
    without a tenant is the default tenant, as it is for notebook storage; it is not
    everyone. No principal (personal mode) or ``admin:*`` sees every session.
    """
    principal = get_principal()
    if principal is None or principal.has_scope("admin:*"):
        return True
    tenant = session.opened_by[1] if session.opened_by else None
    return (tenant or DEFAULT_TENANT_ID) == (principal.tenant or DEFAULT_TENANT_ID)

"""MCP server: expose the live notebook session to an external coding agent.

Mounted at ``/mcp`` (streamable HTTP) when ``mcp_enabled`` is set. Wraps the
:class:`~strata.notebook.ops.NotebookOps` contract the ``strata`` CLI drives,
so an agent operates a warm session (its artifact cache and cell state) rather
than an offline copy.

Needs the ``[mcp]`` extra: :func:`build_mcp_app` returns ``None`` without it.
Tool logic lives in module-level ``_*`` functions that take a
``SessionManager``, testable without an MCP client; :func:`build_mcp_app`
registers thin wrappers whose docstrings are the agent-facing tool descriptions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strata.auth import get_principal, principal_context
from strata.notebook.ops import LocalNotebookOps, NotebookOpsError
from strata.notebook.scopes import required_scope_for_tool, session_visible_to_caller

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from strata.notebook.session import SessionManager
    from strata.types import Principal


logger = logging.getLogger(__name__)


def _resolve_ops(session_manager: SessionManager, session_id: str) -> LocalNotebookOps:
    """Wrap the server's warm session for *session_id* in ``LocalNotebookOps``.

    Raises ``ValueError`` (a tool error to the agent) when no such session is open:
    MCP operates on sessions the UI or CLI already opened, not arbitrary paths.
    """
    return LocalNotebookOps.from_session(_live_session(session_manager, session_id))


def _list_notebooks(session_manager: SessionManager) -> list[dict[str, Any]]:
    """Return one ``{session_id, name, path}`` entry per open session the caller may use."""
    notebooks: list[dict[str, Any]] = []
    for session_id in session_manager.list_sessions():
        session = session_manager.get_session(session_id)
        if session is None or not session_visible_to_caller(session):
            continue
        notebooks.append(
            {
                "session_id": session_id,
                "name": session.notebook_state.name,
                "path": str(session.path),
            }
        )
    return notebooks


def _get_notebook(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return every cell's curated view for an open session, in order."""
    ops = _resolve_ops(session_manager, session_id)
    return {"cells": [cell.model_dump(mode="json") for cell in ops.list_cells()]}


def _get_cell(session_manager: SessionManager, session_id: str, cell_id: str) -> dict[str, Any]:
    """Return one cell's curated view (source, status, outputs, …)."""
    return _resolve_ops(session_manager, session_id).get_cell(cell_id).model_dump(mode="json")


# So the handed-over file opens in the right tool. Unknown types keep ``.bin``.
_OUTPUT_EXTENSIONS = {
    "image/png": ".png",
    "text/markdown": ".md",
    "json/object": ".json",
    "arrow/ipc": ".arrow",
    "pickle/object": ".pickle",
}


def _save_cell_output(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    index: int = -1,
) -> dict[str, Any]:
    """Write one display output into the notebook and return where it landed.

    Always the notebook's own ``.strata/outputs/``, never a caller-named path, so
    the tool cannot become an arbitrary filesystem write.
    """
    from strata.notebook.ops import display_output_at

    session = _live_session(session_manager, session_id)
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise ValueError(f"no cell with id {cell_id!r}")
    output, resolved = display_output_at(cell, index)

    out_dir = session.path / ".strata" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _OUTPUT_EXTENSIONS.get(output.content_type or "", ".bin")
    dest = out_dir / f"{cell_id}-{resolved}{ext}"
    saved = _resolve_ops(session_manager, session_id).save_output(cell_id, dest, index=resolved)
    return saved.model_dump(mode="json")


def _dag(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return the dependency graph (edges, topological order, roots, leaves)."""
    return _resolve_ops(session_manager, session_id).dag().model_dump(mode="json")


def _get_variable(session_manager: SessionManager, session_id: str, name: str) -> dict[str, Any]:
    """Return the cell that defines *name*.

    When *name* is not defined, returns ``defined: False`` plus the available
    variable names, so a miss doubles as discovery.
    """
    ops = _resolve_ops(session_manager, session_id)
    producers = ops.dag().variable_producer
    producer = producers.get(name)
    if producer is None:
        return {"variable": name, "defined": False, "available": sorted(producers)}
    if producer.startswith(("sweep:", "fanout:")):
        # No single producing cell: list each instance and the name `lineage`
        # takes for it.
        from strata.notebook.dag import SweepProducer

        session = _live_session(session_manager, session_id)
        group = session.dag.variable_producer.get(name) if session.dag else None
        variants: list[dict[str, str]] = []
        if isinstance(group, SweepProducer):
            variants = [
                {
                    "variant": variant_name,
                    "cell_id": cell_id,
                    # Fan-out instances share a cell and differ by ``@variant=``;
                    # sweep members are separate cells storing the plain name.
                    "lineage_variable": (
                        f"{name}@variant={variant_name}" if group.fanout_cell is not None else name
                    ),
                }
                for variant_name, cell_id in group.variants
            ]
        return {"variable": name, "defined": True, "defined_in": producer, "variants": variants}
    # Let a get_cell failure propagate rather than report "defined".
    cell = ops.get_cell(producer).model_dump(mode="json")
    return {"variable": name, "defined": True, "defined_in": cell["id"], "cell": cell}


async def _set_variant(
    session_manager: SessionManager,
    session_id: str,
    group: str,
    active: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Switch a variant group's active variant and/or its mode, as the UI tab strip does.

    The group and variant names are checked against the ones cells declare: the
    writer would otherwise add a junk ``[[variant_group]]`` block to the committed
    notebook.toml for a typo, or report a variant the DAG is not using.
    """
    if active is None and mode is None:
        raise ValueError("Provide `active` and/or `mode`.")
    if mode is not None and mode not in ("switch", "sweep"):
        raise ValueError(f"unknown mode {mode!r} (switch|sweep)")

    session = _live_session(session_manager, session_id)
    cells = session.notebook_state.cells
    groups = sorted({c.variant_group for c in cells if c.variant_group})
    if group not in groups:
        raise ValueError(
            f"no variant group {group!r}. Declared: {', '.join(groups) or 'none'}. "
            "A group exists once a cell carries `# @variant <group> <name>`."
        )
    if active is not None:
        names = sorted(
            {c.variant_name for c in cells if c.variant_group == group and c.variant_name}
        )
        if active not in names:
            raise ValueError(
                f"group {group!r} has no variant {active!r}. Variants: {', '.join(names)}"
            )

    # Mode first, matching the route (`active` is ignored in sweep mode).
    if mode is not None:
        session.set_variant_mode(group, mode)
    if active is not None:
        session.set_variant_active(group, active)
    # Without the broadcast an attached viewer keeps stale tabs and badges.
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"variant {group} → {active or mode}")
    return {
        "variant_groups": [
            vg.model_dump(mode="json") for vg in session.notebook_state.variant_groups
        ],
        "cells": [
            cell.model_dump(mode="json")
            for cell in _resolve_ops(session_manager, session_id).list_cells()
        ],
    }


def _status(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return a compact per-cell status + staleness summary."""
    return _resolve_ops(session_manager, session_id).status().model_dump(mode="json")


async def _run_cell(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    mode: str = "normal",
) -> dict[str, Any]:
    """Execute a cell in the warm session, broadcasting live frames to spectators.

    Goes through ``execute_cell_and_broadcast`` (not the silent
    ``LocalNotebookOps.run_cell``) so an attached browser or TUI sees the run live.
    Returns the agent-facing ``RunResult`` view.
    """
    from strata.notebook.ops import NotebookOpsError, _run_result_from_wire
    from strata.notebook.ws import NotebookBusyError, execute_cell_exclusive

    if mode not in ("normal", "rerun", "force"):
        raise ValueError(f"unknown run mode {mode!r} (normal|rerun|force)")

    session = _live_session(session_manager, session_id)
    if session.notebook_state.get_cell(cell_id) is None:
        raise NotebookOpsError(f"no cell with id {cell_id!r}")

    block_reason = session.environment_execution_block_message()
    if block_reason:
        raise ValueError(block_reason)

    # Same reservation as the WS handlers, so an agent run can't race a browser.
    try:
        result = await execute_cell_exclusive(
            session,
            cell_id,
            session_id,
            mode=mode,  # type: ignore[arg-type]
        )
    except NotebookBusyError as exc:
        raise NotebookOpsError(f"{exc} Retry after the current run finishes.")
    if result is None:
        raise NotebookOpsError(f"cell {cell_id!r} could not be executed")
    run = _run_result_from_wire(result.to_dict()).model_dump(mode="json")
    await _agent_note(session_id, "mcp", f"ran cell {cell_id} → {run['status']}")
    return run


async def _set_widget_value(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Set a widget cell's controls and re-materialize it, as dragging the slider does.

    Persists the values, re-runs the widget in force mode, marks downstream stale,
    and for a ``# @live`` widget chains the cost-gated cascade. Widget selection is
    runtime state, so editing the cell source alone cannot change it.
    """
    from strata.notebook.models import CellLanguage
    from strata.notebook.ops import NotebookOpsError, _run_result_from_wire
    from strata.notebook.widget_analyzer import analyze_widget_cell, coerce_widget_values
    from strata.notebook.ws import NotebookBusyError, apply_widget_values, execute_cell_exclusive

    session = _live_session(session_manager, session_id)
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None:
        raise NotebookOpsError(f"no cell with id {cell_id!r}")
    if cell.language != CellLanguage.WIDGET:
        raise NotebookOpsError(f"cell {cell_id!r} is a {cell.language} cell, not a widget")

    descriptors = analyze_widget_cell(cell.source).descriptors
    coerced = coerce_widget_values(descriptors, values)
    if not coerced:
        declared = ", ".join(sorted(d.name for d in descriptors)) or "none"
        raise NotebookOpsError(
            f"no value in {sorted(values)} matches a control of cell {cell_id!r} "
            f"(declared: {declared})"
        )

    block_reason = session.environment_execution_block_message()
    if block_reason:
        raise ValueError(block_reason)

    try:
        result = await execute_cell_exclusive(
            session,
            cell_id,
            session_id,
            mode="force",
            # Same path as the WS handler: values written under the reservation,
            # and a `# @live` widget chains the same cost-gated cascade.
            operation=lambda execution_state: apply_widget_values(
                session, cell_id, coerced, execution_state, session_id
            ),
        )
    except NotebookBusyError as exc:
        raise NotebookOpsError(f"{exc} Retry after the current run finishes.")
    if result is None:
        raise NotebookOpsError(f"widget cell {cell_id!r} could not be re-materialized")

    # No reload: the shared path already broadcast everything, and a reload
    # from disk would label the cascade's fresh cells stale.
    await _agent_note(
        session_id,
        "mcp",
        "widget " + ", ".join(f"{name}={value}" for name, value in sorted(coerced.items())),
    )
    return {
        "cell_id": cell_id,
        "values": dict(cell.widget_values),
        "run": _run_result_from_wire(result.to_dict()).model_dump(mode="json"),
    }


async def _run_tests(
    session_manager: SessionManager, session_id: str, cell_id: str
) -> dict[str, Any]:
    """Run a cell's unit tests in the warm session and return per-test outcomes."""
    ops = _resolve_ops(session_manager, session_id)
    result = await ops.run_tests(cell_id)
    await _agent_note(
        session_id,
        "mcp",
        f"ran tests for {cell_id} → {result.passed} passed, {result.failed} failed",
    )
    return result.model_dump(mode="json")


async def _broadcast_notebook(session_id: str, session: Any) -> None:
    """Push a full ``notebook_state`` to the session's WS spectators after a mutation.

    The offline ``LocalNotebookOps`` verbs do not broadcast; this makes an agent's
    edits appear live in an attached browser or TUI.
    """
    from strata.notebook.ws import broadcast_notebook_sync

    await broadcast_notebook_sync(session_id, session)


async def _agent_note(session_id: str, source: str, text: str) -> None:
    """Surface a one-line ``agent_note`` frame in an attached terminal viewer's Agent tab.

    ``source`` is ``"mcp"`` for narrated tool actions and ``"agent"`` for explicit
    notes. A no-op when nothing is attached.
    """
    from strata.notebook.protocol import MessageType
    from strata.notebook.ws import _broadcast_message, _make_message, next_notebook_sequence

    await _broadcast_message(
        session_id,
        # A real sequence: a hard-coded 0 reads as a gap to a client.
        _make_message(
            MessageType.AGENT_NOTE,
            next_notebook_sequence(session_id),
            {"source": source, "text": text},
        ),
    )


def _live_session(session_manager: SessionManager, session_id: str):
    """Return the server's warm session for *session_id*, or raise ``ValueError``."""
    session = session_manager.get_session(session_id)
    if session is None or not session_visible_to_caller(session):
        raise ValueError(f"no open notebook session {session_id!r}; call list_notebooks first")
    # An agent's tool call is someone working in the notebook.
    session.touch()
    return session


async def _sync_and_broadcast(session_id: str, session: Any) -> None:
    """Broadcast the live session after a file mutation.

    The mutation already reloaded it: ``LocalNotebookOps.from_session`` verbs and the
    session's variant setters ``reload()`` the server's live session in place, and a
    second reload would recompute staleness again on the event loop.
    """
    await _broadcast_notebook(session_id, session)


async def _add_cell(
    session_manager: SessionManager,
    session_id: str,
    source: str,
    after: str | None = None,
    language: str = "python",
    author: str | None = None,
) -> dict[str, Any]:
    """Add a new cell (backend-minted id), then sync + broadcast the new state."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session, author=author).add_cell(
        source, after=after, language=language
    )
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"added {language} cell {view.id}")
    return view.model_dump(mode="json")


async def _run_snippet(
    session_manager: SessionManager,
    session_id: str,
    source: str,
    after: str | None = None,
    language: str = "python",
    author: str | None = None,
) -> dict[str, Any]:
    """Add a cell and run it immediately (the one-call scratchpad primitive).

    Returns the new cell view with the run result under ``run``.
    """
    view = await _add_cell(session_manager, session_id, source, after, language, author)
    run = await _run_cell(session_manager, session_id, view["id"], "normal")
    # The _add_cell view is pre-run; re-fetch to include rendered outputs.
    view = _get_cell(session_manager, session_id, view["id"])
    view["run"] = run
    return view


async def _edit_cell(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    source: str,
    author: str | None = None,
) -> dict[str, Any]:
    """Replace a cell's source, then sync + broadcast the new state."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session, author=author).edit_cell(cell_id, source)
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"edited cell {cell_id}")
    return view.model_dump(mode="json")


async def _remove_cell(
    session_manager: SessionManager, session_id: str, cell_id: str
) -> dict[str, Any]:
    """Delete a cell (and its source / test files), then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    LocalNotebookOps.from_session(session).remove_cell(cell_id)
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"removed cell {cell_id}")
    return {"removed": cell_id}


async def _move_cell(
    session_manager: SessionManager, session_id: str, cell_id: str, index: int
) -> dict[str, Any]:
    """Move a cell to ``index`` in notebook order, then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    cells = LocalNotebookOps.from_session(session).move_cell(cell_id, index)
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"moved cell {cell_id} to position {index}")
    return {"cells": [cell.model_dump(mode="json") for cell in cells]}


async def _add_dependency(
    session_manager: SessionManager, session_id: str, package: str
) -> dict[str, Any]:
    """Add a Python dependency (``uv add``) to the warm session, then broadcast.

    ``mutate_dependency`` updates the live session in place, so no reload is needed.
    """
    session = _live_session(session_manager, session_id)
    result = await LocalNotebookOps.from_session(session).add_dependency(package)
    await _broadcast_notebook(session_id, session)
    await _agent_note(session_id, "mcp", f"added dependency {package}")
    return result.model_dump(mode="json")


async def _remove_dependency(
    session_manager: SessionManager, session_id: str, package: str
) -> dict[str, Any]:
    """Remove a Python dependency (``uv remove``) from the warm session, then broadcast."""
    session = _live_session(session_manager, session_id)
    result = await LocalNotebookOps.from_session(session).remove_dependency(package)
    await _broadcast_notebook(session_id, session)
    await _agent_note(session_id, "mcp", f"removed dependency {package}")
    return result.model_dump(mode="json")


async def _note(session_manager: SessionManager, session_id: str, message: str) -> dict[str, Any]:
    """Push an explicit narration line into the terminal viewer's Agent tab."""
    _live_session(session_manager, session_id)  # validate the session exists
    await _agent_note(session_id, "agent", message)
    return {"ok": True}


def _list_workers(session_manager: SessionManager, session_id: str) -> dict[str, Any]:
    """Return the notebook's registered workers + the default (read-only)."""
    return _resolve_ops(session_manager, session_id).list_workers().model_dump(mode="json")


async def _add_worker(
    session_manager: SessionManager,
    session_id: str,
    name: str,
    url: str,
    transport: str = "direct",
    runtime_id: str | None = None,
    token_env: str | None = None,
    set_default: bool = False,
) -> dict[str, Any]:
    """Register an executor worker in the notebook, then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session).add_worker(
        name,
        url=url,
        transport=transport,
        runtime_id=runtime_id,
        token_env=token_env,
        set_default=set_default,
    )
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"registered worker {name} → {url}")
    return view.model_dump(mode="json")


async def _remove_worker(
    session_manager: SessionManager, session_id: str, name: str
) -> dict[str, Any]:
    """Remove a notebook-scoped worker, then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session).remove_worker(name)
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"removed worker {name}")
    return view.model_dump(mode="json")


async def _set_default_worker(
    session_manager: SessionManager, session_id: str, name: str | None
) -> dict[str, Any]:
    """Set (or clear) the notebook default worker, then sync + broadcast."""
    session = _live_session(session_manager, session_id)
    view = LocalNotebookOps.from_session(session).set_default_worker(name)
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"default worker → {name or 'local'}")
    return view.model_dump(mode="json")


async def _connect_ssh_worker(
    session_manager: SessionManager,
    session_id: str,
    ssh_target: str,
    name: str | None = None,
    set_default: bool = True,
    install: bool = True,
) -> dict[str, Any]:
    """Provision, tunnel and register a remote worker over SSH, then broadcast.

    The tunnel must live in this process, so the server-owned supervisor runs in a
    thread. Returns the tunnel record plus the updated worker list.
    """
    import asyncio
    from dataclasses import asdict

    from strata.notebook.routes import get_worker_supervisor
    from strata.notebook.ssh_worker_service import establish_ssh_worker

    session = _live_session(session_manager, session_id)
    record = await asyncio.to_thread(
        establish_ssh_worker,
        session,
        get_worker_supervisor(),
        ssh_target=ssh_target,
        name=name,
        install=install,
        set_default=set_default,
    )
    await _sync_and_broadcast(session_id, session)
    await _agent_note(
        session_id, "mcp", f"connected ssh worker {record.name} → {record.ssh_target}"
    )
    return {"worker": asdict(record), **_list_workers(session_manager, session_id)}


async def _disconnect_ssh_worker(
    session_manager: SessionManager,
    session_id: str,
    name: str,
    stop_remote: bool = False,
) -> dict[str, Any]:
    """Close a worker's SSH tunnel and remove its registration, then broadcast."""
    import asyncio

    from strata.notebook.routes import get_worker_supervisor
    from strata.notebook.ssh_worker_service import teardown_ssh_worker

    session = _live_session(session_manager, session_id)
    existed = await asyncio.to_thread(
        teardown_ssh_worker, session, get_worker_supervisor(), name, stop_remote=stop_remote
    )
    await _sync_and_broadcast(session_id, session)
    await _agent_note(session_id, "mcp", f"disconnected ssh worker {name}")
    return {"torn_down": existed, **_list_workers(session_manager, session_id)}


def _cell_output(session_manager: SessionManager, session_id: str, cell_id: str, variable: str):
    """The artifact a cell stored for one of its variables.

    Only variables a downstream cell reads become artifacts, so the error for a
    missing one says that too: it is the case an agent can act on.
    """
    session = _live_session(session_manager, session_id)
    manager = session.get_artifact_manager()
    for name, artifact in manager.list_cell_artifacts(cell_id):
        if name == variable:
            return manager.artifact_store, artifact
    # ``__console__`` / ``__display__N`` are the cell's records, not variables it defined.
    stored = sorted(
        name for name, _ in manager.list_cell_artifacts(cell_id) if not name.startswith("__")
    )
    raise ValueError(
        f"Cell {cell_id} has no stored output named {variable!r}. "
        f"Stored: {', '.join(stored) or 'none'}. Only variables a downstream "
        f"cell reads are kept as artifacts."
    )


def _chain(store, artifact, max_depth: int = 10) -> list[dict[str, Any]]:
    """Every step behind an artifact, newest first, as the page would show it."""
    from strata.services.artifact import ArtifactService

    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=artifact.id,
        version=artifact.version,
        tenant_filter=None,
        max_depth=max_depth,
    )
    return [
        {
            "uri": node.uri,
            "type": node.type,
            "artifact_id": node.artifact_id,
            "version": node.version,
            "transform": node.transform_ref,
            "source": node.source,
            "build_env": node.build_env,
            "principal": node.principal,
            "content_sha256": node.content_sha256,
        }
        for node in lineage.nodes
    ]


def _lineage(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    variable: str,
    max_depth: int = 10,
) -> dict[str, Any]:
    """The chain behind one of a cell's outputs."""
    store, artifact = _cell_output(session_manager, session_id, cell_id, variable)
    return {
        "artifact_id": artifact.id,
        "version": artifact.version,
        "provenance_hash": artifact.provenance_hash,
        "content_sha256": artifact.content_sha256,
        "steps": _chain(store, artifact, max_depth),
    }


def _promote(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    variable: str,
    name: str,
    alias: str | None = None,
    tags: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Copy a cell's output and its chain to the team store, and name it there."""
    from strata.artifact_transfer import RemoteStore, promote_artifact

    config = _server_config()
    base_url = getattr(config, "notebook_remote_store_url", None)
    if not base_url:
        raise ValueError(
            "No team store is configured, so there is nowhere to promote to. "
            "Set notebook_remote_store_url on the server."
        )

    store, artifact = _cell_output(session_manager, session_id, cell_id, variable)
    from strata.auth import remote_store_headers

    target = RemoteStore(str(base_url), remote_store_headers(config))
    promotion = promote_artifact(
        store, target, artifact, name=name, alias=alias, tags=dict(tags or {})
    )
    return {
        "status": "pending" if promotion.alias_pending else "applied",
        "name": promotion.name,
        "artifact_uri": f"strata://artifact/{promotion.ref}",
        "copied": promotion.copied,
        "alias": promotion.alias,
        "store": str(base_url),
    }


def _publish_preflight(
    session_manager: SessionManager, session_id: str, cell_id: str, variable: str
) -> dict[str, Any]:
    """What publishing this output would put behind a link anyone can open.

    The whole chain travels: every upstream step's code and environment become
    readable by anyone with the URL.
    """
    store, artifact = _cell_output(session_manager, session_id, cell_id, variable)
    steps = _chain(store, artifact)
    return {
        "artifact_id": artifact.id,
        "version": artifact.version,
        "exposes": steps,
        "step_count": len(steps),
        "reads_source_of": [s["artifact_id"] for s in steps if s.get("source")],
        "note": (
            "Publishing mints a URL that needs no credentials. Every step "
            "listed here becomes readable through it, including the code each "
            "one ran."
        ),
    }


def _publish(
    session_manager: SessionManager,
    session_id: str,
    cell_id: str,
    variable: str,
    title: str | None = None,
) -> dict[str, Any]:
    """Mint the public link, after copying the chain to the store that serves it."""
    from strata.artifact_transfer import copy_chain

    store, artifact = _cell_output(session_manager, session_id, cell_id, variable)
    served = _served_store()
    published_id, published_version = artifact.id, artifact.version
    copied = 0
    if served is not None and served.db_path != store.db_path:
        # The link resolves from the served store, not the notebook's own;
        # without the copy the token lands where the page route never reads.
        written, landed = copy_chain(store, served, artifact, 10)
        copied = len(written)
        published_id, _, landed_version = landed.partition("@v=")
        published_version = int(landed_version)
    else:
        served = store

    # Stamp the caller for the audit row, as REST and CLI do. The tenant is
    # the artifact's own: the store refuses any other.
    caller = get_principal()
    publication = served.publish_artifact(
        published_id,
        published_version,
        title=title,
        published_by=caller.id if caller is not None else None,
    )
    return {
        # Empty when the version was already published: the store keeps only the
        # token's hash, so the link is shown once, when it is minted.
        "token": publication.token,
        "id": publication.id,
        "artifact_uri": f"strata://artifact/{published_id}@v={published_version}",
        "title": publication.title,
        "content_sha256": publication.content_sha256,
        "copied": copied,
    }


def _server_config():
    """The running server's config."""
    from strata.server import get_state

    return get_state().config


def _served_store():
    """The store a published link resolves from, or ``None`` if none is set.

    The server's own instance, with its configured metadata DSN and blob backend.
    """
    from strata.artifact_store import get_artifact_store

    return get_artifact_store()


def _caller(context: Any) -> Principal | None:
    """The authenticated principal behind one MCP request, or ``None`` without auth.

    Parsed as the HTTP auth middleware does; the middleware has already refused a
    request without valid credentials.
    """
    from strata.auth import AuthError, parse_api_key_principal, parse_principal, verify_proxy_token

    config = _server_config()
    if not getattr(config, "principal_auth_enabled", False):
        return None
    if context is None:
        return None
    try:
        request = context.request_context.request
    except ValueError:
        # mcp 2 raises this outside a request (a tool called in-process).
        return None
    if request is None:
        return None
    headers = dict(request.headers)
    # Re-check the proxy token: a mount served another way would otherwise
    # trust the identity headers.
    if config.auth_mode != "api_key" and not verify_proxy_token(
        request.headers.get(config.proxy_token_header), config.proxy_token
    ):
        return None
    try:
        if config.auth_mode == "api_key":
            return parse_api_key_principal(headers, config)
        return parse_principal(headers, config)
    except AuthError:
        return None


def _mcp_import_failure(missing: str | None) -> str:
    """Why an installed mcp could not give the notebook its MCPServer.

    Either mcp is older than 2 (before FastMCP became MCPServer), or it is mcp 2
    missing its own dependency *missing*.
    """
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    try:
        installed = version("mcp")
    except PackageNotFoundError:
        # A bare source tree on sys.path has no distribution metadata.
        installed = "(version unknown)"
    else:
        if Version(installed).major < 2:
            return (
                f"mcp {installed} is installed, but the notebook's /mcp endpoint needs "
                "mcp>=2 (install strata-notebook[mcp]); /mcp is off"
            )
    return (
        f"mcp {installed} is installed, but importing it failed: no module named "
        f"{missing!r} (reinstall strata-notebook[mcp]); /mcp is off"
    )


def build_mcp_app(session_manager: SessionManager) -> Starlette | None:
    """Build the streamable-HTTP MCP ASGI app, or ``None`` if ``[mcp]`` is absent.

    Mount it at ``/mcp``; the host app's lifespan must enter its lifespan (see
    ``server.py``).

    Parameters
    ----------
    session_manager : SessionManager
        The server's live session registry; tools resolve warm sessions from it.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            # Another tool may need this mcp version: warn, don't refuse to start.
            logger.warning(_mcp_import_failure(exc.name))
        return None

    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    class AuthorizingMCPServer(MCPServer):
        """Every tool call runs as the caller that made it, within its scopes.

        The caller is read from the tool call's own HTTP request, not the task serving
        the MCP session: one session's requests can carry different credentials. The
        principal is current for the call, so ``get_principal`` sees the caller. Scopes
        come from the same table the REST routes and WebSocket frames use.
        """

        async def call_tool(
            self, name: str, arguments: dict[str, Any], context: Any | None = None
        ) -> Any:
            principal = _caller(context)
            required = required_scope_for_tool(name)
            if getattr(_server_config(), "principal_auth_enabled", False) and (
                principal is None or not principal.has_scope(required)
            ):
                raise ToolError(f"'{name}' requires the {required} scope")
            with principal_context(principal):
                try:
                    return await super().call_tool(name, arguments, context)
                except UnexpectedToolError as exc:
                    # The tools raise these for the agent to read; anything else stays generic.
                    if isinstance(exc.__cause__, ValueError | NotebookOpsError):
                        raise ToolError(f"{exc}: {exc.__cause__}") from exc.__cause__
                    raise

    mcp = AuthorizingMCPServer("strata-notebook")

    @mcp.tool()
    def list_notebooks() -> list[dict[str, Any]]:
        """List the notebook sessions currently open on the Strata server.

        Returns one entry per session with its ``session_id`` (use it as the
        ``session_id`` argument to the other tools), human-readable ``name``,
        and on-disk ``path``. Sessions are opened by the notebook UI or the
        ``strata`` CLI; this tool does not open them.
        """
        return _list_notebooks(session_manager)

    @mcp.tool()
    def get_notebook(session_id: str) -> dict[str, Any]:
        """Return every cell of an open notebook session, in notebook order.

        Each cell view includes its id, source, language, status, defines /
        references, and rendered outputs.
        """
        return _get_notebook(session_manager, session_id)

    @mcp.tool()
    def get_cell(session_id: str, cell_id: str) -> dict[str, Any]:
        """Return one cell's full curated view: source, status, and outputs.

        Errors if the notebook session or the cell id does not exist.
        """
        return _get_cell(session_manager, session_id, cell_id)

    @mcp.tool()
    def save_cell_output(session_id: str, cell_id: str, index: int = -1) -> dict[str, Any]:
        """Write a cell's display output to a file and return its path.

        Use this to *look at* a plot or an image. `get_cell` tells you a cell
        produced an `image/png` and how big it is, but an image has no useful
        text preview, so reading the file is the only way to see it. Open the
        returned `path` with your own file-reading tool.

        `index` picks which display output in emission order; the default
        `-1` is the last one, which is what a cell ending in an expression
        produced. The file lands in the notebook's own `.strata/outputs/`
        and is overwritten on each call.
        """
        return _save_cell_output(session_manager, session_id, cell_id, index)

    @mcp.tool()
    async def set_variant(
        session_id: str,
        group: str,
        active: str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Choose which variant of a group runs, or sweep them all.

        `active` picks one variant by name (switch mode). `mode` is `switch`
        (one at a time) or `sweep` (every variant runs, and a downstream cell
        reading the group receives a `{variant: value}` dict). Send either or
        both. Returns the group's state and the notebook's cells, whose
        staleness has been recomputed against the new selection.
        """
        return await _set_variant(session_manager, session_id, group, active, mode)

    @mcp.tool()
    def get_variable(session_id: str, name: str) -> dict[str, Any]:
        """Look up the cell that defines a variable: "do I already have `name`?"

        Use this before recomputing something: if `name` already exists, reference
        it in a new cell instead of rebuilding it. Returns the defining cell
        (source, status, outputs) under ``cell`` with ``defined: true``; if it
        isn't defined, ``defined: false`` plus the ``available`` variable names.
        """
        return _get_variable(session_manager, session_id, name)

    @mcp.tool()
    def dag(session_id: str) -> dict[str, Any]:
        """Return the notebook's dependency graph.

        Includes variable-level ``edges``, a ``topological_order``, and the
        ``roots`` / ``leaves``, which show how cell outputs feed downstream cells.
        """
        return _dag(session_manager, session_id)

    @mcp.tool()
    def status(session_id: str) -> dict[str, Any]:
        """Return a compact per-cell status + staleness summary for a session.

        Use it to see which cells are ready, stale (and why), or idle before
        deciding what to run.
        """
        return _status(session_manager, session_id)

    @mcp.tool()
    async def run_cell(session_id: str, cell_id: str, mode: str = "normal") -> dict[str, Any]:
        """Execute a cell in a warm session and return the run outcome.

        ``mode`` is one of: ``normal`` (use the cache, re-run stale upstreams
        first), ``rerun`` (bypass this cell's cache, still refresh upstreams), or
        ``force`` ("run this only": run against whatever upstream artifacts
        already exist). The run is broadcast live, so a browser or terminal
        viewer attached to the session watches it happen. Returns status
        (ok / error), cache hit, duration, and captured stdout / stderr; call
        get_cell afterwards for the rendered outputs.
        """
        return await _run_cell(session_manager, session_id, cell_id, mode)

    @mcp.tool()
    async def set_widget_value(
        session_id: str, cell_id: str, values: dict[str, Any]
    ) -> dict[str, Any]:
        """Set a widget cell's controls and re-run it at the new values.

        ``values`` maps control name to value, as ``get_cell`` reports them
        under ``controls``; send only the ones you are changing. The widget
        re-materializes and everything downstream goes stale, exactly as when a
        person moves the slider, and anyone watching the session sees it.
        Returns the controls' new values and the widget's run outcome.
        """
        return await _set_widget_value(session_manager, session_id, cell_id, values)

    @mcp.tool()
    async def run_tests(session_id: str, cell_id: str) -> dict[str, Any]:
        """Run a Python cell's unit tests (``cells/{cell_id}.test.py``).

        Returns pass / fail / error / skip counts and per-test cases. Errors if
        the cell has no test file.
        """
        return await _run_tests(session_manager, session_id, cell_id)

    @mcp.tool()
    async def add_cell(
        session_id: str,
        source: str,
        after: str | None = None,
        language: str = "python",
        author: str | None = None,
    ) -> dict[str, Any]:
        """Add a new cell and return it (the server mints the cell id).

        ``after`` inserts the cell after that cell id (omit to append at the
        end). ``language`` is one of python, markdown, sql, r, prompt. The new
        cell appears live in any attached viewer.

        ``author`` names you on the cell, so a person opening the notebook can
        tell which cells an agent wrote. Send the same value on every call:
        your own name or id. On a server that authenticates its callers the
        authenticated identity is used instead and this is ignored.
        """
        return await _add_cell(session_manager, session_id, source, after, language, author)

    @mcp.tool()
    async def run_snippet(
        session_id: str,
        source: str,
        after: str | None = None,
        language: str = "python",
        author: str | None = None,
    ) -> dict[str, Any]:
        """Add a cell and run it in one call: the scratchpad primitive.

        Prefer this over add_cell + run_cell for quick exploration: it mints a
        cell, executes it (normal mode: a cache hit if the source + inputs are
        unchanged), and returns the new cell view with the run outcome (status,
        cache_hit, stdout, stderr) nested under ``run``. The run appears live in
        any attached viewer. Use ``add_cell`` without a run only when you want to
        stage a cell without executing it.

        ``author`` names you on the cell, so a person opening the notebook can
        tell which cells an agent wrote. Send the same value on every call:
        your own name or id. On a server that authenticates its callers the
        authenticated identity is used instead and this is ignored.
        """
        return await _run_snippet(session_manager, session_id, source, after, language, author)

    @mcp.tool()
    async def edit_cell(
        session_id: str, cell_id: str, source: str, author: str | None = None
    ) -> dict[str, Any]:
        """Replace a cell's source and return the updated cell.

        Downstream cells that consumed the old output become stale; use status /
        run_cell to re-materialize them.

        ``author`` names you on the cell, so a person opening the notebook can
        tell which cells an agent wrote. Send the same value on every call:
        your own name or id. On a server that authenticates its callers the
        authenticated identity is used instead and this is ignored.
        """
        return await _edit_cell(session_manager, session_id, cell_id, source, author)

    @mcp.tool()
    async def remove_cell(session_id: str, cell_id: str) -> dict[str, Any]:
        """Delete a cell and its source / test files. Returns the removed id."""
        return await _remove_cell(session_manager, session_id, cell_id)

    @mcp.tool()
    async def move_cell(session_id: str, cell_id: str, index: int) -> dict[str, Any]:
        """Move a cell to a new 0-based position and return the new cell order."""
        return await _move_cell(session_manager, session_id, cell_id, index)

    @mcp.tool()
    async def add_dependency(session_id: str, package: str) -> dict[str, Any]:
        """Add a Python dependency to the notebook (runs ``uv add``).

        ``package`` is a uv/pip requirement (e.g. ``polars`` or ``polars>=1.0``).
        Returns whether the lockfile changed; cells that import the package can
        then be run. May take a few seconds while the environment resolves.
        """
        return await _add_dependency(session_manager, session_id, package)

    @mcp.tool()
    async def remove_dependency(session_id: str, package: str) -> dict[str, Any]:
        """Remove a Python dependency from the notebook (runs ``uv remove``)."""
        return await _remove_dependency(session_manager, session_id, package)

    @mcp.tool()
    async def note(session_id: str, message: str) -> dict[str, Any]:
        """Post a short note into the terminal viewer's Agent tab for the human watching.

        Your own actions (running / editing cells, etc.) already appear there
        automatically. Use this to narrate your reasoning or plan, e.g. "About
        to refactor featurize into two cells", so the person watching the
        notebook in the terminal viewer (strata-notebook-tui) can follow what
        you're doing.
        """
        return await _note(session_manager, session_id, message)

    @mcp.tool()
    def list_workers(session_id: str) -> dict[str, Any]:
        """List the notebook's registered workers and which is the default.

        Each worker has a ``name`` (use it as ``# @worker <name>`` in a cell, or
        as the notebook default), its ``backend`` / ``transport`` / ``url``, and
        ``is_default``. The built-in ``local`` worker is always present. Use this
        before routing a cell to confirm a worker exists.
        """
        return _list_workers(session_manager, session_id)

    @mcp.tool()
    async def add_worker(
        session_id: str,
        name: str,
        url: str,
        transport: str = "direct",
        runtime_id: str | None = None,
        token_env: str | None = None,
        set_default: bool = False,
    ) -> dict[str, Any]:
        """Register a remote executor worker so cells can run on it.

        ``url`` is the worker's ``/v1/execute`` endpoint (e.g.
        ``http://127.0.0.1:9000/v1/execute``); ``transport`` is ``direct`` for a
        plain HTTP worker. Set ``token_env`` to the name of an env var holding the
        worker's bearer token, ``runtime_id`` to a stable fingerprint of the
        worker's environment (so its cached results don't collide with local
        runs), and ``set_default=True`` to route every cell there by default.
        Re-adding an existing name replaces it. Then route a cell with
        ``# @worker <name>`` (or rely on the default). Returns the updated worker
        list.
        """
        return await _add_worker(
            session_manager,
            session_id,
            name,
            url,
            transport,
            runtime_id,
            token_env,
            set_default,
        )

    @mcp.tool()
    async def set_default_worker(session_id: str, name: str | None = None) -> dict[str, Any]:
        """Set the notebook's default worker (``name=None`` or ``"local"`` clears it).

        Cells without a ``# @worker`` annotation run on the default. Returns the
        updated worker list.
        """
        return await _set_default_worker(session_manager, session_id, name)

    @mcp.tool()
    async def remove_worker(session_id: str, name: str) -> dict[str, Any]:
        """Remove a notebook-scoped worker (clears the default if it named it)."""
        return await _remove_worker(session_manager, session_id, name)

    @mcp.tool()
    async def connect_ssh_worker(
        session_id: str,
        ssh_target: str,
        name: str | None = None,
        set_default: bool = True,
        install: bool = True,
    ) -> dict[str, Any]:
        """Run cells on a remote box the user gave you SSH access to.

        When the user hands you an SSH target for compute (e.g. a GPU box),
        call this: it connects to ``ssh_target`` (``user@host`` or an
        ~/.ssh/config alias), installs strata-worker there if missing, launches
        it, opens a secure tunnel, and registers it as a worker, so cells then
        run on that box, cached by provenance like everything else. With
        ``set_default`` (the default) every cell runs there; give a specific cell
        ``# @worker local`` to keep it on this machine. A remote cell runs on the
        *box's* filesystem, so its mounts and absolute paths are the box's, not
        yours. Needs key-based SSH (no password prompts). May take a while on
        first connect while it installs. Returns the tunnel record + worker list.
        """
        return await _connect_ssh_worker(
            session_manager, session_id, ssh_target, name, set_default, install
        )

    @mcp.tool()
    async def disconnect_ssh_worker(
        session_id: str, name: str, stop_remote: bool = False
    ) -> dict[str, Any]:
        """Close a remote SSH worker's tunnel and unregister it.

        With ``stop_remote`` also stops the strata-worker process on the box; by
        default it's left running so a later reconnect is fast.
        """
        return await _disconnect_ssh_worker(session_manager, session_id, name, stop_remote)

    @mcp.tool()
    def lineage(
        session_id: str, cell_id: str, variable: str, max_depth: int = 10
    ) -> dict[str, Any]:
        """Read the chain behind one of a cell's outputs.

        Every step that produced it, newest first, each with the code it ran,
        the environment it ran in, who computed it and the digest of its bytes.
        ``variable`` is the name the cell assigned; only variables a downstream
        cell reads are stored, and the error lists what is.
        """
        return _lineage(session_manager, session_id, cell_id, variable, max_depth)

    @mcp.tool()
    def promote(
        session_id: str,
        cell_id: str,
        variable: str,
        name: str,
        alias: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Share a cell's output with the team, under a name they can ask for.

        Copies the artifact and everything behind it into the team's store.
        The chain travels because that store's cache is keyed by provenance, so
        each ancestor that arrives saves the next person the same computation.
        This mints no public link; use ``publish`` for that.

        ``alias`` (e.g. champion) is moved to this version; a protected alias
        comes back as ``pending`` for someone else to approve. Needs a team
        store configured on the server.
        """
        return _promote(session_manager, session_id, cell_id, variable, name, alias, tags)

    @mcp.tool()
    def publish_preflight(session_id: str, cell_id: str, variable: str) -> dict[str, Any]:
        """See what publishing an output would expose, before publishing it.

        Publishing mints a URL that needs no credentials, and the whole chain
        goes with it: every upstream step's code and environment become
        readable by anyone with the link. Show this list to the user and get
        their agreement before calling ``publish``.
        """
        return _publish_preflight(session_manager, session_id, cell_id, variable)

    @mcp.tool()
    def publish(
        session_id: str, cell_id: str, variable: str, title: str | None = None
    ) -> dict[str, Any]:
        """Mint a public link to a cell's output. Ask the user first.

        This is not undoable by you: the URL needs no credentials and exposes
        the whole chain behind the artifact. Call ``publish_preflight`` and put
        what it returns in front of the user before calling this. The token is
        returned only when the link is minted; for a version already published it
        is empty. Withdrawing is ``strata artifact unpublish <token or id>``.
        """
        return _publish(session_manager, session_id, cell_id, variable, title)

    # The default path would nest the endpoint at "/mcp/mcp".
    app = mcp.streamable_http_app(streamable_http_path="/")
    # Lets tests read the tool list without an MCP client.
    app.state.mcp_server = mcp
    return app

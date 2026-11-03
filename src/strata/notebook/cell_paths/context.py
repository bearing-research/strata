"""What an extracted cell path shares with the rest of the executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.notebook.execution_result import CellExecutionResult
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost
    from strata.notebook.models import NotebookState


@dataclass(frozen=True)
class PathContext:
    """Only what a moved path uses, added as each path moves."""

    # The standard provenance of a cell, from its id and source.
    compute_provenance: Callable[[str, str], Awaitable[_CellProvenance]]
    # What the cell's run moved: by mount name, and by table URI.
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    # Run the cell's upstreams, from its id, start time and execution method: their failure
    # as this cell's result, or ``None``.
    materialize_upstreams_or_failure: Callable[
        [str, float, str], Awaitable[CellExecutionResult | None]
    ]
    # Store displays that came as text, not files; what was stored, with its artifact URIs.
    store_inline_display_outputs: Callable[..., list[dict[str, Any]]]


def no_interpreter_message(session: ExecutorHost) -> str:
    """Why cell code did not run: the session has no interpreter.

    PATH ``python`` is never used instead: its result would be stored under
    this notebook's provenance.
    """
    return session.environment_execution_block_message() or (
        "Notebook environment is not ready: it has no Python interpreter."
    )


def effective_worker_name(
    notebook_state: NotebookState, cell_id: str, annotation_worker: str | None
) -> str:
    """The worker a run of the cell targets: annotation, then cell, then notebook, then local."""
    if annotation_worker:
        return annotation_worker
    cell = notebook_state.get_cell(cell_id)
    if cell and cell.worker:
        return cell.worker
    return notebook_state.worker or "local"


def artifact_content_type(artifact: Any) -> str:
    """Read an artifact's stored content_type from its transform_spec params."""
    ct = artifact.params.get("content_type")
    return str(ct) if isinstance(ct, str) and ct else "pickle/object"

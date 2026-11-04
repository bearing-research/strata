"""What an extracted cell path shares with the rest of the executor."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mounts import MountWrite

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost
    from strata.notebook.harness_user import LocalExecutionRefused
    from strata.notebook.models import NotebookState

logger = logging.getLogger(__name__)


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


def _writes_entry(writes: list[MountWrite]) -> dict[str, Any] | None:
    """The writes index entry for a good run that wrote *writes*; None when it wrote nothing."""
    from strata.notebook.artifact_integration import caller_principal_id

    if not writes:
        return None
    return {
        "ran_at": int(time.time() * 1000),
        "principal": caller_principal_id(),
        "mounts": [write.model_dump() for write in writes],
    }


class RemoteExecutionError(RuntimeError):
    """Execution failure with structured remote metadata for notebook UX."""

    def __init__(
        self,
        message: str,
        *,
        remote_build_state: str | None = None,
        remote_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.remote_build_state = remote_build_state
        self.remote_error_code = remote_error_code


def _store_failure_result(
    result: CellExecutionResult, store_error: str | None
) -> CellExecutionResult:
    """A single run that succeeded and could not store its outputs."""
    return CellExecutionResult(
        cell_id=result.cell_id,
        success=False,
        stdout=result.stdout,
        stderr=result.stderr,
        outputs=result.outputs,
        duration_ms=result.duration_ms,
        error=store_error
        or "Cell executed successfully but failed to store output artifacts. Check server logs.",
        execution_method=result.execution_method,
    )


def _mount_sync_failure_result(result: CellExecutionResult, exc: Exception) -> CellExecutionResult:
    """A single run that succeeded and could not sync its read-write mounts back."""
    return CellExecutionResult(
        cell_id=result.cell_id,
        success=False,
        stdout=result.stdout,
        stderr=result.stderr,
        outputs=result.outputs,
        duration_ms=result.duration_ms,
        error=f"Cell executed successfully but failed to sync read-write mounts: {exc}",
        execution_method=result.execution_method,
        mutation_warnings=result.mutation_warnings,
    )


def _worker_mount_writes(result: Any, mount_specs: list[MountSpec]) -> list[MountWrite]:
    """What a remote worker reports it wrote, for the cell's own read-write mounts only.

    The report comes from another machine, perhaps another Strata version: an entry that
    does not parse, or names a mount this cell does not have, is dropped.
    """
    raw = result.get("mount_writes") if isinstance(result, dict) else None
    if not isinstance(raw, list):
        return []
    rw = {m.name: m.uri for m in mount_specs if m.mode == MountMode.READ_WRITE}
    writes: list[MountWrite] = []
    for entry in raw:
        try:
            write = MountWrite.model_validate(entry)
        except ValidationError:
            logger.warning("Ignoring a malformed mount write report from a worker")
            continue
        if rw.get(write.mount) == write.uri:
            writes.append(write)
    return writes


def _refused_result(exc: LocalExecutionRefused) -> dict[str, Any]:
    """A harness result for cell code this host would not start."""
    return {"success": False, "error": str(exc), "stderr": "", "stdout": "", "variables": {}}

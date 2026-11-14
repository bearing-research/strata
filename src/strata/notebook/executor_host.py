"""What ``CellExecutor`` asks of the session that owns it, and of nothing else.

This names the boundary; it does not narrow it. Every member is one the executor or a cell
path it hands the session to (keys, prompt, SQL, widget, LLM config) uses today.
``tests/notebook/test_executor_host.py`` fails on a use that is not listed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import cellgraph as cg
    from vestigo.store import ArtifactVersion

    from strata.notebook.artifact_integration import NotebookArtifactManager
    from strata.notebook.dag import NotebookDag
    from strata.notebook.env import LocalPackages
    from strata.notebook.execution_result import CachedConsole
    from strata.notebook.models import CellOutput, DatasetSpec, InputWeight, NotebookState
    from strata.notebook.pool import WarmProcessPool
    from strata.notebook.session import NotebookSession


class ExecutorHost(Protocol):
    """The session, as a cell's execution sees it."""

    @property
    def id(self) -> str: ...

    @property
    def path(self) -> Path: ...

    @property
    def notebook_state(self) -> NotebookState: ...

    @property
    def analysis(self) -> cg.Analysis | None: ...

    @property
    def dag(self) -> NotebookDag | None: ...

    @property
    def venv_python(self) -> Path | None: ...

    @property
    def warm_pool(self) -> WarmProcessPool | None: ...

    @property
    def r_warm_pool(self) -> WarmProcessPool | None: ...

    @property
    def opened_by(self) -> tuple[str, str | None] | None: ...

    def get_artifact_manager(self) -> NotebookArtifactManager: ...

    def lake_config(self) -> Any: ...

    def parse_artifact_uri(self, artifact_uri: str) -> tuple[str, int]: ...

    def environment_attestation_error(self) -> str | None: ...

    def environment_execution_block_message(self) -> str | None: ...

    def cell_env_hash(self, cell: Any, packages: LocalPackages) -> str: ...

    def cell_test_fingerprint(
        self, cell_id: str, source: str, test_source: str
    ) -> tuple[str, str, str | None]: ...

    def wired_variables(self, cell_id: str, upstream_id: str) -> set[str]: ...

    def collect_input_refs(self, cell_id: str, *, variant: str | None = None) -> dict[str, str]: ...

    def resolve_cached_console(
        self, cell_id: str, provenance_hash: str, *, variant: str | None = None
    ) -> CachedConsole | None: ...

    def resolve_cached_display_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        current_outputs: list[CellOutput],
        variant: str | None = None,
    ) -> list[CellOutput]: ...

    def stored_display_outputs(self, artifacts: list[ArtifactVersion]) -> list[CellOutput]: ...

    def apply_execution_result_metadata(
        self, cell_id: str, result: Any, *, instance: bool = False
    ) -> None: ...

    def persist_display_outputs(
        self, cell_id: str, display_outputs: list[dict[str, Any]] | None
    ) -> None: ...

    def reads_approved(self, cell_id: str, machine: str, fingerprint: str) -> bool: ...

    def record_cell_inputs(
        self, cell_id: str, machine: str, weights: list[InputWeight]
    ) -> bool: ...

    def record_cell_test_inputs(self, cell_id: str, inputs: list[str], env_hash: str) -> str: ...

    def record_successful_execution_provenance(
        self,
        cell_id: str,
        provenance_hash: str,
        source_hash: str,
        env_hash: str,
        *,
        table_reads: list[dict[str, Any]],
        reads: dict[str, Any] | None,
        key_parts: dict[str, Any] | None,
        writes: dict[str, Any] | None = None,
        reopen_identity: str | None = None,
    ) -> None: ...

    def remember_dataset_fingerprint(self, spec: DatasetSpec, fingerprint: str) -> None: ...


if TYPE_CHECKING:

    def _a_session_is_a_host(session: NotebookSession) -> ExecutorHost:
        """``ty check`` fails here when ``NotebookSession`` stops satisfying the protocol."""
        return session

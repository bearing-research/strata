"""Session management for open notebooks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import time as _time
import tomllib
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.annotations import parse_annotations
from strata.notebook.causality import CausalityChain, compute_causality_on_staleness, skip_none
from strata.notebook.dag import CellAnalysisWithId, NotebookDag
from strata.notebook.dependencies import (
    DependencyChangeResult,
    EnvironmentOperationLog,
    RequirementsImportResult,
    _get_notebook_lock,
    import_environment_yaml_text,
    import_environment_yaml_text_streaming,
    import_requirements_text,
    import_requirements_text_streaming,
    list_dependencies,
    list_r_packages,
)
from strata.notebook.env import (
    compute_execution_env_hash,
    compute_lockfile_hash,
    narrow_env_for_provenance,
)
from strata.notebook.models import (
    CellOutput,
    CellStaleness,
    CellState,
    CellStatus,
    NotebookState,
    StalenessReason,
    VariantGroupState,
    VariantMember,
)
from strata.notebook.mounts import MountFingerprinter, resolve_cell_mounts
from strata.notebook.parser import parse_notebook
from strata.notebook.protocol import MessageType
from strata.notebook.provenance import (
    compute_provenance_hash,
    compute_source_hash,
    derive_subkey,
)
from strata.notebook.python_versions import (
    read_requested_python_minor,
    read_venv_runtime_python_version,
)
from strata.notebook.runtime_state import (
    EnvironmentRuntime,
    RRuntime,
    load_runtime_state,
    save_runtime_state,
)
from strata.notebook.timing import NotebookTimingRecorder
from strata.notebook.workers import (
    build_worker_catalog,
    resolve_worker_spec,
    worker_runtime_identity,
    worker_supports_notebook_execution,
)
from strata.notebook.writer import (
    _renv_sync,
    _uv_sync,
    update_cell_display_outputs,
    update_environment_metadata,
)
from strata.notebook.ws_payloads import (
    cell_status_payload,
    environment_job_event_payload,
)

if TYPE_CHECKING:
    from strata.notebook.artifact_integration import NotebookArtifactManager
    from strata.notebook.pool import WarmProcessPool

logger = logging.getLogger(__name__)
_ENVIRONMENT_JOB_HISTORY_LIMIT = 8

_VARIANT_LINE_RE = re.compile(
    r"^(\s*#\s*@variant\s+\S+\s+)\S+(.*)$",
    re.MULTILINE,
)


def _next_variant_name(active_name: str, taken: set[str]) -> str:
    """Return ``<active>_copy`` (or ``<active>_copy2``, ``_copy3``, …)."""
    candidate = f"{active_name}_copy"
    if candidate not in taken:
        return candidate
    n = 2
    while f"{active_name}_copy{n}" in taken:
        n += 1
    return f"{active_name}_copy{n}"


def _rewrite_variant_annotation(source: str, group: str, new_name: str) -> str:
    """Replace the first ``# @variant <group> <old>`` line with ``<new>``.

    If the source doesn't already contain a variant annotation (the
    active cell somehow lost its annotation), prepend a fresh one so the
    new sibling still joins the group.
    """
    new_source, count = _VARIANT_LINE_RE.subn(rf"\g<1>{new_name}\g<2>", source, count=1)
    if count == 0:
        return f"# @variant {group} {new_name}\n{source}"
    return new_source


@dataclass
class ExecutionSample:
    """One execution timing sample for profiling and estimates."""

    duration_ms: float
    cache_hit: bool


@dataclass
class DependencyMutationOutcome:
    """Result of a notebook dependency mutation."""

    result: DependencyChangeResult
    staleness_map: dict[str, CellStaleness]


@dataclass
class RequirementsImportOutcome:
    """Result of importing notebook dependencies from requirements text."""

    result: RequirementsImportResult
    staleness_map: dict[str, CellStaleness]


@dataclass(frozen=True)
class CellStateSnapshot:
    """Per-cell state slice used to diff before/after a staleness recompute.

    Attributes
    ----------
    status : str
        Cell status value (e.g. ``"ready"``, ``"stale"``, ``"running"``).
    reasons : tuple of str
        Staleness reason values in the order they were recorded.
    causality : dict of {str : Any} or None
        Wire-format causality chain (``asdict``-serialized with None fields
        stripped), or ``None`` when the cell has no causality entry.
    """

    status: str
    reasons: tuple[str, ...]
    causality: dict[str, Any] | None


@dataclass
class EnvironmentJobSnapshot:
    """One notebook-scoped background environment operation."""

    id: str
    action: str
    command: str
    status: str
    started_at: int
    package: str | None = None
    phase: str | None = None
    duration_ms: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    finished_at: int | None = None
    lockfile_changed: bool = False
    stale_cell_count: int = 0
    stale_cell_ids: list[str] = field(default_factory=list)
    error: str | None = None


class NotebookSession:
    """Holds state for one open notebook.

    Attributes:
        id: Session ID
        notebook_state: Current notebook state
        path: Path to notebook directory
        venv_python: Path to python executable in notebook venv
        dag: The computed DAG for the notebook
        artifact_manager: NotebookArtifactManager for this notebook (M4)
    """

    def __init__(self, notebook_state: NotebookState, path: Path):
        """Initialize a notebook session.

        Args:
            notebook_state: NotebookState from parser
            path: Path to notebook directory
        """
        from strata.notebook.env_backend import EnvironmentBackend, get_backend

        self.id: str = str(uuid.uuid4())
        self.notebook_state = notebook_state
        self.path = Path(path)
        self.venv_python: Path | None = None
        self.dag: NotebookDag | None = None
        # Environment backend — Phase 1 always resolves to UvBackend.
        # Phase 2 will use detection + notebook.toml override to pick
        # between UvBackend and AttachedBackend.
        self.backend: EnvironmentBackend = get_backend(self.path)

        # M4: Initialize artifact manager
        from strata.notebook.artifact_integration import NotebookArtifactManager

        self.artifact_manager = NotebookArtifactManager(
            notebook_id=notebook_state.id,
            artifact_dir=path / ".strata" / "artifacts",
        )

        # M6: Initialize warm process pool (optional)
        self.warm_pool: WarmProcessPool | None = None
        self.r_warm_pool: WarmProcessPool | None = None

        # Session TTL tracking
        self.last_accessed: float = _time.time()

        # v1.1: Execution history for profiling and duration estimates.
        self.execution_history: dict[str, list[ExecutionSample]] = {}

        # v1.1: Causality chains for stale cells
        self.causality_map: dict[str, CausalityChain] = {}

        # Environment/runtime sync state for the current notebook venv.
        self.environment_sync_state: str = "unknown"
        self.environment_sync_error: str | None = None
        self.environment_sync_notice: str | None = None
        self.environment_last_synced_at: int | None = None
        self.environment_last_sync_duration_ms: int | None = None
        self.environment_python_version: str = ""
        self.environment_interpreter_source: str = "unknown"
        self.environment_job: EnvironmentJobSnapshot | None = None
        self.environment_job_history: list[EnvironmentJobSnapshot] = []
        self.environment_job_task: asyncio.Task[None] | None = None
        self._environment_state_lock = threading.RLock()
        self._synchronous_environment_mutation: str | None = None
        self._load_environment_job_history()

        # Analyze all cells and build DAG
        self._analyze_and_build_dag()
        self._run_annotation_validation()
        # Pull secrets from the configured manager (if any) and merge
        # them into the env map before cells start seeing env.
        self._apply_configured_secrets()

    def _apply_configured_secrets(self) -> None:
        """Fetch + merge secrets from the configured provider, if any.

        Updates notebook_state.env, env_sources, env_fetch_error,
        env_fetched_at in place. Non-destructive on no-op notebooks —
        when there's no ``[secret_manager]`` block, env_sources is still
        stamped with ``manual`` for every existing key so the UI has
        a consistent source map to render.

        Also mirrors the fresh env into each cell's resolved env so
        the executor (which reads cell.env) sees the fetched values.
        """
        from strata.notebook.secret_manager import apply_secrets_to_notebook_state

        apply_secrets_to_notebook_state(self.notebook_state)
        # Rebuild per-cell resolved env, preserving cell-level
        # overrides — same pattern as update_notebook_env_endpoint.
        for cell in self.notebook_state.cells:
            resolved = dict(self.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved

    def refresh_secrets(self):
        """Re-fetch secrets and re-merge into env. Used by the Refresh button."""
        self._apply_configured_secrets()

    def _run_annotation_validation(self) -> None:
        """Validate annotations across all cells. Called on open/reload only."""
        for cell in self.notebook_state.cells:
            diagnostics = validate_cell_annotations(cell, self.notebook_state)
            cell.annotation_diagnostics = diagnostics
            for d in diagnostics:
                logger.warning(
                    "annotation diagnostic notebook=%s cell=%s code=%s: %s",
                    self.notebook_state.id,
                    cell.id,
                    d.code,
                    d.message,
                )

    def mark_environment_pending(self, notice: str | None = None) -> None:
        """Mark the notebook environment as pending background initialization."""
        self.venv_python = None
        self.environment_python_version = ""
        self.environment_interpreter_source = "unknown"
        self.environment_sync_state = "pending"
        self.environment_sync_error = None
        self.environment_sync_notice = notice or (
            "Notebook environment is being created in the background. "
            "Running cells is disabled until it finishes."
        )
        self.environment_last_synced_at = None
        self.environment_last_sync_duration_ms = None

    def touch(self) -> None:
        """Record recent activity for TTL accounting."""
        self.last_accessed = _time.time()

    def set_variant_active(self, group: str, variant_name: str) -> None:
        """Switch the active variant for ``group``.

        Persists the selection to ``notebook.toml`` and reloads so the
        DAG, cell-level ``variant_active`` flags, and downstream
        staleness all recompute against the new selection. Reloading
        already handles "downstream provenance hashes change because
        their input artifact ID points at a different cell" correctly,
        so cells that need re-running are marked STALE in the usual way.
        """
        from strata.notebook.writer import set_variant_active as _set_variant_active

        _set_variant_active(self.path, group, variant_name)
        self.reload()

    def set_variant_mode(self, group: str, mode: str) -> None:
        """Switch a variant group between ``"switch"`` and ``"sweep"`` mode.

        Persists to ``notebook.toml`` and reloads so the DAG rebuilds (sweep →
        all members run, the producer fans out) and downstream staleness
        recomputes — switch→sweep changes a downstream's input set from one
        artifact to the grouped variant map, so consumers restalen in the usual
        way. ``mode`` is validated by the caller; unknown values persist but are
        treated as ``"switch"`` at execution time.
        """
        from strata.notebook.writer import set_variant_mode as _set_variant_mode

        _set_variant_mode(self.path, group, mode)
        self.reload()

    def remove_cell(self, cell_id: str) -> None:
        """Delete a cell, with variant-aware cleanup.

        When the deleted cell is part of a variant group:

        - Group has other members: just remove the cell. If it was the
          active variant, promote the next member in source order so the
          toml's ``active`` pointer doesn't dangle.
        - Group has only this member: remove the cell *and* drop the
          ``[[variant_group]]`` entry — the group dissolves.

        Non-variant cells take the unchanged code path: remove and reload.
        """
        from strata.notebook.writer import (
            remove_cell_from_notebook,
            remove_variant_group_entry,
        )
        from strata.notebook.writer import (
            set_variant_active as _set_variant_active,
        )

        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            raise ValueError(f"Cell {cell_id} not found")

        group_id = cell.variant_group
        resolved = None
        if group_id is not None:
            resolved = next(
                (g for g in self.notebook_state.variant_groups if g.group == group_id),
                None,
            )

        # Decide the variant_group toml fixup *before* the cell goes away,
        # while we still have the resolved group's member ordering.
        promote_to: str | None = None
        drop_group = False
        if resolved is not None:
            remaining = [m for m in resolved.members if m.cell_id != cell_id]
            if not remaining:
                drop_group = True
            elif cell.id == resolved.active_cell_id:
                # The active variant is going away — promote the
                # first-in-source-order survivor so the toml pointer stays
                # valid (otherwise reload would emit variant_active_unknown
                # and fall back implicitly, which works but is noisy).
                promote_to = remaining[0].name

        remove_cell_from_notebook(self.path, cell_id)

        if drop_group and group_id is not None:
            remove_variant_group_entry(self.path, group_id)
        elif promote_to is not None and group_id is not None:
            _set_variant_active(self.path, group_id, promote_to)

        self.reload()

    def add_variant(self, group: str) -> tuple[str, str]:
        """Add a sibling variant to ``group``, cloning the active variant.

        Returns ``(new_variant_name, new_cell_id)``. The new variant is
        placed immediately after the last existing member in source order,
        becomes active on creation, and starts as a copy of the active
        variant's body with the ``# @variant`` line rewritten to its name.

        Names auto-generate as ``<active>_copy``, ``<active>_copy2``, …
        — the user renames by editing the annotation line in source,
        which is consistent with how every other cell-metadata edit works.

        Raises ``ValueError`` if ``group`` doesn't exist in the resolved
        variant groups (caller should surface a 404 / WS error).
        """
        from strata.notebook.writer import add_cell_to_notebook, write_cell

        resolved = next(
            (g for g in self.notebook_state.variant_groups if g.group == group),
            None,
        )
        if resolved is None:
            raise ValueError(f"Variant group {group!r} does not exist")

        active_cell = self.notebook_state.get_cell(resolved.active_cell_id)
        if active_cell is None:
            # Defensive: resolution gave us an active_cell_id that doesn't
            # match any cell. Shouldn't happen but bail cleanly if it does.
            raise ValueError(f"Active variant cell for group {group!r} not found")

        taken = {m.name for m in resolved.members}
        new_name = _next_variant_name(resolved.active_name, taken)
        new_cell_id = uuid.uuid4().hex[:8]
        last_member_cell_id = resolved.members[-1].cell_id

        add_cell_to_notebook(
            self.path,
            new_cell_id,
            after_cell_id=last_member_cell_id,
            language=active_cell.language,
        )
        new_source = _rewrite_variant_annotation(active_cell.source, group, new_name)
        write_cell(self.path, new_cell_id, new_source)

        # Switch active to the new variant. set_variant_active reloads,
        # so the DAG / staleness / variant flags refresh in one pass.
        self.set_variant_active(group, new_name)
        return new_name, new_cell_id

    def reload(self) -> None:
        """Reload notebook state from disk."""
        previous_cells = {cell.id: cell.model_copy(deep=True) for cell in self.notebook_state.cells}
        previous_runtime_identities = {
            cell.id: self._effective_worker_runtime_identity(cell)
            for cell in self.notebook_state.cells
        }
        self.notebook_state = parse_notebook(self.path)
        # Re-analyze all cells and rebuild DAG
        self._analyze_and_build_dag()
        self._run_annotation_validation()
        self._apply_configured_secrets()
        # Restore prior display outputs and provenance history *before*
        # computing staleness. compute_staleness() compares the cell's
        # ``last_provenance_hash`` against the newly-resolved env / source
        # / inputs — without restoring it first, every cell falls back to
        # IDLE and we lose the ability to mark cells as STALE.
        self._restore_execution_history(previous_cells)
        self.compute_staleness()
        self._restore_ready_runtime_state(previous_cells, previous_runtime_identities)

    def _analyze_and_build_dag(self) -> None:
        """Analyze all cells and build the DAG.

        Updates notebook_state with defines/references/upstream/downstream/isLeaf.
        """
        # Per-language analyzer dispatch lives in ``strata.notebook.languages``;
        # adding a new language is a registry entry, not a branch here.
        from strata.notebook.languages import analyze_cell_by_language

        cell_analyses = []
        for cell in self.notebook_state.cells:
            analyzed = analyze_cell_by_language(cell, self)
            defines = list(analyzed.defines)
            references = list(analyzed.references)
            mutation_defines = list(analyzed.mutation_defines)

            # Loop cells read the carry variable from upstream on iter 0
            # even when Python scoping sees it as a local (because the
            # body both reads and rebinds it). Record the carry as a
            # reference so the DAG links the loop cell to the upstream
            # that seeds its initial state.
            annotations = parse_annotations(cell.source)
            if (
                annotations.loop is not None
                and annotations.loop.carry
                and annotations.loop.carry not in references
                and annotations.loop.start_from_cell is None
            ):
                references = references + [annotations.loop.carry]

            variant_group = annotations.variant.group if annotations.variant is not None else None
            variant_name = annotations.variant.name if annotations.variant is not None else None
            cell_analyses.append(
                CellAnalysisWithId(
                    id=cell.id,
                    defines=defines,
                    references=references,
                    after=list(annotations.after),
                    variant_group=variant_group,
                    variant_name=variant_name,
                    per_variant=annotations.per_variant,
                    per_variant_group=annotations.per_variant_group,
                )
            )
            # Update cell with analysis results
            cell.defines = defines
            cell.references = references
            cell.mutation_defines = mutation_defines
            cell.variant_group = variant_group
            cell.variant_name = variant_name
            # Default to active until the DAG resolution proves otherwise.
            cell.variant_active = True

        # Build DAG
        try:
            self.dag = NotebookDag.from_cells(
                cell_analyses,
                variant_active_selections=self.notebook_state.variant_active_selections,
                variant_modes=self.notebook_state.variant_modes,
            )

            # Update cells with DAG information
            for cell in self.notebook_state.cells:
                cell.upstream_ids = self.dag.cell_upstream.get(cell.id, [])
                cell.downstream_ids = self.dag.cell_downstream.get(cell.id, [])
                cell.is_leaf = cell.id in self.dag.leaves
                cell.variant_active = cell.id not in self.dag.inactive_cells

            # Surface resolved variant groups for the API/frontend.
            self.notebook_state.variant_groups = [
                VariantGroupState(
                    group=group.group,
                    active_name=group.active_name,
                    active_cell_id=group.active_cell_id,
                    mode=group.mode,
                    members=[
                        VariantMember(
                            cell_id=cid,
                            name=name,
                            is_active=(cid == group.active_cell_id),
                        )
                        for cid, name in group.members
                    ],
                )
                for group in self.dag.variant_groups
            ]

        except ValueError as e:
            # Cycle detected or variant collision — log but don't crash.
            logger.warning("DAG build failed: %s", e)
            self.dag = None

    def _restore_execution_history(self, previous_cells: dict[str, Any]) -> None:
        """Restore per-cell execution history that's not persisted to notebook.toml.

        Display outputs, artifact URIs, and the last-seen provenance / source
        / env hashes all live only in the session. After a reload() the
        newly-parsed cells are at their defaults, so we copy the history
        back from the pre-reload snapshot whenever the cell is unambiguously
        the same (same id, same source). Status and staleness are not
        touched here — compute_staleness() runs afterwards and may
        legitimately downgrade a previously-READY cell to STALE if the
        resolved env / upstream provenance has changed.
        """
        for cell in self.notebook_state.cells:
            previous = previous_cells.get(cell.id)
            if previous is None or previous.source != cell.source:
                continue

            cell.artifact_uri = previous.artifact_uri
            cell.artifact_uris = dict(previous.artifact_uris)
            cell.display_outputs = [
                output.model_copy(deep=True) for output in previous.display_outputs
            ]
            cell.display_output = (
                previous.display_output.model_copy(deep=True)
                if previous.display_output is not None
                else None
            )
            cell.cache_hit = previous.cache_hit
            cell.execution_method = previous.execution_method
            cell.remote_worker = previous.remote_worker
            cell.remote_transport = previous.remote_transport
            cell.remote_build_id = previous.remote_build_id
            cell.remote_build_state = previous.remote_build_state
            cell.remote_error_code = previous.remote_error_code
            cell.last_provenance_hash = previous.last_provenance_hash
            cell.last_source_hash = previous.last_source_hash
            cell.last_env_hash = previous.last_env_hash
            cell.widget_values = dict(previous.widget_values)

    def _restore_ready_runtime_state(
        self,
        previous_cells: dict[str, Any],
        previous_runtime_identities: dict[str, str | None],
    ) -> None:
        """Preserve READY status for cells whose runtime identity is unchanged.

        Runs after compute_staleness(): for cells that staleness could not
        classify as READY (e.g., leaves without canonical artifacts) we
        promote them back to READY when the entire runtime identity
        matches the pre-reload snapshot. History fields were already
        restored by ``_restore_execution_history``.
        """
        for cell in self.notebook_state.cells:
            previous = previous_cells.get(cell.id)
            can_restore_ready_state = (
                previous is not None
                and previous.source == cell.source
                and previous.status == CellStatus.READY
                and cell.status == CellStatus.IDLE
                and previous.worker == cell.worker
                and previous.worker_override == cell.worker_override
                and previous.env == cell.env
                and previous.env_overrides == cell.env_overrides
                and previous.upstream_ids == cell.upstream_ids
                and previous.downstream_ids == cell.downstream_ids
                and previous.mounts == cell.mounts
                and previous.is_leaf == cell.is_leaf
                and previous_runtime_identities.get(cell.id)
                == self._effective_worker_runtime_identity(cell)
            )
            if not can_restore_ready_state:
                continue

            cell.status = CellStatus.READY
            cell.staleness = CellStaleness(status=CellStatus.READY, reasons=[])
            self.causality_map.pop(cell.id, None)

    def re_analyze_cell(self, cell_id: str) -> None:
        """Re-analyze a single cell and rebuild the DAG.

        Args:
            cell_id: ID of the cell to re-analyze
        """
        # Find the cell
        cell = self.notebook_state.get_cell(cell_id)
        if not cell:
            return

        # Per-language analyzer dispatch lives in
        # ``strata.notebook.languages``; ``analyze_cell_by_language``
        # returns the unified ``AnalyzedCell`` shape regardless of the
        # underlying language.
        from strata.notebook.languages import analyze_cell_by_language

        analyzed = analyze_cell_by_language(cell, self)
        cell.defines = list(analyzed.defines)
        cell.references = list(analyzed.references)

        # Rebuild full DAG (since one cell changed, downstream may be affected)
        self._analyze_and_build_dag()

    def _resolve_sql_dialect(self, cell) -> str | None:
        """Look up the sqlglot dialect for a SQL cell's connection.

        Walks: cell source → ``# @sql connection=<name>`` →
        ``notebook.connections[<name>]`` → ``DriverAdapter.sqlglot_dialect``.

        Returns ``None`` when any step is unresolved — the connection
        isn't declared, the driver isn't registered, or the cell has
        no ``# @sql`` annotation. The analyzer treats ``None`` as
        "skip table extraction"; the executor re-resolves at execute
        time when the connection MUST exist.
        """
        from strata.notebook.annotations import parse_annotations

        annotations = parse_annotations(cell.source)
        if annotations.sql is None or not annotations.sql.connection:
            return None
        connection_name = annotations.sql.connection
        connection = next(
            (c for c in self.notebook_state.connections if c.name == connection_name),
            None,
        )
        if connection is None:
            return None
        try:
            from strata.notebook.sql.registry import get_adapter

            adapter = get_adapter(connection.driver)
        except (KeyError, ImportError):
            return None
        return adapter.sqlglot_dialect

    def get_artifact_manager(self) -> NotebookArtifactManager:
        """Get the artifact manager for this session.

        Returns:
            NotebookArtifactManager instance
        """
        return self.artifact_manager

    def compute_staleness(self) -> dict[str, CellStaleness]:
        """Compute staleness status for all cells.

        Walk cells in topological order and check if cached artifacts
        match the current provenance hash. Updates cell.staleness.
        Also computes causality chains for stale cells (v1.1).

        Returns:
            Dict mapping cell_id -> CellStaleness
        """
        staleness_map: dict[str, CellStaleness] = {}
        stale_cells: set[str] = set()  # Track stale cells for propagation
        if self.dag is None:
            # No DAG — all cells are idle
            for cell in self.notebook_state.cells:
                staleness_map[cell.id] = CellStaleness(status=CellStatus.IDLE)
            self._apply_staleness_map(staleness_map)
            self.causality_map = {}
            return staleness_map

        # Walk cells in topological order
        for cell_id in self.dag.topological_order:
            cell = self.notebook_state.get_cell(cell_id)
            if cell is None:
                continue

            # Languages that skip the provenance chain (today: markdown,
            # since it's pure prose with no inputs or subprocess) are
            # always READY — no hashing, no cache lookup. The protocol
            # exposes this via ``skips_execution_provenance`` so adding
            # another no-execution language (TBD) doesn't need an edit
            # here.
            from strata.notebook.languages import get_language_executor

            language_executor = get_language_executor(cell.language)
            if language_executor.skips_execution_provenance:
                staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                continue

            # If ANY upstream cell is stale, this cell's inputs will change
            # once the upstream re-runs, so it too is out of date and must
            # re-run before it can be trusted. How we surface that depends
            # on whether this cell already holds a result (#361):
            #   - it ran before (``last_provenance_hash`` set) → STALE with
            #     an UPSTREAM reason, so the UI reads "stale · upstream
            #     changed" rather than a bare IDLE (matches how a user
            #     watching a cascade thinks about it).
            #   - it never ran (fresh notebook) → IDLE: there is no cached
            #     result to invalidate, and it can't be evaluated until the
            #     upstream produces its inputs.
            # Either way it propagates: downstream cells are out of date too.
            has_stale_upstream = any(uid in stale_cells for uid in cell.upstream_ids)

            if has_stale_upstream:
                if cell.last_provenance_hash is not None:
                    staleness_map[cell_id] = CellStaleness(
                        status=CellStatus.STALE, reasons=[StalenessReason.UPSTREAM]
                    )
                else:
                    staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                stale_cells.add(cell_id)
                continue

            effective_worker = self._effective_worker_name(cell)
            worker_spec = resolve_worker_spec(
                self.notebook_state,
                effective_worker,
            )
            if not worker_supports_notebook_execution(worker_spec):
                staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                stale_cells.add(cell_id)
                continue

            # Compute current provenance hash
            source_hash = compute_source_hash(cell.source)
            runtime_env = self._collect_runtime_env(cell)
            env_hash = compute_execution_env_hash(
                self.path,
                runtime_env,
                runtime_identity=self._effective_worker_runtime_identity(cell),
            )

            # Get input hashes from upstream artifacts. Use the same
            # per-variable artifact selection as execution, not the legacy
            # single artifact_uri field.
            input_hashes = self._collect_input_hashes(cell_id)
            mount_fingerprints, has_rw_mount = self._collect_mount_fingerprints(cell)

            if has_rw_mount:
                staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                stale_cells.add(cell_id)
                continue

            table_fingerprints = self._collect_table_fingerprints(cell)

            provenance_hash = compute_provenance_hash(
                input_hashes + mount_fingerprints + table_fingerprints,
                source_hash,
                env_hash,
            )

            # Check if cached artifact exists.
            # The executor stores per-variable provenance hashes:
            #   sha256(f"{provenance_hash}:{var_name}")
            # so we must check with the same scheme.
            cached_outputs = self._resolve_cached_outputs(cell_id, provenance_hash)
            cached_display_outputs = self._resolve_cached_display_outputs(
                cell_id,
                provenance_hash,
                cell.display_outputs,
            )

            if cached_outputs is None:
                if cached_display_outputs:
                    cell.display_outputs = cached_display_outputs
                    cell.display_output = cached_display_outputs[-1]
                    staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                else:
                    # Languages with an alternate per-variable cache
                    # scheme (today PROMPT + SQL) store artifacts under
                    # a hash the generic per-variable lookup above
                    # can't match. The wrapper persists the generic
                    # provenance hash via
                    # ``record_successful_execution_provenance`` so we
                    # preserve READY status when it matches despite a
                    # cache miss. Same logic kicks in for leaf cells.
                    can_preserve_uncached_ready = (
                        (cell.is_leaf or language_executor.has_alternate_cache_scheme)
                        and cell.status == CellStatus.READY
                        and cell.last_provenance_hash == provenance_hash
                    )
                    if can_preserve_uncached_ready:
                        staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                    else:
                        # No cached artifact — cell is stale/idle unless we can
                        # prove it still matches the last successful uncached run.
                        staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                        stale_cells.add(cell_id)
            else:
                # Artifact exists — mark as ready
                staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                # Populate per-variable artifact URIs
                for var_name, (artifact_id, version) in cached_outputs.items():
                    uri = f"strata://artifact/{artifact_id}@v={version}"
                    cell.artifact_uris[var_name] = uri
                    cell.artifact_uri = uri  # backward compat
                cell.display_outputs = cached_display_outputs or []
                cell.display_output = cached_display_outputs[-1] if cached_display_outputs else None

        self._apply_staleness_map(staleness_map)

        # v1.1: Compute causality chains for stale cells
        self.causality_map = compute_causality_on_staleness(self)

        return staleness_map

    def _apply_staleness_map(self, staleness_map: dict[str, CellStaleness]) -> None:
        """Persist computed staleness back onto in-memory cell state."""
        for cell in self.notebook_state.cells:
            staleness = staleness_map.get(cell.id)
            if staleness is None:
                continue
            cell.staleness = staleness
            cell.status = staleness.status
            if staleness.status != CellStatus.READY:
                cell.cache_hit = False

    def mark_executed_ready(self, cell_id: str) -> None:
        """Preserve a just-executed cell as ready in backend state.

        Some cells, especially leaves, are intentionally not cacheable via the
        canonical artifact path. They should still appear as successfully run
        immediately after execution, even though a later staleness recompute
        may otherwise classify them as idle.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return

        cell.staleness = CellStaleness(status=CellStatus.READY, reasons=[])
        cell.status = CellStatus.READY
        self.causality_map.pop(cell_id, None)

    def mark_cell_running(self, cell_id: str) -> None:
        """Mark a cell as currently executing in backend state.

        Single controlled entry point used by every execution-driving
        path (REST execute, WS direct/cascade/run_all, agent execute).
        Direct ``cell.status = CellStatus.RUNNING`` mutations elsewhere
        drift from this canonical setter and race with concurrent
        execution paths writing different statuses to the same cell.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.status = CellStatus.RUNNING

    def mark_cell_error(self, cell_id: str) -> list[str]:
        """Mark a cell as errored in backend state.

        Companion to ``mark_cell_running`` / ``mark_executed_ready`` —
        the controlled way to record an execution failure on the cell.

        Also walks the DAG downstream and flips any cell currently in
        ``READY`` (i.e. showing a cached output from a previous good
        run of the failed cell) to ``STALE``. Without this, the
        downstream cell keeps reading its cached artifact — whose
        inputs were materialised from the now-broken upstream's last
        success — and the UI shows green even though the upstream
        cell is red. Returns the list of downstream cells whose
        status flipped, so the caller can broadcast cell-status
        updates for them.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return []
        cell.status = CellStatus.ERROR
        if self.dag is None:
            return []
        affected: list[str] = []
        seen: set[str] = set()
        queue = list(self.dag.cell_downstream.get(cell_id, []))
        while queue:
            nid = queue.pop()
            if nid in seen:
                continue
            seen.add(nid)
            downstream_cell = self.notebook_state.get_cell(nid)
            if downstream_cell is not None and downstream_cell.status == CellStatus.READY:
                downstream_cell.status = CellStatus.STALE
                affected.append(nid)
            queue.extend(self.dag.cell_downstream.get(nid, []))
        return affected

    def apply_execution_result_metadata(self, cell_id: str, result: Any) -> None:
        """Persist transient execution metadata onto the session cell state."""
        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return

        cell.execution_method = result.execution_method
        # A cache hit replays display outputs from the artifact store but
        # carries no fresh console — stdout/stderr aren't part of the cached
        # artifact. Writing the empty cache-hit console here would make
        # ``update_cell_console_output`` *unlink* the file the original
        # execution wrote, so a re-run that hits cache (e.g. a second
        # ``strata run``) would silently delete recoverable print() output.
        # Leave the persisted console untouched on cache hits.
        if result.success and not result.cache_hit:
            cell.console_stdout = result.stdout or ""
            cell.console_stderr = result.stderr or ""
            # Console output lives in .strata/console/, not notebook.toml —
            # invariant 6: runtime writers never touch notebook.toml.
            from strata.notebook.writer import update_cell_console_output

            update_cell_console_output(self.path, cell_id, result.stdout or "", result.stderr or "")
        if result.success and result.display_outputs:
            cell.display_outputs = [CellOutput(**output) for output in result.display_outputs]
            cell.display_output = cell.display_outputs[-1]
        elif result.success:
            cell.display_outputs = []
            cell.display_output = None
        elif not result.success:
            cell.display_outputs = []
            cell.display_output = None

        if (
            result.remote_worker
            or result.remote_transport
            or result.remote_build_id
            or result.remote_build_state
            or result.remote_error_code
        ):
            cell.remote_worker = result.remote_worker
            cell.remote_transport = result.remote_transport
            if result.execution_method == "cached":
                if result.remote_build_id is not None:
                    cell.remote_build_id = result.remote_build_id
                if result.remote_build_state is not None:
                    cell.remote_build_state = result.remote_build_state
                if result.remote_error_code is not None:
                    cell.remote_error_code = result.remote_error_code
            else:
                cell.remote_build_id = result.remote_build_id
                cell.remote_build_state = result.remote_build_state
                cell.remote_error_code = result.remote_error_code
            return

        if result.execution_method != "cached":
            cell.remote_worker = None
            cell.remote_transport = None
            cell.remote_build_id = None
            cell.remote_build_state = None
            cell.remote_error_code = None

    def record_successful_execution_provenance(
        self,
        cell_id: str,
        provenance_hash: str,
        source_hash: str,
        env_hash: str,
    ) -> None:
        """Persist the last successful execution provenance for uncached cells.

        Updates the in-memory cell state and also writes to
        ``.strata/runtime.json`` so ``compute_staleness`` can classify
        the cell correctly after a notebook reopen without requiring a
        re-execution.
        """
        from strata.notebook.runtime_state import persist_cell_provenance

        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return
        cell.last_provenance_hash = provenance_hash
        cell.last_source_hash = source_hash
        cell.last_env_hash = env_hash
        persist_cell_provenance(
            self.path,
            cell_id,
            last_provenance_hash=provenance_hash,
            last_source_hash=source_hash,
            last_env_hash=env_hash,
        )

    def serialize_cell(self, cell: CellState) -> dict[str, Any]:
        """Serialize a cell with session-coupled overlays.

        Wraps ``CellState.serialize()`` (cell-only view) with the three
        overlays that need session state: hydrated display outputs (which
        read from the artifact store), causality chains, and DAG shadow
        warnings.
        """
        data = cell.serialize()
        if cell.display_outputs:
            data["display_outputs"] = [
                self._hydrate_display_output(output) for output in cell.display_outputs
            ]
        if cell.display_output is not None:
            data["display_output"] = self._hydrate_display_output(cell.display_output)
        causality = self.causality_map.get(cell.id)
        if causality is not None:
            data["causality"] = asdict(causality, dict_factory=skip_none)
        if self.dag and cell.id in self.dag.shadow_warnings:
            data["shadow_warnings"] = self.dag.shadow_warnings[cell.id]
        from strata.notebook.models import CellLanguage

        if cell.language == CellLanguage.WIDGET:
            # Controls (parsed from source) + their current runtime values, so the
            # frontend can render the panel without a separate round-trip.
            from strata.notebook.widget_analyzer import analyze_widget_cell

            descriptors = analyze_widget_cell(cell.source).descriptors
            data["widget"] = {
                "descriptors": [
                    {"name": d.name, "kind": d.kind, "params": d.params, "default": d.default}
                    for d in descriptors
                ],
                "values": dict(cell.widget_values),
            }
        return data

    def persist_display_outputs(
        self, cell_id: str, display_outputs: list[dict[str, Any]] | None
    ) -> None:
        """Persist display metadata to ``.strata/runtime.json`` for reopen restoration.

        Display outputs are runtime state, not committed config — same
        reason as console output, per CLAUDE.md invariant 6.
        """
        update_cell_display_outputs(self.path, cell_id, display_outputs)

    def persist_display_output(self, cell_id: str, display_output: dict[str, Any] | None) -> None:
        """Backward-compatible single-display wrapper."""
        self.persist_display_outputs(cell_id, [display_output] if display_output else None)

    def _resolve_cached_display_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        current_outputs: list[CellOutput],
    ) -> list[CellOutput]:
        """Return cached ordered display outputs for a cell when available."""
        if not current_outputs:
            return []

        resolved: list[CellOutput] = []
        notebook_id = self.notebook_state.id
        for index, current_output in enumerate(current_outputs):
            artifact_id = f"nb_{notebook_id}_cell_{cell_id}_var___display__{index}"
            expected_hash = hashlib.sha256(
                f"{provenance_hash}:__display__{index}".encode()
            ).hexdigest()
            artifact = self.artifact_manager.artifact_store.get_latest_version(artifact_id)
            if artifact is None or artifact.provenance_hash != expected_hash:
                return []

            artifact_uri = f"strata://artifact/{artifact.id}@v={artifact.version}"
            output = current_output.model_copy(deep=True)
            output.artifact_uri = artifact_uri
            hydrated = self._hydrate_display_output(output)
            resolved.append(CellOutput(**hydrated) if hydrated is not None else output)
        return resolved

    def _hydrate_display_output(self, output: CellOutput | dict[str, Any]) -> dict[str, Any] | None:
        """Return a serialized display payload with any transient inline data added."""
        raw = output.model_dump() if isinstance(output, CellOutput) else dict(output)
        if raw.get("content_type") == "text/markdown":
            artifact_uri = raw.get("artifact_uri")
            if not isinstance(artifact_uri, str) or not artifact_uri:
                return raw

            if isinstance(raw.get("markdown_text"), str):
                return raw

            try:
                artifact_id, version = self._parse_artifact_uri(artifact_uri)
                blob = self.artifact_manager.load_artifact_data(artifact_id, version)
            except Exception:
                return raw

            raw["markdown_text"] = blob.decode("utf-8", errors="replace")
            return raw

        if raw.get("content_type") != "image/png":
            return raw

        artifact_uri = raw.get("artifact_uri")
        if not isinstance(artifact_uri, str) or not artifact_uri:
            return raw

        if isinstance(raw.get("inline_data_url"), str) and raw["inline_data_url"]:
            return raw

        try:
            artifact_id, version = self._parse_artifact_uri(artifact_uri)
            blob = self.artifact_manager.load_artifact_data(artifact_id, version)
        except Exception:
            return raw

        raw["inline_data_url"] = f"data:image/png;base64,{base64.b64encode(blob).decode('ascii')}"
        return raw

    @staticmethod
    def _parse_artifact_uri(artifact_uri: str) -> tuple[str, int]:
        """Parse a canonical artifact URI into (artifact_id, version)."""
        parts = artifact_uri.split("/")
        artifact_id = parts[-1].split("@")[0]
        version = int(parts[-1].split("@v=")[1])
        return artifact_id, version

    def serialize_cells(self) -> list[dict[str, Any]]:
        """Serialize all cells with runtime-derived metadata."""
        return [self.serialize_cell(cell) for cell in self.notebook_state.cells]

    def capture_cell_state_snapshot(self) -> dict[str, CellStateSnapshot]:
        """Capture cell status/reasons/causality for diffing after a recompute.

        Returns
        -------
        dict of {str : CellStateSnapshot}
            Mapping from cell ID to its current snapshot. Callers recompute
            staleness, build fresh snapshots, and broadcast only the cells
            whose snapshot changed.
        """
        snapshot: dict[str, CellStateSnapshot] = {}
        for cell in self.notebook_state.cells:
            causality = self.causality_map.get(cell.id)
            status = cell.status.value if isinstance(cell.status, CellStatus) else str(cell.status)
            reasons = tuple(
                reason.value for reason in (cell.staleness.reasons if cell.staleness else [])
            )
            snapshot[cell.id] = CellStateSnapshot(
                status=status,
                reasons=reasons,
                causality=asdict(causality, dict_factory=skip_none) if causality else None,
            )
        return snapshot

    def serialize_notebook_state(self) -> dict[str, Any]:
        """Serialize notebook state with enriched cell metadata."""
        data = self.notebook_state.model_dump()
        data["cells"] = self.serialize_cells()
        data["environment"] = self.serialize_environment_state()
        data["environment_job"] = self.serialize_environment_job_state()
        data["environment_job_history"] = self.serialize_environment_job_history()
        data["r_environment"] = self.serialize_r_environment_state()
        return data

    def _probe_python_version(self, python_executable: Path) -> str:
        """Return ``major.minor.micro`` for a Python interpreter when available."""
        cfg_version = read_venv_runtime_python_version(python_executable)
        if cfg_version:
            return cfg_version

        try:
            result = subprocess.run(
                [
                    str(python_executable),
                    "-c",
                    (
                        "import sys; "
                        "print("
                        "f'{sys.version_info.major}."
                        "{sys.version_info.minor}."
                        "{sys.version_info.micro}'"
                        ")"
                    ),
                ],
                cwd=str(self.path),
                capture_output=True,
                check=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return ""

        return result.stdout.strip()

    def _read_persisted_environment_metadata(self) -> EnvironmentRuntime:
        """Best-effort read of the persisted environment metadata.

        Lives in ``.strata/runtime.json`` under ``environment`` — the
        values change on every ``uv sync`` and are not user-authored,
        so they do not belong in the committed ``notebook.toml``.
        """
        from strata.notebook.runtime_state import load_runtime_state

        return load_runtime_state(self.path).environment

    def _resolved_package_count(self) -> int:
        """Count resolved packages from ``uv.lock`` when present."""
        lockfile = self.path / "uv.lock"
        if not lockfile.exists():
            return 0

        try:
            with open(lockfile, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            logger.debug("Failed to parse uv.lock for %s", self.path, exc_info=True)
            return 0

        packages = data.get("package", [])
        return len(packages) if isinstance(packages, list) else 0

    def serialize_environment_state(self) -> dict[str, Any]:
        """Serialize the live notebook environment state for the UI."""
        dependencies = list_dependencies(self.path)
        requested_python_version = read_requested_python_minor(self.path) or ""
        return {
            "requested_python_version": requested_python_version,
            "runtime_python_version": self.environment_python_version,
            "python_version": self.environment_python_version,
            "lockfile_hash": compute_lockfile_hash(self.path),
            "package_count": len(dependencies),
            "declared_package_count": len(dependencies),
            "resolved_package_count": self._resolved_package_count(),
            "sync_state": self.environment_sync_state,
            "sync_error": self.environment_sync_error,
            "sync_notice": self.environment_sync_notice,
            "last_synced_at": self.environment_last_synced_at,
            "last_sync_duration_ms": self.environment_last_sync_duration_ms,
            "has_lockfile": (self.path / "uv.lock").exists(),
            "venv_python": str(self.venv_python) if self.venv_python else None,
            "interpreter_source": self.environment_interpreter_source,
        }

    @property
    def _cached_system_r_version(self) -> str | None:
        """One-shot R version probe, cached per session.

        ``_probe_r_version`` spawns ``Rscript`` (10s timeout); we
        only need it once per session — the system R binary doesn't
        change underneath us. Without the cache, every
        ``serialize_r_environment_state`` call (run on every state
        sync, env refresh, dep mutation) would burn a subprocess
        spawn just to show "R 4.6.0" in the header.
        """
        if hasattr(self, "_system_r_version_cache"):
            return self._system_r_version_cache
        version = self._probe_r_version()
        self._system_r_version_cache = version
        return version

    def serialize_r_environment_state(self, *, include_packages: bool = False) -> dict[str, Any]:
        """Serialize the R-side runtime environment for the UI.

        ``has_lockfile`` is derived from disk — the UI must show R
        information for any notebook that ships a ``renv.lock``,
        including ones that never successfully synced (so the user
        can see *why* the env is broken). The other fields come
        from ``RRuntime`` in ``.strata/runtime.json`` and reflect
        the *last successful* sync; ``sync_error`` carries the
        *latest attempt's* error.

        ``include_packages`` (default ``False``): when True, spawn
        ``Rscript`` to list the renv project library. The default
        is deliberately False because this path is called from
        ``serialize_notebook_state`` and ``_serialize_environment_payload``
        (both fire on every state sync / env refresh / dependency
        mutation) and a synchronous Rscript spawn on each call would
        block the open response on a multi-second probe. The R env
        panel calls a dedicated ``GET /v1/notebooks/{id}/r-packages``
        route to fetch the package list separately when it mounts.

        ``sync_state`` is derived for the UI:

        * ``absent``    — no ``renv.lock`` on disk (Python-only).
        * ``never``     — lockfile exists, but no sync has ever
                          succeeded (last_synced_at == 0) and the
                          latest attempt didn't fail (no
                          sync_error). Typically the brand-new
                          state right after adding renv.lock.
        * ``ok``        — last sync matched the current lockfile
                          hash and no error.
        * ``outdated``  — last good sync was against a different
                          lockfile (user edited renv.lock); no
                          error from the latest attempt yet.
        * ``failed``    — the latest sync attempt failed
                          (``sync_error`` is set).
        """
        runtime = load_runtime_state(self.path).r
        lockfile = self.path / "renv.lock"
        has_lockfile = lockfile.exists()

        current_lock_hash = ""
        if has_lockfile:
            try:
                current_lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
            except OSError:
                pass

        sync_state: str
        if not has_lockfile:
            sync_state = "absent"
        elif runtime.sync_error:
            sync_state = "failed"
        elif runtime.last_synced_at == 0:
            sync_state = "never"
        elif current_lock_hash and runtime.lock_hash == current_lock_hash:
            sync_state = "ok"
        else:
            sync_state = "outdated"

        # Listing the project library is a ~1-2s Rscript spawn — the
        # default ``include_packages=False`` keeps state-sync paths
        # fast. The dedicated R-packages route opts in by passing
        # ``include_packages=True``. When the notebook has no
        # lockfile, skip the spawn even on opt-in — there's no R env
        # to enumerate.
        packages: list[dict[str, str]] = []
        packages_status = "absent"
        packages_error: str | None = None
        if include_packages and has_lockfile:
            listing = list_r_packages(self.path)
            packages = [{"name": pkg.name, "version": pkg.version} for pkg in listing.packages]
            packages_status = listing.status
            packages_error = listing.error

        return {
            "has_lockfile": has_lockfile,
            "current_lock_hash": current_lock_hash,
            # The fields below are last-successful-sync state.
            "lock_hash": runtime.lock_hash,
            # ``r_version`` is the version recorded at the last good
            # renv sync; ``system_r_version`` is the version of the
            # Rscript on PATH right now. Falls back so the UI can
            # always show *some* R version next to the status pill
            # — without this the pre-init R card said "Not set up"
            # with no other state, which read as broken even when
            # R cells were running fine against the system library.
            "r_version": runtime.r_version,
            "system_r_version": self._cached_system_r_version,
            "last_synced_at": runtime.last_synced_at,
            "sync_state": sync_state,
            "sync_error": runtime.sync_error or None,
            # Package list + listing-probe outcome. ``packages_status``
            # disambiguates "the probe failed" from "the library is
            # empty" — both produce an empty ``packages`` array.
            # ``packages_status`` values: ``"absent"`` (no lockfile or
            # ``include_packages=False`` — no probe attempted),
            # ``"ok"``, ``"rscript_missing"``, ``"renv_not_active"``,
            # or ``"failed"``. ``packages_error`` carries a short
            # message when ``packages_status == "failed"``.
            "packages": packages,
            "packages_status": packages_status,
            "packages_error": packages_error,
        }

    def serialize_environment_job_state(self) -> dict[str, Any] | None:
        """Serialize the current or most recent environment job when present."""
        with self._environment_state_lock:
            if self.environment_job is not None and self.environment_job.status == "running":
                return asdict(self.environment_job)
            if self.environment_job_history:
                return asdict(self.environment_job_history[0])
            return None

    def serialize_environment_job_history(self) -> list[dict[str, Any]]:
        """Serialize recent finished environment jobs, newest first."""
        with self._environment_state_lock:
            return [asdict(job) for job in self.environment_job_history]

    def _environment_job_history_path(self) -> Path:
        """Return the persisted recent-job history path for this notebook."""
        return self.path / ".strata" / "environment_jobs.json"

    def _load_environment_job_history(self) -> None:
        """Load recent finished environment jobs from notebook runtime state."""
        history_path = self._environment_job_history_path()
        if not history_path.exists():
            return
        try:
            raw = json.loads(history_path.read_text())
        except Exception:
            logger.warning(
                "Failed to read environment job history for %s", self.path, exc_info=True
            )
            return
        history = [EnvironmentJobSnapshot(**item) for item in raw]
        self.environment_job_history = [
            job for job in history if job.status in {"completed", "failed"}
        ][:_ENVIRONMENT_JOB_HISTORY_LIMIT]

    def _persist_environment_job_history(self) -> None:
        """Persist recent finished environment jobs to notebook runtime state."""
        history_path = self._environment_job_history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            json.dumps(
                [
                    asdict(job)
                    for job in self.environment_job_history[:_ENVIRONMENT_JOB_HISTORY_LIMIT]
                ],
                indent=2,
                sort_keys=True,
            )
        )

    def _record_finished_environment_job(self, job: EnvironmentJobSnapshot) -> None:
        """Add a finished job to recent history and persist it."""
        with self._environment_state_lock:
            remaining = [
                existing for existing in self.environment_job_history if existing.id != job.id
            ]
            self.environment_job_history = [job, *remaining][:_ENVIRONMENT_JOB_HISTORY_LIMIT]
            try:
                self._persist_environment_job_history()
            except Exception:
                logger.warning(
                    "Failed to persist environment job history for %s", self.path, exc_info=True
                )

    def has_active_environment_mutation(self) -> bool:
        """Return whether an environment change is currently in progress."""
        with self._environment_state_lock:
            return (
                self.environment_job is not None and self.environment_job.status == "running"
            ) or self._synchronous_environment_mutation is not None

    def _active_environment_mutation_label(self) -> str | None:
        """Return the label of the current environment mutation, if any."""
        with self._environment_state_lock:
            if self.environment_job is not None and self.environment_job.status == "running":
                if self.environment_job.action == "import":
                    return "environment import"
                if self.environment_job.package:
                    return f"{self.environment_job.action} {self.environment_job.package}"
                return self.environment_job.action
            return self._synchronous_environment_mutation

    def _has_active_execution(self) -> bool:
        """Return whether cell execution is currently active for this notebook."""
        if any(cell.status == CellStatus.RUNNING for cell in self.notebook_state.cells):
            return True
        try:
            from strata.notebook.ws import notebook_has_active_execution

            return notebook_has_active_execution(self.id)
        except Exception:
            return False

    def environment_execution_block_message(self) -> str | None:
        """Return the reason cell execution should be blocked, if any."""
        label = self._active_environment_mutation_label()
        if label is None:
            if self.environment_sync_state == "pending":
                return (
                    self.environment_sync_notice
                    or "Notebook environment is being created in the background. "
                    "Running cells is disabled until it finishes."
                )
            if (
                self.venv_python is None
                and self.environment_interpreter_source == "unknown"
                and self.environment_sync_state in {"failed", "unknown"}
            ):
                if self.environment_sync_error:
                    return f"Notebook environment is not ready. {self.environment_sync_error}"
                return (
                    "Notebook environment is not ready. Running cells is disabled "
                    "until it finishes initializing."
                )
            return None
        return f"Environment update in progress. Running cells is disabled until {label} finishes."

    def _assert_environment_job_can_start(self, action_label: str) -> None:
        """Reject starting a new environment update when the notebook is busy."""
        if self.has_active_environment_mutation():
            active_label = self._active_environment_mutation_label() or "environment update"
            raise RuntimeError(f"Another environment update is already in progress: {active_label}")
        if self._has_active_execution():
            raise RuntimeError(
                "Notebook execution is currently running. Wait for execution to "
                f"finish before starting {action_label}."
            )

    def _begin_synchronous_environment_mutation(self, label: str) -> None:
        """Reserve the notebook environment for a synchronous mutation path."""
        with self._environment_state_lock:
            self._assert_environment_job_can_start(label)
            self._synchronous_environment_mutation = label

    def _end_synchronous_environment_mutation(self) -> None:
        """Release the synchronous environment mutation reservation."""
        with self._environment_state_lock:
            self._synchronous_environment_mutation = None

    def serialize_worker_catalog(self) -> list[dict[str, Any]]:
        """Serialize the worker catalog visible to this notebook."""
        return build_worker_catalog(self.notebook_state)

    def _resolve_cached_outputs(
        self, cell_id: str, provenance_hash: str
    ) -> dict[str, tuple[str, int]] | None:
        """Return canonical output artifacts matching current provenance.

        The cache lookup is valid only if every consumed variable for this cell
        has a canonical artifact in this notebook whose provenance matches the
        per-variable hash used by the executor.
        """
        consumed_vars = self.dag.consumed_variables.get(cell_id, set()) if self.dag else set()
        if consumed_vars:
            first_var = sorted(consumed_vars)[0]
            lookup_hash = derive_subkey(provenance_hash, first_var)
        else:
            lookup_hash = provenance_hash

        cached = self.artifact_manager.find_cached(lookup_hash)
        if cached is None:
            return None

        if not consumed_vars:
            return {}

        notebook_id = self.notebook_state.id
        cached_outputs: dict[str, tuple[str, int]] = {}
        for var_name in sorted(consumed_vars):
            canonical_id = f"nb_{notebook_id}_cell_{cell_id}_var_{var_name}"
            expected_hash = derive_subkey(provenance_hash, var_name)
            canonical = self.artifact_manager.artifact_store.get_latest_version(
                canonical_id,
            )
            if canonical is None or canonical.provenance_hash != expected_hash:
                return None
            cached_outputs[var_name] = (canonical.id, canonical.version)

        return cached_outputs

    def _collect_input_hashes(self, cell_id: str) -> list[str]:
        """Provenance hashes from upstream artifacts (sweep refs grouped).

        The single source of truth for input-hash collection: the executor's
        provenance computation and causality explanations both delegate here, so
        a sweep downstream's *stored* hash and its *staleness recheck* agree. A
        reference sourced from a sweep group collapses to one deterministic
        ``sweep:<var>:<name>=<hash>;…`` string (otherwise the stored grouped hash
        would never match an ungrouped recompute → perpetual staleness).
        """
        from strata.notebook.dag import SweepProducer

        cell = self.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return []

        dag = self.dag
        hashes: list[str] = []
        sweep_buckets: dict[str, list[tuple[str, str]]] = {}

        def _hash_from_uri(uri: str) -> str | None:
            try:
                tail = uri.split("/")[-1]
                artifact_id = tail.split("@")[0]
                version = int(tail.split("@v=")[1])
            except (IndexError, ValueError):
                return None
            artifact = self.artifact_manager.artifact_store.get_artifact(artifact_id, version)
            return artifact.provenance_hash if artifact else None

        for upstream_id in cell.upstream_ids:
            upstream_cell = self.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            uri_items: list[tuple[str | None, str]] = list(upstream_cell.artifact_uris.items())
            if not uri_items and upstream_cell.artifact_uri:
                uri_items = [(None, upstream_cell.artifact_uri)]

            for var_name, uri in uri_items:
                provenance_hash = _hash_from_uri(uri)
                if provenance_hash is None:
                    continue
                if var_name is None:
                    hashes.append(provenance_hash)
                    continue
                producer = dag.variable_producer.get(var_name) if dag else None
                if isinstance(producer, SweepProducer):
                    variant_name = next(
                        (name for name, cid in producer.variants if cid == upstream_id),
                        None,
                    )
                    if variant_name is not None:
                        sweep_buckets.setdefault(var_name, []).append(
                            (variant_name, provenance_hash)
                        )
                        continue
                hashes.append(provenance_hash)

        for var_name, pairs in sweep_buckets.items():
            joined = ";".join(f"{name}={h}" for name, h in sorted(pairs))
            hashes.append(f"sweep:{var_name}:{joined}")

        return hashes

    def _collect_mount_fingerprints(self, cell: Any) -> tuple[list[str], bool]:
        """Return deterministic mount provenance components for a cell.

        Cell mounts already include notebook defaults from parser.py. Source
        annotations can override them again at execution time, so staleness
        must merge both layers exactly like the executor does.
        """
        annotations = parse_annotations(cell.source)
        merged_mounts = resolve_cell_mounts([], cell.mounts, annotations.mounts)

        mount_fingerprints: list[str] = []
        has_rw_mount = False
        for mount in sorted(merged_mounts, key=lambda m: m.name):
            fingerprint = MountFingerprinter.fingerprint_mount_sync(mount)
            if fingerprint is None:
                has_rw_mount = True
            else:
                mount_fingerprints.append(f"{mount.name}:{fingerprint}")

        return mount_fingerprints, has_rw_mount

    def _collect_table_fingerprints(self, cell: Any) -> list[str]:
        """Return ``@table`` snapshot fingerprints for a cell's provenance.

        Must mirror the executor's ``_compute_cell_provenance`` exactly: it
        folds ``table_fingerprints`` into the provenance hash, so staleness
        has to as well or an ``@table`` cell's stored artifacts are keyed
        under a hash this check never reproduces — the cell (and everything
        downstream) then resolves to idle forever. ``fingerprint_tables``
        never raises (random fingerprint on an unreachable catalog), so a
        lake outage shows the cell stale rather than crashing the recompute.
        """
        annotations = parse_annotations(cell.source)
        if not annotations.tables:
            return []
        from strata.notebook.tables import fingerprint_tables

        fingerprints, _ = fingerprint_tables(annotations.tables, self._lake_config())
        return fingerprints

    def _lake_config(self):
        """Server config when running inside the server, else loaded fresh."""
        try:
            from strata.server import get_state

            return get_state().config
        except RuntimeError:
            from strata.config import StrataConfig

            return StrataConfig.load()

    def _collect_runtime_env(self, cell: Any) -> dict[str, str]:
        """Return the provenance-relevant runtime env for a cell.

        The cell receives every notebook-level env var as ambient process
        environment at execution time, but only the keys it actually
        declares or references participate in its provenance hash. This
        prevents unrelated cells from being invalidated when an API key
        or similar ambient secret is added at the notebook level.
        """
        annotations = parse_annotations(cell.source)
        resolved = dict(cell.env)
        resolved.update(annotations.env)
        declared = set(annotations.env) | set(getattr(cell, "env_overrides", {}) or {})
        return narrow_env_for_provenance(cell.source, resolved, declared)

    def _effective_worker_name(self, cell: Any) -> str | None:
        """Return the effective worker name with annotation precedence."""
        annotations = parse_annotations(cell.source)
        if annotations.worker:
            return annotations.worker
        if cell.worker:
            return cell.worker
        return self.notebook_state.worker

    def _effective_worker_runtime_identity(self, cell: Any) -> str | None:
        """Return the worker runtime identity used in provenance."""
        return worker_runtime_identity(
            self.notebook_state,
            self._effective_worker_name(cell),
        )

    def record_execution(self, cell_id: str, duration_ms: float, cache_hit: bool) -> None:
        """Record a cell execution for profiling (v1.1).

        Args:
            cell_id: ID of the executed cell
            duration_ms: Execution duration in milliseconds
            cache_hit: Whether this was a cache hit
        """
        if cell_id not in self.execution_history:
            self.execution_history[cell_id] = []
        self.execution_history[cell_id].append(
            ExecutionSample(duration_ms=duration_ms, cache_hit=cache_hit)
        )

    def get_estimated_duration(self, cell_id: str) -> int:
        """Get estimated execution duration based on history.

        Args:
            cell_id: Cell ID

        Returns:
            Estimated duration in ms, or 0 if no history
        """
        history = self.execution_history.get(cell_id, [])
        for sample in reversed(history):
            if not sample.cache_hit:
                return int(sample.duration_ms)
        return 0

    def get_profiling_summary(self) -> dict:
        """Get notebook-level profiling summary (v1.1).

        Returns:
            Dict with total execution time, cache savings, artifact sizes,
            and per-cell profiling data.
        """
        total_execution_ms = 0
        cache_hits = 0
        cache_misses = 0
        total_artifact_bytes = 0

        cell_profiles = []
        for cell in self.notebook_state.cells:
            history = self.execution_history.get(cell.id, [])
            last_duration = history[-1].duration_ms if history else 0
            is_cached = history[-1].cache_hit if history else cell.cache_hit

            total_execution_ms += int(sum(sample.duration_ms for sample in history))
            cache_hits += sum(1 for sample in history if sample.cache_hit)
            cache_misses += sum(1 for sample in history if not sample.cache_hit)

            cell_name = cell.defines[0] if cell.defines else cell.id
            cell_profiles.append(
                {
                    "cell_id": cell.id,
                    "cell_name": cell_name,
                    "status": cell.status,
                    "duration_ms": int(last_duration),
                    "cache_hit": is_cached,
                    "artifact_uri": cell.artifact_uri,
                    "execution_count": len(history),
                }
            )

        # Estimate cache savings: sum of historical durations for cached cells
        cache_savings_ms = 0
        for cell in self.notebook_state.cells:
            history = self.execution_history.get(cell.id, [])
            last_non_cached_duration: int | None = None
            for sample in history:
                if sample.cache_hit:
                    if last_non_cached_duration is not None:
                        cache_savings_ms += last_non_cached_duration
                else:
                    last_non_cached_duration = int(sample.duration_ms)

        return {
            "total_execution_ms": int(total_execution_ms),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_savings_ms": cache_savings_ms,
            "total_artifact_bytes": total_artifact_bytes,
            "cell_profiles": cell_profiles,
        }

    def ensure_venv_synced(self) -> None:
        """Ensure venv is set up by running ``uv sync``.

        Idempotent — typically <1 s when venv already exists.
        On failure the session still opens (venv_python falls back to
        ``python`` in PATH) so tests without ``uv`` keep working.
        """
        started = _time.perf_counter()
        ok = _uv_sync(
            self.path,
            python_version=read_requested_python_minor(self.path),
        )
        self._apply_uv_sync_result(
            ok,
            duration_ms=int((_time.perf_counter() - started) * 1000),
        )

    def _apply_uv_sync_result(self, ok: bool, *, duration_ms: int) -> None:
        """Update runtime state after a uv sync attempt."""
        self.environment_last_synced_at = int(_time.time() * 1000)
        self.environment_last_sync_duration_ms = duration_ms

        venv_python = self.path / ".venv" / "bin" / "python"
        if venv_python.exists():
            self.venv_python = venv_python
            self.environment_interpreter_source = "venv"
            self.environment_python_version = self._probe_python_version(venv_python)
            self.environment_sync_state = "ready"
            self.environment_sync_error = None
            if ok:
                self.environment_sync_notice = None
            else:
                self.environment_sync_notice = (
                    "Environment refresh failed, but the existing notebook venv is "
                    "still available and will be used."
                )
                logger.warning(
                    "uv sync failed for %s, using existing notebook venv",
                    self.path,
                )
            return

        if ok:
            self.venv_python = Path("python")
            self.environment_interpreter_source = "path"
            self.environment_sync_state = "fallback"
            self.environment_sync_error = (
                "uv sync succeeded but the notebook venv interpreter was not "
                "found; using python from PATH."
            )
            self.environment_sync_notice = None
            self.environment_python_version = self._probe_python_version(self.venv_python)
            logger.warning(
                "uv sync succeeded but .venv/bin/python not found in %s",
                self.path,
            )
            return

        self.venv_python = Path("python")
        self.environment_interpreter_source = "path"
        self.environment_sync_state = "failed"
        self.environment_sync_error = (
            "Environment refresh failed and no notebook venv is available; "
            "notebook execution will fall back to python from PATH."
        )
        self.environment_sync_notice = None
        self.environment_python_version = self._probe_python_version(self.venv_python)
        logger.warning(
            "uv sync failed and no notebook venv is available for %s",
            self.path,
        )

    def ensure_renv_synced(self) -> None:
        """Ensure the notebook's R environment matches its ``renv.lock``.

        Mirror of ``ensure_venv_synced`` for the R side. No-op when the
        notebook has no ``renv.lock``. When the lockfile exists:

        1. Hash the lockfile bytes. If the stored ``r.lock_hash`` in
           ``.strata/runtime.json`` matches **and** the project's
           ``renv/library`` directory still exists, skip the
           ``Rscript`` spawn entirely — reopens against an
           unchanged-and-still-installed lockfile are free.
        2. Otherwise call ``_renv_sync`` synchronously. On success,
           write the new hash + sync timestamp + R version into
           ``runtime.json`` and clear any prior ``sync_error``. On
           failure, record the error in ``runtime.json`` but leave
           the last-good ``lock_hash`` / ``r_version`` /
           ``last_synced_at`` alone — the UI distinguishes "last
           good state was X, latest attempt failed" from "never
           synced".

        Runtime sync state lives in ``.strata/runtime.json``
        (per-session, gitignored) rather than the committed
        ``notebook.toml``. Re-opens with a cached library therefore
        do not churn ``notebook.toml`` — its ``updated_at`` only
        bumps on real structural edits.
        """
        lockfile = self.path / "renv.lock"
        if not lockfile.exists():
            # Python-only notebook (or R notebook pre-init). Clear any
            # stale R runtime state so a notebook that previously had a
            # lockfile and now doesn't doesn't keep a phantom hash or
            # error message.
            self._clear_r_runtime_if_present()
            return

        try:
            lock_bytes = lockfile.read_bytes()
        except OSError as exc:
            logger.warning("Could not read renv.lock to hash: %s", exc)
            return
        lock_hash = hashlib.sha256(lock_bytes).hexdigest()

        previous = load_runtime_state(self.path).r
        if (
            previous.lock_hash == lock_hash
            and not previous.sync_error
            and self._renv_library_present()
        ):
            # Cached restore — the on-disk library already matches the
            # lockfile, the last sync succeeded, AND the project
            # library directory hasn't been deleted out from under us.
            # Skip the ~1-2s Rscript spawn. Matters in particular for
            # the reuse-existing-session path which fires on every
            # reopen.
            logger.debug(
                "renv sync skipped for %s — lockfile hash unchanged + library present (%s)",
                self.path,
                lock_hash[:12],
            )
            return

        started = _time.perf_counter()
        ok = _renv_sync(self.path)
        duration_ms = int((_time.perf_counter() - started) * 1000)

        if not ok:
            # ``_renv_sync`` already logged the cause (Rscript missing,
            # timeout, non-zero exit). Record the failure in
            # runtime.json so the UI can render the failed state,
            # but keep the last-good ``lock_hash`` / ``r_version`` /
            # ``last_synced_at`` so users see "you had a working env
            # at <T>, the most recent attempt against <new lockfile>
            # failed" rather than losing the prior state.
            err_message = (
                f"renv::restore() failed after {duration_ms}ms. "
                "Check Rscript is on PATH and renv.lock is well-formed."
            )
            self._record_r_sync_failure(err_message)
            logger.warning(
                "renv sync failed for %s after %dms; R cells will run against the system R library",
                self.path,
                duration_ms,
            )
            return

        self._persist_r_runtime_success(
            lock_hash=lock_hash,
            last_synced_at=int(_time.time() * 1000),
            r_version=self._probe_r_version(),
        )

    def _renv_library_present(self) -> bool:
        """Probe whether the project's renv library exists *and* has content.

        renv installs into ``<notebook>/renv/library/<platform>/<R>/<pkg>``.
        An empty ``renv/library`` directory is meaningless — it can
        survive a wiped or never-completed restore while the
        runtime metadata claims a successful sync. Requiring the
        directory to be non-empty catches the "renv/library was
        recreated empty (test fixture, mid-aborted restore,
        manual cleanup script that left the parent dir)" case the
        bare existence check missed.

        This is still a cheap probe — it stops at the first directory
        entry. Validating each package's integrity would require
        spawning ``renv::status()``, defeating the short-circuit
        purpose (the whole point is to skip Rscript on a fast path).
        """
        library = self.path / "renv" / "library"
        if not library.is_dir():
            return False
        try:
            # ``next(iter(...))`` returns the first child entry or
            # raises ``StopIteration`` if the dir is empty.
            next(iter(library.iterdir()))
        except StopIteration:
            return False
        except OSError:
            # Permission denied / I/O error — assume not usable
            # rather than short-circuiting against a broken library.
            return False
        return True

    def _probe_r_version(self) -> str | None:
        """Best-effort: ask ``Rscript`` for its version string.

        Returns ``None`` when ``Rscript`` isn't on PATH or the probe
        fails for any reason — ``RRuntime.r_version`` tolerates an
        empty string, so a failed probe records lock_hash + timestamp
        without it.
        """
        rscript = shutil.which("Rscript")
        if rscript is None:
            return None
        try:
            proc = subprocess.run(  # noqa: S603 — rscript resolved via shutil.which
                [rscript, "-e", "cat(R.version$major, R.version$minor, sep='.')"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.debug("R version probe failed: %s", exc)
            return None
        if proc.returncode != 0:
            return None
        version = proc.stdout.strip()
        return version or None

    def _persist_r_runtime_success(
        self,
        *,
        lock_hash: str,
        last_synced_at: int,
        r_version: str | None,
    ) -> None:
        """Persist R sync state after a successful ``renv::restore()``.

        Overwrites the runtime entry — clears any prior ``sync_error``
        (the latest attempt succeeded) and stamps the new hash +
        timestamp + R version.
        """
        try:
            state = load_runtime_state(self.path)
            state.r = RRuntime(
                lock_hash=lock_hash,
                r_version=r_version or "",
                last_synced_at=last_synced_at,
                sync_error="",
            )
            save_runtime_state(self.path, state)
        except Exception as exc:
            logger.warning("Skipping R runtime persist; write failed: %s", exc)

    def _record_r_sync_failure(self, error: str) -> None:
        """Record a failed ``renv::restore()`` attempt.

        Keeps the last-good ``lock_hash`` / ``r_version`` /
        ``last_synced_at`` so the UI can show "you had a working
        env at <T>, the most recent attempt failed". Sets the
        ``sync_error`` field so the UI knows there's a problem.
        """
        try:
            state = load_runtime_state(self.path)
            state.r = RRuntime(
                lock_hash=state.r.lock_hash,
                r_version=state.r.r_version,
                last_synced_at=state.r.last_synced_at,
                sync_error=error,
            )
            save_runtime_state(self.path, state)
        except Exception as exc:
            logger.warning("Skipping R runtime failure record; write failed: %s", exc)

    def _clear_r_runtime_if_present(self) -> None:
        """Reset the R runtime entry when the notebook has no ``renv.lock``.

        Default ``RRuntime()`` matches the Python-only baseline. No-op
        when the entry is already empty so a Python-only open doesn't
        churn ``runtime.json``.
        """
        try:
            state = load_runtime_state(self.path)
            if state.r == RRuntime():
                return
            state.r = RRuntime()
            save_runtime_state(self.path, state)
        except Exception as exc:
            logger.debug("R runtime clear skipped: %s", exc)

    def refresh_environment_runtime(self) -> None:
        """Refresh runtime metadata from an existing notebook venv.

        Dependency mutations already run ``uv add`` / ``uv remove``, which
        update ``pyproject.toml``, rewrite ``uv.lock``, and sync ``.venv``.
        Re-running ``uv sync`` immediately afterwards is redundant and can be
        expensive, so the fast path just reuses the already-updated notebook
        interpreter and re-probes lightweight runtime metadata.

        If the notebook venv is unexpectedly missing, fall back to the normal
        ``ensure_venv_synced()`` path so correctness wins over speed.
        """
        venv_python = self.path / ".venv" / "bin" / "python"
        if not venv_python.exists():
            if self.has_active_environment_mutation():
                self.mark_environment_pending()
                return
            logger.warning(
                "Notebook venv missing after dependency change for %s; falling back to uv sync",
                self.path,
            )
            self.ensure_venv_synced()
            return

        started = _time.perf_counter()
        self.venv_python = venv_python
        self.environment_interpreter_source = "venv"
        persisted = self._read_persisted_environment_metadata()
        persisted_runtime_python = (
            persisted.runtime_python_version or persisted.python_version
        ).strip()
        self.environment_python_version = persisted_runtime_python or self._probe_python_version(
            venv_python
        )
        self.environment_sync_state = "ready"
        self.environment_sync_error = None
        self.environment_sync_notice = None
        self.environment_last_synced_at = persisted.last_synced_at or int(_time.time() * 1000)
        self.environment_last_sync_duration_ms = int((_time.perf_counter() - started) * 1000)

    def _should_start_warm_pool(self) -> bool:
        """Return whether the notebook has a stable enough runtime for warm workers."""
        if self.has_active_environment_mutation():
            return False
        return self.environment_sync_state in {"ready", "fallback"}

    async def _ensure_warm_pool_started(self) -> None:
        """Create and start the warm process pool when the runtime is ready."""
        if self.warm_pool is not None or not self._should_start_warm_pool():
            return

        from strata.notebook.pool import WarmProcessPool

        self.warm_pool = WarmProcessPool(
            notebook_dir=self.path,
            pool_size=2,
            python_executable=self.venv_python or Path("python"),
        )
        try:
            task = asyncio.get_running_loop().create_task(self.warm_pool.start())
            self.warm_pool.track_background_task(task)
        except RuntimeError:
            pass  # No running loop; pool stays cold until first acquire
        self.start_r_pool_background()

    def _has_r_cells(self) -> bool:
        """Whether any cell in the notebook is an R cell."""
        from strata.notebook.models import CellLanguage

        return any(cell.language == CellLanguage.R for cell in self.notebook_state.cells)

    def start_r_pool_background(self) -> None:
        """Create and start the warm R pool when the notebook needs one.

        Mirrors the Python pool but is gated harder: only notebooks that
        actually contain R cells (and machines with Rscript) pay for warm
        R workers. Safe to call repeatedly — it no-ops once created.
        """
        if self.r_warm_pool is not None or not self._should_start_warm_pool():
            return
        if not self._has_r_cells():
            return

        import shutil as _shutil

        rscript = _shutil.which("Rscript")
        if rscript is None:
            return

        from strata.notebook.pool import WarmProcessPool

        pool_worker = Path(__file__).parent / "languages" / "r" / "pool_worker.R"
        self.r_warm_pool = WarmProcessPool(
            notebook_dir=self.path,
            pool_size=2,
            worker_command=[rscript, str(pool_worker), str(self.path)],
            # R startup + renv activation can take far longer than the
            # Python worker's import warm-up.
            ready_timeout_seconds=60.0,
        )
        try:
            task = asyncio.get_running_loop().create_task(self.r_warm_pool.start())
            self.r_warm_pool.track_background_task(task)
        except RuntimeError:
            pass  # No running loop; pool stays cold until first acquire

    async def _invalidate_warm_pool_for_environment_change(self) -> None:
        """Invalidate the warm pools after the runtime environment changes."""
        if self.warm_pool is not None:
            try:
                self.warm_pool.python_executable = str(self.venv_python or Path("python"))
                await self.warm_pool.invalidate()
                logger.info("Warm pool invalidated after environment change")
            except Exception:
                logger.exception("Failed to invalidate warm pool")
        if self.r_warm_pool is not None:
            try:
                await self.r_warm_pool.invalidate()
                logger.info("R warm pool invalidated after environment change")
            except Exception:
                logger.exception("Failed to invalidate R warm pool")

    async def sync_environment(self) -> dict[str, CellStaleness]:
        """Re-sync the notebook environment and refresh runtime metadata."""
        old_hash = compute_lockfile_hash(self.path)
        await asyncio.to_thread(self.ensure_venv_synced)
        await self._invalidate_warm_pool_for_environment_change()
        try:
            await self._ensure_warm_pool_started()
        except Exception:
            logger.warning("Failed to start warm pool after sync for %s", self.path, exc_info=True)

        try:
            await asyncio.to_thread(update_environment_metadata, self.path)
        except Exception:
            logger.exception("Failed to update environment metadata")

        new_hash = compute_lockfile_hash(self.path)
        if new_hash != old_hash:
            return self.compute_staleness()
        return {}

    async def on_dependencies_changed(self) -> None:
        """React to dependency changes (lockfile updated).

        Refreshes runtime metadata from the already-updated notebook venv,
        invalidates the warm pool, and recomputes lockfile hash for
        provenance. Called after ``uv add`` / ``uv remove``.
        """
        # 1. Dependency mutation already synced .venv. Reuse that interpreter
        #    instead of immediately running a second uv sync.
        await asyncio.to_thread(self.refresh_environment_runtime)
        await self._invalidate_warm_pool_for_environment_change()

        # 2. Recompute lockfile hash (triggers cache invalidation on next exec)
        new_hash = compute_lockfile_hash(self.path)
        logger.info("Lockfile hash updated to %.12s after dependency change", new_hash)

        # 3. Persist environment metadata in notebook.toml
        try:
            await asyncio.to_thread(update_environment_metadata, self.path)
        except Exception:
            logger.exception("Failed to update environment metadata")

    async def mutate_dependency(self, package: str, *, action: str) -> DependencyMutationOutcome:
        """Apply a dependency mutation without blocking the event loop."""
        from strata.notebook.dependencies import add_dependency, remove_dependency

        # add_dependency and remove_dependency have different keyword-only params,
        # so a shared `op` variable narrows to a union callable that ty won't pass
        # to asyncio.to_thread. Dispatch at the call site so each to_thread sees a
        # single concrete signature (both accept (path, package) positionally).
        if action == "add":
            result = await asyncio.to_thread(add_dependency, self.path, package)
        elif action == "remove":
            result = await asyncio.to_thread(remove_dependency, self.path, package)
        else:
            raise ValueError(f"Unknown dependency action: {action}")

        staleness_map: dict[str, CellStaleness] = {}
        if getattr(result, "success", False) and getattr(result, "lockfile_changed", False):
            await self.on_dependencies_changed()
            staleness_map = self.compute_staleness()

        return DependencyMutationOutcome(
            result=result,
            staleness_map=staleness_map,
        )

    async def import_requirements(self, requirements_text: str) -> RequirementsImportOutcome:
        """Replace direct notebook dependencies from requirements text."""
        result = await asyncio.to_thread(
            import_requirements_text,
            self.path,
            requirements_text,
        )

        staleness_map: dict[str, CellStaleness] = {}
        if getattr(result, "success", False) and getattr(result, "lockfile_changed", False):
            await self.on_dependencies_changed()
            staleness_map = self.compute_staleness()

        return RequirementsImportOutcome(
            result=result,
            staleness_map=staleness_map,
        )

    async def import_environment_yaml(
        self, environment_yaml_text: str
    ) -> RequirementsImportOutcome:
        """Best-effort import of Conda-style ``environment.yaml``."""
        result = await asyncio.to_thread(
            import_environment_yaml_text,
            self.path,
            environment_yaml_text,
        )

        staleness_map: dict[str, CellStaleness] = {}
        if getattr(result, "success", False) and getattr(result, "lockfile_changed", False):
            await self.on_dependencies_changed()
            staleness_map = self.compute_staleness()

        return RequirementsImportOutcome(
            result=result,
            staleness_map=staleness_map,
        )

    def wait_for_environment_job_task(self) -> asyncio.Task[None] | None:
        """Return the current environment job task, if any."""
        with self._environment_state_lock:
            return self.environment_job_task

    async def wait_for_environment_job(self) -> None:
        """Wait for the currently active environment job to finish."""
        task = self.wait_for_environment_job_task()
        if task is not None:
            await task

    async def submit_environment_job(
        self,
        *,
        action: str,
        package: str | None = None,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
        python_version: str | None = None,
    ) -> EnvironmentJobSnapshot:
        """Start an asynchronous notebook environment job."""
        # R-side actions (``r_init``, ``r_add``) reuse the env-job
        # machinery — same job tracking, same WS broadcast frames,
        # same staleness propagation. The dispatch in
        # ``_run_environment_job`` switches on ``action`` to call
        # the right helper.
        valid_actions = {
            "add",
            "remove",
            "sync",
            "import",
            "change_python",
            "r_init",
            "r_add",
        }
        if action not in valid_actions:
            raise ValueError(f"Unsupported environment job action: {action}")

        if action == "import":
            if (requirements_text is None) == (environment_yaml_text is None):
                raise ValueError(
                    "Import environment jobs require exactly one of requirements_text "
                    "or environment_yaml_text"
                )
        if action == "change_python" and not python_version:
            raise ValueError("change_python jobs require python_version")

        # R-side validation: catch obvious user errors at submission
        # rather than letting the subprocess fail mid-job.
        if action in {"r_init", "r_add"} and not shutil.which("Rscript"):
            raise ValueError(
                "Rscript not found on PATH. Install R "
                "(https://cran.r-project.org/) before initialising renv."
            )
        if action == "r_add":
            from strata.notebook.dependencies import is_valid_r_package_name

            if not package or not is_valid_r_package_name(package):
                raise ValueError(
                    "r_add requires a valid R package name (match ``[A-Za-z][A-Za-z0-9.]*``)."
                )
            if not (self.path / "renv.lock").exists():
                raise ValueError(
                    "renv not initialised in this notebook. Click "
                    "'Initialize renv' before adding R packages."
                )

        if action == "import":
            action_label = (
                "requirements import"
                if requirements_text is not None
                else "environment.yaml import"
            )
        elif action == "change_python":
            action_label = f"change Python to {python_version}"
        elif action == "r_init":
            action_label = "renv::init"
        elif action == "r_add":
            action_label = f"renv::install {package}"
        else:
            action_label = f"{action} {package}".strip()
        with self._environment_state_lock:
            self._assert_environment_job_can_start(action_label)
            requested_python = read_requested_python_minor(self.path)
            command = "uv sync"
            if action == "add" and package:
                command = f"uv add {package}"
            elif action == "remove" and package:
                command = f"uv remove {package}"
            elif action == "sync" and requested_python:
                command = f"uv sync --python {requested_python}"
            elif action == "change_python":
                command = f"uv sync (python {python_version})"
            elif action == "r_init":
                command = "Rscript -e 'renv::init(bare = TRUE)'"
            elif action == "r_add":
                command = (
                    f'Rscript -e \'renv::install("{package}"); '
                    'renv::snapshot(type = "all", prompt = FALSE)\''
                )

            job = EnvironmentJobSnapshot(
                id=str(uuid.uuid4()),
                action=action,
                package=package,
                command=command,
                status="running",
                phase="uv_running",
                started_at=int(_time.time() * 1000),
            )
            self.environment_job = job

        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_STARTED, job)
        task = asyncio.create_task(
            self._run_environment_job(
                job,
                requirements_text=requirements_text,
                environment_yaml_text=environment_yaml_text,
                python_version=python_version,
            )
        )
        with self._environment_state_lock:
            self.environment_job_task = task
        return job

    async def _run_environment_job(
        self,
        job: EnvironmentJobSnapshot,
        *,
        requirements_text: str | None = None,
        environment_yaml_text: str | None = None,
        python_version: str | None = None,
    ) -> None:
        """Execute a background environment job and publish updates."""
        stale_cell_ids: list[str] = []
        import_result: RequirementsImportResult | None = None
        try:
            if job.action == "sync":
                stale_cell_ids = await self._run_sync_environment_job(job)
            elif job.action == "import":
                stale_cell_ids, import_result = await self._run_import_environment_job(
                    job,
                    requirements_text=requirements_text,
                    environment_yaml_text=environment_yaml_text,
                )
            elif job.action == "change_python":
                assert python_version is not None
                stale_cell_ids = await self._run_change_python_environment_job(
                    job, new_minor=python_version
                )
            elif job.action in {"r_init", "r_add"}:
                stale_cell_ids = await self._run_r_environment_job(job)
            else:
                assert job.package is not None
                stale_cell_ids = await self._run_dependency_environment_job(
                    job,
                    action=job.action,
                    package=job.package,
                )
            job.status = "completed"
            job.phase = "completed"
        except Exception as exc:
            logger.exception("Environment job %s failed for %s", job.action, self.path)
            job.status = "failed"
            job.phase = "failed"
            job.error = str(exc)
        finally:
            job.finished_at = int(_time.time() * 1000)
            job.duration_ms = job.finished_at - job.started_at
            self._record_finished_environment_job(job)
            payload: dict[str, Any] = {
                "environment_job": asdict(job),
                "environment_job_history": self.serialize_environment_job_history(),
                "cells": self.serialize_cells(),
                **{
                    "lockfile_changed": job.lockfile_changed,
                    "stale_cell_count": job.stale_cell_count,
                    "stale_cell_ids": stale_cell_ids,
                },
            }
            if import_result is not None:
                payload["warnings"] = list(import_result.warnings)
                payload["imported_count"] = import_result.imported_count
            if job.status == "completed":
                payload.update(
                    {
                        "environment": self.serialize_environment_state(),
                        # Include r_environment in the finished payload so
                        # successful r_init / r_add jobs flip the R panel
                        # to the synced state without a manual reopen.
                        # Pre-PR H this field was omitted; the store's
                        # ``syncNotebookREnvironmentFromBackend`` skipped
                        # the update and the card kept showing the
                        # pre-job state (System R / outdated).
                        "r_environment": self.serialize_r_environment_state(),
                        "dependencies": [
                            {
                                "name": dep.name,
                                "version": str(dep.version) if dep.version else None,
                                "specifier": str(dep.specifier) if dep.specifier else None,
                            }
                            for dep in list_dependencies(self.path)
                        ],
                    }
                )
                from strata.notebook.dependencies import list_resolved_dependencies

                payload["resolved_dependencies"] = [
                    {
                        "name": dep.name,
                        "version": str(dep.version) if dep.version else None,
                        "specifier": str(dep.specifier) if dep.specifier else None,
                    }
                    for dep in list_resolved_dependencies(self.path)
                ]
            await self._broadcast_environment_job_message(
                MessageType.ENVIRONMENT_JOB_FINISHED,
                payload,
            )
            if job.action in {"add", "remove"}:
                legacy_payload = {
                    "action": job.action,
                    "package": job.package,
                    "success": job.status == "completed",
                    "error": job.error,
                    "lockfile_changed": job.lockfile_changed,
                    "stale_cell_count": job.stale_cell_count,
                    "cells": payload["cells"],
                }
                if "environment" in payload:
                    legacy_payload["environment"] = payload["environment"]
                    legacy_payload["dependencies"] = payload.get("dependencies", [])
                    legacy_payload["resolved_dependencies"] = payload.get(
                        "resolved_dependencies", []
                    )
                await self._broadcast_environment_job_message(
                    MessageType.DEPENDENCY_CHANGED,
                    legacy_payload,
                )
                await self._broadcast_environment_staleness_updates(job.stale_cell_ids)
            with self._environment_state_lock:
                if self.environment_job is job:
                    self.environment_job = None
                current_task = asyncio.current_task()
                if self.environment_job_task is current_task:
                    self.environment_job_task = None

    async def _run_r_environment_job(self, job: EnvironmentJobSnapshot) -> list[str]:
        """Run ``r_init`` / ``r_add`` as a background job.

        Parallel to ``_run_dependency_environment_job`` for the R side.
        ``renv_init`` / ``renv_add`` are native async streaming calls
        (PR G) — the ``on_update`` callback fires per stdout/stderr
        chunk so ``environment_job_progress`` frames go out live
        during a multi-minute ``arrow`` compile, and the R card's
        stdout tail in the env panel actually populates during the
        run instead of only at the end.

        Staleness propagation reuses ``_finalize_environment_job``:
        when ``renv.lock`` changes, ``compute_lockfile_hash`` (which
        folds renv.lock since #78) reports a different hash and
        every cell's env_hash drifts. Python cells go stale alongside
        R cells — that's an over-stale gap from sharing one
        lockfile hash; making the env hash language-aware is a
        Phase 2 follow-up (issue to file).
        """
        from strata.notebook.dependencies import renv_add, renv_init

        old_lockfile_hash = compute_lockfile_hash(self.path)
        on_update = lambda stream, text, truncated: self._update_environment_job_stream(  # noqa: E731
            job,
            stream=stream,
            text=text,
            truncated=truncated,
        )

        if job.action == "r_init":
            result = await renv_init(self.path, on_update=on_update)
        elif job.action == "r_add":
            assert job.package is not None
            result = await renv_add(self.path, job.package, on_update=on_update)
        else:  # pragma: no cover — guarded by submit_environment_job
            raise RuntimeError(f"Unsupported R job action: {job.action!r}")

        if result.operation_log is not None:
            self._apply_environment_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or f"{job.action} failed")

        # After a successful ``renv::init`` / ``renv::install`` + snapshot
        # the project library matches the new ``renv.lock``. Persist that
        # as the last-good sync so ``sync_state`` reports ``ok`` (not
        # ``never`` / ``outdated``) and the next session open doesn't
        # spuriously re-run ``renv::restore()`` to rebuild what we just
        # built. The R version comes from the cached system probe — by
        # construction the library was just populated with this R.
        renv_lock = self.path / "renv.lock"
        if renv_lock.exists():
            new_lockfile_hash = hashlib.sha256(renv_lock.read_bytes()).hexdigest()
            self._persist_r_runtime_success(
                lock_hash=new_lockfile_hash,
                last_synced_at=int(_time.time() * 1000),
                r_version=self._cached_system_r_version,
            )

        job.lockfile_changed = compute_lockfile_hash(self.path) != old_lockfile_hash
        return await self._finalize_environment_job(job, lockfile_changed=job.lockfile_changed)

    async def _run_dependency_environment_job(
        self,
        job: EnvironmentJobSnapshot,
        *,
        action: str,
        package: str,
    ) -> list[str]:
        """Run ``uv add`` / ``uv remove`` as a background job."""
        timeout = 120
        display_name = f"uv {action}"
        old_lockfile_hash = compute_lockfile_hash(self.path)
        lock = _get_notebook_lock(self.path)
        await asyncio.to_thread(lock.acquire)
        try:
            on_update = lambda stream, text, truncated: self._update_environment_job_stream(  # noqa: E731
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            )
            if action == "add":
                result = await self.backend.add_streaming(
                    package, timeout=timeout, on_update=on_update
                )
            elif action == "remove":
                result = await self.backend.remove_streaming(
                    package, timeout=timeout, on_update=on_update
                )
            else:
                raise RuntimeError(f"Unsupported dependency action: {action!r}")
        finally:
            lock.release()

        self._apply_environment_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or f"{display_name} failed")

        job.lockfile_changed = compute_lockfile_hash(self.path) != old_lockfile_hash
        return await self._finalize_environment_job(job, lockfile_changed=job.lockfile_changed)

    async def _run_import_environment_job(
        self,
        job: EnvironmentJobSnapshot,
        *,
        requirements_text: str | None,
        environment_yaml_text: str | None,
    ) -> tuple[list[str], RequirementsImportResult]:
        """Run a requirements/environment.yaml import as a background job."""
        job.phase = "preparing_import"
        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)

        if requirements_text is not None:
            result = await import_requirements_text_streaming(
                self.path,
                requirements_text,
                on_update=lambda stream, text, truncated: self._update_environment_job_stream(
                    job,
                    stream=stream,
                    text=text,
                    truncated=truncated,
                ),
            )
        else:
            assert environment_yaml_text is not None
            result = await import_environment_yaml_text_streaming(
                self.path,
                environment_yaml_text,
                on_update=lambda stream, text, truncated: self._update_environment_job_stream(
                    job,
                    stream=stream,
                    text=text,
                    truncated=truncated,
                ),
            )

        self._apply_environment_operation_log(job, result.operation_log)
        if not result.success:
            raise RuntimeError(result.error or "Environment import failed")

        stale_cell_ids = await self._finalize_environment_job(
            job,
            lockfile_changed=result.lockfile_changed,
        )
        return stale_cell_ids, result

    async def _run_change_python_environment_job(
        self,
        job: EnvironmentJobSnapshot,
        *,
        new_minor: str,
    ) -> list[str]:
        """Change ``requires-python`` + rebuild the venv on the new minor.

        Rollback policy on uv sync failure: restore the previous
        ``requires-python`` and re-sync to the old interpreter. Best-
        effort — if the rollback sync also fails the notebook is left
        with the old pyproject and no venv, and we surface the error
        via the job's operation log.
        """
        from strata.notebook.writer import update_requires_python

        old_minor = read_requested_python_minor(self.path)
        await asyncio.to_thread(update_requires_python, self.path, new_minor)

        # Wipe the existing .venv so uv sync builds against the new
        # interpreter rather than reporting an "interpreter mismatch"
        # error and refusing to proceed.
        venv_dir = self.path / ".venv"
        if venv_dir.exists():
            await asyncio.to_thread(shutil.rmtree, venv_dir, ignore_errors=True)

        old_lockfile_hash = compute_lockfile_hash(self.path)
        lock = _get_notebook_lock(self.path)
        await asyncio.to_thread(lock.acquire)
        try:
            on_update = lambda stream, text, truncated: self._update_environment_job_stream(  # noqa: E731
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            )
            result = await self.backend.sync_streaming(
                python_version=None,
                timeout=180,
                on_update=on_update,
            )
        finally:
            lock.release()

        self._apply_environment_operation_log(job, result.operation_log)
        if not result.success:
            # Rollback: restore previous requires-python and re-sync.
            if old_minor:
                try:
                    await asyncio.to_thread(update_requires_python, self.path, old_minor)
                    rollback = await self.backend.sync_streaming(
                        python_version=None,
                        timeout=180,
                        on_update=on_update,
                    )
                    self._apply_environment_operation_log(job, rollback.operation_log)
                except Exception:
                    logger.exception("Failed to rollback python-version change for %s", self.path)
            raise RuntimeError(result.error or f"uv sync failed for Python {new_minor}")

        job.lockfile_changed = compute_lockfile_hash(self.path) != old_lockfile_hash
        return await self._finalize_environment_job(
            job,
            lockfile_changed=True,
        )

    async def _run_sync_environment_job(
        self,
        job: EnvironmentJobSnapshot,
    ) -> list[str]:
        """Run ``uv sync`` as a background job."""
        old_lockfile_hash = compute_lockfile_hash(self.path)
        requested_python = read_requested_python_minor(self.path)
        result = await self.backend.sync_streaming(
            python_version=requested_python,
            timeout=60,
            on_update=lambda stream, text, truncated: self._update_environment_job_stream(
                job,
                stream=stream,
                text=text,
                truncated=truncated,
            ),
        )
        self._apply_environment_operation_log(job, result.operation_log)
        self._apply_uv_sync_result(
            result.success,
            duration_ms=result.operation_log.duration_ms or 0,
        )
        if not result.success:
            raise RuntimeError(result.error or "uv sync failed")

        return await self._finalize_environment_job(
            job,
            lockfile_changed=compute_lockfile_hash(self.path) != old_lockfile_hash,
            refresh_runtime=False,
        )

    async def _finalize_environment_job(
        self,
        job: EnvironmentJobSnapshot,
        *,
        lockfile_changed: bool,
        refresh_runtime: bool = True,
    ) -> list[str]:
        """Refresh runtime metadata and staleness after a successful env mutation."""
        if refresh_runtime:
            job.phase = "refreshing_runtime"
            await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
            await asyncio.to_thread(self.refresh_environment_runtime)

        job.phase = "invalidating_warm_pool"
        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        await self._invalidate_warm_pool_for_environment_change()

        job.phase = "recomputing_staleness"
        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        try:
            await asyncio.to_thread(update_environment_metadata, self.path)
        except Exception:
            logger.exception("Failed to update environment metadata")

        staleness_map = self.compute_staleness()
        stale_cell_ids = [
            cell_id
            for cell_id, staleness in staleness_map.items()
            if staleness.status != CellStatus.READY
        ]
        job.phase = "starting_warm_pool"
        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)
        try:
            await self._ensure_warm_pool_started()
        except Exception:
            logger.warning(
                "Failed to start warm pool after environment job for %s",
                self.path,
                exc_info=True,
            )
        job.lockfile_changed = lockfile_changed
        job.stale_cell_count = len(stale_cell_ids)
        job.stale_cell_ids = stale_cell_ids
        return stale_cell_ids

    def _apply_environment_operation_log(
        self,
        job: EnvironmentJobSnapshot,
        operation_log: EnvironmentOperationLog | None,
    ) -> None:
        """Copy final command log details onto a job snapshot."""
        if operation_log is None:
            return
        job.command = operation_log.command or job.command
        job.duration_ms = operation_log.duration_ms
        job.stdout = operation_log.stdout
        job.stderr = operation_log.stderr
        job.stdout_truncated = operation_log.stdout_truncated
        job.stderr_truncated = operation_log.stderr_truncated

    async def _update_environment_job_stream(
        self,
        job: EnvironmentJobSnapshot,
        *,
        stream: str,
        text: str,
        truncated: bool,
    ) -> None:
        """Update a running job's live stdout/stderr snapshot and broadcast it."""
        if stream == "stdout":
            job.stdout = text
            job.stdout_truncated = truncated
        else:
            job.stderr = text
            job.stderr_truncated = truncated
        await self._broadcast_environment_job_event(MessageType.ENVIRONMENT_JOB_PROGRESS, job)

    async def _broadcast_environment_job_event(
        self,
        event_type: MessageType,
        job: EnvironmentJobSnapshot,
    ) -> None:
        """Broadcast a single environment-job state snapshot over notebook WS.

        Carries the started / progress frames; the payload is validated through
        ``EnvironmentJobModel`` so the wire shape is the documented contract.
        """
        await self._broadcast_environment_job_message(
            event_type,
            environment_job_event_payload(asdict(job)),
        )

    async def _broadcast_environment_job_message(
        self,
        event_type: MessageType,
        payload: dict[str, Any],
    ) -> None:
        """Send a structured notebook environment-job message to WS clients."""
        try:
            from strata.notebook.ws import broadcast_notebook_message, next_notebook_sequence
        except Exception:
            return

        await broadcast_notebook_message(
            self.id,
            {
                "type": event_type,
                "seq": next_notebook_sequence(self.id),
                "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "payload": payload,
            },
        )

    async def _broadcast_environment_staleness_updates(self, cell_ids: list[str]) -> None:
        """Broadcast current stale/idle statuses after an environment mutation."""
        if not cell_ids:
            return
        try:
            from strata.notebook.ws import broadcast_notebook_message, next_notebook_sequence
        except Exception:
            return

        for cell_id in cell_ids:
            cell = next(
                (candidate for candidate in self.notebook_state.cells if candidate.id == cell_id),
                None,
            )
            if cell is None:
                continue
            status = cell.status.value if isinstance(cell.status, CellStatus) else str(cell.status)
            causality = self.causality_map.get(cell.id)
            payload = cell_status_payload(
                cell.id,
                status,
                staleness_reasons=[
                    reason.value for reason in (cell.staleness.reasons if cell.staleness else [])
                ],
                causality=asdict(causality, dict_factory=skip_none)
                if causality is not None
                else None,
            )

            await broadcast_notebook_message(
                self.id,
                {
                    "type": MessageType.CELL_STATUS,
                    "seq": next_notebook_sequence(self.id),
                    "ts": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                    "payload": payload,
                },
            )


class SessionManager:
    """Manages multiple open notebooks by ID.

    Sessions are evicted after ``SESSION_TTL_SECONDS`` of inactivity
    or when ``MAX_SESSIONS`` is exceeded (oldest evicted first).
    """

    MAX_SESSIONS = 50
    SESSION_TTL_SECONDS = 4 * 3600  # 4 hours

    def __init__(self):
        """Initialize session manager."""
        self._sessions: dict[str, NotebookSession] = {}

    def _find_session_by_path(self, directory: Path) -> NotebookSession | None:
        """Return an existing live session for *directory*, if any."""
        target = Path(directory).resolve()
        for session in self._sessions.values():
            try:
                if session.path.resolve() == target:
                    return session
            except FileNotFoundError:
                continue
        return None

    def open_notebook(
        self,
        directory: Path,
        *,
        skip_initial_venv_sync: bool = False,
        defer_initial_venv_sync: bool = False,
        reuse_existing: bool = False,
        timing: NotebookTimingRecorder | None = None,
    ) -> NotebookSession:
        """Open a notebook directory.

        Args:
            directory: Path to notebook directory
            skip_initial_venv_sync: Reuse an already-created notebook venv and
                only refresh lightweight runtime metadata on first open.
            defer_initial_venv_sync: Mark the notebook environment as pending
                background initialization instead of synchronizing it during open.
            reuse_existing: Reuse an already-open in-memory session for the
                same path instead of constructing a new one.
            timing: Optional request timing recorder for internal phases.

        Returns:
            NotebookSession for the opened notebook
        """
        self._evict_stale()

        if reuse_existing:
            existing = self._find_session_by_path(Path(directory))
            if existing is not None:
                if existing._has_active_execution():
                    existing.touch()
                    return existing
                if existing.has_active_environment_mutation():
                    existing.mark_environment_pending()
                    existing.touch()
                    return existing
                if timing is None:
                    existing.reload()
                else:
                    with timing.phase("session_reload"):
                        existing.reload()
                try:
                    if timing is None:
                        existing.refresh_environment_runtime()
                    else:
                        with timing.phase("session_env_refresh"):
                            existing.refresh_environment_runtime()
                except Exception as e:
                    logger.warning("Failed to refresh existing notebook runtime: %s", e)
                # Re-check renv.lock on every reopen. The hash short-circuit
                # inside ``ensure_renv_synced`` keeps unchanged-lockfile
                # reopens free; without this call, a notebook whose
                # ``renv.lock`` changes while the session is still cached
                # in the manager would silently run against the old R
                # library on next open.
                try:
                    if timing is None:
                        existing.ensure_renv_synced()
                    else:
                        with timing.phase("session_renv_sync"):
                            existing.ensure_renv_synced()
                except Exception as e:
                    logger.warning("Failed to re-sync renv for existing session: %s", e)
                existing.touch()
                return existing

        if timing is None:
            notebook_state = parse_notebook(Path(directory))
        else:
            with timing.phase("session_parse"):
                notebook_state = parse_notebook(Path(directory))
        session = NotebookSession(notebook_state, Path(directory))

        # Ensure venv is ready. Freshly-created notebooks may already have a
        # synced .venv from writer.create_notebook(), so avoid immediately
        # paying for a second uv sync and just refresh runtime metadata.
        try:
            if defer_initial_venv_sync:
                session.mark_environment_pending()
            elif skip_initial_venv_sync:
                if timing is None:
                    session.refresh_environment_runtime()
                else:
                    with timing.phase("session_env_refresh"):
                        session.refresh_environment_runtime()
            else:
                if timing is None:
                    session.ensure_venv_synced()
                else:
                    with timing.phase("session_env_sync"):
                        session.ensure_venv_synced()
        except Exception as e:
            # Log warning but don't fail — notebook can still be opened,
            # it just won't be able to execute cells
            logger.warning("Failed to sync venv: %s", e)

        # Ensure the R environment matches its lockfile. No-op when
        # the notebook has no ``renv.lock``, so Python-only notebooks
        # don't pay the Rscript probe cost. Same "log + continue"
        # contract as the venv sync above: a failed R sync doesn't
        # block opening the notebook.
        try:
            if timing is None:
                session.ensure_renv_synced()
            else:
                with timing.phase("session_renv_sync"):
                    session.ensure_renv_synced()
        except Exception as e:
            logger.warning("Failed to sync renv: %s", e)

        # M6: Initialize and start warm process pool
        try:
            if session._should_start_warm_pool():
                if timing is None:
                    from strata.notebook.pool import WarmProcessPool

                    session.warm_pool = WarmProcessPool(
                        notebook_dir=Path(directory),
                        pool_size=2,
                        python_executable=session.venv_python or Path("python"),
                    )
                    # Start pool in background (don't block on notebook open)
                    import asyncio

                    try:
                        task = asyncio.get_running_loop().create_task(session.warm_pool.start())
                        session.warm_pool.track_background_task(task)
                    except RuntimeError:
                        pass  # No running loop; pool stays cold until first acquire
                else:
                    with timing.phase("session_warm_pool"):
                        from strata.notebook.pool import WarmProcessPool

                        session.warm_pool = WarmProcessPool(
                            notebook_dir=Path(directory),
                            pool_size=2,
                            python_executable=session.venv_python or Path("python"),
                        )
                        # Start pool in background (don't block on notebook open)
                        import asyncio

                        try:
                            task = asyncio.get_running_loop().create_task(session.warm_pool.start())
                            session.warm_pool.track_background_task(task)
                        except RuntimeError:
                            pass  # No running loop; pool stays cold until first acquire
        except Exception as e:
            logger.warning("Failed to initialize warm pool: %s", e)

        # R warm pool: only for notebooks that actually contain R cells.
        try:
            session.start_r_pool_background()
        except Exception as e:
            logger.warning("Failed to initialize R warm pool: %s", e)

        if timing is None:
            session.compute_staleness()
        else:
            with timing.phase("session_staleness"):
                session.compute_staleness()

        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> NotebookSession | None:
        """Get a session by ID, updating its last-accessed timestamp.

        Args:
            session_id: Session ID

        Returns:
            NotebookSession or None if not found
        """
        session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def _has_active_websocket(self, session_id: str) -> bool:
        """Return whether a notebook session currently has connected sockets."""
        try:
            from strata.notebook.ws import _notebook_connections
        except Exception:
            return False
        return bool(_notebook_connections.get(session_id))

    def _evict_stale(self) -> None:
        """Remove sessions not accessed within TTL and enforce max count."""
        now = _time.time()
        stale = [
            sid
            for sid, s in self._sessions.items()
            if (
                not self._has_active_websocket(sid)
                and now - s.last_accessed > self.SESSION_TTL_SECONDS
            )
        ]
        for sid in stale:
            logger.info("Evicting stale session %s", sid)
            self.close_session(sid)

        # Enforce max count — evict oldest if over limit
        while len(self._sessions) >= self.MAX_SESSIONS:
            evictable = [sid for sid in self._sessions if not self._has_active_websocket(sid)]
            if not evictable:
                logger.warning(
                    "Session limit exceeded (%d) but all sessions have active websockets",
                    len(self._sessions),
                )
                break
            oldest_id = min(evictable, key=lambda sid: self._sessions[sid].last_accessed)
            logger.info("Evicting oldest session %s (max %d reached)", oldest_id, self.MAX_SESSIONS)
            self.close_session(oldest_id)

    def close_session(self, session_id: str) -> None:
        """Close a session and release resources.

        Args:
            session_id: Session ID
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        # Drain warm pools if present
        for pool in (session.warm_pool, session.r_warm_pool):
            if pool is None:
                continue
            import asyncio

            drain = getattr(pool, "drain", None)
            shutdown_nowait = getattr(pool, "shutdown_nowait", None)
            try:
                if callable(drain):
                    asyncio.get_running_loop().create_task(drain())
            except RuntimeError:
                if callable(shutdown_nowait):
                    shutdown_nowait()
            else:
                if not callable(drain) and callable(shutdown_nowait):
                    shutdown_nowait()

    def list_sessions(self) -> list[str]:
        """List all open session IDs.

        Returns:
            List of session IDs
        """
        return list(self._sessions.keys())

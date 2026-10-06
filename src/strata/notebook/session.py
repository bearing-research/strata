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
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.notebook import console_relay
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
    AnnotationDiagnostic,
    CellOutput,
    CellStaleness,
    CellState,
    CellStatus,
    DatasetSpec,
    MountMode,
    NotebookState,
    StalenessReason,
    VariantGroupState,
    VariantMember,
)
from strata.notebook.mounts import resolve_cell_mounts
from strata.notebook.parser import parse_notebook
from strata.notebook.presence import SessionPresence
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
    MAX_EXECUTION_SAMPLES,
    EnvironmentRuntime,
    RRuntime,
    load_runtime_state,
    persist_cell_execution_sample,
    persist_environment_synced_lockfile_hash,
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
    _is_sensitive_env_key,
    _renv_sync,
    _uv_sync,
    drop_blanked_secrets,
    update_cell_display_outputs,
    update_environment_metadata,
)
from strata.notebook.ws_payloads import (
    cell_status_payload,
    environment_job_event_payload,
)

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactVersion
    from strata.notebook.artifact_integration import NotebookArtifactManager
    from strata.notebook.pool import WarmProcessPool
    from strata.notebook.secret_manager import SecretFetchResult
    from strata.notebook.ws_payloads import SessionClosedReason

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

    Prepends a fresh annotation if the source has none, so the new sibling still
    joins the group.
    """
    new_source, count = _VARIANT_LINE_RE.subn(rf"\g<1>{new_name}\g<2>", source, count=1)
    if count == 0:
        return f"# @variant {group} {new_name}\n{source}"
    return new_source


def stored_display(artifact: ArtifactVersion) -> tuple[dict[str, Any], int] | None:
    """A display artifact's own description and its run's display count.

    ``None`` for an artifact written before displays recorded themselves; callers
    fall back to the cell's current description.
    """
    if not artifact.transform_spec:
        return None
    params = json.loads(artifact.transform_spec).get("params", {})
    if "display" not in params or "display_count" not in params:
        return None
    return json.loads(params["display"]), int(params["display_count"])


def _value_outlives_provenance(cell: CellState) -> bool:
    """Whether a cell's value can change while its provenance hash does not.

    True for ``# @nocache`` cells and cells with a read-write mount; their
    downstream cache keys follow the bytes rather than the hash. Everything else
    stays keyed on provenance, since bytes can differ across machines for the same
    result and keying on them would split the team cache.
    """
    annotations = parse_annotations(cell.source)
    if annotations.nocache:
        return True
    mounts = resolve_cell_mounts([], cell.mounts, annotations.mounts)
    return any(mount.mode == MountMode.READ_WRITE for mount in mounts)


@dataclass(frozen=True)
class _OutsideWorld:
    """What one cell's staleness depends on beyond this process.

    Gathered before the staleness lock is taken: it reaches a catalog, a registry
    and ``@fetch`` URLs, and the broadcast path takes the lock on the event loop.
    """

    mount_fingerprints: list[str]
    has_rw_mount: bool
    table_fingerprints: list[str]
    fetch_fingerprints: list[str]
    dataset_fingerprints: list[str]


@dataclass
class ExecutionSample:
    """One execution timing sample for profiling and estimates."""

    duration_ms: float
    cache_hit: bool
    # Set only for a shared team store hit: who computed it and what their run cost.
    # The estimator prices a hit against the last local uncached run, which a
    # teammate's hit never had, so the saving rides on the sample.
    from_team: bool = False
    team_principal: str | None = None
    team_saved_ms: int = 0
    team_promotion: str | None = None


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

    ``reasons`` keeps recorded order; ``causality`` is the wire-format chain with
    None fields stripped, or ``None`` when the cell has none.
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
    """Holds state for one open notebook."""

    def __init__(self, notebook_state: NotebookState, path: Path, *, fetch_secrets: bool = True):
        """``fetch_secrets=False`` leaves the first fetch to :meth:`refresh_secrets_async`."""
        from strata.notebook.env_backend import EnvironmentBackend, get_backend

        self.id: str = str(uuid.uuid4())
        self.notebook_state = notebook_state
        self.path = Path(path)
        # Unset until a sync or ``refresh_environment_runtime`` records it; while None,
        # cells do not run. Seeding it from an existing .venv would let a sync that
        # raises on open leave cells running in the old environment with no notice.
        self.venv_python: Path | None = None
        self.dag: NotebookDag | None = None
        # None when the build succeeded; consumers must tell "no dependencies" from
        # "could not be computed".
        self.dag_error: str | None = None
        self.backend: EnvironmentBackend = get_backend(self.path)

        from strata.notebook.artifact_integration import NotebookArtifactManager

        self.artifact_manager = NotebookArtifactManager(
            notebook_id=notebook_state.id,
            artifact_dir=path / ".strata" / "artifacts",
        )

        # One staleness computation at a time: it mutates the cells it walks (status,
        # staleness, artifact uris), and an off-loop walk and an on-loop recompute
        # would overwrite each other's status, which the cascade planner reads.
        # Reentrant so the off-loop wrapper can hold it across its call. A threading
        # lock because a session outlives any single event loop, and an asyncio.Lock
        # awaited from another loop never wakes.
        self._staleness_lock = threading.RLock()

        self.warm_pool: WarmProcessPool | None = None
        self.r_warm_pool: WarmProcessPool | None = None

        # The manager swaps in its own clock, so tests can age a session.
        self.activity_clock: Callable[[], float] = _time.time
        self.last_accessed: float = self.activity_clock()
        # (principal, tenant) of whoever opened it; open reuses it only for them,
        # and MCP hides it from other tenants.
        self.opened_by: tuple[str, str | None] | None = None

        # Seeded from ``.strata/runtime.json`` so the cache-savings figure survives a restart.
        self.execution_history: dict[str, list[ExecutionSample]] = (
            self._load_persisted_execution_history()
        )

        self.causality_map: dict[str, CausalityChain] = {}

        self.presence = SessionPresence()

        # Keyed by (variable, reference), with when: staleness re-asks the registry at
        # most every ``datasets.STALE_CHECK_SECONDS``.
        self._dataset_checks: dict[tuple[str, str], tuple[float, str]] = {}

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

        # The last secret fetch and the ``[secret_manager]`` config it was made for.
        self._secret_fetch: tuple[dict[str, Any], SecretFetchResult | None] | None = None

        self._analyze_and_build_dag()
        self._run_annotation_validation()
        # Merge manager secrets into env before cells see it.
        if fetch_secrets:
            self.refresh_secrets()
        else:
            self._merge_secrets()

    def _merge_secrets(self) -> None:
        """Merge the last secret fetch into env, unless ``[secret_manager]`` changed since.

        Updates ``notebook_state.env`` and the ``env_sources`` / ``env_fetch_error`` /
        ``env_fetched_at`` fields in place; with no fetch every key is stamped
        ``manual``. Mirrors the env into each cell's resolved env, which the executor
        reads. A reload lands here rather than refetching: a fetch is a network call,
        so it happens on open and on refresh only.
        """
        from strata.notebook.secret_manager.session_integration import merge_secrets_into_state

        result = None
        if self._secret_fetch is not None:
            config, fetched = self._secret_fetch
            if config == self.notebook_state.secret_manager_config:
                result = fetched
        merge_secrets_into_state(self.notebook_state, result)
        # Preserve cell-level overrides, as update_notebook_env_endpoint does.
        for cell in self.notebook_state.cells:
            resolved = dict(self.notebook_state.env)
            resolved.update(cell.env_overrides or {})
            cell.env = resolved

    def refresh_secrets(self) -> None:
        """Re-fetch secrets and re-merge into env, blocking; servers use the async form."""
        from strata.notebook.secret_manager import fetch_configured_secrets

        config = dict(self.notebook_state.secret_manager_config)
        self._secret_fetch = (config, fetch_configured_secrets(self.notebook_state))
        self._merge_secrets()

    async def refresh_secrets_async(self) -> None:
        """Re-fetch secrets in a worker thread and re-merge into env (the Refresh button).

        Gives up after ``SECRET_FETCH_TIMEOUT_SECONDS`` and records that as the fetch
        error. Recomputes staleness when the merge changed a cell's env, since a
        referenced secret enters provenance.
        """
        from strata.notebook.secret_manager import SecretFetchResult, fetch_configured_secrets
        from strata.notebook.secret_manager.provider import SECRET_FETCH_TIMEOUT_SECONDS

        config = dict(self.notebook_state.secret_manager_config)
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(fetch_configured_secrets, self.notebook_state),
                SECRET_FETCH_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            result = SecretFetchResult.failure(
                str(config.get("provider") or ""),
                f"The secret manager did not answer within {SECRET_FETCH_TIMEOUT_SECONDS:g}s.",
            )
        envs_before = [cell.env for cell in self.notebook_state.cells]
        # Recorded against the config it used, so a reload that changed the config
        # meanwhile does not merge it.
        self._secret_fetch = (config, result)
        self._merge_secrets()
        if [cell.env for cell in self.notebook_state.cells] != envs_before:
            await self.compute_staleness_async()

    def _run_annotation_validation(
        self, previous: dict[str, list[AnnotationDiagnostic]] | None = None
    ) -> None:
        """Validate annotations across all cells (on open/reload only).

        A diagnostic is logged once: not again while the cell keeps carrying it, nor
        when ``previous`` (the cells a reload replaced) already held it.
        """
        for cell in self.notebook_state.cells:
            seen = (
                previous.get(cell.id, []) if previous is not None else cell.annotation_diagnostics
            )
            diagnostics = validate_cell_annotations(cell, self.notebook_state)
            cell.annotation_diagnostics = diagnostics
            for d in diagnostics:
                if d in seen:
                    continue
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
        """Record user activity (an edit, a run, a focus) for idle accounting."""
        self.last_accessed = self.activity_clock()

    def set_variant_active(self, group: str, variant_name: str) -> None:
        """Switch the active variant for ``group``.

        Persists to ``notebook.toml`` and reloads, so the DAG, ``variant_active`` flags
        and downstream staleness recompute against the new selection.
        """
        from strata.notebook.writer import set_variant_active as _set_variant_active

        _set_variant_active(self.path, group, variant_name)
        self.reload()

    def set_variant_mode(self, group: str, mode: str) -> None:
        """Switch a variant group between ``"switch"`` and ``"sweep"`` mode.

        Persists to ``notebook.toml`` and reloads, so the DAG and downstream staleness
        recompute. ``mode`` is validated by the caller; an unknown value persists but
        runs as ``"switch"``.
        """
        from strata.notebook.writer import set_variant_mode as _set_variant_mode

        _set_variant_mode(self.path, group, mode)
        self.reload()

    def remove_cell(self, cell_id: str) -> None:
        """Delete a cell, with variant-aware cleanup.

        Removing the active variant promotes the next member in source order so
        ``active`` does not dangle; removing a group's last member also drops its
        ``[[variant_group]]`` entry.
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

        # Decide the variant_group toml fixup while the group's member order is still known.
        promote_to: str | None = None
        drop_group = False
        if resolved is not None:
            remaining = [m for m in resolved.members if m.cell_id != cell_id]
            if not remaining:
                drop_group = True
            elif cell.id == resolved.active_cell_id:
                # Promote the first survivor so the toml pointer stays valid (else reload
                # emits variant_active_unknown).
                promote_to = remaining[0].name

        remove_cell_from_notebook(self.path, cell_id)

        if drop_group and group_id is not None:
            remove_variant_group_entry(self.path, group_id)
        elif promote_to is not None and group_id is not None:
            _set_variant_active(self.path, group_id, promote_to)

        self.reload()

    def add_variant(self, group: str, author: str | None = None) -> tuple[str, str]:
        """Add a sibling variant to ``group`` by cloning the active variant.

        The new cell goes after the group's last member, becomes active, and gets a
        ``# @variant`` line with an auto-generated name (``<active>_copy``,
        ``<active>_copy2``, ...). ``author`` is credited. Returns
        ``(new_variant_name, new_cell_id)``.

        Raises ``ValueError`` if ``group`` is not a resolved variant group.
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
            # Shouldn't happen, but bail cleanly.
            raise ValueError(f"Active variant cell for group {group!r} not found")

        taken = {m.name for m in resolved.members}
        new_name = _next_variant_name(resolved.active_name, taken)
        new_cell_id = uuid.uuid4().hex[:8]
        last_member_cell_id = resolved.members[-1].cell_id

        # The caller's authorship, not the cloned cell's: `created_by` records who
        # added this cell, and inheriting the origin's author would attribute one
        # principal's action to another.
        add_cell_to_notebook(
            self.path,
            new_cell_id,
            after_cell_id=last_member_cell_id,
            language=active_cell.language,
            author=author,
        )
        new_source = _rewrite_variant_annotation(active_cell.source, group, new_name)
        write_cell(self.path, new_cell_id, new_source, author=author)

        # set_variant_active reloads, so DAG / staleness / variant flags refresh in one pass.
        self.set_variant_active(group, new_name)
        return new_name, new_cell_id

    def reload(self, *, keep_typed_secrets: bool = True) -> None:
        """Reload notebook state from disk.

        Secret values typed this session are never written to disk, so they are carried
        over for keys the disk still leaves blank. ``keep_typed_secrets=False`` is for a
        caller that sets the whole env itself.
        """
        from strata.notebook.secret_manager.session_integration import MANUAL_SOURCE

        previous_cells = {cell.id: cell.model_copy(deep=True) for cell in self.notebook_state.cells}
        previous_runtime_identities = {
            cell.id: self._effective_worker_runtime_identity(cell)
            for cell in self.notebook_state.cells
        }
        previous_env = self.notebook_state.env
        previous_sources = self.notebook_state.env_sources
        self.notebook_state = parse_notebook(self.path)
        if keep_typed_secrets:
            disk_env = self.notebook_state.env
            # The writer drops an [env] block that holds only blanked secrets.
            block_dropped = not disk_env
            for key, value in previous_env.items():
                if not value or not _is_sensitive_env_key(key):
                    continue
                if previous_sources.get(key, MANUAL_SOURCE) != MANUAL_SOURCE:
                    continue  # provider-fetched; _merge_secrets restores it
                if disk_env.get(key) == "" or (block_dropped and key not in disk_env):
                    disk_env[key] = value
        self._analyze_and_build_dag()
        self._run_annotation_validation(
            {cell_id: cell.annotation_diagnostics for cell_id, cell in previous_cells.items()}
        )
        self._merge_secrets()
        # Restore ``last_provenance_hash`` *before* computing staleness, or every cell
        # falls back to IDLE and none can be marked STALE.
        self._restore_execution_history(previous_cells)
        self.compute_staleness()
        self._restore_ready_runtime_state(previous_cells, previous_runtime_identities)

    def _analyze_and_build_dag(self) -> None:
        """Analyze all cells, build the DAG, and update each cell's DAG fields."""
        from strata.notebook.languages import analyze_cell_by_language

        cell_analyses = []
        for cell in self.notebook_state.cells:
            analyzed = analyze_cell_by_language(cell, self)
            defines = list(analyzed.defines)
            references = list(analyzed.references)
            mutation_defines = list(analyzed.mutation_defines)

            # Loop cells read the carry from upstream on iter 0 even though Python scoping
            # sees it as local, so record it as a reference to wire the seeding upstream.
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
            builtin_references = list(analyzed.builtin_references)
            cell_analyses.append(
                CellAnalysisWithId(
                    id=cell.id,
                    defines=defines,
                    references=references,
                    builtin_references=builtin_references,
                    after=list(annotations.after),
                    variant_group=variant_group,
                    variant_name=variant_name,
                    per_variant=annotations.per_variant,
                    per_variant_group=annotations.per_variant_group,
                )
            )
            cell.defines = defines
            cell.references = references
            cell.mutation_defines = mutation_defines
            cell.builtin_references = builtin_references
            cell.variant_group = variant_group
            cell.variant_name = variant_name
            # Active until the DAG resolution proves otherwise.
            cell.variant_active = True

        try:
            self.dag_error = None
            self.dag = NotebookDag.from_cells(
                cell_analyses,
                variant_active_selections=self.notebook_state.variant_active_selections,
                variant_modes=self.notebook_state.variant_modes,
            )

            for cell in self.notebook_state.cells:
                cell.upstream_ids = self.dag.cell_upstream.get(cell.id, [])
                cell.downstream_ids = self.dag.cell_downstream.get(cell.id, [])
                cell.is_leaf = cell.id in self.dag.leaves
                cell.variant_active = cell.id not in self.dag.inactive_cells

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
            # Cycle or variant collision: log, don't crash. Keep the reason, since a None
            # DAG otherwise reads as a notebook with no dependencies and "is this variable
            # defined?" would answer a confident no.
            logger.warning("DAG build failed: %s", e)
            self.dag = None
            self.dag_error = str(e)

    def _restore_execution_history(self, previous_cells: dict[str, Any]) -> None:
        """Restore per-cell execution history that ``notebook.toml`` does not hold.

        After a reload, copies display outputs, artifact URIs and last-seen hashes from
        the pre-reload snapshot for cells with the same id and source. Status is left
        to ``compute_staleness()``, which may downgrade a READY cell.
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

    def _restore_alternate_scheme_outputs(self, cell: Any) -> None:
        """Point a cell at the artifacts its last successful run stored.

        For cell kinds keyed under their own scheme (SQL, prompt, widget) nothing else
        repopulates ``artifact_uris`` on open, and a downstream cell would otherwise
        compute a provenance hash that no longer matches its record.
        """
        store = self.get_artifact_manager().artifact_store
        notebook_id = self.notebook_state.id
        for name in cell.defines:
            artifact = store.get_latest_version(f"nb_{notebook_id}_cell_{cell.id}_var_{name}")
            if artifact is not None:
                uri = f"strata://artifact/{artifact.id}@v={artifact.version}"
                cell.artifact_uris[name] = uri
                cell.artifact_uri = uri

    def _restore_ready_runtime_state(
        self,
        previous_cells: dict[str, Any],
        previous_runtime_identities: dict[str, str | None],
    ) -> None:
        """Restore READY for cells whose whole runtime identity is unchanged.

        Runs after ``compute_staleness()``, for cells it could not classify READY (e.g.
        leaves without canonical artifacts).
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
        """Re-analyze a single cell and rebuild the DAG."""
        cell = self.notebook_state.get_cell(cell_id)
        if not cell:
            return

        from strata.notebook.languages import analyze_cell_by_language

        analyzed = analyze_cell_by_language(cell, self)
        cell.defines = list(analyzed.defines)
        cell.references = list(analyzed.references)
        cell.builtin_references = list(analyzed.builtin_references)

        # One cell changed, so downstream may be affected.
        self._analyze_and_build_dag()

    def get_artifact_manager(self) -> NotebookArtifactManager:
        return self.artifact_manager

    async def compute_staleness_async(
        self, executing: str | None = None
    ) -> dict[str, CellStaleness]:
        """``compute_staleness`` in a worker thread.

        Staleness reads the outside world (``@fetch``, ``@dataset``, ``@table``) with
        blocking calls, and runs on every debounced source flush; on the event loop one
        unreachable host would stall every socket and route. Calls are serialized
        because the work mutates the cells it walks.
        """
        return await asyncio.to_thread(self.compute_staleness, executing)

    def compute_staleness(self, executing: str | None = None) -> dict[str, CellStaleness]:
        """Compute staleness for all cells, updating ``cell.staleness`` and causality chains.

        Reads the outside world for ``@fetch``, ``@dataset`` and ``@table`` cells, so
        async callers want :meth:`compute_staleness_async`. ``executing`` is a cell
        running during this recompute: its status stays running and it is left out of
        the returned map.
        """
        # Outside reads (an ``@fetch`` gets 60s) happen before the lock: they mutate
        # nothing, and holding the lock across them would block the broadcast path,
        # which takes it on the event loop and would freeze every socket.
        prefetched = self._outside_world_fingerprints()
        with self._staleness_lock:
            return self._compute_staleness_locked(prefetched, executing)

    def _outside_world_fingerprints(self) -> dict[str, _OutsideWorld]:
        """Fingerprint what each cell reads from outside, before the lock.

        Cells with nothing external cost nothing. A cell with external inputs is
        fingerprinted even if the walk will skip it for a stale upstream; the fetch
        cache and dataset memo throttle their own checks.
        """
        gathered: dict[str, _OutsideWorld] = {}
        if self.dag is None:
            return gathered
        from strata.notebook.languages import get_language_executor

        for cell_id in self.dag.topological_order:
            cell = self.notebook_state.get_cell(cell_id)
            if cell is None or get_language_executor(cell.language).skips_execution_provenance:
                continue
            gathered[cell_id] = self._outside_world_for(cell)
        return gathered

    def _outside_world_for(self, cell: Any) -> _OutsideWorld:
        """One cell's outside-world fingerprints."""
        mount_fingerprints, has_rw_mount = self._collect_mount_fingerprints(cell)
        return _OutsideWorld(
            mount_fingerprints=mount_fingerprints,
            has_rw_mount=has_rw_mount,
            table_fingerprints=self._collect_table_fingerprints(cell),
            fetch_fingerprints=self._collect_fetch_fingerprints(cell),
            dataset_fingerprints=self._collect_dataset_fingerprints(cell),
        )

    def _compute_staleness_locked(
        self, prefetched: dict[str, _OutsideWorld], executing: str | None = None
    ) -> dict[str, CellStaleness]:
        staleness_map: dict[str, CellStaleness] = {}
        stale_cells: set[str] = set()  # for propagation
        if self.dag is None:
            for cell in self.notebook_state.cells:
                staleness_map[cell.id] = CellStaleness(status=CellStatus.IDLE)
            self._apply_staleness_map(staleness_map, executing)
            self.causality_map = {}
            return staleness_map

        for cell_id in self.dag.topological_order:
            cell = self.notebook_state.get_cell(cell_id)
            if cell is None:
                continue

            # Languages with ``skips_execution_provenance`` (markdown) are always READY.
            from strata.notebook.languages import get_language_executor

            language_executor = get_language_executor(cell.language)
            if language_executor.skips_execution_provenance:
                staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                continue

            # A stale upstream makes this cell out of date too, and it propagates. If it ran
            # before (``last_provenance_hash`` set) it shows STALE with an UPSTREAM reason;
            # if it never ran, IDLE: there is no result to invalidate.
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

            source_hash = compute_source_hash(cell.source)
            runtime_env = self._collect_runtime_env(cell)
            env_hash = compute_execution_env_hash(
                self.path,
                runtime_env,
                runtime_identity=self._effective_worker_runtime_identity(cell),
            )

            # Same per-variable artifact selection as execution.
            input_hashes = self._collect_input_hashes(cell_id)
            # Missing only if the DAG moved meanwhile, a race this walk loses anyway (the
            # next flush recomputes), so read our own rather than skip the cell's mounts.
            outside = prefetched.get(cell_id)
            if outside is None:
                outside = self._outside_world_for(cell)

            if outside.has_rw_mount:
                staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                stale_cells.add(cell_id)
                continue

            provenance_hash = compute_provenance_hash(
                input_hashes
                + outside.mount_fingerprints
                + outside.table_fingerprints
                + outside.fetch_fingerprints
                + outside.dataset_fingerprints,
                source_hash,
                env_hash,
            )

            # Per-variable hashes: sha256(f"{provenance_hash}:{var_name}").
            cached_outputs = self._resolve_cached_outputs(cell_id, provenance_hash)
            cached_display_outputs = self._resolve_cached_display_outputs(
                cell_id,
                provenance_hash,
                cell.display_outputs,
            )

            if cached_outputs is None:
                # Resolving a display artifact for a cell keyed under its own scheme only says
                # the generic triplet is unchanged; the connection and cache policy aren't in
                # that hash, so the branch below settles readiness. Restore what it showed
                # either way.
                keyed_elsewhere = (
                    language_executor.has_alternate_cache_scheme
                    or parse_annotations(cell.source).per_variant
                )
                if cached_display_outputs and not keyed_elsewhere:
                    cell.display_outputs = cached_display_outputs
                    cell.display_output = cached_display_outputs[-1]
                    staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                else:
                    if cached_display_outputs:
                        cell.display_outputs = cached_display_outputs
                        cell.display_output = cached_display_outputs[-1]
                    # Alternate-scheme cells (PROMPT, SQL, ``# @per_variant`` fan-out) store under
                    # hashes the generic lookup can't match; the generic hash is recorded via
                    # ``record_successful_execution_provenance`` so a match preserves status.
                    # They are preserved from IDLE too, since a cold open starts every cell IDLE.
                    # A leaf still needs READY: its structural comparison on reload (mounts,
                    # worker, env) catches what its hash does not.
                    allowed_status = (
                        (CellStatus.READY, CellStatus.IDLE)
                        if keyed_elsewhere
                        else (CellStatus.READY,)
                    )
                    # The generic hash isn't all these cells cache on (a SQL cell's connection and
                    # cache policy, a prompt's model), so the cell's own identity must match what it
                    # recorded; None means it can't be settled without a probe. Only needed when
                    # resurrecting from IDLE (a cold open); a READY cell ran in this session.
                    identity_required = keyed_elsewhere and cell.status == CellStatus.IDLE
                    identity = self.reopen_identity(cell) if identity_required else None
                    identity_holds = not identity_required or (
                        identity is not None and identity == (cell.last_reopen_identity or "")
                    )
                    can_preserve_uncached_ready = (
                        (cell.is_leaf or keyed_elsewhere)
                        and cell.status in allowed_status
                        and cell.last_provenance_hash == provenance_hash
                        and identity_holds
                    )
                    if can_preserve_uncached_ready:
                        staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                        # Outputs live under a per-language scheme, so artifact uris are still empty
                        # here, and downstream cells read them to build their provenance.
                        self._restore_alternate_scheme_outputs(cell)
                    elif self._upstream_moved_under(cell, source_hash, env_hash):
                        # Upstream artifacts were replaced by newer versions with this cell's source
                        # and env unchanged: stale, not idle (which would read as "never ran").
                        staleness_map[cell_id] = CellStaleness(
                            status=CellStatus.STALE, reasons=[StalenessReason.UPSTREAM]
                        )
                        stale_cells.add(cell_id)
                    else:
                        # No cached artifact: idle unless it provably matches the last
                        # successful uncached run.
                        staleness_map[cell_id] = CellStaleness(status=CellStatus.IDLE, reasons=[])
                        stale_cells.add(cell_id)
            else:
                staleness_map[cell_id] = CellStaleness(status=CellStatus.READY, reasons=[])
                for var_name, (artifact_id, version) in cached_outputs.items():
                    uri = f"strata://artifact/{artifact_id}@v={version}"
                    cell.artifact_uris[var_name] = uri
                    cell.artifact_uri = uri  # backward compat
                cell.display_outputs = cached_display_outputs or []
                cell.display_output = cached_display_outputs[-1] if cached_display_outputs else None

        self._apply_staleness_map(staleness_map, executing)

        self.causality_map = compute_causality_on_staleness(self)

        return staleness_map

    def _upstream_moved_under(self, cell: CellState, source_hash: str, env_hash: str) -> bool:
        """Whether *cell*'s last result is out of date only because an upstream got a newer version.

        Covers two cases the walk would otherwise mark idle ("never ran"): the
        consumer of a changed ``# @nocache`` producer, and the direct downstream of an
        upstream that was edited and re-run. Decided only from the last result's
        record: same upstream artifacts at newer versions, and unchanged source and env
        hashes. The last result is the variable artifact, or for a leaf its first
        display output.
        """
        uri = cell.artifact_uri or next(
            (output.artifact_uri for output in cell.display_outputs if output.artifact_uri),
            None,
        )
        if not uri:
            return False
        try:
            artifact_id, version = self._parse_artifact_uri(uri)
        except (IndexError, ValueError):
            return False
        artifact = self.artifact_manager.artifact_store.get_artifact(artifact_id, version)
        if artifact is None or not artifact.transform_spec or not artifact.input_versions:
            return False

        params = json.loads(artifact.transform_spec).get("params", {})
        if params.get("source_hash") != source_hash or params.get("env_hash") != env_hash:
            return False

        def _versions(refs: Iterable[str]) -> dict[str, int]:
            out: dict[str, int] = {}
            for ref in refs:
                if not ref.startswith("strata://artifact/"):
                    continue  # a fetch or a dataset, not an upstream cell
                try:
                    ref_id, ref_version = self._parse_artifact_uri(ref)
                except (IndexError, ValueError):
                    continue
                out[ref_id] = ref_version
            return out

        recorded = _versions(json.loads(artifact.input_versions))
        current = _versions(self._collect_input_refs(cell.id))
        return (
            bool(recorded)
            and recorded.keys() == current.keys()
            and any(current[ref_id] != recorded[ref_id] for ref_id in recorded)
        )

    def _apply_staleness_map(
        self, staleness_map: dict[str, CellStaleness], executing: str | None = None
    ) -> None:
        """Write computed staleness back onto in-memory cell state."""
        for cell in self.notebook_state.cells:
            if cell.id == executing:
                # Still running: a verdict now would tell every client it had stopped.
                # Removed from the broadcast map too.
                staleness_map.pop(cell.id, None)
                continue
            staleness = staleness_map.get(cell.id)
            if staleness is None:
                continue
            cell.staleness = staleness
            if self._failure_still_stands(cell):
                # A failed cell stored no artifact, so the walk may call it idle, stale or even
                # ready. Until it is edited or succeeds again (a cache hit counts), the
                # failure is the truth about it.
                cell.status = CellStatus.ERROR
                cell.cache_hit = False
                # The WebSocket broadcasts from the map, so set it there too, or clients see
                # `idle` for a failed cell.
                cell.staleness = CellStaleness(status=CellStatus.ERROR, reasons=staleness.reasons)
                staleness_map[cell.id] = cell.staleness
                continue
            cell.status = staleness.status
            if staleness.status != CellStatus.READY:
                cell.cache_hit = False

    @staticmethod
    def _failure_still_stands(cell: CellState) -> bool:
        """Whether ``cell``'s recorded error is still about the source it has."""
        return cell.current_error() is not None

    def mark_executed_ready(self, cell_id: str) -> None:
        """Mark a just-executed cell READY.

        Some cells (especially leaves) are not cacheable via the canonical artifact
        path, but should still show as run until the next staleness recompute.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return

        cell.staleness = CellStaleness(status=CellStatus.READY, reasons=[])
        cell.status = CellStatus.READY
        self.causality_map.pop(cell_id, None)

    def mark_cell_running(self, cell_id: str) -> None:
        """Mark a cell as currently executing.

        The single setter every execution path uses; writing ``cell.status`` directly
        races with other paths.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is not None:
            cell.status = CellStatus.RUNNING

    def mark_cell_error(self, cell_id: str) -> list[str]:
        """Mark a cell as errored and flip its READY downstream cells to STALE.

        Otherwise a downstream cell would keep showing green on a cached result built
        from the broken upstream's last success. Returns the flipped cell ids for the
        caller to broadcast.
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
        # Keep the error against its source, so an agent returning to a failed cell
        # doesn't have to re-run it (side effects and all) to see what went wrong.
        had_error = cell.error is not None
        if result.success:
            cell.error = None
            cell.error_source_hash = None
        else:
            # A Python traceback is the fuller form of the error message.
            detail = getattr(result, "traceback", None)
            cell.error = (detail or result.error or "").strip()
            cell.error_source_hash = compute_source_hash(cell.source)
        if cell.error is not None or had_error:
            # Runtime state, not notebook.toml. Written only when an error appears or
            # clears, so a green run doesn't rewrite the file.
            from strata.notebook.runtime_state import persist_cell_error

            persist_cell_error(
                self.path,
                cell_id,
                error=cell.error,
                source_hash=cell.error_source_hash,
            )
        # Write on a real execution or a cache hit that replays console. A hit with no
        # console must not write: ``update_cell_console_output`` would unlink the
        # original run's output. A failed run is a real execution.
        if not result.cache_hit or result.stdout or result.stderr:
            cell.console_stdout = result.stdout or ""
            cell.console_stderr = result.stderr or ""
            # Runtime writers never touch notebook.toml.
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

        Also writes ``.strata/runtime.json`` so ``compute_staleness`` classifies the
        cell correctly after a reopen.
        """
        from strata.notebook.runtime_state import persist_cell_provenance

        cell = self.notebook_state.get_cell(cell_id)
        if cell is None:
            return
        cell.last_provenance_hash = provenance_hash
        cell.last_source_hash = source_hash
        cell.last_env_hash = env_hash
        # So a reopen compares like with like, not on a hash that never covered the
        # connection read or the model asked.
        cell.last_reopen_identity = self.reopen_identity(cell)
        persist_cell_provenance(
            self.path,
            cell_id,
            last_provenance_hash=provenance_hash,
            last_source_hash=source_hash,
            last_env_hash=env_hash,
            last_reopen_identity=cell.last_reopen_identity,
        )

    def reopen_identity(self, cell: CellState) -> str | None:
        """What this cell's cache scheme rests on beyond the generic triplet.

        ``None`` when the language cannot settle it without going out to the world, or
        when asking raised: a reopen must not act on an identity nobody can reproduce.
        """
        from strata.notebook.languages import get_language_executor

        try:
            language_executor = get_language_executor(cell.language)
            return language_executor.reopen_identity(cell, self)
        except Exception:
            logger.debug("reopen identity unavailable for cell %s", cell.id, exc_info=True)
            return None

    def serialize_cell(self, cell: CellState) -> dict[str, Any]:
        """Serialize a cell with session-coupled overlays.

        Adds hydrated display outputs, causality chains and DAG shadow warnings to
        ``CellState.serialize()``, and masks secret env values.
        """
        from strata.notebook.secret_manager.session_integration import mask_env

        data = cell.serialize()
        sources = self.notebook_state.env_sources
        data["env"] = mask_env(cell.env, sources)
        data["env_overrides"] = mask_env(cell.env_overrides, sources)
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
        # A viewer joining mid-run gets what the running cell has printed so far;
        # clients cleared the last run's console when this run started.
        live = console_relay.live_console(self.id, cell.id)
        for stream in ("stdout", "stderr"):
            if stream in live:
                data[f"console_{stream}"] = live[stream]
            elif cell.status == CellStatus.RUNNING:
                data[f"console_{stream}"] = ""
        return data

    def persist_display_outputs(
        self, cell_id: str, display_outputs: list[dict[str, Any]] | None
    ) -> None:
        """Persist display metadata to ``.strata/runtime.json`` for restoration on reopen."""
        update_cell_display_outputs(self.path, cell_id, display_outputs)

    def persist_display_output(self, cell_id: str, display_output: dict[str, Any] | None) -> None:
        """Persist a single display output (``None`` clears)."""
        self.persist_display_outputs(cell_id, [display_output] if display_output else None)

    def _resolve_cached_display_outputs(
        self,
        cell_id: str,
        provenance_hash: str,
        current_outputs: list[CellOutput],
    ) -> list[CellOutput]:
        """Return the display outputs cached for ``provenance_hash``, if all are.

        Each display artifact records its own description and its run's display count,
        so the set comes back exactly as that run left it, not with the cell's current
        previews. Artifacts written before that are bounded and described by
        ``current_outputs``.
        """
        notebook_id = self.notebook_state.id
        store = self.artifact_manager.artifact_store

        def _matching(index: int) -> ArtifactVersion | None:
            artifact_id = f"nb_{notebook_id}_cell_{cell_id}_var___display__{index}"
            expected = hashlib.sha256(f"{provenance_hash}:__display__{index}".encode()).hexdigest()
            artifact = store.get_latest_version(artifact_id)
            if artifact is None or artifact.provenance_hash != expected:
                return None
            return artifact

        first = _matching(0)
        if first is None:
            return []

        described = stored_display(first)
        pairs: list[tuple[ArtifactVersion, CellOutput]] = []
        if described is not None:
            for index in range(described[1]):
                artifact = first if index == 0 else _matching(index)
                stored = stored_display(artifact) if artifact is not None else None
                if artifact is None or stored is None:
                    return []
                pairs.append((artifact, CellOutput(**stored[0])))
        else:
            if not current_outputs:
                return []
            for index, current_output in enumerate(current_outputs):
                artifact = first if index == 0 else _matching(index)
                if artifact is None:
                    return []
                pairs.append((artifact, current_output.model_copy(deep=True)))

        resolved: list[CellOutput] = []
        for artifact, output in pairs:
            output.artifact_uri = f"strata://artifact/{artifact.id}@v={artifact.version}"
            hydrated = self._hydrate_display_output(output)
            resolved.append(CellOutput(**hydrated) if hydrated is not None else output)
        return resolved

    def _resolve_cached_console(self, cell_id: str, provenance_hash: str) -> tuple[str, str] | None:
        """Return cached ``(stdout, stderr)`` for a leaf cell's identical provenance, else ``None``.

        Keyed by ``derive_subkey(provenance_hash, "__console__")``, so it replays only
        on an identical provenance.
        """
        notebook_id = self.notebook_state.id
        artifact_id = f"nb_{notebook_id}_cell_{cell_id}_var___console__"
        expected_hash = hashlib.sha256(f"{provenance_hash}:__console__".encode()).hexdigest()
        artifact = self.artifact_manager.artifact_store.get_latest_version(artifact_id)
        if artifact is None or artifact.provenance_hash != expected_hash:
            return None
        try:
            blob = self.artifact_manager.load_artifact_data(artifact_id, artifact.version)
            payload = json.loads(blob)
        except (ValueError, OSError, KeyError):
            return None
        return str(payload.get("stdout", "")), str(payload.get("stderr", ""))

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

    def read_display_blob(self, output: CellOutput) -> bytes:
        """The stored bytes behind one display output.

        Raises ``ValueError`` when the output was never stored as an artifact, or the
        artifact it names is gone.
        """
        artifact_uri = output.artifact_uri
        if not artifact_uri:
            raise ValueError("display output is not backed by an artifact")
        artifact_id, version = self._parse_artifact_uri(artifact_uri)
        return self.artifact_manager.load_artifact_data(artifact_id, version)

    @staticmethod
    def _parse_artifact_uri(artifact_uri: str) -> tuple[str, int]:
        """Parse a canonical artifact URI into ``(artifact_id, version)``.

        Splits on the *last* ``@v=``, since a fan-out id carries its own ``@``
        (``..._var_score@variant=triple``). Raises ``ValueError`` when there is no version.
        """
        artifact_id, sep, version = artifact_uri.split("/")[-1].rpartition("@v=")
        if not sep:
            raise ValueError(f"not a versioned artifact URI: {artifact_uri!r}")
        return artifact_id, int(version)

    def serialize_cells(self) -> list[dict[str, Any]]:
        """Serialize all cells with runtime-derived metadata."""
        return [self.serialize_cell(cell) for cell in self.notebook_state.cells]

    def capture_cell_state_snapshot(self) -> dict[str, CellStateSnapshot]:
        """Capture each cell's status/reasons/causality, by cell id, for diffing after a recompute.

        Callers recompute staleness and broadcast only the cells whose snapshot changed.
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
        from strata.notebook.secret_manager.session_integration import mask_env

        data = self.notebook_state.model_dump()
        data["env"] = mask_env(self.notebook_state.env, self.notebook_state.env_sources)
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
        """Best-effort read of environment metadata from ``.strata/runtime.json``."""
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

        The probe spawns ``Rscript`` (10s timeout), and this is read on every state
        sync, env refresh and dependency mutation.
        """
        if hasattr(self, "_system_r_version_cache"):
            return self._system_r_version_cache
        version = self._probe_r_version()
        self._system_r_version_cache = version
        return version

    def serialize_r_environment_state(self, *, include_packages: bool = False) -> dict[str, Any]:
        """Serialize the R-side runtime environment for the UI.

        ``has_lockfile`` comes from disk, so a notebook whose ``renv.lock`` never
        synced still shows why. Other fields reflect the last successful sync;
        ``sync_error`` is the latest attempt's error.

        ``include_packages`` spawns ``Rscript`` to list the renv library. Off by
        default because this runs on every state sync; the R env panel fetches the list
        from ``GET /v1/notebooks/{id}/r-packages`` instead.

        ``sync_state``:

        * ``absent``: no ``renv.lock`` (Python-only).
        * ``never``: lockfile exists, no sync ever succeeded, latest attempt did not fail.
        * ``ok``: last sync matched the current lockfile hash, no error.
        * ``outdated``: last good sync was against a different lockfile, no error.
        * ``failed``: the latest sync attempt failed (``sync_error`` is set).
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

        # Listing the library is a ~1-2s Rscript spawn, so only the R-packages route
        # opts in; with no lockfile there is nothing to enumerate.
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
            # Last-successful-sync state below.
            "lock_hash": runtime.lock_hash,
            # ``r_version`` is from the last good renv sync; ``system_r_version`` is the
            # Rscript on PATH now, so the UI always has some R version to show.
            "r_version": runtime.r_version,
            "system_r_version": self._cached_system_r_version,
            "last_synced_at": runtime.last_synced_at,
            "sync_state": sync_state,
            "sync_error": runtime.sync_error or None,
            # ``packages_status`` tells a failed probe from an empty library: ``"absent"``
            # (no lockfile or no probe), ``"ok"``, ``"rscript_missing"``,
            # ``"renv_not_active"`` or ``"failed"`` (with ``packages_error``).
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
        from strata.notebook.quiesce import execution_block

        held = execution_block(self.path)
        if held is not None:
            return held
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
        from strata.notebook.quiesce import execution_block

        held = execution_block(self.path)
        if held is not None:
            raise RuntimeError(held)
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

        A hit requires every consumed variable to have a canonical artifact in this
        notebook whose provenance matches the executor's per-variable hash.
        """
        consumed_vars = self.dag.consumed_variables.get(cell_id, set()) if self.dag else set()
        if not consumed_vars:
            # A leaf's only product is its console, stored under its own subkey; that is
            # what the executor replays as a hit, so it is what makes the cell ready.
            if self._resolve_cached_console(cell_id, provenance_hash) is not None:
                return {}
            return None

        first_var = sorted(consumed_vars)[0]
        cached = self.artifact_manager.find_cached(derive_subkey(provenance_hash, first_var))
        if cached is None:
            return None

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
        """Provenance hashes from upstream artifacts, with sweep refs grouped.

        The executor's provenance and the causality explanations both use this, so a
        sweep downstream's stored hash and its staleness recheck agree. A sweep-group
        reference collapses to one deterministic ``sweep:<var>:<name>=<hash>;...``
        string; otherwise the cell would be perpetually stale.
        """
        from strata.notebook.dag import SweepProducer

        cell = self.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return []

        dag = self.dag
        hashes: list[str] = []
        sweep_buckets: dict[str, list[tuple[str, str]]] = {}

        store = self.artifact_manager.artifact_store

        def _hash_from_uri(uri: str, *, by_content: bool) -> str | None:
            try:
                artifact_id, version = self._parse_artifact_uri(uri)
            except ValueError:
                return None
            artifact = store.get_artifact(artifact_id, version)
            if artifact is None:
                return None
            if not by_content:
                return artifact.provenance_hash
            # The producer's provenance hash is the same every run, so only the bytes say
            # the value changed. Recorded at finalize; backfilled once if absent.
            digest = artifact.content_sha256 or store.content_digest(artifact_id, version)
            if digest is None:
                return artifact.provenance_hash
            return derive_subkey(artifact.provenance_hash, f"content={digest}")

        # For a content-keyed producer only the variables this cell reads are inputs,
        # so a changing sibling it never reads doesn't miss the cache. Provenance-keyed
        # producers keep every variable: their hashes move together, and narrowing
        # would rekey every existing downstream cell.
        reads = set(cell.references) | set(cell.builtin_references)

        for upstream_id in cell.upstream_ids:
            upstream_cell = self.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue
            by_content = _value_outlives_provenance(upstream_cell)

            uri_items: list[tuple[str | None, str]] = list(upstream_cell.artifact_uris.items())
            if not uri_items and upstream_cell.artifact_uri:
                uri_items = [(None, upstream_cell.artifact_uri)]

            for var_name, uri in uri_items:
                if by_content and var_name is not None and var_name not in reads:
                    continue
                fanout = dag.variable_producer.get(var_name) if dag and var_name else None
                if (
                    var_name is not None
                    and isinstance(fanout, SweepProducer)
                    and fanout.fanout_cell == upstream_id
                ):
                    # A @per_variant cell keeps one URI per variable (last variant stored), but a
                    # consumer reads every instance, so key on all of them.
                    instances = []
                    for variant_name, _ in fanout.variants:
                        instance = store.get_latest_version(
                            self.artifact_manager.cell_artifact_id(
                                upstream_id, var_name, variant=variant_name
                            )
                        )
                        if instance is None:
                            continue  # not run yet: the loader drops it too
                        instance_hash = _hash_from_uri(
                            f"strata://artifact/{instance.id}@v={instance.version}",
                            by_content=by_content,
                        )
                        instances.append(f"{variant_name}={instance_hash}")
                    hashes.append(f"fanout:{var_name}:{';'.join(sorted(instances))}")
                    continue
                provenance_hash = _hash_from_uri(uri, by_content=by_content)
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

    def _fanout_group_of(self, cell_id: str) -> str | None:
        """The sweep group ``cell_id`` fans out over, if it is a fan-out cell."""
        from strata.notebook.dag import SweepProducer

        if self.dag is None:
            return None
        for producer in self.dag.variable_producer.values():
            if isinstance(producer, SweepProducer) and producer.fanout_cell == cell_id:
                return producer.group
        return None

    def _collect_input_refs(self, cell_id: str, *, variant: str | None = None) -> dict[str, str]:
        """Upstream artifact refs in the form the lineage walk resolves.

        Returns ``{strata://artifact/<id>@v=<n>: <id>@v=<n>}``, the shape
        ``services.artifact._input_version_to_artifact_ref`` and
        ``artifact_cli._walk_lineage`` follow; anything else is an unidentifiable leaf.

        Built from ``artifact_uris`` (the exact ``id@v=n`` the inputs were loaded
        from), not from input provenance hashes: a hash is not an identity. An
        identical cell in another notebook sharing the store hashes the same, so
        ``find_by_provenance`` could return that notebook's artifact. This also needs
        no store lookups.
        """
        cell = self.notebook_state.get_cell(cell_id)
        if cell is None or not cell.upstream_ids:
            return {}

        from strata.notebook.dag import SweepProducer

        # A fan-out instance over the same group binds only its own variant as a
        # scalar, so recording the whole set would name variants it never read.
        own_group = self._fanout_group_of(cell_id) if variant is not None else None

        refs: dict[str, str] = {}
        for upstream_id in cell.upstream_ids:
            upstream_cell = self.notebook_state.get_cell(upstream_id)
            if upstream_cell is None:
                continue

            uris: list[str] = []
            for var_name, uri in upstream_cell.artifact_uris.items():
                producer = self.dag.variable_producer.get(var_name) if self.dag else None
                if isinstance(producer, SweepProducer) and producer.fanout_cell == upstream_id:
                    # A fan-out keeps one URI per variable (last variant stored), while a collapse
                    # consumer read every instance; a chained instance read only its own variant.
                    wanted = (
                        [(variant, None)]
                        if own_group == producer.group and variant is not None
                        else producer.variants
                    )
                    for variant_name, _ in wanted:
                        instance = self.artifact_manager.artifact_store.get_latest_version(
                            self.artifact_manager.cell_artifact_id(
                                upstream_id, var_name, variant=variant_name
                            )
                        )
                        if instance is not None:
                            uris.append(f"strata://artifact/{instance.id}@v={instance.version}")
                    continue
                uris.append(uri)
            # Only with no per-variable URIs at all: a fan-out upstream whose instances are
            # all missing must not fall back to the cell-level URI (an unrelated variant).
            if not upstream_cell.artifact_uris and upstream_cell.artifact_uri:
                uris = [upstream_cell.artifact_uri]

            for uri in uris:
                ref = uri.split("/")[-1]
                # Variants and loop iterations each get their own edge; nothing here groups them.
                if "@v=" in ref:
                    refs[uri] = ref
        return refs

    def _collect_mount_fingerprints(self, cell: Any) -> tuple[list[str], bool]:
        """Return deterministic mount provenance components for a cell.

        Source annotations can override the notebook-default mounts at execution time,
        so this merges both layers exactly as the executor does.
        """
        annotations = parse_annotations(cell.source)
        merged_mounts = resolve_cell_mounts([], cell.mounts, annotations.mounts)

        # Same storage options the executor fingerprints with, or a mount reached through
        # a credential lists differently and the cell never matches its own artifacts.
        from strata.notebook.credentials import CredentialResolver
        from strata.notebook.mounts import MountResolver, mount_fingerprint_sync

        resolver = MountResolver(
            cache_dir=self.path / ".strata" / "mount_cache",
            credential_resolver=CredentialResolver.from_config(
                self._lake_config(), env=dict(self.notebook_state.env)
            ),
        )
        mount_fingerprints: list[str] = []
        has_rw_mount = False
        for mount in sorted(merged_mounts, key=lambda m: m.name):
            fingerprint = mount_fingerprint_sync(resolver, mount)
            if fingerprint is None:
                has_rw_mount = True
            else:
                mount_fingerprints.append(fingerprint)

        return mount_fingerprints, has_rw_mount

    def _collect_table_fingerprints(self, cell: Any) -> list[str]:
        """Return ``@table`` snapshot fingerprints for a cell's provenance.

        Must match the executor's ``_compute_cell_provenance``, or the stored artifacts
        are keyed under a hash this check never reproduces and the cell (and its
        downstream) reads idle forever. Never raises: an unreachable catalog shows the
        cell stale.
        """
        annotations = parse_annotations(cell.source)
        tables = list(annotations.tables)
        config = self._lake_config()
        if annotations.sql is not None:
            # Only a SQL cell: the SQL package needs the [sql] extra.
            from strata.notebook.sql.lake import lake_tables, with_notebook_catalogs

            tables += lake_tables(self.notebook_state, cell.source)
            config = with_notebook_catalogs(config, self.notebook_state)
        if not tables:
            return []
        from strata.notebook.tables import fingerprint_tables

        fingerprints, _ = fingerprint_tables(tables, config, dict(self.notebook_state.env))
        return fingerprints

    def _collect_fetch_fingerprints(self, cell: Any) -> list[str]:
        """``@fetch`` fingerprints for staleness, matching the executor's.

        Checked at most every ``STALE_CHECK_SECONDS``, since staleness runs on each
        source edit and the executor checks again before every run. Never raises.
        """
        annotations = parse_annotations(cell.source)
        if not annotations.fetches:
            return []
        from strata.notebook.fetch import FetchCache, guard_settings

        allowed_hosts, allow_local = guard_settings(self._lake_config())
        cache = FetchCache(self.path, allowed_hosts=allowed_hosts, allow_local=allow_local)
        return [
            cache.fingerprint(spec) for spec in sorted(annotations.fetches, key=lambda s: s.name)
        ]

    def _collect_dataset_fingerprints(self, cell: Any) -> list[str]:
        """``@dataset`` fingerprints for staleness, matching the executor's.

        The registry (possibly remote) is asked at most every ``STALE_CHECK_SECONDS``
        per declaration; the executor resolves before every run and records the answer
        here. Never raises: an unresolvable name fingerprints as stale.
        """
        annotations = parse_annotations(cell.source)
        if not annotations.datasets:
            return []
        from strata.notebook import datasets

        fingerprints: list[str] = []
        for spec in sorted(annotations.datasets, key=lambda s: s.name):
            checked = self._dataset_checks.get((spec.name, spec.reference))
            if (
                checked is not None
                and _time.monotonic() - checked[0] < datasets.STALE_CHECK_SECONDS
            ):
                fingerprints.append(checked[1])
                continue
            try:
                fingerprint = datasets.registry_for(self._lake_config()).resolve(spec).fingerprint
            except datasets.DatasetError:
                fingerprint = datasets.unresolved_fingerprint(spec)
            self.remember_dataset_fingerprint(spec, fingerprint)
            fingerprints.append(fingerprint)
        return fingerprints

    def remember_dataset_fingerprint(self, spec: DatasetSpec, fingerprint: str) -> None:
        """Record what *spec* resolved to now, for staleness to reuse."""
        self._dataset_checks[(spec.name, spec.reference)] = (_time.monotonic(), fingerprint)

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

        Every notebook env var reaches the process, but only keys the cell declares or
        references enter its provenance, so adding an unrelated notebook-level secret
        does not invalidate it.
        """
        annotations = parse_annotations(cell.source)
        resolved = drop_blanked_secrets(cell.env)
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

    def _load_persisted_execution_history(self) -> dict[str, list[ExecutionSample]]:
        """Read per-cell execution timings back from ``.strata/runtime.json``.

        Runs before the notebook is read; a missing or unreadable file means no history.
        """
        history: dict[str, list[ExecutionSample]] = {}
        for cell_id, entry in load_runtime_state(self.path).cells.items():
            samples = [
                ExecutionSample(
                    duration_ms=float(sample["duration_ms"]),
                    cache_hit=bool(sample["cache_hit"]),
                    from_team=bool(sample.get("from_team")),
                    team_principal=sample.get("team_principal"),
                    team_saved_ms=int(sample.get("team_saved_ms") or 0),
                    team_promotion=sample.get("team_promotion"),
                )
                for sample in entry.execution_samples
            ]
            if samples:
                history[cell_id] = samples
        return history

    def record_execution(
        self,
        cell_id: str,
        duration_ms: float,
        cache_hit: bool,
        *,
        from_team: bool = False,
        team_principal: str | None = None,
        team_saved_ms: int = 0,
        team_promotion: str | None = None,
    ) -> None:
        """Record a cell execution for profiling.

        Args:
            from_team: Whether the result came from the shared team store. Explicit
                because an anonymous store has no principal and a publisher may
                record no duration.
            team_principal: Who computed it, when the store recorded an author.
            team_saved_ms: What their run cost, hence what this hit saved; this user
                has no local run to price it against.
            team_promotion: The promotion that put the result in the team store, if any.
        """
        if cell_id not in self.execution_history:
            self.execution_history[cell_id] = []
        self.execution_history[cell_id].append(
            ExecutionSample(
                duration_ms=duration_ms,
                cache_hit=cache_hit,
                from_team=from_team,
                team_principal=team_principal,
                team_saved_ms=team_saved_ms,
                team_promotion=team_promotion,
            )
        )
        # Mirrored to disk; trim to the file's cap so memory and disk don't drift across a reopen.
        self.execution_history[cell_id] = self.execution_history[cell_id][-MAX_EXECUTION_SAMPLES:]
        persist_cell_execution_sample(
            self.path,
            cell_id,
            duration_ms=duration_ms,
            cache_hit=cache_hit,
            from_team=from_team,
            team_principal=team_principal,
            team_saved_ms=team_saved_ms,
            team_promotion=team_promotion,
        )

    def get_estimated_duration(self, cell_id: str) -> int:
        """Estimated execution duration in ms from history, or 0 without history."""
        history = self.execution_history.get(cell_id, [])
        for sample in reversed(history):
            if not sample.cache_hit:
                return int(sample.duration_ms)
        return 0

    def get_profiling_summary(self) -> dict:
        """Notebook-level profiling: total time, cache savings, artifact sizes, per-cell data."""
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

        # A local hit is priced against the cell's last uncached run. A team hit has
        # no such run, so its price rides on the sample from the publisher's run.
        cache_savings_ms = 0
        team_cache_savings_ms = 0
        team_cache_hits = 0
        team_contributors: set[str] = set()
        team_promotions: set[str] = set()
        for cell in self.notebook_state.cells:
            history = self.execution_history.get(cell.id, [])
            last_non_cached_duration: int | None = None
            for sample in history:
                if sample.cache_hit:
                    if sample.team_saved_ms:
                        cache_savings_ms += sample.team_saved_ms
                        team_cache_savings_ms += sample.team_saved_ms
                    elif last_non_cached_duration is not None:
                        cache_savings_ms += last_non_cached_duration
                    if sample.from_team:
                        team_cache_hits += 1
                        if sample.team_principal:
                            team_contributors.add(sample.team_principal)
                        if sample.team_promotion:
                            team_promotions.add(sample.team_promotion)
                else:
                    last_non_cached_duration = int(sample.duration_ms)

        return {
            "total_execution_ms": int(total_execution_ms),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_savings_ms": cache_savings_ms,
            # The share from someone else's machine: is the shared store earning its keep?
            "team_cache_savings_ms": team_cache_savings_ms,
            "team_cache_hits": team_cache_hits,
            "team_contributors": sorted(team_contributors),
            # Which promotions these hits drew on, for deciding whether promoting is worth it.
            "team_promotions": sorted(team_promotions),
            "total_artifact_bytes": total_artifact_bytes,
            "cell_profiles": cell_profiles,
        }

    def ensure_venv_synced(self) -> None:
        """Ensure the venv is set up by running ``uv sync`` (idempotent).

        On failure the session still opens, with ``venv_python`` falling back to
        ``python`` on PATH.
        """
        from strata.notebook.env_backend import UvBackend

        started = _time.perf_counter()
        python_version = read_requested_python_minor(self.path)
        if isinstance(self.backend, UvBackend):
            ok = _uv_sync(self.path, python_version=python_version)
        else:
            # Never sync a shared environment through the notebook's link: ``uv sync``
            # would (un)install inside an env other notebooks use. The backend syncs into
            # the key and moves this notebook's link.
            ok = self.backend.sync(python_version=python_version, timeout=60).success
        self._apply_uv_sync_result(
            ok,
            duration_ms=int((_time.perf_counter() - started) * 1000),
        )

    def environment_attestation_error(self) -> str | None:
        """Why this environment cannot be published from, or ``None`` if it can.

        A failed ``uv sync`` keeps the previous venv and leaves the state ``ready`` (so
        a transient failure does not lock the owner out), after which artifacts are
        stamped with an environment they were not built in. In a shared store that
        result becomes everyone's cache hit. Returns a reason rather than a bool since
        "never synced" and "sync failed and the lockfile moved on" need different
        fixes. Does not inspect the venv's contents, so a hand-installed package still
        slips past.
        """
        # Only a known system-python fallback disqualifies. ``unknown`` is the normal
        # state for a directly constructed session (CLI, MCP ops, scratchpad).
        if self.environment_interpreter_source == "path":
            return (
                "the notebook has no venv and is running system python, "
                "which uv.lock does not describe"
            )

        realized = self._read_persisted_environment_metadata().synced_lockfile_hash
        if not realized:
            return "no successful environment sync has been recorded for this notebook"
        if realized != compute_lockfile_hash(self.path):
            return "the installed environment does not match uv.lock"

        # `compute_lockfile_hash` folds in renv.lock, so a `uv sync` before a failed
        # `renv::restore()` can satisfy it. Check R against its own last-good hash.
        renv_lock = self.path / "renv.lock"
        if renv_lock.exists():
            from strata.notebook.dependencies import _renv_lockfile_hash

            state = load_runtime_state(self.path)
            if state.r.lock_hash != _renv_lockfile_hash(self.path):
                return "the installed R library does not match renv.lock"

        return None

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
                # Record what was installed, not what is declared: when a sync fails the venv
                # keeps its old contents while the lockfile moves on.
                persist_environment_synced_lockfile_hash(
                    self.path, compute_lockfile_hash(self.path)
                )
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

        No-op without ``renv.lock``. Skips ``Rscript`` when the lockfile hash matches
        ``r.lock_hash`` in ``.strata/runtime.json`` and the renv library is present.
        Otherwise runs ``_renv_sync``: success records hash, timestamp and R version and
        clears ``sync_error``; failure records the error but keeps the last-good fields.
        State lives in ``runtime.json``, so reopens never churn ``notebook.toml``.
        """
        if not (self.path / "renv.lock").exists():
            # Python-only (or pre-init R): clear stale R state so a removed lockfile leaves
            # no phantom hash or error.
            self._clear_r_runtime_if_present()
            return

        lock_hash = self._renv_restore_due()
        if lock_hash is None:
            return

        started = _time.perf_counter()
        ok = _renv_sync(self.path)
        duration_ms = int((_time.perf_counter() - started) * 1000)

        if not ok:
            # ``_renv_sync`` already logged the cause. Record the failure for the UI but
            # keep the last-good ``lock_hash`` / ``r_version`` / ``last_synced_at``.
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

    def _renv_restore_due(self) -> str | None:
        """The ``renv.lock`` hash when the R library needs a restore, else ``None``.

        ``None`` too without a readable ``renv.lock``.
        """
        lockfile = self.path / "renv.lock"
        try:
            lock_bytes = lockfile.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Could not read renv.lock to hash: %s", exc)
            return None
        lock_hash = hashlib.sha256(lock_bytes).hexdigest()

        previous = load_runtime_state(self.path).r
        if (
            previous.lock_hash == lock_hash
            and not previous.sync_error
            and self._renv_library_present()
        ):
            # Library matches the lockfile, last sync succeeded and the library dir still
            # exists: skip the ~1-2s Rscript spawn (this fires on every reopen).
            logger.debug(
                "renv sync skipped for %s: lockfile hash unchanged and library present (%s)",
                self.path,
                lock_hash[:12],
            )
            return None
        return lock_hash

    def _renv_library_present(self) -> bool:
        """Whether the project's renv library exists *and* is non-empty.

        An empty ``renv/library`` can survive a wiped or aborted restore while runtime
        metadata claims success. Stops at the first entry; checking package integrity
        would need ``renv::status()`` and defeat the fast path.
        """
        library = self.path / "renv" / "library"
        if not library.is_dir():
            return False
        try:
            next(iter(library.iterdir()))
        except StopIteration:
            return False
        except OSError:
            # Permission denied / I/O error: assume not usable.
            return False
        return True

    def _probe_r_version(self) -> str | None:
        """Best-effort ``Rscript`` version string, or ``None`` if unavailable."""
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

        Stamps the new hash, timestamp and R version and clears any prior ``sync_error``.
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
        """Record a failed ``renv::restore()`` attempt in ``sync_error``.

        Keeps the last-good ``lock_hash`` / ``r_version`` / ``last_synced_at`` so the
        UI can show when the env last worked.
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

        No-op when already empty, so a Python-only open does not churn ``runtime.json``.
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
        """Refresh runtime metadata from an existing notebook venv without ``uv sync``.

        ``uv add`` / ``uv remove`` already synced the venv, so this only re-probes. If
        the venv is missing it falls back to ``ensure_venv_synced()``.
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
        # Deliberately does NOT attest the environment: this also runs on every reopen,
        # where nothing was installed, so a failed sync could be laundered by reloading
        # the tab. The caller that knows an install happened attests instead.
        self.environment_last_synced_at = persisted.last_synced_at or int(_time.time() * 1000)
        self.environment_last_sync_duration_ms = int((_time.perf_counter() - started) * 1000)

    def _should_start_warm_pool(self) -> bool:
        """Return whether the notebook has a stable enough runtime for warm workers."""
        if _session_setting("notebook_warm_pool_size") == 0:
            return False
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
            pool_size=_session_setting("notebook_warm_pool_size"),
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

        Only for notebooks with R cells on machines with Rscript. Safe to call
        repeatedly.
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
            pool_size=_session_setting("notebook_warm_pool_size"),
            worker_command=[rscript, str(pool_worker), str(self.path)],
            # R startup + renv activation can take far longer than Python's warm-up.
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
        """React to a lockfile update after ``uv add`` / ``uv remove``.

        Refreshes runtime metadata, invalidates the warm pool, and recomputes the
        lockfile hash for provenance.
        """
        # 1. Dependency mutation already synced .venv; reuse it instead of a second sync.
        await asyncio.to_thread(self.refresh_environment_runtime)
        # ``uv add`` / ``uv remove`` installed into the venv, so this realized the
        # lockfile; without attesting, every dependency change would disable publishing
        # until the next full sync.
        await asyncio.to_thread(
            persist_environment_synced_lockfile_hash,
            self.path,
            compute_lockfile_hash(self.path),
        )
        await self._invalidate_warm_pool_for_environment_change()

        # 2. New lockfile hash invalidates caches on next exec.
        new_hash = compute_lockfile_hash(self.path)
        logger.info("Lockfile hash updated to %.12s after dependency change", new_hash)

        # 3. Persist environment metadata
        try:
            await asyncio.to_thread(update_environment_metadata, self.path)
        except Exception:
            logger.exception("Failed to update environment metadata")

    async def mutate_dependency(self, package: str, *, action: str) -> DependencyMutationOutcome:
        """Apply a dependency mutation without blocking the event loop."""
        from strata.notebook.dependencies import add_dependency, remove_dependency

        # The two functions have different keyword-only params, so a shared `op` is a
        # union callable ty won't pass to asyncio.to_thread. Dispatch per call site.
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
        # R actions reuse the env-job machinery; ``_run_environment_job`` dispatches on ``action``.
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

        # Catch obvious user errors at submission rather than mid-job.
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
                        # So successful r_init / r_add jobs flip the R panel to synced
                        # without a reopen.
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

        ``renv_init`` / ``renv_add`` stream, so ``environment_job_progress`` frames go
        out live during a long compile. A changed ``renv.lock`` changes the shared
        lockfile hash, so Python cells go stale along with R cells (over-stale).
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

        # The library now matches the new ``renv.lock``: record it as the last good
        # sync so ``sync_state`` reads ``ok`` and the next open doesn't re-restore.
        # The R version is the cached system probe, which just populated the library.
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
        """Change ``requires-python`` and rebuild the venv on the new minor.

        If ``uv sync`` fails, restores the previous ``requires-python`` and re-syncs.
        If that also fails the notebook is left with the old pyproject and no venv,
        and the error goes to the job's operation log.
        """
        from strata.notebook.writer import update_requires_python

        old_minor = read_requested_python_minor(self.path)
        await asyncio.to_thread(update_requires_python, self.path, new_minor)

        # Otherwise uv sync reports an "interpreter mismatch" and refuses.
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
        # No-op without ``renv.lock``; R restores whatever uv did, as an open did inline.
        try:
            await asyncio.to_thread(self.ensure_renv_synced)
        except Exception as exc:
            logger.warning("Failed to sync renv: %s", exc)
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
        """Broadcast one environment-job state snapshot (started / progress) over the notebook WS.

        The payload is validated through ``EnvironmentJobModel``, the documented wire shape.
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


def _prune_artifacts_in_background(session: NotebookSession) -> threading.Thread | None:
    """Drop the notebook's older cell values in a background thread.

    Only under a server, whose config sets how many to keep; a CLI run or test
    leaves the store alone. Returns the thread, or ``None`` if none started.
    """
    from strata.notebook.harness_user import running_server_config

    # Read as harness_user reads it: a config without these fields prunes nothing.
    config = running_server_config()
    keep = getattr(config, "notebook_keep_superseded_versions", 0)
    if not keep:
        return None
    min_idle_seconds = getattr(config, "artifact_gc_min_idle_seconds", 0.0)
    manager = session.artifact_manager

    def prune() -> None:
        from strata.notebook.quiesce import NotebookQuiesced

        try:
            result = manager.prune(keep, min_idle_seconds)
        except NotebookQuiesced as exc:
            logger.info("Not pruning %s's artifacts: %s", session.path, exc)
            return
        except Exception:
            logger.exception("Pruning %s's artifacts failed", session.path)
            return
        if result["deleted_count"]:
            logger.info(
                "Pruned %d earlier cell value(s), %d bytes, from %s",
                result["deleted_count"],
                result["deleted_bytes"],
                session.path,
            )

    thread = threading.Thread(target=prune, name="notebook-artifact-prune", daemon=True)
    thread.start()
    return thread


def _session_setting(name: str) -> Any:
    """A ``notebook_*`` session setting from the server's config, or its default."""
    from strata.config import StrataConfig

    default = StrataConfig.model_fields[name].default
    try:
        from strata.server import get_state

        config = get_state().config
    except RuntimeError:
        return default
    return getattr(config, name, default)


def available_memory_mb() -> int | None:
    """The host's ``MemAvailable`` in MiB, or ``None`` where ``/proc/meminfo`` is absent."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except FileNotFoundError:
        return None
    return None


class SessionManager:
    """Manages open notebook sessions by ID.

    A session nobody has edited, run or focused for ``notebook_session_ttl_seconds``
    is closed, as are the least recently used beyond ``notebook_max_sessions``, and,
    with ``notebook_session_min_available_mb`` set, idle ones while memory is short.
    A session with a running cell, a soft lock or a quiesce hold is never closed.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = _time.time,
        available_memory_mb: Callable[[], int | None] = available_memory_mb,
    ):
        self._sessions: dict[str, NotebookSession] = {}
        self._clock = clock
        self._available_memory_mb = available_memory_mb
        self._memory_unreadable_logged = False

    def _find_session_by_path(
        self, directory: Path, opened_by: tuple[str, str | None] | None
    ) -> NotebookSession | None:
        """Return a live session for *directory* that *opened_by* opened, if any."""
        target = Path(directory).resolve()
        for session in self._sessions.values():
            if session.opened_by != opened_by:
                continue
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
        opened_by: tuple[str, str | None] | None = None,
        timing: NotebookTimingRecorder | None = None,
    ) -> NotebookSession:
        """Open a notebook directory and return its session.

        Args:
            skip_initial_venv_sync: Reuse an existing venv and only refresh
                lightweight runtime metadata on first open.
            defer_initial_venv_sync: Sync nothing during open: when the environment
                needs a ``uv sync`` or ``renv`` restore, mark it pending for the caller
                to run as an environment job.
            reuse_existing: Return an already-open session for the same path that
                the same ``opened_by`` opened.
            opened_by: ``(principal, tenant)`` of the caller, recorded on a new session.
            timing: Request timing recorder for internal phases.
        """
        self._evict_stale(making_room=True)

        if reuse_existing:
            existing = self._find_session_by_path(Path(directory), opened_by)
            if existing is not None:
                if existing._has_active_execution():
                    existing.touch()
                    return existing
                if existing.has_active_environment_mutation():
                    existing.mark_environment_pending()
                    existing.touch()
                    return existing
                # A reused session may have been open for days; prune as a fresh open does.
                _prune_artifacts_in_background(existing)
                if timing is None:
                    existing.reload()
                else:
                    with timing.phase("session_reload"):
                        existing.reload()
                # No interpreter means the sync raised on open. Refreshing would make cells
                # runnable on whatever .venv holds; syncing again keeps the failure visible.
                if existing.venv_python is None:
                    phase, prepare = "session_env_sync", existing.ensure_venv_synced
                else:
                    phase, prepare = "session_env_refresh", existing.refresh_environment_runtime
                if defer_initial_venv_sync and (
                    phase == "session_env_sync"
                    or not (existing.path / ".venv" / "bin" / "python").exists()
                    or existing._renv_restore_due() is not None
                ):
                    existing.mark_environment_pending()
                    existing.touch()
                    return existing
                try:
                    if timing is None:
                        prepare()
                    else:
                        with timing.phase(phase):
                            prepare()
                except Exception as e:
                    logger.warning("Failed to refresh existing notebook runtime: %s", e)
                # Re-check renv.lock on every reopen (cheap when unchanged), or a lockfile
                # changed while the session was cached runs against the old R library.
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
        # The caller fetches secrets with ``refresh_secrets_async``, off the event loop.
        session = NotebookSession(notebook_state, Path(directory), fetch_secrets=False)
        _prune_artifacts_in_background(session)

        # A fresh notebook may already have a synced .venv from create_notebook(), so
        # refresh runtime metadata instead of a second uv sync.
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
            # The notebook still opens; it just can't execute cells.
            logger.warning("Failed to sync venv: %s", e)

        # No-op without ``renv.lock``. A failed R sync doesn't block opening. Deferred,
        # the environment job restores it.
        if not defer_initial_venv_sync:
            try:
                if timing is None:
                    session.ensure_renv_synced()
                else:
                    with timing.phase("session_renv_sync"):
                        session.ensure_renv_synced()
            except Exception as e:
                logger.warning("Failed to sync renv: %s", e)

        try:
            if session._should_start_warm_pool():
                if timing is None:
                    from strata.notebook.pool import WarmProcessPool

                    session.warm_pool = WarmProcessPool(
                        notebook_dir=Path(directory),
                        pool_size=_session_setting("notebook_warm_pool_size"),
                        python_executable=session.venv_python or Path("python"),
                    )
                    # Don't block notebook open.
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
                            pool_size=_session_setting("notebook_warm_pool_size"),
                            python_executable=session.venv_python or Path("python"),
                        )
                        # Don't block notebook open.
                        import asyncio

                        try:
                            task = asyncio.get_running_loop().create_task(session.warm_pool.start())
                            session.warm_pool.track_background_task(task)
                        except RuntimeError:
                            pass  # No running loop; pool stays cold until first acquire
        except Exception as e:
            logger.warning("Failed to initialize warm pool: %s", e)

        # Only for notebooks that contain R cells.
        try:
            session.start_r_pool_background()
        except Exception as e:
            logger.warning("Failed to initialize R warm pool: %s", e)

        if timing is None:
            session.compute_staleness()
        else:
            with timing.phase("session_staleness"):
                session.compute_staleness()

        session.opened_by = opened_by
        session.activity_clock = self._clock
        session.touch()
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> NotebookSession | None:
        """Get a session by ID, or None. A lookup is not activity; callers touch."""
        return self._sessions.get(session_id)

    def _closable(self, session: NotebookSession) -> bool:
        """Whether closing *session* now loses nothing but its warm processes."""
        from strata.notebook.presence import lock_window_seconds
        from strata.notebook.quiesce import execution_block

        if session._has_active_execution() or session.has_active_environment_mutation():
            return False
        if session.presence.holds_lock(lock_window_seconds()):
            return False
        return execution_block(session.path) is None

    def _least_recently_used(self) -> str | None:
        """The closable session with the oldest activity, if any."""
        closable = [s for s in self._sessions.values() if self._closable(s)]
        if not closable:
            return None
        return min(closable, key=lambda s: s.last_accessed).id

    def _evict_stale(self, *, making_room: bool = False) -> None:
        """Close idle sessions, then the least recently used beyond the maximum.

        With *making_room*, one under the maximum, for the session about to open.
        """
        now = self._clock()
        ttl = _session_setting("notebook_session_ttl_seconds")
        for sid, session in list(self._sessions.items()):
            if session._has_active_execution():
                # Idleness starts when the run ends, not when it was asked for.
                session.touch()
            elif now - session.last_accessed > ttl and self._closable(session):
                logger.info("Closing session %s: idle for %ds", sid, now - session.last_accessed)
                self.close_session(sid, reason="idle")

        limit = _session_setting("notebook_max_sessions") - (1 if making_room else 0)
        while len(self._sessions) > limit:
            oldest_id = self._least_recently_used()
            if oldest_id is None:
                logger.warning(
                    "Session limit exceeded (%d) but every session is busy", len(self._sessions)
                )
                break
            logger.info("Closing session %s: over the limit of %d", oldest_id, limit)
            self.close_session(oldest_id, reason="session_limit")

    async def relieve_memory_pressure(self) -> None:
        """Close the least recently used idle sessions while memory is below the floor."""
        floor = _session_setting("notebook_session_min_available_mb")
        if floor is None:
            return
        while True:
            available = await asyncio.to_thread(self._available_memory_mb)
            if available is None:
                if not self._memory_unreadable_logged:
                    self._memory_unreadable_logged = True
                    logger.warning(
                        "notebook_session_min_available_mb is set, but this host has no "
                        "/proc/meminfo; sessions are not closed for memory"
                    )
                return
            if available >= floor:
                return
            victim = self._least_recently_used()
            if victim is None:
                logger.warning(
                    "Available memory %d MB is below %d MB and no session is idle",
                    available,
                    floor,
                )
                return
            logger.info("Closing session %s: available memory %d MB", victim, available)
            # Wait for its processes to exit, so the next reading sees the memory back.
            await asyncio.gather(
                *self.close_session(victim, reason="memory"), return_exceptions=True
            )

    async def sweep(self) -> None:
        """The periodic pass: idle, over-limit and memory-pressure closes."""
        self._evict_stale()
        await self.relieve_memory_pressure()

    def close_session(
        self, session_id: str, *, reason: SessionClosedReason = "closed"
    ) -> list[asyncio.Task]:
        """Close a session, tell its clients why, and release its warm processes.

        Returns the tasks still finishing that (the notice, the pool drains).
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            return []
        # Local import: ws imports this.
        from strata.notebook.ws import end_session_clients, forget_notebook_execution_state

        tasks: list[asyncio.Task] = []
        # Before the forget below: the notice takes the session's next sequence.
        notice = end_session_clients(session_id, reason)
        if notice is not None:
            tasks.append(notice)
        # The outbound sequence counter outlives a disconnect, and this is every way a
        # session ends (delete and close routes, the sweep).
        forget_notebook_execution_state(session_id)
        for pool in (session.warm_pool, session.r_warm_pool):
            if pool is None:
                continue
            drain = getattr(pool, "drain", None)
            shutdown_nowait = getattr(pool, "shutdown_nowait", None)
            try:
                if callable(drain):
                    tasks.append(asyncio.get_running_loop().create_task(drain()))
            except RuntimeError:
                if callable(shutdown_nowait):
                    shutdown_nowait()
            else:
                if not callable(drain) and callable(shutdown_nowait):
                    shutdown_nowait()
        return tasks

    def list_sessions(self) -> list[str]:
        """List all open session IDs."""
        return list(self._sessions.keys())

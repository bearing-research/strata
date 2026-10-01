"""Persistent per-notebook runtime state in ``.strata/runtime.json`` (gitignored).

Anything that changes on every run or sync (display outputs, per-cell
provenance, the last ``uv sync``) lives here so ``notebook.toml`` stays stable
under version control.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from strata.notebook.quiesce import refuses_while_held

SCHEMA_VERSION = 1
_RUNTIME_FILENAME = "runtime.json"

# Profiling needs the sequence (a cache hit credits the last uncached duration
# before it), but this file is rewritten on every execution, so keep it short.
MAX_EXECUTION_SAMPLES = 50


@dataclass
class CellRuntime:
    """Per-cell runtime state: execution provenance and display outputs."""

    last_provenance_hash: str | None = None
    last_source_hash: str | None = None
    last_env_hash: str | None = None
    # A language's own cache identity (a SQL cell's connection, a prompt
    # cell's model). The generic triplet above misses it, so a reopen comparing
    # only that would wrongly call the cell ready.
    last_reopen_identity: str | None = None
    # Last failure and the source hash it applies to, so a later process
    # (offline CLI, reopen) can read it without re-running.
    last_error: str | None = None
    last_error_source_hash: str | None = None
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    display: dict[str, Any] | None = None
    test_result: dict[str, Any] | None = None
    # Widget control values by variable name: runtime state, so a slider drag
    # does not churn ``notebook.toml``.
    widget_values: dict[str, Any] = field(default_factory=dict)
    # Recent ``{duration_ms, cache_hit}`` timings, oldest first, capped at
    # ``MAX_EXECUTION_SAMPLES``; persisted so cache savings survive a restart.
    execution_samples: list[dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Whether this entry carries no state; empty entries are stripped on save."""
        return not (
            self.last_provenance_hash
            or self.last_source_hash
            or self.last_env_hash
            or self.last_reopen_identity
            or self.last_error
            or self.display_outputs
            or self.display
            or self.test_result
            or self.widget_values
            or self.execution_samples
        )


@dataclass
class EnvironmentRuntime:
    """Snapshot of the notebook's runtime environment after a ``uv sync``.

    All fields default to empty, so missing keys read back as a well-formed value.
    """

    requested_python_version: str = ""
    runtime_python_version: str = ""
    lockfile_hash: str = ""
    python_version: str = ""
    package_count: int = 0
    declared_package_count: int = 0
    resolved_package_count: int = 0
    has_lockfile: bool = False
    last_synced_at: int = 0
    # The lockfile hash a ``uv sync`` actually realized. It differs from
    # ``lockfile_hash`` (what was on disk) when a sync failed and the old venv
    # was kept, so a cell would run in one env while provenance claims another.
    # Empty until a sync succeeds.
    synced_lockfile_hash: str = ""


@dataclass
class RRuntime:
    """Snapshot of the notebook's R runtime after ``renv::restore()``.

    ``lock_hash`` / ``r_version`` / ``last_synced_at`` describe the last
    successful sync and survive a failed re-sync. ``sync_error`` is the latest
    attempt's error, cleared on success. ``has_lockfile`` is not stored: it is
    derived from disk at serialization time.
    """

    lock_hash: str = ""
    r_version: str = ""
    last_synced_at: int = 0
    sync_error: str = ""


@dataclass
class RuntimeState:
    """Root of ``.strata/runtime.json``: keyed cells plus environment snapshots."""

    schema_version: int = SCHEMA_VERSION
    cells: dict[str, CellRuntime] = field(default_factory=dict)
    environment: EnvironmentRuntime = field(default_factory=EnvironmentRuntime)
    r: RRuntime = field(default_factory=RRuntime)

    def get_or_create_cell(self, cell_id: str) -> CellRuntime:
        """Return the per-cell entry, creating it on demand."""
        if cell_id not in self.cells:
            self.cells[cell_id] = CellRuntime()
        return self.cells[cell_id]

    def prune_cell(self, cell_id: str) -> None:
        """Remove a per-cell entry (when the cell is deleted)."""
        self.cells.pop(cell_id, None)


def runtime_state_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / _RUNTIME_FILENAME


def load_runtime_state(notebook_dir: Path) -> RuntimeState:
    """Return the runtime-state document, or a fresh empty one.

    A missing or unparseable file yields an empty ``RuntimeState``: runtime data
    is not authoritative and must not stop a notebook opening. A schema mismatch
    raises from the dataclass constructor.
    """
    path = runtime_state_path(notebook_dir)
    if not path.exists():
        return RuntimeState()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return RuntimeState()
    return RuntimeState(
        schema_version=data.get("schema_version", SCHEMA_VERSION),
        cells={cid: CellRuntime(**entry) for cid, entry in data.get("cells", {}).items()},
        environment=EnvironmentRuntime(**data.get("environment", {})),
        r=RRuntime(**data.get("r", {})),
    )


@refuses_while_held
def save_runtime_state(notebook_dir: Path, state: RuntimeState) -> None:
    """Atomically persist the runtime-state document."""
    path = runtime_state_path(notebook_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.cells = {cid: entry for cid, entry in state.cells.items() if not entry.is_empty()}
    state.schema_version = SCHEMA_VERSION

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2, sort_keys=True)
    os.replace(tmp_name, path)


def persist_cell_provenance(
    notebook_dir: Path,
    cell_id: str,
    *,
    last_provenance_hash: str | None,
    last_source_hash: str | None,
    last_env_hash: str | None,
    last_reopen_identity: str | None = None,
) -> None:
    """Persist the last successful execution provenance for a cell.

    Lets ``compute_staleness`` tell ``STALE`` from ``IDLE`` after the cell's
    canonical artifact has been evicted.
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.last_provenance_hash = last_provenance_hash or None
    entry.last_source_hash = last_source_hash or None
    entry.last_env_hash = last_env_hash or None
    entry.last_reopen_identity = last_reopen_identity
    save_runtime_state(notebook_dir, state)


def persist_cell_error(
    notebook_dir: Path,
    cell_id: str,
    *,
    error: str | None,
    source_hash: str | None,
) -> None:
    """Record (or clear) what a cell's last run failed with.

    Stored with the source hash it happened at, so the error is dropped once the
    source changes.
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.last_error = error or None
    entry.last_error_source_hash = source_hash if error else None
    save_runtime_state(notebook_dir, state)


def persist_environment_synced_lockfile_hash(notebook_dir: Path, lockfile_hash: str) -> None:
    """Record the lockfile a successful ``uv sync`` actually installed.

    Kept apart from the rest of the environment snapshot, which records what is
    declared on disk and is written unconditionally: when the two disagree, the
    venv does not match the lockfile provenance is computed from.
    """
    state = load_runtime_state(notebook_dir)
    if state.environment.synced_lockfile_hash == lockfile_hash:
        # The common reopen case. A rewrite costs a full serialize and widens
        # the window for clobbering a concurrent execution's write.
        return
    state.environment.synced_lockfile_hash = lockfile_hash
    save_runtime_state(notebook_dir, state)


def persist_cell_execution_sample(
    notebook_dir: Path,
    cell_id: str,
    *,
    duration_ms: float,
    cache_hit: bool,
    from_team: bool = False,
    team_principal: str | None = None,
    team_saved_ms: int = 0,
    team_promotion: str | None = None,
) -> None:
    """Append one execution timing to a cell's persisted history.

    Feeds the profiling summary's cache-savings figure across restarts. Trimmed
    to the newest ``MAX_EXECUTION_SAMPLES`` since the file is rewritten every run.
    ``team_principal`` / ``team_saved_ms`` are set only for shared-store hits and
    must be stored: the reader never ran the cell, so there is no local run to
    price the hit against.
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    sample: dict[str, Any] = {"duration_ms": float(duration_ms), "cache_hit": bool(cache_hit)}
    # Only for a team hit, so ordinary runs do not grow dead keys.
    if from_team:
        sample["from_team"] = True
    if team_principal:
        sample["team_principal"] = team_principal
    if team_saved_ms:
        sample["team_saved_ms"] = int(team_saved_ms)
    if team_promotion:
        sample["team_promotion"] = team_promotion
    entry.execution_samples = [*entry.execution_samples, sample][-MAX_EXECUTION_SAMPLES:]
    save_runtime_state(notebook_dir, state)


def persist_cell_widget_values(
    notebook_dir: Path,
    cell_id: str,
    values: dict[str, Any],
) -> dict[str, Any]:
    """Merge *values* into a widget cell's persisted control values; return the merged map.

    Only the named controls change.
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.widget_values = {**entry.widget_values, **values}
    save_runtime_state(notebook_dir, state)
    return dict(entry.widget_values)


def persist_cell_test_result(
    notebook_dir: Path,
    cell_id: str,
    test_result: dict[str, Any] | None,
) -> None:
    """Persist a cell's last unit-test result; ``None`` clears it."""
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.test_result = test_result or None
    save_runtime_state(notebook_dir, state)


def migrate_from_legacy_notebook_toml(
    notebook_dir: Path,
    toml_data: dict[str, Any],
) -> bool:
    """Move legacy runtime fields out of notebook.toml into ``runtime.json``.

    Migrates per-cell ``display_outputs`` / ``display`` and the ``[environment]``
    section. Returns ``True`` when anything moved, so the caller rewrites
    notebook.toml. Idempotent.
    """
    state = load_runtime_state(notebook_dir)
    migrated = False

    legacy_artifacts = toml_data.get("artifacts")
    if isinstance(legacy_artifacts, dict):
        for cell_id, cell_artifacts in legacy_artifacts.items():
            if not isinstance(cell_artifacts, dict):
                continue
            entry = state.get_or_create_cell(cell_id)
            raw_outputs = cell_artifacts.get("display_outputs")
            if isinstance(raw_outputs, list) and not entry.display_outputs:
                cleaned = [dict(output) for output in raw_outputs if isinstance(output, dict)]
                if cleaned:
                    entry.display_outputs = cleaned
                    migrated = True
            raw_display = cell_artifacts.get("display")
            if isinstance(raw_display, dict) and raw_display and entry.display is None:
                entry.display = dict(raw_display)
                migrated = True

    legacy_environment = toml_data.get("environment")
    if isinstance(legacy_environment, dict) and legacy_environment:
        if state.environment == EnvironmentRuntime():
            state.environment = EnvironmentRuntime(**legacy_environment)
            migrated = True

    if migrated:
        save_runtime_state(notebook_dir, state)

    return migrated

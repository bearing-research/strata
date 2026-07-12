"""Persistent per-notebook runtime state.

``notebook.toml`` holds stable notebook configuration — cell list,
worker config, notebook-level env, mounts. Anything that changes on
every execution or background sync (display outputs, per-cell
provenance hashes, the last ``uv sync`` timestamp) lives here
instead. Storing it separately keeps ``notebook.toml`` diff-friendly
for version control and means example notebooks don't churn under
Git every time someone runs them.

The file is ``.strata/runtime.json`` and is gitignored alongside the
rest of ``.strata/``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_RUNTIME_FILENAME = "runtime.json"


@dataclass
class CellRuntime:
    """Per-cell runtime state — execution provenance and display outputs."""

    last_provenance_hash: str | None = None
    last_source_hash: str | None = None
    last_env_hash: str | None = None
    display_outputs: list[dict[str, Any]] = field(default_factory=list)
    display: dict[str, Any] | None = None
    test_result: dict[str, Any] | None = None
    # Current values of a widget cell's controls, keyed by variable name. The
    # committed source declares the controls + defaults; the user-set value is
    # runtime state (a slider drag must not churn ``notebook.toml``).
    widget_values: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Whether this entry carries no useful state.

        Empty entries are stripped on save so the file stays tidy.
        """
        return not (
            self.last_provenance_hash
            or self.last_source_hash
            or self.last_env_hash
            or self.display_outputs
            or self.display
            or self.test_result
            or self.widget_values
        )


@dataclass
class EnvironmentRuntime:
    """Snapshot of the notebook's runtime environment after a ``uv sync``.

    All fields default to empty / zero so missing keys on the disk read
    side and partial migrations resolve to a well-formed dataclass
    without manual ``setdefault`` calls.
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


@dataclass
class RRuntime:
    """Snapshot of the notebook's R-side runtime after ``renv::restore()``.

    Mirror of ``EnvironmentRuntime`` for R. Lives in ``.strata/runtime.json``
    rather than the committed ``notebook.toml`` so the per-session
    ``last_synced_at`` doesn't churn the on-disk notebook definition.

    Fields:

    * ``lock_hash`` / ``r_version`` / ``last_synced_at`` — the *last
      successful* sync. A failed re-sync leaves these alone so the UI
      can still show "last good state was X".
    * ``sync_error`` — error message from the most recent sync attempt.
      Cleared on success. Non-empty value means the latest attempt
      failed; the rest of the fields may still reflect a prior good
      sync (or all be empty if no sync has ever succeeded).

    All fields default to empty / zero. ``has_lockfile`` is NOT
    cached here — it's a current-disk fact (``renv.lock`` exists)
    derived at serialization time, not a "did we successfully sync
    once" historical claim.
    """

    lock_hash: str = ""
    r_version: str = ""
    last_synced_at: int = 0
    sync_error: str = ""


@dataclass
class RuntimeState:
    """Root of ``.strata/runtime.json`` — keyed cells + environment snapshot."""

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
        """Remove a per-cell entry — callers do this when the cell is deleted."""
        self.cells.pop(cell_id, None)


def runtime_state_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / _RUNTIME_FILENAME


def load_runtime_state(notebook_dir: Path) -> RuntimeState:
    """Return the runtime-state document, or a fresh empty shell.

    A missing or unparseable file produces an empty ``RuntimeState`` —
    runtime data is augmenting state, not authoritative, so a corrupt
    ``runtime.json`` must not prevent a notebook from opening.
    Schema-level mismatches (e.g. a hand-edited file with stray fields)
    raise from the dataclass constructor; the on-disk shape is
    Strata-written and trusted to match.
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
) -> None:
    """Persist the last successful execution provenance for a cell.

    These hashes let ``compute_staleness`` tell ``STALE`` from
    ``IDLE`` for cells whose canonical artifact has been evicted — a
    must-have for loop cells and long-lived notebooks re-opened after
    a GC pass. They live in ``.strata/runtime.json`` so they survive
    reopens without polluting the committed ``notebook.toml``.
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.last_provenance_hash = last_provenance_hash or None
    entry.last_source_hash = last_source_hash or None
    entry.last_env_hash = last_env_hash or None
    save_runtime_state(notebook_dir, state)


def persist_cell_test_result(
    notebook_dir: Path,
    cell_id: str,
    test_result: dict[str, Any] | None,
) -> None:
    """Persist (or clear) a cell's last unit-test result.

    Results live in ``.strata/runtime.json`` — they're runtime state, not
    committed config, so re-running a cell's tests never churns
    ``notebook.toml``. Passing ``None`` clears a prior result (the entry is
    pruned on save if it carries nothing else).
    """
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)
    entry.test_result = test_result or None
    save_runtime_state(notebook_dir, state)


def migrate_from_legacy_notebook_toml(
    notebook_dir: Path,
    toml_data: dict[str, Any],
) -> bool:
    """One-time migration of runtime fields out of notebook.toml.

    Returns ``True`` when at least one field was migrated so callers
    know to rewrite notebook.toml without the legacy sections.

    Scope for this migration step:

    * ``artifacts.<cell_id>.display_outputs`` / ``display`` →
      ``runtime.json`` ``cells.<cell_id>.display_outputs`` / ``display``.
    * The ``[cache]`` section is dropped because it's never been used.

    Migrations for environment metadata and per-cell provenance hashes
    land in later commits; this helper is additive and re-entrant, so
    running it twice is harmless.
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

"""Parse notebook directory and load notebook.toml + cell sources."""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

from strata.notebook.models import (
    CellMeta,
    CellOutput,
    CellState,
    ConnectionSpec,
    MalformedConnection,
    MountSpec,
    NotebookState,
    NotebookToml,
    VariantGroupConfig,
    WorkerSpec,
)


def _parse_connections(
    toml_data: dict,
) -> tuple[list[ConnectionSpec], list[MalformedConnection]]:
    """Split ``[connections.<name>]`` blocks into valid and malformed.

    TOML shape: ``{"connections": {"<name>": {"driver": ..., ...}, ...}}``.
    Two outputs:

    1. Valid ``ConnectionSpec``s — fully parsed, ready for the adapter.
    2. ``MalformedConnection`` records carrying the raw body and a
       human-readable error. The annotation_validation layer reads
       these to surface diagnostics; the writer round-trips them so a
       transient typo doesn't get erased by an unrelated save.

    Pydantic's ``extra="allow"`` on ``ConnectionSpec`` preserves
    driver-specific keys (``uri``, ``host``, ``account``, ...) for the
    adapter to interpret.

    Path values are kept exactly as written on disk so a save
    round-trips byte-for-byte. The cell executor resolves relative
    SQLite paths against the notebook directory at adapter-open time
    (see ``cell_executor._resolve_runtime_spec``), so the adapter
    itself stays a pure in-process call site without any notebook
    awareness.
    """
    raw = toml_data.get("connections")
    if not isinstance(raw, dict):
        return [], []
    valid: list[ConnectionSpec] = []
    malformed: list[MalformedConnection] = []
    for name, body in raw.items():
        name_str = str(name)
        if not isinstance(body, dict):
            malformed.append(
                MalformedConnection(
                    name=name_str,
                    body={},
                    error="connection body must be a TOML table",
                )
            )
            continue
        if "driver" not in body:
            malformed.append(
                MalformedConnection(
                    name=name_str,
                    body=dict(body),
                    error="connection is missing required 'driver' key",
                )
            )
            continue
        try:
            valid.append(ConnectionSpec(name=name_str, **body))
        except Exception as exc:
            malformed.append(
                MalformedConnection(
                    name=name_str,
                    body=dict(body),
                    error=f"validation failed: {exc}",
                )
            )
    return valid, malformed


def parse_notebook(directory: Path) -> NotebookState:
    """Parse notebook directory, load notebook.toml and cell files.

    Args:
        directory: Path to notebook directory

    Returns:
        NotebookState with all cells loaded

    Raises:
        FileNotFoundError: If notebook.toml is missing
    """
    directory = Path(directory)
    notebook_toml_path = directory / "notebook.toml"

    if not notebook_toml_path.exists():
        raise FileNotFoundError(f"notebook.toml not found at {notebook_toml_path}")

    # Read notebook.toml
    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Move runtime fields (display outputs, cache block) out of
    # notebook.toml on first open of a legacy notebook. The helper is
    # additive and a no-op once the migration has happened.
    from strata.notebook.runtime_state import (
        load_runtime_state,
        migrate_from_legacy_notebook_toml,
    )
    from strata.notebook.writer import _env_has_meaningful_content

    has_legacy_cache = "cache" in toml_data
    has_legacy_environment = isinstance(toml_data.get("environment"), dict) and bool(
        toml_data.get("environment")
    )
    # Drop an ``[env]`` block that has no meaningful content (empty, or
    # only blanked sensitive-key placeholders). This cleans up pollution
    # from earlier runs where a user typed an API key in the Runtime
    # panel and the sensitive-key blanking left an empty slot in the
    # committed notebook.toml.
    legacy_env = toml_data.get("env")
    has_empty_env_block = isinstance(legacy_env, dict) and not _env_has_meaningful_content(
        legacy_env
    )
    needs_rewrite = migrate_from_legacy_notebook_toml(directory, toml_data) or has_legacy_cache
    if needs_rewrite or has_legacy_environment or has_empty_env_block:
        toml_data.pop("artifacts", None)
        toml_data.pop("cache", None)
        toml_data.pop("environment", None)
        if has_empty_env_block:
            toml_data.pop("env", None)
        _rewrite_notebook_toml(notebook_toml_path, toml_data)
    # Even when nothing was migrated (runtime.json already exists), drop
    # the legacy sections from the in-memory parse result so downstream
    # code does not see them — the authoritative values live in
    # runtime.json from here on.
    toml_data.pop("artifacts", None)
    toml_data.pop("cache", None)
    toml_data.pop("environment", None)

    runtime_state = load_runtime_state(directory)
    runtime_cells = runtime_state.cells

    # Parse into NotebookToml
    # Get created_at and updated_at, defaulting to now if not present
    created_at = toml_data.get("created_at")
    if created_at is None:
        created_at = datetime.now(tz=UTC)

    updated_at = toml_data.get("updated_at")
    if updated_at is None:
        updated_at = datetime.now(tz=UTC)

    _parsed_connections, _parsed_malformed = _parse_connections(toml_data)
    _parsed_variant_groups = _parse_variant_groups(toml_data)

    notebook_toml = NotebookToml(
        notebook_id=toml_data.get("notebook_id", ""),
        name=toml_data.get("name", "Untitled Notebook"),
        created_at=created_at,
        updated_at=updated_at,
        worker=toml_data.get("worker"),
        timeout=toml_data.get("timeout"),
        env=toml_data.get("env", {}),
        workers=[WorkerSpec(**worker) for worker in toml_data.get("workers", [])],
        cells=[CellMeta(**cell_meta) for cell_meta in toml_data.get("cells", [])],
        mounts=[MountSpec(**m) for m in toml_data.get("mounts", [])],
        connections=_parsed_connections,
        malformed_connections=_parsed_malformed,
        variant_groups=_parsed_variant_groups,
        ai=toml_data.get("ai", {}),
        secret_manager=toml_data.get("secret_manager", {}),
        strata=toml_data.get("strata", {}),
        artifacts=toml_data.get("artifacts", {}),
        environment=toml_data.get("environment", {}),
        cache=toml_data.get("cache", {}),
    )

    # Load cell sources
    cells_dir = directory / "cells"
    cell_states: list[CellState] = []

    # Build notebook-level mount defaults (keyed by name for cell overrides)
    notebook_mounts = {m.name: m for m in notebook_toml.mounts}

    for cell_meta in notebook_toml.cells:
        cell_file = cells_dir / cell_meta.file
        source = ""

        if cell_file.exists():
            with open(cell_file, encoding="utf-8") as f:
                source = f.read()

        # Resolve mounts: notebook-level defaults, overridden by cell-level
        resolved_mounts = dict(notebook_mounts)
        for m in cell_meta.mounts:
            resolved_mounts[m.name] = m
        resolved_worker = cell_meta.worker or notebook_toml.worker
        resolved_timeout = (
            cell_meta.timeout if cell_meta.timeout is not None else notebook_toml.timeout
        )
        resolved_env = dict(notebook_toml.env)
        resolved_env.update(cell_meta.env)
        runtime_cell = runtime_cells.get(cell_meta.id)
        if runtime_cell is not None:
            display_outputs = [CellOutput(**d) for d in runtime_cell.display_outputs]
            if not display_outputs and runtime_cell.display:
                display_outputs = [CellOutput(**runtime_cell.display)]
        else:
            display_outputs = []

        # Restore console output from .strata/console/
        from strata.notebook.writer import load_cell_console_output

        console_stdout, console_stderr = load_cell_console_output(directory, cell_meta.id)

        # Persisted execution provenance from ``.strata/runtime.json``.
        # compute_staleness() compares these against freshly-computed
        # hashes, so hydrating them at open lets a reopened notebook
        # correctly classify cells as READY / STALE without a
        # re-execution.
        cell_states.append(
            CellState(
                id=cell_meta.id,
                source=source,
                language=cell_meta.language,
                order=cell_meta.order,
                worker=resolved_worker,
                worker_override=cell_meta.worker,
                timeout=resolved_timeout,
                timeout_override=cell_meta.timeout,
                env=resolved_env,
                env_overrides=dict(cell_meta.env),
                mounts=list(resolved_mounts.values()),
                mount_overrides=list(cell_meta.mounts),
                display_outputs=display_outputs,
                display_output=display_outputs[-1] if display_outputs else None,
                console_stdout=console_stdout,
                console_stderr=console_stderr,
                last_provenance_hash=runtime_cell.last_provenance_hash if runtime_cell else None,
                last_source_hash=runtime_cell.last_source_hash if runtime_cell else None,
                last_env_hash=runtime_cell.last_env_hash if runtime_cell else None,
            )
        )

    # Sort by order
    cell_states.sort(key=lambda c: c.order)

    return NotebookState(
        id=notebook_toml.notebook_id,
        name=notebook_toml.name,
        owner=notebook_toml.owner,
        worker=notebook_toml.worker,
        timeout=notebook_toml.timeout,
        env=dict(notebook_toml.env),
        workers=list(notebook_toml.workers),
        mounts=list(notebook_toml.mounts),
        connections=list(notebook_toml.connections),
        malformed_connections=list(notebook_toml.malformed_connections),
        secret_manager_config=dict(notebook_toml.secret_manager),
        cells=cell_states,
        variant_active_selections={vg.group: vg.active for vg in notebook_toml.variant_groups},
        path=directory,
        created_at=notebook_toml.created_at,
        updated_at=notebook_toml.updated_at,
    )


def _parse_variant_groups(toml_data: dict) -> list[VariantGroupConfig]:
    """Parse ``[[variant_group]]`` entries into VariantGroupConfig.

    Malformed entries (missing ``group`` / ``active``, or values that
    don't match the identifier pattern) are dropped silently here;
    annotation_validation surfaces a ``variant_active_unknown`` diagnostic
    if the named active variant doesn't exist in the cells.
    """
    raw = toml_data.get("variant_group")
    if not isinstance(raw, list):
        return []
    out: list[VariantGroupConfig] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            out.append(
                VariantGroupConfig(group=entry.get("group", ""), active=entry.get("active", ""))
            )
        except Exception:
            continue
    return out


def _rewrite_notebook_toml(path: Path, toml_data: dict) -> None:
    """Write a pre-parsed TOML dict back to disk (used by migration).

    Unlike ``write_notebook_toml`` this preserves whatever shape the
    caller has constructed, including fields not modelled by
    ``NotebookToml``. Used when the migration helper has already
    stripped legacy runtime sections from ``toml_data``.
    """
    with open(path, "wb") as f:
        tomli_w.dump(toml_data, f)

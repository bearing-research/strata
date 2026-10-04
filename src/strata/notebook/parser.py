"""Parse notebook directory and load notebook.toml + cell sources."""

from __future__ import annotations

import os
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from strata.notebook.models import (
    CellMeta,
    CellOutput,
    CellState,
    CellTestResult,
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
    """Split ``[connections.<name>]`` blocks into valid specs and malformed records.

    ``MalformedConnection`` keeps the raw body and an error: annotation
    validation surfaces it, and the writer round-trips it so a typo is not
    erased by an unrelated save. ``extra="allow"`` keeps driver-specific keys.
    Paths stay as written; relative ones are resolved at adapter-open time
    (``cell_executor._resolve_runtime_spec``).
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


def _parse_catalogs(toml_data: dict) -> dict[str, dict]:
    """``[catalogs.<name>]`` tables; anything else under ``catalogs`` is not a catalog."""
    raw = toml_data.get("catalogs")
    if not isinstance(raw, dict):
        return {}
    return {str(name): dict(body) for name, body in raw.items() if isinstance(body, dict)}


def parse_notebook(directory: Path) -> NotebookState:
    """Parse a notebook directory: notebook.toml plus every cell file.

    Raises:
        FileNotFoundError: If notebook.toml is missing.
    """
    directory = Path(directory)
    notebook_toml_path = directory / "notebook.toml"

    if not notebook_toml_path.exists():
        raise FileNotFoundError(f"notebook.toml not found at {notebook_toml_path}")

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Move legacy runtime fields out of notebook.toml; a no-op once migrated.
    from strata.notebook.runtime_state import (
        load_runtime_state,
        migrate_from_legacy_notebook_toml,
    )
    from strata.notebook.writer import _env_has_meaningful_content

    has_legacy_cache = "cache" in toml_data
    has_legacy_environment = isinstance(toml_data.get("environment"), dict) and bool(
        toml_data.get("environment")
    )
    # Drop an ``[env]`` block that is empty or holds only blanked secrets.
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
    # Drop legacy sections even when nothing migrated: runtime.json is
    # authoritative for them.
    toml_data.pop("artifacts", None)
    toml_data.pop("cache", None)
    toml_data.pop("environment", None)

    runtime_state = load_runtime_state(directory)
    runtime_cells = runtime_state.cells

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
        owner=toml_data.get("owner"),
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
        catalogs=_parse_catalogs(toml_data),
        secret_manager=toml_data.get("secret_manager", {}),
        r=toml_data.get("r", {}),
        artifacts=toml_data.get("artifacts", {}),
        environment=toml_data.get("environment", {}),
        cache=toml_data.get("cache", {}),
    )

    cells_dir = directory / "cells"
    cell_states: list[CellState] = []

    notebook_mounts = {m.name: m for m in notebook_toml.mounts}

    for cell_meta in notebook_toml.cells:
        cell_file = cells_dir / cell_meta.file
        source = ""

        if cell_file.exists():
            with open(cell_file, encoding="utf-8") as f:
                source = f.read()

        # A hand-edited id with path separators must not escape ``cells/``.
        test_path = os.path.realpath(os.path.join(cells_dir, f"{cell_meta.id}.test.py"))
        cells_root = os.path.realpath(cells_dir)
        test_source = ""
        if test_path.startswith(cells_root + os.sep) and os.path.isfile(test_path):
            with open(test_path, encoding="utf-8") as f:
                test_source = f.read()

        # Cell-level mounts override notebook-level defaults
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

        test_result = (
            CellTestResult(**runtime_cell.test_result)
            if runtime_cell is not None and runtime_cell.test_result
            else None
        )

        from strata.notebook.writer import load_cell_console_output

        console_stdout, console_stderr = load_cell_console_output(directory, cell_meta.id)

        # Persisted provenance lets compute_staleness() classify a reopened
        # notebook's cells as READY / STALE without re-executing.
        cell_states.append(
            CellState(
                id=cell_meta.id,
                source=source,
                test_source=test_source,
                test_result=test_result,
                language=cell_meta.language,
                order=cell_meta.order,
                created_by=cell_meta.created_by,
                updated_by=cell_meta.updated_by,
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
                last_reopen_identity=(runtime_cell.last_reopen_identity if runtime_cell else None),
                error=runtime_cell.last_error if runtime_cell else None,
                error_source_hash=(runtime_cell.last_error_source_hash if runtime_cell else None),
                widget_values=dict(runtime_cell.widget_values) if runtime_cell else {},
            )
        )

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
        catalogs=dict(notebook_toml.catalogs),
        secret_manager_config=dict(notebook_toml.secret_manager),
        r=dict(notebook_toml.r),
        cells=cell_states,
        variant_active_selections={vg.group: vg.active for vg in notebook_toml.variant_groups},
        variant_modes={vg.group: vg.mode for vg in notebook_toml.variant_groups},
        path=directory,
        created_at=notebook_toml.created_at,
        updated_at=notebook_toml.updated_at,
    )


def _parse_variant_groups(toml_data: dict) -> list[VariantGroupConfig]:
    """Parse ``[[variant_group]]`` entries into VariantGroupConfig.

    Malformed entries are dropped silently; annotation validation reports an
    active variant that does not exist.
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
                VariantGroupConfig(
                    group=entry.get("group", ""),
                    active=entry.get("active", ""),
                    mode=entry.get("mode", "switch"),
                )
            )
        except Exception:
            continue
    return out


def _rewrite_notebook_toml(path: Path, toml_data: dict) -> None:
    """Write a pre-parsed TOML dict back to disk for migration.

    Unlike ``write_notebook_toml``, keeps fields ``NotebookToml`` does not model.
    """
    from strata.notebook.writer import _dump_notebook_toml

    with open(path, "wb") as f:
        _dump_notebook_toml(toml_data, f)

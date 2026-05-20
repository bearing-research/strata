"""Write notebook changes back to disk (notebook.toml and cell files)."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import time
import tomllib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import tomli_w

from strata.notebook.models import (
    CellMeta,
    ConnectionSpec,
    MalformedConnection,
    MountSpec,
    NotebookToml,
    WorkerSpec,
)
from strata.notebook.python_versions import (
    current_python_minor,
    format_requires_python,
    normalize_python_minor,
    read_requested_python_minor,
    read_venv_runtime_python_version,
)

if TYPE_CHECKING:
    pass


def _serialize_mounts(mounts: list[MountSpec]) -> list[dict[str, Any]]:
    """Convert mount specs into TOML-friendly dicts."""
    return [
        {
            "name": mount.name,
            "uri": mount.uri,
            "mode": mount.mode.value,
            **({"pin": mount.pin} if mount.pin is not None else {}),
            **({"options": dict(mount.options)} if mount.options else {}),
        }
        for mount in mounts
    ]


_SENSITIVE_KEY_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def _is_sensitive_env_key(key: str) -> bool:
    """Return True if the env var name looks like a secret."""
    upper = key.upper()
    return any(pattern in upper for pattern in _SENSITIVE_KEY_PATTERNS)


def _serialize_env(env: dict[str, str]) -> dict[str, str]:
    """Convert env vars into a TOML-friendly dict.

    Values for sensitive-looking keys (API keys, tokens, passwords) are
    stripped so they never reach disk. The key names are preserved so
    the notebook remembers *which* vars are configured — the user
    re-enters values via the Runtime panel on next open.
    """
    return {
        key: ("" if _is_sensitive_env_key(key) else value) for key, value in sorted(env.items())
    }


def _env_has_meaningful_content(env: dict[str, str]) -> bool:
    """Return True if the env dict has any non-empty, non-sensitive value.

    An ``[env]`` block where every entry is either empty or a blanked
    sensitive-key placeholder carries no real configuration — just
    noise that pollutes committed notebooks. This predicate lets the
    writer skip persisting such blocks entirely.
    """
    for key, value in env.items():
        if not value:
            continue
        if _is_sensitive_env_key(key):
            continue
        return True
    return False


_AUTH_INDIRECTION_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")


def is_auth_indirection(value: object) -> bool:
    """Return True for ``${VAR}`` env-var indirections.

    Used by both the writer (to scrub literals before disk) and the
    annotation_validation layer (to warn about literals before they
    get scrubbed). The regex matches ASCII identifiers wrapped in
    ``${…}`` — empty braces, lowercase-only names, and shell-style
    ``$VAR`` forms are all rejected on purpose to keep the contract
    narrow and unambiguous.
    """
    return isinstance(value, str) and bool(_AUTH_INDIRECTION_RE.match(value))


def _scrub_auth_for_disk(auth: dict[str, Any]) -> dict[str, Any]:
    """Replace literal auth values with empty strings before writing.

    Mirrors the existing ``_serialize_env`` behavior for sensitive env
    keys: the key name is preserved so the notebook remembers *which*
    credentials are configured, but the literal value is dropped so
    secrets never reach disk. Only ``${VAR}`` indirections pass
    through unchanged. Non-string values (a typo writing
    ``password = 1234``) are coerced to empty string.
    """
    return {k: (v if is_auth_indirection(v) else "") for k, v in auth.items()}


def _serialize_connections(
    connections: list[ConnectionSpec],
    malformed: list[MalformedConnection] | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert connection specs into the ``[connections.<name>]`` TOML shape.

    Round-trips both valid and malformed blocks so an unrelated save
    (cell add/remove, worker change) doesn't erase a hand-edited
    typo. ``auth`` values that aren't ``${VAR}`` indirections are
    blanked here so secrets never reach disk; the validation layer
    surfaces the corresponding diagnostic before the user saves.
    """
    out: dict[str, dict[str, Any]] = {}
    for conn in connections:
        body = conn.model_dump(exclude={"name"})
        if body.get("auth"):
            body["auth"] = _scrub_auth_for_disk(body["auth"])
        else:
            body.pop("auth", None)
        if not body.get("options"):
            body.pop("options", None)
        out[conn.name] = body

    for mal in malformed or []:
        if mal.name in out:
            # Same name appears as both valid and malformed — the valid
            # entry wins. This shouldn't happen in practice (the parser
            # only emits a malformed record when the valid path errors)
            # but guards against double-write.
            continue
        body = dict(mal.body)
        if isinstance(body.get("auth"), dict):
            body["auth"] = _scrub_auth_for_disk(body["auth"])
        out[mal.name] = body

    return out


def _serialize_workers(workers: list[WorkerSpec]) -> list[dict[str, object]]:
    """Convert worker specs into TOML-friendly dicts."""
    return [
        {
            "name": worker.name,
            "backend": worker.backend.value,
            **({"runtime_id": worker.runtime_id} if worker.runtime_id else {}),
            **({"config": worker.config} if worker.config else {}),
        }
        for worker in workers
    ]


def _sanitize_display_output_for_toml(
    display_output: dict[str, object] | None,
) -> dict[str, object] | None:
    """Strip transient fields before persisting cell display metadata.

    Uses ``to_serialization_safe`` from the serializer module as the
    single boundary for TOML/JSON compatibility — no separate None
    stripping needed.
    """
    from strata.notebook.serializer import to_serialization_safe

    if display_output is None:
        return None

    persisted = dict(display_output)
    persisted.pop("inline_data_url", None)
    persisted.pop("file", None)
    persisted.pop("markdown_text", None)
    cleaned = {key: value for key, value in persisted.items() if value is not None}
    return to_serialization_safe(cleaned)


def _sanitize_display_outputs_for_toml(
    display_outputs: list[dict[str, object]] | None,
) -> list[dict[str, object]]:
    """Strip transient fields from a display output list before persistence."""
    if not display_outputs:
        return []

    persisted_outputs: list[dict[str, object]] = []
    for display_output in display_outputs:
        persisted = _sanitize_display_output_for_toml(display_output)
        if persisted:
            persisted_outputs.append(persisted)
    return persisted_outputs


def write_cell(notebook_dir: Path, cell_id: str, source: str) -> None:
    """Write cell source to disk.

    Args:
        notebook_dir: Path to notebook directory
        cell_id: Cell ID
        source: Cell source code

    Raises:
        ValueError: If cell not found in notebook.toml
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    # Read current notebook.toml
    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Find cell metadata
    cells_data = toml_data.get("cells", [])
    cell_meta = None
    for cell in cells_data:
        if cell.get("id") == cell_id:
            cell_meta = cell
            break

    if cell_meta is None:
        # FileNotFoundError (not ValueError) so route handlers can map
        # this to 404 without ambiguity with validation errors.
        raise FileNotFoundError(f"Cell {cell_id} not found in notebook.toml")

    # Write cell file
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    cell_file = cells_dir / cell_meta["file"]
    cell_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cell_file, "w", encoding="utf-8") as f:
        f.write(source)


def write_notebook_toml(notebook_dir: Path, toml: NotebookToml) -> None:
    """Write notebook.toml to disk.

    ``notebook.toml`` holds stable notebook configuration only — cell
    list, workers, env, mounts, timeout, ai. Anything that changes on
    every execution or background sync (display outputs, console, per-
    cell provenance hashes, the last ``uv sync`` timestamp) lives in
    ``.strata/runtime.json`` via ``runtime_state.py``. Callers that
    only need to persist runtime state must not touch this file — that
    keeps ``updated_at`` meaningful as a structural-change signal.

    Args:
        notebook_dir: Path to notebook directory
        toml: NotebookToml model to write
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    # Convert to dict for TOML serialization
    toml_data = {
        "notebook_id": toml.notebook_id,
        "name": toml.name,
        **({"owner": toml.owner} if toml.owner else {}),
        "created_at": (
            toml.created_at.isoformat()
            if isinstance(toml.created_at, datetime)
            else toml.created_at
        ),
        "updated_at": (
            toml.updated_at.isoformat()
            if isinstance(toml.updated_at, datetime)
            else toml.updated_at
        ),
        "cells": [
            {
                "id": cell.id,
                "file": cell.file,
                "language": cell.language,
                "order": cell.order,
                **({"worker": cell.worker} if cell.worker is not None else {}),
                **({"timeout": cell.timeout} if cell.timeout is not None else {}),
                **(
                    {"env": _serialize_env(cell.env)}
                    if cell.env and _env_has_meaningful_content(cell.env)
                    else {}
                ),
                "mounts": _serialize_mounts(cell.mounts),
            }
            for cell in toml.cells
        ],
        **({"worker": toml.worker} if toml.worker is not None else {}),
        **({"timeout": toml.timeout} if toml.timeout is not None else {}),
        **(
            {"env": _serialize_env(toml.env)}
            if toml.env and _env_has_meaningful_content(toml.env)
            else {}
        ),
        "workers": _serialize_workers(toml.workers),
        "mounts": _serialize_mounts(toml.mounts),
        **(
            {
                "connections": _serialize_connections(
                    toml.connections,
                    toml.malformed_connections,
                )
            }
            if toml.connections or toml.malformed_connections
            else {}
        ),
        **(
            {
                "variant_group": [
                    {"group": vg.group, "active": vg.active} for vg in toml.variant_groups
                ]
            }
            if toml.variant_groups
            else {}
        ),
        **({"ai": toml.ai} if toml.ai else {}),
        **({"secret_manager": toml.secret_manager} if toml.secret_manager else {}),
        # Runtime state that used to live in this file — ``artifacts``
        # (display outputs), ``environment`` (sync timestamps, package
        # counts), ``cache`` — now lives in ``.strata/runtime.json``.
        # Keeping it out of ``notebook.toml`` means example notebooks
        # stop producing multi-KB git diffs every time someone runs
        # them.
    }

    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def create_notebook(
    parent_dir: Path,
    name: str,
    python_version: str | None = None,
    *,
    initialize_environment: bool = True,
    owner: str | None = None,
) -> Path:
    """Create a new notebook directory with notebook.toml and pyproject.toml.

    Args:
        parent_dir: Parent directory for the notebook
        name: Notebook name (used for folder and notebook name)
        python_version: Requested notebook Python major.minor version
        initialize_environment: Whether to create the notebook venv immediately
        owner: Opaque identity string stamped into notebook.toml. None means
            unowned (the default for non-shared deployments). Set by callers
            that have resolved a caller identity from a request header.

    Returns:
        Path to created notebook directory
    """
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    requested_python_version = (
        normalize_python_minor(python_version)
        if python_version is not None
        else current_python_minor()
    )

    # Validate notebook name
    if "/" in name or "\\" in name or ".." in name or "\0" in name:
        raise ValueError("Notebook name contains invalid characters")

    # Create notebook directory (slugify the name)
    notebook_dir = parent_dir / name.lower().replace(" ", "_")
    notebook_dir.mkdir(exist_ok=True)

    # Create cells subdirectory
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    # Preserve existing notebook ID if notebook.toml already exists.
    # Overwriting the ID would orphan all artifacts keyed to the old ID.
    existing_toml_path = notebook_dir / "notebook.toml"
    existing_notebook_id = None
    existing_created_at = None
    existing_cells: list[CellMeta] = []
    if existing_toml_path.exists():
        try:
            with open(existing_toml_path, "rb") as f:
                raw = tomllib.load(f)
            existing_notebook_id = raw.get("notebook_id")
            raw_created = raw.get("created_at")
            if isinstance(raw_created, datetime):
                existing_created_at = raw_created
            for c in raw.get("cells", []):
                if isinstance(c, dict) and "id" in c and "file" in c:
                    existing_cells.append(
                        CellMeta(
                            id=c["id"],
                            file=c["file"],
                            language=c.get("language", "python"),
                            order=c.get("order", 0),
                        )
                    )
        except Exception:
            pass

    notebook_id = existing_notebook_id or str(uuid.uuid4())

    # Preserve owner if the existing notebook.toml had one — only stamp the
    # incoming owner on a genuinely new notebook so re-creating with a
    # different identity doesn't silently take over someone else's work.
    existing_owner: str | None = None
    if existing_toml_path.exists():
        try:
            with open(existing_toml_path, "rb") as f:
                raw_existing = tomllib.load(f)
            raw_owner = raw_existing.get("owner")
            if isinstance(raw_owner, str) and raw_owner:
                existing_owner = raw_owner
        except Exception:
            pass

    # Create notebook.toml
    now = datetime.now(tz=UTC)
    notebook_toml = NotebookToml(
        notebook_id=notebook_id,
        name=name,
        owner=existing_owner if existing_owner is not None else owner,
        created_at=existing_created_at or now,
        updated_at=now,
        cells=existing_cells,
    )
    write_notebook_toml(notebook_dir, notebook_toml)

    # Create pyproject.toml (minimal). The notebook runtime expects
    # pyarrow (DataFrame / Series / ndarray serialization), orjson
    # (manifest I/O), and cloudpickle (default object codec) to be
    # importable inside the notebook venv. They're cheap wheels on all
    # supported platforms; baking them into the template avoids
    # silent fallbacks to slower stdlib json / stdlib pickle.
    pyproject_content = f'''[project]
name = "{name.lower().replace(" ", "-")}"
version = "0.1.0"
description = ""
requires-python = "{format_requires_python(requested_python_version)}"
dependencies = [
    "pyarrow>=18.0.0",
    "orjson>=3.10.0",
    "cloudpickle>=3.0.0",
]

[tool.uv]
'''

    with open(notebook_dir / "pyproject.toml", "w", encoding="utf-8") as f:
        f.write(pyproject_content)

    if initialize_environment:
        # Run uv sync to create venv + uv.lock (best-effort)
        _uv_sync(notebook_dir, python_version=requested_python_version)

        # Populate environment section with lockfile hash + python version
        _update_environment_metadata(notebook_dir)

    return notebook_dir


_logger = logging.getLogger(__name__)


def _uv_sync(notebook_dir: Path, *, timeout: int = 60, python_version: str | None = None) -> bool:
    """Run ``uv sync`` in *notebook_dir*.

    Returns True on success, False on failure (logged, never raised).
    """
    command = ["uv", "sync"]
    if python_version is not None:
        command.extend(["--python", normalize_python_minor(python_version)])
    try:
        subprocess.run(
            command,
            cwd=str(notebook_dir),
            timeout=timeout,
            capture_output=True,
            check=True,
        )
        _logger.debug("uv sync succeeded in %s", notebook_dir)
        return True
    except FileNotFoundError:
        _logger.warning("uv not found on PATH — skipping venv creation")
    except subprocess.TimeoutExpired:
        _logger.warning("uv sync timed out after %ds in %s", timeout, notebook_dir)
    except subprocess.CalledProcessError as exc:
        _logger.warning(
            "uv sync failed in %s: %s",
            notebook_dir,
            exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)",
        )
    return False


def _update_environment_metadata(notebook_dir: Path) -> None:
    """Persist the environment-metadata snapshot for a notebook.

    Records lockfile_hash, python_version, package counts, and the
    last ``uv sync`` timestamp so clients can detect environment
    changes without recomputing hashes themselves. Lives in
    ``.strata/runtime.json`` under ``environment`` — these values
    change on every sync and don't belong in the committed
    ``notebook.toml``.
    """
    from strata.notebook.dependencies import list_dependencies
    from strata.notebook.env import compute_lockfile_hash
    from strata.notebook.runtime_state import (
        EnvironmentRuntime,
        load_runtime_state,
        save_runtime_state,
    )

    if not (notebook_dir / "notebook.toml").exists():
        return

    requested_python_version = read_requested_python_minor(notebook_dir) or ""
    runtime_python_version = ""
    venv_python = notebook_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        runtime_python_version = read_venv_runtime_python_version(venv_python) or ""
        if not runtime_python_version:
            try:
                result = subprocess.run(
                    [
                        str(venv_python),
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
                    cwd=str(notebook_dir),
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=10,
                )
                runtime_python_version = result.stdout.strip()
            except Exception:
                _logger.debug("Failed to probe notebook venv python version", exc_info=True)

    resolved_package_count = 0
    lock_path = notebook_dir / "uv.lock"
    if lock_path.exists():
        try:
            with open(lock_path, "rb") as f:
                lock_data = tomllib.load(f)
            packages = lock_data.get("package", [])
            resolved_package_count = len(packages) if isinstance(packages, list) else 0
        except Exception:
            _logger.debug("Failed to parse uv.lock for env metadata", exc_info=True)

    declared_package_count = len(list_dependencies(notebook_dir))
    state = load_runtime_state(notebook_dir)
    state.environment = EnvironmentRuntime(
        requested_python_version=requested_python_version,
        runtime_python_version=runtime_python_version,
        lockfile_hash=compute_lockfile_hash(notebook_dir),
        python_version=runtime_python_version,
        package_count=declared_package_count,
        declared_package_count=declared_package_count,
        resolved_package_count=resolved_package_count,
        has_lockfile=lock_path.exists(),
        last_synced_at=int(time.time() * 1000),
    )
    save_runtime_state(notebook_dir, state)


def update_environment_metadata(notebook_dir: Path) -> None:
    """Public API: refresh ``[environment]`` in ``notebook.toml``.

    Called after ``uv add`` / ``uv remove`` to persist the new lockfile hash.
    """
    _update_environment_metadata(notebook_dir)


def add_cell_to_notebook(
    notebook_dir: Path,
    cell_id: str,
    after_cell_id: str | None = None,
    language: str = "python",
) -> None:
    """Add a new cell to the notebook.

    Args:
        notebook_dir: Path to notebook directory
        cell_id: New cell ID
        after_cell_id: Cell ID to add after (None = at end)
        language: Cell language ("python", "prompt", "markdown", or "sql")
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Calculate order. "Right after X" means landing between X and the
    # next cell in source order — a plain ``X.order + 0.5`` collides
    # whenever two inserts target the same parent (the second insert
    # ends up tied with the first, and stable sort makes "after X"
    # actually mean "after the cell that was added previously"). Use
    # the midpoint to the next cell instead.
    cells_data = toml_data.get("cells", [])
    if after_cell_id:
        idx = next((i for i, c in enumerate(cells_data) if c.get("id") == after_cell_id), None)
        if idx is not None:
            after_order = cells_data[idx].get("order", 0)
            next_order = None
            for other in cells_data:
                other_order = other.get("order", 0)
                if other_order > after_order and (next_order is None or other_order < next_order):
                    next_order = other_order
            order = after_order + 1.0 if next_order is None else (after_order + next_order) / 2.0
        else:
            order = len(cells_data)
    else:
        order = len(cells_data)

    # Create cell file. Markdown cells use ``.md`` so the file is editable
    # outside the notebook UI (in any markdown-aware editor) without the
    # ``.py`` extension confusing syntax highlighters.
    extension = "md" if language == "markdown" else "py"
    cell_filename = f"{cell_id}.{extension}"
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    with open(cells_dir / cell_filename, "w", encoding="utf-8") as f:
        f.write("")

    # Add to cells list
    cells_data.append(
        {
            "id": cell_id,
            "file": cell_filename,
            "language": language,
            "order": order,
        }
    )

    # Re-sort cells by order
    cells_data.sort(key=lambda c: c.get("order", 0))

    toml_data["cells"] = cells_data
    toml_data["updated_at"] = datetime.now(tz=UTC).isoformat()

    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def remove_cell_from_notebook(notebook_dir: Path, cell_id: str) -> None:
    """Remove a cell from the notebook.

    Args:
        notebook_dir: Path to notebook directory
        cell_id: Cell ID to remove

    Raises:
        ValueError: If cell not found
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    cells_data = toml_data.get("cells", [])
    cell_meta = None
    cell_idx = None

    for i, cell in enumerate(cells_data):
        if cell.get("id") == cell_id:
            cell_meta = cell
            cell_idx = i
            break

    if cell_meta is None:
        # FileNotFoundError (not ValueError) so route handlers can map
        # this to 404 without ambiguity with validation errors.
        raise FileNotFoundError(f"Cell {cell_id} not found")

    # Remove cell file
    cells_dir = notebook_dir / "cells"
    cell_file = cells_dir / cell_meta["file"]
    if cell_file.exists():
        cell_file.unlink()

    # Remove from cells list
    cells_data.pop(cell_idx)
    toml_data["cells"] = cells_data
    toml_data["updated_at"] = datetime.now(tz=UTC).isoformat()

    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def reorder_cells(notebook_dir: Path, cell_ids: list[str]) -> None:
    """Reorder cells in the notebook.

    Args:
        notebook_dir: Path to notebook directory
        cell_ids: Ordered list of cell IDs
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    cells_data = toml_data.get("cells", [])

    # Create mapping of cell_id to metadata
    cell_map = {cell.get("id"): cell for cell in cells_data}

    # Reorder and update order field
    new_cells = []
    for i, cell_id in enumerate(cell_ids):
        if cell_id in cell_map:
            cell = cell_map[cell_id]
            cell["order"] = i
            new_cells.append(cell)

    toml_data["cells"] = new_cells
    toml_data["updated_at"] = datetime.now(tz=UTC).isoformat()

    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def update_requires_python(notebook_dir: Path, new_minor: str) -> str:
    """Rewrite ``requires-python`` in a notebook's pyproject.toml.

    Returns the *previous* requires-python value so the caller can
    roll back (eg. when the subsequent ``uv sync`` fails on the new
    minor). ``new_minor`` is normalized via
    ``format_requires_python`` so the on-disk form is always the
    canonical ``==X.Y.*`` spec.

    The pyproject is rewritten in place — uv.lock and .venv are NOT
    touched here; the caller orchestrates the rest of the version-
    change pipeline.
    """
    pyproject_path = Path(notebook_dir) / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"Notebook pyproject not found: {pyproject_path}")

    text = pyproject_path.read_text(encoding="utf-8")
    new_spec = format_requires_python(new_minor)
    new_line = f'requires-python = "{new_spec}"'

    pattern = re.compile(r'^requires-python\s*=\s*"[^"]*"', re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Notebook pyproject is missing a requires-python line: {pyproject_path}")

    old_line = match.group(0)
    # Capture the old spec value (between quotes) for the return.
    old_spec_match = re.search(r'"([^"]*)"', old_line)
    old_spec = old_spec_match.group(1) if old_spec_match else ""

    if old_line == new_line:
        # Caller is responsible for the equal-to-current no-op short
        # circuit; if it gets here anyway, no harm done.
        return old_spec

    updated = text[: match.start()] + new_line + text[match.end() :]
    pyproject_path.write_text(updated, encoding="utf-8")
    return old_spec


def rename_notebook(notebook_dir: Path, new_name: str) -> None:
    """Rename the notebook.

    Args:
        notebook_dir: Path to notebook directory
        new_name: New notebook name
    """
    normalized_name = new_name.strip()

    if not normalized_name:
        raise ValueError("Notebook name cannot be empty")
    if (
        "/" in normalized_name
        or "\\" in normalized_name
        or ".." in normalized_name
        or "\0" in normalized_name
    ):
        raise ValueError("Notebook name contains invalid characters")

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("name") == normalized_name:
            return False
        toml_data["name"] = normalized_name
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def delete_notebook_directory(notebook_dir: Path) -> None:
    """Delete a notebook directory and all notebook-owned runtime state."""
    notebook_dir = Path(notebook_dir).resolve()
    notebook_toml_path = notebook_dir / "notebook.toml"

    if notebook_dir.is_symlink():
        raise ValueError("Refusing to delete a symlinked notebook directory")
    if not notebook_dir.exists():
        raise FileNotFoundError(f"Notebook directory not found: {notebook_dir}")
    if not notebook_dir.is_dir():
        raise ValueError(f"Notebook path is not a directory: {notebook_dir}")
    if not notebook_toml_path.is_file():
        raise ValueError(f"Notebook directory missing notebook.toml: {notebook_dir}")

    shutil.rmtree(notebook_dir)


def _apply_notebook_toml_update(
    notebook_dir: Path,
    mutate: Callable[[dict[str, Any]], bool],
) -> None:
    """Load ``notebook.toml``, apply ``mutate``, rewrite only when the
    mutator reports a real change.

    ``mutate`` receives the loaded dict and returns ``True`` iff it
    modified something worth persisting. When it returns ``False`` the
    file is left untouched — no rewrite, no ``updated_at`` bump — so
    ``updated_at`` keeps tracking actual structural edits.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    if not mutate(toml_data):
        return

    toml_data["updated_at"] = datetime.now(tz=UTC).isoformat()
    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def update_notebook_mounts(notebook_dir: Path, mounts: list[MountSpec]) -> None:
    """Persist notebook-level mount defaults."""
    new_mounts = _serialize_mounts(mounts)

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("mounts", []) == new_mounts:
            return False
        toml_data["mounts"] = new_mounts
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_connections(
    notebook_dir: Path,
    connections: list[ConnectionSpec],
    malformed: list[MalformedConnection] | None = None,
) -> None:
    """Persist notebook-level ``[connections.<name>]`` blocks.

    Round-trips both valid and malformed connections (same as the
    full-notebook writer) so a transient parse error in one entry
    doesn't get erased by an unrelated edit elsewhere. Auth values
    are scrubbed at the serializer boundary — literal secrets are
    blanked before they reach disk; ``${VAR}`` indirections survive.

    The ``[connections]`` table is dropped when the resulting dict
    is empty so a notebook with no connections doesn't carry a
    stub block.
    """
    new_connections = _serialize_connections(connections, malformed)

    def mutate(toml_data: dict[str, Any]) -> bool:
        existing = toml_data.get("connections")
        if not new_connections:
            # Treat "no [connections] block" and "empty dict" as the same
            # state so an empty save on a notebook that never had
            # connections doesn't rewrite the file (and re-serialize
            # array-of-tables to inline, churning updated_at).
            if not existing:
                return False
            toml_data.pop("connections", None)
            return True
        if existing == new_connections:
            return False
        toml_data["connections"] = new_connections
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_worker(notebook_dir: Path, worker: str | None) -> None:
    """Persist the notebook-level default worker."""

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("worker") == worker:
            return False
        if worker is None:
            toml_data.pop("worker", None)
        else:
            toml_data["worker"] = worker
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_workers(notebook_dir: Path, workers: list[WorkerSpec]) -> None:
    """Persist notebook-scoped worker definitions."""
    new_workers = _serialize_workers(workers)

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("workers", []) == new_workers:
            return False
        toml_data["workers"] = new_workers
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_timeout(notebook_dir: Path, timeout: float | None) -> None:
    """Persist the notebook-level default timeout."""

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("timeout") == timeout:
            return False
        if timeout is None:
            toml_data.pop("timeout", None)
        else:
            toml_data["timeout"] = timeout
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_env(notebook_dir: Path, env: dict[str, str]) -> None:
    """Persist notebook-level default environment variables.

    The persistable env block drops any entry that has no meaningful
    content (empty values and blanked sensitive keys). If that block
    is byte-identical to what's already on disk, the call is a no-op —
    we skip the rewrite so ``updated_at`` keeps tracking genuine
    structural edits. Typing an API key in the Runtime panel therefore
    doesn't churn the committed notebook.toml.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    new_env: dict[str, str] | None = (
        _serialize_env(env) if env and _env_has_meaningful_content(env) else None
    )
    existing_env = toml_data.get("env")

    # Compare the effective persistable state. ``None`` means "block
    # should not appear"; equality on dicts handles the value changes.
    if new_env == existing_env or (new_env is None and not existing_env):
        return

    if new_env is None:
        toml_data.pop("env", None)
    else:
        toml_data["env"] = new_env
    toml_data["updated_at"] = datetime.now(tz=UTC).isoformat()

    with open(notebook_toml_path, "wb") as f:
        tomli_w.dump(toml_data, f)


def update_cell_display_outputs(
    notebook_dir: Path,
    cell_id: str,
    display_outputs: list[dict[str, object]] | None,
) -> None:
    """Persist or clear ordered display output metadata for a cell.

    Stored in ``.strata/runtime.json`` — display outputs change every
    execution, so they'd churn ``notebook.toml`` under Git if kept
    there. The same file also holds per-cell provenance hashes and
    the last ``uv sync`` timestamp; see ``runtime_state.py``.
    """
    from strata.notebook.runtime_state import load_runtime_state, save_runtime_state

    notebook_dir = Path(notebook_dir)
    state = load_runtime_state(notebook_dir)
    entry = state.get_or_create_cell(cell_id)

    persisted_displays = _sanitize_display_outputs_for_toml(display_outputs)
    if persisted_displays:
        entry.display_outputs = persisted_displays
        entry.display = persisted_displays[-1]
    else:
        entry.display_outputs = []
        entry.display = None

    save_runtime_state(notebook_dir, state)


_SECRET_MANAGER_CONFIG_KEYS = ("provider", "project_id", "environment", "path", "base_url")


def set_variant_active(
    notebook_dir: Path,
    group: str,
    variant_name: str,
) -> None:
    """Set the active variant for ``group`` in notebook.toml.

    Updates the matching ``[[variant_group]]`` entry, or appends one if
    none exists. Bumps ``updated_at`` (variant selection is a structural
    change — it changes which cells participate in the executable graph).

    Group membership itself is declared by ``# @variant`` annotations in
    cell source; this function only persists the active-variant pointer.
    Validation that ``variant_name`` actually matches a member of the
    group is left to ``annotation_validation`` (``variant_active_unknown``
    surfaces the drift if the user picks a name no cell provides).
    """

    def mutate(toml_data: dict[str, Any]) -> bool:
        entries = toml_data.get("variant_group", [])
        if not isinstance(entries, list):
            entries = []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("group") == group:
                if entry.get("active") == variant_name:
                    return False
                entry["active"] = variant_name
                toml_data["variant_group"] = entries
                return True
        entries.append({"group": group, "active": variant_name})
        toml_data["variant_group"] = entries
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def remove_variant_group_entry(notebook_dir: Path, group: str) -> None:
    """Drop the ``[[variant_group]]`` entry for ``group``, if any.

    Used when the last member of a variant group is removed — there's
    no group left to point at, so the toml entry is no longer meaningful.
    """

    def mutate(toml_data: dict[str, Any]) -> bool:
        entries = toml_data.get("variant_group")
        if not isinstance(entries, list):
            return False
        new_entries = [
            entry
            for entry in entries
            if not (isinstance(entry, dict) and entry.get("group") == group)
        ]
        if len(new_entries) == len(entries):
            return False
        if new_entries:
            toml_data["variant_group"] = new_entries
        else:
            toml_data.pop("variant_group", None)
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_secret_manager(notebook_dir: Path, config: dict[str, Any]) -> None:
    """Persist the [secret_manager] block in notebook.toml.

    Accepts only a fixed whitelist of keys so arbitrary runtime state
    can't leak into the committed TOML via this path. Passing an empty
    dict removes the block entirely — the UI "disconnect" action.
    """
    cleaned: dict[str, Any] = {}
    for key in _SECRET_MANAGER_CONFIG_KEYS:
        value = config.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value:
                continue
        cleaned[key] = value

    def mutate(toml_data: dict[str, Any]) -> bool:
        existing = toml_data.get("secret_manager")
        existing_dict = existing if isinstance(existing, dict) else None
        if not cleaned:
            if not existing_dict:
                return False
            toml_data.pop("secret_manager", None)
            return True
        if existing_dict == cleaned:
            return False
        toml_data["secret_manager"] = cleaned
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_notebook_ai_model(notebook_dir: Path, model: str) -> None:
    """Update the notebook's default LLM model in [ai] section."""

    def mutate(toml_data: dict[str, Any]) -> bool:
        ai = toml_data.get("ai", {})
        if not isinstance(ai, dict):
            ai = {}
        if ai.get("model") == model:
            return False
        ai["model"] = model
        toml_data["ai"] = ai
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


def update_cell_console_output(
    notebook_dir: Path,
    cell_id: str,
    stdout: str,
    stderr: str,
) -> None:
    """Persist stdout/stderr for a cell so they survive notebook reopens.

    Written to ``.strata/console/{cell_id}.json`` — separate from
    notebook.toml to keep configuration and runtime state apart.
    Truncated to 10 KB per stream.
    """
    max_len = 10_000
    console_dir = Path(notebook_dir) / ".strata" / "console"
    console_dir.mkdir(parents=True, exist_ok=True)

    console_file = console_dir / f"{cell_id}.json"
    if stdout or stderr:
        import json

        with open(console_file, "w", encoding="utf-8") as f:
            json.dump(
                {"stdout": stdout[:max_len], "stderr": stderr[:max_len]},
                f,
            )
    elif console_file.exists():
        console_file.unlink()


def load_cell_console_output(notebook_dir: Path, cell_id: str) -> tuple[str, str]:
    """Load persisted stdout/stderr for a cell.

    Returns ``(stdout, stderr)`` — empty strings if nothing is persisted.
    """
    console_file = Path(notebook_dir) / ".strata" / "console" / f"{cell_id}.json"
    if not console_file.exists():
        return "", ""
    try:
        import json

        with open(console_file, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("stdout", ""), data.get("stderr", "")
    except Exception:
        return "", ""


def update_cell_display_output(
    notebook_dir: Path,
    cell_id: str,
    display_output: dict[str, object] | None,
) -> None:
    """Backward-compatible wrapper for persisting a single display output."""
    update_cell_display_outputs(
        notebook_dir,
        cell_id,
        [display_output] if display_output is not None else None,
    )

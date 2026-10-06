"""Write notebook changes back to disk (notebook.toml and cell files)."""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO

import filelock
import tomli_w
from packaging.requirements import Requirement

from strata.notebook.dependencies import rscript_env, uv_env
from strata.notebook.harness_user import (
    HarnessUser,
    LocalExecutionRefused,
    hand_over,
    resolve_harness_user,
    spawn_kwargs,
)
from strata.notebook.layout import write_gitignore
from strata.notebook.models import (
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
from strata.notebook.quiesce import refuses_while_held

if TYPE_CHECKING:
    pass


def _serialize_mounts(mounts: list[MountSpec]) -> list[dict[str, Any]]:
    """Convert mount specs into TOML-friendly dicts, omitting empty ``options``."""
    result: list[dict[str, Any]] = []
    for mount in mounts:
        data = mount.model_dump(mode="json", exclude_none=True)
        if not data.get("options"):
            data.pop("options", None)
        result.append(data)
    return result


# Rendered as ``[[name]]`` blocks: tomli_w always picks the inline
# ``name = [{...}]`` form for simple dicts, which makes git diffs messy.
_ARRAY_OF_TABLES_SECTIONS = ("workers", "mounts", "variant_group")


def _dump_notebook_toml(data: dict[str, Any], fp: Any) -> None:
    """Serialize a notebook.toml dict, forcing array-of-tables sections.

    ``[[workers]]``, ``[[mounts]]`` and ``[[variant_group]]`` are written as
    hand-formatted blocks after the ``tomli_w`` output, matching the committed
    example notebooks.
    """
    aot: dict[str, list[dict[str, Any]]] = {}
    for key in _ARRAY_OF_TABLES_SECTIONS:
        value = data.get(key)
        if isinstance(value, list) and value:
            aot[key] = value
            data = {k: v for k, v in data.items() if k != key}

    tomli_w.dump(data, fp)

    for name, items in aot.items():
        for item in items:
            simple = {k: v for k, v in item.items() if not isinstance(v, dict)}
            nested = {k: v for k, v in item.items() if isinstance(v, dict)}
            fp.write(f"\n[[{name}]]\n".encode())
            if simple:
                tomli_w.dump(simple, fp)
            for nested_key, nested_value in nested.items():
                fp.write(f"\n[{name}.{nested_key}]\n".encode())
                tomli_w.dump(nested_value, fp)


def _replace_file_atomically(path: Path, write: Callable[[BinaryIO], Any]) -> None:
    """Write *path* through *write* into a temp sibling, fsync it, then ``os.replace`` it.

    A crash or a full disk mid-write leaves the previous file intact instead of
    truncated.
    """
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp_path, "xb") as f:
            # Keep a mode the user narrowed (e.g. a notebook.toml holding an [ai] api_key).
            with contextlib.suppress(FileNotFoundError):
                os.chmod(tmp_path, path.stat().st_mode & 0o7777)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with UTF-8 *text*."""
    _replace_file_atomically(path, lambda f: f.write(text.encode("utf-8")))


def _write_notebook_toml_atomic(notebook_toml_path: Path, toml_data: dict[str, Any]) -> None:
    """Serialize *toml_data* and atomically replace ``notebook.toml``.

    A truncated file would lose the cell list and orphan artifacts.
    """
    _replace_file_atomically(notebook_toml_path, lambda f: _dump_notebook_toml(toml_data, f))


_SENSITIVE_KEY_PATTERNS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL")


def _is_sensitive_env_key(key: str) -> bool:
    """Return True if the env var name looks like a secret."""
    upper = key.upper()
    return any(pattern in upper for pattern in _SENSITIVE_KEY_PATTERNS)


def _serialize_env(env: dict[str, str]) -> dict[str, str]:
    """Convert env vars into a TOML-friendly dict, blanking sensitive values.

    Key names are kept so the notebook remembers which vars are configured;
    the user re-enters the values.
    """
    return {
        key: ("" if _is_sensitive_env_key(key) else value) for key, value in sorted(env.items())
    }


def _env_has_meaningful_content(env: dict[str, str]) -> bool:
    """True if the env dict has any non-empty, non-sensitive value.

    Lets the writer skip an ``[env]`` block that holds only empty or blanked entries.
    """
    for key, value in env.items():
        if not value:
            continue
        if _is_sensitive_env_key(key):
            continue
        return True
    return False


def drop_blanked_secrets(env: Mapping[str, str]) -> dict[str, str]:
    """Drop empty sensitive entries so they do not mask the server's own value.

    The writer blanks sensitive values on disk, so an empty one is a placeholder
    for re-entry, not an intentional override. Empty non-sensitive values stay.
    """
    return {key: value for key, value in env.items() if value or not _is_sensitive_env_key(key)}


_AUTH_INDIRECTION_RE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")


def is_auth_indirection(value: object) -> bool:
    """Return True for ``${VAR}`` env-var indirections.

    Shared by the writer and annotation validation. Deliberately narrow: empty
    braces, lowercase-only names and bare ``$VAR`` are rejected.
    """
    return isinstance(value, str) and bool(_AUTH_INDIRECTION_RE.match(value))


def _scrub_auth_for_disk(auth: dict[str, Any]) -> dict[str, Any]:
    """Blank literal auth values before writing; only ``${VAR}`` indirections pass.

    Key names are kept, like ``_serialize_env``. Non-string values become ``""``.
    """
    return {k: (v if is_auth_indirection(v) else "") for k, v in auth.items()}


def _serialize_connections(
    connections: list[ConnectionSpec],
    malformed: list[MalformedConnection] | None = None,
) -> dict[str, dict[str, Any]]:
    """Convert connection specs into the ``[connections.<name>]`` TOML shape.

    Malformed blocks round-trip too, so an unrelated save keeps a hand-edited
    typo. ``auth`` values other than ``${VAR}`` indirections are blanked.
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
        if body.get("credential") is None:
            body.pop("credential", None)
        out[conn.name] = body

    for mal in malformed or []:
        if mal.name in out:
            # The parser only emits a malformed record when the valid path errors; guard
            # against a double write anyway (valid wins).
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
            # Drop unset keys, and omit the block when nothing is set (an empty model is truthy).
            **(
                {"config": cfg}
                if (cfg := worker.config.model_dump(mode="json", exclude_none=True))
                else {}
            ),
        }
        for worker in workers
    ]


def _sanitize_display_output_for_toml(
    display_output: dict[str, object] | None,
) -> dict[str, object] | None:
    """Strip transient fields before persisting cell display metadata.

    ``to_serialization_safe`` is the single TOML/JSON compatibility boundary. The
    target is runtime.json, so a ``None`` inside a value stays null.
    """
    from strata.notebook.serializer import to_serialization_safe

    if display_output is None:
        return None

    persisted = dict(display_output)
    persisted.pop("inline_data_url", None)
    persisted.pop("file", None)
    persisted.pop("markdown_text", None)
    cleaned = {key: value for key, value in persisted.items() if value is not None}
    return to_serialization_safe(cleaned, keep_none=True)


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


@refuses_while_held
def write_cell(notebook_dir: Path, cell_id: str, source: str, author: str | None = None) -> None:
    """Write cell source to disk.

    ``author`` is recorded on the cell only when it differs from the current
    one, so a person editing their own cell never rewrites ``notebook.toml``
    and debounced flushes do not produce commit-worthy diffs.

    Raises:
        ValueError: If the cell is not in notebook.toml.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    cells_data = toml_data.get("cells", [])
    cell_meta = None
    for cell in cells_data:
        if cell.get("id") == cell_id:
            cell_meta = cell
            break

    if cell_meta is None:
        # FileNotFoundError, not ValueError, so handlers can map it to 404.
        raise FileNotFoundError(f"Cell {cell_id} not found in notebook.toml")

    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)
    cell_file = cells_dir / cell_meta["file"]
    cell_file.parent.mkdir(parents=True, exist_ok=True)

    _write_text_atomic(cell_file, source)

    if author and cell_meta.get("updated_by") != author:
        # Through the helper, which re-reads: rewriting the snapshot loaded above would
        # drop a structural edit that landed meanwhile (an offline `strata cell add`, a
        # reorder) and orphan that cell's source. No `updated_at` bump: an author change
        # is not a structural edit.
        def _stamp(data: dict[str, Any]) -> bool:
            for entry in data.get("cells", []):
                if entry.get("id") == cell_id and entry.get("updated_by") != author:
                    entry["updated_by"] = author
                    return True
            return False

        _apply_notebook_toml_update(notebook_dir, _stamp, bump_updated_at=False)


@refuses_while_held
def write_cell_tests(notebook_dir: Path, cell_id: str, test_source: str) -> None:
    """Write (or clear) a cell's unit-test source in ``cells/{cell_id}.test.py``.

    Empty or whitespace source removes the file instead.

    Raises:
        FileNotFoundError: If the cell is not in notebook.toml.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    if not any(cell.get("id") == cell_id for cell in toml_data.get("cells", [])):
        # FileNotFoundError, not ValueError, so handlers can map it to 404.
        raise FileNotFoundError(f"Cell {cell_id} not found in notebook.toml")

    cells_dir = notebook_dir / "cells"
    # Validated above, but basename so a path-separator id can never escape ``cells/``.
    test_file = cells_dir / os.path.basename(f"{cell_id}.test.py")

    if test_source.strip():
        cells_dir.mkdir(exist_ok=True)
        _write_text_atomic(test_file, test_source)
    elif test_file.exists():
        test_file.unlink()


@refuses_while_held
def write_notebook_toml(notebook_dir: Path, toml: NotebookToml) -> None:
    """Write notebook.toml to disk.

    It holds stable configuration only; per-execution state lives in
    ``.strata/runtime.json`` (``runtime_state.py``). Runtime-only callers must
    not touch it, so ``updated_at`` stays a structural-change signal.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    toml_data = {
        "notebook_id": toml.notebook_id,
        "name": toml.name,
        "created_at": toml.created_at,
        "updated_at": toml.updated_at,
        "cells": [
            {
                "id": cell.id,
                "file": cell.file,
                "language": cell.language,
                "order": cell.order,
                **({"worker": cell.worker} if cell.worker is not None else {}),
                **({"timeout": cell.timeout} if cell.timeout is not None else {}),
                # Rebuilt field by field, so anything omitted here is erased on round-trip.
                **({"created_by": cell.created_by} if cell.created_by else {}),
                **({"updated_by": cell.updated_by} if cell.updated_by else {}),
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
                    {
                        "group": vg.group,
                        "active": vg.active,
                        # Only when non-default, so switch-mode notebooks don't churn.
                        **({"mode": vg.mode} if vg.mode != "switch" else {}),
                    }
                    for vg in toml.variant_groups
                ]
            }
            if toml.variant_groups
            else {}
        ),
        **({"ai": toml.ai} if toml.ai else {}),
        **({"catalogs": toml.catalogs} if toml.catalogs else {}),
        **({"secret_manager": toml.secret_manager} if toml.secret_manager else {}),
        **({"r": toml.r} if toml.r else {}),
        # Runtime state (display outputs, sync timestamps, cache) lives in
        # ``.strata/runtime.json`` so running a notebook doesn't churn git diffs.
    }

    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


# Imported directly by harness/pool_worker/serializer inside the notebook venv.
# Specifiers come from strata-notebook's own Requires-Dist so bounds never drift;
# Requirement.parse handles both core deps and the [notebook] extra.
_NOTEBOOK_RUNTIME_PACKAGES: tuple[str, ...] = ("pyarrow", "orjson", "cloudpickle")


def _notebook_runtime_specifiers() -> list[str]:
    """PEP 508 specifiers for the notebook venv's runtime deps, from installed metadata."""
    try:
        reqs = metadata("strata-notebook").get_all("Requires-Dist") or []
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "strata-notebook distribution metadata not found — install via "
            "`uv sync` so importlib.metadata can supply notebook venv "
            "dependency specifiers"
        ) from exc

    found: dict[str, str] = {}
    for raw in reqs:
        req = Requirement(raw)
        if req.name in _NOTEBOOK_RUNTIME_PACKAGES and req.name not in found:
            found[req.name] = f"{req.name}{req.specifier}"

    missing = [name for name in _NOTEBOOK_RUNTIME_PACKAGES if name not in found]
    if missing:
        raise RuntimeError(
            f"strata-notebook metadata missing required notebook runtime "
            f"deps {missing} — they must stay in pyproject.toml (core deps "
            f"or [project.optional-dependencies].notebook) for the venv "
            f"template to find them"
        )
    return [found[name] for name in _NOTEBOOK_RUNTIME_PACKAGES]


def create_notebook(
    parent_dir: Path,
    name: str,
    python_version: str | None = None,
    *,
    initialize_environment: bool = True,
    write_gitignore_file: bool = True,
    project_mount: str | None = None,
) -> Path:
    """Create a new notebook directory with notebook.toml and pyproject.toml.

    On an existing notebook only missing scaffolding (``cells/``,
    ``.gitignore``) is added; ``project_mount`` does not apply.

    Args:
        write_gitignore_file: Write a runtime-state .gitignore; an existing one
            is never replaced.
        project_mount: Variable name for a pinned read-only mount of
            *parent_dir*, so cells read project files without absolute paths.
            Pinned, so it never makes cells stale.

    Returns:
        Path to the created notebook directory.
    """
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True, exist_ok=True)
    requested_python_version = (
        normalize_python_minor(python_version)
        if python_version is not None
        else current_python_minor()
    )

    if "/" in name or "\\" in name or ".." in name or "\0" in name:
        raise ValueError("Notebook name contains invalid characters")

    notebook_dir = parent_dir / name.lower().replace(" ", "_")
    notebook_dir.mkdir(exist_ok=True)

    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    # An existing notebook is left as it is: rewriting its notebook.toml and
    # pyproject.toml from what this function knows would drop its settings and
    # dependencies. Only missing scaffolding is added.
    if (notebook_dir / "notebook.toml").exists():
        if write_gitignore_file:
            write_gitignore(notebook_dir)
        return notebook_dir

    notebook_id = str(uuid.uuid4())

    # Pinned read-only mount of the project dir, so cells use `open(<name> / "file")`.
    # Pinned means no hashing: project churn (including .strata/) never re-stales cells.
    mounts: list[MountSpec] = []
    if project_mount is not None:
        if not project_mount.isidentifier():
            raise ValueError(f"--project-mount name {project_mount!r} must be a valid identifier")
        mounts.append(MountSpec(name=project_mount, uri=f"file://{parent_dir}", pin="project-root"))

    now = datetime.now(tz=UTC)
    notebook_toml = NotebookToml(
        notebook_id=notebook_id,
        name=name,
        created_at=now,
        updated_at=now,
        cells=[],
        mounts=mounts,
    )
    write_notebook_toml(notebook_dir, notebook_toml)

    # pyarrow, orjson and cloudpickle are baked in to avoid silent fallbacks to
    # slower stdlib json and pickle.
    pyproject_data: dict[str, Any] = {
        "project": {
            "name": name.lower().replace(" ", "-"),
            "version": "0.1.0",
            "description": "",
            "requires-python": format_requires_python(requested_python_version),
            "dependencies": _notebook_runtime_specifiers(),
        },
        "tool": {"uv": {}},
    }

    with open(notebook_dir / "pyproject.toml", "wb") as f:
        tomli_w.dump(pyproject_data, f)

    # Before any sync: a directory that gains .venv before its ignore rule can be
    # `git add -A`'d at exactly the wrong moment.
    if write_gitignore_file:
        write_gitignore(notebook_dir)

    if initialize_environment:
        # Best-effort; creates venv + uv.lock
        synced = _uv_sync(notebook_dir, python_version=requested_python_version)

        _update_environment_metadata(notebook_dir)

        # Records the lockfile only if it was actually installed: a failed best-effort
        # sync must not leave the notebook claiming an environment it lacks.
        if synced:
            from strata.notebook.env import compute_lockfile_hash
            from strata.notebook.runtime_state import (
                persist_environment_synced_lockfile_hash,
            )

            persist_environment_synced_lockfile_hash(
                notebook_dir, compute_lockfile_hash(notebook_dir)
            )

    return notebook_dir


_logger = logging.getLogger(__name__)


def _uv_sync(notebook_dir: Path, *, timeout: int = 60, python_version: str | None = None) -> bool:
    """Run ``uv sync`` in *notebook_dir*; False on failure (logged, never raised)."""
    command = ["uv", "sync"]
    if python_version is not None:
        command.extend(["--python", normalize_python_minor(python_version)])
    try:
        subprocess.run(
            command,
            cwd=str(notebook_dir),
            env=uv_env(),
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


def _renv_sync(notebook_dir: Path, *, timeout: int = 600) -> bool:
    """Run ``renv::restore()`` in *notebook_dir* so its library matches ``renv.lock``.

    Returns False on failure (logged, never raised), so a missing R install or
    stale lockfile does not crash the caller. The 10-minute default (vs uv's 60s)
    covers compiling CRAN sources on platforms without binaries.
    """
    # Before the Rscript lookup: R cells with no ``renv.lock`` yet is the normal
    # pre-init state and must succeed whether or not R is installed.
    if not (notebook_dir / "renv.lock").exists():
        _logger.debug("renv.lock missing in %s — nothing to restore", notebook_dir)
        return True

    # Building a package from source runs its configure script, so the restore runs
    # as the harness user; a service-mode server with no harness user skips it.
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        _logger.warning(
            "renv restore skipped in %s: it builds R packages from source, which runs "
            "their code. %s",
            notebook_dir,
            exc,
        )
        return False

    # renv has no locking of its own, so a server and a ``strata run`` in another
    # process must not restore concurrently. After the other finishes, ours is a
    # fast no-op. Acquired before the Rscript lookup so the contended path is
    # deterministic.
    from strata.notebook.dependencies import renv_process_lock

    process_lock = renv_process_lock(notebook_dir)
    try:
        process_lock.acquire(timeout=timeout)
    except filelock.Timeout:
        _logger.warning(
            "renv restore skipped in %s — another process held the renv lock for over %ds",
            notebook_dir,
            timeout,
        )
        return False

    try:
        from strata.notebook.env_backend import shared_root

        root = shared_root()
        if root is not None:
            # One library per renv.lock, restored into the shared store once.
            from strata.notebook.shared_env import restore_r_library

            return restore_r_library(
                notebook_dir,
                root,
                lambda env: _renv_restore_locked(
                    notebook_dir, timeout=timeout, env=env, harness_user=harness_user
                ),
            )
        return _renv_restore_locked(notebook_dir, timeout=timeout, harness_user=harness_user)
    finally:
        process_lock.release()


def _renv_restore_locked(
    notebook_dir: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    harness_user: HarnessUser | None = None,
) -> bool:
    """Run ``renv::restore()`` with the cross-process lock already held.

    Runs as *harness_user* when set, who is given what the restore writes: the
    notebook's ``renv/``, the shared library it links to, and the package
    cache *env* names.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        _logger.warning("Rscript not found on PATH — skipping renv restore")
        return False

    if harness_user is not None:
        renv_dir = notebook_dir / "renv"
        renv_dir.mkdir(exist_ok=True)
        library = renv_dir / "library"
        cache = (env or {}).get("RENV_PATHS_CACHE")
        if library.is_symlink():
            # Checked, not trusted: the harness user owns renv/ after a restore and can
            # repoint this link. Only a library in the shared store is ours to hand over.
            target = library.resolve()
            store = Path(cache).resolve().parent if cache else None
            if store is None or target.parent != store:
                _logger.warning(
                    "renv restore skipped in %s: renv/library points outside the "
                    "shared R library store (%s)",
                    notebook_dir,
                    target,
                )
                return False
            hand_over(target, harness_user)
        hand_over(renv_dir, harness_user)
        if cache:
            Path(cache).mkdir(parents=True, exist_ok=True)
            hand_over(Path(cache), harness_user)

    try:
        # No ``--vanilla``: the project ``.Rprofile`` sources ``renv/activate.R``, without
        # which ``renv::restore()`` targets the user's default lib (usually without renv).
        subprocess.run(
            [rscript, "-e", "renv::restore(prompt = FALSE)"],
            cwd=str(notebook_dir),
            timeout=timeout,
            capture_output=True,
            check=True,
            env=rscript_env(harness_user, env),
            **spawn_kwargs(harness_user),
        )
        _logger.debug("renv::restore() succeeded in %s", notebook_dir)
        return True
    except subprocess.TimeoutExpired:
        _logger.warning("renv::restore() timed out after %ds in %s", timeout, notebook_dir)
    except subprocess.CalledProcessError as exc:
        _logger.warning(
            "renv::restore() failed in %s: %s",
            notebook_dir,
            exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)",
        )
    return False


def _update_environment_metadata(notebook_dir: Path) -> None:
    """Persist the environment snapshot in ``.strata/runtime.json`` under ``environment``.

    Lockfile hash, python version, package counts and the last ``uv sync`` time,
    so clients can detect environment changes without hashing. Not in
    ``notebook.toml``, since it changes on every sync.
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
    # Everything below is rewritten from what is declared on disk. The attestation
    # records what was *installed*, so it must survive a refresh or every sync
    # would silently revoke it.
    realized = state.environment.synced_lockfile_hash
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
        synced_lockfile_hash=realized,
    )
    save_runtime_state(notebook_dir, state)


@refuses_while_held
def update_environment_metadata(notebook_dir: Path) -> None:
    """Refresh the environment snapshot in ``.strata/runtime.json`` after ``uv add``/``remove``."""
    _update_environment_metadata(notebook_dir)


@refuses_while_held
def add_cell_to_notebook(
    notebook_dir: Path,
    cell_id: str,
    after_cell_id: str | None = None,
    language: str = "python",
    author: str | None = None,
) -> None:
    """Add a new cell to the notebook.

    Args:
        after_cell_id: Cell to insert after; None appends.
        author: Who is adding it; None records nobody.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    # Midpoint to the next cell: ``X.order + 0.5`` ties when two inserts target the
    # same parent, and stable sort then puts the second after the first insert.
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

    # Language-matching extension so the file works in outside editors. The harness
    # reads source by content, so everything else (SQL included) uses ``.py``.
    extension_by_language = {"markdown": "md", "r": "r", "widget": "widget"}
    extension = extension_by_language.get(language, "py")
    cell_filename = f"{cell_id}.{extension}"
    cells_dir = notebook_dir / "cells"
    cells_dir.mkdir(exist_ok=True)

    # Widget cells start with a usable starter control; every other language starts empty.
    starter_source = (
        "alpha = slider(0, 1, step=0.01, default=0.5)\n" if language == "widget" else ""
    )
    with open(cells_dir / cell_filename, "w", encoding="utf-8") as f:
        f.write(starter_source)

    entry = {
        "id": cell_id,
        "file": cell_filename,
        "language": language,
        "order": order,
    }
    if author:
        # Both: an added, unedited cell was last changed by whoever added it.
        entry["created_by"] = author
        entry["updated_by"] = author
    cells_data.append(entry)

    cells_data.sort(key=lambda c: c.get("order", 0))

    toml_data["cells"] = cells_data
    toml_data["updated_at"] = datetime.now(tz=UTC)

    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


@refuses_while_held
def remove_cell_from_notebook(notebook_dir: Path, cell_id: str) -> None:
    """Remove a cell from the notebook.

    Raises:
        ValueError: If the cell is not found.
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
        # FileNotFoundError, not ValueError, so handlers can map it to 404.
        raise FileNotFoundError(f"Cell {cell_id} not found")

    cells_dir = notebook_dir / "cells"
    cell_file = cells_dir / cell_meta["file"]
    if cell_file.exists():
        cell_file.unlink()
    # Otherwise it stays in the committed tree and in every export.
    (cells_dir / os.path.basename(f"{cell_id}.test.py")).unlink(missing_ok=True)

    cells_data.pop(cell_idx)
    toml_data["cells"] = cells_data
    toml_data["updated_at"] = datetime.now(tz=UTC)

    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


@refuses_while_held
def reorder_cells(notebook_dir: Path, cell_ids: list[str]) -> None:
    """Reorder cells in the notebook to match *cell_ids*."""
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    cells_data = toml_data.get("cells", [])

    cell_map = {cell.get("id"): cell for cell in cells_data}

    new_cells = []
    for cell_id in cell_ids:
        if cell_id in cell_map:
            new_cells.append(cell_map[cell_id])

    # Keep every on-disk cell absent from ``cell_ids``: callers pass a snapshot from
    # open time, so cells added since (by a server session, the TUI, another CLI)
    # would otherwise be deleted from committed config. Unknown cells keep their
    # relative order after the explicitly ordered ones.
    named = set(cell_ids)
    new_cells.extend(cell for cell in cells_data if cell.get("id") not in named)

    for i, cell in enumerate(new_cells):
        cell["order"] = i

    toml_data["cells"] = new_cells
    toml_data["updated_at"] = datetime.now(tz=UTC)

    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


@refuses_while_held
def update_requires_python(notebook_dir: Path, new_minor: str) -> str:
    """Rewrite ``requires-python`` in pyproject.toml as ``==X.Y.*``; return the previous value.

    The previous value lets the caller roll back if the next ``uv sync`` fails.
    uv.lock and .venv are not touched.
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
    old_spec_match = re.search(r'"([^"]*)"', old_line)
    old_spec = old_spec_match.group(1) if old_spec_match else ""

    if old_line == new_line:
        # The caller handles the no-op case; harmless if it gets here anyway.
        return old_spec

    updated = text[: match.start()] + new_line + text[match.end() :]
    _write_text_atomic(pyproject_path, updated)
    return old_spec


@refuses_while_held
def rename_notebook(notebook_dir: Path, new_name: str) -> None:
    """Rename the notebook."""
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


@refuses_while_held
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


@refuses_while_held
def _apply_notebook_toml_update(
    notebook_dir: Path,
    mutate: Callable[[dict[str, Any]], bool],
    *,
    bump_updated_at: bool = True,
) -> None:
    """Load ``notebook.toml``, apply ``mutate``, and rewrite only if it reports a change.

    ``mutate`` returns True iff it changed something; otherwise nothing is
    written and ``updated_at`` is not bumped. ``bump_updated_at=False`` is for
    committed but non-structural changes (who last edited a cell), since the
    discover list sorts by ``updated_at``.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    if not mutate(toml_data):
        return

    if bump_updated_at:
        toml_data["updated_at"] = datetime.now(tz=UTC)
    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


@refuses_while_held
def update_notebook_mounts(notebook_dir: Path, mounts: list[MountSpec]) -> None:
    """Persist notebook-level mount defaults."""
    new_mounts = _serialize_mounts(mounts)

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("mounts", []) == new_mounts:
            return False
        toml_data["mounts"] = new_mounts
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


@refuses_while_held
def update_notebook_connections(
    notebook_dir: Path,
    connections: list[ConnectionSpec],
    malformed: list[MalformedConnection] | None = None,
) -> None:
    """Persist notebook-level ``[connections.<name>]`` blocks.

    Malformed entries round-trip and literal auth secrets are blanked, as in
    the full writer. An empty ``[connections]`` table is dropped.
    """
    new_connections = _serialize_connections(connections, malformed)

    def mutate(toml_data: dict[str, Any]) -> bool:
        existing = toml_data.get("connections")
        if not new_connections:
            # No block and an empty dict are the same state, so an empty save doesn't
            # rewrite the file (churning array-of-tables and updated_at).
            if not existing:
                return False
            toml_data.pop("connections", None)
            return True
        if existing == new_connections:
            return False
        toml_data["connections"] = new_connections
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


@refuses_while_held
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


@refuses_while_held
def update_notebook_workers(notebook_dir: Path, workers: list[WorkerSpec]) -> None:
    """Persist notebook-scoped worker definitions."""
    new_workers = _serialize_workers(workers)

    def mutate(toml_data: dict[str, Any]) -> bool:
        if toml_data.get("workers", []) == new_workers:
            return False
        toml_data["workers"] = new_workers
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


@refuses_while_held
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


@refuses_while_held
def update_notebook_env(notebook_dir: Path, env: dict[str, str]) -> None:
    """Persist notebook-level default environment variables.

    Entries with no meaningful content are dropped, and the call is a no-op if
    the result matches disk, so typing an API key does not churn notebook.toml.
    """
    notebook_dir = Path(notebook_dir)
    notebook_toml_path = notebook_dir / "notebook.toml"

    with open(notebook_toml_path, "rb") as f:
        toml_data = tomllib.load(f)

    new_env: dict[str, str] | None = (
        _serialize_env(env) if env and _env_has_meaningful_content(env) else None
    )
    existing_env = toml_data.get("env")

    # ``None`` means the block should not appear.
    if new_env == existing_env or (new_env is None and not existing_env):
        return

    if new_env is None:
        toml_data.pop("env", None)
    else:
        toml_data["env"] = new_env
    toml_data["updated_at"] = datetime.now(tz=UTC)
    _write_notebook_toml_atomic(notebook_toml_path, toml_data)


@refuses_while_held
def update_cell_display_outputs(
    notebook_dir: Path,
    cell_id: str,
    display_outputs: list[dict[str, object]] | None,
) -> None:
    """Persist or clear a cell's ordered display output metadata in ``.strata/runtime.json``."""
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


@refuses_while_held
def set_variant_active(
    notebook_dir: Path,
    group: str,
    variant_name: str,
) -> None:
    """Set the active variant for ``group`` in notebook.toml; bumps ``updated_at``.

    Updates or appends the ``[[variant_group]]`` entry. Membership comes from
    ``# @variant`` annotations; an unknown name is reported by annotation
    validation, not checked here.
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


@refuses_while_held
def set_variant_mode(notebook_dir: Path, group: str, mode: str) -> None:
    """Set the execution ``mode`` (``switch`` or ``sweep``) for ``group``; bumps ``updated_at``.

    ``switch`` is the default and is written by removing the key; a new entry
    gets an empty ``active`` (first variant in source order).
    """

    def mutate(toml_data: dict[str, Any]) -> bool:
        entries = toml_data.get("variant_group", [])
        if not isinstance(entries, list):
            entries = []
        for entry in entries:
            if isinstance(entry, dict) and entry.get("group") == group:
                current = entry.get("mode", "switch")
                if current == mode:
                    return False
                if mode == "switch":
                    entry.pop("mode", None)
                else:
                    entry["mode"] = mode
                toml_data["variant_group"] = entries
                return True
        if mode == "switch":
            # Switch is the default; nothing to persist.
            return False
        entries.append({"group": group, "active": "", "mode": mode})
        toml_data["variant_group"] = entries
        return True

    _apply_notebook_toml_update(notebook_dir, mutate)


@refuses_while_held
def remove_variant_group_entry(notebook_dir: Path, group: str) -> None:
    """Drop the ``[[variant_group]]`` entry for ``group``, if any (its last member was removed)."""

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


@refuses_while_held
def update_notebook_secret_manager(notebook_dir: Path, config: dict[str, Any]) -> None:
    """Persist the ``[secret_manager]`` block in notebook.toml.

    Only whitelisted keys are written. An empty dict removes the block (the UI's
    "disconnect").
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


@refuses_while_held
def update_cell_console_output(
    notebook_dir: Path,
    cell_id: str,
    stdout: str,
    stderr: str,
) -> None:
    """Persist a cell's stdout/stderr to ``.strata/console/{cell_id}.json``.

    Truncated to 10,000 characters per stream; both empty removes the file.
    """
    max_len = 10_000
    console_dir = Path(notebook_dir) / ".strata" / "console"
    console_dir.mkdir(parents=True, exist_ok=True)

    console_file = console_dir / f"{cell_id}.json"
    if stdout or stderr:
        import json

        _write_text_atomic(
            console_file, json.dumps({"stdout": stdout[:max_len], "stderr": stderr[:max_len]})
        )
    elif console_file.exists():
        console_file.unlink()


def load_cell_console_output(notebook_dir: Path, cell_id: str) -> tuple[str, str]:
    """Load persisted ``(stdout, stderr)`` for a cell; empty strings if none."""
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


@refuses_while_held
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

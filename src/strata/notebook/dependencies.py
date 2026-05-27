"""Dependency management for notebooks.

Wraps ``uv add`` / ``uv remove`` to manage Python packages in a
notebook's virtual environment.  After every mutation the lockfile
is re-synced so that ``uv.lock`` and ``.venv/`` stay consistent.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import subprocess
import threading
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import tomli_w
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

logger = logging.getLogger(__name__)
_MAX_OPERATION_LOG_CHARS = 12_000

# Serialize concurrent uv add/remove per notebook directory.
# Without this, two concurrent ``uv add`` calls for the same notebook
# can corrupt pyproject.toml / uv.lock.
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_notebook_lock(notebook_dir: Path) -> threading.Lock:
    """Get or create a per-notebook lock for uv operations."""
    key = str(notebook_dir.resolve())
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DependencyInfo:
    """One declared or resolved notebook dependency.

    Attributes
    ----------
    name : str
        PEP 503 canonical project name.
    version : Version or None
        Concrete pinned version parsed as PEP 440 (only set for entries
        read from ``uv.lock``). Render with ``str(...)`` at serialization
        boundaries.
    specifier : SpecifierSet or None
        Declared version constraint (only set for entries read from
        ``pyproject.toml``). Stored as a parsed ``SpecifierSet`` so semantic
        equality works; render with ``str(...)`` at serialization boundaries.
    """

    name: str
    version: Version | None = None
    specifier: SpecifierSet | None = None


@dataclass
class EnvironmentOperationLog:
    """Structured command details for environment/package operations."""

    command: str
    duration_ms: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False


@dataclass
class DependencyChangeResult:
    """Result of adding or removing a dependency."""

    success: bool
    package: str
    action: str  # "add" | "remove"
    error: str | None = None
    lockfile_changed: bool = False
    dependencies: list[DependencyInfo] = field(default_factory=list)
    operation_log: EnvironmentOperationLog | None = None


@dataclass
class RequirementsImportResult:
    """Result of importing notebook dependencies from requirements text."""

    success: bool
    error: str | None = None
    lockfile_changed: bool = False
    dependencies: list[DependencyInfo] = field(default_factory=list)
    imported_count: int = 0
    warnings: list[str] = field(default_factory=list)
    operation_log: EnvironmentOperationLog | None = None


@dataclass
class RequirementsPreviewResult:
    """Preview of importing notebook dependencies from external text."""

    dependencies: list[DependencyInfo] = field(default_factory=list)
    normalized_requirements: list[str] = field(default_factory=list)
    imported_count: int = 0
    warnings: list[str] = field(default_factory=list)
    additions: list[DependencyInfo] = field(default_factory=list)
    removals: list[DependencyInfo] = field(default_factory=list)
    unchanged: list[DependencyInfo] = field(default_factory=list)


@dataclass
class _UvCommandResult:
    """Internal subprocess result wrapper for uv commands."""

    success: bool
    error: str | None
    operation_log: EnvironmentOperationLog


class _BoundedOutputBuffer:
    """Accumulate subprocess output without letting UI payloads grow unbounded."""

    def __init__(self) -> None:
        self._text = ""
        self.truncated = False

    def append(self, value: str) -> None:
        if not value:
            return
        if self.truncated:
            return
        remaining = _MAX_OPERATION_LOG_CHARS - len(self._text)
        if remaining <= 0:
            self.truncated = True
            return
        if len(value) <= remaining:
            self._text += value
            return
        self._text += value[:remaining]
        self.truncated = True

    @property
    def text(self) -> str:
        return self._text.strip()


def _normalize_output_text(value: str | bytes | None) -> str:
    """Normalize subprocess output into a safe UI string."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value.strip()


def _trim_output_for_ui(value: str | bytes | None) -> tuple[str, bool]:
    """Trim command output so REST payloads stay bounded."""
    text = _normalize_output_text(value)
    if len(text) <= _MAX_OPERATION_LOG_CHARS:
        return text, False
    return text[:_MAX_OPERATION_LOG_CHARS], True


def _format_command_for_ui(command: list[str]) -> str:
    """Render a subprocess command for UI/debugging."""
    return " ".join(shlex.quote(part) for part in command)


def _run_uv_command(
    notebook_dir: Path,
    args: list[str],
    *,
    timeout: int,
    display_name: str,
) -> _UvCommandResult:
    """Run a uv command and capture bounded UI logs."""
    command = ["uv", *args]
    started = time.perf_counter()
    formatted_command = _format_command_for_ui(command)

    try:
        completed = subprocess.run(
            command,
            cwd=str(notebook_dir),
            timeout=timeout,
            capture_output=True,
            check=True,
            text=True,
        )
        stdout, stdout_truncated = _trim_output_for_ui(completed.stdout)
        stderr, stderr_truncated = _trim_output_for_ui(completed.stderr)
        return _UvCommandResult(
            success=True,
            error=None,
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            ),
        )
    except FileNotFoundError:
        return _UvCommandResult(
            success=False,
            error="uv not found on PATH",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _trim_output_for_ui(exc.stdout)
        stderr, stderr_truncated = _trim_output_for_ui(exc.stderr)
        return _UvCommandResult(
            success=False,
            error=f"{display_name} timed out after {timeout}s",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            ),
        )
    except subprocess.CalledProcessError as exc:
        stdout, stdout_truncated = _trim_output_for_ui(exc.stdout)
        stderr, stderr_truncated = _trim_output_for_ui(exc.stderr)
        error_detail = stderr or stdout or f"{display_name} exited with status {exc.returncode}"
        return _UvCommandResult(
            success=False,
            error=f"{display_name} failed: {error_detail}",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            ),
        )


async def run_uv_command_streaming(
    notebook_dir: Path,
    args: list[str],
    *,
    timeout: int,
    display_name: str,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
) -> _UvCommandResult:
    """Run a uv command asynchronously and surface bounded live stdout/stderr."""
    command = ["uv", *args]
    started = time.perf_counter()
    formatted_command = _format_command_for_ui(command)
    stdout_buffer = _BoundedOutputBuffer()
    stderr_buffer = _BoundedOutputBuffer()

    async def _emit_update(stream_name: str, text: str, truncated: bool) -> None:
        if on_update is None:
            return
        maybe_awaitable = on_update(stream_name, text, truncated)
        if asyncio.iscoroutine(maybe_awaitable):
            await maybe_awaitable

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(notebook_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return _UvCommandResult(
            success=False,
            error="uv not found on PATH",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )

    async def _read_stream(
        stream: asyncio.StreamReader | None,
        name: str,
        buffer: _BoundedOutputBuffer,
    ) -> None:
        if stream is None:
            return
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            text = chunk.decode(errors="replace")
            buffer.append(text)
            await _emit_update(name, buffer.text, buffer.truncated)

    stdout_task = asyncio.create_task(_read_stream(process.stdout, "stdout", stdout_buffer))
    stderr_task = asyncio.create_task(_read_stream(process.stderr, "stderr", stderr_buffer))

    try:
        await asyncio.wait_for(
            asyncio.gather(stdout_task, stderr_task, process.wait()),
            timeout=timeout,
        )
    except TimeoutError:
        process.kill()
        await asyncio.gather(stdout_task, stderr_task, process.wait(), return_exceptions=True)
        return _UvCommandResult(
            success=False,
            error=f"{display_name} timed out after {timeout}s",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stdout=stdout_buffer.text,
                stderr=stderr_buffer.text,
                stdout_truncated=stdout_buffer.truncated,
                stderr_truncated=stderr_buffer.truncated,
            ),
        )

    stdout = stdout_buffer.text
    stderr = stderr_buffer.text
    operation_log = EnvironmentOperationLog(
        command=formatted_command,
        duration_ms=int((time.perf_counter() - started) * 1000),
        stdout=stdout,
        stderr=stderr,
        stdout_truncated=stdout_buffer.truncated,
        stderr_truncated=stderr_buffer.truncated,
    )

    if process.returncode == 0:
        return _UvCommandResult(
            success=True,
            error=None,
            operation_log=operation_log,
        )

    error_detail = stderr or stdout or f"{display_name} exited with status {process.returncode}"
    return _UvCommandResult(
        success=False,
        error=f"{display_name} failed: {error_detail}",
        operation_log=operation_log,
    )


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def list_dependencies(notebook_dir: Path) -> list[DependencyInfo]:
    """List current project dependencies from pyproject.toml.

    Parses the ``[project] dependencies`` array.  Does **not** shell out.
    """
    deps_list = _read_project_dependency_strings(notebook_dir)
    results: list[DependencyInfo] = []
    for dep_str in deps_list:
        name, specifier = _split_requirement(dep_str)
        results.append(DependencyInfo(name=name, specifier=specifier))

    return results


def list_resolved_dependencies(notebook_dir: Path) -> list[DependencyInfo]:
    """List resolved packages from ``uv.lock`` when present."""
    lockfile_path = notebook_dir / "uv.lock"
    if not lockfile_path.exists():
        return []

    try:
        with open(lockfile_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        logger.debug("Failed to parse uv.lock in %s", notebook_dir, exc_info=True)
        return []

    packages = data.get("package", [])
    if not isinstance(packages, list):
        return []

    resolved: list[DependencyInfo] = []
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = package.get("name")
        version_raw = package.get("version")
        if not isinstance(name, str):
            continue
        parsed_version: Version | None = None
        if version_raw is not None:
            try:
                parsed_version = Version(str(version_raw))
            except InvalidVersion:
                logger.debug(
                    "Skipping non-PEP 440 version %r for %s in %s",
                    version_raw,
                    name,
                    notebook_dir,
                )
        resolved.append(
            DependencyInfo(
                name=canonicalize_name(name),
                version=parsed_version,
                specifier=None,
            )
        )

    resolved.sort(key=lambda dep: dep.name)
    return resolved


@dataclass
class RPackageInfo:
    """One R package installed in the notebook's library.

    Lightweight parallel to ``DependencyInfo`` for the R side — the
    env panel renders both lists side by side. ``version`` is the
    string CRAN/renv stores (R doesn't use PEP 440); render as-is.
    """

    name: str
    version: str


def list_r_packages(notebook_dir: Path, *, timeout: int = 30) -> list[RPackageInfo]:
    """List R packages installed in the notebook's library.

    Spawns ``Rscript`` with ``cwd=notebook_dir`` so the project's
    ``.Rprofile`` (renv activator) puts the project library on
    ``.libPaths()`` before ``installed.packages()`` runs. With
    renv, that returns the per-notebook library; without renv it
    falls through to the system library — the latter is the
    pre-bootstrap state.

    Returns ``[]`` when ``Rscript`` isn't on PATH, the subprocess
    fails, or the output can't be parsed. Same "log + continue"
    contract the rest of the R surface uses; the env panel renders
    an empty list rather than erroring.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        return []

    # ``installed.packages()`` returns a character matrix; loop in
    # R rather than using ``apply`` because ``apply`` on a 1-row
    # matrix returns a vector and the parsing diverges. The for-loop
    # always emits ``<name>\t<version>\n`` per package, no
    # special-casing needed.
    r_snippet = (
        "ip <- installed.packages()\n"
        "for (i in seq_len(nrow(ip))) {\n"
        '  cat(ip[i, "Package"], ip[i, "Version"], sep = "\\t")\n'
        '  cat("\\n")\n'
        "}"
    )

    try:
        proc = subprocess.run(  # noqa: S603 — rscript resolved via shutil.which
            [rscript, "-e", r_snippet],
            cwd=str(notebook_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("R package listing failed: %s", exc)
        return []

    if proc.returncode != 0:
        logger.debug(
            "R package listing returned non-zero (%d): %s",
            proc.returncode,
            proc.stderr.strip()[:200],
        )
        return []

    packages: list[RPackageInfo] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split("\t")
        if len(parts) >= 2 and parts[0] and parts[1]:
            packages.append(RPackageInfo(name=parts[0], version=parts[1]))
    packages.sort(key=lambda pkg: pkg.name)
    return packages


def export_requirements_text(notebook_dir: Path) -> str:
    """Export direct notebook dependencies as ``requirements.txt`` text."""
    deps_list = _read_project_dependency_strings(notebook_dir)
    if not deps_list:
        return ""
    return "\n".join(deps_list) + "\n"


def preview_requirements_text(
    notebook_dir: Path,
    requirements_text: str,
) -> RequirementsPreviewResult:
    """Preview replacing direct notebook dependencies from requirements text."""
    normalized_requirements = parse_requirements_text(requirements_text)
    preview_dependencies = _dependency_info_from_requirement_strings(normalized_requirements)
    additions, removals, unchanged = _diff_dependency_sets(
        list_dependencies(notebook_dir),
        preview_dependencies,
    )
    return RequirementsPreviewResult(
        dependencies=preview_dependencies,
        normalized_requirements=normalized_requirements,
        imported_count=len(preview_dependencies),
        additions=additions,
        removals=removals,
        unchanged=unchanged,
    )


def import_requirements_text(
    notebook_dir: Path,
    requirements_text: str,
    *,
    timeout: int = 180,
) -> RequirementsImportResult:
    """Replace direct notebook dependencies from ``requirements.txt`` text."""
    normalized_requirements = parse_requirements_text(requirements_text)
    pyproject_path = notebook_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return RequirementsImportResult(
            success=False,
            error="pyproject.toml not found",
        )

    old_lockfile_hash = _lockfile_hash(notebook_dir)
    old_pyproject = pyproject_path.read_bytes()
    lockfile_path = notebook_dir / "uv.lock"
    old_lockfile = lockfile_path.read_bytes() if lockfile_path.exists() else None

    lock = _get_notebook_lock(notebook_dir)
    with lock:
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            return RequirementsImportResult(
                success=False,
                error=f"Failed to parse pyproject.toml: {exc}",
            )

        project = data.setdefault("project", {})
        if not isinstance(project, dict):
            return RequirementsImportResult(
                success=False,
                error="pyproject.toml project section is invalid",
            )
        project["dependencies"] = normalized_requirements

        try:
            with open(pyproject_path, "wb") as f:
                tomli_w.dump(data, f)
        except Exception as exc:
            return RequirementsImportResult(
                success=False,
                error=f"Failed to write pyproject.toml: {exc}",
            )

        command_result = _run_uv_command(
            notebook_dir,
            ["sync"],
            timeout=timeout,
            display_name="uv sync",
        )
        if command_result.success:
            logger.info(
                "Imported %s requirements into %s",
                len(normalized_requirements),
                notebook_dir,
            )
        else:
            _restore_dependency_files(pyproject_path, old_pyproject, lockfile_path, old_lockfile)
            return RequirementsImportResult(
                success=False,
                error=command_result.error,
                operation_log=command_result.operation_log,
            )

    new_lockfile_hash = _lockfile_hash(notebook_dir)
    return RequirementsImportResult(
        success=True,
        lockfile_changed=old_lockfile_hash != new_lockfile_hash,
        dependencies=list_dependencies(notebook_dir),
        imported_count=len(normalized_requirements),
        operation_log=command_result.operation_log,
    )


async def import_requirements_text_streaming(
    notebook_dir: Path,
    requirements_text: str,
    *,
    timeout: int = 180,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
) -> RequirementsImportResult:
    """Replace direct notebook dependencies from ``requirements.txt`` with live logs."""
    normalized_requirements = parse_requirements_text(requirements_text)
    pyproject_path = notebook_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return RequirementsImportResult(
            success=False,
            error="pyproject.toml not found",
        )

    old_lockfile_hash = _lockfile_hash(notebook_dir)
    old_pyproject = pyproject_path.read_bytes()
    lockfile_path = notebook_dir / "uv.lock"
    old_lockfile = lockfile_path.read_bytes() if lockfile_path.exists() else None

    lock = _get_notebook_lock(notebook_dir)
    await asyncio.to_thread(lock.acquire)
    try:
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
        except Exception as exc:
            return RequirementsImportResult(
                success=False,
                error=f"Failed to parse pyproject.toml: {exc}",
            )

        project = data.setdefault("project", {})
        if not isinstance(project, dict):
            return RequirementsImportResult(
                success=False,
                error="pyproject.toml project section is invalid",
            )
        project["dependencies"] = normalized_requirements

        try:
            with open(pyproject_path, "wb") as f:
                tomli_w.dump(data, f)
        except Exception as exc:
            _restore_dependency_files(pyproject_path, old_pyproject, lockfile_path, old_lockfile)
            return RequirementsImportResult(
                success=False,
                error=f"Failed to write pyproject.toml: {exc}",
            )

        command_result = await run_uv_command_streaming(
            notebook_dir,
            ["sync"],
            timeout=timeout,
            display_name="uv sync",
            on_update=on_update,
        )
        if command_result.success:
            logger.info(
                "Imported %s requirements into %s",
                len(normalized_requirements),
                notebook_dir,
            )
        else:
            _restore_dependency_files(pyproject_path, old_pyproject, lockfile_path, old_lockfile)
            return RequirementsImportResult(
                success=False,
                error=command_result.error,
                operation_log=command_result.operation_log,
            )
    finally:
        lock.release()

    new_lockfile_hash = _lockfile_hash(notebook_dir)
    return RequirementsImportResult(
        success=True,
        lockfile_changed=old_lockfile_hash != new_lockfile_hash,
        dependencies=list_dependencies(notebook_dir),
        imported_count=len(normalized_requirements),
        operation_log=command_result.operation_log,
    )


def import_environment_yaml_text(
    notebook_dir: Path,
    environment_yaml_text: str,
    *,
    timeout: int = 180,
) -> RequirementsImportResult:
    """Best-effort import of Conda-style ``environment.yaml`` into notebook deps."""
    requirements, warnings = parse_environment_yaml_text(environment_yaml_text)
    result = import_requirements_text(
        notebook_dir,
        "\n".join(requirements),
        timeout=timeout,
    )
    result.warnings = warnings
    return result


async def import_environment_yaml_text_streaming(
    notebook_dir: Path,
    environment_yaml_text: str,
    *,
    timeout: int = 180,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
) -> RequirementsImportResult:
    """Best-effort ``environment.yaml`` import with live ``uv sync`` output."""
    requirements, warnings = parse_environment_yaml_text(environment_yaml_text)
    result = await import_requirements_text_streaming(
        notebook_dir,
        "\n".join(requirements),
        timeout=timeout,
        on_update=on_update,
    )
    result.warnings = warnings
    return result


def preview_environment_yaml_text(
    notebook_dir: Path,
    environment_yaml_text: str,
) -> RequirementsPreviewResult:
    """Preview best-effort import of Conda-style ``environment.yaml`` text."""
    normalized_requirements, warnings = parse_environment_yaml_text(environment_yaml_text)
    preview_dependencies = _dependency_info_from_requirement_strings(normalized_requirements)
    additions, removals, unchanged = _diff_dependency_sets(
        list_dependencies(notebook_dir),
        preview_dependencies,
    )
    return RequirementsPreviewResult(
        dependencies=preview_dependencies,
        normalized_requirements=normalized_requirements,
        imported_count=len(preview_dependencies),
        warnings=warnings,
        additions=additions,
        removals=removals,
        unchanged=unchanged,
    )


def add_dependency(
    notebook_dir: Path,
    package: str,
    *,
    timeout: int = 120,
) -> DependencyChangeResult:
    """Add a Python package to the notebook.

    Runs ``uv add <package>`` which updates pyproject.toml, resolves
    dependencies, writes uv.lock, and syncs .venv.

    Args:
        notebook_dir: Path to notebook directory
        package: Package specifier (e.g. ``"requests"`` or ``"pandas>=2.0"``)
        timeout: Subprocess timeout in seconds

    Returns:
        DependencyChangeResult with success status
    """
    lock = _get_notebook_lock(notebook_dir)
    with lock:
        return _add_dependency_locked(notebook_dir, package, timeout=timeout)


def _add_dependency_locked(
    notebook_dir: Path, package: str, *, timeout: int = 120
) -> DependencyChangeResult:
    old_lockfile_hash = _lockfile_hash(notebook_dir)

    command_result = _run_uv_command(
        notebook_dir,
        ["add", package],
        timeout=timeout,
        display_name="uv add",
    )
    if command_result.success:
        logger.info("uv add %s succeeded in %s", package, notebook_dir)
    else:
        return DependencyChangeResult(
            success=False,
            package=package,
            action="add",
            error=command_result.error,
            operation_log=command_result.operation_log,
        )

    new_lockfile_hash = _lockfile_hash(notebook_dir)
    return DependencyChangeResult(
        success=True,
        package=package,
        action="add",
        lockfile_changed=old_lockfile_hash != new_lockfile_hash,
        dependencies=list_dependencies(notebook_dir),
        operation_log=command_result.operation_log,
    )


def remove_dependency(
    notebook_dir: Path,
    package: str,
    *,
    timeout: int = 120,
) -> DependencyChangeResult:
    """Remove a Python package from the notebook.

    Runs ``uv remove <package>`` which updates pyproject.toml,
    re-resolves, writes uv.lock, and syncs .venv.

    Args:
        notebook_dir: Path to notebook directory
        package: Package name to remove
        timeout: Subprocess timeout in seconds

    Returns:
        DependencyChangeResult with success status
    """
    lock = _get_notebook_lock(notebook_dir)
    with lock:
        return _remove_dependency_locked(notebook_dir, package, timeout=timeout)


def _remove_dependency_locked(
    notebook_dir: Path, package: str, *, timeout: int = 120
) -> DependencyChangeResult:
    old_lockfile_hash = _lockfile_hash(notebook_dir)

    command_result = _run_uv_command(
        notebook_dir,
        ["remove", package],
        timeout=timeout,
        display_name="uv remove",
    )
    if command_result.success:
        logger.info("uv remove %s succeeded in %s", package, notebook_dir)
    else:
        return DependencyChangeResult(
            success=False,
            package=package,
            action="remove",
            error=command_result.error,
            operation_log=command_result.operation_log,
        )

    new_lockfile_hash = _lockfile_hash(notebook_dir)
    return DependencyChangeResult(
        success=True,
        package=package,
        action="remove",
        lockfile_changed=old_lockfile_hash != new_lockfile_hash,
        dependencies=list_dependencies(notebook_dir),
        operation_log=command_result.operation_log,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lockfile_hash(notebook_dir: Path) -> str:
    """Compute hash of uv.lock for change detection."""
    from strata.notebook.env import compute_lockfile_hash

    return compute_lockfile_hash(notebook_dir)


def parse_requirements_text(requirements_text: str) -> list[str]:
    """Parse a small supported subset of ``requirements.txt`` syntax."""
    requirements: list[str] = []
    seen_names: set[str] = set()

    for raw_line in requirements_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            raise ValueError("Unsupported requirements entry. Use plain package specifiers only.")
        if " #" in line:
            line = line.split(" #", 1)[0].strip()

        validated = _validate_requirement_specifier(line)
        requirement_name, _ = _split_requirement(validated)
        canonical = canonicalize_name(requirement_name)
        if canonical in seen_names:
            raise ValueError(f"Duplicate requirement: {requirement_name}")
        seen_names.add(canonical)
        requirements.append(validated)

    return requirements


def parse_environment_yaml_text(environment_yaml_text: str) -> tuple[list[str], list[str]]:
    """Translate a subset of Conda ``environment.yaml`` into pip requirements."""
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ValueError("PyYAML is required to import environment.yaml") from exc

    try:
        data = yaml.safe_load(environment_yaml_text) or {}
    except Exception as exc:
        raise ValueError(f"Failed to parse environment.yaml: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("environment.yaml must contain a mapping at the top level")

    dependencies = data.get("dependencies", [])
    if not isinstance(dependencies, list):
        raise ValueError("environment.yaml dependencies must be a list")

    warnings: list[str] = []
    requirements: list[str] = []
    seen_names: set[str] = set()

    channels = data.get("channels")
    if isinstance(channels, list) and channels:
        warnings.append(
            "Ignored conda channels from environment.yaml; notebook "
            "environments use pip/uv resolution."
        )

    def add_requirement(requirement: str) -> None:
        validated = _validate_requirement_specifier(requirement)
        requirement_name, _ = _split_requirement(validated)
        canonical = canonicalize_name(requirement_name)
        if canonical in seen_names:
            raise ValueError(f"Duplicate requirement: {requirement_name}")
        seen_names.add(canonical)
        requirements.append(validated)

    for entry in dependencies:
        if isinstance(entry, str):
            translated, entry_warning = _translate_conda_dependency(entry)
            if entry_warning:
                warnings.append(entry_warning)
            if translated:
                add_requirement(translated)
            continue

        if isinstance(entry, dict):
            pip_entries = entry.get("pip")
            if isinstance(pip_entries, list):
                for pip_entry in pip_entries:
                    if not isinstance(pip_entry, str):
                        warnings.append(
                            "Ignored non-string pip dependency entry in environment.yaml."
                        )
                        continue
                    add_requirement(pip_entry.strip())
                continue

            warnings.append("Ignored unsupported mapping entry in environment.yaml dependencies.")
            continue

        warnings.append("Ignored unsupported dependency entry in environment.yaml.")

    return requirements, warnings


def _read_project_dependency_strings(notebook_dir: Path) -> list[str]:
    """Read raw dependency strings from ``pyproject.toml``."""
    pyproject_path = notebook_dir / "pyproject.toml"
    if not pyproject_path.exists():
        return []

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    deps_list: list[str] = data.get("project", {}).get("dependencies", [])
    return [str(dep) for dep in deps_list]


def _dependency_info_from_requirement_strings(
    requirements: list[str],
) -> list[DependencyInfo]:
    """Convert normalized requirement strings to dependency metadata."""
    results: list[DependencyInfo] = []
    for requirement in requirements:
        name, specifier = _split_requirement(requirement)
        results.append(DependencyInfo(name=name, specifier=specifier))
    return results


def _split_requirement(dep_str: str) -> tuple[str, SpecifierSet | None]:
    """Split a requirement into canonical name and parsed specifier.

    Uses ``packaging.requirements.Requirement`` for PEP 508 parsing and
    ``packaging.utils.canonicalize_name`` for PEP 503 name normalization.
    Falls back to returning the raw string as the name (and ``None`` for
    the specifier) when the entry can't be parsed, so malformed
    ``pyproject.toml`` entries still surface in the UI rather than crashing.
    """
    try:
        req = Requirement(dep_str)
    except InvalidRequirement:
        return dep_str.strip(), None
    specifier = req.specifier if req.specifier else None
    return canonicalize_name(req.name), specifier


def _diff_dependency_sets(
    current: list[DependencyInfo],
    target: list[DependencyInfo],
) -> tuple[list[DependencyInfo], list[DependencyInfo], list[DependencyInfo]]:
    """Diff dependency sets by canonical name and semantic specifier equality.

    Names are already PEP 503 canonical at construction (``Pandas`` and
    ``pandas`` collapse); specifiers are ``SpecifierSet`` instances, so
    ``==`` is structural — ``>=1.0,<2.0`` matches ``<2.0,>=1.0``.
    """
    current_map = {dep.name: dep for dep in current}
    target_map = {dep.name: dep for dep in target}

    additions: list[DependencyInfo] = []
    removals: list[DependencyInfo] = []
    unchanged: list[DependencyInfo] = []

    for name, target_dep in target_map.items():
        current_dep = current_map.get(name)
        if current_dep is None:
            additions.append(target_dep)
        elif current_dep.specifier == target_dep.specifier:
            unchanged.append(target_dep)
        else:
            additions.append(target_dep)
            removals.append(current_dep)

    for name, current_dep in current_map.items():
        if name not in target_map:
            removals.append(current_dep)

    additions.sort(key=lambda dep: dep.name)
    removals.sort(key=lambda dep: dep.name)
    unchanged.sort(key=lambda dep: dep.name)
    return additions, removals, unchanged


def _validate_requirement_specifier(requirement: str) -> str:
    """Validate a supported requirement line.

    Accepts plain PEP 508 requirements (name, optional extras, optional
    version specifier); rejects environment markers and URL/direct
    references, which are out of scope for this notebook surface.
    """
    normalized = requirement.strip()
    if not normalized:
        raise ValueError("Requirement cannot be empty")
    if len(normalized) > 200:
        raise ValueError("Requirement specifier too long")
    try:
        req = Requirement(normalized)
    except InvalidRequirement as exc:
        raise ValueError(f"Invalid requirement: {exc}") from exc
    if req.marker is not None:
        raise ValueError("Environment markers are not supported in notebook requirements")
    if req.url is not None:
        raise ValueError("URL / direct-reference requirements are not supported")
    return normalized


def _translate_conda_dependency(dependency: str) -> tuple[str | None, str | None]:
    """Best-effort conversion from a Conda dependency string to a pip requirement."""
    normalized = dependency.strip()
    if not normalized:
        return None, None

    warning: str | None = None
    if "::" in normalized:
        _, normalized = normalized.split("::", 1)
        warning = "Ignored conda channel prefixes in environment.yaml; using package names only."

    lowered = normalized.lower()
    if lowered == "pip":
        return None, "Ignored explicit pip bootstrap entry from environment.yaml."
    if lowered.startswith("python"):
        return (
            None,
            "Ignored python version pin from environment.yaml; notebook "
            "Python is managed separately.",
        )

    if (
        "==" not in normalized
        and "!=" not in normalized
        and ">=" not in normalized
        and "<=" not in normalized
        and "~=" not in normalized
        and "=" in normalized
    ):
        pieces = normalized.split("=")
        if len(pieces) == 2 and pieces[0] and pieces[1]:
            normalized = f"{pieces[0]}=={pieces[1]}"
        else:
            return (
                None,
                f"Ignored unsupported conda dependency entry: {dependency}",
            )

    return normalized, warning


def _restore_dependency_files(
    pyproject_path: Path,
    old_pyproject: bytes,
    lockfile_path: Path,
    old_lockfile: bytes | None,
) -> None:
    """Restore dependency files after a failed import attempt."""
    pyproject_path.write_bytes(old_pyproject)
    if old_lockfile is None:
        if lockfile_path.exists():
            lockfile_path.unlink()
    else:
        lockfile_path.write_bytes(old_lockfile)

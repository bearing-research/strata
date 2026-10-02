"""Dependency management for notebooks via ``uv add`` / ``uv remove``.

Every mutation re-syncs the lockfile so ``uv.lock`` and ``.venv/`` stay consistent.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import filelock
import tomli_w
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from strata.notebook.harness_user import (
    HarnessUser,
    LocalExecutionRefused,
    identity_env,
    resolve_harness_user,
    running_server_config,
    spawn_kwargs,
)

logger = logging.getLogger(__name__)
_MAX_OPERATION_LOG_CHARS = 12_000

# Per-notebook locks: concurrent ``uv add`` calls can corrupt uv.lock.
_locks: dict[str, threading.Lock] = {}
_locks_lock = threading.Lock()


def _get_notebook_lock(notebook_dir: Path) -> threading.Lock:
    """Get or create a per-notebook lock for uv operations."""
    key = str(notebook_dir.resolve())
    with _locks_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def renv_process_lock(notebook_dir: Path) -> filelock.FileLock:
    """Cross-process lock guarding renv mutations for *notebook_dir*.

    renv has no locking of its own, and a server and ``strata run`` can sync the same
    dir at once. The lock file lives under ``.strata/`` so it is never committed. uv
    needs no equivalent: it locks its venv and cache itself.
    """
    lock_dir = notebook_dir / ".strata"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(str(lock_dir / "renv-process.lock"))


# --- Result types ---


@dataclass
class DependencyInfo:
    """One declared or resolved notebook dependency.

    ``name`` is PEP 503 canonical. ``version`` is set only for ``uv.lock`` entries and
    ``specifier`` only for ``pyproject.toml`` ones; render both with ``str(...)``.
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


UV_NOT_FOUND_MESSAGE = (
    "uv not found on PATH. Install uv "
    "(https://docs.astral.sh/uv/getting-started/installation/) or add it to PATH "
    "— the installer puts uv in ~/.local/bin, which a non-login shell "
    "(ssh, cron) often doesn't include."
)


def resolve_uv() -> str | None:
    """Locate ``uv``: PATH first, then uv's installer dirs.

    Non-login shells (``ssh host 'strata run ...'``, cron) often lack
    ``~/.local/bin`` and ``~/.cargo/bin`` on PATH.
    """
    found = shutil.which("uv")
    if found:
        return found
    for base in ("~/.local/bin", "~/.cargo/bin"):
        candidate = Path(base).expanduser() / "uv"
        if candidate.is_file():
            return str(candidate)
    return None


# Vars that point uv at the server's environment instead of the notebook's
# ``.venv``. A foreign VIRTUAL_ENV only warns, but in every operation log.
_FOREIGN_ENVIRONMENT_VARS = ("UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV")


def uv_env(
    base: Mapping[str, str] | None = None, extra: dict[str, str] | None = None
) -> dict[str, str]:
    """The environment for a uv command on a notebook.

    *base* (default: the server's environment) minus the variables that point uv at
    another environment, then *extra* on top. In service mode uv installs wheels
    only: building an sdist runs its build backend as the server's user, which the
    harness user exists to keep cell code away from.
    """
    env = {
        name: value
        for name, value in (os.environ if base is None else base).items()
        if name not in _FOREIGN_ENVIRONMENT_VARS
    }
    if getattr(running_server_config(), "deployment_mode", None) == "service":
        env["UV_NO_BUILD"] = "1"
    env.update(extra or {})
    return env


def _run_uv_command(
    notebook_dir: Path,
    args: list[str],
    *,
    timeout: int,
    display_name: str,
    env: dict[str, str] | None = None,
) -> _UvCommandResult:
    """Run a uv command and capture bounded UI logs.

    *env* adds to the inherited environment, e.g. ``UV_PROJECT_ENVIRONMENT``.
    """
    started = time.perf_counter()
    uv = resolve_uv()
    if uv is None:
        return _UvCommandResult(
            success=False,
            error=UV_NOT_FOUND_MESSAGE,
            operation_log=EnvironmentOperationLog(
                command=_format_command_for_ui(["uv", *args]),
                duration_ms=int((time.perf_counter() - started) * 1000),
            ),
        )
    command = [uv, *args]
    formatted_command = _format_command_for_ui(["uv", *args])

    try:
        completed = subprocess.run(
            command,
            cwd=str(notebook_dir),
            env=uv_env(extra=env),
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
    env: dict[str, str] | None = None,
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
            env=uv_env(extra=env),
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


# --- Core operations ---


def list_dependencies(notebook_dir: Path) -> list[DependencyInfo]:
    """List declared dependencies from pyproject.toml without shelling out."""
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
    """One R package installed in the notebook's renv project library.

    ``version`` is the string CRAN/renv stores (not PEP 440); render it as-is.
    """

    name: str
    version: str


@dataclass
class RPackageListing:
    """Result of listing R packages installed in the project library.

    ``packages`` is empty on failure as well as for an empty library; ``error`` is
    set only on failure. ``status`` values:

    * ``"ok"``: ``packages`` is the live library content (possibly empty).
    * ``"rscript_missing"``: Rscript is not on PATH.
    * ``"renv_not_active"``: ``.Rprofile`` did not activate renv (before
      ``renv::init()``, or a broken activator); treat as empty.
    * ``"failed"``: timeout, non-zero exit or malformed output; ``error`` says which.
    """

    packages: list[RPackageInfo]
    status: str
    error: str | None = None


def rscript_env(
    harness_user: HarnessUser | None, extra: dict[str, str] | None = None
) -> dict[str, str] | None:
    """The environment for an Rscript started in a notebook directory.

    As the server: the server's own with *extra* on top, ``None`` when there is
    nothing to add. As the harness user: what a cell is given, because such an
    Rscript sources the notebook's ``.Rprofile`` and runs package configure scripts.
    """
    if harness_user is None:
        return {**os.environ, **extra} if extra else None
    from strata.notebook.harness_env import configured_allowlist, harness_env

    return identity_env(harness_env(configured_allowlist(), extra), harness_user)


def list_r_packages(notebook_dir: Path, *, timeout: int = 30) -> RPackageListing:
    """List R packages installed in the notebook's renv project library.

    Runs ``Rscript`` in *notebook_dir* so the renv activator in ``.Rprofile`` loads,
    and scopes ``installed.packages()`` to the project library; unscoped it lists
    every ``.libPaths()`` entry, base R included. When renv is not loadable the
    snippet prints ``RENV_NOT_ACTIVE`` and the status is ``"renv_not_active"``.
    """
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused as exc:
        return RPackageListing(packages=[], status="failed", error=str(exc))
    rscript = shutil.which("Rscript")
    if rscript is None:
        return RPackageListing(packages=[], status="rscript_missing", error=None)

    # ``tryCatch`` turns a missing renv namespace into ``RENV_NOT_ACTIVE``
    # rather than a bare nonzero exit. Loop, not ``apply``: a 1-row matrix
    # collapses to a vector.
    r_snippet = "\n".join(
        [
            "lib <- tryCatch(",
            "  renv::paths$library(project = getwd()),",
            "  error = function(e) NULL",
            ")",
            "if (is.null(lib) || !dir.exists(lib)) {",
            '  cat("RENV_NOT_ACTIVE\\n")',
            "} else {",
            "  ip <- installed.packages(lib.loc = lib)",
            "  for (i in seq_len(nrow(ip))) {",
            '    cat(ip[i, "Package"], ip[i, "Version"], sep = "\\t")',
            '    cat("\\n")',
            "  }",
            "}",
        ]
    )

    try:
        proc = subprocess.run(  # noqa: S603 — rscript resolved via shutil.which
            [rscript, "-e", r_snippet],
            cwd=str(notebook_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=rscript_env(harness_user),
            **spawn_kwargs(harness_user),
        )
    except subprocess.TimeoutExpired as exc:
        logger.debug("R package listing timed out: %s", exc)
        return RPackageListing(
            packages=[], status="failed", error=f"Rscript timed out after {timeout}s"
        )
    except OSError as exc:
        logger.debug("R package listing failed to spawn: %s", exc)
        return RPackageListing(packages=[], status="failed", error=str(exc))

    if proc.returncode != 0:
        snippet = proc.stderr.strip()[:200] or proc.stdout.strip()[:200]
        logger.debug(
            "R package listing returned non-zero (%d): %s",
            proc.returncode,
            snippet,
        )
        return RPackageListing(
            packages=[],
            status="failed",
            error=snippet or f"Rscript exited with code {proc.returncode}",
        )

    if "RENV_NOT_ACTIVE" in proc.stdout:
        return RPackageListing(packages=[], status="renv_not_active", error=None)

    packages: list[RPackageInfo] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split("	")
        if len(parts) >= 2 and parts[0] and parts[1]:
            packages.append(RPackageInfo(name=parts[0], version=parts[1]))
    packages.sort(key=lambda pkg: pkg.name)
    return RPackageListing(packages=packages, status="ok", error=None)


# --- R bootstrap + install (same lock and operation log as ``uv add``) ---

# CRAN names only: the name is interpolated into the Rscript snippet.
_R_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")


def is_valid_r_package_name(name: str) -> bool:
    """Whether *name* is a syntactically valid CRAN package name."""
    return bool(_R_PACKAGE_NAME_RE.fullmatch(name or ""))


_R_INSTALL_REFUSAL = (
    "R packages are not installed from the notebook on this server: renv builds "
    "them from source, which runs their code, and it writes renv.lock and "
    ".Rprofile into the notebook directory, which cell code may not change. Add "
    "the package to renv.lock where the notebook is authored; the server restores "
    "it as the harness user (STRATA_NOTEBOOK_HARNESS_USER) when the notebook opens."
)


def _r_install_refusal() -> str | None:
    """Why ``renv_init`` / ``renv_add`` may not run here, ``None`` when they may.

    They run as this process or not at all: an install writes files in the notebook
    directory the harness user cannot write, and it may not run as a server that
    isolates cell code.
    """
    try:
        harness_user = resolve_harness_user()
    except LocalExecutionRefused:
        return _R_INSTALL_REFUSAL
    return None if harness_user is None else _R_INSTALL_REFUSAL


@dataclass
class _RscriptCommandResult:
    """Internal subprocess result wrapper for Rscript invocations."""

    success: bool
    error: str | None
    operation_log: EnvironmentOperationLog


def _run_rscript_command(
    notebook_dir: Path,
    snippet: str,
    *,
    timeout: int,
    display_name: str,
) -> _RscriptCommandResult:
    """Run an Rscript ``-e`` snippet and capture bounded UI logs, like ``_run_uv_command``."""
    rscript = shutil.which("Rscript")
    formatted_command = f"Rscript -e {shlex.quote(snippet)}"

    if rscript is None:
        return _RscriptCommandResult(
            success=False,
            error="Rscript not found on PATH",
            operation_log=EnvironmentOperationLog(command=formatted_command),
        )

    started = time.perf_counter()
    try:
        completed = subprocess.run(  # noqa: S603 — rscript resolved via shutil.which
            [rscript, "-e", snippet],
            cwd=str(notebook_dir),
            timeout=timeout,
            capture_output=True,
            check=True,
            text=True,
        )
    except subprocess.TimeoutExpired as exc:
        stdout, stdout_truncated = _trim_output_for_ui(exc.stdout)
        stderr, stderr_truncated = _trim_output_for_ui(exc.stderr)
        return _RscriptCommandResult(
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
        return _RscriptCommandResult(
            success=False,
            error=f"{display_name} failed (exit {exc.returncode})",
            operation_log=EnvironmentOperationLog(
                command=formatted_command,
                duration_ms=int((time.perf_counter() - started) * 1000),
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            ),
        )

    stdout, stdout_truncated = _trim_output_for_ui(completed.stdout)
    stderr, stderr_truncated = _trim_output_for_ui(completed.stderr)
    return _RscriptCommandResult(
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


async def run_rscript_command_streaming(
    notebook_dir: Path,
    snippet: str,
    *,
    timeout: int,
    display_name: str,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
    env: dict[str, str] | None = None,
) -> _RscriptCommandResult:
    """Run an Rscript ``-e`` snippet asynchronously with streamed stdout/stderr.

    The R counterpart of ``run_uv_command_streaming``: an ``arrow`` source compile
    during ``renv::init`` can take 5 to 10 minutes, and *on_update* lets the env
    panel show its output live.
    """
    rscript = shutil.which("Rscript")
    formatted_command = f"Rscript -e {shlex.quote(snippet)}"

    if rscript is None:
        return _RscriptCommandResult(
            success=False,
            error="Rscript not found on PATH",
            operation_log=EnvironmentOperationLog(command=formatted_command),
        )

    started = time.perf_counter()
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
            rscript,
            "-e",
            snippet,
            cwd=str(notebook_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **env} if env else None,
        )
    except FileNotFoundError:
        return _RscriptCommandResult(
            success=False,
            error="Rscript not found on PATH",
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
        return _RscriptCommandResult(
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

    operation_log = EnvironmentOperationLog(
        command=formatted_command,
        duration_ms=int((time.perf_counter() - started) * 1000),
        stdout=stdout_buffer.text,
        stderr=stderr_buffer.text,
        stdout_truncated=stdout_buffer.truncated,
        stderr_truncated=stderr_buffer.truncated,
    )

    if process.returncode == 0:
        return _RscriptCommandResult(success=True, error=None, operation_log=operation_log)

    return _RscriptCommandResult(
        success=False,
        error=f"{display_name} failed (exit {process.returncode})",
        operation_log=operation_log,
    )


@dataclass
class RJobResult:
    """Result of an R env job (``renv::init`` / ``renv::install``).

    ``lockfile_changed`` means the same as on ``DependencyChangeResult``, so staleness
    propagates after an ``r_add`` as after a Python ``add``.
    """

    success: bool
    action: str  # "r_init" | "r_add"
    package: str | None
    lockfile_changed: bool = False
    error: str | None = None
    operation_log: EnvironmentOperationLog | None = None


def _renv_lockfile_hash(notebook_dir: Path) -> str:
    """SHA-256 of ``renv.lock``, or of ``b""`` when absent.

    Goes through ``_fold_lockfile_into_hash`` so the read takes the path CodeQL's
    ``py/path-injection`` model accepts.
    """
    import hashlib

    from strata.notebook.env import _fold_lockfile_into_hash

    hasher = hashlib.sha256()
    _fold_lockfile_into_hash(hasher, notebook_dir, "renv.lock", tag=None)
    return hasher.hexdigest()


async def _run_renv_mutation(
    notebook_dir: Path,
    snippet: str,
    *,
    timeout: int,
    display_name: str,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None,
) -> _RscriptCommandResult:
    """Run a snippet that installs into the notebook's R library.

    With the shared backend the library is shared, so it is never installed into:
    the notebook first moves onto a private library restored from the package cache,
    and once the snippet has written ``renv.lock`` that library is adopted under the
    new lock's key. A failed snippet links the notebook back to the library for the
    lock it still has.
    """
    from strata.notebook.env_backend import shared_root

    root = shared_root()
    if root is None:
        return await run_rscript_command_streaming(
            notebook_dir, snippet, timeout=timeout, display_name=display_name, on_update=on_update
        )
    from strata.notebook.shared_env import (
        adopt_r_library,
        detach_r_library,
        link_r_library_if_built,
        r_env,
    )

    await asyncio.to_thread(detach_r_library, notebook_dir, root)
    # Restore first, or the adopted library holds only the new package and
    # every notebook on the new lock links to it without restoring.
    snippet = "renv::restore(prompt = FALSE)\n" + snippet
    result = await run_rscript_command_streaming(
        notebook_dir,
        snippet,
        timeout=timeout,
        display_name=display_name,
        on_update=on_update,
        env=r_env(root),
    )
    if result.success:
        await asyncio.to_thread(adopt_r_library, notebook_dir, root)
    else:
        # Relink the library for the unchanged lock, if built.
        await asyncio.to_thread(link_r_library_if_built, notebook_dir, root)
    return result


async def renv_init(
    notebook_dir: Path,
    *,
    timeout: int = 900,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
) -> RJobResult:
    """Bootstrap renv in *notebook_dir* with streamed Rscript output.

    ``on_update(stream, text, truncated)`` fires per output chunk; the 900s default
    covers source compiles on platforms without binaries. Holds the per-notebook
    lock and the cross-process ``renv_process_lock`` throughout, both acquired via
    ``asyncio.to_thread`` so a held lock does not block the event loop.
    """
    refusal = _r_install_refusal()
    if refusal is not None:
        return RJobResult(success=False, action="r_init", package=None, error=refusal)
    lock = _get_notebook_lock(notebook_dir)
    await asyncio.to_thread(lock.acquire)
    try:
        process_lock = renv_process_lock(notebook_dir)
        try:
            await asyncio.to_thread(process_lock.acquire, timeout)
        except filelock.Timeout:
            return RJobResult(
                success=False,
                action="r_init",
                package=None,
                error=(
                    "Another process is mutating this notebook's R environment "
                    f"(renv lock held for over {timeout}s). Retry once it finishes."
                ),
            )
        try:
            return await _renv_init_locked(notebook_dir, timeout=timeout, on_update=on_update)
        finally:
            process_lock.release()
    finally:
        lock.release()


async def _renv_init_locked(
    notebook_dir: Path,
    *,
    timeout: int,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None,
) -> RJobResult:
    old_hash = _renv_lockfile_hash(notebook_dir)
    # The harness needs jsonlite + arrow in the project library (activate
    # scopes ``.libPaths()`` to it). ``bare = TRUE`` skips the post-init
    # snapshot, so snapshot explicitly or ``renv.lock`` never appears.
    snippet = "\n".join(
        [
            'if (!requireNamespace("renv", quietly = TRUE)) {',
            '  install.packages("renv", repos = "https://cloud.r-project.org", quiet = TRUE)',
            "}",
            "renv::init(bare = TRUE)",
            'renv::install(c("jsonlite", "arrow"))',
            'renv::snapshot(type = "all", prompt = FALSE)',
        ]
    )
    result = await _run_renv_mutation(
        notebook_dir,
        snippet,
        timeout=timeout,
        display_name="renv::init",
        on_update=on_update,
    )
    if not result.success:
        return RJobResult(
            success=False,
            action="r_init",
            package=None,
            error=result.error,
            operation_log=result.operation_log,
        )
    new_hash = _renv_lockfile_hash(notebook_dir)
    return RJobResult(
        success=True,
        action="r_init",
        package=None,
        lockfile_changed=old_hash != new_hash,
        operation_log=result.operation_log,
    )


async def renv_add(
    notebook_dir: Path,
    package: str,
    *,
    timeout: int = 600,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None = None,
) -> RJobResult:
    """Install an R package and snapshot the lockfile, with streamed output.

    ``snapshot(type = "all")`` makes the lockfile reflect the library rather than
    only packages referenced in source. The name is concatenated into the Rscript
    body, so names that are not valid CRAN names are rejected first.
    """
    if not is_valid_r_package_name(package):
        return RJobResult(
            success=False,
            action="r_add",
            package=package,
            error=(
                f"Invalid R package name: {package!r}. CRAN names match "
                "[A-Za-z][A-Za-z0-9.]* — no dashes, no shell metacharacters."
            ),
        )
    refusal = _r_install_refusal()
    if refusal is not None:
        return RJobResult(success=False, action="r_add", package=package, error=refusal)
    lock = _get_notebook_lock(notebook_dir)
    await asyncio.to_thread(lock.acquire)
    try:
        process_lock = renv_process_lock(notebook_dir)
        try:
            await asyncio.to_thread(process_lock.acquire, timeout)
        except filelock.Timeout:
            return RJobResult(
                success=False,
                action="r_add",
                package=package,
                error=(
                    "Another process is mutating this notebook's R environment "
                    f"(renv lock held for over {timeout}s). Retry once it finishes."
                ),
            )
        try:
            return await _renv_add_locked(
                notebook_dir, package, timeout=timeout, on_update=on_update
            )
        finally:
            process_lock.release()
    finally:
        lock.release()


async def _renv_add_locked(
    notebook_dir: Path,
    package: str,
    *,
    timeout: int,
    on_update: Callable[[str, str, bool], Awaitable[None] | None] | None,
) -> RJobResult:
    old_hash = _renv_lockfile_hash(notebook_dir)
    # Safe to embed unescaped: ``is_valid_r_package_name`` allows only [A-Za-z0-9.].
    snippet = f'renv::install("{package}"); renv::snapshot(type = "all", prompt = FALSE)'
    result = await _run_renv_mutation(
        notebook_dir,
        snippet,
        timeout=timeout,
        display_name="renv::install",
        on_update=on_update,
    )
    if not result.success:
        return RJobResult(
            success=False,
            action="r_add",
            package=package,
            error=result.error,
            operation_log=result.operation_log,
        )
    new_hash = _renv_lockfile_hash(notebook_dir)
    return RJobResult(
        success=True,
        action="r_add",
        package=package,
        lockfile_changed=old_hash != new_hash,
        operation_log=result.operation_log,
    )


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

        from strata.notebook.env_backend import get_backend

        command_result = get_backend(notebook_dir).sync(python_version=None, timeout=timeout)
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

        from strata.notebook.env_backend import get_backend

        command_result = await get_backend(notebook_dir).sync_streaming(
            python_version=None, timeout=timeout, on_update=on_update
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
    dev: bool = False,
    timeout: int = 120,
) -> DependencyChangeResult:
    """Add a Python package to the notebook with ``uv add`` (``--dev`` when *dev*).

    Updates pyproject.toml, writes uv.lock and syncs .venv. A dev-group package is
    synced but excluded from the cell-provenance env hash, so dev tooling does not
    invalidate cell caches.
    """
    lock = _get_notebook_lock(notebook_dir)
    with lock:
        return _add_dependency_locked(notebook_dir, package, dev=dev, timeout=timeout)


def _add_dependency_locked(
    notebook_dir: Path, package: str, *, dev: bool = False, timeout: int = 120
) -> DependencyChangeResult:
    old_lockfile_hash = _lockfile_hash(notebook_dir)

    from strata.notebook.env_backend import get_backend

    command_result = get_backend(notebook_dir).add(package, timeout=timeout, dev=dev)
    if command_result.success:
        logger.info("uv add %s%s succeeded in %s", "--dev " if dev else "", package, notebook_dir)
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


def ensure_dev_tool(
    notebook_dir: Path,
    tool: str,
    *,
    timeout: int = 120,
) -> DependencyChangeResult:
    """Provision a dev tool (pytest / ruff / ty / mypy) into the notebook.

    Adds *tool* to the ``dev`` group, which stays out of the cell-provenance env hash,
    so provisioning never invalidates a cell. Tool-backed features call this rather
    than running ``uv add --dev`` themselves.
    """
    return add_dependency(notebook_dir, tool, dev=True, timeout=timeout)


def remove_dependency(
    notebook_dir: Path,
    package: str,
    *,
    timeout: int = 120,
) -> DependencyChangeResult:
    """Remove a Python package with ``uv remove``, re-resolving and syncing .venv."""
    lock = _get_notebook_lock(notebook_dir)
    with lock:
        return _remove_dependency_locked(notebook_dir, package, timeout=timeout)


def _remove_dependency_locked(
    notebook_dir: Path, package: str, *, timeout: int = 120
) -> DependencyChangeResult:
    old_lockfile_hash = _lockfile_hash(notebook_dir)

    from strata.notebook.env_backend import get_backend

    command_result = get_backend(notebook_dir).remove(package, timeout=timeout)
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


# --- Helpers ---


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

    An unparseable entry returns the raw string and ``None``, so a malformed
    ``pyproject.toml`` entry still shows in the UI.
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

    ``SpecifierSet`` equality is structural: ``>=1.0,<2.0`` equals ``<2.0,>=1.0``.
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
    """Validate a plain PEP 508 requirement line.

    Environment markers and URL/direct references are rejected as out of scope.
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
    # The interpreter pin only: python-dateutil and friends are ordinary packages.
    if re.match(r"python(?:$|[\s=<>!~\[])", lowered):
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

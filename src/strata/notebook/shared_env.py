"""One environment per lockfile, shared by every notebook with that lock.

Selected with ``notebook_env_backend = "shared"``. Environments live under
``notebook_shared_env_dir`` (default: ``envs`` beside the notebook storage
directory), one directory per **key**, and a notebook's ``.venv`` is a symlink
to its key's directory. A second notebook with the same lock does not install
anything: its sync is the link.

The key is ``sha256`` of the raw ``uv.lock`` bytes together with the exact
interpreter build and platform tag. It is deliberately not the provenance
environment hash, which leaves the dev group out and so would put two different
installed environments under one key.

A shared environment is never changed in place. ``add`` and ``remove`` change
the notebook's ``pyproject.toml`` and ``uv.lock`` without syncing, then sync,
which lands in another key and moves only that notebook's link.

Each key keeps a reference file per notebook linked to it, and ``collect``
removes a key no notebook links to once it has gone unused for
``notebook_shared_env_ttl_days``. A reference whose notebook now links
elsewhere, or no longer exists, does not keep a key alive.

R libraries are shared the same way, one per ``renv.lock`` (see the R section
below).

POSIX only: the link is a symlink.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import filelock

from strata.notebook.dependencies import (
    EnvironmentOperationLog,
    _run_uv_command,
    _UvCommandResult,
    resolve_uv,
    run_uv_command_streaming,
)
from strata.notebook.env_backend import _StreamCallback

# Written last, once an install succeeded: a directory without it is a sync
# that died part way and is installed again.
COMPLETE_MARKER = ".strata-env-complete"
_REFS = "refs"
_PROBE = (
    "import platform, sys, sysconfig; "
    "print(sys.implementation.name, platform.python_version(), sys.abiflags, "
    "sysconfig.get_platform())"
)


def _failure(command: str, error: str) -> _UvCommandResult:
    return _UvCommandResult(
        success=False, error=error, operation_log=EnvironmentOperationLog(command=command)
    )


def _combined(results: list[_UvCommandResult]) -> _UvCommandResult:
    """One result for a sequence of uv commands, as the UI shows one log."""
    last = results[-1]
    logs = [result.operation_log for result in results]
    return _UvCommandResult(
        success=last.success,
        error=last.error,
        operation_log=EnvironmentOperationLog(
            command=" && ".join(log.command for log in logs),
            duration_ms=sum(log.duration_ms or 0 for log in logs),
            stdout="".join(log.stdout for log in logs),
            stderr="".join(log.stderr for log in logs),
            stdout_truncated=any(log.stdout_truncated for log in logs),
            stderr_truncated=any(log.stderr_truncated for log in logs),
        ),
    )


def _key_lock(root: Path, key: str) -> filelock.FileLock:
    """Held while a key's environment is installed, linked or removed.

    Not thread-local, and with a finite timeout: the streaming sync acquires it
    on a worker thread and releases it on the event loop's, and filelock's
    same-thread deadlock check misreads that as a second holder once the
    worker thread is reused. An hour is longer than any install is allowed.
    """
    return filelock.FileLock(str(root / f"{key}.lock"), thread_local=False, timeout=3600)


def _ref_name(notebook_dir: Path) -> str:
    return hashlib.sha256(str(notebook_dir.resolve()).encode()).hexdigest()[:32]


class SharedEnvBackend:
    """Environment backend whose ``.venv`` is a link into a shared, keyed store."""

    name = "uv-shared"
    supports_mutations = True

    def __init__(self, notebook_dir: Path, root: Path) -> None:
        self.notebook_dir = Path(notebook_dir)
        # Absolute, because links point into it and references are compared
        # against where links point.
        self.root = Path(root).resolve()

    def python_executable(self) -> Path:
        return self.notebook_dir / ".venv" / "bin" / "python"

    # --- keys ---

    def _interpreter(self, python_version: str | None) -> tuple[Path, str] | str:
        """The interpreter a sync would use, and its exact build; or why not."""
        uv = resolve_uv()
        if uv is None:
            return "uv not found on PATH"
        # --system: the base interpreter, not the one inside whichever
        # environment .venv links to now.
        args = [uv, "python", "find", "--system"] + ([python_version] if python_version else [])
        try:
            found = subprocess.run(
                args,
                cwd=self.notebook_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            interpreter = Path(found.stdout.strip())
            build = subprocess.run(
                [str(interpreter), "-c", _PROBE],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            ).stdout.strip()
        except subprocess.CalledProcessError as exc:
            return f"no interpreter for this notebook: {(exc.stderr or exc.stdout).strip()}"
        except subprocess.TimeoutExpired:
            return "timed out looking for this notebook's interpreter"
        return interpreter, build

    def key(self, build: str) -> str:
        """The environment key for the notebook's current lock and *build*."""
        lock = hashlib.sha256((self.notebook_dir / "uv.lock").read_bytes()).hexdigest()
        return hashlib.sha256(f"{lock}\n{build}".encode()).hexdigest()[:32]

    # --- linking ---

    def _link(self, key: str) -> None:
        """Point the notebook's ``.venv`` at *key*, and move its reference."""
        target = self.root / key
        venv = self.notebook_dir / ".venv"
        ref = _ref_name(self.notebook_dir)
        if venv.is_symlink():
            previous = Path(os.readlink(venv))
            if previous.parent == self.root and previous.name != key:
                (self.root / _REFS / previous.name / ref).unlink(missing_ok=True)
        elif venv.exists():
            # A per-notebook environment from before the switch.
            shutil.rmtree(venv)
        refs = self.root / _REFS / key
        refs.mkdir(parents=True, exist_ok=True)
        (refs / ref).write_text(str(self.notebook_dir.resolve()))
        staged = self.notebook_dir / f".venv.link-{uuid.uuid4().hex[:8]}"
        staged.symlink_to(target, target_is_directory=True)
        os.replace(staged, venv)
        # Last used, for the sweep.
        os.utime(target / COMPLETE_MARKER)

    def _prepare(self, python_version: str | None) -> tuple[Path, str, filelock.FileLock] | str:
        found = self._interpreter(python_version)
        if isinstance(found, str):
            return found
        interpreter, build = found
        key = self.key(build)
        self.root.mkdir(parents=True, exist_ok=True)
        return interpreter, key, _key_lock(self.root, key)

    def _install_args(self, interpreter: Path) -> list[str]:
        return ["sync", "--frozen", "--python", str(interpreter)]

    def _install_env(self, key: str) -> dict[str, str]:
        return {"UV_PROJECT_ENVIRONMENT": str(self.root / key)}

    def _detached(self) -> dict[str, str]:
        """For commands that change only the notebook's files: point uv at an
        environment that does not exist, so it neither reads nor touches the
        one .venv links to."""
        return {"UV_PROJECT_ENVIRONMENT": str(self.root / ".no-environment")}

    def _is_complete(self, key: str) -> bool:
        return (self.root / key / COMPLETE_MARKER).exists()

    # --- the backend surface ---

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        locked = _run_uv_command(
            self.notebook_dir,
            ["lock"],
            timeout=timeout,
            display_name="uv lock",
            env=self._detached(),
        )
        if not locked.success:
            return locked
        prepared = self._prepare(python_version)
        if isinstance(prepared, str):
            return _failure("uv sync", prepared)
        interpreter, key, lock = prepared
        results = [locked]
        with lock:
            if not self._is_complete(key):
                installed = _run_uv_command(
                    self.notebook_dir,
                    self._install_args(interpreter),
                    timeout=timeout,
                    display_name="uv sync",
                    env=self._install_env(key),
                )
                results.append(installed)
                if not installed.success:
                    return _combined(results)
                (self.root / key / COMPLETE_MARKER).touch()
            self._link(key)
        return _combined(results)

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        locked = await run_uv_command_streaming(
            self.notebook_dir,
            ["lock"],
            timeout=timeout,
            display_name="uv lock",
            on_update=on_update,
            env=self._detached(),
        )
        if not locked.success:
            return locked
        prepared = await asyncio.to_thread(self._prepare, python_version)
        if isinstance(prepared, str):
            return _failure("uv sync", prepared)
        interpreter, key, lock = prepared
        results = [locked]
        await asyncio.to_thread(lock.acquire)
        try:
            if not self._is_complete(key):
                installed = await run_uv_command_streaming(
                    self.notebook_dir,
                    self._install_args(interpreter),
                    timeout=timeout,
                    display_name="uv sync",
                    on_update=on_update,
                    env=self._install_env(key),
                )
                results.append(installed)
                if not installed.success:
                    return _combined(results)
                (self.root / key / COMPLETE_MARKER).touch()
            self._link(key)
        finally:
            lock.release()
        return _combined(results)

    def add(self, package: str, *, timeout: int, dev: bool = False) -> _UvCommandResult:
        args = ["add", "--no-sync", *(["--dev"] if dev else []), package]
        changed = _run_uv_command(
            self.notebook_dir, args, timeout=timeout, display_name="uv add", env=self._detached()
        )
        if not changed.success:
            return changed
        return _combined([changed, self.sync(python_version=None, timeout=timeout)])

    def remove(self, package: str, *, timeout: int) -> _UvCommandResult:
        changed = _run_uv_command(
            self.notebook_dir,
            ["remove", "--no-sync", package],
            timeout=timeout,
            display_name="uv remove",
            env=self._detached(),
        )
        if not changed.success:
            return changed
        return _combined([changed, self.sync(python_version=None, timeout=timeout)])

    async def add_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        changed = await run_uv_command_streaming(
            self.notebook_dir,
            ["add", "--no-sync", package],
            timeout=timeout,
            display_name="uv add",
            on_update=on_update,
            env=self._detached(),
        )
        if not changed.success:
            return changed
        synced = await self.sync_streaming(
            python_version=None, timeout=timeout, on_update=on_update
        )
        return _combined([changed, synced])

    async def remove_streaming(
        self,
        package: str,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        changed = await run_uv_command_streaming(
            self.notebook_dir,
            ["remove", "--no-sync", package],
            timeout=timeout,
            display_name="uv remove",
            on_update=on_update,
            env=self._detached(),
        )
        if not changed.success:
            return changed
        synced = await self.sync_streaming(
            python_version=None, timeout=timeout, on_update=on_update
        )
        return _combined([changed, synced])

    async def lock_streaming(
        self,
        *,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        return await run_uv_command_streaming(
            self.notebook_dir,
            ["lock"],
            timeout=timeout,
            display_name="uv lock",
            on_update=on_update,
            env=self._detached(),
        )


# --- R: one renv library per renv.lock ------------------------------------
#
# The same store holds R libraries under ``r/``, one per key of the raw
# ``renv.lock`` bytes and the exact R build, apart from the Python keys so a
# change to one lock does not rebuild the other language's environment. A
# notebook's ``renv/library`` is a link to its key's directory: every Rscript
# (the harness, the warm pool, a restore) reads the library through renv's
# project path with nothing to configure. renv's package cache
# (``RENV_PATHS_CACHE``) lives on the same volume, under ``r/cache``, so a
# library built for a changed lock links the packages it already has.
#
# A shared library is never changed in place either. Installing a package
# first detaches the notebook onto a private library restored from the cache,
# and once ``renv.lock`` is written the library is adopted under its new key.

R_DIR = "r"
R_CACHE = "cache"
_R_PROBE = 'cat(R.version$version.string, R.version$platform, sep = " ")'


def r_build() -> str | None:
    """The R build a restore would use, or None without Rscript."""
    rscript = shutil.which("Rscript")
    if rscript is None:
        return None
    try:
        return subprocess.run(
            [rscript, "--vanilla", "-e", _R_PROBE],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def r_key(notebook_dir: Path, build: str) -> str:
    lock = hashlib.sha256((notebook_dir / "renv.lock").read_bytes()).hexdigest()
    return hashlib.sha256(f"{lock}\n{build}".encode()).hexdigest()[:32]


def r_env(root: Path) -> dict[str, str]:
    """Environment for an Rscript that installs: the package cache in the store."""
    return {"RENV_PATHS_CACHE": str(Path(root).resolve() / R_DIR / R_CACHE)}


def _r_library(notebook_dir: Path) -> Path:
    return notebook_dir / "renv" / "library"


def _link_r_library(notebook_dir: Path, r_root: Path, key: str) -> None:
    """Point the notebook's ``renv/library`` at *key*, and move its reference."""
    library = _r_library(notebook_dir)
    ref = _ref_name(notebook_dir)
    if library.is_symlink():
        previous = Path(os.readlink(library))
        if previous.parent == r_root and previous.name != key:
            (r_root / _REFS / previous.name / ref).unlink(missing_ok=True)
    elif library.exists():
        # A per-notebook library from before the switch, or a private one
        # already adopted.
        shutil.rmtree(library)
    refs = r_root / _REFS / key
    refs.mkdir(parents=True, exist_ok=True)
    (refs / ref).write_text(str(notebook_dir.resolve()))
    library.parent.mkdir(parents=True, exist_ok=True)
    staged = library.parent / f"library.link-{uuid.uuid4().hex[:8]}"
    staged.symlink_to(r_root / key, target_is_directory=True)
    os.replace(staged, library)


def restore_r_library(
    notebook_dir: Path, root: Path, restore: Callable[[dict[str, str]], bool]
) -> bool:
    """Link the notebook to the library for its ``renv.lock``, restoring it once.

    *restore* runs ``renv::restore()`` in the notebook with the environment it
    is given; it writes through the link into the keyed directory. A second
    notebook with the same lock and R build only links.
    """
    notebook_dir = Path(notebook_dir)
    r_root = Path(root).resolve() / R_DIR
    build = r_build()
    if build is None:
        return False
    key = r_key(notebook_dir, build)
    r_root.mkdir(parents=True, exist_ok=True)
    library = _r_library(notebook_dir)
    with _key_lock(r_root, key):
        target = r_root / key
        if not (target / COMPLETE_MARKER).exists():
            target.mkdir(exist_ok=True)
            # The restore writes through the link, so the library the notebook
            # has is moved aside rather than removed: a restore that fails
            # (CRAN unreachable, a package that will not build) leaves the
            # notebook with the packages it had this morning.
            previous = None
            if library.exists() and not library.is_symlink():
                previous = library.with_name(f"library.previous-{uuid.uuid4().hex[:8]}")
                os.replace(library, previous)
            _link_r_library(notebook_dir, r_root, key)
            if not restore(r_env(root)):
                if previous is not None:
                    library.unlink(missing_ok=True)
                    os.replace(previous, library)
                return False
            if previous is not None:
                shutil.rmtree(previous, ignore_errors=True)
            (target / COMPLETE_MARKER).touch()
        _link_r_library(notebook_dir, r_root, key)
        os.utime(target / COMPLETE_MARKER)
    return True


def link_r_library_if_built(notebook_dir: Path, root: Path) -> bool:
    """Link the notebook to the built library for its lock, if there is one.

    For a mutation that failed: the notebook goes back to the shared library it
    was using, and when there is none its private one is left alone rather than
    removed — it holds whatever the restore before the failure had installed.
    """
    notebook_dir = Path(notebook_dir)
    build = r_build()
    if build is None or not (notebook_dir / "renv.lock").exists():
        return False
    r_root = Path(root).resolve() / R_DIR
    key = r_key(notebook_dir, build)
    if not (r_root / key / COMPLETE_MARKER).exists():
        return False
    with _key_lock(r_root, key):
        _link_r_library(notebook_dir, r_root, key)
    return True


def detach_r_library(notebook_dir: Path, root: Path) -> bool:
    """Give the notebook a private, empty library in place of a shared one.

    Returns whether it was linked, in which case the caller restores the lock
    into the private library (from the cache) before changing it.
    """
    library = _r_library(Path(notebook_dir))
    if not library.is_symlink():
        return False
    r_root = Path(root).resolve() / R_DIR
    previous = Path(os.readlink(library))
    if previous.parent == r_root:
        (r_root / _REFS / previous.name / _ref_name(Path(notebook_dir))).unlink(missing_ok=True)
    library.unlink()
    library.mkdir()
    return True


def adopt_r_library(notebook_dir: Path, root: Path) -> bool:
    """Move a private library under the key of the ``renv.lock`` it was built
    for, or drop it for the one already there, and link the notebook to it."""
    notebook_dir = Path(notebook_dir)
    library = _r_library(notebook_dir)
    build = r_build()
    if build is None or not (notebook_dir / "renv.lock").exists():
        return False
    if library.is_symlink() or not library.is_dir():
        return False
    r_root = Path(root).resolve() / R_DIR
    key = r_key(notebook_dir, build)
    r_root.mkdir(parents=True, exist_ok=True)
    with _key_lock(r_root, key):
        target = r_root / key
        if not (target / COMPLETE_MARKER).exists():
            shutil.rmtree(target, ignore_errors=True)
            shutil.move(str(library), str(target))
            (target / COMPLETE_MARKER).touch()
        _link_r_library(notebook_dir, r_root, key)
    return True


@dataclass
class Collection:
    """What one sweep removed and what it kept because a notebook links to it."""

    removed: list[str] = field(default_factory=list)
    referenced: list[str] = field(default_factory=list)


def collect(root: Path, *, ttl_days: float, now: float | None = None) -> Collection:
    """Remove environments no notebook links to that have gone unused for
    *ttl_days*. A key some notebook still links to is never removed.

    Python environments and R libraries alike; an R key is reported as
    ``r/<key>``.
    """
    root = Path(root).resolve()
    now = time.time() if now is None else now
    result = Collection()
    _collect(root, Path(".venv"), ttl_days, now, result, prefix="")
    _collect(root / R_DIR, Path("renv") / "library", ttl_days, now, result, prefix=f"{R_DIR}/")
    return result


def _collect(
    root: Path, link: Path, ttl_days: float, now: float, result: Collection, *, prefix: str
) -> None:
    if not root.is_dir():
        return
    skipped = {_REFS, R_DIR} if not prefix else {_REFS, R_CACHE}
    for env_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if env_dir.name in skipped or env_dir.name.startswith("."):
            continue
        key = env_dir.name
        with _key_lock(root, key):
            marker = env_dir / COMPLETE_MARKER
            if not marker.exists():
                # Not an environment this ever finished building. A half-built
                # one is rebuilt in place, and anything else under the root is
                # somebody's directory, not a key.
                continue
            refs = root / _REFS / key
            live = False
            for ref in list(refs.iterdir()) if refs.is_dir() else []:
                try:
                    venv = Path(ref.read_text()) / link
                    points_here = venv.is_symlink() and Path(os.readlink(venv)) == env_dir
                except OSError:
                    # A notebook on a volume that is not mounted right now says
                    # nothing about whether it still links here.
                    live = True
                    continue
                if points_here:
                    live = True
                else:
                    ref.unlink()
            if live:
                result.referenced.append(prefix + key)
                continue
            last_used = marker.stat().st_mtime
            if now - last_used < ttl_days * 86400:
                continue
            shutil.rmtree(env_dir)
            if refs.is_dir():
                shutil.rmtree(refs)
            result.removed.append(prefix + key)

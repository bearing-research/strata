"""A notebook's locked environment on a worker (the manifest's ``environment``).

A server sends the notebook's lock with the cell::

    {"key": "<uv_lock_key of uv.lock>", "python": "3.13",
     "lockfile": "<uv.lock>", "pyproject": "<pyproject.toml>"}

and the worker runs the cell in that exact environment, built once per lock and
interpreter build under ``STRATA_WORKER_ENV_ROOT`` and reused for the same key.
With ``STRATA_WORKER_ENV_REGISTRY_URL`` set, a missing environment is fetched
from ``<registry>/<key>/<interpreter>/<platform>`` as a ``.tar.gz``, and built
locally when the registry answers 404. Workers
advertise support in ``/health`` (``locked_environments``); the server sends
``environment`` only to those.

An R cell's ``environment`` is ``{"key": "<renv_lock_key of renv.lock>",
"lockfile": "<renv.lock>"}``: the worker restores it with renv into one library
per lock and R build under ``<root>/r``, fetched first from
``<registry>/r/<key>/<R version>/<platform>``, and runs ``harness.R`` with that
library first on ``R_LIBS`` (``locked_r_environments``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import filelock
import httpx

from strata.notebook.env import renv_lock_key, uv_lock_key

ENV_ROOT_VAR = "STRATA_WORKER_ENV_ROOT"
REGISTRY_VAR = "STRATA_WORKER_ENV_REGISTRY_URL"
COMPLETE_MARKER = ".strata-env-complete"
INSTALL_TIMEOUT_SECONDS = 900
_KEY = re.compile(r"^[0-9a-f]{64}$")
# The build, ``cpython-3.13.1 linux-x86_64``: the registry path's last two segments.
_PROBE = (
    "import platform, sys, sysconfig; "
    "print(f'{sys.implementation.name}-{platform.python_version()}{sys.abiflags}', "
    "sysconfig.get_platform())"
)


class WorkerEnvironmentError(RuntimeError):
    """The locked environment could not be prepared; the cell cannot run."""


@dataclass(frozen=True)
class PreparedEnvironment:
    python: Path
    key: str
    installed: bool


def env_root() -> Path:
    configured = os.environ.get(ENV_ROOT_VAR)
    return Path(configured) if configured else Path.home() / ".strata" / "worker-envs"


def _validated(spec: Any) -> dict[str, str]:
    if not isinstance(spec, dict):
        raise WorkerEnvironmentError("environment must be an object")
    key = str(spec.get("key") or "")
    if not _KEY.match(key):
        raise WorkerEnvironmentError("environment.key must be a sha256 hex digest")
    lockfile = spec.get("lockfile")
    pyproject = spec.get("pyproject")
    if not isinstance(lockfile, str) or not isinstance(pyproject, str):
        raise WorkerEnvironmentError("environment needs lockfile and pyproject text")
    try:
        matches = uv_lock_key(lockfile) == key
    except tomllib.TOMLDecodeError as exc:
        raise WorkerEnvironmentError(f"environment.lockfile is not TOML: {exc}") from exc
    if not matches:
        raise WorkerEnvironmentError("environment.lockfile does not match environment.key")
    python = spec.get("python")
    return {
        "key": key,
        "lockfile": lockfile,
        "pyproject": pyproject,
        "python": str(python) if python else "",
    }


def _interpreter(python: str) -> tuple[Path, str]:
    """The interpreter uv would use for *python* on this machine, and its build."""
    uv = shutil.which("uv")
    if uv is None:
        raise WorkerEnvironmentError("uv is not installed on this worker")
    args = [uv, "python", "find", "--system", *([python] if python else [])]
    try:
        found = subprocess.run(args, capture_output=True, text=True, check=True, timeout=60)
        interpreter = Path(found.stdout.strip())
        build = subprocess.run(
            [str(interpreter), "-c", _PROBE],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise WorkerEnvironmentError(
            f"no Python {python or ''} on this worker: {(exc.stderr or exc.stdout).strip()}"
        ) from exc
    return interpreter, build


def _install(spec: dict[str, str], interpreter: Path, env_dir: Path) -> None:
    """``uv sync`` the lock into *env_dir*."""
    project = env_dir.with_name(env_dir.name + ".project")
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text(spec["pyproject"])
    (project / "uv.lock").write_text(spec["lockfile"])
    try:
        subprocess.run(
            [
                shutil.which("uv") or "uv",
                "sync",
                "--frozen",
                "--no-install-project",
                "--python",
                str(interpreter),
            ],
            cwd=project,
            env={
                **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
                "UV_PROJECT_ENVIRONMENT": str(env_dir),
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkerEnvironmentError(
            f"uv sync failed on this worker: {exc.stderr.strip()}"
        ) from exc


def _fetch(url: str, env_dir: Path) -> bool:
    """Unpack *url* into *env_dir*; False when the registry has no such environment."""
    with tempfile.TemporaryDirectory(dir=env_dir.parent) as scratch:
        archive = Path(scratch) / "environment.tar.gz"
        try:
            with httpx.stream("GET", url, timeout=INSTALL_TIMEOUT_SECONDS) as response:
                if response.status_code == 404:
                    return False
                response.raise_for_status()
                with open(archive, "wb") as out:
                    for chunk in response.iter_bytes():
                        out.write(chunk)
        except httpx.HTTPError as exc:
            raise WorkerEnvironmentError(f"could not fetch environment from {url}: {exc}") from exc
        unpacked = Path(scratch) / "environment"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        shutil.rmtree(env_dir, ignore_errors=True)
        os.replace(unpacked, env_dir)
    return True


def _prepare(spec: dict[str, str]) -> PreparedEnvironment:
    interpreter, build = _interpreter(spec["python"])
    root = env_root()
    root.mkdir(parents=True, exist_ok=True)
    directory_key = hashlib.sha256(f"{spec['key']}\n{build}".encode()).hexdigest()[:32]
    env_dir = root / directory_key
    lock = filelock.FileLock(str(root / f"{directory_key}.lock"), timeout=INSTALL_TIMEOUT_SECONDS)
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    with lock:
        installed = False
        if not (env_dir / COMPLETE_MARKER).exists():
            registry = os.environ.get(REGISTRY_VAR, "").rstrip("/")
            url = f"{registry}/{spec['key']}/{build.replace(' ', '/')}"
            if not registry or not _fetch(url, env_dir):
                _install(spec, interpreter, env_dir)
            if not python.exists():
                # If marked complete first, an archive with no interpreter would fail every
                # cell and never be fetched again.
                raise WorkerEnvironmentError(
                    f"environment {directory_key} has no interpreter at {python}"
                )
            (env_dir / COMPLETE_MARKER).touch()
            installed = True
    if not python.exists():
        raise WorkerEnvironmentError(f"environment {directory_key} has no interpreter at {python}")
    return PreparedEnvironment(python=python, key=directory_key, installed=installed)


async def ensure_environment(spec: Any) -> PreparedEnvironment:
    """The interpreter to run a cell with, building its environment if needed."""
    return await asyncio.to_thread(_prepare, _validated(spec))


def environment_spec(notebook_dir: Path, python: str | None) -> dict[str, str] | None:
    """What a server sends for *notebook_dir*, or None without a lock."""
    lockfile = notebook_dir / "uv.lock"
    pyproject = notebook_dir / "pyproject.toml"
    if not lockfile.exists() or not pyproject.exists():
        return None
    lock_text = lockfile.read_text()
    return {
        "key": uv_lock_key(lock_text),
        "python": python or "",
        "lockfile": lock_text,
        "pyproject": pyproject.read_text(),
    }


# --- R: one renv library per renv.lock and R build ---

R_DIR = "r"
# The build, ``R-4.4.1 x86_64-pc-linux-gnu``: the registry path's last two segments.
_R_PROBE = 'cat(paste0("R-", R.version$major, ".", R.version$minor), R.version$platform)'
# The library comes from the environment, so no path is quoted into R source.
_R_RESTORE = (
    'renv::restore(lockfile = "renv.lock", library = Sys.getenv("STRATA_R_LIBRARY"), '
    "prompt = FALSE)"
)
_renv_found: dict[str, bool] = {}


@dataclass(frozen=True)
class PreparedRLibrary:
    library: Path
    key: str
    installed: bool


def renv_available(rscript: str) -> bool:
    """Whether *rscript* can load renv, which restoring a lock needs."""
    if rscript not in _renv_found:
        try:
            probe = subprocess.run(
                [rscript, "-e", 'if (!requireNamespace("renv", quietly = TRUE)) quit(status = 1)'],
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            # Unanswered, not "no": a loaded machine must not lose the feature for good.
            return False
        _renv_found[rscript] = probe.returncode == 0
    return _renv_found[rscript]


def _validated_r(spec: Any) -> dict[str, str]:
    if not isinstance(spec, dict):
        raise WorkerEnvironmentError("environment must be an object")
    key = str(spec.get("key") or "")
    if not _KEY.match(key):
        raise WorkerEnvironmentError("environment.key must be a sha256 hex digest")
    lockfile = spec.get("lockfile")
    if not isinstance(lockfile, str):
        raise WorkerEnvironmentError("environment needs the renv.lock text")
    try:
        packages = json.loads(lockfile).get("Packages")
    except (ValueError, AttributeError) as exc:
        raise WorkerEnvironmentError(f"environment.lockfile is not an renv.lock: {exc}") from exc
    if not isinstance(packages, dict):
        raise WorkerEnvironmentError("environment.lockfile has no Packages")
    if renv_lock_key(lockfile) != key:
        raise WorkerEnvironmentError("environment.lockfile does not match environment.key")
    return {"key": key, "lockfile": lockfile}


def _r_build(rscript: str) -> str:
    try:
        return subprocess.run(
            [rscript, "--vanilla", "-e", _R_PROBE],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise WorkerEnvironmentError(
            f"could not ask this worker's R its version: {(exc.stderr or exc.stdout).strip()}"
        ) from exc


def _restore_r(rscript: str, lockfile: str, library: Path) -> None:
    """``renv::restore`` the lock into *library*, with renv's package cache beside it."""
    project = library.with_name(library.name + ".project")
    project.mkdir(parents=True, exist_ok=True)
    (project / "renv.lock").write_text(lockfile)
    library.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [rscript, "-e", _R_RESTORE],
            cwd=project,
            env={
                **os.environ,
                "STRATA_R_LIBRARY": str(library),
                "RENV_PATHS_CACHE": str(library.parent / "cache"),
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkerEnvironmentError(
            f"renv::restore failed on this worker: {(exc.stderr or exc.stdout).strip()}"
        ) from exc


def _prepare_r(spec: dict[str, str], rscript: str) -> PreparedRLibrary:
    build = _r_build(rscript)
    root = env_root() / R_DIR
    root.mkdir(parents=True, exist_ok=True)
    directory_key = hashlib.sha256(f"{spec['key']}\n{build}".encode()).hexdigest()[:32]
    library = root / directory_key
    lock = filelock.FileLock(str(root / f"{directory_key}.lock"), timeout=INSTALL_TIMEOUT_SECONDS)
    with lock:
        installed = False
        if not (library / COMPLETE_MARKER).exists():
            registry = os.environ.get(REGISTRY_VAR, "").rstrip("/")
            url = f"{registry}/{R_DIR}/{spec['key']}/{build.replace(' ', '/')}"
            if not registry or not _fetch(url, library):
                _restore_r(rscript, spec["lockfile"], library)
            # renv installs itself only on activation, so its own entry is not required.
            missing = sorted(
                name
                for name in json.loads(spec["lockfile"])["Packages"]
                if name != "renv" and not (library / name / "DESCRIPTION").exists()
            )
            if missing:
                # Marked complete, a library short of the lock would never be rebuilt.
                raise WorkerEnvironmentError(
                    f"R library {directory_key} lacks {', '.join(missing)} from the lock"
                )
            (library / COMPLETE_MARKER).touch()
            installed = True
    return PreparedRLibrary(library=library, key=directory_key, installed=installed)


async def ensure_r_library(spec: Any, rscript: str) -> PreparedRLibrary:
    """The library to run an R cell against, restoring its lock if needed."""
    return await asyncio.to_thread(_prepare_r, _validated_r(spec), rscript)


def r_environment_spec(notebook_dir: Path) -> dict[str, str] | None:
    """What a server sends for an R cell of *notebook_dir*, or None without ``renv.lock``."""
    lockfile = notebook_dir / "renv.lock"
    if not lockfile.exists():
        return None
    # Bytes, not read_text: newline translation would change the key.
    text = lockfile.read_bytes().decode()
    return {"key": renv_lock_key(text), "lockfile": text}

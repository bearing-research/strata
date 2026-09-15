"""A notebook's locked environment on a worker (the manifest's ``environment``).

A worker used to run every cell with its own interpreter, so a cell that
imported something the image lacked failed, and a cell that imported something
at another version than the notebook's lock computed something else. A server
that knows the worker can do better sends the notebook's lock with the cell:

    {"key": "<sha256 of uv.lock>", "python": "3.13",
     "lockfile": "<uv.lock>", "pyproject": "<pyproject.toml>"}

and the worker runs the cell in that exact environment, built once per lock and
interpreter build under ``STRATA_WORKER_ENV_ROOT`` and reused by every later
cell with the same key. With ``STRATA_WORKER_ENV_REGISTRY_URL`` set, a missing
environment is fetched from ``<registry>/<key>`` as a ``.tar.gz`` of the
environment directory instead of installed.

A worker says it can do this in ``/health`` (``locked_environments``); the
server sends ``environment`` only to workers that do.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import filelock
import httpx

ENV_ROOT_VAR = "STRATA_WORKER_ENV_ROOT"
REGISTRY_VAR = "STRATA_WORKER_ENV_REGISTRY_URL"
COMPLETE_MARKER = ".strata-env-complete"
INSTALL_TIMEOUT_SECONDS = 900
_KEY = re.compile(r"^[0-9a-f]{64}$")
_PROBE = (
    "import platform, sys, sysconfig; "
    "print(sys.implementation.name, platform.python_version(), sys.abiflags, "
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
    if hashlib.sha256(lockfile.encode()).hexdigest() != key:
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


def _fetch(registry: str, key: str, env_dir: Path) -> None:
    """Unpack ``<registry>/<key>`` into *env_dir*."""
    url = f"{registry.rstrip('/')}/{key}"
    with tempfile.TemporaryDirectory(dir=env_dir.parent) as scratch:
        archive = Path(scratch) / "environment.tar.gz"
        try:
            with httpx.stream("GET", url, timeout=INSTALL_TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with open(archive, "wb") as out:
                    for chunk in response.iter_bytes():
                        out.write(chunk)
        except httpx.HTTPError as exc:
            raise WorkerEnvironmentError(
                f"could not fetch environment {key} from {url}: {exc}"
            ) from exc
        unpacked = Path(scratch) / "environment"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        shutil.rmtree(env_dir, ignore_errors=True)
        os.replace(unpacked, env_dir)


def _prepare(spec: dict[str, str]) -> PreparedEnvironment:
    interpreter, build = _interpreter(spec["python"])
    root = env_root()
    root.mkdir(parents=True, exist_ok=True)
    directory_key = hashlib.sha256(f"{spec['key']}\n{build}".encode()).hexdigest()[:32]
    env_dir = root / directory_key
    lock = filelock.FileLock(str(root / f"{directory_key}.lock"), timeout=INSTALL_TIMEOUT_SECONDS)
    with lock:
        installed = False
        if not (env_dir / COMPLETE_MARKER).exists():
            registry = os.environ.get(REGISTRY_VAR)
            if registry:
                _fetch(registry, spec["key"], env_dir)
            else:
                _install(spec, interpreter, env_dir)
            (env_dir / COMPLETE_MARKER).touch()
            installed = True
    python = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
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
        "key": hashlib.sha256(lock_text.encode()).hexdigest(),
        "python": python or "",
        "lockfile": lock_text,
        "pyproject": pyproject.read_text(),
    }

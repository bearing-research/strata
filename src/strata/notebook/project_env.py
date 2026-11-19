"""A scratchpad's environment: its project's when it has one, else one of its own.

The project's environment is found by asking uv with a dry run, which changes nothing;
whether uv manages it is decided by files (a ``uv.lock``, a ``[tool.uv]`` table), since
the dry run succeeds on any ``pyproject.toml``. Cells run in an overlay venv,
``.venv-overlay``, which adds the project's site-packages and only the runtime packages
the project lacks, so the project's versions win.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from strata.notebook.env import ATTACHED_IDENTITY, OVERLAY_DIR

# The interpreter a scratchpad's cells run in instead (``--python``); the daemon reads it.
PYTHON_OVERRIDE = "STRATA_SCRATCH_PYTHON"

if TYPE_CHECKING:
    from strata.notebook.dependencies import _UvCommandResult
    from strata.notebook.env_backend import _StreamCallback

# What the overlay installed, and from which base interpreter.
_STATE = ".strata-overlay.json"
_PTH = "_strata_project.pth"
_TIMEOUT = 600

_PROJECT_PROBE = """\
import importlib.util, json, site, sys
print(json.dumps({
    "base": sys._base_executable,
    "site": site.getsitepackages(),
    "missing": [n for n in sys.argv[1:] if importlib.util.find_spec(n) is None],
}))
"""

_IDENTITY_PROBE = """\
import importlib.metadata as m, json, platform, re, sys, sysconfig

def editable(dist):
    try:
        url = json.loads(dist.read_text("direct_url.json") or "{}")
    except ValueError:
        return False
    return bool(url.get("dir_info", {}).get("editable"))

names = set()
for dist in m.distributions():
    name = dist.metadata["Name"]
    if name and not editable(dist):
        names.add(re.sub(r"[-_.]+", "-", name).lower() + "==" + dist.version)
build = (sys.implementation.name, platform.python_version(), getattr(sys, "abiflags", ""),
         sysconfig.get_platform())
print(json.dumps({"build": " ".join(build), "distributions": sorted(names)}))
"""


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _is_venv(path: Path) -> bool:
    return (path / "pyvenv.cfg").is_file()


@dataclass(frozen=True)
class ProjectEnv:
    """Where a scratchpad's cells run, and what chose it."""

    anchor: Path
    mode: Literal["attached", "owned"]
    # The project environment's interpreter, when attached.
    python: Path | None
    # uv, VIRTUAL_ENV, .venv, STRATA_SCRATCH_PYTHON, or why the scratchpad owns one.
    how: str
    # uv manages it, and its dry run reported changes: `uv sync --inexact` first.
    sync: bool = False
    # The files whose change means detecting again.
    watched: tuple[Path, ...] = field(default=())

    def record(self) -> dict[str, str]:
        """``notebook.toml``'s ``[scratchpad_environment]``."""
        if self.python is None:
            return {"mode": self.mode}
        return {"mode": self.mode, "attach": str(self.python)}


def detect(anchor: Path, environ: dict[str, str] | None = None) -> ProjectEnv:
    """The environment *anchor*'s scratchpad runs in."""
    environ = dict(os.environ) if environ is None else environ
    watched = (anchor / "pyproject.toml", anchor / "uv.lock", anchor / ".venv")
    override = environ.get(PYTHON_OVERRIDE)
    if override:
        return ProjectEnv(anchor, "attached", Path(override), PYTHON_OVERRIDE)

    found = _dry_run(anchor, environ)
    if found is None and _uv_managed(anchor, anchor):
        # uv's, but uv cannot plan it (a broken lock, an unresolvable dependency): the
        # sync reports why, and no cell runs, rather than a fresh environment of its own.
        found = anchor, anchor / ".venv", True
    if found is not None:
        workspace, environment, pending = found
        if _uv_managed(anchor, workspace):
            return ProjectEnv(
                anchor,
                "attached",
                venv_python(environment),
                "uv",
                sync=pending,
                watched=(*watched, *_workspace_files(workspace)),
            )

    active = environ.get("VIRTUAL_ENV")
    # `uv run` and uvx set it to the environment Strata itself runs in, not the project's.
    if active and Path(active).resolve() != Path(sys.prefix).resolve() and _is_venv(Path(active)):
        return ProjectEnv(
            anchor, "attached", venv_python(Path(active)), "VIRTUAL_ENV", watched=watched
        )
    if _is_venv(anchor / ".venv"):
        return ProjectEnv(
            anchor, "attached", venv_python(anchor / ".venv"), ".venv", watched=watched
        )
    return ProjectEnv(anchor, "owned", None, "no project environment", watched=watched)


def _project_uv_vars(environ: dict[str, str]) -> dict[str, str]:
    """The user's own settings for which environment and interpreter uv uses, which
    :func:`~strata.notebook.dependencies.uv_env` drops for a notebook's commands."""
    return {k: environ[k] for k in ("UV_PROJECT_ENVIRONMENT", "UV_PYTHON") if k in environ}


def _dry_run(anchor: Path, environ: dict[str, str]) -> tuple[Path, Path, bool] | None:
    """The workspace root, the environment and whether a sync would change anything, as
    ``uv sync --dry-run`` from *anchor* reports them; ``None`` when uv finds no project."""
    from strata.notebook.dependencies import resolve_uv, uv_env

    uv = resolve_uv()
    if uv is None:
        return None
    try:
        done = subprocess.run(
            [uv, "sync", "--dry-run", "--output-format", "json"],
            cwd=anchor,
            env=uv_env(extra=_project_uv_vars(environ)),
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return parse_dry_run(done.stdout, done.stderr, anchor)


_TEXT_ENVIRONMENT = re.compile(r"^Would (?:use|create) project environment at: (.+)$", re.M)


def parse_dry_run(stdout: str, stderr: str, cwd: Path) -> tuple[Path, Path, bool] | None:
    """What :func:`_dry_run` reads. The JSON is marked preview, so the text lines are the
    fallback; from those the workspace is the environment's parent and a sync always runs."""
    try:
        data = json.loads(stdout)
        workspace = Path(data["project"]["workspace"]["path"])
        environment = Path(data["sync"]["environment"]["path"])
        pending = (
            data["sync"]["action"] != "check"
            or bool(data["sync"].get("changes"))
            or data.get("lock", {}).get("action", "check") != "check"
        )
        return workspace, environment, pending
    except (ValueError, KeyError, TypeError):
        pass
    match = _TEXT_ENVIRONMENT.search(stderr)
    if match is None:
        return None
    environment = cwd / match.group(1).strip()
    return environment.parent, environment, True


def _has_tool_uv(pyproject: Path) -> bool:
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    tool = data.get("tool")
    return isinstance(tool, dict) and isinstance(tool.get("uv"), dict)


def _uv_managed(anchor: Path, workspace: Path) -> bool:
    return (
        (workspace / "uv.lock").is_file()
        or _has_tool_uv(anchor / "pyproject.toml")
        or _has_tool_uv(workspace / "pyproject.toml")
    )


def _workspace_files(workspace: Path) -> list[Path]:
    """The workspace's lock and every member's ``pyproject.toml``, as its lock names them."""
    files = [workspace / "pyproject.toml", workspace / "uv.lock"]
    try:
        with open(workspace / "uv.lock", "rb") as f:
            packages = tomllib.load(f).get("package", [])
    except (OSError, tomllib.TOMLDecodeError):
        return files
    for package in packages if isinstance(packages, list) else []:
        source = package.get("source") if isinstance(package, dict) else None
        if not isinstance(source, dict):
            continue
        for kind in ("editable", "virtual"):
            path = source.get(kind)
            if isinstance(path, str) and path != ".":
                files.append(workspace / path / "pyproject.toml")
    return files


def fingerprint(paths: tuple[Path, ...] | list[Path]) -> tuple[tuple[str, int, int] | None, ...]:
    """Each path's modification time and size, ``None`` for one that is missing."""
    seen: list[tuple[str, int, int] | None] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            seen.append(None)
            continue
        seen.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(seen)


@dataclass
class Overlay:
    """The overlay venv once reconciled with the project environment."""

    python: Path
    # The project's site-packages: a change there (an install) means reconciling again.
    site: list[Path]
    identity: str


def _run(args: list[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout)


def _uv(notebook_dir: Path, args: list[str], what: str) -> str | None:
    """Run uv on the overlay; return its error, if any."""
    from strata.notebook.dependencies import _run_uv_command

    done = _run_uv_command(notebook_dir, args, timeout=_TIMEOUT, display_name=what)
    return None if done.success else done.error


def reconcile(notebook_dir: Path, project_python: Path, runtime: tuple[str, ...]) -> Overlay | str:
    """Bring ``.venv-overlay`` in line with the environment *project_python* runs in: its
    site-packages, plus those of *runtime* it lacks. Returns the overlay, or why not."""
    overlay = notebook_dir / OVERLAY_DIR
    # Gone until the overlay holds again, so a failure never leaves the old one keying cells.
    (overlay / ATTACHED_IDENTITY).unlink(missing_ok=True)
    try:
        probed: dict[str, Any] = json.loads(
            _run([str(project_python), "-c", _PROJECT_PROBE, *runtime]).stdout
        )
    except FileNotFoundError:
        return f"the project's interpreter {project_python} does not exist"
    except subprocess.CalledProcessError as exc:
        return f"the project's interpreter {project_python} failed: {exc.stderr.strip()}"
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return f"the project's interpreter {project_python} could not be asked: {exc}"

    state_path = overlay / _STATE
    try:
        state = json.loads(state_path.read_text())
    except (OSError, ValueError):
        state = {}
    python = venv_python(overlay)
    if state.get("base") != probed["base"] or not python.exists():
        shutil.rmtree(overlay, ignore_errors=True)
        state = {}
        error = _uv(
            notebook_dir,
            ["venv", "--no-project", "--python", probed["base"], str(overlay)],
            "uv venv",
        )
        if error is not None:
            return error

    held = set(state.get("installed", []))
    missing = set(probed["missing"])
    # One the project has now would shadow the project's own version.
    if held - missing:
        error = _uv(
            notebook_dir,
            ["pip", "uninstall", "--python", str(python), *sorted(held - missing)],
            "uv pip uninstall",
        )
        if error is not None:
            return error
    if missing - held:
        # Bare names: a version floor could shadow nothing, but would refuse a project pin.
        error = _uv(
            notebook_dir,
            ["pip", "install", "--python", str(python), *sorted(missing - held)],
            "uv pip install",
        )
        if error is not None:
            return error

    try:
        own_site = _run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]
        ).stdout.strip()
        # After the overlay's own site-packages, so it fills gaps and shadows nothing.
        lines = [f"import site; site.addsitedir({str(s)!r})\n" for s in probed["site"]]
        (Path(own_site) / _PTH).write_text("".join(lines))
        state_path.write_text(
            json.dumps({"base": probed["base"], "installed": sorted(missing)}, indent=2) + "\n"
        )
        identity = _identity(python)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return f"the scratchpad's overlay environment could not be set up: {exc}"
    (overlay / ATTACHED_IDENTITY).write_text(identity)
    return Overlay(python=python, site=[Path(s) for s in probed["site"]], identity=identity)


def _identity(python: Path) -> str:
    """The environment's distributions (editable ones are code, keyed by what runs read)
    and interpreter build, as *python* sees them."""
    probed = json.loads(_run([str(python), "-c", _IDENTITY_PROBE]).stdout)
    return hashlib.sha256(json.dumps(probed, sort_keys=True).encode()).hexdigest()


def remove_overlay(notebook_dir: Path) -> None:
    """Drop the overlay of a scratchpad that now owns its environment."""
    shutil.rmtree(notebook_dir / OVERLAY_DIR, ignore_errors=True)


def _refused(command: str, error: str) -> _UvCommandResult:
    from strata.notebook.dependencies import EnvironmentOperationLog, _UvCommandResult

    return _UvCommandResult(
        success=False, error=error, operation_log=EnvironmentOperationLog(command=command)
    )


class ProjectEnvBackend:
    """A scratchpad that runs in its project's environment, through the overlay.

    The environment is the project's to change, so this changes nothing in it but what
    ``uv run`` would: a uv-managed project is synced inexactly when uv says it is behind.
    """

    name = "project"
    supports_mutations = False

    def __init__(self, notebook_dir: Path, env: ProjectEnv | None = None) -> None:
        self.notebook_dir = Path(notebook_dir)
        self._env = env
        # The project's site-packages, as the last sync found them.
        self._site: list[Path] = []
        self._seen: tuple[Any, ...] | None = None

    @property
    def env(self) -> ProjectEnv:
        if self._env is None:
            self._env = detect(self.notebook_dir.parent)
        return self._env

    def python_executable(self) -> Path:
        return venv_python(self.notebook_dir / OVERLAY_DIR)

    def changed(self) -> bool:
        """Whether the project's files or installed packages changed since the last sync,
        or none succeeded yet."""
        return self._seen is None or self._seen != self._observe()

    def _observe(self) -> tuple[Any, ...]:
        return fingerprint(self.env.watched), fingerprint(self._site)

    def sync(self, *, python_version: str | None, timeout: int) -> _UvCommandResult:
        """Sync a uv project that is behind, then reconcile the overlay with it.

        *python_version* is the scratchpad's own pin, which the project's interpreter
        replaces.
        """
        from strata.notebook.dependencies import EnvironmentOperationLog, _UvCommandResult
        from strata.notebook.writer import _NOTEBOOK_RUNTIME_PACKAGES

        env = self.env
        assert env.python is not None  # attached
        synced = None
        if env.sync:
            from strata.notebook.dependencies import _run_uv_command

            synced = _run_uv_command(
                env.anchor,
                ["sync", "--inexact"],
                timeout=timeout,
                display_name="uv sync",
                env=_project_uv_vars(dict(os.environ)),
            )
            if not synced.success:
                return synced
        overlay = reconcile(self.notebook_dir, env.python, _NOTEBOOK_RUNTIME_PACKAGES)
        if isinstance(overlay, str):
            return _refused("uv sync", overlay)
        self._site = overlay.site
        self._seen = self._observe()
        return synced or _UvCommandResult(
            success=True, error=None, operation_log=EnvironmentOperationLog(command="uv sync")
        )

    def _mutation(self, verb: str, package: str) -> _UvCommandResult:
        env = self.env
        if env.how == "uv":
            fix = f"`uv {verb} {package}` in {env.anchor}"
        else:
            fix = f"{verb} it there ({env.python})"
        return _refused(
            f"uv {verb}",
            f"this scratchpad runs in your project's environment; {verb} it in the project: {fix}",
        )

    def add(self, package: str, *, timeout: int, dev: bool = False) -> _UvCommandResult:
        return self._mutation("add", package)

    def remove(self, package: str, *, timeout: int) -> _UvCommandResult:
        return self._mutation("remove", package)

    async def sync_streaming(
        self,
        *,
        python_version: str | None,
        timeout: int,
        on_update: _StreamCallback | None,
    ) -> _UvCommandResult:
        import asyncio

        return await asyncio.to_thread(self.sync, python_version=python_version, timeout=timeout)

    async def add_streaming(
        self, package: str, *, timeout: int, on_update: _StreamCallback | None
    ) -> _UvCommandResult:
        return self._mutation("add", package)

    async def remove_streaming(
        self, package: str, *, timeout: int, on_update: _StreamCallback | None
    ) -> _UvCommandResult:
        return self._mutation("remove", package)

    async def lock_streaming(
        self, *, timeout: int, on_update: _StreamCallback | None
    ) -> _UvCommandResult:
        return _refused(
            "uv lock",
            "this scratchpad runs in your project's environment; its lock is the project's",
        )

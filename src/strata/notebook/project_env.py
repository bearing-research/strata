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
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from strata.notebook.env import ATTACHED_IDENTITY, OVERLAY_DIR, OVERLAY_STATE

# The interpreter a scratchpad's cells run in instead (``--python``); the daemon reads it.
PYTHON_OVERRIDE = "STRATA_SCRATCH_PYTHON"

if TYPE_CHECKING:
    from strata.notebook.dependencies import _UvCommandResult
    from strata.notebook.env_backend import _StreamCallback

_PTH = "_strata_project.pth"
_TIMEOUT = 600
# What the identity was probed from, and what to observe to know it still holds.
_OBSERVED = ".strata-observed.json"
# A probe imports the warm list (torch, say), so it may take a while: all its rounds.
_PROBE_TIMEOUT = 300
# Each round with a dropped warm entry shortens the list; one this unsettled loads none.
_WARM_ROUNDS = 5
# An owned scratchpad's E, beside its own ``.venv`` (which ``uv sync`` may rebuild).
_OWNED = Path(".strata") / "environment"
_HARNESS = Path(__file__).parent / "harness.py"

logger = logging.getLogger(__name__)

# Each runtime package's installed version: None for one missing, "" for one importable
# with no metadata to say.
_PROJECT_PROBE = """\
import importlib.metadata, importlib.util, json, site, sys

def version(name):
    if importlib.util.find_spec(name) is None:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""

print(json.dumps({
    "base": sys._base_executable,
    "site": site.getsitepackages(),
    "versions": {n: version(n) for n in sys.argv[1:]},
}))
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
        """What :func:`write_record` keeps of it."""
        if self.python is None:
            return {"mode": self.mode, "found_by": self.how}
        return {"mode": self.mode, "attach": str(self.python), "found_by": self.how}


# Runtime state: it follows the machine and the launch, so not ``notebook.toml``.
_RECORD = Path(".strata") / "scratchpad_environment.json"


def write_record(notebook_dir: Path, env: ProjectEnv) -> None:
    """Record where the scratchpad's cells run, for the next open and for reports."""
    record = env.record()
    if read_record(notebook_dir) == record:
        return
    path = notebook_dir / _RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(partial, path)


def read_record(notebook_dir: Path) -> dict[str, str] | None:
    try:
        return json.loads((Path(notebook_dir) / _RECORD).read_text())
    except FileNotFoundError:
        return None


def detect(anchor: Path, environ: dict[str, str] | None = None) -> ProjectEnv:
    """The environment *anchor*'s scratchpad runs in."""
    environ = dict(os.environ) if environ is None else environ
    watched = (anchor / "pyproject.toml", anchor / "uv.lock", anchor / ".venv")
    override = environ.get(PYTHON_OVERRIDE)
    if override:
        return ProjectEnv(anchor, "attached", Path(override), PYTHON_OVERRIDE)

    found = _dry_run(anchor, environ)
    if found is None:
        # uv's, but uv cannot plan it (a broken lock, an unresolvable dependency): the
        # sync reports why, and no cell runs, rather than a fresh environment of its own.
        workspace = _locked_root(anchor)
        if _uv_managed(anchor, workspace):
            found = workspace, workspace / environ.get("UV_PROJECT_ENVIRONMENT", ".venv"), True
    if found is not None:
        workspace, environment, pending = found
        if _uv_managed(anchor, workspace):
            return ProjectEnv(
                anchor,
                "attached",
                venv_python(environment),
                "uv",
                sync=pending,
                watched=tuple(dict.fromkeys((*watched, *_workspace_files(workspace)))),
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


def _locked_root(anchor: Path) -> Path:
    """The nearest directory at or above *anchor* with a ``uv.lock``, within its repository
    and below home; else *anchor*."""
    home = Path.home()
    for directory in [anchor, *anchor.parents]:
        if (directory / "uv.lock").is_file():
            return directory
        if (directory / ".git").exists() or directory == home:
            break
    return anchor


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
            # Inexact, as the sync it plans: what the lock lacks is not a change.
            [uv, "sync", "--dry-run", "--inexact", "--output-format", "json"],
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


def reconcile(notebook_dir: Path, project_python: Path, runtime: list[str]) -> Overlay | str:
    """Bring ``.venv-overlay`` in line with the environment *project_python* runs in: its
    site-packages, plus the *runtime* requirements it lacks. Returns the overlay, or why
    not, which is also a project version *runtime* rules out: it cannot be shadowed."""
    # Under E's lock: a key checked meanwhile would probe a half-built overlay.
    with _identity_lock(notebook_dir, owned=False):
        return _reconcile(notebook_dir, project_python, runtime)


def _reconcile(notebook_dir: Path, project_python: Path, runtime: list[str]) -> Overlay | str:
    from packaging.requirements import Requirement
    from packaging.version import InvalidVersion

    from strata.notebook.harness_user import LocalExecutionRefused

    overlay = notebook_dir / OVERLAY_DIR
    # Gone until the overlay holds again, so a failure never leaves the old one keying cells.
    (overlay / ATTACHED_IDENTITY).unlink(missing_ok=True)
    wanted = {r.name: r for r in map(Requirement, runtime)}
    try:
        # The last line: what a ``.pth`` prints at startup comes before it.
        out = _run([str(project_python), "-c", _PROJECT_PROBE, *wanted]).stdout.strip()
        probed: dict[str, Any] = json.loads(out.splitlines()[-1] if out else "")
    except FileNotFoundError:
        return f"the project's interpreter {project_python} does not exist"
    except subprocess.CalledProcessError as exc:
        return f"the project's interpreter {project_python} failed: {exc.stderr.strip()}"
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return f"the project's interpreter {project_python} could not be asked: {exc}"

    state_path = overlay / OVERLAY_STATE
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

    for name, found in probed["versions"].items():
        try:
            too_old = bool(found) and not wanted[name].specifier.contains(found, prereleases=True)
        except InvalidVersion:
            too_old = False
        if too_old:
            return (
                f"the project's {name} is {found}, and the scratchpad needs {wanted[name]}; "
                "upgrade it in the project"
            )
    held = set(state.get("installed", []))
    missing = {name for name, found in probed["versions"].items() if found is None}
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
        # No dependencies: uv cannot see the project's, and one installed here would
        # shadow the project's own (the runtime packages need none).
        error = _uv(
            notebook_dir,
            [
                "pip",
                "install",
                "--no-deps",
                "--python",
                str(python),
                *(str(wanted[name]) for name in sorted(missing - held)),
            ],
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
        state = {"base": probed["base"], "installed": sorted(missing)}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        identity = _probe_identity(notebook_dir, probed["site"])
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        LocalExecutionRefused,
    ) as exc:
        return f"the scratchpad's overlay environment could not be set up: {exc}"
    return Overlay(python=python, site=[Path(s) for s in probed["site"]], identity=identity)


def _identity_dir(notebook_dir: Path, owned: bool) -> Path:
    return notebook_dir / (_OWNED if owned else OVERLAY_DIR)


def _cell_python(notebook_dir: Path, owned: bool) -> Path:
    return venv_python(notebook_dir / (".venv" if owned else OVERLAY_DIR))


def _cell_env() -> tuple[dict[str, str], dict[str, Any]]:
    """The environment a cell starts with, and how it is spawned: as the cells' user."""
    from strata.notebook.harness_env import configured_allowlist, harness_env
    from strata.notebook.harness_user import identity_env, resolve_harness_user, spawn_kwargs

    user = resolve_harness_user()
    env = identity_env(harness_env(configured_allowlist()), user)
    return dict(os.environ if env is None else env), spawn_kwargs(user)


def _probe_identity(notebook_dir: Path, sites: list[str], *, owned: bool = False) -> str:
    """Probe E in the cells' interpreter (the overlay, or an owned scratchpad's ``.venv``),
    as a cell started there would find it, and record it with what proves it. The old
    identity goes first and the new one comes last, so a failure leaves none keying cells.
    """
    overlay = _identity_dir(notebook_dir, owned)
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / ATTACHED_IDENTITY).unlink(missing_ok=True)
    import time

    env, spawn = _cell_env()
    warm: list[str] | None = None
    # One deadline for every round: each waits under E's lock, which every key takes.
    deadline = time.monotonic() + _PROBE_TIMEOUT
    # A warm entry that loads tracked code is dropped; a later one may have loaded it too,
    # unseen while the dropped one held it, so the rest are tried again in a fresh process.
    # A list still unsettled after that many rounds is given up: none always settles.
    for round_ in range(_WARM_ROUNDS + 1):
        if round_ == _WARM_ROUNDS:
            warm = []
        done = subprocess.run(
            [
                str(_cell_python(notebook_dir, owned)),
                str(_HARNESS),
                "--probe-environment",
                str(notebook_dir.parent),
                "-" if warm is None else json.dumps(warm),
                *sites,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=max(deadline - time.monotonic(), 1.0),
            cwd=notebook_dir,
            env=env,
            **spawn,
        )
        # The last line: what a ``.pth`` or an import prints comes before it.
        lines = done.stdout.strip().splitlines()
        probed = json.loads(lines[-1] if lines else "")
        if not probed["warm_dropped"]:
            break
        warm = probed["warm"]
    identity = hashlib.sha256(json.dumps(probed["identity"], sort_keys=True).encode()).hexdigest()
    record = {
        "sites": sites,
        "variables": probed["identity"]["variables"],
        "observed": probed["observed"],
        "anchor_listing": probed["anchor_listing"],
        "base_names": probed["base_names"],
        "base": probed["identity"]["base"],
        "roots": probed["roots"],
        "warm": probed["warm"],
    }
    _write(overlay / _OBSERVED, json.dumps(record, indent=2) + "\n")
    _write(overlay / ATTACHED_IDENTITY, identity)
    _failed.pop((Path(notebook_dir), owned), None)
    return identity


def _write(path: Path, text: str) -> None:
    import tempfile

    fd, partial = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def read_observed(notebook_dir: Path, *, owned: bool = False) -> dict[str, Any] | None:
    try:
        found = json.loads((_identity_dir(notebook_dir, owned) / _OBSERVED).read_text())
    except (OSError, ValueError):
        return None
    return found if isinstance(found, dict) else None


_identity_locks: dict[tuple[Path, bool], threading.Lock] = {}
_identity_locks_lock = threading.Lock()
# A probe that failed, per notebook: what it was tried against, the identity it gave, when.
_failed: dict[tuple[Path, bool], tuple[str, bytes, float]] = {}
# How long a failure stands with nothing it could see changed, before one more try.
_RETRY_SECONDS = 300.0


def _owned(notebook_dir: Path) -> bool:
    record = read_record(notebook_dir)
    return record is not None and record.get("mode") == "owned"


def environment_record(notebook_dir: Path) -> dict[str, Any] | None:
    """What a scratchpad's E was last probed from: the attached overlay's, or the owned
    environment's."""
    return read_observed(notebook_dir, owned=_owned(notebook_dir))


def base_image(notebook_dir: Path) -> tuple[str, list[str]] | None:
    """What a scratchpad's warm worker spawned now is built from: E as last found (its
    stamp) and the warm list probed with it; ``None`` before any probe, or for a notebook.
    """
    stamp = _identity_dir(notebook_dir, _owned(notebook_dir)) / ATTACHED_IDENTITY
    try:
        # The identity, the list, the identity: a probe writes the list then the
        # identity, so one between the reads cannot pair its list with the old identity.
        identity = stamp.read_text()
        record = environment_record(notebook_dir)
        again = stamp.read_text()
    except OSError:
        return None
    if record is None or again != identity:
        return None
    return identity, [str(m) for m in record.get("warm") or []]


def base_image_holds(notebook_dir: Path, stamp: str | None) -> bool:
    """Whether a warm worker built on *stamp* may run a cell now (design 4.2): E is the
    one it was built on, and that E's base record B still holds. Stats, so off the loop.
    """
    from strata.notebook.code_reads import base_candidates

    found = base_image(notebook_dir)
    if found is None or stamp is None:
        # Built with no E, it may run only while there is none.
        return found is None and stamp is None
    if found[0] != stamp:
        return False
    record = environment_record(notebook_dir) or {}
    base = record.get("base") or {}
    now = base_candidates(str(notebook_dir.parent), record.get("base_names") or [])
    if all(base.get(path) == seen for path, seen in now.items()):
        return True
    # E's B no longer holds: found again now, so the replacement is built on the new one.
    current_identity(notebook_dir, owned=_owned(notebook_dir))
    return False


def _identity_lock(notebook_dir: Path, *, owned: bool) -> threading.Lock:
    with _identity_locks_lock:
        return _identity_locks.setdefault((Path(notebook_dir), owned), threading.Lock())


def owned_identity(notebook_dir: Path, *, check: bool = True) -> bytes | None:
    """An owned scratchpad's E, which its ``uv.lock`` does not hold (where imports resolve,
    what would shadow a preloaded module); ``None`` for any other notebook, and for one
    whose environment is not built yet, where no cell runs. Without *check*, the last one
    found, never probing (on the event loop)."""
    record = read_record(notebook_dir)
    if record is None or record.get("mode") != "owned":
        return None
    if not _cell_python(notebook_dir, owned=True).exists():
        return None
    if not check:
        try:
            return (_identity_dir(notebook_dir, True) / ATTACHED_IDENTITY).read_bytes()
        except OSError:
            return None
    return current_identity(notebook_dir, owned=True)


def current_identity(notebook_dir: Path, *, owned: bool = False) -> bytes:
    """E, once its observations show it still holds, else probed again (design 4.2):
    before any key is computed, so a changed environment never meets an old result.
    It may start a probe, so never on the event loop.

    A probe that fails gives an identity nothing matches: a rerun, never a stale hit. The
    same one while nothing it could see changes, so the keys it gives agree, and no probe
    for a while.
    """
    import time

    from strata.notebook.harness_user import LocalExecutionRefused

    key = (Path(notebook_dir), owned)
    where = _identity_dir(notebook_dir, owned)
    with _identity_lock(notebook_dir, owned=owned):
        env, _ = _cell_env_or_none()
        record = read_observed(notebook_dir, owned=owned)
        if env is not None and record is not None and _holds(notebook_dir, record, where, env):
            try:
                found = (where / ATTACHED_IDENTITY).read_bytes()
            except OSError:
                pass
            else:
                _failed.pop(key, None)
                return found
        state = _state(notebook_dir, record, env)
        before = _failed.get(key)
        if before is not None and before[0] == state:
            if time.monotonic() - before[2] < _RETRY_SECONDS:
                return before[1]
        sites = list(record.get("sites") or []) if record is not None else []
        try:
            if env is None:
                raise LocalExecutionRefused("no cell may start here")
            identity = _probe_identity(notebook_dir, sites, owned=owned).encode()
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
            LocalExecutionRefused,
        ) as exc:
            logger.warning("Could not probe %s's environment: %s", notebook_dir, exc)
            token = (
                before[1]
                if before is not None and before[0] == state
                else b"unprobed:" + os.urandom(16).hex().encode()
            )
            _failed[key] = (state, token, time.monotonic())
            return token
        _failed.pop(key, None)
        return identity


def _cell_env_or_none() -> tuple[dict[str, str] | None, dict[str, Any]]:
    from strata.notebook.harness_user import LocalExecutionRefused

    try:
        return _cell_env()
    except LocalExecutionRefused:
        # No cell can start, nor the probe; nothing holds.
        return None, {}


def _state(notebook_dir: Path, record: dict[str, Any] | None, env: dict[str, str] | None) -> str:
    """All a probe could see change, observed now: what a failed one is remembered by."""
    from strata.notebook.code_reads import IMPORT_VARIABLES, base_candidates, observe, watch_state

    anchor = str(notebook_dir.parent)
    record = record or {}
    return json.dumps(
        [
            None if env is None else {k: env.get(k) for k in IMPORT_VARIABLES},
            {p: watch_state(p, mode) for p, (mode, _) in (record.get("observed") or {}).items()},
            observe(anchor, listing=True)[0],
            base_candidates(anchor, record.get("base_names") or []),
        ],
        sort_keys=True,
    )


def _holds(notebook_dir: Path, record: dict[str, Any], where: Path, env: dict[str, str]) -> bool:
    """Whether what E was probed from is as it was: a few hundred stats and small reads."""
    from strata.notebook.code_reads import (
        IMPORT_VARIABLES,
        base_candidates,
        base_names_to_check,
        observe,
        watch_state,
    )

    # The cells' environment, not the server's: an allowlist may leave a variable out.
    if {k: env.get(k) for k in IMPORT_VARIABLES} != record.get("variables"):
        return False
    for path, (mode, seen) in (record.get("observed") or {}).items():
        if watch_state(path, mode) != seen:
            return False
    anchor = str(notebook_dir.parent)
    names = record.get("base_names") or []
    base = record.get("base") or {}
    listing = observe(anchor, listing=True)[0]
    moved = listing != record.get("anchor_listing")
    if not moved:
        # Unchanged names; what is already there can change within (an edit, a new
        # ``__init__.py`` in a folder).
        names = base_names_to_check(anchor, names, base)
    if any(base.get(p) != seen for p, seen in base_candidates(anchor, names).items()):
        return False
    if moved:
        # Something in the project folder changed, but nothing that shadows.
        record["anchor_listing"] = listing
        try:
            _write(where / _OBSERVED, json.dumps(record, indent=2) + "\n")
        except OSError as exc:
            logger.debug("Could not update %s's observations: %s", notebook_dir, exc)
    return True


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
        """Detected when first needed, which runs uv: off the event loop."""
        if self._env is None:
            self._env = detect(self.notebook_dir.parent)
        return self._env

    def described(self) -> dict[str, str]:
        """Where its cells run, as last detected; never detects."""
        if self._env is not None:
            return self._env.record()
        return read_record(self.notebook_dir) or {"mode": "attached"}

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
        from strata.notebook.writer import _notebook_runtime_specifiers

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
        overlay = reconcile(self.notebook_dir, env.python, _notebook_runtime_specifiers())
        if isinstance(overlay, str):
            return _refused("uv sync", overlay)
        self._site = overlay.site
        self._seen = self._observe()
        return synced or _UvCommandResult(
            success=True, error=None, operation_log=EnvironmentOperationLog(command="uv sync")
        )

    def _mutation(self, verb: str, package: str) -> _UvCommandResult:
        described = self.described()
        if described.get("found_by") == "uv":
            fix = f"`uv {verb} {package}` in {self.notebook_dir.parent}"
        elif described.get("attach"):
            fix = f"{verb} it there ({described['attach']})"
        else:
            fix = f"{verb} it with the tool that manages it"
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

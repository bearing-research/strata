"""Find ``uv``, and refuse to start without one.

Notebooks shell out to ``uv`` for per-notebook venvs, so a server or command
with no uv to run would fail later with a confusing subprocess error. Any
environment will do: uv is a dependency, so pip, pipx and conda installs bring it.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

_UV_EXE = "uv" + (sysconfig.get_config_var("EXE") or "")


def resolve_uv(*, users_project: bool = False) -> str | None:
    """Locate ``uv``: ``STRATA_UV``, the uv installed with Strata, then the user's uv.

    The bundled uv comes first so Strata runs the uv it was tested with. A project
    the user manages (*users_project*) is the exception: their uv wrote its lock,
    and a ``[tool.uv] required-version`` may pin it. A ``STRATA_UV`` that names no
    file is no uv: nothing else stands in for the one asked for.
    """
    override = os.environ.get("STRATA_UV")
    if override:
        return override if Path(override).is_file() else None
    if users_project and (found := _users_uv()):
        return found
    return _bundled_uv() or _users_uv()


def _bundled_uv() -> str | None:
    """The binary of the ``uv`` dependency, wherever its install put it."""
    try:
        from uv import find_uv_bin
    except ImportError:
        # conda-forge's uv package is the binary alone: in the environment's scripts dir,
        # or its Library/bin on Windows.
        for directory in (sysconfig.get_path("scripts"), Path(sys.prefix) / "Library" / "bin"):
            candidate = Path(directory) / _UV_EXE
            if candidate.is_file():
                return str(candidate)
        return None
    try:
        return find_uv_bin()
    except FileNotFoundError:
        return None


def _users_uv() -> str | None:
    """uv on PATH, else in uv's installer dirs: non-login shells (``ssh host 'strata
    run ...'``, cron) often lack ``~/.local/bin`` and ``~/.cargo/bin`` on PATH."""
    found = shutil.which("uv")
    if found:
        return found
    for base in ("~/.local/bin", "~/.cargo/bin"):
        candidate = Path(base).expanduser() / _UV_EXE
        if candidate.is_file():
            return str(candidate)
    return None


def assert_uv_available() -> None:
    """Exit with status 1 if no uv can be found."""
    if resolve_uv() is not None:
        return
    override = os.environ.get("STRATA_UV")
    if override:
        print(f"error: STRATA_UV is {override}, which is not a file.", file=sys.stderr)
        raise SystemExit(1)
    print(
        "error: Strata needs uv, and none was found.\n"
        "  The uv installed with Strata is missing, and no uv is on PATH or in\n"
        "  ~/.local/bin or ~/.cargo/bin.\n"
        f"  Current Python: {sys.executable}\n\n"
        "Reinstall strata-notebook (its uv dependency brings the binary), or set\n"
        "STRATA_UV to a uv binary (https://docs.astral.sh/uv/).",
        file=sys.stderr,
    )
    raise SystemExit(1)

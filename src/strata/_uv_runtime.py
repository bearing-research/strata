"""Guard that Strata runs inside a uv-managed Python env.

Notebooks shell out to ``uv`` for per-notebook venvs, so we refuse to start
elsewhere rather than fail later with a confusing subprocess error. Detection
reads the ``uv = <version>`` line uv writes into ``<sys.prefix>/pyvenv.cfg``.
"""

from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path

from uv import find_uv_bin

_UV_MARKER_PREFIX = "uv = "


def resolve_uv(*, users_project: bool = False) -> str | None:
    """Locate ``uv``: ``STRATA_UV``, the uv package's binary, then the user's uv.

    The package's binary (``find_uv_bin``: where an install of the ``uv`` wheel put
    it) comes first so Strata runs the uv it was tested with. A project the user
    manages (*users_project*) is the exception: their uv wrote its lock, and a
    ``[tool.uv] required-version`` may pin it.
    """
    override = os.environ.get("STRATA_UV")
    if override:
        return override
    if users_project and (found := _users_uv()):
        return found
    try:
        return find_uv_bin()
    except FileNotFoundError:
        # The uv package is installed but not its binary where find_uv_bin looks.
        return _users_uv()


def _users_uv() -> str | None:
    """uv on PATH, else in uv's installer dirs: non-login shells (``ssh host 'strata
    run ...'``, cron) often lack ``~/.local/bin`` and ``~/.cargo/bin`` on PATH."""
    found = shutil.which("uv")
    if found:
        return found
    name = "uv" + (sysconfig.get_config_var("EXE") or "")
    for base in ("~/.local/bin", "~/.cargo/bin"):
        candidate = Path(base).expanduser() / name
        if candidate.is_file():
            return str(candidate)
    return None


def is_uv_managed_runtime() -> bool:
    """Return True if the current Python is a uv-created virtual env."""
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if not cfg.exists():
        return False
    for line in cfg.read_text().splitlines():
        if line.strip().startswith(_UV_MARKER_PREFIX):
            return True
    return False


def assert_uv_managed_runtime() -> None:
    """Exit with status 1 if not launched from a uv-managed env."""
    if is_uv_managed_runtime():
        return
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        # A venv, but not one uv made: python -m venv, virtualenv, pip.
        found = f"{cfg} has no `uv = ` line, so uv did not create this environment"
    else:
        # conda and a bare interpreter have no pyvenv.cfg to point at.
        found = "not a virtual environment (conda or a system Python)"
    print(
        "error: Strata runs only inside a uv-managed Python environment;\n"
        "plain `pip install` is not supported.\n"
        f"  Current Python: {sys.executable}\n"
        f"  {found}\n\n"
        "Install uv (https://docs.astral.sh/uv/), then either\n"
        "  uv tool install strata-notebook && strata-notebook\n"
        "(a released version, in its own env; conda and pip environments are\n"
        "left as they are), or from a checkout\n"
        "  uv sync && uv run strata-notebook",
        file=sys.stderr,
    )
    raise SystemExit(1)

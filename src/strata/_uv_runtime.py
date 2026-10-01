"""Guard that Strata runs inside a uv-managed Python env.

Notebooks shell out to ``uv`` for per-notebook venvs, so we refuse to start
elsewhere rather than fail later with a confusing subprocess error. Detection
reads the ``uv = <version>`` line uv writes into ``<sys.prefix>/pyvenv.cfg``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_UV_MARKER_PREFIX = "uv = "


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

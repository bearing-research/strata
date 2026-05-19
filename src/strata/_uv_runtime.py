"""Strata is only supported when launched from a uv-managed Python env.

The notebook subsystem shells out to ``uv`` to manage per-notebook venvs
(see ``strata.notebook.env_backend.UvBackend``), and the rest of the
project's dev workflow assumes uv as the install path. We refuse to
start outside a uv-managed runtime rather than fail later with a
confusing subprocess error.

Detection looks at ``<sys.prefix>/pyvenv.cfg`` for the ``uv = <version>``
line that uv writes when it creates a venv. ``uv run`` and ``uvx`` both
produce envs with this marker.
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
    print(
        "error: Strata requires a uv-managed Python environment.\n"
        f"  Current Python: {sys.executable}\n"
        f"  Looked for a `uv = ...` line in: {cfg}\n\n"
        "Install via `uv sync` (project dev) or `uvx strata` (runtime),\n"
        "then re-launch Strata from that environment.",
        file=sys.stderr,
    )
    raise SystemExit(1)

"""The script the suite puts where a notebook's interpreter goes."""

from __future__ import annotations

import shlex
import shutil
import sys
from pathlib import Path


def write_interpreter_shim(path: Path, env: dict[str, str] | None = None) -> None:
    """Make *path* a script that runs this interpreter on its arguments, with *env* added.

    A wrapper, not a symlink: Python started through a symlink looks for pyvenv.cfg beside
    the symlink, finds none, and comes up as the base interpreter without this venv's packages.
    """
    # dash, not /bin/sh: on macOS that is bash 3.2, which reads its script from descriptor
    # 255 and closes it on exec, so a pipe the spawner passed the harness there is gone.
    shell = shutil.which("dash") or "/bin/sh"
    assignments = "".join(f"{name}={shlex.quote(value)} " for name, value in (env or {}).items())
    path.write_text(
        f'#!{shell}\n{assignments}exec {shlex.quote(sys.executable)} "$@"\n', encoding="utf-8"
    )
    path.chmod(0o755)

"""A cell runs in its notebook's ``work/``, not beside ``notebook.toml``.

Loaded by path into the harnesses, so stdlib only. A harness starts in the notebook's
directory and moves into ``work/`` just before the cell's code: a cell writes its files
there, and a file it wrote earlier is there to read back.
"""

from __future__ import annotations

import os
from typing import Any


def enter(manifest: dict[str, Any]) -> None:
    """Change to the manifest's ``work_dir``, when it names one.

    A server names it relative to the notebook's directory, where its harnesses start, so
    no directory above the notebook needs to be searchable by the cell's user.
    """
    work_dir = manifest.get("work_dir")
    if work_dir:
        os.chdir(work_dir)


def hint(exc: BaseException, manifest: dict[str, Any]) -> str | None:
    """For a relative path missing from ``work/`` but beside the notebook, say ``../<path>``.

    Only for a server's run, whose ``work/`` is in the notebook's directory; looked up
    relatively, like ``enter``. Added to the exception as a note, so the traceback
    carries it too.
    """
    if not manifest.get("notebook_dir") or not isinstance(exc, FileNotFoundError):
        return None
    if not exc.filename:
        return None
    name = os.fsdecode(exc.filename)
    if os.path.isabs(name) or os.path.exists(name):
        return None
    if not os.path.exists(os.path.join(os.pardir, name)):
        return None
    text = (
        f"{name} is in the notebook's directory, not in the cell's working directory "
        f"work/: read it as ../{name}"
    )
    if hasattr(exc, "add_note"):
        exc.add_note(text)
    return text

"""What in a notebook directory belongs in version control, and what does not.

``notebook.toml`` is committed config and ``.strata/`` is runtime state; this
module is the single statement of that split, which the docs page renders.
"""

from __future__ import annotations

import os
from pathlib import Path

# All reproducible from the committed set, and none of it portable between
# machines (a .venv built on a Mac is broken on Linux).
IGNORED_PATTERNS: tuple[str, ...] = (
    ".strata/",
    ".venv/",
    "renv/library/",
    "renv/staging/",
    "__pycache__/",
    "*.pyc",
    # Rewritten by every `strata agent` launch: the server's port and session id.
    "/.mcp.json",
    "/CLAUDE.md",
)

_GITIGNORE_HEADER = """\
# Strata notebook: runtime state, not source.
#
# Everything here is rebuilt from notebook.toml, cells/, pyproject.toml and
# uv.lock. .strata/ holds display outputs, console logs and the artifact
# store; .venv/ and renv/library/ hold installed packages. `strata agent`
# writes .mcp.json and CLAUDE.md on each launch.
"""


def gitignore_contents() -> str:
    """The ``.gitignore`` a new notebook directory gets."""
    return _GITIGNORE_HEADER + "\n".join(IGNORED_PATTERNS) + "\n"


def committed_paths(notebook_dir: Path) -> list[Path]:
    """Existing files under *notebook_dir* that belong in version control, sorted.

    Paths are relative to *notebook_dir*. Includes cell tests and the lockfile.
    """
    notebook_dir = Path(notebook_dir)
    found: list[Path] = []

    for name in ("notebook.toml", "pyproject.toml", "uv.lock", "renv.lock", ".gitignore"):
        if (notebook_dir / name).exists():
            found.append(Path(name))

    cells_dir = notebook_dir / "cells"
    if cells_dir.is_dir():
        for path in cells_dir.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts:
                continue
            found.append(path.relative_to(notebook_dir))

    return sorted(found)


def cell_source_path(notebook_dir: Path, cell_id: str, file: str) -> Path:
    """Where a cell's source lives: ``cells/<file>``, refused if that leaves ``cells/``.

    ``file`` comes from ``notebook.toml``, which a cloned or imported notebook writes;
    an absolute path or ``..`` would otherwise read, overwrite or delete any file.

    Raises:
        ValueError: The resolved path is not inside ``cells/``.
    """
    cells_dir = Path(notebook_dir) / "cells"
    target = os.path.realpath(os.path.join(cells_dir, file))
    if not target.startswith(os.path.realpath(cells_dir) + os.sep):
        raise ValueError(f"cell {cell_id!r} names a source file outside cells/: {file!r}")
    return cells_dir / file


def write_gitignore(notebook_dir: Path) -> bool:
    """Write the notebook's ``.gitignore`` unless one exists; return whether it wrote.

    Never overwrites: an existing one may carry an enclosing project's rules.
    """
    target = Path(notebook_dir) / ".gitignore"
    if target.exists():
        return False
    target.write_text(gitignore_contents(), encoding="utf-8")
    return True

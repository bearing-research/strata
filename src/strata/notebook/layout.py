"""What belongs in version control, and what does not.

The split is real and load-bearing: ``notebook.toml`` is committed config,
``.strata/`` is runtime state. Until now it was documented and nowhere else —
no function returned the set, and nothing wrote a ``.gitignore``, so a new
notebook directory had none and ``git add -A`` swept in a virtualenv, a
SQLite artifact store, and every blob it held.

Stating it once, in code, means the docs page can be a rendering of these
rules rather than a second source that drifts from them.
"""

from __future__ import annotations

from pathlib import Path

# Runtime state, build output, and installed environments. Everything here is
# reproducible from the committed set, and none of it is portable between
# machines — a .venv committed on a Mac is worse than useless on Linux.
IGNORED_PATTERNS: tuple[str, ...] = (
    ".strata/",
    ".venv/",
    "renv/library/",
    "renv/staging/",
    "__pycache__/",
    "*.pyc",
)

_GITIGNORE_HEADER = """\
# Strata notebook: runtime state, not source.
#
# Everything here is rebuilt from notebook.toml, cells/, pyproject.toml and
# uv.lock. .strata/ holds display outputs, console logs and the artifact
# store; .venv/ and renv/library/ hold installed packages.
"""


def gitignore_contents() -> str:
    """The ``.gitignore`` a new notebook directory gets."""
    return _GITIGNORE_HEADER + "\n".join(IGNORED_PATTERNS) + "\n"


def committed_paths(notebook_dir: Path) -> list[Path]:
    """Files under *notebook_dir* that belong in version control, sorted.

    Paths are relative to *notebook_dir*, and only what exists is returned:
    this answers "what should be committed from this directory as it stands",
    which is the question both ``export`` and a git status check are asking.

    Cell sources include their tests — a cell's tests are as much a part of
    the notebook as the cell — and the lockfile is included because an
    environment nobody can reproduce makes the rest of it decorative.
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


def write_gitignore(notebook_dir: Path) -> bool:
    """Write the notebook's ``.gitignore``, unless one is already there.

    Never overwrites: a directory that already has one may be inside a project
    with its own rules, and silently replacing them would be a worse failure
    than not writing at all. Returns whether it wrote.
    """
    target = Path(notebook_dir) / ".gitignore"
    if target.exists():
        return False
    target.write_text(gitignore_contents(), encoding="utf-8")
    return True

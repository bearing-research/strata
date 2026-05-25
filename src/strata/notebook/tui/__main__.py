"""Module entrypoint: ``python -m strata.notebook.tui``.

Mirrors the ``strata-notebook-tui`` console script so the TUI is usable
without installing the project's bin shims (handy when iterating in a
worktree before ``uv sync`` rebuilds the entry point).
"""

from __future__ import annotations

from strata.notebook.tui.cli import main

if __name__ == "__main__":
    main()

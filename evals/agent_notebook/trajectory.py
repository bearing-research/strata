"""Driver-agnostic representation of an agent run.

A :class:`Trajectory` is the normalized form both drivers produce and every
grader consumes: an ordered list of :class:`ToolEvent`, plus the final text and
whether the run terminated cleanly. Keeping this independent of Claude Code's
wire format is what lets the deterministic replay driver (CI) and the real
Claude Code driver (local) share one set of graders.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Notebook tools that *do work* (mutate or execute), vs. read-only inspection.
# The in-tool rate is about work: reads and notes don't count either way.
WORK_TOOLS = frozenset(
    {
        "run_cell",
        "run_tests",
        "add_cell",
        "edit_cell",
        "remove_cell",
        "move_cell",
        "add_dependency",
        "remove_dependency",
    }
)
READ_TOOLS = frozenset({"list_notebooks", "get_notebook", "get_cell", "dag", "status", "note"})

# The ways an agent routes *around* the notebook — the escape hatches the
# working agreement tells it not to use. Each is a distinct bypass:
#
# * bash-python — running Python/tests via Bash instead of run_cell. Matched at
#   a word boundary so `pythonpath=...` or a filename containing "python" don't
#   trip it; `uv run python` / `uv run pytest` count.
# * bash-install — managing dependencies via Bash (pip/uv/conda) instead of
#   add_dependency, so the change never lands in the notebook's committed env.
# * cell-file-edit — editing a cell's source file directly (Write/Edit) instead
#   of edit_cell/add_cell, so the DAG and the watcher never see it.
_PY_ESCAPE = re.compile(r"(?:^|[\s;&|()`])(?:uv\s+run\s+)?(?:python3?|ipython|pytest)(?:\s|$)")
_INSTALL_ESCAPE = re.compile(
    r"(?:^|[\s;&|()`])(?:pip\s+install|uv\s+pip\s+install|uv\s+add|conda\s+install)\b"
)
_CELL_FILE = re.compile(r"[\\/]cells[\\/][^\\/]+\.py$")
_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})


@dataclass
class ToolEvent:
    """One tool call the agent made, as reported by a driver."""

    name: str
    arguments: dict = field(default_factory=dict)

    @property
    def notebook_tool(self) -> str | None:
        """The bare notebook tool name if this is an MCP strata call, else None.

        Claude Code namespaces MCP tools as ``mcp__<server>__<tool>`` and
        sanitizes the server name (``strata-notebook`` → ``strata_notebook``),
        so we split on ``__`` and key off the trailing tool segment rather than
        matching an exact prefix.
        """
        parts = self.name.split("__")
        if len(parts) >= 3 and parts[0] == "mcp":
            return parts[-1]
        return None

    @property
    def is_notebook_work(self) -> bool:
        return self.notebook_tool in WORK_TOOLS

    @property
    def is_notebook_read(self) -> bool:
        return self.notebook_tool in READ_TOOLS

    @property
    def escape_reason(self) -> str | None:
        """Why this call routes around the notebook, or None if it doesn't."""
        if self.name == "Bash":
            command = str(self.arguments.get("command", ""))
            if _PY_ESCAPE.search(command):
                return "bash-python"
            if _INSTALL_ESCAPE.search(command):
                return "bash-install"
            return None
        if self.name in _EDIT_TOOLS:
            path = str(self.arguments.get("file_path") or self.arguments.get("path") or "")
            if _CELL_FILE.search(path):
                return "cell-file-edit"
        return None

    @property
    def is_escape(self) -> bool:
        return self.escape_reason is not None

    def escape_detail(self) -> str:
        """A short human label for the escape (for reports)."""
        if self.name == "Bash":
            return str(self.arguments.get("command", ""))
        path = str(self.arguments.get("file_path") or self.arguments.get("path") or "")
        return f"{self.name} {path}"


@dataclass
class Trajectory:
    """The full record of one agent run against a notebook."""

    events: list[ToolEvent] = field(default_factory=list)
    final_text: str = ""
    ok: bool = True
    raw: object = None

    @property
    def tool_calls(self) -> int:
        return len(self.events)

    def work_events(self) -> list[ToolEvent]:
        return [e for e in self.events if e.is_notebook_work]

    def escape_events(self) -> list[ToolEvent]:
        return [e for e in self.events if e.is_escape]

"""Driver-agnostic representation of an agent run.

Both drivers produce a :class:`Trajectory` and every grader consumes one, so it
stays independent of Claude Code's wire format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Notebook tools that *do work* (mutate or execute), vs. read-only inspection.
# The in-tool rate is about work: reads and notes don't count either way.
WORK_TOOLS = frozenset(
    {
        "run_cell",
        "run_snippet",
        "run_tests",
        "add_cell",
        "edit_cell",
        "remove_cell",
        "move_cell",
        "add_dependency",
        "remove_dependency",
    }
)
READ_TOOLS = frozenset(
    {"list_notebooks", "get_notebook", "get_cell", "get_variable", "dag", "status", "note"}
)

# Ways an agent routes *around* the notebook:
# * bash-python: Python/tests via Bash instead of run_cell. Word-bounded so
#   `pythonpath=...` does not trip it; `uv run python` counts.
# * bash-install: pip/uv/conda via Bash instead of add_dependency, so the
#   notebook's committed env never changes.
# * cell-file-edit: Write/Edit on a cell file, which the DAG never sees.
_PY_ESCAPE = re.compile(r"(?:^|[\s;&|()`])(?:uv\s+run\s+)?(?:python3?|ipython|pytest)(?:\s|$)")
_INSTALL_ESCAPE = re.compile(
    r"(?:^|[\s;&|()`])(?:pip\s+install|uv\s+pip\s+install|uv\s+add|conda\s+install)\b"
)
_CELL_FILE = re.compile(r"[\\/]cells[\\/][^\\/]+\.py$")

# The `strata` CLI is the other way to drive the notebook (the scratchpad path,
# off the MCP on-ramp). Its calls are notebook work or reads, NOT escapes, so a
# `strata` segment's own arguments must not trip the escape detectors. The work
# set mirrors WORK_TOOLS.
_CLI_WORK = re.compile(
    r"(?:^|[\s;&|()`])strata\s+"
    r"(?:cell\s+(?:add|edit|run|test|rm|mv|annotate)|dep\s+(?:add|rm|remove))\b"
)
_CLI_READ = re.compile(r"(?:^|[\s;&|()`])strata\s+(?:cell\s+(?:list|show)|dag|status)\b")
_EDIT_TOOLS = frozenset({"Write", "Edit", "MultiEdit", "NotebookEdit"})

# Classify Bash per segment so a `strata` call cannot mask an escape in another
# segment. Not split on bare `|`, which shows up far more often inside a `-c`
# payload than as a pipe. A `-c` payload with `;` then a python token can still
# be misread; acceptable for eval scoring.
_CMD_SEP = re.compile(r"&&|\|\||;")


def _bash_segments(command: str) -> list[str]:
    return _CMD_SEP.split(command)


@dataclass
class ToolEvent:
    """One tool call the agent made, as reported by a driver."""

    name: str
    arguments: dict = field(default_factory=dict)

    @property
    def notebook_tool(self) -> str | None:
        """The bare notebook tool name if this is an MCP strata call, else None.

        Keys off the last ``__`` segment because Claude Code sanitizes the
        server name in ``mcp__<server>__<tool>``.
        """
        parts = self.name.split("__")
        if len(parts) >= 3 and parts[0] == "mcp":
            return parts[-1]
        return None

    @property
    def _bash_command(self) -> str:
        return str(self.arguments.get("command", "")) if self.name == "Bash" else ""

    @property
    def is_notebook_work(self) -> bool:
        if self.notebook_tool in WORK_TOOLS:
            return True
        return any(_CLI_WORK.search(seg) for seg in _bash_segments(self._bash_command))

    @property
    def is_notebook_read(self) -> bool:
        if self.notebook_tool in READ_TOOLS:
            return True
        return any(_CLI_READ.search(seg) for seg in _bash_segments(self._bash_command))

    @property
    def escape_reason(self) -> str | None:
        """Why this call routes around the notebook, or None if it doesn't."""
        if self.name == "Bash":
            for seg in _bash_segments(self._bash_command):
                if _CLI_WORK.search(seg) or _CLI_READ.search(seg):
                    continue
                if _PY_ESCAPE.search(seg):
                    return "bash-python"
                if _INSTALL_ESCAPE.search(seg):
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

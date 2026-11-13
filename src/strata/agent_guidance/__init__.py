"""What coding agents are told about Strata, from one file (``scratchpad.md``).

Each channel (the scratchpad MCP server's instructions, the notebook's
``CLAUDE.md`` block) reads its section here, so the advice cannot drift apart.
"""

from __future__ import annotations

import re
from functools import cache
from importlib.resources import files

_MARKER = re.compile(r"^<!-- section: (\w+) -->$", re.MULTILINE)


@cache
def _sections() -> dict[str, str]:
    text = files(__package__).joinpath("scratchpad.md").read_text(encoding="utf-8")
    # split with one group: [preamble, name, body, name, body, ...]
    parts = _MARKER.split(text)
    return {name: body.strip() + "\n" for name, body in zip(parts[1::2], parts[2::2], strict=True)}


def section(name: str) -> str:
    """The text of one tagged section of ``scratchpad.md``; ``KeyError`` if there is none."""
    return _sections()[name]

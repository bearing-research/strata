# strata-scratchpad (Claude Code plugin)

The [Strata](https://github.com/bearing-research/strata) scratchpad for Claude Code:
a persistent, cached Python session in your project's environment, used instead of
`python -c` and temp scripts. A value the agent computed earlier is still there, and
comes back instantly while nothing it depends on has changed.

## What's in it

- **The `strata` MCP server** (`strata mcp`, loaded up front): the `run` and
  `scratchpad` tools, over one daemon per project, which the first call starts.
- **Two hooks**, at session start and on each prompt: what the project's scratchpad
  already holds, each cell checked against the files it read (`cached`, `stale`, or
  `as of` the last pass). They run `strata-agent-hook`, installed beside `strata`,
  and say nothing in a project with no scratchpad.

## Prerequisites

The `strata` CLI, 0.12 or later, on the `PATH` Claude Code starts with:

```
uv tool install "strata-notebook[mcp]"
```

Claude Code started from a GUI or an IDE may not have uv's tool directory
(`~/.local/bin`) on its `PATH`; `strata agent install`, which writes the server by its
absolute path, avoids that.

## Install

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

A plugin cannot write a permission, so each `run` asks until you allow
`mcp__plugin_strata-scratchpad_strata__*`, nor the `CLAUDE.md` block that tells
subagents about the scratchpad. `strata agent install` writes the same server and
hooks with both. Use one or the other: with both, each prompt gets two blocks naming
two servers.

See [Agent scratchpad](https://bearingresearch.com/strata/latest/notebook/scratchpad/).

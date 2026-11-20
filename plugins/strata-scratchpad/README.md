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
  `as of` the last pass). They run on `python3` alone and say nothing in a project
  with no scratchpad.

## Prerequisites

The `strata` CLI on `PATH`:

```
uv tool install "strata-notebook[mcp]"
```

## Install

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

`strata agent install` writes the same server and hooks, and also the
`mcp__strata__*` permission and a `CLAUDE.md` block, which a plugin cannot. Use one
or the other.

`hooks/agent_hook.py` is a copy of `strata/agent_hook.py`; a test keeps them
identical.

See [Agent scratchpad](https://bearingresearch.com/strata/latest/notebook/scratchpad/).

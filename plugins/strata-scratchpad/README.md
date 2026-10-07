# strata-scratchpad (Claude Code plugin)

Makes a coding agent reach for a **persistent, cached [Strata](https://github.com/bearing-research/strata)
notebook** as its scratchpad (quick computations, data exploration, sanity
checks) instead of throwaway `python -c` / temp scripts. Work persists, is
watchable in a live terminal viewer, and unchanged runs are instant.

## What's in it

- **`strata-scratchpad` skill**, model-invoked: when the agent is about to run
  ad-hoc Python, it uses a notebook cell (`strata cell add … --run`) instead.
- **`/strata-scratchpad:scratch` command**, human-invoked: explicitly set up a
  scratchpad in the current project and switch the agent to it.

## Prerequisites

The `strata` CLI must be installed and on `PATH`, version with `strata cell add --run`
(the one-call add-and-run primitive). `uv tool install "strata-notebook[mcp,tui]"`
installs it with what `strata agent` needs.

## Install

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

Once installed the skill is auto-discovered in any project. A human can watch the
agent's scratchpad live with `strata agent ./scratch`, which starts a server
scoped to the project; plain `strata watch ./scratch` only works against a server
whose `STRATA_NOTEBOOK_STORAGE_DIR` covers the project.

See [Driving a notebook with a coding agent](https://bearing-research.github.io/strata/latest/notebook/agent/).

# The notebook as an agent's scratchpad

A coding agent working on *something else* — debugging your service, exploring a
dataset, checking what a function returns — constantly needs to run a bit of
throwaway Python. The usual way is `python -c` or a temp script: work that
vanishes the moment it finishes, that nobody can see, and that recomputes from
scratch every time.

A Strata notebook is a better home for exactly that. The agent adds a cell and
runs it; the cell is versioned, content-addressed, and cached.

!!! info "This is not the same as [driving a notebook](agent.md)"

    There the notebook is **the deliverable** — you asked for it, you watch it
    being built, you keep it. Here it is **disposable infrastructure** for some
    other task, created by the agent on demand, and you may never look at it.

    Same machinery, opposite intent. The distinction matters mostly for setup:
    driving starts with you running `strata agent`; scratchpad use starts with
    the agent noticing it has a better option than `python -c`.

## Set it up once, in any project

The behavior travels with the package, but the plugin is the reliable way to get
it into every Claude Code session — it also bundles a
`/strata-scratchpad:scratch` command:

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

Once installed the skill is auto-discovered in **every** project, so the agent
reaches for a cached notebook cell even when it wasn't launched anywhere near a
notebook. The `strata` CLI still needs to be on `PATH`. The plugin source lives
in the repo at `plugins/strata-scratchpad/`.

The skill also ships inside `strata-notebook` itself (under
`<site-packages>/strata/.agents/skills/`), which some agents discover on their
own; the plugin removes the "some".

## Why it beats a temp script

- **Unchanged work never recomputes.** Every cell is cached by provenance
  (`sha256(inputs + source + env)`), *including a leaf cell that only `print`s*.
  An agent that re-runs an unchanged diagnostic gets its output back instantly.
  The expensive step it ran ten turns ago is still a cache hit now — which is
  the whole point, because agents re-derive the same thing constantly.
- **It persists and it can be seen.** Each snippet is a real cell on disk, not
  an invisible script in `/tmp`. You can open it later, or watch it live.
- **Side effects stay honest.** A cell that writes a file, calls an API, or
  reads the clock must not replay a stale result. Those get
  [`# @nocache`](annotations.md#nocache) and always re-execute; everything else
  stays cached.

## Why the agent actually reaches for it

A better option that costs more gets ignored. Two things keep the cost equal:

- **One call, not three.** `run_snippet(session_id, source)` over
  [MCP](mcp.md), or `strata cell add … -c 'src' --run` on the CLI, adds a cell
  **and** runs it **and** returns `stdout` — the same shape as `python -c`.
- **No setup step.** The skill tells the agent how to create or reuse a
  scratch notebook itself, so there is no moment where using the notebook means
  stopping to configure one.

## Looking at it

You don't have to, and usually won't. When you want to:

```bash
strata watch ./scratch        # live terminal viewer, read-only
```

or open the notebook in the [web UI](../getting-started/notebook.md). Everything
in [Watching an agent work](agent.md#watching-an-agent-work) applies here too.

## Related

- [Driving a notebook with a coding agent](agent.md): the other use case, where
  the notebook is the point.
- [MCP Server](mcp.md): the tool list and the endpoint's security model.
- [Cell Annotations](annotations.md): `# @nocache` and the rest.

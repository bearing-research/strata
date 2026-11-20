# Agent scratchpad

A coding agent working on *something else* (debugging your service, exploring a
dataset, checking what a function returns) keeps running a bit of throwaway
Python. The usual way is `python -c` or a temp script: work that vanishes when it
finishes, that nobody can see, and that recomputes from scratch every time.

The Strata scratchpad is a better home for exactly that: a persistent, cached
Python session in your project's environment, which the agent reaches through two
MCP tools, `run` and `scratchpad`. A value it computed ten turns ago, or in
yesterday's session, is still there, and comes back instantly while nothing it
depends on has changed.

!!! info "This is not the same as [driving a notebook](agent.md)"

    There the notebook is **the deliverable**: you asked for it, you watch it
    being built, you keep it. Here it is **disposable infrastructure** for some
    other task, created on the agent's first call, and you may never look at it.

## Set it up

### 1. Install the `strata` CLI

```bash
uv tool install "strata-notebook[mcp]"
```

A `uv tool` install is the lasting choice: the agent's config names this `strata`
by its path, so it must outlive the shell you ran it from.

### 2. Wire it into the agent

```bash
strata agent install            # Claude Code and Codex, whichever are installed
strata agent install claude     # or just one
```

For Claude Code this writes the user-scope `strata` MCP server, the
`mcp__strata__*` permission, two hooks (below) and a short block in `CLAUDE.md`
for subagents that do not see MCP instructions. For Codex it writes the server and
the block in `AGENTS.md`. `strata agent uninstall` removes exactly what it wrote.
What each piece is, and how a file you edited since is handled, is in
[`strata agent install`](mcp.md#install-it-into-the-agent-strata-agent-install).

If you would rather install through Claude Code's plugin manager, the plugin
bundles the same server and hooks, but not the permission (allow
`mcp__plugin_strata-scratchpad_strata__*` yourself) or the `CLAUDE.md` block, which
a plugin cannot write. Use one or the other, not both:

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

### 3. Confirm it engaged

Start a session in any project and ask for something only Python can answer:

```text
Fit a linear regression to the points (1,2), (2,4.1), (3,5.9), (4,8.2)
and tell me the slope and intercept.
```

The agent should call `run` rather than `python -c`. The first call in a project
creates `.strata-scratch/` at the project's root and starts its daemon; its result
carries `setup_ms`, what that took. Then, from a shell in the same project:

```bash
strata scratch show             # what the scratchpad holds, as the agent sees it
```

## What the agent is told

Three channels; the first two read one source (`strata/agent_guidance/scratchpad.md`),
and the hooks' lines are written by `strata/agent_hook.py`:

- **The MCP server's instructions**: use `run` instead of `python -c` or a temp
  script, look in `scratchpad` before computing, put a slow step in its own cell.
- **The `CLAUDE.md` block**, the same advice for subagents and harnesses without
  MCP, naming `strata scratch run` as the shell equivalent.
- **The hooks.** At session start, and on each prompt for the cells the prompt
  names, Claude Code is told what the scratchpad holds:

  ```
  <strata_scratchpad note="Strata scratchpad for this project; build on these with mcp__strata__run">
  - df (cell c1a2b3c4): 38 s to build, cached
  - model (cell c9f0d1e2): 2 s to build, stale, src/proj/train.py changed
  </strata_scratchpad>
  ```

  A plugin install names its own tool, `mcp__plugin_strata-scratchpad_strata__run`.
  `cached` always means checked just now: the hook looks at every file a cell read
  again before saying it, so a file you edited while no daemon ran makes its cell
  stale, never cached. What it cannot check within its budget is said `as of` the
  last pass. It never starts the daemon, and says nothing in a project with no
  scratchpad.

## When a cell runs again

A cell is keyed on its source, the values it uses, the environment, and every file
its run read or imported: the project's own modules included. Editing
`src/proj/features.py` re-runs the cells that imported it and what depends on them,
and nothing else. Editing a file a cell opened does the same.

Two things the key cannot see:

- **What a process the cell started read** (a `subprocess`, a C extension reading
  a file). After editing such a file, the agent passes `fresh=True`.
- **A value that changes by itself**: the clock, `random`, a live API. A cell that
  reads one, or exists for its side effect (writing a file, calling an API),
  starts with [`# @nocache`](../reference/annotations/python.md#nocache) and always
  runs.

How the read set is recorded and checked is in
[A project's own daemon](mcp.md#a-projects-own-daemon).

## Where it lives

- `.strata-scratch/` at the project's root (the nearest `pyproject.toml`, bounded
  by the git root): cells, values, and the digest the hooks read. Add it to
  `.gitignore`; deleting it costs reruns only.
- Cells run in the project's environment when it has one (`.venv`, or a uv
  project's), else in the scratchpad's own. Their cwd is the scratchpad's `work/`
  directory, so the agent uses absolute paths for project files.
- One daemon per scratchpad, started by the first call and stopped after 30 idle
  minutes; agents in the same project share it.

```bash
strata scratch ls               # every scratchpad: size, last use, daemon
strata scratch gc               # sweep one now, by idle days then its soft limit
strata scratch rm               # stop its daemon and delete .strata-scratch/
```

## Looking at it

You don't have to, and usually won't. Each `run` result carries a `url`: the
scratchpad open in the notebook UI, served by its daemon, with every cell the agent
ran. `strata scratch status` prints the daemon's address too.

## From a shell

An agent with only a shell, or you, can use the same session:

```bash
strata scratch run -c 'import pandas as pd; df = pd.read_parquet("/repo/data.parquet"); df.shape'
strata scratch show df
```

They are the `run` and `scratchpad` tools, printing the same JSON.

## Related

- [MCP server](mcp.md#scratchpad-profile): the scratchpad profile in full, its
  daemon, and the installer's pieces.
- [Use a coding agent](agent.md): the other use case, where the notebook is the
  point.
- [Annotations](../reference/annotations/index.md): `# @nocache` and the rest.

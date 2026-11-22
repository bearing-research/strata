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

!!! info "This is not the same as [driving a notebook](https://bearingresearch.com/strata/dev/notebook/agent/)"

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
strata agent install                    # every agent whose CLI is on PATH; asks first
strata agent install --target claude    # or just one
strata agent install --yes              # without asking, as from a script
```

For Claude Code this writes the user-scope `strata` MCP server, the
`mcp__strata__*` permission, two hooks (below) and a short block in `CLAUDE.md`
for subagents that do not see MCP instructions. For Codex it writes the server and
the block in `AGENTS.md`. `strata agent uninstall` removes exactly what it wrote.
What each piece is, and how a file you edited since is handled, is in
[`strata agent install`](https://bearingresearch.com/strata/dev/notebook/mcp/#install-it-into-the-agent-strata-agent-install).

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
  starts with [`# @nocache`](https://bearingresearch.com/strata/dev/reference/annotations/python/#nocache) and always
  runs.

How the read set is recorded and checked is in
[A project's own daemon](https://bearingresearch.com/strata/dev/notebook/mcp/#a-projects-own-daemon).

## What it saves

Measured with the
[uptake eval](https://github.com/bearing-research/strata/tree/main/evals/agent_notebook/uptake):
headless Claude Code on a small uv project whose questions read 1M order events
through the project's own loader. That loader stands in for a warehouse export and
takes 20 s every time it runs (`HARBOR_LOAD_SECONDS`), so what is measured is a slow
load that a session keeps needing. Each trial is a fresh copy of the project and a
config of its own. Arm A cannot reach Strata at all, and arm C is set up by
`strata agent install`, with the scratchpad created beforehand as a returning
user's would be. The 29 tasks come in five kinds, each run three times per arm; 10
of them, the holdout, drawn by a fixed seed before any arm C run, were never looked
at while the guidance and the tools were tuned.

Median wall time of C over A, with its 95% bootstrap interval:

| Kind | What the agent does | Tuning tasks | Holdout |
| --- | --- | --- | --- |
| Iteration | four questions in a row over the same data | 0.42 [0.38, 0.55] | 0.67 [0.53, 0.84] |
| Plot | a chart from the data | 0.54 [0.41, 0.58] | 0.85 [0.61, 1.10] |
| Probing | calls project code with varied arguments | 1.04 [0.83, 1.12] | 1.01 [0.88, 1.60] |
| Neutral | either tool does | 0.99 [0.61, 2.32] | 1.01 [0.80, 1.42] |
| Must-not | edits and file reads, no Python to run | 1.13 [0.75, 1.56] | 1.04 [0.58, 1.58] |

- **The data is loaded once.** In iteration and plot trials, the loader ran once in
  the whole trial in 21 of 21 of C's (holdout: 9 of 9), and in 3 of 21 of A's
  (2 of 9): A loaded the events again, where C used the value already in the
  scratchpad. On the tuning tasks an iteration trial's median took A 130 s and C
  55 s (holdout: 92 s and 62 s). Most of that is the loads not repeated, so a load
  that takes a second would save far less; only the 20 s load was measured.
- **It answers as often.** Every trial in both arms completed: 57 of 57 per arm on
  the tuning tasks, 30 of 30 on the holdout.
- **It stays out of the way.** On must-not tasks C used the scratchpad in 1 of 15
  trials (holdout: 0 of 9), and those tasks took about as long as in A.
- **It costs the same.** $7.86 for C's 57 tuning trials against $8.01 for A's.

These are one model (`claude-opus-5-5`, Claude Code 2.1.330) on one machine (an
Apple silicon Mac), with three runs per task, and an interval that crosses 1 is no
evidence of a saving. The holdout's plot row is a single task. The rows ran on
different Strata commits:

- **Tuning tasks except probing:** arm C's branch
  ([#1598](https://github.com/bearing-research/strata/pull/1598)), before
  [#1599](https://github.com/bearing-research/strata/pull/1599).
- **Tuning probing:** a rerun after
  [#1600](https://github.com/bearing-research/strata/pull/1600), which starts the
  daemon as the session opens. On #1598's branch it took 1.53 [1.30, 2.04] times
  A's: the first `run` waited about 4 s for the daemon, and a value's preview showed
  only its type's name, which #1599 changed to its repr.
- **Holdout:** not rerun after #1600; it ran at #1599, so its probing row still
  includes the daemon wait of about 4 s.

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

- [MCP server](https://bearingresearch.com/strata/dev/notebook/mcp/#scratchpad-profile): the scratchpad profile in full, its
  daemon, and the installer's pieces.
- [Use a coding agent](https://bearingresearch.com/strata/dev/notebook/agent/): the other use case, where the notebook is the
  point.
- [Annotations](https://bearingresearch.com/strata/dev/reference/annotations/): `# @nocache` and the rest.

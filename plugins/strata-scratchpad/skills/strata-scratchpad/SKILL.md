---
name: strata-scratchpad
description: Use a persistent, cached Strata notebook as a scratchpad for throwaway Python — quick computations, data exploration, sanity checks, inspecting a value — instead of writing temp scripts or running `python -c`. Triggers whenever you are about to run ad-hoc Python to compute, inspect, or explore, and the `strata` CLI is available.
---

# Strata as your scratchpad

When you need to run throwaway Python — compute a number, inspect a dataframe,
sanity-check a function, explore some data — **put it in a Strata notebook cell
instead of a `/tmp` script or `python -c`.** A cell costs the same one call, but
unlike a scratch script it **persists, is inspectable, and caches** — so you
build on your own prior work instead of recomputing it.

This is a *stateful* scratchpad, not a stateless shell. Every cell and its
variables stay for the whole session; you can look back at them, reference them,
and edit-and-rerun them.

## Decision rule

> Any Python you'd run to check or compute something → a notebook cell via
> add-and-run, **not** `Bash: python -c` and **not** a temp `.py` file.

## Setup (once per project)

Create a scratchpad notebook if one doesn't exist yet, and gitignore it:

```bash
strata new scratch --parent . --no-env --project-mount   # ./scratch/, project mounted as `project`
echo "scratch/" >> .gitignore              # optional; it's your workspace
```

`--project-mount` mounts the project dir read-only as a `project` Path variable in
every cell, so you can read project files as `project / "file"` (see below).

If a `strata-notebook` **MCP server is connected** (you were launched via
`strata agent`, or `.mcp.json` registers one), prefer the MCP tools below —
your runs then appear **live in the human's terminal viewer**. Otherwise use the
`strata` CLI against the directory; a human can attach a viewer anytime with
`strata watch ./scratch`.

## Run a snippet (one call)

CLI (works anywhere, no server needed):

```bash
strata cell add ./scratch -c 'print(sum(range(6)) / 6)' --run
```

Returns JSON: the new cell plus a `run` block with `status`, `cache_hit`,
`stdout`, `stderr`. The first run of new code is always cold (`cache_hit: false`)
— that's expected.

MCP (when a session is connected — visible live in the viewer):

```
run_snippet(session_id, "print(sum(range(6)) / 6)")
```

**Third-party libraries:** a fresh scratchpad ships only the notebook runtime
(`pyarrow`, `orjson`, `cloudpickle`). Add anything else first, then use it:

```bash
strata dep add ./scratch numpy         # or: add_dependency(session_id, "numpy")
strata cell add ./scratch -c 'import numpy as np; print(np.arange(6).mean())' --run
```

## Reading files from your project

A cell executes with its **working directory set to the notebook dir** (e.g.
`./scratch`), **not** the directory you launched from — so a bare
`open("data.txt")` won't find a file in your project. If you created the
scratchpad with `--project-mount` (above), read project files through the
`project` Path variable, which points at the project root:

```bash
strata cell add ./scratch -c 'import json; print(len(json.load(open(project / "events.json"))))' --run
```

Without the mount, pass an **absolute path** instead (`open("/abs/path/events.json")`),
or add a per-cell `# @mount data /abs/path ro` (injects `data` as a `pathlib.Path`).

## Look before you compute

Before recomputing something, check what the scratchpad already holds:

```bash
strata status ./scratch              # cells, their variables, staleness
strata cell show ./scratch --var df  # the cell that defines `df` — or, if it's not
                                     # defined, the list of variables that ARE
strata cell show ./scratch <cell_id> # a specific cell's source + output + console
```

(MCP: `get_variable(session_id, "df")`.) If a variable you need already exists,
**reference it in a new cell** rather than recomputing it — the DAG wires the
dependency and the upstream stays cached.

## The caching payoff — how to get it

- **Put an expensive step in its own cell whose result a later cell consumes.**
  That result is cached by provenance across the whole session: iterate on the
  downstream cell all you like, the expensive upstream is a cache hit and never
  re-executes. (This is the point — don't recompute the slow thing.)
- **To change a result, edit its cell and re-run** (`strata cell edit … && strata
  cell run …`, or `edit_cell` + `run_cell`). Only that cell and what's downstream
  recompute; everything upstream stays cached.
- **Prefer editing an existing cell over adding a new identical one.** A leaf
  cell that only prints replays its cached output on an *unchanged same-cell*
  re-run; re-adding the same snippet as a new cell runs cold.

## The one exception — side effects and fresh values

A cell whose point is a **side effect** (writing a file, calling an API, mutating
external state) or a **fresh value** (the clock, `random`, a live endpoint) must
not replay a cached result. Put `# @nocache` on its first line so it always
re-executes:

```bash
strata cell add ./scratch -c '# @nocache
import time; print(time.time())' --run
```

## Don'ts

- Don't fall back to `python -c` / temp scripts for exploration — that work is
  invisible, uncached, and thrown away.
- Don't delete a diagnostic cell just because it "finished" — leave it; an
  unchanged re-run is instant, and the human can see it.
- Don't read `./scratch/.strata/` directly — it's machine-managed runtime state.

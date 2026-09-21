# The notebook as an agent's scratchpad

A coding agent working on *something else* (debugging your service, exploring a
dataset, checking what a function returns) constantly needs to run a bit of
throwaway Python. The usual way is `python -c` or a temp script: work that
vanishes the moment it finishes, that nobody can see, and that recomputes from
scratch every time.

A Strata notebook is a better home for exactly that. The agent adds a cell and
runs it; the cell is versioned, content-addressed, and cached.

!!! info "This is not the same as [driving a notebook](agent.md)"

    There the notebook is **the deliverable**: you asked for it, you watch it
    being built, you keep it. Here it is **disposable infrastructure** for some
    other task, created by the agent on demand, and you may never look at it.

    Same machinery, opposite intent. The distinction matters mostly for setup:
    driving starts with you running `strata agent`; scratchpad use starts with
    the agent noticing it has a better option than `python -c`.

## Set it up

### 1. Put the `strata` CLI on your `PATH`

```bash
uv tool install strata-notebook
```

The agent shells out to `strata`, so this is the one hard requirement. Check it
with `strata --help`.

### 2. Install the plugin

In any Claude Code session:

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

Once installed the skill is auto-discovered in **every** project, so the agent
reaches for a cached notebook cell even when it wasn't launched anywhere near a
notebook. The plugin also bundles a `/strata-scratchpad:scratch` command, and
its source lives in the repo at `plugins/strata-scratchpad/`.

The skill also ships inside `strata-notebook` itself (under
`<site-packages>/strata/.agents/skills/`), which some agents discover on their
own; the plugin removes the "some".

### 3. Confirm it engaged

This is worth doing once, because the whole thing is invisible by design.
Nothing announces itself, and an agent that quietly kept using `python -c` looks
identical from the outside.

Ask an agent, in any project, for something it **cannot** answer with shell
tools. The skill only triggers when it was about to run ad-hoc *Python*, so a
question `find`/`awk` can answer proves nothing:

```text
Fit a linear regression to the points (1,2), (2,4.1), (3,5.9), (4,8.2)
and tell me the slope and intercept.
```

Then look for the notebook:

```bash
ls ./scratch/cells/            # one .py file per snippet it ran
strata cell list ./scratch     # the same thing as JSON, with each cell's output
```

If `./scratch` exists and has cells in it, the skill engaged. If it does not,
the agent answered with `python -c` and the skill did not fire. Check that
`strata --help` works **in the agent's environment**, which is the usual cause.

### 4. Nothing else, for the agent

The agent needs no server and no session: it drives the notebook through the
`strata` CLI against the directory. (Looking at it yourself is the one thing
that does need a server; see [Looking at it](#looking-at-it) below.)

It creates the scratch notebook the first time it needs one, with

```bash
strata new scratch --parent . --no-env --project-mount
```

which also mounts the project read-only as a `project` variable, so cells read
your files as `project / "some/file.py"`. Adding `scratch/` to `.gitignore` is
optional, since it is your workspace and some people keep it.

!!! warning "The project mount is pinned, so editing a file does not restale a cell"

    The mount is written with `pin = "project-root"`, which fingerprints the
    pin rather than the tree. That is deliberate: hashing a whole repository
    before every cell run would cost more than the cell. The consequence is
    that a cell which read `project / "config.yaml"` keeps returning what that
    file said the first time, however often you edit it, and nothing marks the
    cell stale.

    A cell that reads a file you are actively changing wants
    [`# @nocache`](annotations.md#nocache), the same as a clock read. A cell
    that reads a file you are not changing is fine as it is, and is the case
    the pin exists for.

## Why it beats a temp script

- **Unchanged work never recomputes.** Every cell is cached by provenance
  (`sha256(inputs + source + env)`), *including a leaf cell that only `print`s*.
  An agent that re-runs an unchanged diagnostic gets its output back instantly.
  The expensive step it ran ten turns ago is still a cache hit now, which is
  the whole point, because agents re-derive the same thing constantly.
- **It persists and it can be seen.** Each snippet is a real cell on disk, not
  an invisible script in `/tmp`. You can open it later, or watch it live.
- **Side effects stay honest.** A cell that writes a file, calls an API, reads
  the clock, or reads a file that changes underneath it must not replay a stale
  result. Those get [`# @nocache`](annotations.md#nocache) and always
  re-execute; everything else stays cached.

## Why the agent actually reaches for it

A better option that costs more gets ignored. Two things keep the cost equal:

- **One call, not three.** `run_snippet(session_id, source)` over
  [MCP](mcp.md), or `strata cell add … -c 'src' --run` on the CLI, adds a cell
  **and** runs it **and** returns `stdout`, the same shape as `python -c`.
- **No setup step.** The skill tells the agent how to create or reuse a
  scratch notebook itself, so there is no moment where using the notebook means
  stopping to configure one.

## Looking at it

You don't have to, and usually won't. When you want to, the simplest way is:

```bash
strata agent ./scratch        # starts a server scoped to it, then attaches the viewer
```

!!! warning "`strata watch ./scratch` on its own usually won't work"

    A server only opens notebooks inside its configured storage root, which
    defaults to `~/.strata/notebooks`, and a scratchpad lives in your project.
    Pointed at a default server, opening it is rejected with *"Invalid notebook
    path: must be inside configured notebook storage"*, and the viewer sits at
    `connecting…` rather than saying so.

    `strata agent` avoids this because it starts a server scoped to the
    notebook's parent directory. If you would rather use a server you already
    run, start it with its root over your project:

    ```bash
    STRATA_NOTEBOOK_STORAGE_DIR=. strata-notebook &
    strata watch ./scratch
    ```

Once something is attached, everything in
[Watching an agent work](agent.md#watching-an-agent-work) applies here too,
including opening it in the [web UI](../getting-started/notebook.md) instead.

One thing changes for the agent, though: `strata cell run ./scratch …` opens
its own offline session on the directory, so its runs do not appear in the
viewer you just attached, and both sessions are writing the same files. Once
someone is watching, point the agent at that server instead: MCP, or
`--server`/`--session` on the same commands, both covered in the
[CLI guide](cli.md#working-against-a-live-session-server-session).

## Related

- [Driving a notebook with a coding agent](agent.md): the other use case, where
  the notebook is the point.
- [MCP Server](mcp.md): the tool list and the endpoint's security model.
- [Cell Annotations](annotations.md): `# @nocache` and the rest.

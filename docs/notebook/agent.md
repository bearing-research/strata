# Driving a notebook with a coding agent

`strata agent <notebook-dir>` is a one-command on-ramp: it stands up everything
a coding agent (Claude Code) needs to drive a **live** Strata notebook, and
attaches a terminal viewer so you watch it happen in real time.

This is the **coding agent** quickstart. The same notebook can also be driven
by hand in the [web UI](../getting-started/notebook.md) or watched from a
[terminal](tui.md).

!!! info "Not the same as [using a notebook as a scratchpad](scratchpad.md)"

    Here the notebook is **the deliverable**: you asked for it, you watch it get
    built, you keep it. An agent can also use a notebook as **disposable
    infrastructure** while working on something else entirely — running
    throwaway Python that happens to be cached instead of thrown away. That is
    [a separate setup](scratchpad.md) with a different starting point: this page
    starts with you running a command, that one starts with the agent noticing
    it has a better option than `python -c`.

It exists because the pieces were already there (the [MCP server](mcp.md), the
[`strata` CLI](cli.md) ops, the [terminal viewer](tui.md)), but wiring them
together by hand is a fiddly, ordered dance: enable the MCP endpoint *before*
the server boots, open a session (the agent can't do that itself), point the
agent at the right session, and only then start driving. `strata agent` does all
of it in one step.

## Prerequisites

The `[mcp]` extra (the agent endpoint) and [Claude Code](https://claude.com/claude-code)
— or another MCP-capable coding agent — on your `PATH`. Add `[tui]` for the
terminal viewer; skip it if you plan to watch in a browser and pass `--no-tui`:

=== "From PyPI"

    ```bash
    uv tool install "strata-notebook[mcp,tui]"
    ```

=== "From source"

    ```bash
    uv sync --extra mcp --extra tui     # or: uv sync --all-extras
    ```

## Use it

```bash
strata agent ./my-notebook
```

That single command:

1. **creates-or-opens** the notebook directory (scaffolds a new one if needed),
2. **starts** a notebook server with the MCP endpoint enabled (or **reuses** one
   already running on the target URL),
3. **opens a session**, the step the agent cannot do itself, since the MCP
   tools only see notebooks that are already open,
4. **writes** a `.mcp.json` and a managed block in `CLAUDE.md` into the notebook
   directory, and
5. **attaches** the read-only TUI to that session.

It prints the one line you run on the agent's side, then hands the terminal to
the TUI. In a second terminal:

```bash
cd ./my-notebook && claude
```

Claude Code discovers the notebook's `.mcp.json`, connects to the `strata-notebook`
MCP server, reads the `CLAUDE.md` working agreement, and starts building the
notebook by adding and running cells — each of which lights up live in whatever
you have watching, the TUI in the first terminal by default. See
[Watching an agent work](#watching-an-agent-work) for the alternatives.

When you quit the TUI, the server `strata agent` started is shut down with it. If
you pointed it at a server you started yourself, that server is left running.

## Watching an agent work

`strata agent` attaches the TUI because it is the option that needs no extra
step. **It is not the only one, and it is not the richest.**

Everything that watches a notebook is a WebSocket client on the same session,
and the server broadcasts each `cell_status` / `cell_output` / `cell_console`
frame to all of them. Opening the notebook a second time reuses the existing
session rather than creating a new one, so any of these lands on the session the
agent is driving:

| How | What you get | When it fits |
| --- | --- | --- |
| **Web UI** — open the server URL in a browser and open the same notebook | The full editor: outputs, plots, the DAG, the inspector. **Not read-only** — you can edit and run cells alongside the agent | You want to see rendered output, or take over |
| **TUI** — `strata watch ./my-notebook` | Read-only live spectator in the terminal | No browser, over SSH, or beside the agent in a split terminal |
| **Any WebSocket client** | The raw frames | You are building your own view — see the [client protocol](../reference/notebook-protocol.md) |
| **Nothing, then read the directory** | `cells/*.py` is the source; `.strata/runtime.json` holds display outputs, provenance and timings; `.strata/console/` holds per-cell stdout/stderr | You would rather review afterwards than watch |

The last row is worth knowing: a notebook is a directory of ordinary files, so
`strata cell show`, `strata dag`, `git diff` and your editor all work on an
agent's output with no live connection at all.

One caveat on that row. Live *cell status* — running, ready, errored — belongs
to the session, not to disk, so `strata status` run against the directory
reports `idle` for cells the agent has already executed. What it does tell you
offline is real and separately useful: each cell's **staleness** and why. For
"what is it doing right now", attach one of the live views above.

If you would rather keep a browser tab than a terminal viewer, start with
`strata agent --no-tui` and open the notebook in the web UI instead.

## What gets written into the notebook

- **`.mcp.json`**: registers the running server's `/mcp` endpoint as an MCP
  server named `strata-notebook`, so Claude Code auto-connects when launched in
  the directory. Overwritten on each launch (it points at the current port).
- **`CLAUDE.md`**: a working agreement telling the agent to drive the notebook
  through the MCP tools rather than writing throwaway `.py` scripts, so its work
  is captured as versioned, content-addressed, cached cells that you can see.
  Only the region between the `<!-- strata:agent:start -->` /
  `<!-- strata:agent:end -->` markers is managed; your own notes in the same file
  are preserved and rewritten around.

Both are safe to commit; they make the notebook agent-ready for anyone who
clones it.

## Options

| Flag | Effect |
| --- | --- |
| `--server URL` | Server base URL; reused if already running there, else started. Default `http://localhost:8765` (or `$STRATA_TUI_SERVER`). |
| `--python X.Y` | Python `major.minor` for a newly created notebook's venv. |
| `--no-env` | When creating a notebook, skip building its venv now. |
| `--no-tui` | Set up and open the session but don't attach the TUI (useful when you keep a browser tab open instead, or drive the launcher from a script). |
| `--worker-ssh user@host` | Provision a remote worker over SSH and route cells to it (see below). |

## Running cells on a remote machine

Hand the agent an SSH target and it can run the heavy cells on that box: a GPU
machine, a bigger instance, or one closer to the data. The agent calls the
`connect_ssh_worker` tool (its working agreement tells it to when you give it a
target); or wire it up as the session opens with `strata agent … --worker-ssh
user@gpu-box`. Strata installs the worker on the box if needed, tunnels to it,
and makes it the default; keep a specific cell local with `# @worker local`. See
[Run cells on a machine you can SSH to](workers.md#run-cells-on-a-machine-you-can-ssh-to).

## Reusing a server you already run

If a server is already listening at the target URL, `strata agent` reuses it
instead of starting one, but it must have the MCP endpoint enabled, and it must
be allowed to open the notebook. A server confines notebooks to its configured
storage root, so a reused server only opens notebooks that live under that root.
If it can't, stop it and let `strata agent` start one scoped to your notebook.

## Related

- [The notebook as an agent's scratchpad](scratchpad.md): the other use case,
  where the notebook is disposable rather than the deliverable.
- [MCP Server](mcp.md): the full tool list and the endpoint's security model.
- [Terminal Viewer (TUI)](tui.md): the live spectator `strata agent` attaches.
- [Authoring Programmatically](agent-authoring.md): the same ops as a Python API.
- [Distributed Workers](workers.md#run-cells-on-a-machine-you-can-ssh-to): running cells on a remote box over SSH.

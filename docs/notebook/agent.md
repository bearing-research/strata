# Driving a notebook with a coding agent

`strata agent <notebook-dir>` is a one-command on-ramp: it stands up everything
a coding agent (Claude Code) needs to drive a **live** Strata notebook, and
attaches a terminal viewer so you watch it happen in real time.

It exists because the pieces were already there — the [MCP server](mcp.md), the
[`strata` CLI](cli.md) ops, the [terminal viewer](tui.md) — but wiring them
together by hand is a fiddly, ordered dance: enable the MCP endpoint *before*
the server boots, open a session (the agent can't do that itself), point the
agent at the right session, and only then start driving. `strata agent` does all
of it in one step.

## Prerequisites

The `[mcp]` (agent endpoint) and `[tui]` (terminal viewer) extras:

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
3. **opens a session** — the step the agent cannot do itself, since the MCP
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
notebook by adding and running cells — each of which **lights up live in the TUI**
you left running in the first terminal.

When you quit the TUI, the server `strata agent` started is shut down with it. If
you pointed it at a server you started yourself, that server is left running.

## Why it's a good agent scratchpad

Coding agents usually explore by writing throwaway `.py` scripts to a temp
directory and running them — work that vanishes the moment it finishes, can't be
seen, and recomputes from scratch every time. A Strata notebook is a better home
for exactly that kind of scratch work, and the working agreement above points the
agent at it:

- **It persists and it's visible.** Each snippet becomes a versioned,
  content-addressed cell you can see building live in the TUI — not an invisible
  script in `/tmp`.
- **Unchanged work never recomputes.** Every cell is cached by provenance
  (`sha256(inputs + source + env)`), *including a leaf cell that only `print`s* —
  so an agent that re-runs an unchanged diagnostic cell gets its output back
  instantly instead of paying for it again. The expensive step an agent ran ten
  turns ago is still a cache hit now.
- **Side effects stay honest.** A cell that writes a file, calls an API, or reads
  the clock shouldn't replay a stale result. The agent marks those with
  [`# @nocache`](annotations.md#nocache) so they always re-execute; everything
  else stays cached.

The net effect: the agent's exploration is captured, watchable, and free to
re-derive — a durable scratchpad instead of a pile of discarded scripts.

Two things make the agent actually reach for it:

- **One-call runs.** `run_snippet(session_id, source)` (MCP) and `strata cell add
  … -c 'src' --run` (CLI) add a cell **and** run it in a single call, returning
  `stdout` — the same cost as `python -c`, so a cell isn't the slower path.
- **A shipped skill.** `strata-notebook` installs a `strata-scratchpad` skill
  (under `<site-packages>/strata/.agents/skills/`) that a coding agent discovers
  in **any** project — so it knows to use a cached notebook cell for throwaway
  Python even when it isn't launched inside a notebook directory. A human can
  attach a live viewer at any moment with [`strata watch ./scratch`](tui.md).

## Installing the scratchpad skill as a plugin

The skill travels with the `strata-notebook` package, but the most reliable way
to give any Claude Code session the scratchpad behavior is the **plugin**, which
also bundles a `/strata-scratchpad:scratch` command:

```
/plugin marketplace add bearing-research/strata
/plugin install strata-scratchpad@strata
```

Once installed the skill is auto-discovered in every project (the `strata` CLI
still needs to be on `PATH`). The plugin source lives in the repo at
`plugins/strata-scratchpad/`.

## What gets written into the notebook

- **`.mcp.json`** — registers the running server's `/mcp` endpoint as an MCP
  server named `strata-notebook`, so Claude Code auto-connects when launched in
  the directory. Overwritten on each launch (it points at the current port).
- **`CLAUDE.md`** — a working agreement telling the agent to drive the notebook
  through the MCP tools rather than writing throwaway `.py` scripts, so its work
  is captured as versioned, content-addressed, cached cells that you can see.
  Only the region between the `<!-- strata:agent:start -->` /
  `<!-- strata:agent:end -->` markers is managed; your own notes in the same file
  are preserved and rewritten around.

Both are safe to commit — they make the notebook agent-ready for anyone who
clones it.

## Options

| Flag | Effect |
| --- | --- |
| `--server URL` | Server base URL; reused if already running there, else started. Default `http://localhost:8765` (or `$STRATA_TUI_SERVER`). |
| `--python X.Y` | Python `major.minor` for a newly created notebook's venv. |
| `--no-env` | When creating a notebook, skip building its venv now. |
| `--no-tui` | Set up and open the session but don't attach the TUI (useful when you keep a browser tab open instead, or drive the launcher from a script). |

## Reusing a server you already run

If a server is already listening at the target URL, `strata agent` reuses it
instead of starting one — but it must have the MCP endpoint enabled, and it must
be allowed to open the notebook. A server confines notebooks to its configured
storage root, so a reused server only opens notebooks that live under that root.
If it can't, stop it and let `strata agent` start one scoped to your notebook.

## Related

- [MCP Server](mcp.md) — the full tool list and the endpoint's security model.
- [Terminal Viewer (TUI)](tui.md) — the live spectator `strata agent` attaches.
- [Authoring Programmatically](agent-authoring.md) — the same ops as a Python API.

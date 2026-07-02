# MCP Server

Strata Notebook can expose a **running notebook session** to an external coding
agent (Claude Code, and any other [Model Context
Protocol](https://modelcontextprotocol.io) client) over an HTTP endpoint at
`/mcp`. The agent gets the same operations the [`strata` CLI](cli.md) drives —
read, run, author, manage dependencies — but against a **warm session**: its
populated artifact cache and current cell state, not an offline copy.

Because the tools reuse Strata's broadcasting execution paths, the browser UI
and the [terminal viewer](tui.md) double as a **live view** of the agent at
work — cells flip status, outputs render, and new cells appear as the agent
edits them.

## Enable it

The endpoint is **off by default** and is **personal-mode only** — it has no
per-request authentication, so it grants a caller full control of the session
and is safe only behind a loopback, single-user deployment. (Starting the
server in service mode with the flag set is rejected at startup.)

```bash
uv sync --extra mcp          # or: uv tool install "strata-notebook[mcp]"
STRATA_MCP_ENABLED=true uv run python -m strata
```

Then register it with your agent. For Claude Code:

```bash
claude mcp add --transport http strata http://localhost:8765/mcp
```

If the flag is set but the `[mcp]` extra is not installed, the server logs a
warning and starts normally without the endpoint.

## Workflow

Sessions are opened by the notebook UI or the CLI; the MCP tools operate on
sessions that are **already open** (they do not open notebooks from a path).
The typical loop:

1. You open a notebook in the browser (or with `strata`).
2. `list_notebooks` → the agent gets the `session_id`.
3. The agent inspects (`get_notebook` / `get_cell` / `dag` / `status`), edits
   (`add_cell` / `edit_cell` / …), runs (`run_cell` / `run_tests`), and manages
   dependencies (`add_dependency` / `remove_dependency`) — all against that
   session, while you watch it happen in the browser or the TUI.

## Tools

| Tool | Description |
| --- | --- |
| `list_notebooks` | The sessions currently open on the server: `session_id`, `name`, `path`. |
| `get_notebook(session_id)` | Every cell of a session, in order. |
| `get_cell(session_id, cell_id)` | One cell: source, status, outputs. |
| `dag(session_id)` | The dependency graph — edges, topological order, roots, leaves. |
| `status(session_id)` | Per-cell status + staleness summary. |
| `run_cell(session_id, cell_id, mode)` | Execute a cell (`normal` / `rerun` / `force`), broadcast live. |
| `run_tests(session_id, cell_id)` | Run a cell's `cells/{id}.test.py`. |
| `add_cell(session_id, source, after?, language?)` | Add a cell (server mints the id). |
| `edit_cell(session_id, cell_id, source)` | Replace a cell's source. |
| `remove_cell(session_id, cell_id)` | Delete a cell and its files. |
| `move_cell(session_id, cell_id, index)` | Reorder a cell. |
| `add_dependency(session_id, package)` | `uv add` a dependency. |
| `remove_dependency(session_id, package)` | `uv remove` a dependency. |
| `note(session_id, message)` | Post a line into the Agent panel for the human watching. |

`run_cell` modes match the UI and CLI: `normal` uses the cache and re-runs stale
upstreams first; `rerun` bypasses the target's cache but still refreshes
upstreams; `force` ("run this only") runs against whatever upstream artifacts
already exist.

## Watching the agent

The built-in AI panel streams its own reasoning into the **Agent panel**. An
external agent's reasoning lives in its own client, so instead its **tool
actions are narrated there automatically** — "ran cell abc → ok", "added python
cell def", "added dependency polars" — as it works. It can also call the `note`
tool to post an explicit line of narration ("about to refactor featurize into
two cells"). Open the notebook in the browser or the
[terminal viewer](tui.md) and you can follow along in real time.

## Relationship to the CLI

The MCP tools and the `strata` CLI share one operation contract
([`NotebookOps`](agent-authoring.md)) and return the same curated views. The CLI
is the right tool for **offline / headless** authoring (write files, `strata
run`); the MCP server is for driving a **live, warm session** — rich outputs,
partial re-runs against a populated cache, and edits a human watches in real
time. See [Authoring Programmatically](agent-authoring.md) for the file + CLI
loop that needs no server at all.

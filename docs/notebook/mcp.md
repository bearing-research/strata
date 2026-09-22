# MCP Server

Strata Notebook can expose a **running notebook session** to an external coding
agent (Claude Code, and any other [Model Context
Protocol](https://modelcontextprotocol.io) client) over an HTTP endpoint at
`/mcp`. The agent gets the same operations the [`strata` CLI](cli.md) drives -
read, run, author, manage dependencies - but against a **warm session**: its
populated artifact cache and current cell state, not an offline copy.

Because the tools reuse Strata's broadcasting execution paths, the browser UI
and the [terminal viewer](tui.md) double as a **live view** of the agent at
work - cells flip status, outputs render, and new cells appear as the agent
edits them.

## Enable it

The endpoint is **off by default**. On a personal server there is one user and
nothing to check, so a caller has full control of the session. Keep it behind
loopback.

On a **service-mode** server it needs principal auth (`auth_mode` of
`trusted_proxy` or `api_key`). Service mode without principal auth is rejected
at startup when the flag is set. Each tool call then runs as the caller its HTTP
request names. Authorship, team-store attribution and the registry audit record
that caller, and the call is checked against the same notebook scopes as the
REST routes and the WebSocket:

| Scope | Tools |
|---|---|
| `notebook:read` | `list_notebooks`, `get_notebook`, `get_cell`, `save_cell_output`, `get_variable`, `dag`, `status`, `list_workers`, `lineage`, `publish_preflight` |
| `notebook:write` | `add_cell`, `edit_cell`, `remove_cell`, `move_cell`, `note`, `add_worker`, `set_default_worker`, `set_variant`, `remove_worker`, `disconnect_ssh_worker`, `promote` |
| `notebook:execute` | `run_cell`, `run_tests`, `run_snippet`, `set_widget_value`, `add_dependency`, `remove_dependency`, `connect_ssh_worker`, and any tool not listed |
| `artifacts:publish` | `publish` |

Every open session on the server is visible to any caller holding
`notebook:read`: a server serves one organization, and the check is scopes, not
ownership.

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

!!! tip "One-command setup"
    [`strata agent <notebook-dir>`](agent.md) does the enable-open-register-watch
    steps below for you: it starts the server with this endpoint on, opens a
    session, writes the `.mcp.json` an agent auto-connects to, and attaches the
    TUI. Reach for it unless you want to wire the pieces up by hand.

## Workflow

Sessions are opened by the notebook UI or the CLI; the MCP tools operate on
sessions that are **already open** (they do not open notebooks from a path).
The typical loop:

1. You open a notebook in the browser (or with `strata`).
2. `list_notebooks` → the agent gets the `session_id`.
3. The agent inspects (`get_notebook` / `get_cell` / `dag` / `status`), edits
   (`add_cell` / `edit_cell` / …), runs (`run_cell` / `run_tests`), and manages
   dependencies (`add_dependency` / `remove_dependency`) - all against that
   session, while you watch it happen in the browser or the TUI.

## Tools

| Tool | Description |
| --- | --- |
| `list_notebooks` | The sessions currently open on the server: `session_id`, `name`, `path`. |
| `get_notebook(session_id)` | Every cell of a session, in order. |
| `get_cell(session_id, cell_id)` | One cell: source, status, outputs. For a widget cell, its controls and what each is set to. |
| `save_cell_output(session_id, cell_id, index=-1)` | Write a display output (a plot, an image) to the notebook's `.strata/outputs/` and return the path, so the agent can open it. The path is on the **server's** machine: an agent on another host should use `strata cell output --server … --session …`, which downloads and writes the file locally. |
| `get_variable(session_id, name)` | The cell that defines a variable, "do I already have `name`?"; else the available names. For a swept variable, each variant's cell and the name `lineage` takes for it. |
| `set_variant(session_id, group, active?, mode?)` | Pick which variant of a group runs, or set the group to `switch` / `sweep`. |
| `dag(session_id)` | The dependency graph - edges, topological order, roots, leaves. |
| `status(session_id)` | Per-cell status + staleness summary. |
| `run_cell(session_id, cell_id, mode)` | Execute a cell (`normal` / `rerun` / `force`), broadcast live. |
| `set_widget_value(session_id, cell_id, values)` | Set a widget cell's controls and re-run it at the new values, the same thing moving the slider does. |
| `run_tests(session_id, cell_id)` | Run a cell's `cells/{id}.test.py`. |
| `add_cell(session_id, source, after?, language?, author?)` | Add a cell (server mints the id). `language` is one of `python`, `markdown`, `sql`, `r`, `prompt`, `widget`. |
| `run_snippet(session_id, source, after?, language?, author?)` | Add a cell **and run it** in one call; returns the cell view with the run outcome nested under `run`. The scratchpad primitive. |
| `edit_cell(session_id, cell_id, source, author?)` | Replace a cell's source. |
| `remove_cell(session_id, cell_id)` | Delete a cell and its files. |
| `move_cell(session_id, cell_id, index)` | Reorder a cell. |
| `add_dependency(session_id, package)` | `uv add` a dependency. |
| `remove_dependency(session_id, package)` | `uv remove` a dependency. |
| `note(session_id, message)` | Post a line into the Agent panel for the human watching. |
| `list_workers(session_id)` | The notebook's registered workers and which is the default. |
| `add_worker(session_id, name, url, transport?, token_env?, runtime_id?, set_default?)` | Register a remote executor worker so cells can run on it. |
| `set_default_worker(session_id, name?)` | Set the notebook's default worker (`name` omitted/`local` clears it). |
| `remove_worker(session_id, name)` | Remove a notebook-scoped worker. |
| `connect_ssh_worker(session_id, ssh_target, name?, set_default?, install?)` | Provision + tunnel + register a worker on a machine you reach over SSH (see [Distributed Workers](workers.md#run-cells-on-a-machine-you-can-ssh-to)). |
| `disconnect_ssh_worker(session_id, name, stop_remote?)` | Close an SSH worker's tunnel and unregister it. |
| `lineage(session_id, cell_id, variable, max_depth?)` | The chain behind one of a cell's outputs: every step with the code it ran, the environment it ran in, who computed it and the digest of its bytes. For one instance of a `# @per_variant` cell, name it `variable@variant=<name>` (for example `score@variant=triple`); the error on a bare name lists the stored ones. |
| `promote(session_id, cell_id, variable, name, alias?, tags?)` | Copy the output **and everything behind it** into the team's store, under a name colleagues can ask for. Mints no public link. A protected alias comes back `pending`. Needs `STRATA_NOTEBOOK_REMOTE_STORE_URL`. |
| `publish_preflight(session_id, cell_id, variable)` | What publishing would expose - the whole chain, step by step. Read this to the user before `publish`. |
| `publish(session_id, cell_id, variable, title?)` | Mint a URL that needs no credentials. Copies the chain into the store the link resolves from first. |

Pass the same `author` on every authoring call — your own name or id. It is
recorded on the cell as `created_by` / `updated_by` and shown in the cell view,
so a person opening the notebook can tell which cells an agent wrote. On a
server that authenticates its callers the authenticated identity is used
instead and `author` is ignored; on a personal server there is nothing to check
it against, so it is a claim rather than a fact.

`promote` and `publish` are different acts. Promoting puts a result where
colleagues' cells already read from, inside a store that still needs
credentials; publishing mints a link that needs none and exposes every upstream
step's code and environment along with the result. `publish_preflight` returns
that exposure list, and an agent should put it in front of the user and get
their agreement before calling `publish`. Withdrawing is
`strata artifact unpublish <token>`.

A **widget cell** declares controls; what they are set to is runtime state,
not source. Editing the cell therefore cannot change what the notebook
computes, and reading the source cannot tell you what it is computing.
`get_cell` reports a widget's `controls` with each one's kind, declared default
and current `value`, and `set_widget_value(session_id, cell_id,
{"utilization": 0.9})` sets them and re-materializes the cell, marking
everything downstream stale. Send only the controls you are changing. A
notebook that is mid-run refuses the call and leaves the stored values alone,
so retry once that run finishes.

`run_cell` modes match the UI and CLI: `normal` uses the cache and re-runs stale
upstreams first; `rerun` bypasses the target's cache but still refreshes
upstreams; `force` ("run this only") runs against whatever upstream artifacts
already exist.

`set_variant` drives [variant groups](annotations.md#variant-cells):
`set_variant(session_id, "policy", active="service")` picks one variant,
`set_variant(session_id, "policy", mode="sweep")` runs them all, and either
argument can be sent alone. It makes the same two calls the UI's tab strip
does, so staleness recomputes against the new selection; the equivalent REST
route is `PUT /v1/notebooks/{session_id}/variant-groups/{group}`.

For a swept variable, `get_variable` names each instance and the spelling
`lineage` takes for it (`score@variant=triple` for a `# @per_variant` cell's
instances; the plain name on the member cell for a sweep group).

## Watching the agent

The built-in AI panel streams its own reasoning into the **Agent panel**. An
external agent's reasoning lives in its own client, so instead its **tool
actions are narrated there automatically** - "ran cell abc → ok", "added python
cell def", "added dependency polars" - as it works. It can also call the `note`
tool to post an explicit line of narration ("about to refactor featurize into
two cells"). Open the notebook in the browser or the
[terminal viewer](tui.md) and you can follow along in real time.

## Relationship to the CLI

The MCP tools and the `strata` CLI share one operation contract
([`NotebookOps`](agent-authoring.md)) and return the same curated views. The CLI
is the right tool for **offline / headless** authoring (write files, `strata
run`); the MCP server is for driving a **live, warm session** - rich outputs,
partial re-runs against a populated cache, and edits a human watches in real
time. See [Authoring Programmatically](agent-authoring.md) for the file + CLI
loop that needs no server at all.

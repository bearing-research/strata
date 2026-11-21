# MCP server

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
| `notebook:read` | `list_notebooks`, `get_notebook`, `get_cell`, `save_cell_output`, `get_variable`, `dag`, `status`, `list_workers`, `lineage`, `publish_preflight`, `scratchpad` |
| `notebook:write` | `add_cell`, `edit_cell`, `remove_cell`, `move_cell`, `note`, `add_worker`, `set_default_worker`, `set_variant`, `remove_worker`, `disconnect_ssh_worker`, `promote` |
| `notebook:write` and `notebook:execute` | `run_snippet`, `run`: each writes a cell's source, then runs it |
| `notebook:execute` | `run_cell`, `cancel_cell`, `run_tests`, `set_widget_value`, `add_dependency`, `remove_dependency`, `connect_ssh_worker`, and any tool not listed |
| `artifacts:publish` | `publish` |

Within a tenant, every open session is visible to any caller holding
`notebook:read`: the check is scopes, not ownership. A session records the
tenant of the caller who opened or created it, and a caller from another tenant
neither sees it in `list_notebooks` nor reaches it by `session_id`, as with
tenant-scoped artifacts. `admin:*` sees every tenant's sessions, and a caller
the proxy sends without a tenant is the default tenant (`_default`), so its
sessions are visible only to other tenantless callers and to `admin:*`. The
notebook REST routes and the WebSocket apply the same rule.

=== "From PyPI"

    ```bash
    uv tool install "strata-notebook[mcp,tui]"
    STRATA_MCP_ENABLED=true strata-notebook
    ```

=== "From source"

    ```bash
    uv sync --extra mcp
    STRATA_MCP_ENABLED=true uv run python -m strata
    ```

Then register it with your agent. For Claude Code:

```bash
claude mcp add --transport http strata http://localhost:8765/mcp
```

An agent on another machine uses the server's name or address instead
(`http://devbox.lan:8765/mcp`). `/mcp` answers to the same `Host` names as
the rest of the server: `localhost`, any IP literal, `STRATA_HOST` and the
names in `STRATA_ALLOWED_HOSTS` (see
[Deployment modes](../deployment/modes.md#personal-mode)). Any other `Host`
is refused (in service mode, only once `STRATA_ALLOWED_HOSTS` is set). A request from a browser page must also be same-origin or come
from an origin in `STRATA_CORS_ALLOW_ORIGINS`.

The `[mcp]` extra needs `mcp` 2.2 or newer. If the flag is set but the extra is
not installed, the server logs a warning and starts normally without the
endpoint. With `mcp` 1 installed (another tool may pin it), `/mcp` stays off
the same way and the server logs that the installed `mcp` is too old.

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
   (`add_cell` / `edit_cell` / …), runs (`run_cell` / `run_tests`, and `cancel_cell` to stop one), and manages
   dependencies (`add_dependency` / `remove_dependency`) - all against that
   session, while you watch it happen in the browser or the TUI.

A call that fails for a reason the agent can act on (an unknown session or
cell, a bad argument, a busy notebook) returns that reason in the tool error,
for example `no open notebook session '…'; call list_notebooks first`. An
unexpected server failure returns only `Error executing tool <name>`; the
details stay in the server log.

## Scratchpad profile

With `STRATA_MCP_PROFILE=scratchpad` the server lists two tools instead of the
full set, both over the server's **one** open session, so neither takes a
`session_id`. The initialize response carries instructions telling the agent
when to reach for them, and both tools are marked to stay loaded rather than
sit behind Claude Code's tool search.

| Tool | Description |
| --- | --- |
| `run(source, cell=None, fresh=False)` | Without `cell`, add a Python cell and run it; with `cell`, replace that cell's source and run it. `fresh=True` ignores this cell's cache (the `rerun` mode). Returns `cell`, `status`, `cache_hit`, `duration_ms`, `saved_ms` on a hit, `reran` (the upstream cells brought up to date first, each with why), `value` (what a trailing expression showed), `stdout` / `stderr`, `traceback`, `diagnostics` (files it read without declaring them, or values it mutated, each with its fix), `defines`, `path` (a file holding an image the cell showed) and `url` (the notebook in the web UI). A `cell` that is not Python is refused. |
| `scratchpad(name=None)` | Without `name`: each cell's id, `defines`, `status`, why it is stale and what its last uncached run took, plus the notebook directory, the cells' `cwd` (its `work/`), the `python` they run on, the `environment` that is (`attached` to the project's, with its `project_python` and what found it, or `owned`) and the time the cache has saved. With `name`: the cell that defines it, its source and shown `value`, or the names that exist. |

A failure that is not the cell's own exception (no session or several, an
unknown or non-Python cell, an environment that is not ready, an annotation that
cannot be read, a busy notebook, which leaves the cells as they were)
returns `status: error` with a `fix` line naming the next step. The other tools
stay registered and callable by name; `STRATA_MCP_TOOLS=all` lists them too.

### A project's own daemon

A scratchpad belongs to a project, found from where a command runs: the nearest
directory at or above it that holds a `pyproject.toml`, searched no higher than
the git repository's root or your home directory, whichever is nearer. So in a
monorepo, `services/api/` with its own `pyproject.toml` is its own project, and a
command run in its `src/` serves `services/api/`'s scratchpad. With no
`pyproject.toml` found, the repository is the project (never your home directory,
should it be one), or outside one the directory itself. A command run inside a
notebook, `.strata-scratch/` included, serves the project the notebook is in.
`--root DIR` on any `strata scratch` command or on `strata mcp`, or
`STRATA_SCRATCH_ROOT` for a config that cannot pass arguments, names the project
instead.

`strata scratch start` in a project creates `.strata-scratch/` at its root, a
notebook with its own `.gitignore` of `*`, so the project's git never sees it and
its `.gitignore` is left alone. It then starts a server for it on a free loopback
port, with the scratchpad profile and
`STRATA_IDLE_EXIT_MINUTES=30` (or the value already set), opens the notebook and
waits for its environment. The daemon's address, pid and session are written to
`.strata-scratch/.strata/server.json`, so the next client finds it instead of
starting another, and a file lock makes two clients asking at once share one.
A daemon another Strata version started is stopped and replaced once it is idle
(`GET /health/busy`); while it has a request, a socket or a run in flight it serves the
call as it is, the result's `note` says so, and the next call asks again. The server's
output goes to `.strata-scratch/.strata/server.log`; it keeps remote-mount copies
and shared environments under `~/.strata`, as a personal server does.
Cells read the project read-only as `project` (`project / "data.csv"`).

Cells run in the project's own environment when it has one: the one `uv run
python` would use there, found by uv's dry run, which creates nothing. A project
is uv's when its workspace root has a `uv.lock` or its `pyproject.toml` has a
`[tool.uv]` table; the scratchpad then runs `uv sync --inexact` first when uv says
the environment is missing or behind, as `uv run` does, and a sync that fails comes
back from `run` with uv's error, and no cell runs until it succeeds. A project uv
does not manage (pip, poetry) is used as it is, never synced: the environment
`VIRTUAL_ENV` names, unless it is the one Strata runs in, else a `.venv` in the
project. Anything else (no environment, a project in another language, no
project) gets one of its own, `.strata-scratch/.venv`, where `add_dependency`
works; in the project's, it is refused with the project's command (`uv add
numpy`). The cells' interpreter is `.strata-scratch/.venv-overlay`, which adds the
project's site-packages and only those of `pyarrow`, `orjson` and `cloudpickle`
the project lacks (at Strata's minimum version, without their dependencies), so
the project's versions win; a project whose own is older than Strata needs is
refused with that named, since the overlay cannot replace it. Cells there are keyed
on the environment's installed distributions (name and version, plus what one from
a URL, a repository or a path installed; editable ones aside, whose code is keyed
by what runs read), the interpreter's build and where imports resolve: the
search path, the import hooks, where each editable points, the `PYTHON*`
variables that move imports, and whether a file in the project would shadow a
module loaded before the cell started (a new `orjson.py` there changes every
key). That is checked before every lookup, with a few hundred stats, and probed
again when it changed, so an install, a repointed editable or a new `PYTHONPATH`
misses rather than replays. A scratchpad with an environment of its own is keyed on
where its imports resolve too, beside its `uv.lock`.

Before a scratchpad's cell starts, the modules its project and the scratchpad
declare as dependencies are imported, on a warm worker and on a cold run alike, so
a cell's result does not depend on which ran it. A dependency whose import loads
code a run would record (an editable sibling, the project's own package, or an
installed one that imports either) is left out, and is imported by the cell, which
records it. The list is part of the key and re-derived when a `pyproject.toml`
changes. A warm worker built before the environment changed, or before a file in
the project began to shadow a module it loaded, is replaced when claimed, never
used. The read audit counts what is
installed in the project's environment as the environment and does not warn about it. Before each
`run`, a change to the project's `pyproject.toml` files, its `uv.lock` or its
installed packages syncs it again, once no cell is running; whether a project has an environment at all is
decided again when the scratchpad opens, and recorded in
`.strata-scratch/.strata/scratchpad_environment.json`. `--python PATH` on `strata
mcp` or `strata scratch start`, or `STRATA_SCRATCH_PYTHON`, names the interpreter
instead. A client asking for another than the daemon's replaces it once idle, and
while it is busy is refused with both named; a client asking for none keeps the
daemon's. A scratchpad cell's `# @env` may not set `PYTHONPATH`, `PYTHONHOME`,
`PYTHONSAFEPATH`, `PYTHONNOUSERSITE` or `PYTHONPLATLIBDIR` (`env_moves_imports`,
which keeps it from running): its key does not follow where imports resolve.
Its `notebook.toml` says `scratchpad = true`, so it keeps every variable a cell defines,
read or not: the first cell to read one loads it rather than rerunning the cell that
made it. A plain notebook stores only what a cell downstream reads.

A scratchpad's cells run as under `python -c` in the project: its folder is first on
`sys.path`, so a module beside them imports, and shadows an installed one of its
name. They run on this machine only, in either environment mode: a `# @worker` other
than `local`, or a default worker, is an error that keeps the cell from running, and
`add_worker`, `set_default_worker` and `connect_ssh_worker` refuse a scratchpad
(for remote compute, use a notebook). A scratchpad's Python cell is also keyed on
what its run read: each file it opened or directory it listed (pyarrow's, polars'
and duckdb's readers included), the code it imported from outside its environment,
and each place an import looked and found nothing. A file it only checked with
`exists()`, or one a subprocess read, is not recorded. Edit `project / "config.yaml"` or a module
of the project and the cells that read it go stale with reason `code`
(`changed_code` names the file), then run again; put the file back and they hit
again. A cell that changed a file it had read, or whose file changed while it ran,
keeps its outputs for that run only, as does a loop or `@each` cell that read
files. Since such a run is keyed on what it imported, opened and listed, the read
audit does not warn about those reads. Run All runs a scratchpad's cells one at a time, so each run records its
own reads, and a run never resumes a dead run's checkpoint, whose reads are unknown.
The project's modules are compiled from the very source a run read, never from a
`.pyc` that may predate it; the compiled code is kept in
`.strata-scratch/.strata/pycache/` under the source's sha256, its path and the
interpreter (256 MiB, least recently used first), so an unchanged module is not
compiled again. A damaged or misplaced entry, one whose checksum or key does not
match, is compiled again, never run; like the rest of `.strata/`, the cache trusts
what writes there. A module whose loader compiles code of its own (an instrumenting
import hook) is never cached, and with `notebook_harness_user` set nothing is.
`strata scratch status` prints the running daemon and `strata scratch stop`
stops it. Once it is stopped, deleting `.strata-scratch/` costs reruns, nothing else.

A scratchpad keeps superseded results, so reverting an edit hits again, and its
daemon sweeps it at start and hourly instead of keeping a count per value:
results unused for 14 days go, then the least recently used while the store is
over its 5 GiB soft limit (`STRATA_NOTEBOOK_SCRATCH_GC_*` in
[Configuration](../reference/configuration.md)). Live cells' values are never
swept, so the limit can be exceeded; when it is, `scratchpad` returns a
`storage` entry with the overage and the cells holding the most. `strata scratch
ls` lists every scratchpad with its size, last use, overage and whether its
daemon is up; `strata scratch gc` sweeps one now (`--all` for every one) and
`strata scratch rm` stops its daemon and deletes it. With its daemon running, `gc`
leaves retiring cells to the daemon, which does it only when nothing runs.

`run` adds a cell per call, so cells have a way out too. A cell nothing reads,
with no `# @name`, that has not been run, edited or shown by `scratchpad` for 14
days (`STRATA_NOTEBOOK_SCRATCH_RETIRE_DAYS`) is retired before the sweep: its
source and tests go to `.strata-scratch/retired.jsonl`, it leaves the notebook, and the
sweep frees every value it stored unless something else holds one (a name, a
pin, a live cell reading it). A cell it read may then retire the same way. A
later `run` that needs a retired cell's variable, a `run` naming its id, or a
`scratchpad` look for its variable answers with a `fix` naming the cell, when it
retired, and its source to run again. `strata scratch retire
<cell>` retires one by hand, unless another cell reads it.

### From an agent's config: `strata mcp`

`strata mcp` serves the project's scratchpad over stdio, the form an agent's MCP
config starts as a command:

```json
{"mcpServers": {"strata": {"command": "strata", "args": ["mcp"]}}}
```

It answers the handshake and the tool list itself. In a project that already has
a scratchpad (`.strata-scratch/`), the session's first tool list starts the
project's daemon in the background, so the first call does not wait for it; a
call made while it starts waits for that one, hearing its steps. A client that
only connects, as a health check does, starts nothing. Elsewhere a session that never calls a
tool starts nothing: the first tool call finds the project's daemon or starts it
(creating `.strata-scratch/`). Every call runs there. The
project is the one `strata mcp` starts in
([A project's own daemon](#a-projects-own-daemon)). While the scratchpad is set
up, each step is sent as MCP progress when the call asks for it, and the result
of a call that had to set it up carries `setup_ms`, the
milliseconds each step took. A call that cannot reach the daemon (it stopped while
idle) starts it again and is sent once more; a call the agent cancels is cancelled
on the daemon too. Needs the `[mcp]` extra.

### Install it into the agent: `strata agent install`

```bash
strata agent install            # every agent whose CLI is on PATH; asks first
strata agent install --target claude --yes
strata agent uninstall
```

`install` writes, for Claude Code, the user-scope MCP entry above (through
`claude mcp add-json`, with the absolute path of the `strata` it runs as and
`alwaysLoad`), the `mcp__strata__*` permission in `~/.claude/settings.json`, and a
short block in `~/.claude/CLAUDE.md` naming the tools and the shell commands
below, for subagents and harnesses that do not see MCP instructions. For Codex it
writes an `[mcp_servers.strata]` block in `~/.codex/config.toml` and the same block
in `~/.codex/AGENTS.md`. `CLAUDE_CONFIG_DIR` and `CODEX_HOME` are honoured. A
hand-copied `strata-scratchpad` skill it finds is moved to
`~/.strata/agent-install/retired-skills/`, since the server and the block replace it,
and `uninstall` puts it back.

For Claude Code it also adds two hooks to `settings.json`, at session start and on each
prompt. They tell the agent what the project's scratchpad already holds before it
computes anything: each cell's variables, what it took to build, and whether it is
`cached`, `stale` (and which file changed, or `upstream`) or `failed`. The prompt hook
says only the cells whose id or variables the prompt names. The hooks read a digest the
daemon writes (`.strata-scratch/.strata/digest.json`) and look at each cell's recorded
reads and the environment's watched paths again first, so a file edited while the daemon
was stopped makes its cell stale; what one run's budget (1000 paths, 80 ms) leaves
unchecked is said `as of` the digest's time, never `cached`, as is a cell downstream of
one the digest does not hold (a SQL, prompt or R cell, or one not run). They never start
the daemon.

Every piece is marker-fenced or keyed. `uninstall` puts back the original bytes of a
file left as `install` wrote it and removes only Strata's part from one edited since.
It works from the directories `install` wrote into, whatever `CLAUDE_CONFIG_DIR` says
now. A `strata` MCP server of your own (Claude Code's, or a `config.toml`
`[mcp_servers.strata]`), a missing `claude` CLI, or a `settings.json` that is not valid
JSON, is refused before anything is written: remove yours (`claude mcp remove --scope
user strata`) and install again. An install that fails halfway is recorded, so
`uninstall` still removes what it wrote. With no `claude` CLI to remove Strata's MCP
entry, `uninstall` leaves the Claude Code target as installed, its server being still
configured: run it again with `claude` on PATH, or remove the entry from
`~/.claude.json` yourself first. The `strata` written is the one
that runs `install`: refused from uv's cache (`uvx`), and warned about from a project's
environment, since every project's agent would use it; `uv tool install` is the
lasting choice. Ending `strata scratch run` (Ctrl-C, or a shell tool's timeout
terminating it) cancels its cell on the daemon, or the daemon's setup, killing a daemon
not yet recorded. When another Strata version
wrote the files, `strata mcp` rewrites them as it starts.

### From a shell: `strata scratch run` and `show`

```bash
strata scratch run -c 'import pandas as pd; df = pd.read_parquet("/repo/data.parquet"); df.shape'
strata scratch run --cell c1a2b3c4 -c 'df.describe()'   # replace that cell's source
strata scratch show df                                   # the cell defining df, its value
```

The `run` and `scratchpad` tools for an agent that has only a shell: each command
is one tool call on the project's daemon, starting it if none runs, and prints the
tool's result as JSON. `run` exits 1 when the cell failed; `--file` reads the source
from a file (`-` for stdin) and `--fresh` runs it again whatever its cache says.
Needs the `[mcp]` extra.

## Tools

| Tool | Description |
| --- | --- |
| `list_notebooks` | The sessions currently open on the server: `session_id`, `name`, `path`. |
| `get_notebook(session_id)` | Every cell of a session, in order. |
| `get_cell(session_id, cell_id)` | One cell: source, status, outputs. A failed cell has its `error` and `error_full`, the traceback from the cell's first frame and the one with the harness frames too. For a widget cell, its controls and what each is set to. |
| `save_cell_output(session_id, cell_id, index=-1)` | Write a display output (a plot, an image) to the notebook's `.strata/outputs/` and return the path, so the agent can open it. The path is on the **server's** machine: an agent on another host should use `strata cell output --server … --session …`, which downloads and writes the file locally. |
| `get_variable(session_id, name)` | The cell that defines a variable, "do I already have `name`?"; else the available names. For a swept variable, each variant's cell and the name `lineage` takes for it. |
| `set_variant(session_id, group, active?, mode?)` | Pick which variant of a group runs, or set the group to `switch` / `sweep`. |
| `dag(session_id)` | The dependency graph - edges, topological order, roots, leaves. |
| `status(session_id)` | Per-cell status + staleness summary. A ready cell's `changed_reads` lists files it read without declaring them that changed since: its result may be out of date, and only a `rerun` reads them again. A scratchpad cell stale for `code` has `changed_code`, the file its last run read that changed. |
| `run_cell(session_id, cell_id, mode, confirm_reads, approve_reads)` | Execute a cell (`normal` / `rerun` / `force`), broadcast live. A cell that raised has its `traceback` and `traceback_full`; `run_diagnostics` names what the run read without declaring it, each with its fix. With `confirm_reads`, a run over the server's read threshold returns `status: error`, `error_code: large_read` and the estimate in `reads` instead; `approve_reads` runs it. |
| `cancel_cell(session_id, cell_id)` | Cancel the cell's run, whoever started it, as the browser's stop button does: the run stops and the cell is left idle. A `run_cell` waiting on it returns `status: error` with `error_code: cancelled`. Returns `{cell_id, cancelled, status}`; `cancelled` is `false` when the cell was not running. |
| `set_widget_value(session_id, cell_id, values)` | Set a widget cell's controls and re-run it at the new values, the same thing moving the slider does. |
| `run_tests(session_id, cell_id)` | Run a cell's `cells/{id}.test.py`. |
| `add_cell(session_id, source, after?, language?, author?)` | Add a cell (server mints the id). `language` is one of `python`, `markdown`, `sql`, `r`, `prompt`, `widget`. |
| `run_snippet(session_id, source, after?, language?, author?)` | Add a cell **and run it** in one call; returns the cell view with the run outcome nested under `run`. The scratchpad primitive. |
| `edit_cell(session_id, cell_id, source, author?)` | Replace a cell's source. |
| `remove_cell(session_id, cell_id)` | Delete a cell and its files. |
| `move_cell(session_id, cell_id, index)` | Reorder a cell. |
| `add_dependency(session_id, package)` | `uv add` a dependency. |
| `remove_dependency(session_id, package)` | `uv remove` a dependency. |
| `note(session_id, message)` | Post a line into the terminal viewer's Agent tab for the human watching. |
| `list_workers(session_id)` | The notebook's registered workers and which is the default. |
| `add_worker(session_id, name, url, transport?, token_env?, runtime_id?, set_default?)` | Register a remote executor worker so cells can run on it. Refused where the server manages worker definitions (service mode, or `STRATA_NOTEBOOK_WORKERS_EDITABLE=false`), as are `remove_worker` and `connect_ssh_worker`. |
| `set_default_worker(session_id, name?)` | Set the notebook's default worker (`name` omitted/`local` clears it). |
| `remove_worker(session_id, name)` | Remove a notebook-scoped worker. |
| `connect_ssh_worker(session_id, ssh_target, name?, set_default?, install?)` | Provision + tunnel + register a worker on a machine you reach over SSH (see [Workers](workers.md#run-cells-on-a-machine-you-can-ssh-to)). |
| `disconnect_ssh_worker(session_id, name, stop_remote?)` | Close an SSH worker's tunnel and unregister it. |
| `lineage(session_id, cell_id, variable, max_depth?)` | The chain behind one of a cell's outputs: every step with the code it ran, the environment it ran in, who computed it and the digest of its bytes. For one instance of a `# @per_variant` cell, name it `variable@variant=<name>` (for example `score@variant=triple`); the error on a bare name lists the stored ones. `variable` may be `__mount__<name>`, what a run wrote through its `rw` mount `<name>`. |
| `promote(session_id, cell_id, variable, name, alias?, tags?)` | Copy the output **and everything behind it** into the team's store, under a name colleagues can ask for. Mints no public link. A protected alias comes back `pending`. Needs `STRATA_NOTEBOOK_REMOTE_STORE_URL`. |
| `publish_preflight(session_id, cell_id, variable)` | What publishing would expose - the whole chain, step by step. Read this to the user before `publish`. |
| `publish(session_id, cell_id, variable, title?)` | Mint a URL that needs no credentials. Copies the chain into the store the link resolves from first. |

Pass the same `author` on every authoring call: your own name or id. It is
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
their agreement before calling `publish`. `publish` returns the `token` only
when it mints the link; for a version already published it is empty, with the
publication's `id`. Withdrawing is `strata artifact unpublish <token or id>`.

A **widget cell** declares controls; what they are set to is runtime state,
not source. Editing the cell therefore cannot change what the notebook
computes, and reading the source cannot tell you what it is computing.
`get_cell` reports a widget's `controls` with each one's kind, declared default
and current `value` (the default itself, for a control nobody has moved, since
that is what the cell runs at), and `set_widget_value(session_id, cell_id,
{"utilization": 0.9})` sets them and re-materializes the cell, marking
everything downstream stale. On a [`# @live`](../guides/cells/widget.md#live-mode) widget it
runs the same cost-gated cascade a drag does, so the cheap downstream cells
come back computed rather than stale. Send only the controls you are changing.
A notebook that is mid-run refuses the call and leaves the stored values alone,
so retry once that run finishes.

`run_cell` modes match the UI and CLI: `normal` uses the cache and re-runs stale
upstreams first; `rerun` bypasses the target's cache but still refreshes
upstreams; `force` ("run this only") runs against whatever upstream artifacts
already exist.

On a server that [asks before a large read](lake-aware-cells.md#asking-before-a-large-read),
pass `confirm_reads=true` so a run that would move more than the threshold
returns, rather than starting:

```json
{"cell_id": "...", "status": "error", "error_code": "large_read",
 "error": "<the server's message>",
 "reads": [{"cell_id": "...", "machine": "...", "inputs": [...],
            "bytes_to_move": 0, "cost": 0.0}]}
```

`reads` has one entry per cell of the run (its stale upstreams included) that
reads past the threshold with no approval kept. Ask the user, then call again
with `approve_reads=true`; the yes is remembered for that cell, machine and
inputs, so the same run does not ask again. Without either, the run goes ahead
and the reads index records that it was not asked.

`set_variant` drives [variant groups](../guides/cells/loops-and-variants.md#variant-cells):
`set_variant(session_id, "policy", active="service")` picks one variant,
`set_variant(session_id, "policy", mode="sweep")` runs them all, and either
argument can be sent alone. It makes the same two calls the UI's tab strip
does, so staleness recomputes against the new selection; the equivalent REST
route is `PUT /v1/notebooks/{session_id}/variant-groups/{group}`.

For a swept variable, `get_variable` names each instance and the spelling
`lineage` takes for it (`score@variant=triple` for a `# @per_variant` cell's
instances; the plain name on the member cell for a sweep group).

## Watching the agent

An external agent's reasoning lives in its own client, so instead its **tool
actions are narrated automatically** into the terminal viewer's **Agent** tab -
"ran cell abc → ok", "added python cell def", "added dependency polars" - as it
works. It can also call the `note`
tool to post an explicit line of narration ("about to refactor featurize into
two cells"). Open the notebook in the [terminal viewer](tui.md) and you can
follow along in real time; the browser UI does not show these notes.

## Relationship to the CLI

The MCP tools and the `strata` CLI share one operation contract
([`NotebookOps`](agent-authoring.md)) and return the same curated views. The CLI
is the right tool for **offline / headless** authoring (write files, `strata
run`); the MCP server is for driving a **live, warm session** - rich outputs,
partial re-runs against a populated cache, and edits a human watches in real
time. See [Authoring Programmatically](agent-authoring.md) for the file + CLI
loop that needs no server at all.

# Terminal Viewer (TUI)

`strata-notebook-tui` is a **read-only terminal spectator** for a live notebook
session. It attaches to a running `strata-notebook` server over the same
WebSocket protocol the web UI uses, then renders the notebook and streams every
update - cell status changes, console output, results, cascade and
environment-job progress, and an AI agent's activity - as they happen.

It does not edit or run cells. Its purpose is **watching**: open a notebook in
the web UI (or let an AI agent drive it), and follow along in a terminal - a
second pane, an SSH session, or a tmux window beside your editor.

This is the **terminal** quickstart. Something else has to be driving the
notebook - either you in the [web UI](../getting-started/notebook.md), or a
[coding agent](agent.md).

## Install

The viewer ships behind the `tui` extra (it pulls in `textual` and `grandalf`,
plus `textual-image` and `Pillow` for inline image rendering; `httpx` and
`websockets` are already core dependencies):

```bash
uv tool install "strata-notebook[tui]"      # global: strata-notebook, strata-notebook-tui, strata, strata-worker
# or, inside a project venv:
uv pip install "strata-notebook[tui]"
```

Plain `strata-notebook` (without `[tui]`) installs the server but not the
viewer's dependencies, so `strata-notebook-tui` will fail to start.

## Workflow

The viewer is a client - you always have a **server**, a **session**, and the
**viewer watching it**.

### 1. Start the server

```bash
strata-notebook                # serves http://127.0.0.1:8765 (web UI + REST + WS)
```

### 2. Have a notebook

Scaffold one from the terminal:

```bash
strata new my-notebook         # creates ./my-notebook
```

…or create / open one in the web UI at `:8765`. (`strata-notebook-tui` can open
an existing notebook directory itself - see `--notebook` below - but it can't
create one.)

### 3. Watch it

```bash
strata-notebook-tui --notebook ./my-notebook   # open/reuse a session for that path, then attach
```

Or, if a session is already open (for example you opened the notebook in the web
UI), attach without naming it:

```bash
strata-notebook-tui            # auto-attaches the only running session, or shows a picker
```

The `strata` CLI wraps the same spectator as **`strata watch`**, so you don't need
the separate binary on `PATH`:

```bash
strata watch ./my-notebook     # open/reuse a session for that path, then attach
strata watch --session <id>    # attach to a specific running session
strata watch                   # auto-attach the only session, or show a picker
```

`strata watch` is watch-only: unlike [`strata agent`](agent.md), it writes no
`.mcp.json` / `CLAUDE.md` and never owns the server lifecycle - it just shows a
notebook live. Point it at a scratchpad an agent is driving to see the cells land
in real time.

## Watching an agent

The viewer's headline use case is following an AI agent as it drives a notebook
in another terminal:

1. Start the server and open the notebook in the web UI.
2. Kick off the [AI agent](ai.md) there.
3. In a terminal, run `strata-notebook-tui` and open the **Agent** tab.

The agent's reasoning streams into the Agent tab while cells flip status, the
selection follows the running cell, and results render live. Agent
confirmation prompts are shown as *"awaiting driver confirmation"* - the viewer
is read-only, so the web UI (the driver) answers them.

## Layout and keys

![The terminal viewer: a cell list on the left with status glyphs and last-run
times including a cached entry, and on the right the selected cell's source above
its output rendered as a table.](../assets/tui-layout.svg)

The left pane lists the cells (status glyph, name, and last-run time); the right
pane is split into two tab-groups for the selected cell - the **code** on top
(Source, Tests source) and the **runtime** on the bottom (Output, Console, Agent,
Results). A header line shows the connection state, follow mode, and any
cascade / environment / agent activity.

When the driver runs a cell's [unit tests](testing.md), the result shows as a
badge on that cell in the list - `✓ 4/4` (green), `✗ 2/4` on failure, `·stale`
when the cell changed since the run - and the header notes the test run. The
top **Tests** tab (`3`) shows the test *source*; the bottom **Results** tab (`7`)
shows the individual outcomes for the selected cell: each test as `✓`/`✗`/`⚠`/`○`
with its failure message (the rewritten-assert diff) underneath - the pytest run,
in the terminal.

| Key | Action |
| --- | --- |
| `1` | Focus the cell list (`↑`/`↓` move the selection) |
| `2` / `3` | Top group: Source / Tests source |
| `4` / `5` / `6` / `7` | Bottom group: Output / Console / Agent / Results |
| `↑` `↓` `PgUp` `PgDn` `Home` `End` | Scroll the focused pane |
| `n` / `p` | **Data viewer:** next / previous page of a large table |
| `s` | **Data viewer:** sort by the focused column (asc → desc → off) |
| `e` | **Data viewer:** export the table to a CSV in the current directory |
| `f` | Toggle **follow mode** (auto-select the running cell) |
| `d` | Show the notebook **DAG** (layered ASCII; `Esc`/`d`/`q` to close) |
| `i` | Enlarge the selected cell's **image** output to full screen (`Esc`/`i`/`q` to close) |
| `r` | Force an immediate resync (the viewer also auto-resyncs in the background) |
| `Ctrl`+`←` / `Ctrl`+`→` | Resize the **cell-list ↔ detail** boundary |
| `Ctrl`+`↑` / `Ctrl`+`↓` | Resize the **top ↔ bottom** detail boundary |
| `Ctrl`+`x` | Reset the panel layout to defaults |
| `?` | Show the keybinding reference (`Esc`/`?`/`q` to close) |
| `q` | Quit |

The tabs render richly: source is **syntax-highlighted** by cell language,
markdown cells and markdown outputs render as formatted markdown, a single DataFrame / table output
renders in an **interactive data viewer** - a scrollable table you can page (`n`/`p`),
sort by the focused column (`s`), and export to CSV (`e`), backed by the full
cached artifact rather than the 20-row preview - and image outputs (e.g.
matplotlib figures) render **inline** - using the terminal's graphics protocol
(kitty / iTerm2 / Sixel) where available, degrading to Unicode half-blocks
otherwise.

Cell status glyphs:

| Glyph | Status |
| --- | --- |
| `○` | idle |
| `▶` | running |
| `✓` | ready |
| `✗` | error |
| `⊘` | stale |
| `…` | queued |

## Options

```bash
strata-notebook-tui [--session ID | --notebook PATH] [--server URL] [--user-header NAME --user VALUE]
```

| Option | Purpose |
| --- | --- |
| `--session ID` | Attach to a specific running session id (skips the picker). |
| `--notebook PATH` | Open / reuse a session for a notebook directory path (the path must exist on the server's filesystem). |
| `--server URL` | Base URL of the server (default: `$STRATA_TUI_SERVER` or `http://localhost:8765`). |
| `--user-header NAME` | Identity header name, matching the server's `personal_mode_user_header`. |
| `--user VALUE` | Identity header value - needed to attach to an owned notebook. |

## Authentication

Against a plain local server (the default), no auth flags are needed. If the
server runs behind a proxy with per-user scoping
(`STRATA_PERSONAL_MODE_USER_HEADER`), pass `--user-header` and `--user` matching
what the proxy injects; the WebSocket attach is owner-gated, so a notebook
created under your identity needs the same identity to watch. Point at a
non-default server with `--server` (or the `STRATA_TUI_SERVER` environment
variable). See [Privacy & Sharing](../deployment/privacy.md) for the per-user
scoping model.

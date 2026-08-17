# Recording the agent demo GIF

The 20-second clip that shows a coding agent building this notebook live — and
Strata reusing the cached model when only the evaluation changes. This is the
GIF in the top-level README.

## Setup

- **Two terminal panes, side by side** (iTerm2 split, tmux, or two windows).
  Left = the coding agent. Right = the Strata TUI.
- **Large font** (≥ 16pt) and a **dark theme** — the GIF gets scaled down, so
  legibility matters.
- A scratch directory to build in (not this committed example — the agent writes
  its own copy):

```bash
mkdir /tmp/demo && cd /tmp/demo
```

## Panes

**Right pane — the notebook + live TUI:**

```bash
strata agent .
```

This starts a server with the MCP endpoint on, opens a session, writes
`.mcp.json` + `CLAUDE.md`, and attaches the read-only TUI. Leave it running —
this is what the viewer watches light up.

**Left pane — the agent:**

```bash
cd /tmp/demo && claude
```

Claude Code auto-connects to the `strata-notebook` MCP server and reads the
working agreement.

## The two beats

**Beat 1 — build (agent drives, TUI lights up).** In the left pane:

> Train a random forest on some synthetic data and report its accuracy — build it in the notebook.

The agent adds a training cell and an evaluation cell and runs them. The right
pane shows the cells appear and the model train (~2s), then go green. *This is
the "an agent is driving my notebook" moment.*

**Beat 2 — the cache payoff.** Then:

> Also report the confusion matrix.

The agent edits **only** the evaluation cell. The right pane shows the model cell
stay **cached** — it is not retrained — while just the evaluation recomputes.
*This is the differentiator: content-addressed, never recompute unchanged work.*

## Capture

- **macOS**: record the region spanning both panes with **[Kap](https://getkap.co)**
  (free) → export as GIF. Or QuickTime screen recording → convert with
  [`gifski`](https://gif.ski) for a smaller, sharper GIF.
- Target **15–25 s** and **< 5 MB** (GitHub inlines images up to 10 MB, but
  smaller loads faster). Trim dead air; start on the empty TUI, end on the cached
  model + new confusion matrix.
- Save it to **`docs/assets/agent-demo.gif`**; the follow-up change wires that
  path into the README hero and `docs/notebook/agent.md`.

## Suggested caption

> A coding agent builds a Strata notebook live — and never retrains the model when only the evaluation changes.

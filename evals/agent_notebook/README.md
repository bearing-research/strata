# Agent-notebook eval suite

Does a coding agent, handed a notebook by `strata agent`, actually **drive the
notebook** — or does it route around it with scratch Python? This suite measures
that.

The headline metric is the **in-tool rate**: of the work actions an agent took,
what fraction went through the notebook's MCP tools (`add_cell`, `edit_cell`,
`run_cell`, …) versus an **escape** — a bypass of the notebook. Three escape
modes are detected:

- **bash-python** — running Python/tests via Bash instead of `run_cell`
- **bash-install** — `pip`/`uv`/`conda` installs via Bash instead of `add_dependency`
- **cell-file-edit** — editing a cell's source file directly (Write/Edit) instead
  of `edit_cell`/`add_cell`

Task completion and efficiency are reported alongside.

## How it's structured

Graders consume a driver-agnostic `Trajectory` (a list of tool calls), so the
same scoring runs against two backends:

| Driver | What it is | When |
| --- | --- | --- |
| `claude_code` | Real Claude Code headless (`claude -p`) against the live `/mcp` server | Local, on-demand |
| `replay` | Recorded transcripts, no server/LLM | CI (deterministic) |

Per task the runner drives the **real on-ramp** — it reuses the same
`agent_launch` helpers `strata agent` uses (create-or-open → spawn server with
MCP → open session → write `.mcp.json` + `CLAUDE.md`) — then hands the prepared
notebook to the driver and scores the run.

## Run it locally (real Claude Code)

Requires `claude` on `PATH`, the `[mcp]` + `[tui]` extras, and Claude Code
authenticated. Permissions are bypassed **on purpose** so the agent is free to
reach for Bash — whether it does is the whole measurement.

```bash
uv run python -m evals.agent_notebook.runner --driver claude_code \
    --out report.json                 # all tasks
uv run python -m evals.agent_notebook.runner --driver claude_code \
    --tasks build_summary,fix_bug     # a subset
```

Each task builds a throwaway notebook, runs the agent against it, then
`strata run`s the result for the completion check.

## Run it in CI (deterministic replay)

No server, no venv, no LLM — scores the committed transcripts and asserts the
harness + graders behave. This is what `tests/evals/test_agent_notebook.py`
covers, and you can run the suite directly:

```bash
uv run python -m evals.agent_notebook.runner --driver replay \
    --transcripts evals/agent_notebook/transcripts
```

**Note:** in replay mode the tool calls aren't executed against a live
notebook, so the **completion** column is only meaningful for seeded tasks — the
in-tool rate and efficiency are what replay validates. Completion is a
`claude_code`-path signal.

## Recording a real run for replay

Capture a real Claude Code session's `stream-json` and drop it in as
`transcripts/<task_id>.jsonl`; the replay driver parses it the same way the live
driver does. This lets you freeze a real trajectory and iterate on the graders
offline.

## Repeats

Agent runs are non-deterministic, so a single pass can be a lucky (or unlucky)
sweep. `--repeats N` runs each task N times; the report aggregates per task
(mean and **min** in-tool rate, escapes, pass count) so the headline is robust:

```bash
uv run python -m evals.agent_notebook.runner --driver claude_code --repeats 3
```

## Tasks

| id | what it exercises |
| --- | --- |
| `build_summary` | Build from scratch: create a DataFrame, compute an aggregate |
| `train_eval` | Add a dependency, fit a model, compute accuracy |
| `fix_bug` | Debug a seeded failing cell and make the notebook run clean |
| `extend_dag` | Add a downstream cell on an existing variable, run only what's needed |
| `compute_stats` | Two dependent cells over a range (pure Python) |
| `aggregate_group` | pandas groupby aggregate |
| `explore_inline` | Compute an answer from inline data (tempts a scratch script) |
| `add_missing_dep` | Fix a missing import — tempts a Bash `pip`/`uv` install |
| `refactor_split` | Split one cell into two — tempts editing the cell file directly |

The last three are deliberate **escape-tempters**: each has a tempting bypass the
working agreement asks the agent to resist, and the graders flag it if taken.

Add a scenario by appending a `Task` to `tasks.py` (name the expected variables
in the prompt so completion is checkable without an LLM judge). If you want the
replay/CI path to cover it, drop a matching transcript in `transcripts/`.

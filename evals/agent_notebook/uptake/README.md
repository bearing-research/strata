# Uptake task bank

The un-primed tasks that measure whether an agent reaches for the Strata scratchpad
on its own (design note `design-agent-uptake.md`, section 5).

- **`template/`:** `harbor`, a small uv workspace with an editable sibling package
  (`packages/fleet`), a seeded bug (`tests/test_text.py` fails), and a slow
  `harbor.data.load_events()` (20 s, `HARBOR_LOAD_SECONDS`). Each load appends a
  line to the file named by `HARBOR_LOAD_LOG`, so a run's loads are counted outside
  the agent's reach. A line is tagged with `STRATA_CELL_ID`, which a Strata cell's
  process has set, so a load done in a cell names the cell.
- **`workspace.py`:** `build(dest)` copies the template and writes 1M seeded order
  events to `data/events.parquet`. Their shape gives each question one answer.
- **`tasks.py`:** 29 tasks of five kinds: iteration (four turns over the same
  data), probing (calling project code with varied arguments), plot, neutral
  (either tool is fine) and must-not (edits and file reads, where using the
  scratchpad is a false positive). Each turn's answer is computed from the
  workspace: a number or word in the reply, or a check of the files left behind.
  `HOLDOUT` is the seeded third of each kind that tuning never sees.

```python
from evals.agent_notebook.uptake import tasks, workspace

work = workspace.build(Path("/tmp/run-1/harbor"))
task = tasks.TASKS_BY_ID["iter_west"]
for question, expect in zip(task.turns, task.expected(work)):
    reply = ...  # the agent's answer to question.prompt
    expect.passes(reply, work)
```

## Running trials

POSIX only (macOS, Linux). Each trial runs headless Claude Code (`claude -p`,
stream-json) in a fresh copy of the project (built, `uv sync`ed, committed) with its
own `CLAUDE_CONFIG_DIR`, so nothing from your own config reaches it. That dir has no
login, so `ANTHROPIC_API_KEY` must be set, and every run is billed to it. An iteration
task's follow-ups are sent one at a time with `--resume`. The agent's `PATH` leaves out
the eval's own environment, whose python can import strata.

Trials must live where no folder above them holds a `CLAUDE.md`, `.claude/` settings, a
`.git` or a `pyproject.toml`: Claude Code loads memory from every parent folder, so a
trial under this checkout (or your home, with its `~/.claude/CLAUDE.md`) would be told
about Strata. `run` refuses such an `--out`; use one under `/tmp`.

| Arm | Setup |
| --- | --- |
| A, control | every `strata*` command on `PATH` answers "command not found"; a `PreToolUse` hook denies any Bash command that runs one or imports `strata` |
| B, today | the shipped `strata-scratchpad` skill in the config dir; this checkout's `strata` on `PATH` |
| Cp, the design's C' | a Strata server of the trial's own (`STRATA_MCP_PROFILE=scratchpad`, every store under the trial folder) with one notebook open, whose environment has the project installed editable; a user-scope HTTP MCP entry for it, `alwaysLoad: true`, written with `claude mcp add-json` |

```bash
uv run python -m evals.agent_notebook.uptake run --arm A --out /tmp/uptake-runs/ --split train --runs 3
uv run python -m evals.agent_notebook.uptake run --arm B --out /tmp/uptake-runs/ --task iter_west
uv run python -m evals.agent_notebook.uptake report /tmp/uptake-runs/
```

`run` skips trials already in `--out`, so an interrupted batch picks up where it
stopped. Each trial folder keeps the project as the agent left it, each turn's stream,
the load counter and `trial.json`. A trial is **discarded**, never scored, when a turn's
`system/init` event does not show the arm's setup (in Cp, anything but `strata`
connected with `run` and `scratchpad` loaded; a strata MCP server in A or B; the skill
missing in B or present in A or Cp), when Cp's server does not come up, when a turn
never reached the model, when a resumed turn starts a new session, or when a turn runs
past its timeout (30 min).

`report` gives, per arm and kind: completion, strata use (for must-not tasks, the
false positive rate), payoff (`load_events()` ran once in the whole trial), median
wall time, cost, each rate with its 95% Wilson interval; and each arm against A, on
the tasks both ran: completion and payoff differences (Newcombe) and the ratio of
median wall times (bootstrap). It warns when more than 10% of an arm's runs were
discarded or arm A used strata, either of which means the numbers are not to be
trusted. The model, Claude Code, uv and Strata versions are recorded with each trial.

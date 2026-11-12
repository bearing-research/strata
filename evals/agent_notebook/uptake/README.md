# Uptake task bank

The un-primed tasks that measure whether an agent reaches for the Strata scratchpad
on its own (design note `design-agent-uptake.md`, section 5). Nothing here runs an
agent; the A/B runner that does comes next.

- **`template/`:** `harbor`, a small uv workspace with an editable sibling package
  (`packages/fleet`), a seeded bug (`tests/test_text.py` fails), and a slow
  `harbor.data.load_events()` (20 s, `HARBOR_LOAD_SECONDS`). Each load appends a
  line to the file named by `HARBOR_LOAD_LOG`, tagged with `STRATA_CELL_ID` when a
  cell ran it, so a run's loads are counted outside the agent's reach.
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

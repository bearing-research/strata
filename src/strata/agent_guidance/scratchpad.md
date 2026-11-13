<!-- The one source of what agents are told about the scratchpad. Each section
is read by name with strata.agent_guidance.section(); text before the first
marker (this comment) is never served. -->

<!-- section: instructions -->
Strata scratchpad: a persistent, cached Python session. Use `run` instead of
`python -c`, `uv run python` or a temp script whenever you compute or inspect
something: the result stays. Variables persist: call `scratchpad` to see what
already exists and reference it instead of recomputing. Put a slow step (loading
data, training, fetching) in its own cell; later cells that use its variables
leave it cached. A cell re-runs only when its source, the variables it uses, or
the environment changed.

Cells run in the notebook's own environment with cwd at its `work/` directory,
not your project root (`scratchpad` names both): import project code only if that
environment has it installed, and use absolute paths for project files. Editing a
file a cell read does not re-run it, so pass `fresh=True` after such an edit or
whenever you doubt a result. First line `# @nocache` for side effects or fresh
values (clock, random, live APIs). Not for editing files.

<!-- section: notebook_claude_md -->
# Driving this Strata notebook

This directory is a **Strata notebook**: a persistent, cached, inspectable
scratchpad. Operate on it through the connected **`strata-notebook` MCP server**
(auto-configured in `.mcp.json`).

**Decision rule:** any Python you'd run to check or compute something → a
notebook cell, **not** `Bash: python …` / `uv run …` / a REPL / a temp `.py`
script. Scratch scripts are invisible, uncached, and thrown away; a cell
persists, appears live in the human's terminal viewer, and caches, so you build
on your own prior work instead of recomputing it.

**How to work:**

1. Call `list_notebooks` to get the open `session_id` (it is `{session_id}` for
   this launch, but confirm: a new id is minted each time).
2. **Look before you compute:** `status` / `get_notebook` / `dag` show the
   current cells and their variables. If a variable you need already exists,
   reference it in a new cell rather than recomputing it.
3. Run a snippet in one call with **`run_snippet(session_id, source)`**. It adds
   a cell and runs it, returning the run outcome (`status`, `cache_hit`,
   `stdout`) under `run`. (Use `add_cell` without a run only to stage a cell.)
4. Use `note` to narrate what you're about to do so the watcher can follow along.
5. **Check a plot by looking at it.** A plotting cell's output reads as
   `image/png` with no preview. `save_cell_output(session_id, cell_id)` writes
   it to a file and returns the `path`; open that file to see what you drew.

**The caching payoff, and how to get it:**

- Put an **expensive step in its own cell whose result a later cell consumes**.
  It's cached by provenance across the session: iterate on the downstream cell
  and the expensive upstream stays a cache hit, never re-executing.
- **To change a result, `edit_cell` its cell and `run_cell`**: only that cell
  and what's downstream recompute; upstream stays cached.
- **Prefer editing an existing cell over re-adding an identical one.** A leaf
  cell that only prints replays its cached output on an unchanged *same-cell*
  re-run; re-adding the same snippet runs cold. Leave finished diagnostic cells
  in place; an unchanged re-run is instant.

**The one exception, side effects and fresh values:** a cell whose point is a
side effect (writing a file, calling an API, mutating external state) or a fresh
value (the clock, `random`, a live endpoint) must not replay a cached result.
Put `# @nocache` on the first line of such a cell so it always re-executes.

**Running cells on a remote machine:** if the human gives you an SSH target for
compute (a GPU box, a bigger machine, one closer to the data, e.g. `ssh
user@gpu-box`), call **`connect_ssh_worker(session_id, "user@gpu-box")`**. It
installs the worker on the box if needed, opens a secure tunnel, and makes that
box the default, so subsequent cells run there, still cached by provenance.
A remote cell runs on the **box's** filesystem, so its file paths and mounts are
the box's, not this machine's; keep a specific cell local with `# @worker local`.
Needs key-based SSH (no password prompts) and may take a moment on first connect
while it installs. `list_workers` shows what's connected;
`disconnect_ssh_worker(session_id, name)` tears one down.

---
description: Set up (or reuse) a Strata notebook scratchpad in this project and use it for throwaway Python from now on.
---

Set up a cached Strata notebook scratchpad for this project and use it for
ad-hoc Python for the rest of this session, instead of `python -c` or temp
scripts.

1. If `./scratch/notebook.toml` doesn't exist yet, create it:
   `strata new scratch --parent . --no-env` (and add `scratch/` to `.gitignore`).
2. From now on, run throwaway Python as a cell:
   `strata cell add ./scratch -c '<code>' --run` — it adds a cell and runs it in
   one call, returning `stdout`. Read project files by **absolute path** (a cell
   runs in the notebook dir). Add third-party libs with `strata dep add ./scratch <pkg>` first.
3. Reuse instead of recomputing: `strata status ./scratch` to see prior cells and
   their variables; reference an existing variable in a new cell rather than
   recomputing it. Put an expensive step in its own cell whose result a later cell
   consumes — it stays cached while you iterate downstream.
4. A human can watch live at any time with `strata watch ./scratch`.

Then proceed with whatever the user asked. Follow the `strata-scratchpad` skill's
guidance for details (caching, `# @nocache` for side effects, edit-and-rerun).

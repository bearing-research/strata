# Library cells

A *library cell* is one that defines reusable Python: functions, classes,
constants, code that other cells will import and call. Strata Notebook
re-executes the producing cell's source as a synthetic Python module so
downstream cells get a fresh, deterministic copy of each definition.

Originally the producing cell had to be **pure**: imports, defs, classes,
and literal-constant assignments only. A single setup line like
`df = load_data()` would block every helper in the same cell from being
shared.

The planner now **slices** the cell instead: it keeps imports, defs,
classes, and literal constants, drops everything else, and validates that
the slice is self-contained. Cells that mix runtime work with library code
can now share the library code cleanly. This notebook walks through what
works, what's relaxed, and what's still blocked.

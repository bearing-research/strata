# library_cells

Walks through cross-cell library code in Strata Notebook: pure module
cells, mixed runtime+library cells, PEP 563 type annotations, and the
limitations slicing keeps in place.

## What it covers

| Cell | Demonstrates |
| ---- | ------------ |
| `pure_lib` / `use_pure` | Classic pure module cell, imports + defs + literal const, no runtime work. Source survives verbatim. |
| `mixed_lib` / `use_mixed` | The slicing payoff: a single cell mixes runtime setup (`raw_min`, `raw_max`, a `print`) with reusable helpers (`clamp`, `CLAMP_MIN`). Runtime values still flow through the artifact path; helpers ride the synthetic module. |
| `typed_lib` / `use_typed` | `from __future__ import annotations` lets a helper reference a type defined elsewhere, `symtable` correctly drops stringified annotations from the free-variable check. |
| `blocked` / `try_blocked` | Closure over a runtime value blocks export. The diagnostic pinpoints the function (`is_outlier`) and the unresolved name (`runtime_threshold`). |

## Running

```bash
STRATA_DEPLOYMENT_MODE=personal \
STRATA_NOTEBOOK_STORAGE_DIR=/path/to/strata/examples \
uv run python -m strata
```

Then open the notebook UI and pick **Library Cells** from the discover
list. Run the cells top-to-bottom; the `blocked` cell is meant to fail
so you can see the new diagnostic.

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

## Pure module cell (baseline)

<sub>**kind** python</sub>

```python
# @name Pure module cell (baseline)
#
# This is the classic shape: imports, a literal constant, and a couple
# of helpers. No runtime work at module scope. The cell is exported
# verbatim (no slicing), so comments and formatting survive.

import math

CIRCLE_PRECISION = 4


def area(radius: float) -> float:
    return round(math.pi * radius * radius, CIRCLE_PRECISION)


def perimeter(radius: float) -> float:
    return round(2 * math.pi * radius, CIRCLE_PRECISION)
```

## Use the pure-cell helpers

<sub>**kind** python</sub>

```python
# @name Use the pure-cell helpers
#
# ``area`` and ``perimeter`` ride the synthetic-module path; running
# this cell re-executes the slice in a fresh module and pulls out
# the requested symbols.

circle = {
    "radius": 7.5,
    "area": area(7.5),
    "perimeter": perimeter(7.5),
    "precision_used": CIRCLE_PRECISION,
}
```

# Mixed cells (the new payoff)

The next cell mixes runtime setup with reusable helpers. Before slicing
this whole cell would have been blocked because of the runtime
assignments, the user would have to split them into separate cells just
to share `clamp` downstream.

After slicing, the planner drops the runtime lines from the synthetic
module and exports the helper. The runtime variables still flow through
the regular artifact path, so downstream cells see them too.

## Mixed: setup + helpers in one cell

<sub>**kind** python</sub>

```python
# @name Mixed: setup + helpers in one cell
#
# Slicing keeps:
#   - the import (math)
#   - the literal constant (CLAMP_MIN, CLAMP_MAX)
#   - the def (clamp)
#
# and drops the runtime statements (raw_min, raw_max, the print). The
# runtime variables are still visible to downstream cells via the
# regular artifact path; the def + constants ride the synthetic module.

import math

# --- Runtime setup, dropped from the synthetic module ----------------
# (Pre-slicing, these three lines would have blocked the whole cell.)
raw_min = round(-math.tau * 7, 2)  # function call → not a literal
raw_max = round(math.tau * 16, 2)
print(f"loaded raw bounds: [{raw_min}, {raw_max}]")

# --- Library code, kept in the slice ---------------------------------
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0


def clamp(value: float) -> float:
    """Pin *value* into ``[CLAMP_MIN, CLAMP_MAX]``."""
    return max(CLAMP_MIN, min(CLAMP_MAX, value))
```

## Use the mixed-cell helpers

<sub>**kind** python</sub>

```python
# @name Use the mixed-cell helpers
#
# This cell consumes both the library code (``clamp``, ``CLAMP_MIN``)
# *and* a runtime variable (``raw_max``) from the producing cell;
# the slicer routes each through the right path automatically.

clamped_examples = {
    "raw_max_clamped": clamp(raw_max),  # raw_max is runtime → artifact path
    "clamp_at_floor": clamp(-99),
    "clamp_at_ceil": clamp(999),
    "clamp_min_constant": CLAMP_MIN,  # CLAMP_MIN is literal → module path
}
```

# Typed library cells (PEP 563)

Type annotations participate in the free-variable check by default , 
`def f(x: SomeType): ...` would block export when `SomeType` isn't
bound in the slice.

Adding `from __future__ import annotations` to the cell relaxes this:
PEP 563 stringifies annotations and `symtable` correctly drops them from
the reference set. That makes it easy to write helpers that hint at types
defined elsewhere (e.g. `pyarrow.Table`) without forcing every consumer
to import the same names.

## Typed helper using PEP 563 annotations

<sub>**kind** python</sub>

```python
# @name Typed helper using PEP 563 annotations
#
# ``Table`` is *not* imported in this cell, but the future import keeps
# the annotation from being evaluated at module load. The function
# still works at call time because we only access duck-typed
# attributes (``num_rows``, ``num_columns``).

from __future__ import annotations


def describe_table(table: Table) -> dict:  # noqa: F821 - PEP 563 stringified
    """Summarize an Arrow-like table without needing the type at module scope."""
    return {"rows": table.num_rows, "columns": table.num_columns}
```

## Use the typed helper

<sub>**kind** python</sub>

```python
# @name Use the typed helper
#
# The consumer imports ``pyarrow`` itself (it actually needs the
# concrete class, since it's calling the constructor). The library
# cell got away with just the *name* in an annotation.

import pyarrow as pa

table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
summary = describe_table(table)
```

# Limitations

Slicing has a hard rule: the synthetic module is built from **one cell's
source only**: it can't reach across cells to find names. Three failure
modes follow from that:

1. **Closures over runtime values.** A helper that references a value
   computed at runtime in the same cell can't be exported, because
   that value won't exist when the synthetic module re-executes.
2. **Cross-cell imports.** `import math` in cell A doesn't make `math`
   visible inside a helper exported by cell B, each cell that hosts
   library code has to carry its own imports.
3. **Cross-cell helpers.** A helper can't call another helper that
   lives in a different cell. Move them into the same cell, or
   duplicate the dependency.

The next cell triggers the first failure mode on purpose so you can see
the diagnostic, it should error with a message naming the unresolved
variable.

## Closure over a runtime value (intentionally blocked)

<sub>**kind** python</sub>

```python
# @name Closure over a runtime value (intentionally blocked)
#
# ``runtime_threshold`` is computed at runtime, so it gets dropped from
# the slice. ``is_outlier`` references it as a free variable, which
# leaves the synthetic module unable to resolve the name at call time.
#
# Running this cell errors at execution time. The diagnostic message
# names both the function (``is_outlier``) and the unresolved variable
# (``runtime_threshold``), pointing the user straight at the fix.
#
# To unblock: move ``runtime_threshold`` into its own cell so the slice
# only contains ``is_outlier``, and let the threshold flow through the
# regular artifact path.

import math

runtime_threshold = math.sqrt(9)


def is_outlier(value: float) -> bool:
    return value > runtime_threshold
```

## Downstream consumer (would call is_outlier)

<sub>**kind** python</sub>

```python
# @name Downstream consumer (would call is_outlier)
#
# Without this reference, the producing cell would just be a private
# helper and the planner would let it slide. Naming ``is_outlier``
# here is what tells Strata "this is meant to be shared," and is
# what makes the export-blocked diagnostic fire on the cell above.

would_be_outliers = is_outlier(7)
```

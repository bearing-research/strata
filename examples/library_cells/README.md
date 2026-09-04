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

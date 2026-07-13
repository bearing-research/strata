# Data Viewer — paging and sorting a large DataFrame

A two-cell notebook that produces DataFrames larger than the inline
preview, so the notebook's **interactive data viewer** switches on.

## What it shows

- A cell whose displayed output is a **2,000-row** DataFrame. Instead of
  a static 20-row table, the output renders as a scrollable grid with a
  footer showing `1–50 of 2,000 rows` and **Prev / Next** paging.
- **Click a column header to sort** — ascending, then descending, then
  clear. Sorting runs server-side over the whole frame (not just the
  visible page), reading the cached Arrow artifact directly.
- A **page-size** selector (25 / 50 / 100 / 250).
- Columns of every dtype: integer (`id`), datetime (`ts`), string
  (`region`, `product`), and float (`unit_price`, `revenue`).

The viewer only appears for table-shaped outputs (pandas / polars
DataFrames + Series, pyarrow Tables, and SQL / R tabular results).
Tensors and scalars keep their existing display.

## Cells

| Cell | What it does |
|---|---|
| `generate` | Builds a 2,000-row transactions frame and displays it. |
| `ranked` | Sorts by revenue into a derived frame — also gets the grid. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Then open `examples/data_viewer` from the Strata home page and run both
cells.

## Try this

1. Run `generate`. Scroll the grid; click **Next** to page through all
   2,000 rows.
2. Click the `revenue` header twice to sort descending — the top row is
   now the largest revenue in the *entire* frame, not just this page.
3. Change the page size to 250 and page again.
4. Run `ranked` and sort it by `ts` — the server re-sorts the derived
   frame on demand.

# Widget Playground — interactive controls driving the DAG

A two-cell notebook: a **widget cell** (a control panel) feeding a Python cell
that renders a DataFrame. Drag a control, and the downstream cell goes stale —
run it and the grid updates.

## What it shows

- A **widget cell** with three controls:
  - `alpha` — a slider (0–1)
  - `n` — a number input (row count)
  - `curve` — a dropdown (`linear` / `sqrt` / `square`)
- A downstream Python cell that reads `alpha`, `n`, `curve` and builds a
  DataFrame — which renders in the **interactive data viewer**. So the two
  features compose: a widget drives the grid.
- **Content-addressed values**: drag `alpha` to a new value and back — the
  second time is a cache hit, no recompute.

## Cells

| Cell | What it does |
|---|---|
| `controls` | A `widget` cell declaring the slider / number / dropdown. |
| `preview` | Reads the control values, builds a DataFrame, displays it. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Open `examples/widget_playground` from the Strata home page.

## Try this

1. Run the `preview` cell — it materializes the widget defaults and shows a
   20-row table in the data viewer.
2. Drag the `alpha` slider. `preview` turns yellow (stale).
3. Run `preview` — the `y` column reflects the new `alpha`.
4. Switch `curve` to `square` and re-run — the shape changes.
5. Drag `alpha` back to its previous value and run — a cache hit.

Widget cells render in the web UI. Edit the control declaration with the
**✎ Edit controls** toggle on the cell.

## Open as an app

Click **App** in the notebook header to open this notebook as a read-only
interactive app — just the controls + the table, no editor. Turn on the
widget cell's **⚡ Live** toggle first, and dragging `alpha` in the app updates
the table live.

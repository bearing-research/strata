# Widget Playground: interactive controls driving the DAG

A three-cell notebook: a **widget cell** (a control panel) feeding two Python
cells, one that renders a DataFrame and one that draws a chart. The widget
cell is `# @live`, so dragging a control re-runs both downstream cells and
the grid and the chart update.

## What it shows

- A **widget cell** with three controls:
  - `alpha`: a slider (0–1)
  - `n`: a number input (row count)
  - `curve`: a dropdown (`linear` / `sqrt` / `square`)
- A downstream Python cell that reads `alpha`, `n`, `curve` and builds a
  DataFrame, which renders in the **interactive data viewer**. So the two
  features compose: a widget drives the grid.
- A second downstream cell that plots the same curve with matplotlib.
- **Live mode**: `# @live` on the widget cell makes a control change re-run
  the cheap downstream cells instead of only marking them stale.
- **Content-addressed values**: drag `alpha` to a new value and back, the
  second time is a cache hit, no recompute.

## Cells

| Cell | What it does |
|---|---|
| `controls` | A `# @live` `widget` cell declaring the slider / number / dropdown. |
| `preview` | Reads the control values, builds a DataFrame, displays it. |
| `plot` | Reads the same values and plots `y` against `x`. |

## Running

From the project root:

```bash
uv run strata-notebook
```

Open `examples/widget_playground` from the Strata home page.

## Try this

1. Run `preview` and `plot`. They read the widget defaults: a 20-row table
   in the data viewer and a straight line.
2. Drag the `alpha` slider. Both cells re-run on their own, and the `y`
   column and the chart reflect the new `alpha`.
3. Switch `curve` to `square`, the shape changes.
4. Drag `alpha` back to its previous value, a cache hit.
5. Turn **⚡ Live** off (or change the annotation to `# @live off`) and drag
   again: now the two cells only turn yellow (stale) until you run them.

Widget cells render in the web UI. Edit the control declaration with the
**✎ Edit controls** toggle on the cell.

## Open as an app

Click **App** in the notebook header to open this notebook as a read-only
interactive app, just the controls, the table and the chart, no editor. With
the widget cell's **⚡ Live** on, as it ships, dragging `alpha` in the app
updates them live.

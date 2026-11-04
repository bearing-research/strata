# Runtime states

How each state a cell can be in looks, everywhere it is shown. The state comes from one place in
each client: `src/utils/cellState.ts` for the web UI, `cell_state` in
`src/strata/notebook/tui/viewmodel.py` for the terminal. Tests hold this table to both.

**A state is never shown by color alone.** Each one has its own icon shape and its own word. Color
only repeats what those say, so the states stay apart for a color-blind reader, on a monochrome
terminal, and in a screenshot pasted into a grey chat window.

| State       | Means                                      | Web token pair        | Web icon          | Label       | Terminal | Terminal color |
| ----------- | ------------------------------------------ | --------------------- | ----------------- | ----------- | -------- | -------------- |
| idle        | No result, or one that can't be judged yet | `--state-idle`        | `Circle`          | idle        | `○`      | `bright_black` |
| queued      | Waiting its turn in this run               | `--state-queued`      | `Clock`           | queued      | `…`      | `bright_black` |
| running     | Computing on this machine                  | `--state-running`     | `CircleDashed`    | running     | `▶`      | `blue`         |
| dispatching | A worker has the run                       | `--state-dispatching` | `Send`            | dispatching | `↗`      | `cyan`         |
| ready       | Current, computed by its last run          | `--state-ready`       | `Circle` (filled) | ready       | `✓`      | `green`        |
| cached      | Current, and its last run was a hit        | `--state-cached`      | `Zap`             | cached      | `◆`      | `bright_cyan`  |
| stale       | Shows a result known not to match          | `--state-stale`       | `Contrast`        | stale       | `⊘`      | `yellow`       |
| error       | Its last run failed                        | `--state-error`       | `X`               | error       | `✗`      | `red`          |
| blocked     | An annotation error keeps it from running  | `--state-blocked`     | `Ban`             | blocked     | `■`      | `bright_red`   |

`ready` and `queued` are real states the cell list reaches. `ready` is not `cached`: a cell you
just ran is current without being a hit.

A cell that is both takes the first of: a run in flight (`running`, `dispatching`, `queued`), then
`blocked`, `error`, `stale`, `cached`/`ready`, `idle`. An annotation error outranks a stale or failed
result because it is what stops the next run. A run in flight outranks it because that run is on
the source from before the error.

## Tokens

Each state has a pair: `--state-<name>` for the icon, border and DAG node stroke, and
`--state-<name>-tint` for the badge and DAG node fill. They live in `src/style.css` and alias the
existing `--accent-*` and `--tint-*` tokens, which alias `--cat-*` per theme:

| State        | Alias                                | Dark      | Light     |
| ------------ | ------------------------------------ | --------- | --------- |
| idle, queued | `--text-muted`, `--tint-muted`       | `#6c7086` | `#8c8fa1` |
| running      | `--accent-primary`, `--tint-primary` | `#89b4fa` | `#1e66f5` |
| dispatching  | `--accent-info`, `--tint-info`       | `#89dceb` | `#04a5e5` |
| ready        | `--accent-success`, `--tint-success` | `#a6e3a1` | `#40a02b` |
| cached       | `--accent-teal`, `--tint-teal`       | `#94e2d5` | `#179299` |
| stale        | `--accent-warning`, `--tint-warning` | `#f9e2af` | `#df8e1d` |
| error        | `--accent-danger`, `--tint-danger`   | `#f38ba8` | `#d20f39` |
| blocked      | `--accent-peach`, `--tint-peach`     | `#fab387` | `#fe640b` |

A label is never written in its state color. Several Latte accents fall under 3:1 on white
(`--accent-warning` is 2.6:1), so a badge sets its word in `--text-primary` on the tint and gives
the state color to the icon. The `.state-badge` class does this; a `.state-<name>` class on any
ancestor sets which state it shows.

## Where each part shows

| Surface            | Icon             | Color                   | Label                                                                                                                          |
| ------------------ | ---------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Cell header        | The gutter dot   | Dot, border, badge tint | A badge, except `idle` and `ready`, whose label is on the dot's tooltip and accessible name, with the run's duration beside it |
| DAG node           | Left of the name | Stroke, fill, icon      | The node's tooltip and accessible name; there is no room in the box                                                            |
| Terminal cell list | The first column | The glyph               | The second column, except `idle` and `ready`, which show the run's duration                                                    |
| Terminal DAG       | Left of the name | None                    | None: the glyph is the state                                                                                                   |

Any new surface that shows a state uses this table rather than picking from the palette.

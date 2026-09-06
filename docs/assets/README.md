# docs/assets

Images the docs site and the top-level `README.md` embed.

Everything here except `agent-demo.gif` is **generated**, not hand-made -
including `tui-cache-payoff.gif`, which is assembled from canned TUI frames
rather than recorded. A screenshot of last release's layout still looks
authoritative, so the way these stay honest is that regenerating all of them is
one command rather than an afternoon of cropping:

```bash
uv run python scripts/capture_docs_shots.py           # everything
uv run python scripts/capture_docs_shots.py --only tui # just the SVGs (fast, no server)
```

Re-run it whenever the notebook UI or the TUI changes shape, and commit the
result alongside that change.

## What is here

| File | Surface | Shown on |
| ---- | ------- | -------- |
| `notebook-anatomy-{light,dark}.png` | Web UI | Notebook quickstart, `README.md` |
| `cascade-stale-{light,dark}.png` | Web UI | Notebook quickstart, step 5 |
| `cell-tests-{light,dark}.png` | Web UI | Cell unit tests, the Tests panel |
| `registry-promote-strip-{light,dark}.png` | Web UI | Registry dashboard, step 3 |
| `registry-tab-{light,dark}.png` | Web UI | Registry dashboard, step 5 |
| `registry-lineage-{light,dark}.png` | Web UI | Registry dashboard, step 7 |
| `tui-layout.svg` | TUI | Terminal viewer |
| `tui-agent-running.svg` | TUI | Driving a notebook with a coding agent |
| `tui-cache-payoff.gif` | TUI | Terminal viewer, watching a re-run |
| `agent-demo.gif` | Screen recording | Not yet recorded - see below |

## Conventions

- **Web shots are PNG and come in both themes.** Reference them with Material's
  `#only-light` / `#only-dark` suffix so a page never shows a dark screenshot on
  a light background. `README.md` uses a `<picture>` element for the same reason.
- **TUI shots are SVG**, exported by Textual. They stay crisp at any zoom, are
  roughly a tenth the size of the equivalent PNG, keep the text selectable, and
  read correctly on either docs theme, so they need only one variant.
- **Alt text is required, and no image carries information the prose omits.** A
  screenshot illustrates; it never explains. Every page must read correctly with
  images blocked.
- `mkdocs build --strict` fails on an image reference with no file behind it, so
  a broken path in `docs/` is caught on every docs PR. `README.md` is not built
  by mkdocs - check that one by eye.

## Animations

`tui-cache-payoff.gif` is generated like the stills, in three stages, because
nothing in the Python environment renders SVG:

```bash
uv run python scripts/capture_docs_shots.py --only anim
```

Textual writes each storyboard beat as an SVG, `frontend/scripts/rasterize-svg.mjs`
turns those into PNG through Playwright (already a frontend dependency, and it
renders SVG the way a reader's browser will), and `scripts/assemble_gif.py`
packs them with per-beat hold times.

Frame timings live in `assemble_gif.py`, not in the storyboard: how long a
reader needs on a frame is a property of the finished animation. Adding or
removing a beat without retiming it is a hard error rather than a silently
mistimed GIF.

This is **not** the agent demo below. It shows the TUI - which is the right-hand
pane of that recording - and carries the cache payoff, but no agent appears in
it. The two are different assets.

## `agent-demo.gif` (to be recorded)

A clip for the README's "cached scratchpad" section: a coding agent builds a
Strata notebook live and reuses the cached model when only the evaluation
changes. Unlike the shots above it cannot be scripted from here - it is a screen
recording.

- **How to record it:** follow [`examples/agent_demo/RECORDING.md`](https://github.com/bearing-research/strata/blob/main/examples/agent_demo/RECORDING.md)
  (two side-by-side panes, `strata agent`, two prompts, ~15-25s, < 5 MB).
- **Where it goes:** save the exported GIF here as `docs/assets/agent-demo.gif`.
- **Wire it in:** uncomment the image line in the README's "Give your coding
  agent a cached scratchpad" section (it's a commented slot so the README never
  renders a broken image before the GIF exists).

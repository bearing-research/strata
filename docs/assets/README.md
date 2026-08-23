# docs/assets

Binary image/media assets referenced by the README and the docs site.

## `agent-demo.gif` (to be recorded)

The hero clip in the top-level `README.md`: a coding agent builds a Strata
notebook live and reuses the cached model when only the evaluation changes.

- **How to record it:** follow [`examples/agent_demo/RECORDING.md`](https://github.com/bearing-research/strata/blob/main/examples/agent_demo/RECORDING.md)
  (two side-by-side panes, `strata agent`, two prompts, ~15–25s, < 5 MB).
- **Where it goes:** save the exported GIF here as `docs/assets/agent-demo.gif`.
- **Wire it in:** uncomment the image line in the README's "Give your coding
  agent a cached scratchpad" section (it's a commented slot so the README never
  renders a broken image before the GIF exists).

Keep assets small (GitHub inlines images up to 10 MB, but smaller loads faster)
and give each a descriptive filename.

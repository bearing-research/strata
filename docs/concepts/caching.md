# How caching works

![A browser on the tutorial notebook over five moments: nothing has run; Run All computes every cell; Run All again and every code cell is a cache hit showing the time it saved; one cell is edited, so it and the chart below read stale while the load cell stays cached; Run All again, and only those two run while the load cell still reads cached, saved 2.1s.](../assets/cache-payoff-light.gif#only-light)
![A browser on the tutorial notebook over five moments: nothing has run; Run All computes every cell; Run All again and every code cell is a cache hit showing the time it saved; one cell is edited, so it and the chart below read stale while the load cell stays cached; Run All again, and only those two run while the load cell still reads cached, saved 2.1s.](../assets/cache-payoff-dark.gif#only-dark)

*Edit one cell in `strata tutorial`: only the cells that read it run again, and the
cell above keeps what it saved. Captured from a real run, not a mock-up.*

## Caching and provenance

The provenance hash determines cache identity. It includes:

| Component | In hash? | Why |
|-----------|----------|-----|
| Source code | Yes | Different code = different result |
| Upstream artifact hashes | Yes | Different inputs = different result |
| Environment lockfile hash | Yes | Different packages = different result |
| Env vars the cell declares (`@env`) or reads by name | Yes | A different setting can change the result |
| Mount fingerprints (URI and contents) | Yes | Different mounted data = different result |
| `@table` snapshot, `@fetch` digest, `@dataset` version | Yes | New data at the source = different result |
| Cell ID | No | Same code in a different cell = same result |
| Notebook ID or project name | No | Same cell in another notebook with the same packages = same result |
| Execution time | No | Same inputs should produce same output |

When you change a cell's source, its provenance hash changes, and all downstream cells become **stale**.

Because the hash describes the computation rather than the machine, a match
found somewhere else is just as valid. That is what the optional
[team cache](../guides/share/team-store.md#the-team-cache-sharing-results-nobody-named) trades on: with a shared store configured, a local
miss asks whether a colleague has already run this exact computation, and a
cell that does run offers its outputs back. Nobody has to decide in advance
which results are worth publishing - the expensive middle of a pipeline is
shared on the same terms as anything else.

A cell's console output is keyed by the same provenance hash and replayed with
its outputs, so a hit shows what the run behind it printed. That holds after you
undo an edit you ran: the cell reads stale, because what it shows is the edit's
run, and running it brings back the first run whole without executing anything.

This applies even to a **leaf cell**, one with no downstream consumer, like a
cell that only `print`s a diagnostic. It has no output artifact to cache, so its
console is the record that it ran, and an unchanged re-run doesn't re-execute.
That is what makes the notebook a usable scratchpad. A cell that must always run, because it has a side effect
(writing a file, calling an API) or produces a fresh value (the clock,
`random`), opts out of caching with [`# @nocache`](../reference/annotations/python.md#nocache).

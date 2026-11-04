# Notebook: Markdown Showcase

# Markdown Showcase

This notebook exercises every markdown rendering path in Strata. It's
deliberately content-only, there's no real "computation" to run. Use it
to verify that the renderer handles the cases listed below, and as a
reference when you want to write a documentation cell.

**Two surfaces are tested:**

1. **Markdown cells** (this cell, and the next several). Source is raw
   markdown; preview renders by default; click anywhere on the preview
   to swap in the editor; blur to render again.
2. **Dynamic markdown output** from Python cells via `Markdown(...)`
   (the last two cells). Same rendering pipeline, different entry
   point, useful when you want a code cell to emit a formatted report.

If anything in the next few cells doesn't render the way you'd expect,
that's a renderer bug worth filing.

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

Headings should shrink with each level and stay distinct from body text.
There should be no `#` characters left in the rendered output, and the
spacing between adjacent headings should be sane.

## Inline emphasis

This is **bold** and so is __this__. This is *italic* and so is _this_.
This is ***both*** at once. This is ~~struck through~~ via GFM.

Code spans use backticks: `df.groupby("ticker").agg("sum")`. Inline
code should keep `_underscores_` and `*asterisks*` literal, they're
data, not formatting.

Edge cases the old renderer was bad at:

- Mid-word italic shouldn't fire: `snake_case_name` stays as one token.
- Math-style underscores: `x_1, x_2, x_3` should also stay literal.
- Asterisks inside code spans: `a * b * c` keeps the stars.
- A pair of backticks containing a backtick: `` `like_this` ``.

## Unordered

- alpha
- beta
- gamma

## Ordered

1. first
2. second
3. third

## Nested

- top level
  - nested level
    - deeply nested
  - back to nested
- another top level
  1. nested ordered one
  2. nested ordered two

## Mixed (ordered + unordered, with content)

1. **Step one**: set up the environment.
   - Install dependencies via `uv sync`.
   - Verify with `uv run pytest tests/notebook/`.
2. **Step two**: run the notebook.
   - Open it via the Strata UI.
   - Click `Run All` to execute every Python cell in order.
3. **Step three**: read the output.

## Plain table

| Provider | Model | Pricing tier |
|----------|-------|--------------|
| OpenAI | gpt-5.4 | Premium |
| Anthropic | claude-sonnet-4-6 | Standard |
| Google | gemini-3-flash | Cheap |

## With column alignment

| Eval | Score | Notes |
|:-----|------:|:------:|
| MMLU-redux | 74.2% | macro |
| GSM8K | 88.1% | exact-match |
| HumanEval | 71.0% | pass@1 |
| TruthfulQA | 62.3% | mc1 |

## Table with inline formatting

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `model_id` | `str` | ✓ | Identifier from the **registry** |
| `score` | `float` | ✓ | Numeric in `[0, 1]` |
| `provenance_hash` | `str` | ✓ | SHA-256 of the methodology chain |
| `notes` | `str` | | Optional *human-readable* commentary |

## Fenced code with a language

```python
def materialize(inputs, transform):
    """The single primitive Strata exposes."""
    provenance = sha256(serialize(inputs) + serialize(transform))
    if cache.has(provenance):
        return cache.get(provenance)
    result = transform.apply(inputs)
    cache.put(provenance, result)
    return result
```

## Fenced code without a language

```
$ STRATA_PORT=8765 strata-notebook
Listening on http://127.0.0.1:8765
```

## Inline + fenced together

The runtime injects `Markdown` into every cell namespace:

```python
display(Markdown("# Live!"))
```

So you can produce documentation from data without importing anything.

## Edge cases

A fence containing markdown special chars:

```markdown
# This stays literal

- *no italics here*
- `no code spans`
- [no links](http://example.com)
```

Fenced HTML, should be HTML-escaped, not rendered:

```html
<script>alert("hi")</script>
```

## Single-level

> Long-horizon workflows have these properties: expensive, iterative,
> branching, failure-prone. What breaks first is not compute, it's
> state. Strata makes state explicit and durable.

## Nested

> Outer layer.
>
> > Quoted within the quote, the inner block should render as a
> > nested blockquote, not lose its prefix.
>
> Back to the outer layer.

## With other markdown inside

> ### Methodology
>
> Every benchmark score this harness produces is paired with a
> **provenance hash** covering:
>
> 1. Dataset version
> 2. Prompt template
> 3. Decoding configuration
> 4. Model identifier
> 5. Harness code
>
> Same inputs + same harness = same hash.

## Horizontal rule

Above the rule.

---

Below the rule.

***

Triple-asterisk rule above this line.

## Inline links

Standard markdown link: [Strata homepage](https://bearingresearch.com/strata/).

Link with a query string:
[search](https://example.com/search?q=strata&page=2).

Link to a fragment: [strata install](https://bearingresearch.com/strata/latest/getting-started/installation/#prerequisites).

## Autolinks (CommonMark)

Wrapped in angle brackets: <https://example.com>.

Email autolink: <hello@example.com>.

## Plain URLs (linkify)

A bare URL inside a sentence, like https://example.com/page, should
auto-detect into a clickable link without explicit markdown syntax.
The old hand-rolled parser missed this case.

## Reference links

Reference-style: [Strata][nb] uses an artifact store at its core.

[nb]: https://bearingresearch.com/strata/

## Link safety

All rendered links should open in a new tab and carry
`rel="noreferrer noopener"`. Right-click → Inspect on any link above
to verify.

## Sanitization checks

Markdown cells (and `Markdown(...)` outputs) are user-controlled and
rendered via `v-html`, so the renderer runs every output through
DOMPurify. Each item below should appear as **literal text or a no-op**
,  never executed, never producing a popup, never with the dangerous
attribute reaching the DOM.

### Script injection

Below should render as plain text inside a paragraph, not run:

&lt;script>alert('XSS via raw script tag')&lt;/script>

### Inline HTML attempts

The renderer is configured with `html: false`, so any inline HTML
we accidentally let through still gets sanitized. The next line should
remain as a paragraph with the angle-bracket text visible:

&lt;img src="x" onerror="alert('attribute XSS')" />

### `javascript:` URLs

A markdown link with a `javascript:` URL should either render as
inert text or be stripped entirely, never as a clickable executor:

[click here please](#)

### `on*` handler attempts inside markdown

Markdown doesn't have a syntax for HTML attributes, but if a future
extension allowed `{onclick=...}` style attribute lists, it must not
plumb through.

### Data URLs

Data URLs in links are sanitized too, DOMPurify drops the href:

[image data url](#)

## Dynamic Markdown via display()

<sub>**kind** python</sub>

```python
# @name Dynamic Markdown via display()
#
# The Markdown(...) wrapper is auto-injected into every cell namespace,
# so a Python cell can emit formatted prose without importing anything.
# This is the path that lets a code cell produce a "report" alongside
# numeric output.

import datetime as _dt

now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

display(
    Markdown(
        f"""
## Generated at runtime

This block was rendered from a **Python cell**, not a markdown cell.

- Current time: `{now}`
- Cell language: `python`
- Display path: `Markdown(text) → text/markdown artifact → renderer`

> The same renderer powers both surfaces, so anything the markdown
> showcase renders correctly will also work here.
"""
    )
)
```

## Templated report from a DataFrame

<sub>**kind** python</sub>

```python
# @name Templated report from a DataFrame
#
# A more realistic dynamic-markdown pattern: take some tabular result
# and render a small report. Useful for "summary at the end of a
# notebook" cells where you want prose + data in one display, not
# stacked output blocks.

import pandas as pd

df = pd.DataFrame(
    {
        "model_id": [
            "claude-sonnet-4-6",
            "gpt-5.4",
            "gemini-3-flash",
            "mistral-large-latest",
        ],
        "mmlu": [0.8412, 0.8205, 0.7891, 0.7634],
        "gsm8k": [0.9421, 0.9387, 0.9012, 0.8723],
        "humaneval": [0.8521, 0.8234, 0.7823, 0.7123],
    }
)

# Score columns to summarize.
score_cols = [c for c in df.columns if c != "model_id"]
df["mean"] = df[score_cols].mean(axis=1)
ranked = df.sort_values("mean", ascending=False).reset_index(drop=True)

best = ranked.iloc[0]
worst = ranked.iloc[-1]
spread = (best["mean"] - worst["mean"]) * 100

# Pretty table: markdown table syntax with right-aligned scores.
table_lines = [
    "| Rank | Model | MMLU | GSM8K | HumanEval | Mean |",
    "|:----:|-------|-----:|------:|----------:|-----:|",
]
for i, row in ranked.iterrows():
    table_lines.append(
        f"| {i + 1} | `{row['model_id']}` "
        f"| {row['mmlu']:.1%} | {row['gsm8k']:.1%} "
        f"| {row['humaneval']:.1%} | {row['mean']:.1%} |"
    )
table = "\n".join(table_lines)

display(
    Markdown(
        f"""
## Mock leaderboard report

**{len(ranked)} models** scored across **{len(score_cols)} evals**. Top
performer: `{best['model_id']}` at `{best['mean']:.1%}` mean. Spread
between best and worst: **{spread:.1f} points**.

{table}

---

*Report generated dynamically from `pd.DataFrame` → `Markdown`,
exactly the pattern you'd use for "summary at end of notebook" cells.*
"""
    )
)
```

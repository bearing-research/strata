# Review Triage: Structured Prompt Output

Demonstrates `# @output_schema` on a prompt cell.

The `triage` cell sends a list of customer reviews to an LLM and gets
back a **schema-validated JSON array**: not free-form text that a
downstream cell has to regex. The schema is passed through as native
structured-output (Anthropic's native tool use, OpenAI's
`response_format: {type: "json_schema"}`, or `json_object` fallback for
providers that don't support schemas).

Because the schema is part of the cell's provenance hash, editing the
schema invalidates the cached response, exactly what you want when
you're iterating on the shape of the output.

## Cells

1. `reviews.py`: hand-picked list of customer reviews
2. `triage.prompt`: prompt cell with `@output_schema` enforcing
   `{sentiment, priority, tags}` per review
3. `summary.py`: pandas aggregation of the structured results

## Running

Set an API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) in the
notebook's Runtime panel, then run-all.

## reviews

<sub>**kind** python</sub>

```python
# A small hand-picked set of customer reviews. Kept inline so the
# notebook runs anywhere: no network, no data file, just enough
# variety to make the triage output interesting.
reviews = [
    "Shipping took two weeks but the product itself works perfectly. Would buy again.",
    "Absolute disaster. Arrived broken, support ignored three emails. Demanding a refund.",
    "It's fine. Does what it says on the box.",
    "Great product and fast shipping, but the packaging was excessive. So much plastic.",
    "Completely failed after 24 hours of use. Battery overheated. Fire hazard.",
    "Best purchase of the year! Setup was a breeze and it integrates with everything.",
    "Missing the power cable. Shop should have caught this before shipping.",
    "Works as advertised. The interface feels a bit dated but the performance is solid.",
]
```

## triage

<sub>**kind** prompt</sub>

*Prompt cell: response intentionally excluded from export.*

```text
# @name triage
# @temperature 0.0
# @output_schema {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "object", "properties": {"review_index": {"type": "integer"}, "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]}, "priority": {"type": "string", "enum": ["low", "medium", "high"]}, "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3}}, "required": ["review_index", "sentiment", "priority", "tags"]}}}, "required": ["items"]}
# @system You are a customer-support triage assistant. Keep tags short (1-2 words).
#
# Free-form language models return free-form text: useful for summaries,
# awkward for pipelines. The `@output_schema` annotation pins the shape
# of this cell's output so downstream cells can destructure fields
# without regex-wrangling the response. Schema changes invalidate the
# cache, so iterating on the schema does what you'd expect.

Triage the following customer reviews. For each one, return:
- `review_index`: the 0-based position of the review in the input list
- `sentiment`: positive / negative / neutral
- `priority`: low / medium / high. "high" means the team should
  escalate immediately (safety issues, demands for refunds, etc).
- `tags`: 1–3 short descriptive tags (e.g. "shipping", "hardware
  failure", "packaging")

Reviews:
{{ reviews }}
```

## triage_summary

<sub>**kind** python</sub>

```python
# @name triage_summary
# Downstream cells consume the LLM output as structured data: no
# parsing, no fallbacks, no "did the model actually return JSON this
# time?" defensive code. The schema guarantees the shape.
import pandas as pd

rows = triage["items"]
df = pd.DataFrame(rows)

print("Per-review triage:")
for row, review in zip(rows, reviews):
    tags = ", ".join(row["tags"])
    print(
        f"  [{row['review_index']}] "
        f"{row['sentiment']:>8} / {row['priority']:>6} "
        f"({tags}): {review[:60]}..."
    )

print()
print("By priority:")
print(df["priority"].value_counts().to_string())

print()
print("Sentiment distribution:")
print(df["sentiment"].value_counts().to_string())

# The final expression becomes the cell's display value: a compact
# record of what the triage flagged as high-priority.
high_priority = df[df["priority"] == "high"]
{
    "total_reviews": len(df),
    "needs_escalation": len(high_priority),
    "negative_rate": float((df["sentiment"] == "negative").mean()),
}
```

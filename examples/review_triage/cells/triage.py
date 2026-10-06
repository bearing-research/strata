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

# Review Triage: Structured Prompt Output

Demonstrates `# @output_schema` on a prompt cell.

The `triage` cell sends a list of customer reviews to an LLM and gets
back a **schema-validated JSON array**: not free-form text that a
downstream cell has to regex. The schema is passed through as native
structured-output (OpenAI's `response_format: {type: "json_schema"}`,
or `json_object` fallback for providers that don't support schemas).

Because the schema is part of the cell's provenance hash, editing the
schema invalidates the cached response, exactly what you want when
you're iterating on the shape of the output.

## Cells

1. `reviews.py`: hand-picked list of customer reviews
2. `triage.py`: prompt cell with `@output_schema` enforcing
   `{sentiment, priority, tags}` per review
3. `summary.py`: pandas aggregation of the structured results

## Running

Set an API key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) in the
notebook's Runtime panel, then run-all.

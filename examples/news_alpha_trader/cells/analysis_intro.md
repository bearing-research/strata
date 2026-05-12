# Turn news into signals

We've pulled raw headlines from the news API. The cell below is a
**prompt cell** — it sends each headline to an LLM with an
`@output_schema` that forces a structured JSON response (ticker,
direction, confidence). The schema-validated result becomes a
content-addressed artifact like any Python cell's output, so
identical headlines + identical prompt template = cache hit, no
extra API call.

After signal extraction, the next few cells persist them, fetch
matching price data, and run risk checks before any orders go out.

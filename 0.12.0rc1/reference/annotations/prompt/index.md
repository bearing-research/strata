# Prompt annotations

Prompt cells (language `prompt`) accept an additional set of annotations that
configure the AI call.

## `@model`

Override the notebook-level AI model for this cell only.

```
# @model claude-sonnet-4-6
Summarize {{ df }} in one paragraph.
```

## `@temperature`

Sampling temperature. Defaults to `0.0`.

```
# @temperature 0.3
```

## `@max_tokens`

Ceiling on output tokens for this call.

```
# @max_tokens 1024
```

## `@system`

System prompt prepended to the conversation.

```
# @system You are a terse data analyst. Answer in bullet points.
```

Only one `@system` line counts: when there are several, the last one wins.

## `@output`

Force the response format.

```
# @output json
```

Accepts `json` (the response is parsed/coerced as JSON) or `text`
(free-form text - the default). Auto-set to `json` when
`@output_schema` is present, so the schema and `# @output json` don't
need to appear together.

## `@output_schema`

Inline JSON Schema pinning the response shape. When provided, Strata
dispatches to provider-native structured output (OpenAI's `json_schema` with
strict mode; Anthropic's native tool-use) so the response comes back as
validated JSON rather than free-form text. Providers without schema support
fall back to `json_object` (valid JSON, shape not enforced), and the
`@validate_retries` loop catches shape violations.

```
# @output_schema {"type": "object", "properties": {"themes": {"type": "array", "items": {"type": "string"}}}, "required": ["themes"]}
```

Editing the schema invalidates the cell's cache: the schema is part of the
provenance hash.

## `@validate_retries`

Total attempts for the validate-and-retry loop (1 initial call + N-1 retries).
Defaults to 3. Only takes effect when `@output_schema` is set; each failed
validation feeds the prior response and path-addressed errors back as a retry
turn.

```
# @validate_retries 5
```

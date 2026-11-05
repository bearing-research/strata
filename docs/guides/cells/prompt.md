# Prompt cells

A prompt cell is a text template that gets rendered with upstream variable values, sent to an AI model, and the response stored as an artifact. Prompt cells participate in the DAG and cache by provenance exactly like Python cells: same inputs + same template + same model config = cache hit, no API call.

Create a prompt cell by picking **Prompt** from the **+ Add cell** menu. You never need to touch `notebook.toml` directly; editing the cell's source, wiring it into the DAG, and persisting the result all happen through the UI.

## Basic syntax

```
# @name summary
Summarize this dataset and return the top 3 findings as a numbered list:

{{ df }}
```

```text title="Output (illustrative model response)"
1. Setosa is linearly separable from versicolor and virginica based on petal dimensions alone.
2. Versicolor and virginica overlap moderately on sepal width but separate well by petal length.
3. The dataset is balanced (50 samples per species) with no missing values.
```

- `{{ df }}` is replaced with a text representation of the upstream variable `df` before sending to the model.
- The model's response is stored as an artifact named `summary` (from `# @name`).
- Downstream cells can read `summary` like any other upstream variable.

## Template syntax

Variables are injected with `{{ expression }}`. The expression is resolved against upstream cell outputs and converted to text using type-specific rules:

| Upstream type     | Text representation                                        |
| ----------------- | ---------------------------------------------------------- |
| pandas DataFrame  | Shape and column list, then every column, as many rows as fit |
| pandas Series     | Name, length and dtype, then as many values as fit         |
| numpy ndarray     | Shape + dtype + first 10 elements                          |
| dict / list       | JSON, indented                                             |
| Python object (stored as a pickle: a class instance, a fitted model) | A placeholder. The server never unpickles a cell's output, so the prompt says the value is a Python object and the cell's console names it. Convert it in the producing cell (`str(model)`, a dict). |
| anything else     | `str(value)`                                               |

Each variable has a 2,000-token budget (about 8,000 characters) per template render. A frame or series drops rows past the budget and ends with `... (N more rows)`; other values are cut with a `... (truncated)` marker.

**Attribute access** is supported for safe read-only operations:

```
{{ df.describe() }}     # OK, pandas describe() is allow-listed
{{ df.head() }}         # OK
{{ df.shape }}          # OK, attribute access (non-callable)
{{ df.to_csv() }}       # blocked (unknown method); left as-is in the template
```

Only a small set of methods is permitted (`describe`, `head`, `tail` on pandas objects), called with no arguments. Arbitrary method calls are blocked to keep template rendering side-effect-free.

## Prompt-cell annotations

| Annotation               | What it does                                                              | Default              |
| ------------------------ | ------------------------------------------------------------------------- | -------------------- |
| `# @name <identifier>`   | Output variable name; must be a Python identifier                         | `result`             |
| `# @model <model_id>`    | Override the notebook-level AI model                                      | From provider config |
| `# @temperature <float>` | Sampling temperature (0.0 = deterministic; see [Caching](#caching) below) | `0.0`                |
| `# @max_tokens <int>`    | Response token ceiling                                                    | `4096`               |
| `# @system <text>`       | System prompt prepended to the request (one line; the last one wins)      | None                 |
| `# @output json\|text`   | Coerce the response to JSON (or keep as free-form text)                   | `text`               |
| `# @output_schema {…}`   | Inline JSON Schema pinning the response shape                             | None                 |
| `# @validate_retries N`  | Total attempts for the validate-and-retry loop (1 initial + N−1 retries)  | `3`                  |
| `# @nocache`             | Call the model on every run; never serve a cache hit                      | Cache on             |

Example using several at once:

```
# @name classification
# @model gpt-5.4
# @temperature 0.0
# @max_tokens 1000
# @system You are a data scientist. Return only valid JSON.
Classify each paper by topic:

{{ sampled_papers }}

Return a JSON object mapping paper ID to topic.
```

## Schema-constrained output

`# @output_schema {...}` pins the shape of the model response to an inline JSON Schema. Strata picks the best provider-native path:

| Provider                             | Enforcement                                                                                                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI / Gemini**                  | Native `response_format: {type: "json_schema"}` for any base URL containing `openai`, which includes Gemini's OpenAI-compatible endpoint. `additionalProperties: false` is auto-injected at every `object` node; strict mode is used when the user's `required` list covers every property (otherwise relaxed to `strict: false`). |
| **Anthropic**                        | Native `/v1/messages` with tool-use: the schema is sent as a tool's `input_schema` and `tool_choice` is forced to that tool. The returned `tool_use.input` is extracted verbatim.                                                         |
| **Mistral / Ollama / vLLM**          | Fallback to `response_format: {type: "json_object"}`: valid JSON guaranteed, shape not enforced server-side. Client-side validation (see below) fills the gap.                                                                           |
| **Servers that reject the extensions** | Some OpenAI-compatible servers 400 on `response_format` or `stream_options` outright. Strata retries once without them, appending a schema-guidance system turn; the result is marked degraded and the validate-and-retry loop carries full enforcement. |

Setting `@output_schema` implies `@output json`; you don't need both.

Validation is lenient about packaging: when a provider wraps otherwise-valid
JSON in code fences or prose, Strata extracts the JSON document before
validating instead of burning a retry on the wrapper.

Example: triage each review into a structured record.

```
# @name triage
# @output_schema {"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"sentiment":{"type":"string","enum":["positive","negative","neutral"]},"priority":{"type":"string","enum":["low","medium","high"]},"tags":{"type":"array","items":{"type":"string"}}},"required":["sentiment","priority","tags"]}}},"required":["items"]}
Triage these customer reviews:

{{ reviews }}

For each review return sentiment, priority, and 1–3 short tags.
```

A downstream cell can then destructure without regex-wrangling:

```python
import pandas as pd

df = pd.DataFrame(triage["items"])
print(df["priority"].value_counts())
```

!!! note "Long schemas don't have an escape hatch yet"
    `# @output_schema` is parsed as a single JSON value from a single
    annotation line (see `prompt_analyzer.py`). For real-world
    schemas this gets long fast and the resulting cell source isn't
    pretty.

    Two pragmatic workarounds while a multi-line / file-reference
    syntax is on the backlog:

    - **Generate the schema in an upstream Python cell** and write
      the schema-driven prompt cell by hand. The Python cell's
      `json.dumps(...)` keeps the schema readable in source; the
      prompt cell pastes the result inline.
    - **Keep the schema in a `schemas/`-style sibling file** inside
      the notebook directory, and have a setup cell load + cache it
      as a string. Downstream prompt cells can use a shorter inline
      schema for caching while documenting the rich version
      elsewhere.

    Track the multi-line annotation feature in the issue tracker if
    this is blocking you.

```text title="Output"
priority
high      4
medium    2
low       1
Name: count, dtype: int64
```

## Validate-and-retry

When `@output_schema` is set, Strata runs a **validate-and-retry loop** after every model call:

1. Parse the response as JSON and run it through `jsonschema`.
2. On success → store the artifact and return.
3. On failure → append the bad response as an `assistant` turn, feed the validator's path-addressed errors back as a `user` turn, and retry.
4. On retry exhaustion → surface a cell error with the last validator messages.

The default is 3 total attempts (1 initial + 2 retries). Override with `# @validate_retries N`. Cumulative input/output tokens across all attempts are recorded on the artifact so cost accounting is accurate. The retry count is surfaced on the cell result (`validation_retries`), and when non-zero the cell's console line ends with `| Validation retries: N`.

Retries are mostly invisible on OpenAI-strict (OpenAI, Gemini) and Anthropic-native paths because the provider enforces the schema at decode time. They earn their keep on the `json_object` fallback path (Mistral, Ollama, vLLM) where the provider only guarantees _syntactic_ JSON.

## Caching

A prompt cell's provenance hash mixes together:

- The rendered template text (after `{{ var }}` injection)
- The provenance hash of each upstream variable the template reads (rendering truncates, so the text alone would miss an edit past the cut)
- Model name
- Temperature
- `@max_tokens`
- System prompt
- Output type (`json` / `text`)
- Output schema fingerprint (when set)

Editing any of these invalidates the cache. In particular, tweaking `@output_schema` on a cached cell forces a fresh call, which is what you want when iterating on the response shape.

!!! tip "Keep temperature at 0.0 for prompt cells"
With `temperature=0.0` the model is deterministic: same inputs → same output, and cache behavior is intuitive. Bumping temperature makes the first response "sticky" in the cache: future runs return the stored stochastic sample rather than re-sampling.

## Provider configuration

Set an API key in the **Runtime panel**, under **Notebook settings** in the sidebar, in Environment Variables. The key determines which provider is used:

| Environment Variable | Provider                        | Default Model        |
| -------------------- | ------------------------------- | -------------------- |
| `ANTHROPIC_API_KEY`  | Anthropic                       | claude-sonnet-4-6    |
| `OPENAI_API_KEY`     | OpenAI                          | gpt-5.4              |
| `GEMINI_API_KEY`     | Google                          | gemini-3-flash       |
| `MISTRAL_API_KEY`    | Mistral                         | mistral-large-latest |
| `STRATA_AI_API_KEY`  | Custom (requires `[ai]` config) | set in `[ai]`        |

**Resolution order** (highest priority wins):

1. `notebook.toml` `[ai]` section, per-notebook advanced overrides (see below)
2. Runtime panel env vars, set in the UI
3. Server config (`STRATA_AI_*` env vars), the admin default

For standard providers you only need step 2: drop your API key into the Runtime panel and Strata picks the matching default base URL and model.

!!! note "Process environment is not consulted"
    A shell-exported `OPENAI_API_KEY` does **not** leak into notebooks. This is intentional: each notebook must explicitly opt in to an AI provider.

For self-hosted models (Ollama, vLLM) or custom endpoints, add an `[ai]` section to `notebook.toml` directly:

```toml
[ai]
base_url = "http://localhost:11434/v1"
model = "llama3"
```

The `[ai]` section accepts `api_key` (use sparingly: it persists in `notebook.toml`; prefer the Runtime panel), `base_url`, `model`, `max_output_tokens` and `timeout_seconds` (default 60, or the server's `STRATA_AI_TIMEOUT_SECONDS`; a prompt cell ignores `# @timeout`). `model` is the notebook's default for every prompt cell, and there is no UI for it: set it here, or override it for one cell with `# @model`. Any service that implements the OpenAI `/v1/chat/completions` endpoint works, including OpenAI, Anthropic, Google and Mistral through their OpenAI-compatible endpoints, Ollama, and self-hosted vLLM, TGI or LiteLLM.

On a server in service mode the request is made by the server, so a `base_url` set here is checked like an `@fetch` URL: a host on a private, loopback or link-local address is refused unless the operator lists it in `STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS`. Personal mode reaches `localhost` as before.

The server's `STRATA_AI_API_KEY` is sent only to the server's own `STRATA_AI_BASE_URL` or a provider's default URL. A `base_url` set here that names another host needs a key from the notebook (the Runtime panel, or `api_key`), in every mode; otherwise the cell reports that the LLM is not configured.

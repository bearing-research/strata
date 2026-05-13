# AI Integration

Strata Notebook has two ways to use AI: **prompt cells** (declarative, part of the DAG) and the **AI assistant** (conversational, in a sidebar panel). Both use the same provider configuration and support any OpenAI-compatible API.

This page covers provider configuration and the AI assistant. For the prompt-cell template syntax, annotations, schema-constrained output, and validate-and-retry loop, see [Cell Types](cells.md#prompt-cells).

---

## Configuration

Set an API key in the **Runtime panel** under Environment Variables. The key determines which provider is used:

| Environment Variable | Provider                        | Default Model        |
| -------------------- | ------------------------------- | -------------------- |
| `ANTHROPIC_API_KEY`  | Anthropic                       | claude-sonnet-4-6    |
| `OPENAI_API_KEY`     | OpenAI                          | gpt-5.4              |
| `GEMINI_API_KEY`     | Google                          | gemini-3-flash       |
| `MISTRAL_API_KEY`    | Mistral                         | mistral-large-latest |
| `STRATA_AI_API_KEY`  | Custom (requires `[ai]` config) |, |

**Resolution order** (highest priority wins):

1. `notebook.toml` `[ai]` section, per-notebook advanced overrides (see below)
2. Runtime panel env vars, set in the UI
3. Server config (`STRATA_AI_*` env vars) admin default

For standard providers you only need step 2: drop your API key into the Runtime panel and Strata auto-picks the matching default base URL and model. The AI panel's model picker lets you switch models without leaving the UI (it persists the choice to `[ai].model`).

!!! note "Process environment is not consulted"
A shell-exported `OPENAI_API_KEY` does **not** leak into notebooks. This is intentional, each notebook must explicitly opt in to an AI provider. See the [Annotations](annotations.md) page for how env vars flow.

### Custom Provider Configuration

For self-hosted models (Ollama, vLLM) or custom endpoints there's no UI for the `base_url` / timeout / token-ceiling fields, so you add an `[ai]` section to `notebook.toml` directly:

```toml
[ai]
base_url = "http://localhost:11434/v1"
model = "llama3"
```

This is the intended escape hatch for advanced config. Fields the `[ai]` section accepts:

- `api_key`, *use sparingly*, persists in `notebook.toml` even for blanked sensitive keys. Prefer the Runtime panel.
- `base_url`
- `model`
- `max_context_tokens`
- `max_output_tokens`
- `timeout_seconds`

### Supported Providers

Any service that implements the OpenAI `/v1/chat/completions` endpoint works, including:

- OpenAI (GPT-4o, GPT-4, GPT-3.5)
- Anthropic (Claude, via their OpenAI-compatible endpoint)
- Google (Gemini, via their OpenAI-compatible endpoint)
- Mistral (Mistral Large, Codestral)
- Ollama (local models)
- vLLM, TGI, LiteLLM (self-hosted)

---

## AI Assistant

The AI assistant is a sidebar panel (toggle with the **AI Assistant** button) that provides conversational access to a model. It operates outside the DAG, it doesn't create artifacts or participate in caching.

### Chat Mode (Enter)

Type a message and press Enter. The assistant streams a response with full conversation context:

- **Conversation memory**: prior turns are sent back to the model so follow-up questions work ("give an example of that", "now do it for column X")
- **Notebook context**: the current notebook state (cell sources, variable definitions, packages) is included in every request as a system prompt
- **Cell context**: optionally select a cell from the dropdown to focus the conversation on that cell's code and errors
- **Code insertion**: assistant responses with fenced code blocks show an "Insert Cell" button to add the code as a new notebook cell

The conversation resets when you click "Clear" or reload the page. History is session-only (not persisted to disk).

### Agent Mode (Shift+Enter)

Type an instruction and press Shift+Enter. The agent autonomously takes actions on the notebook:

**Available tools:**

| Tool                 | Description                                     |
| -------------------- | ----------------------------------------------- |
| `get_notebook_state` | Read all cells, variables, and execution status |
| `create_cell`        | Add a new Python or prompt cell                 |
| `edit_cell`          | Modify an existing cell's source                |
| `delete_cell`        | Remove a cell                                   |
| `run_cell`           | Execute a cell and observe the result           |
| `add_package`        | Install a Python package via uv                 |

The agent runs as a background task with a 10-iteration limit. Progress events appear in the panel as they happen. You can cancel a running agent with the Cancel button.

**Example agent instructions:**

- "Add a cell that loads the iris dataset and prints its shape"
- "Install pandas and create a simple data analysis"
- "Fix the error in cell c3 and run it again"

The agent works best for additive tasks (creating new cells, installing packages). For complex refactoring, use Chat mode to discuss the approach first.

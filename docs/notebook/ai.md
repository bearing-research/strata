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
- `approval_timeout_seconds`, how long an agent confirm prompt waits before being treated as a decline (default 120)

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

### Safety surface

Before granting the assistant write access, understand what it can and can't do.

**Approval-gated tools.** `delete_cell` and `add_package` always go through a confirm prompt in the UI ("agent_confirm_request" WebSocket message) before running. The approval future times out after **120 s by default** and is treated as a decline so a closed tab doesn't leave the loop hanging; configure it with `STRATA_AI_APPROVAL_TIMEOUT_SECONDS` on the server or `approval_timeout_seconds` in the notebook's `[ai]` section. Approval can be skipped with the **Auto-approve** toggle in the AI panel footer — that suppresses the gate for the remainder of the session.

**Non-gated mutating tools.** `create_cell`, `edit_cell`, and `run_cell` execute without prompting. `edit_cell` overwrites the cell source; `run_cell` executes whatever is currently in the cell. Neither has an undo. (Cell source is autosaved to `cells/*.py`, so git is the practical undo for `edit_cell` and `delete_cell`. Side effects of `run_cell` — files written, packages mutated, API calls made — are not reversible.)

**Loop bounds.**

| Bound | Default | Source |
| --- | --- | --- |
| Iterations (tool-use rounds) | **10** | `max_iterations` in `run_agent_loop` |
| Approval timeout | **120 s** | `STRATA_AI_APPROVAL_TIMEOUT_SECONDS` / `[ai] approval_timeout_seconds` |
| Conversation memory | **12 turns** (6 user/assistant pairs) | `HISTORY_MAX_TURNS` |
| Per-call output tokens | `STRATA_AI_MAX_OUTPUT_TOKENS` (default 4096) | LLM config |
| Per-call context tokens | `STRATA_AI_MAX_CONTEXT_TOKENS` (default 100000) | LLM config |

There is **no aggregate token budget** across iterations — a 10-iteration run can consume up to 10× the per-call limits. If you're using a metered provider, expect costs roughly proportional to (notebook context size + conversation history + tool-call traces) × iterations.

**What's NOT bounded.**

- **Package allowlist.** `add_package` accepts any pip-compatible package spec. Approval-gated, so the user sees the spec before install, but there's no server-side allowlist or signature check. `pandas>=2.0` and `evil-package@git+https://...` both pass the same gate.
- **Mount / credential access.** `run_cell` executes in the notebook's normal execution context. It sees the notebook's mounts, env vars (including any unblanked secrets in the runtime panel), and any artifacts already in the store. Don't grant agent access to a notebook with production credentials unless you also trust the assistant's prompts.
- **Network access from cells.** No sandboxing. A cell created and run by the agent can make outbound HTTP calls, read/write to mounted buckets, hit external APIs — same as a cell you wrote by hand.
- **Filesystem reads outside the notebook directory.** Same as a hand-written cell — Python `open()` works wherever the strata-notebook process has permission. Inside a Docker / Fly deployment this is usually limited to the container, but a local-dev `uv run strata-notebook` has full user-account access.

### Package install scoping

`add_package` calls `uv add <package>` against the **per-notebook**
`pyproject.toml` (`dependencies.py:688`). The package gets resolved
into the notebook's local `.venv/`, the local `uv.lock` is updated,
and `pyproject.toml` records the new entry on disk. Three
consequences:

- **Scope is the notebook, not the host.** Other notebooks aren't
  affected; strata-notebook's own venv isn't affected; system Python
  isn't touched.
- **The change is persistent.** Once installed, the package stays
  in the notebook's `pyproject.toml` until removed; closing and
  reopening the notebook doesn't un-do it. Inspect the diff before
  committing to git.
- **Concurrent installs serialize.** `dependencies.py` holds a
  per-notebook lock around `uv add` / `uv remove`, so two agent
  iterations can't race on the same lockfile.

`add_package` cannot escape the notebook venv. It cannot install
into the system Python and cannot reach across to another notebook.

### Concurrent edit with an open editor

If the user has a cell open in the editor while the agent calls
`edit_cell` or `delete_cell`:

- The agent writes the new source to `cells/<id>.py` and calls
  `session.reload()`, then broadcasts a fresh `notebook_state` over
  WebSocket via `broadcast_notebook_sync` (`agent.py:450`).
- Every connected frontend tab — including the user's — replaces its
  cached state with the broadcast. The editor view re-renders with
  the agent's new source.
- **The user's unflushed keystrokes are lost.** Source edits are
  buffered locally in the frontend and flush via debounced
  `cell_source_update` after 2 s idle / on blur / before run. If
  the agent's broadcast arrives while a buffer is pending, the
  buffer is overwritten by the broadcast on the next render. There
  is no merge-conflict prompt.
- For `delete_cell`, the cell disappears from the user's view
  entirely; the editor focus moves to the next cell.
- For `run_cell`, the user's tab sees the cell transition through
  `running → ready` via `cell_status` broadcasts; the cell didn't
  start from a button the user clicked, which can be confusing.

In practice the user is reading the agent's progress in the AI
panel and notices the cell changes there too, but if you're prone
to typing into a buffer while the agent works, save first
(Ctrl+S in the editor, or Shift+Enter to run).

**Conversation memory is per-notebook.** Agent history (the last 12 user/assistant text turns — tool traces are never kept) is persisted to the notebook's `.strata/agent_history.json`, so it survives a `strata-notebook` restart. Clicking **Clear** in the panel removes it, in memory and on disk. Like everything under `.strata/`, it's gitignored runtime state.

**Recommended posture.**

- For routine work, leave Auto-approve off so destructive actions surface a confirm.
- Don't put production database credentials in a notebook the agent has access to; use a separate notebook (or a service-mode deployment with proxy auth).
- After an agent run, review the diff in `cells/*.py` before pushing — the agent can rewrite cells without ceremony.

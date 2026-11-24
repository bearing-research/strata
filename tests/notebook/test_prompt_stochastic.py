"""Each run of a ``# @stochastic`` prompt cell is its own version, as a Python cell's is.

Stored under the cell's plain output key, a second response took the first one's place.
"""

from __future__ import annotations

from unittest import mock

import pytest
from vestigo.provenance import derive_subkey

from strata.notebook.llm import LlmCompletionResult, LlmConfig
from strata.notebook.parser import parse_notebook
from strata.notebook.prompt_executor import compute_prompt_provenance_hash, execute_prompt_cell
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

TEMPLATE = "# @stochastic\n# @temperature 0.8\nName a colour.\n"


@pytest.mark.asyncio
async def test_a_stochastic_prompt_keeps_every_response(tmp_path):
    nb = create_notebook(tmp_path, "colours", initialize_environment=False)
    add_cell_to_notebook(nb, "p1", None, language="prompt")
    write_cell(nb, "p1", TEMPLATE)
    session = NotebookSession(parse_notebook(nb), nb)
    replies = iter(["red", "blue"])

    async def fake(_config, _messages, **_kwargs):
        return LlmCompletionResult(
            content=next(replies), model="m", input_tokens=1, output_tokens=1
        )

    config = LlmConfig(base_url="https://api.openai.com/v1", api_key="sk", model="m")
    with mock.patch("strata.notebook.prompt_executor.chat_completion", fake):
        for _ in range(2):
            result = await execute_prompt_cell(session, "p1", TEMPLATE, config, use_cache=True)
            assert result["success"], result

    store = session.get_artifact_manager().artifact_store
    rows = [
        store.get_artifact(f"nb_{session.notebook_state.id}_cell_p1_var_result", v) for v in (1, 2)
    ]
    assert all(r is not None for r in rows)
    assert [r.state for r in rows] == ["ready", "ready"], "both responses persist"
    assert len({r.provenance_hash for r in rows}) == 2
    plain = compute_prompt_provenance_hash(
        rendered="Name a colour.",
        model="m",
        temperature=0.8,
        system_prompt=None,
        output_type="text",
        output_schema=None,
        max_tokens=config.max_output_tokens,
        input_hashes=[],
    )
    assert store.find_by_provenance(derive_subkey(plain, "result")) is None

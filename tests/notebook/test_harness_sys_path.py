"""A cell imports the user's packages, never Strata's own modules of the same name.

Started as ``python harness.py``, a process puts the harness's directory first on
``sys.path``, so Strata's ``datasets.py`` and ``tables.py`` shadowed Hugging Face
``datasets`` and PyTables. Every process that runs a cell or reads its values starts
with ``-P`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

import strata.notebook
from strata.notebook.executor import CellExecutor
from strata.notebook.llm import LlmCompletionResult, LlmConfig
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.prompt_executor import execute_prompt_cell
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

_PACKAGE = str(Path(strata.notebook.__file__).parent)
# Names of modules in Strata's notebook package that a cell could mean as its own.
_NAMES = ("datasets", "tables", "serializer")
SOURCE = (
    "import importlib.util, json\n"
    "found = {n: getattr(importlib.util.find_spec(n), 'origin', None) for n in "
    f"{_NAMES!r}}}\n"
    "print(json.dumps(found))\n"
)


def _session(tmp_path: Path) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "Shadowing")
    add_cell_to_notebook(notebook_dir, "c1", None)
    write_cell(notebook_dir, "c1", SOURCE)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _assert_nothing_of_strata(stdout: str) -> None:
    found = json.loads(stdout.strip().splitlines()[-1])
    assert set(found) == set(_NAMES)
    assert [o for o in found.values() if o and o.startswith(_PACKAGE)] == []


async def test_a_cold_run(tmp_path):
    result = await CellExecutor(_session(tmp_path)).execute_cell("c1", SOURCE)

    assert result.success, result.error
    assert result.execution_method == "cold"
    _assert_nothing_of_strata(result.stdout)


@pytest.mark.integration
@pytest.mark.warm_pool
async def test_a_warm_run(tmp_path, warm_pool):
    session = _session(tmp_path)
    pool = await warm_pool(session)

    result = await CellExecutor(session, pool).execute_cell("c1", SOURCE)

    assert result.success, result.error
    assert result.execution_method == "warm"
    _assert_nothing_of_strata(result.stdout)


async def test_a_run_all_batch(tmp_path):
    spec = {
        "cell_id": "c1",
        "source": SOURCE,
        "consumed_vars": [],
        "env": {},
        "mount_manifest": {},
        "source_hash": "",
        "env_hash": "",
    }

    result = await CellExecutor(_session(tmp_path)).execute_batch([spec])

    (ran,) = result.cell_results
    assert ran.status == "ok", ran.error
    _assert_nothing_of_strata(ran.stdout)


async def test_a_run_on_a_worker(tmp_path, notebook_executor_server):
    session = _session(tmp_path)
    session.notebook_state.workers = [
        WorkerSpec(
            name="remote",
            backend=WorkerBackendType.EXECUTOR,
            config={"url": notebook_executor_server["execute_url"]},
        )
    ]
    session.notebook_state.get_cell("c1").worker = "remote"

    result = await CellExecutor(session).execute_cell("c1", SOURCE)

    assert result.success, result.error
    assert result.execution_method == "executor"
    _assert_nothing_of_strata(result.stdout)


async def test_reading_an_object_for_a_prompt(tmp_path):
    """The object's ``str()`` runs in its own process, as the producing cell's code."""
    producer = (
        "class Where:\n"
        "    def __str__(self):\n"
        "        import importlib.util, json\n"
        f"        return json.dumps({{n: getattr(importlib.util.find_spec(n), 'origin', None)"
        f" for n in {_NAMES!r}}})\n"
        "where = Where()\n"
    )
    notebook_dir = create_notebook(tmp_path, "Prompted")
    add_cell_to_notebook(notebook_dir, "prod", None)
    write_cell(notebook_dir, "prod", producer)
    add_cell_to_notebook(notebook_dir, "p1", "prod", language="prompt")
    write_cell(notebook_dir, "p1", "{{ where }}")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    produced = await CellExecutor(session).execute_cell("prod", producer)
    assert produced.success, produced.error
    prompts: list[str] = []

    async def fake(_config, messages, **_kwargs):
        prompts.append(messages[-1]["content"])
        return LlmCompletionResult(content="ok", model="m", input_tokens=1, output_tokens=1)

    config = LlmConfig(base_url="https://api.openai.com/v1", api_key="sk", model="m")
    with mock.patch("strata.notebook.prompt_executor.chat_completion", fake):
        result = await execute_prompt_cell(session, "p1", "{{ where }}", config, use_cache=False)

    assert result["success"], result["error"]
    (prompt,) = prompts
    _assert_nothing_of_strata(prompt)

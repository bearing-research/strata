"""What a prompt cell does with an upstream Python object, end to end.

The object is loaded in the notebook's interpreter, as the producing cell's code, and
its ``str()`` is the text. An object with no text of its own stops the cell before the
paid call: its default repr holds an address, so no run would hit the cache either.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

from strata.notebook.annotation_validation import blocking_diagnostic
from strata.notebook.executor import CellExecutor
from strata.notebook.llm import LlmCompletionResult, LlmConfig
from strata.notebook.parser import parse_notebook
from strata.notebook.prompt_executor import (
    _load_upstream_variables,
    _objects_to_text,
    _parse_output,
    execute_prompt_cell,
)
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _session(tmp_path: Path, producer: str, template: str) -> NotebookSession:
    nb = create_notebook(tmp_path, "prompt_objects")
    add_cell_to_notebook(nb, "prod", None)
    write_cell(nb, "prod", producer)
    add_cell_to_notebook(nb, "p1", "prod", language="prompt")
    write_cell(nb, "p1", template)
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


async def _prompt_with(tmp_path: Path, producer: str, template: str):
    """Run *producer*, then the prompt; the result and every prompt the model received."""
    session = _session(tmp_path, producer, template)
    produced = await CellExecutor(session).execute_cell("prod", producer)
    assert produced.success, produced.error
    calls: list[str] = []

    async def fake(_config, messages, **_kwargs):
        calls.append(messages[-1]["content"])
        return LlmCompletionResult(content="ok", model="m", input_tokens=1, output_tokens=1)

    config = LlmConfig(base_url="https://api.openai.com/v1", api_key="sk", model="m")
    with mock.patch("strata.notebook.prompt_executor.chat_completion", fake):
        result = await execute_prompt_cell(session, "p1", template, config, use_cache=False)
    return result, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("producer", "expected"),
    [
        (
            'class C:\n    def __str__(self):\n        print("noise")\n        return "custom"\n'
            "o = C()\n",
            "custom",
        ),
        (
            "from dataclasses import dataclass\n@dataclass\nclass C:\n"
            "    lr: float = 0.1\no = C()\n",
            "C(lr=0.1)",
        ),
    ],
)
async def test_an_object_with_text_reaches_the_model(tmp_path, producer, expected):
    result, calls = await _prompt_with(tmp_path, producer, "{{ o }}")
    assert result["success"], result["error"]
    assert calls == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("producer", "in_error"),
    [
        ("class C:\n    pass\no = C()\n", "has no text of its own"),
        ("def f():\n    pass\no = f\n", "has no text of its own"),
        (
            "from dataclasses import dataclass\nclass P:\n    pass\n"
            "@dataclass\nclass C:\n    p: object\no = C(P())\n",
            "has no text of its own",
        ),
        (
            'class C:\n    def __str__(self):\n        raise RuntimeError("nope")\no = C()\n',
            "RuntimeError: nope",
        ),
    ],
)
async def test_an_object_without_text_stops_before_the_call(tmp_path, producer, in_error):
    result, calls = await _prompt_with(tmp_path, producer, "{{ o }}")
    assert result["success"] is False
    assert in_error in result["error"]
    assert "`o`" in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_a_long_str_is_cut_with_the_marker(tmp_path):
    producer = 'class C:\n    def __str__(self):\n        return "x" * 100000\no = C()\n'
    result, calls = await _prompt_with(tmp_path, producer, "{{ o }}")
    assert result["success"], result["error"]
    assert calls[0] == "x" * 8000 + "\n... (truncated)"


@pytest.mark.asyncio
async def test_a_stored_value_with_no_text_form_is_refused(tmp_path):
    """An R object is not decoded as text: the model would answer the noise anyway."""
    session = _session(tmp_path, "o = 1\n", "{{ o }}")
    variables = {"o": _parse_output(b"\x00binary", "application/x-r-rds")}

    problems = await _objects_to_text(session, variables)

    assert len(problems) == 1
    assert "application/x-r-rds" in problems[0]


@pytest.mark.parametrize(
    ("template", "code", "line"),
    [
        ("{% if o %}a{% endif %}", "prompt_template_invalid", 2),
        ("{{ o | upper }}", "prompt_template_invalid", 2),
        ("{{ o['k'] }}", "prompt_template_invalid", 2),
        ("{{ o(1) }}", "prompt_template_invalid", 2),
        ("{{ o._private }}", "prompt_template_invalid", 2),
        ("{{ }}", "prompt_template_invalid", 2),
        ("open {{ o", "prompt_template_invalid", 2),
        ("first line\nsecond {{ o | upper }}", "prompt_template_invalid", 3),
        ("{{ nope }}", "prompt_input_undefined", None),
    ],
)
def test_a_bad_template_keeps_the_cell_from_running(tmp_path, template, code, line):
    session = _session(tmp_path, "o = 1\n", f"# @name answer\n{template}\n")
    cell = session.notebook_state.get_cell("p1")
    assert cell is not None

    diagnostic = blocking_diagnostic(cell, session.notebook_state, session.analysis)

    assert diagnostic is not None
    assert diagnostic.code == code
    assert diagnostic.line == line


@pytest.mark.parametrize(
    "template",
    [
        "{{ o }}",
        "{{ o.shape }}",
        "{{ o.head() }}",
        r"\{{ literal }} and \{% literal %} and {{ o }}",
        '{"a": {"b": {{ o }}}}',
        "no placeholders",
    ],
)
def test_a_good_template_runs(tmp_path, template):
    session = _session(tmp_path, "o = 1\n", f"# @name answer\n{template}\n")
    cell = session.notebook_state.get_cell("p1")
    assert cell is not None

    assert blocking_diagnostic(cell, session.notebook_state, session.analysis) is None


@pytest.mark.asyncio
async def test_reaching_into_an_object_names_the_object(tmp_path):
    producer = "class C:\n    coef = 3\n    def __str__(self):\n        return 'C'\no = C()\n"
    result, calls = await _prompt_with(tmp_path, producer, "{{ o.coef }}")
    assert result["success"] is False
    assert "`o` is a Python object" in result["error"]
    assert calls == []


@pytest.mark.asyncio
async def test_a_set_reads_the_same_on_every_run(tmp_path):
    """A different order would be a different key: every run a new paid call."""
    from strata.notebook import prompt_executor

    producer = (
        "class C:\n    def __init__(self):\n        self.tags = {f'tag{i}' for i in range(50)}\n"
        "    def __str__(self):\n        return str(self.tags)\no = C()\n"
    )
    session = _session(tmp_path, producer, "{{ o }}")
    assert (await CellExecutor(session).execute_cell("prod", producer)).success

    texts = set()
    for _ in range(3):
        prompt_executor._OBJECT_TEXTS.clear()
        variables, _hashes = _load_upstream_variables(session, "p1")
        assert await _objects_to_text(session, variables) == []
        texts.add(variables["o"])

    assert len(texts) == 1


@pytest.mark.asyncio
async def test_an_object_is_read_once_per_version(tmp_path):
    from strata.notebook import prompt_executor

    producer = "class C:\n    def __str__(self):\n        return 'once'\no = C()\n"
    session = _session(tmp_path, producer, "{{ o }}")
    assert (await CellExecutor(session).execute_cell("prod", producer)).success
    spawn = prompt_executor.asyncio.create_subprocess_exec

    for expected_spawns in (1, 0):
        with mock.patch.object(
            prompt_executor.asyncio, "create_subprocess_exec", side_effect=spawn
        ) as spawned:
            variables, _hashes = prompt_executor._load_upstream_variables(session, "p1")
            assert await prompt_executor._objects_to_text(session, variables) == []
        assert variables["o"] == "once"
        assert spawned.call_count == expected_spawns


def test_problems_keep_their_order_when_one_repeats(tmp_path):
    session = _session(
        tmp_path, "o = 1\n", "# @name answer\n{% if o %}\n{{ o | upper }}\n{% endif %}\n"
    )
    cell = session.notebook_state.get_cell("p1")
    assert cell is not None

    diagnostic = blocking_diagnostic(cell, session.notebook_state, session.analysis)

    assert diagnostic is not None
    assert diagnostic.line == 2


@pytest.mark.asyncio
async def test_a_cancelled_read_leaves_no_process(tmp_path):
    pid_file = tmp_path / "pid"
    producer = (
        "import os, time\nclass C:\n    def __str__(self):\n"
        f"        open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "        time.sleep(60)\n        return 'late'\no = C()\n"
    )
    session = _session(tmp_path / "nb", producer, "{{ o }}")
    assert (await CellExecutor(session).execute_cell("prod", producer)).success
    variables, _hashes = _load_upstream_variables(session, "p1")

    task = asyncio.ensure_future(_objects_to_text(session, variables))
    for _ in range(200):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    pid = int(pid_file.read_text())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)

"""The server never unpickles cell data.

Cell code may run as a separate harness user, and an imported snapshot carries
artifacts someone else wrote: unpickling either in the server would run that
code as the server. The producer here writes a pickle whose load appends the
loading process's pid to a marker file.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from unittest import mock

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _marker_code(path: Path) -> str:
    return f"open({str(path)!r}, 'a').write(str(__import__('os').getpid()) + chr(10))"


class _Marker:
    """Unpickling appends the loading process's pid to ``path``."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __reduce__(self):
        return (exec, (_marker_code(self.path),))


def _producer_source(marker: Path) -> str:
    # The cell controls the bytes it stores: it swaps the codec envelope for the payload.
    return (
        "import sys\n"
        "class Evil:\n"
        "    def __reduce__(self):\n"
        f"        return (exec, ({_marker_code(marker)!r},))\n"
        'sys.modules["_nb_serializer"]._wrap_codec_payload = lambda name, payload: Evil()\n'
        "evil = Evil()\n"
    )


def _session(tmp_path: Path, cells: list[tuple[str, str, str | None, str]]) -> NotebookSession:
    nb = create_notebook(tmp_path, "no_unpickle")
    for cell_id, source, after, language in cells:
        add_cell_to_notebook(nb, cell_id, after, language=language)
        write_cell(nb, cell_id, source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


async def _run(session: NotebookSession, cell_id: str) -> CellExecutor:
    executor = CellExecutor(session)
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    result = await executor.execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return executor


def _server_unpickled(marker: Path) -> bool:
    return marker.exists() and str(os.getpid()) in marker.read_text().split()


def _llm_config():
    from strata.notebook.llm import LlmConfig

    return LlmConfig(base_url="https://api.openai.com/v1", api_key="sk", model="m")


def _fake_llm():
    from strata.notebook.llm import LlmCompletionResult

    calls: list[list[dict]] = []

    async def fake(config, messages, **_kwargs):
        calls.append([dict(m) for m in messages])
        return LlmCompletionResult(content="ok", model="m", input_tokens=1, output_tokens=1)

    return fake, calls


class TestPromptCells:
    @pytest.mark.asyncio
    async def test_object_input_is_a_placeholder_not_unpickled(self, tmp_path):
        from strata.notebook.prompt_executor import _OBJECT_PLACEHOLDER, execute_prompt_cell

        marker = tmp_path / "marker"
        session = _session(
            tmp_path,
            [
                ("prod", _producer_source(marker), None, "python"),
                ("p1", "Describe {{ evil }}", "prod", "prompt"),
            ],
        )
        await _run(session, "prod")
        marker.unlink(missing_ok=True)
        fake, calls = _fake_llm()

        with mock.patch("strata.notebook.prompt_executor.chat_completion", fake):
            result = await execute_prompt_cell(
                session, "p1", "Describe {{ evil }}", _llm_config(), use_cache=False
            )

        assert not _server_unpickled(marker)
        assert result["success"] is True
        assert calls[0][-1]["content"] == f"Describe {_OBJECT_PLACEHOLDER}"
        assert "Not rendered: evil (Python object)" in result["stderr"]

    def test_cached_object_artifact_is_not_unpickled(self, tmp_path):
        """A cache hit reads the content type the artifact declares, which an import can forge."""
        from strata.notebook.prompt_executor import _OBJECT_PLACEHOLDER, _parse_output

        marker = tmp_path / "marker"
        blob = pickle.dumps(_Marker(marker))

        value = _parse_output(blob, "pickle/object")

        assert not marker.exists()
        assert value is _OBJECT_PLACEHOLDER

    @pytest.mark.asyncio
    async def test_json_and_table_inputs_still_render(self, tmp_path):
        from strata.notebook.prompt_executor import execute_prompt_cell

        session = _session(
            tmp_path,
            [
                (
                    "prod",
                    "import pandas as pd\ncfg = {'k': 'v'}\ndf = pd.DataFrame({'x': [41]})\n",
                    None,
                    "python",
                ),
                ("p1", "{{ cfg }} {{ df }}", "prod", "prompt"),
            ],
        )
        await _run(session, "prod")
        fake, calls = _fake_llm()

        with mock.patch("strata.notebook.prompt_executor.chat_completion", fake):
            result = await execute_prompt_cell(session, "p1", "{{ cfg }} {{ df }}", _llm_config())

        rendered = calls[0][-1]["content"]
        assert '"k": "v"' in rendered
        assert "41" in rendered
        assert "Not rendered" not in result["stderr"]


class TestSqlBinds:
    @pytest.mark.asyncio
    async def test_object_bind_is_refused_not_unpickled(self, tmp_path):
        pytest.importorskip("adbc_driver_sqlite")
        marker = tmp_path / "marker"
        session = _session(
            tmp_path,
            [
                ("prod", _producer_source(marker), None, "python"),
                ("q1", "# @sql connection=db\nSELECT :evil AS v\n", "prod", "sql"),
            ],
        )
        toml = session.path / "notebook.toml"
        db = tmp_path / "db.sqlite"
        toml.write_text(
            toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db}"\n'
        )
        session = NotebookSession(parse_notebook(session.path), session.path)
        session.venv_python = Path(sys.executable)
        executor = await _run(session, "prod")
        marker.unlink(missing_ok=True)

        cell = session.notebook_state.get_cell("q1")
        assert cell is not None
        result = await executor.execute_cell("q1", cell.source)

        assert not _server_unpickled(marker)
        assert not result.success
        assert "bind param :evil has unsupported type 'PickledObject'" in (result.error or "")


class TestCellTests:
    @pytest.mark.asyncio
    async def test_inputs_are_deserialized_in_the_test_process(self, tmp_path):
        marker = tmp_path / "marker"
        session = _session(
            tmp_path,
            [
                ("prod", _producer_source(marker), None, "python"),
                ("c2", "y = evil\n", "prod", "python"),
            ],
        )
        executor = await _run(session, "prod")
        marker.unlink(missing_ok=True)

        result = await executor.run_cell_tests("c2", "def test_y(cell):\n    assert cell.y\n")

        assert not _server_unpickled(marker)
        # The load still happens, in the pytest subprocess that runs as the harness user.
        assert marker.exists()
        assert result.errored == 0

"""What a cell actually sees, driven through a real harness spawn.

Lives outside ``tests/notebook/`` on purpose: that package's conftest replaces
``_run_harness`` with a direct spawn to skip ``uv run``, and a filter is only
worth as much as the spawn sites that apply it. Here the shipping code runs.
Item 49.
"""

from __future__ import annotations

import json

import pytest

from strata.notebook.cli import run_main
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture(autouse=True)
def a_server_holding_secrets(monkeypatch):
    monkeypatch.setenv("STRATA_PROXY_TOKEN", "shhh")
    monkeypatch.setenv("HF_TOKEN", "hf_x")


def _notebook(tmp_path):
    nb = create_notebook(tmp_path, "Isolation", initialize_environment=False)
    (nb / ".venv").mkdir(exist_ok=True)  # --no-sync placeholder
    add_cell_to_notebook(nb, "peek", None, language="python")
    write_cell(
        nb,
        "peek",
        "import os\n"
        'print("TOKEN:", os.environ.get("STRATA_PROXY_TOKEN", "<absent>"))\n'
        'print("HF:", os.environ.get("HF_TOKEN", "<absent>"))\n',
    )
    return nb


def _run(nb, capsys):
    assert run_main([str(nb), "--no-sync", "--force", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    return next(c for c in payload["cells"] if c["id"] == "peek")["stdout"]


def test_without_the_setting_a_cell_reads_the_servers_secrets(tmp_path, capsys):
    """Stated as a test because it is what every deployment does today, and
    the reason the setting exists."""
    assert "TOKEN: shhh" in _run(_notebook(tmp_path), capsys)


def test_with_the_setting_it_does_not(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", "HF_TOKEN")

    stdout = _run(_notebook(tmp_path), capsys)

    assert "TOKEN: <absent>" in stdout
    # And the cell still got what it was given, and still ran at all — the
    # half a too-aggressive filter would break.
    assert "HF: hf_x" in stdout

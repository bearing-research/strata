"""What a cell actually sees, driven through a real harness spawn.

Lives outside ``tests/notebook/`` on purpose: that package's conftest swaps the
harness command to skip ``uv run``, and here the shipping command runs too.
Item 49.

The harness-user check at the bottom needs root and a second OS user, so it runs
only where both are arranged: ``STRATA_TEST_HARNESS_USER`` names the user.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

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


_HARNESS_USER = os.environ.get("STRATA_TEST_HARNESS_USER")


@pytest.mark.skipif(
    not (sys.platform == "linux" and _HARNESS_USER and os.geteuid() == 0),
    reason="needs Linux, root, and STRATA_TEST_HARNESS_USER naming a second user",
)
def test_a_cell_run_as_the_harness_user_cannot_read_the_servers_environment(tmp_path, monkeypatch):
    """The property the allowlist alone could not give: the server's
    environment is readable through /proc by anyone running as the server, and
    by nobody else."""
    import pwd

    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    nb = create_notebook(tmp_path, "Dropped", initialize_environment=False)
    # The harness user has to reach the notebook, which a root tmp_path hides.
    os.chmod(tmp_path, 0o755)
    add_cell_to_notebook(nb, "peek", None, language="python")
    source = (
        "import os\n"
        "print('UID:', os.getuid())\n"
        "print('HOME:', os.environ['HOME'])\n"
        "try:\n"
        "    open(f'/proc/{os.getppid()}/environ').read()\n"
        "    print('SERVER ENV: readable')\n"
        "except PermissionError:\n"
        "    print('SERVER ENV: refused')\n"
    )
    write_cell(nb, "peek", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = sys.executable
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(
            config=SimpleNamespace(deployment_mode="service", notebook_harness_user=_HARNESS_USER)
        ),
    )

    result = asyncio.run(CellExecutor(session).execute_cell("peek", source))

    assert result.success, result.error
    entry = pwd.getpwnam(_HARNESS_USER)
    assert f"UID: {entry.pw_uid}" in result.stdout
    assert f"HOME: {entry.pw_dir}" in result.stdout
    assert "SERVER ENV: refused" in result.stdout

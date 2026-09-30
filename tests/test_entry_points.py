"""The uv-only guard runs after argument parsing on both entry points.

`strata --help` used to hit the guard before parsing while `strata-notebook
--help` did not. Now both print help from any Python, and every real command
on both is refused outside a uv-managed environment before it does anything.
"""

from __future__ import annotations

import os

import pytest

from strata import cli, server


@pytest.fixture
def outside_uv(tmp_path, monkeypatch):
    """No pyvenv.cfg at all: the guard treats this as not uv-managed."""
    monkeypatch.setattr("sys.prefix", str(tmp_path))


@pytest.mark.parametrize("main", [cli.main, server.main], ids=["strata", "strata-notebook"])
def test_help_prints_from_any_python(outside_uv, capsys, main):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert "uv-managed" not in captured.err


def test_strata_without_a_command_prints_help_from_any_python(outside_uv, capsys):
    assert cli.main([]) == 0
    assert "usage:" in capsys.readouterr().out


def test_a_strata_command_is_refused_outside_uv(tmp_path, outside_uv, capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["validate", str(tmp_path)])

    assert exc_info.value.code == 1
    assert "uv-managed Python environment" in capsys.readouterr().err


def test_the_server_is_refused_outside_uv_before_it_starts(outside_uv, capsys, monkeypatch):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("the server started"))

    monkeypatch.delenv("STRATA_NOTEBOOK_STORAGE_DIR", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        server.main(["--notebook-dir", "."])

    assert exc_info.value.code == 1
    assert "uv-managed Python environment" in capsys.readouterr().err
    # The refusal comes before the command line is applied to the environment.
    assert "STRATA_NOTEBOOK_STORAGE_DIR" not in os.environ

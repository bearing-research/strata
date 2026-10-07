"""The uv-only guard runs after argument parsing on both entry points.

Both print help from any Python; every real command is refused outside a uv-managed environment.
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


@pytest.fixture
def refused_config(tmp_path, monkeypatch):
    """Service mode with no auth mode: the first start of an upgrade that set none."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("STRATA_DEPLOYMENT_MODE", "service")
    monkeypatch.delenv("STRATA_AUTH_MODE", raising=False)
    monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "art"))
    monkeypatch.setenv("STRATA_S3_SECRET_KEY", "minio-secret-value")


def test_a_refused_config_stops_the_server_with_its_reason(refused_config, monkeypatch):
    """One message on stderr and a non-zero exit, not a pydantic traceback."""
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pytest.fail("the server started"))

    with pytest.raises(SystemExit) as exc_info:
        server.main([])

    message = str(exc_info.value.code)
    assert message.startswith("Strata cannot start: ")
    assert "STRATA_AUTH_MODE" in message
    assert "input_value" not in message and "minio-secret-value" not in message


def test_importing_the_server_with_a_refused_config_does_not_raise(refused_config, monkeypatch):
    """``python -m strata`` imports the module before main() can print the reason."""
    monkeypatch.setattr(server, "_state", None)

    server._mount_mcp_if_enabled()

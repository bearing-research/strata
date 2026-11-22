"""The uv-managed-runtime startup guard."""

from __future__ import annotations

import sysconfig
from pathlib import Path

import pytest
from uv import find_uv_bin

from strata import _uv_runtime
from strata._uv_runtime import assert_uv_managed_runtime, is_uv_managed_runtime


def _write_pyvenv(prefix: Path, body: str) -> None:
    prefix.mkdir(parents=True, exist_ok=True)
    (prefix / "pyvenv.cfg").write_text(body)


def test_is_uv_managed_runtime_true_with_marker(tmp_path, monkeypatch):
    _write_pyvenv(
        tmp_path,
        "home = /opt/python/bin\nimplementation = CPython\nuv = 0.8.13\nversion_info = 3.13.3\n",
    )
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    assert is_uv_managed_runtime() is True


def test_is_uv_managed_runtime_false_without_marker(tmp_path, monkeypatch):
    _write_pyvenv(
        tmp_path,
        "home = /opt/python/bin\nimplementation = CPython\nversion_info = 3.13.3\n",
    )
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    assert is_uv_managed_runtime() is False


def test_is_uv_managed_runtime_false_without_pyvenv_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    assert is_uv_managed_runtime() is False


def test_assert_uv_managed_runtime_passes_with_marker(tmp_path, monkeypatch):
    _write_pyvenv(tmp_path, "uv = 0.8.13\n")
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    assert_uv_managed_runtime()


def test_assert_uv_managed_runtime_exits_without_marker(tmp_path, monkeypatch, capsys):
    _write_pyvenv(tmp_path, "implementation = CPython\n")
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    with pytest.raises(SystemExit) as exc_info:
        assert_uv_managed_runtime()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "uv-managed Python environment" in err
    assert "pip install" in err  # the docs' sentence: not supported
    assert "uv tool install strata-notebook" in err
    assert "uv sync" in err
    # A venv without the marker: say which line is missing, and where.
    assert "no `uv = ` line" in err
    assert str(tmp_path / "pyvenv.cfg") in err


def test_assert_uv_managed_runtime_names_a_conda_or_system_python(tmp_path, monkeypatch, capsys):
    """A conda env or system Python has no pyvenv.cfg, so the message must not point at one."""
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    with pytest.raises(SystemExit):
        assert_uv_managed_runtime()

    err = capsys.readouterr().err
    assert "not a virtual environment (conda or a system Python)" in err
    assert "uv tool install strata-notebook" in err
    assert "Looked for" not in err


# --- resolve_uv ---

_UV = "uv" + (sysconfig.get_config_var("EXE") or "")


def _no_bundled_uv() -> str:
    raise FileNotFoundError("no bundled uv")


def _fake_uv(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    uv = directory / _UV
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    return uv


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.delenv("STRATA_UV", raising=False)


def _home(monkeypatch, home: Path) -> None:
    # expanduser reads USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def test_strata_uv_overrides_every_other_uv(monkeypatch):
    monkeypatch.setenv("STRATA_UV", "/opt/conda/bin/uv")

    assert _uv_runtime.resolve_uv() == "/opt/conda/bin/uv"
    assert _uv_runtime.resolve_uv(users_project=True) == "/opt/conda/bin/uv"


def test_the_bundled_uv_comes_before_path(tmp_path, monkeypatch, no_override):
    """The uv installed with Strata is the one it was tested with, whatever PATH holds."""
    on_path = _fake_uv(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(on_path.parent))

    assert _uv_runtime.resolve_uv() == find_uv_bin() != str(on_path)


def test_a_users_project_gets_the_uv_on_path(tmp_path, monkeypatch, no_override):
    """Their uv wrote the lock, and a required-version may pin it."""
    on_path = _fake_uv(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(on_path.parent))

    assert _uv_runtime.resolve_uv(users_project=True) == str(on_path)


def test_a_users_project_without_a_uv_of_its_own_gets_the_bundled_one(
    tmp_path, monkeypatch, no_override
):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _home(monkeypatch, tmp_path)

    assert _uv_runtime.resolve_uv(users_project=True) == find_uv_bin()


def test_without_a_bundled_uv_path_is_used(tmp_path, monkeypatch, no_override):
    monkeypatch.setattr(_uv_runtime, "find_uv_bin", _no_bundled_uv)
    on_path = _fake_uv(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(on_path.parent))

    assert _uv_runtime.resolve_uv() == str(on_path)


def test_a_non_login_shell_finds_uv_in_its_installer_dir(tmp_path, monkeypatch, no_override):
    """ssh and cron often lack ~/.local/bin on PATH."""
    monkeypatch.setattr(_uv_runtime, "find_uv_bin", _no_bundled_uv)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _home(monkeypatch, tmp_path)
    installed = _fake_uv(tmp_path / ".local" / "bin")

    assert _uv_runtime.resolve_uv() == str(installed)


def test_no_uv_anywhere_is_none(tmp_path, monkeypatch, no_override):
    monkeypatch.setattr(_uv_runtime, "find_uv_bin", _no_bundled_uv)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _home(monkeypatch, tmp_path)

    assert _uv_runtime.resolve_uv() is None

"""Finding uv, and the startup guard that refuses to run without one."""

from __future__ import annotations

import sys
import sysconfig
from pathlib import Path

import pytest
from uv import find_uv_bin

from strata import _uv_runtime

_UV = "uv" + (sysconfig.get_config_var("EXE") or "")


def _fake_uv(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    uv = directory / _UV
    uv.write_text("#!/bin/sh\n")
    uv.chmod(0o755)
    return uv


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.delenv("STRATA_UV", raising=False)


@pytest.fixture
def no_bundled_uv(monkeypatch):
    monkeypatch.setattr(_uv_runtime, "_bundled_uv", lambda: None)


def _home(monkeypatch, home: Path) -> None:
    # expanduser reads USERPROFILE on Windows.
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


# --- resolve_uv ---


def test_strata_uv_overrides_every_other_uv(tmp_path, monkeypatch):
    chosen = _fake_uv(tmp_path / "chosen")
    monkeypatch.setenv("STRATA_UV", str(chosen))
    monkeypatch.setenv("PATH", str(_fake_uv(tmp_path / "bin").parent))

    assert _uv_runtime.resolve_uv() == str(chosen)
    assert _uv_runtime.resolve_uv(users_project=True) == str(chosen)


def test_a_strata_uv_naming_no_file_is_no_uv(tmp_path, monkeypatch):
    """A typo must not quietly run some other uv, nor pass for one."""
    monkeypatch.setenv("STRATA_UV", str(tmp_path / "uvv"))

    assert _uv_runtime.resolve_uv() is None


def test_the_bundled_uv_comes_before_path(tmp_path, monkeypatch, no_override):
    """The uv installed with Strata is the one it was tested with, whatever PATH holds."""
    on_path = _fake_uv(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(on_path.parent))

    assert _uv_runtime.resolve_uv() == find_uv_bin() != str(on_path)


def test_a_conda_uv_is_the_binary_in_the_environment(tmp_path, monkeypatch, no_override):
    """conda-forge's uv package has no Python module: its binary sits in the env's scripts dir."""
    monkeypatch.setitem(sys.modules, "uv", None)  # import uv raises ImportError
    scripts = _fake_uv(tmp_path / "env" / "bin").parent
    monkeypatch.setattr(_uv_runtime.sysconfig, "get_path", lambda name: str(scripts))
    monkeypatch.setenv("PATH", str(_fake_uv(tmp_path / "bin").parent))

    assert _uv_runtime.resolve_uv() == str(scripts / _UV)


def test_a_windows_conda_uv_is_in_library_bin(tmp_path, monkeypatch, no_override):
    monkeypatch.setitem(sys.modules, "uv", None)
    monkeypatch.setattr(_uv_runtime.sysconfig, "get_path", lambda name: str(tmp_path / "Scripts"))
    monkeypatch.setattr(_uv_runtime.sys, "prefix", str(tmp_path))
    installed = _fake_uv(tmp_path / "Library" / "bin")

    assert _uv_runtime._bundled_uv() == str(installed)


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


def test_without_a_bundled_uv_path_is_used(tmp_path, monkeypatch, no_override, no_bundled_uv):
    on_path = _fake_uv(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(on_path.parent))

    assert _uv_runtime.resolve_uv() == str(on_path)


def test_a_non_login_shell_finds_uv_in_its_installer_dir(
    tmp_path, monkeypatch, no_override, no_bundled_uv
):
    """ssh and cron often lack ~/.local/bin on PATH."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _home(monkeypatch, tmp_path)
    installed = _fake_uv(tmp_path / ".local" / "bin")

    assert _uv_runtime.resolve_uv() == str(installed)


def test_no_uv_anywhere_is_none(tmp_path, monkeypatch, no_override, no_bundled_uv):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    _home(monkeypatch, tmp_path)

    assert _uv_runtime.resolve_uv() is None


# --- assert_uv_available ---


def test_any_environment_with_a_uv_may_run_strata(tmp_path, monkeypatch, no_override):
    """A venv uv did not make (pip, pipx) or no venv at all (conda): uv is what counts."""
    (tmp_path / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.13.3\n")
    monkeypatch.setattr("sys.prefix", str(tmp_path))

    _uv_runtime.assert_uv_available()


def test_no_uv_is_refused_with_the_way_out(tmp_path, monkeypatch, capsys, no_override):
    monkeypatch.setattr(_uv_runtime, "resolve_uv", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        _uv_runtime.assert_uv_available()

    assert exc_info.value.code == 1
    err = capsys.readouterr().err
    assert "Strata needs uv" in err
    assert "Reinstall strata-notebook" in err
    assert "STRATA_UV" in err


def test_a_strata_uv_naming_no_file_is_refused_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STRATA_UV", str(tmp_path / "uvv"))

    with pytest.raises(SystemExit):
        _uv_runtime.assert_uv_available()

    assert f"STRATA_UV is {tmp_path / 'uvv'}, which is not a file" in capsys.readouterr().err

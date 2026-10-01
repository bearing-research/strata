"""The uv-managed-runtime startup guard."""

from __future__ import annotations

from pathlib import Path

import pytest

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

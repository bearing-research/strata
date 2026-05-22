"""Direct tests for the notebook-extra precheck in conftest.py."""

from __future__ import annotations

import builtins

import pytest

from tests.notebook.conftest import _check_notebook_extra


def _make_import_blocker(*blocked: str):
    real = builtins.__import__

    def fake(name, *args, **kwargs):
        if name in blocked:
            raise ImportError(f"simulated missing {name}")
        return real(name, *args, **kwargs)

    return fake


def test_check_notebook_extra_passes_when_all_present():
    # No monkeypatch: real env has the [notebook] extra, so this is a no-op.
    _check_notebook_extra()


def test_check_notebook_extra_exits_when_orjson_missing(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("orjson"))
    with pytest.raises(pytest.exit.Exception) as excinfo:
        _check_notebook_extra()
    assert "orjson" in str(excinfo.value)
    assert "uv sync --all-extras" in str(excinfo.value)


def test_check_notebook_extra_exits_when_cloudpickle_missing(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("cloudpickle"))
    with pytest.raises(pytest.exit.Exception) as excinfo:
        _check_notebook_extra()
    assert "cloudpickle" in str(excinfo.value)
    assert "uv sync --all-extras" in str(excinfo.value)


def test_check_notebook_extra_reports_both_when_both_missing(monkeypatch):
    monkeypatch.setattr(builtins, "__import__", _make_import_blocker("orjson", "cloudpickle"))
    with pytest.raises(pytest.exit.Exception) as excinfo:
        _check_notebook_extra()
    msg = str(excinfo.value)
    assert "orjson" in msg
    assert "cloudpickle" in msg

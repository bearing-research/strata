"""The ``strata-notebook --notebook-dir`` flag: where new notebooks are created."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import strata.server as server
from strata.config import StrataConfig

_ENV = "STRATA_NOTEBOOK_STORAGE_DIR"


@pytest.fixture(autouse=True)
def _isolate_storage_env():
    """Save and clear the storage env; the flag sets os.environ for the lifespan config."""
    saved = os.environ.pop(_ENV, None)
    yield
    os.environ.pop(_ENV, None)
    if saved is not None:
        os.environ[_ENV] = saved


def test_no_flag_does_not_set_env():
    args = server._build_server_arg_parser().parse_args([])
    server._apply_server_cli_overrides(args)
    assert _ENV not in os.environ


def test_flag_threads_into_loaded_config(tmp_path):
    target = tmp_path / "nbs"
    args = server._build_server_arg_parser().parse_args(["--notebook-dir", str(target)])
    server._apply_server_cli_overrides(args)
    # env beats pyproject, so the loaded config reflects the flag whatever the repo's
    # [tool.strata] says.
    config = StrataConfig.load(cache_dir=tmp_path / "cache")
    assert config.notebook_storage_dir == target.resolve()


def test_dot_means_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = server._build_server_arg_parser().parse_args(["--notebook-dir", "."])
    server._apply_server_cli_overrides(args)
    config = StrataConfig.load(cache_dir=tmp_path / "cache")
    assert config.notebook_storage_dir == Path(tmp_path).resolve()

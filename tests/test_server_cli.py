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


@pytest.mark.parametrize(
    ("argv0", "prog"),
    [
        ("/venv/lib/python3.14/site-packages/strata/__main__.py", "python -m strata"),
        ("/venv/bin/strata-notebook", "strata-notebook"),
    ],
    ids=["module", "script"],
)
def test_usage_names_the_server_as_invoked(monkeypatch, argv0, prog):
    monkeypatch.setattr("sys.argv", [argv0])
    assert server._build_server_arg_parser().prog == prog


def test_version_prints_the_package_version(capsys):
    from importlib.metadata import version

    with pytest.raises(SystemExit) as exited:
        server._build_server_arg_parser().parse_args(["--version"])

    assert exited.value.code == 0
    assert capsys.readouterr().out == f"strata-notebook {version('strata-notebook')}\n"


def test_without_package_metadata_only_version_says_so(monkeypatch, capsys):
    from importlib.metadata import PackageNotFoundError

    def _missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("strata.cli_version.version", _missing)
    parser = server._build_server_arg_parser()
    assert parser.parse_args([]).notebook_dir is None

    with pytest.raises(SystemExit):
        parser.parse_args(["--version"])

    assert capsys.readouterr().out == "strata-notebook unknown (not installed)\n"


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

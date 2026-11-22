"""What a cell subprocess is allowed to see of the server's environment.

A cell is arbitrary Python spawned with the server's environment. On a shared server that
exposes the remote-store headers, proxy token, worker tokens and data-source credentials
to anyone who can run a cell.
"""

from __future__ import annotations

import os

import pytest

from strata.notebook.harness_env import harness_env, in_environment


@pytest.fixture(autouse=True)
def a_server_with_secrets(monkeypatch):
    monkeypatch.setenv("STRATA_PROXY_TOKEN", "shhh")
    monkeypatch.setenv("STRATA_NOTEBOOK_REMOTE_STORE_HEADERS", '{"X-Strata-Principal": "svc"}')
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("SOME_OTHER_THING", "1")


class TestUnset:
    def test_nothing_is_filtered_by_default(self):
        """Unset, as in every personal deployment: the cell is your own code on your own machine."""
        env = harness_env([])

        assert env["STRATA_PROXY_TOKEN"] == "shhh"
        assert env["AWS_ACCESS_KEY_ID"] == "AKIA"

    def test_none_is_the_same_as_empty(self):
        assert harness_env(None)["STRATA_PROXY_TOKEN"] == "shhh"


class TestSet:
    def test_the_servers_secrets_are_gone(self):
        env = harness_env(["AWS_*"])

        assert "STRATA_PROXY_TOKEN" not in env
        assert "STRATA_NOTEBOOK_REMOTE_STORE_HEADERS" not in env

    def test_what_was_asked_for_is_there(self):
        env = harness_env(["AWS_*", "HF_TOKEN"])

        assert env["AWS_ACCESS_KEY_ID"] == "AKIA"
        assert env["AWS_SECRET_ACCESS_KEY"] == "secret"
        assert env["HF_TOKEN"] == "hf_x"

    def test_what_was_not_asked_for_is_not(self):
        env = harness_env(["AWS_*"])

        assert "HF_TOKEN" not in env
        assert "SOME_OTHER_THING" not in env

    def test_the_essentials_survive_without_being_listed(self, monkeypatch):
        """A subprocess with no PATH does not start, so an allowlist that had
        to name it would be a setting nobody could turn on correctly."""
        monkeypatch.setenv("PATH", "/somewhere/bin")

        env = harness_env(["HF_TOKEN"])

        assert env["PATH"] == "/somewhere/bin"

    def test_a_prefix_rule_cannot_reach_the_secrets(self, monkeypatch):
        """A rule broad enough to catch a credential by accident is the failure
        this setting exists to prevent."""
        env = harness_env(["STRATA_*", "*"])

        assert "STRATA_PROXY_TOKEN" not in env

    def test_naming_one_exactly_hands_it_over(self, monkeypatch):
        """An operator who writes the whole name means it; some deployments pass a STRATA_ setting
        a cell legitimately reads.
        """
        monkeypatch.setenv("STRATA_NOTEBOOK_OBJECT_CODEC", "pickle")

        env = harness_env(["STRATA_NOTEBOOK_OBJECT_CODEC"])

        assert env["STRATA_NOTEBOOK_OBJECT_CODEC"] == "pickle"
        assert "STRATA_PROXY_TOKEN" not in env


class TestTheFloorCarriesNoCredentials:
    """UV_*, PYTHON* and R_* pass without being listed, so a credential under one of
    them would reach every cell: a private index's login is for the server's uv sync."""

    @pytest.fixture(autouse=True)
    def index_credentials(self, monkeypatch):
        monkeypatch.setenv("UV_INDEX_CORP_PASSWORD", "s3cret")
        monkeypatch.setenv("UV_INDEX_CORP_USERNAME", "svc")
        monkeypatch.setenv("UV_PUBLISH_TOKEN", "pypi-tok")
        monkeypatch.setenv("UV_PUBLISH_URL", "https://upload.example")
        # uv also takes an index login embedded in the index URL.
        for name in (
            "UV_INDEX",
            "UV_INDEX_URL",
            "UV_EXTRA_INDEX_URL",
            "UV_DEFAULT_INDEX",
            "UV_INDEX_CORP_URL",
        ):
            monkeypatch.setenv(name, "https://ci:s3cret@pypi.internal/simple")
        monkeypatch.setenv("R_LIBS_SECRET", "x")
        monkeypatch.setenv("PYTHON_KEYRING_TOKEN", "y")
        monkeypatch.setenv("CONDA_PREFIX", "/opt/conda")

    def test_credentials_under_the_essential_prefixes_are_dropped(self):
        env = harness_env(["HF_TOKEN"])

        for name in (
            "UV_INDEX_CORP_PASSWORD",
            "UV_INDEX_CORP_USERNAME",
            "UV_PUBLISH_TOKEN",
            "UV_PUBLISH_URL",
            "UV_INDEX",
            "UV_INDEX_URL",
            "UV_EXTRA_INDEX_URL",
            "UV_DEFAULT_INDEX",
            "UV_INDEX_CORP_URL",
            "R_LIBS_SECRET",
            "PYTHON_KEYRING_TOKEN",
        ):
            assert name not in env, name

    def test_what_uv_python_and_r_need_still_passes(self, monkeypatch):
        monkeypatch.setenv("UV_CACHE_DIR", "/cache/uv")
        monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/opt/uv-python")
        monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
        monkeypatch.setenv("R_LIBS_USER", "/r/lib")
        monkeypatch.setenv("LC_ALL", "C.UTF-8")

        env = harness_env(["HF_TOKEN"])

        assert env["UV_CACHE_DIR"] == "/cache/uv"
        assert env["UV_PYTHON_INSTALL_DIR"] == "/opt/uv-python"
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["R_LIBS_USER"] == "/r/lib"
        assert env["LC_ALL"] == "C.UTF-8"
        assert env["HF_TOKEN"] == "hf_x"

    def test_conda_activation_is_not_a_cell_need(self):
        """Notebook environments are uv venvs; nothing in a cell's start-up reads CONDA_*."""
        assert "CONDA_PREFIX" not in harness_env(["HF_TOKEN"])

    def test_naming_a_credential_exactly_still_hands_it_over(self):
        assert harness_env(["UV_INDEX_CORP_PASSWORD"])["UV_INDEX_CORP_PASSWORD"] == "s3cret"
        assert "UV_INDEX_URL" in harness_env(["UV_INDEX_URL"])


class TestExtra:
    def test_extra_is_set_after_filtering(self):
        """The batch harness is told which file descriptors to use through the
        environment. That is this code talking to itself, not something an
        allowlist should have to know about."""
        env = harness_env(["HF_TOKEN"], {"STRATA_BATCH_FRAME_FD": "7"})

        assert env["STRATA_BATCH_FRAME_FD"] == "7"
        assert "STRATA_PROXY_TOKEN" not in env

    def test_extra_reaches_an_unfiltered_environment_too(self):
        env = harness_env([], {"STRATA_BATCH_FRAME_FD": "7"})

        assert env["STRATA_BATCH_FRAME_FD"] == "7"
        assert env["STRATA_PROXY_TOKEN"] == "shhh"


class TestConfig:
    def test_the_setting_parses_the_way_the_other_lists_do(self):
        from strata.config import StrataConfig

        config = StrataConfig(notebook_harness_env_allowlist="AWS_*, HF_TOKEN")

        assert config.notebook_harness_env_allowlist == ["AWS_*", "HF_TOKEN"]

    def test_the_default_hands_over_everything(self):
        from strata.config import StrataConfig

        assert StrataConfig().notebook_harness_env_allowlist == []


class TestInItsEnvironment:
    """A cell's ``python``, ``uv pip`` and console scripts are the notebook's."""

    @staticmethod
    def _venv(tmp_path):
        (tmp_path / "env" / "bin").mkdir(parents=True)
        (tmp_path / "env" / "pyvenv.cfg").write_text("home = /usr/bin\n")
        return tmp_path / "env" / "bin" / "python"

    def test_the_environment_is_named_and_its_scripts_come_first(self, tmp_path):
        env = in_environment({"PATH": "/usr/bin", "KEPT": "1"}, self._venv(tmp_path))

        assert env == {
            "PATH": f"{tmp_path / 'env' / 'bin'}{os.pathsep}/usr/bin",
            "VIRTUAL_ENV": str(tmp_path / "env"),
            "KEPT": "1",
        }

    def test_the_servers_own_environment_does_not_reach_the_cell(self, tmp_path):
        """A server started by ``uv run`` has a ``VIRTUAL_ENV`` of its own."""
        server = {
            "PATH": "/usr/bin",
            "VIRTUAL_ENV": "/server/venv",
            "UV_PROJECT_ENVIRONMENT": "/server/env",
            "UV_PYTHON": "3.99",
        }

        env = in_environment(server, self._venv(tmp_path))

        assert env["VIRTUAL_ENV"] == str(tmp_path / "env")
        assert "UV_PROJECT_ENVIRONMENT" not in env and "UV_PYTHON" not in env
        assert server["VIRTUAL_ENV"] == "/server/venv", "the caller's mapping was changed"

    def test_an_inherited_environment_is_the_servers_with_the_environment_active(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("PATH", "/usr/bin")

        env = in_environment(None, self._venv(tmp_path))

        assert env["PATH"] == f"{tmp_path / 'env' / 'bin'}{os.pathsep}/usr/bin"
        assert env["HF_TOKEN"] == "hf_x"

    def test_an_interpreter_in_no_environment_changes_nothing(self):
        assert in_environment(None, "python") is None
        assert in_environment({"PATH": "/usr/bin"}, "/usr/bin/python3") == {"PATH": "/usr/bin"}


class TestEverySpawnApplied:
    """The filter is worth what the spawn sites apply it to.

    The cold Python harness is covered in ``tests/test_harness_isolation.py``, through a
    real spawn; the other spawns are checked by capturing what they hand to the OS.
    """

    def _capture(self, monkeypatch, module):
        seen = {}

        async def _fake(*_args, **kwargs):
            seen["env"] = kwargs.get("env")
            raise RuntimeError("spawn intercepted")

        monkeypatch.setattr(module.asyncio, "create_subprocess_exec", _fake)
        return seen

    def test_the_warm_pool_worker_is_filtered(self, tmp_path, monkeypatch):
        """The default WebSocket path. Filtering the cold spawn and not this one
        would leave the secrets readable from almost every cell anyone runs."""
        import asyncio as _asyncio

        from strata.notebook import pool as pool_module

        monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", "HF_TOKEN")
        seen = self._capture(monkeypatch, pool_module)

        warm = pool_module.WarmProcessPool(tmp_path, pool_size=1)
        _asyncio.run(warm._spawn_warm_process())

        assert seen["env"] is not None
        assert "STRATA_PROXY_TOKEN" not in seen["env"]
        assert seen["env"]["HF_TOKEN"] == "hf_x"

    def test_the_warm_pool_inherits_when_unset(self, tmp_path, monkeypatch):
        """An unconfigured deployment hands over the server's whole environment, plus
        only the lifeline fd that lets the worker die with the server."""
        import asyncio as _asyncio

        from strata.notebook import pool as pool_module
        from strata.notebook.lifeline import LIFELINE_FD_ENV

        seen = self._capture(monkeypatch, pool_module)

        warm = pool_module.WarmProcessPool(tmp_path, pool_size=1)
        _asyncio.run(warm._spawn_warm_process())

        inherited = {k: v for k, v in seen["env"].items() if k != LIFELINE_FD_ENV}
        assert inherited == dict(os.environ)

    def test_the_warm_pool_worker_starts_in_the_notebooks_environment(self, tmp_path, monkeypatch):
        import asyncio as _asyncio

        from strata.notebook import pool as pool_module

        monkeypatch.setenv("VIRTUAL_ENV", "/server/venv")
        seen = self._capture(monkeypatch, pool_module)
        (tmp_path / "env" / "bin").mkdir(parents=True)
        (tmp_path / "env" / "pyvenv.cfg").write_text("home = /usr/bin\n")

        warm = pool_module.WarmProcessPool(
            tmp_path, pool_size=1, python_executable=tmp_path / "env" / "bin" / "python"
        )
        _asyncio.run(warm._spawn_warm_process())

        assert seen["env"]["VIRTUAL_ENV"] == str(tmp_path / "env")

    def test_the_r_harness_is_filtered(self, monkeypatch):
        """R cells are cell code like any other."""
        from types import SimpleNamespace

        from strata.config import StrataConfig
        from strata.notebook.executor import CellExecutor

        monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", "HF_TOKEN")
        config = StrataConfig.load()
        stub = SimpleNamespace(_lake_config=lambda: config)

        env = CellExecutor._harness_env(stub)

        assert "STRATA_PROXY_TOKEN" not in env
        assert env["HF_TOKEN"] == "hf_x"

    def test_the_batch_harness_keeps_its_own_descriptors(self, monkeypatch):
        """The batch harness is told which file descriptors to use through the
        environment. Those are this code talking to itself and must survive a
        filter that drops everything else STRATA_."""
        from types import SimpleNamespace

        from strata.config import StrataConfig
        from strata.notebook.executor import CellExecutor

        monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", "HF_TOKEN")
        config = StrataConfig.load()
        stub = SimpleNamespace(_lake_config=lambda: config)

        env = CellExecutor._harness_env(stub, {"STRATA_BATCH_FRAME_FD": "7"})

        assert env["STRATA_BATCH_FRAME_FD"] == "7"
        assert "STRATA_PROXY_TOKEN" not in env

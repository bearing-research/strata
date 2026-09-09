"""What a cell subprocess is allowed to see of the server's environment.

A cell is arbitrary Python spawned with the server's whole environment. On a
laptop that is right and there is nothing to protect; on a shared server it
means every member who can run a cell can read the remote-store headers, the
proxy token, worker tokens and every data-source credential the server holds.
Item 49.
"""

from __future__ import annotations

import pytest

from strata.notebook.harness_env import harness_env


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
        """Every deployment before this setting existed, and every personal one
        after it: the cell is your own code on your own machine."""
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
        """An operator who writes the whole name means it — some deployments
        do pass a STRATA_ setting a cell legitimately reads."""
        monkeypatch.setenv("STRATA_NOTEBOOK_OBJECT_CODEC", "pickle")

        env = harness_env(["STRATA_NOTEBOOK_OBJECT_CODEC"])

        assert env["STRATA_NOTEBOOK_OBJECT_CODEC"] == "pickle"
        assert "STRATA_PROXY_TOKEN" not in env


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


class TestEverySpawnApplied:
    """The filter is worth what the spawn sites apply it to.

    The cold Python harness is covered end to end in
    ``tests/test_harness_isolation.py``, outside this package, because this
    conftest replaces ``_run_harness``. The other three run cell code too and
    are covered here by capturing what they hand to the OS.
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
        """``env=None`` rather than a copy, so an unconfigured deployment gets
        the spawn it always got."""
        import asyncio as _asyncio

        from strata.notebook import pool as pool_module

        seen = self._capture(monkeypatch, pool_module)

        warm = pool_module.WarmProcessPool(tmp_path, pool_size=1)
        _asyncio.run(warm._spawn_warm_process())

        assert seen["env"] is None

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

"""Tests for the runtime worker-token registry + executor token resolution.

An SSH-tunneled worker's token is generated at provisioning time and kept in the
server process, never in notebook.toml. These cover the registry and that the
executor's ``_resolve_worker_token`` finds it by name, without disturbing the
existing token_env / literal precedence.
"""

from __future__ import annotations

import pytest

from strata.notebook.executor import _resolve_worker_token
from strata.notebook.models import WorkerBackendType, WorkerConfig, WorkerSpec
from strata.notebook.worker_secrets import (
    clear_runtime_worker_token,
    get_runtime_worker_token,
    set_runtime_worker_token,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    yield
    for name in ("gpu", "cpu"):
        clear_runtime_worker_token(name)


def _spec(*, token_env: str | None = None, token: str | None = None) -> WorkerSpec:
    return WorkerSpec(
        name="gpu",
        backend=WorkerBackendType.EXECUTOR,
        config=WorkerConfig(
            url="http://127.0.0.1:9000/v1/execute", token_env=token_env, token=token
        ),
    )


def test_registry_round_trip():
    assert get_runtime_worker_token("gpu") is None
    set_runtime_worker_token("gpu", "secret")
    assert get_runtime_worker_token("gpu") == "secret"
    clear_runtime_worker_token("gpu")
    assert get_runtime_worker_token("gpu") is None
    clear_runtime_worker_token("gpu")  # idempotent


def test_resolve_falls_back_to_runtime_token():
    spec = _spec()
    assert _resolve_worker_token(spec) is None
    set_runtime_worker_token("gpu", "rt-secret")
    assert _resolve_worker_token(spec) == "rt-secret"


def test_token_env_wins_over_runtime(monkeypatch):
    monkeypatch.setenv("MY_WORKER_TOKEN", "env-secret")
    set_runtime_worker_token("gpu", "rt-secret")
    assert _resolve_worker_token(_spec(token_env="MY_WORKER_TOKEN")) == "env-secret"


def test_literal_token_wins_over_runtime():
    set_runtime_worker_token("gpu", "rt-secret")
    assert _resolve_worker_token(_spec(token="literal")) == "literal"


def test_no_token_anywhere_is_none():
    assert _resolve_worker_token(_spec()) is None

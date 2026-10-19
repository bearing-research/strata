"""A cell on a server that authenticates callers calls that server with its run's token.

The server here is a live service-mode server under ``trusted_proxy``; the cells run as this
user, since a harness user needs root (``tests/test_harness_isolation.py`` covers that).
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from strata.auth.principal import principal_context
from strata.auth.run_tokens import RUN_TOKEN_HEADER, grant_for
from strata.config import StrataConfig
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.pool import PooledCellExecutor
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from strata.types import Principal
from tests.conftest import LIVE_SERVER_TIMEOUT, find_free_port, run_server
from tests.notebook.test_executor_batch import _cell_spec

ANA = Principal(id="ana", scopes=frozenset({"notebook:write"}))


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A trusted-proxy server whose cells run here, as if under a harness user."""
    config = StrataConfig(
        host="127.0.0.1",
        port=find_free_port(),
        cache_dir=tmp_path / "cache",
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="proxy-secret",
        rate_limit_enabled=False,
    )
    monkeypatch.setattr("strata.notebook.executor.resolve_harness_user", lambda: None)
    monkeypatch.setattr("strata.notebook.pool.resolve_harness_user", lambda: None)
    with run_server(config) as base:
        yield base


def _session(tmp_path: Path, source: str) -> NotebookSession:
    nb = create_notebook(tmp_path, "Tokens")
    add_cell_to_notebook(nb, "c", None)
    write_cell(nb, "c", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _scan_source(table_uri: str) -> str:
    return (
        f"# @table events {table_uri}\n"
        "scan = strata.materialize(inputs=[events], transform={'executor': 'scan@v1'})\n"
        "print('ROWS', scan.to_arrow().num_rows)\n"
        "del scan\n"
    )


def _run(executor: CellExecutor, source: str, principal: Principal | None = ANA):
    async def run():
        if principal is None:
            return await executor.execute_cell("c", source)
        with principal_context(principal):
            return await executor.execute_cell("c", source)

    return asyncio.run(run())


def test_a_cell_scans_a_table_through_its_server(server, tmp_path, temp_warehouse):
    source = _scan_source(temp_warehouse["table_uri"])

    result = _run(CellExecutor(_session(tmp_path, source)), source)

    assert result.success, result.error
    assert "ROWS 500" in result.stdout


def test_a_run_nobody_started_reaches_nothing(server, tmp_path, temp_warehouse):
    source = _scan_source(temp_warehouse["table_uri"])

    result = _run(CellExecutor(_session(tmp_path, source)), source, principal=None)

    assert not result.success
    assert "401" in (result.error or "")


def test_the_token_reaches_the_cell_only_over_its_stdin(server, tmp_path, monkeypatch):
    """Not in the manifest, argv or environment, which other cells as the same user could read."""
    source = f"print('TOKEN', strata._headers['{RUN_TOKEN_HEADER}'])\n"
    session = _session(tmp_path, source)
    seen: list[str] = []
    spawn = asyncio.create_subprocess_exec

    async def spying_spawn(*args, **kwargs):
        seen.append(json.dumps([args, kwargs.get("env")], default=str))
        return await spawn(*args, **kwargs)

    write_manifest = CellExecutor._write_manifest

    def reading_manifest(self, *args, **kwargs):
        path = write_manifest(self, *args, **kwargs)
        seen.append(path.read_text())
        return path

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spying_spawn)
    monkeypatch.setattr(CellExecutor, "_write_manifest", reading_manifest)

    result = _run(CellExecutor(session), source)

    assert result.success, result.error
    token = result.stdout.split("TOKEN ", 1)[1].split()[0]
    assert len(seen) == 2
    assert all(token not in text for text in seen)


def test_a_copied_token_is_refused_once_its_run_ends(server, tmp_path, temp_warehouse):
    source = f"print('TOKEN', strata._headers['{RUN_TOKEN_HEADER}'])\n"

    result = _run(CellExecutor(_session(tmp_path, source)), source)

    token = result.stdout.split("TOKEN ", 1)[1].split()[0]
    response = httpx.post(
        f"{server}/v1/materialize",
        json={
            "inputs": [temp_warehouse["table_uri"]],
            "transform": {"executor": "scan@v1", "params": {}},
            "mode": "stream",
        },
        headers={RUN_TOKEN_HEADER: token},
        timeout=LIVE_SERVER_TIMEOUT,
    )
    assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.warm_pool
async def test_a_warm_cell_scans_through_its_server(server, tmp_path, temp_warehouse, warm_pool):
    source = _scan_source(temp_warehouse["table_uri"])
    session = _session(tmp_path, source)
    executor = CellExecutor(session, await warm_pool(session))

    with principal_context(ANA):
        result = await executor.execute_cell("c", source)

    assert result.success, result.error
    assert result.execution_method == "warm"
    assert "ROWS 500" in result.stdout


def test_a_batched_cell_scans_through_its_server(server, tmp_path, temp_warehouse):
    source = (
        "scan = strata.materialize(\n"
        f"    inputs=[{temp_warehouse['table_uri']!r}], transform={{'executor': 'scan@v1'}}\n"
        ")\n"
        "print('ROWS', scan.to_arrow().num_rows)\n"
        "del scan\n"
    )
    session = _session(tmp_path, source)
    spec = {**_cell_spec("c", source), "strata_url": f"{server}"}

    async def run():
        with principal_context(ANA):
            return await CellExecutor(session).execute_batch([spec])

    batch = asyncio.run(run())

    assert batch.completed, batch.cell_results[0].error
    assert "ROWS 500" in (batch.cell_results[0].stdout or "")


class TestATokenLivesAsLongAsItsHarness:
    """A token's clock starts where its harness's timeout does: when the harness starts."""

    @staticmethod
    def _clock(monkeypatch) -> list[float]:
        """Run tokens' clock only: the time module's would freeze the event loop."""
        from strata.auth import run_tokens

        now = [1000.0]
        monkeypatch.setattr(run_tokens, "time", SimpleNamespace(monotonic=lambda: now[0]))
        return now

    async def test_a_cold_rerun_after_a_warm_process_died_has_a_live_token(
        self, tmp_path, monkeypatch
    ):
        now = self._clock(monkeypatch)
        session = _session(tmp_path, "x = 1\n")
        live: list[bool] = []

        async def died_late(*args, **kwargs):
            now[0] += 30.0 + 61.0  # it ran past the timeout and the margin, then died
            return None

        async def cold(self, manifest_path, venv_python, timeout_seconds, *, run_token=None):
            live.append(run_token is not None and grant_for(run_token) is not None)
            return {"success": True, "variables": {}}

        monkeypatch.setattr(PooledCellExecutor, "execute_with_pool", died_late)
        monkeypatch.setattr(CellExecutor, "_run_harness", cold)
        executor = CellExecutor(session, pool=cast(Any, object()))
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        with principal_context(ANA):
            await executor._dispatch_local(
                "x = 1\n", {}, [], output_dir, Path(sys.executable), {}, 30.0
            )

        assert live == [True]

    async def test_a_run_all_token_is_minted_after_its_inputs_are_staged(
        self, tmp_path, monkeypatch
    ):
        now = self._clock(monkeypatch)
        session = _session(tmp_path, "x = 1\n")
        staging = CellExecutor._batch_resolve_upstream_inputs

        def slow_staging(self, *args, **kwargs):
            now[0] += 600.0 + 61.0  # staging large inputs took a batch timeout and more
            return staging(self, *args, **kwargs)

        minted: list[str | None] = []
        mint = CellExecutor._run_token

        @contextmanager
        def recording(self, timeout_seconds):
            with mint(self, timeout_seconds) as token:
                minted.append(token)
                yield token

        live: list[bool] = []
        spawn = asyncio.create_subprocess_exec

        async def spawning(*args, **kwargs):
            live.append(all(t is not None and grant_for(t) is not None for t in minted))
            return await spawn(*args, **kwargs)

        monkeypatch.setattr(CellExecutor, "_batch_resolve_upstream_inputs", slow_staging)
        monkeypatch.setattr(CellExecutor, "_run_token", recording)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawning)

        with principal_context(ANA):
            batch = await CellExecutor(session).execute_batch([_cell_spec("c", "x = 1\n")])

        assert batch.completed, batch.cell_results[0].error
        assert len(minted) == 1
        assert live == [True]

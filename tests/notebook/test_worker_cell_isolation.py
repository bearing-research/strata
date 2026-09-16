"""What a cell on a worker can see of the worker, and who the server polls.

A worker holds the bearer token that authorizes running code on it and the
credentials it resolves mount and connection names against. A cell is arbitrary
code somebody else wrote; it gets what its manifest carries.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import httpx
import pytest

from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
    create_notebook_executor_app,
)

_SOURCE = "import json, os\nseen = json.dumps(dict(os.environ))\nprint(seen)\n"


async def _run_cell(source: str = _SOURCE) -> dict:
    """Run a cell on an in-process worker; return the harness result."""
    metadata = {
        "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": source,
        "inputs": {},
        "mounts": [],
        "env": {},
        "timeout_seconds": 120,
    }
    transport = httpx.ASGITransport(app=create_notebook_executor_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.post(
            "/v1/notebook-execute",
            files={
                "metadata": ("metadata.json", json.dumps(metadata).encode(), "application/json")
            },
            headers={"Authorization": "Bearer worker-bearer-token"},
            timeout=120,
        )
    assert response.status_code == 200, response.text
    return response.content


def _stdout(bundle: bytes, tmp_path: Path) -> str:
    from strata.notebook.remote_bundle import unpack_notebook_output_bundle

    path = tmp_path / "bundle.tar"
    path.write_bytes(bundle)
    with tarfile.open(path):  # a bundle, not something else
        pass
    return unpack_notebook_output_bundle(path, tmp_path / "out").get("stdout", "")


@pytest.fixture
def worker_secrets(monkeypatch):
    monkeypatch.setenv("STRATA_WORKER_TOKEN", "worker-bearer-token")
    monkeypatch.setenv(
        "STRATA_NOTEBOOK_CREDENTIALS", '{"lab": {"key": "AKIA", "secret": "SUPER-SECRET"}}'
    )
    monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CREDENTIALS", '{"s3": "lab"}')
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.delenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", raising=False)


@pytest.mark.asyncio
async def test_a_cell_cannot_read_the_workers_token_or_credentials(tmp_path, worker_secrets):
    seen = json.loads(_stdout(await _run_cell(), tmp_path))

    assert "STRATA_WORKER_TOKEN" not in seen
    assert "STRATA_NOTEBOOK_CREDENTIALS" not in seen
    assert "STRATA_NOTEBOOK_MOUNT_CREDENTIALS" not in seen
    assert "SUPER-SECRET" not in json.dumps(seen)


@pytest.mark.asyncio
async def test_an_allowlist_narrows_the_rest_of_the_workers_environment(
    tmp_path, monkeypatch, worker_secrets
):
    monkeypatch.setenv("MY_TOOL_HOME", "/opt/tool")
    monkeypatch.setenv("SOMEONE_ELSES", "private")
    monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", "MY_TOOL_HOME")

    seen = json.loads(_stdout(await _run_cell(), tmp_path))

    assert seen.get("MY_TOOL_HOME") == "/opt/tool"
    assert "SOMEONE_ELSES" not in seen
    assert "AWS_SECRET_ACCESS_KEY" not in seen
    assert "PATH" in seen, "a cell still needs the names without which nothing runs"


class TestWhereTheServerPolls:
    """A 202 hands back a job URL. It belongs to the worker the manifest went
    to: anywhere else and a worker could have the server poll a host of its
    choosing, carrying the worker's token."""

    @staticmethod
    def _executor(monkeypatch, tmp_path, job_url: str | None):
        from types import SimpleNamespace

        from strata.notebook.executor import CellExecutor
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import create_notebook

        nb = create_notebook(tmp_path, "poller", initialize_environment=False)
        session = NotebookSession(parse_notebook(nb), nb)
        accepted = httpx.Response(202, json={"job_url": job_url} if job_url else {})
        worker = SimpleNamespace(name="w", config=SimpleNamespace(url="http://worker/v1/execute"))
        return CellExecutor(session), accepted, worker

    @pytest.mark.asyncio
    async def test_a_job_url_on_another_host_is_refused(self, tmp_path, monkeypatch):
        from strata.notebook.executor import RemoteExecutionError

        executor, accepted, worker = self._executor(
            monkeypatch, tmp_path, "http://169.254.169.254/latest/meta-data/"
        )
        cancelled: list[bool] = []

        with pytest.raises(RemoteExecutionError) as raised:
            await executor._await_accepted_job(
                accepted,
                manifest_execute_url="http://worker/v1/execute-manifest",
                headers={"Authorization": "Bearer worker-token"},
                worker_spec=worker,
                timeout_seconds=30.0,
                cancel=_recording(cancelled),
                cell_id=None,
            )

        assert raised.value.remote_error_code == "PROTOCOL_ERROR"
        assert "another host" in str(raised.value)

    @pytest.mark.asyncio
    async def test_a_status_request_that_fails_cancels_the_job(self, tmp_path, monkeypatch):
        from strata.notebook.executor import RemoteExecutionError

        executor, accepted, worker = self._executor(monkeypatch, tmp_path, "/v1/jobs/j1")
        cancelled: list[bool] = []

        async def _cancel():
            cancelled.append(True)

        async def _fail(self, *args, **kwargs):
            raise httpx.ConnectError("connection reset")

        monkeypatch.setattr(httpx.AsyncClient, "get", _fail)

        with pytest.raises(RemoteExecutionError) as raised:
            await executor._await_accepted_job(
                accepted,
                manifest_execute_url="http://worker/v1/execute-manifest",
                headers={},
                worker_spec=worker,
                timeout_seconds=30.0,
                cancel=_cancel,
                cell_id=None,
            )

        assert raised.value.remote_error_code == "JOB_STATUS_FAILED"
        assert cancelled == [True], "the machine was left running for a caller that gave up"


def _recording(calls: list[bool]):
    async def _cancel() -> None:
        calls.append(True)

    return _cancel


def test_the_workers_secrets_leave_the_environment_a_cell_can_reach(monkeypatch):
    """Scrubbing the harness's own copy is not a boundary: the harness runs
    under the worker's uid, so /proc/<ppid>/environ has whatever the worker
    still holds. The entry point takes them out of the environment entirely."""
    import os

    from strata.notebook.remote_executor import (
        _CAPTURED_SECRETS,
        capture_worker_secrets,
        worker_secret,
    )

    monkeypatch.setenv("STRATA_WORKER_TOKEN", "worker-bearer-token")
    monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIALS", '{"lab": {"key": "AKIA"}}')
    _CAPTURED_SECRETS.clear()
    try:
        capture_worker_secrets()

        assert "STRATA_WORKER_TOKEN" not in os.environ
        assert "STRATA_NOTEBOOK_CREDENTIALS" not in os.environ
        # The worker still needs them: it authenticates requests and resolves
        # the credential names a manifest carries.
        assert worker_secret("STRATA_WORKER_TOKEN") == "worker-bearer-token"
        assert "AKIA" in worker_secret("STRATA_NOTEBOOK_CREDENTIALS")
    finally:
        _CAPTURED_SECRETS.clear()


def test_an_allowlist_written_as_json_is_read_the_way_the_server_reads_it(monkeypatch):
    """The setting's validator accepts a JSON array; a worker that read only
    the comma form would narrow a cell's environment to nothing."""
    from strata.config import StrataConfig
    from strata.notebook.remote_executor import _cell_env

    monkeypatch.setenv("MY_TOOL_HOME", "/opt/tool")
    monkeypatch.setenv("SOMEONE_ELSES", "private")
    monkeypatch.setenv("STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST", '["MY_TOOL_HOME"]')

    assert StrataConfig.load().notebook_harness_env_allowlist == ["MY_TOOL_HOME"]
    env = _cell_env()
    assert env.get("MY_TOOL_HOME") == "/opt/tool"
    assert "SOMEONE_ELSES" not in env

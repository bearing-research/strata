"""A service-mode notebook's mount options cannot name credentials, key files or endpoints.

The registry is the operator's. A notebook option over it could send its credential to the
notebook author's host (``endpoint_url``), or have the server read any file it can and
exchange it with a token service the author chose (``token``, ``federated_token_file``).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from strata.config import StrataConfig
from strata.notebook.credentials import CredentialError, CredentialResolver
from strata.notebook.models import MountSpec
from strata.notebook.mounts import MountResolver

REGISTRY = {
    "lab-s3": {"key": "lab-key", "secret": "lab-secret"},
    "lab-gs": {"token": "/run/creds/lab-gs/external-account.json"},
    "lab-az": {"account_name": "labdata", "account_key": "lab-account-key"},
}


def _external_account(tmp_path: Path) -> Path:
    """What an author would point ``token`` at: the server reads ``secret`` and posts it."""
    secret = tmp_path / "server-secret"
    secret.write_text("server-only")
    config = tmp_path / "external-account.json"
    config.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/1/x",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://author.example/token",
                "credential_source": {"file": str(secret)},
            }
        )
    )
    return config


def _refused(tmp_path: Path) -> list[tuple[str, str, dict]]:
    return [
        ("s3", "endpoint_url", {"endpoint_url": "https://author.example"}),
        ("gs", "token", {"token": str(_external_account(tmp_path))}),
        ("gs", "endpoint_url", {"endpoint_url": "https://author.example"}),
        (
            "az",
            "federated_token_file",
            {"federated_token_file": "/etc/hosts", "tenant_id": "t", "client_id": "c"},
        ),
        ("az", "account_name", {"account_name": "author"}),
        ("s3", "client_kwargs", {"client_kwargs": {"endpoint_url": "https://author.example"}}),
        ("s3", "profile", {"profile": "org-admin"}),
    ]


@pytest.fixture
def fetched(monkeypatch) -> list[tuple[str, dict]]:
    """fsspec stopped before any network, recording what each mount would be read with."""
    import fsspec

    seen: list[tuple[str, dict]] = []

    def _filesystem(protocol, **options):
        seen.append((protocol, options))
        raise RuntimeError("stop before any network")

    monkeypatch.setattr(fsspec, "filesystem", _filesystem)
    return seen


def _config(tmp_path: Path, mode: str) -> StrataConfig:
    service = {"auth_mode": "trusted_proxy", "proxy_token": "t"} if mode == "service" else {}
    return StrataConfig(
        deployment_mode=mode,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        notebook_credentials=REGISTRY,
        **service,
    )


def _prepare(tmp_path: Path, mode: str, mount: MountSpec) -> None:
    resolver = MountResolver(
        cache_dir=tmp_path / "mount_cache",
        credential_resolver=CredentialResolver.from_config(_config(tmp_path, mode)),
    )
    asyncio.run(resolver.prepare_mounts([mount]))


@pytest.mark.parametrize("named", [True, False], ids=["named-credential", "no-credential"])
@pytest.mark.parametrize("case", range(7))
def test_service_mode_refuses_an_option_naming_a_secret_file_or_endpoint(
    tmp_path, fetched, case, named
):
    scheme, key, options = _refused(tmp_path)[case]
    mount = MountSpec(
        name="raw",
        uri=f"{scheme}://lab/raw",
        credential=f"lab-{scheme}" if named else None,
        options=options,
    )

    with pytest.raises(CredentialError, match=rf"mount option '{key}' cannot be set"):
        _prepare(tmp_path, "service", mount)
    assert fetched == [], "nothing reached fsspec"


def test_personal_mode_still_takes_them(tmp_path, fetched):
    """The author is the server's owner."""
    mount = MountSpec(
        name="raw",
        uri="s3://lab/raw",
        credential="lab-s3",
        options={"endpoint_url": "https://minio.local", "profile": "dev"},
    )

    with pytest.raises(RuntimeError, match="stop before any network"):
        _prepare(tmp_path, "personal", mount)

    assert fetched[-1][1] == {
        "key": "lab-key",
        "secret": "lab-secret",
        "endpoint_url": "https://minio.local",
        "profile": "dev",
    }


@pytest.mark.parametrize(
    ("scheme", "options"),
    [
        ("s3", {"anon": True, "requester_pays": True, "version_aware": True}),
        ("gs", {"token": "anon", "requester_pays": True}),
        ("az", {"anon": True, "blocksize": 1048576}),
    ],
)
def test_service_mode_passes_the_allowed_options_through(tmp_path, fetched, scheme, options):
    mount = MountSpec(name="raw", uri=f"{scheme}://lab/raw", options=options)

    with pytest.raises(RuntimeError, match="stop before any network"):
        _prepare(tmp_path, "service", mount)

    assert fetched[-1][1] == options


class TestThroughACellRun:
    """A notebook.toml mount's options, run on a service server and sent to a worker."""

    @pytest.fixture
    def server(self, tmp_path, monkeypatch, fetched):
        from strata.notebook.executor import CellExecutor
        from strata.notebook.session import NotebookSession

        config = _config(tmp_path, "service")
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
        return fetched

    @staticmethod
    def _session(parent: Path, worker_url: str | None = None):
        from strata.notebook.models import WorkerBackendType, WorkerSpec
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import (
            add_cell_to_notebook,
            create_notebook,
            update_notebook_mounts,
            update_notebook_workers,
            write_cell,
        )

        parent.mkdir(parents=True, exist_ok=True)
        nb = create_notebook(parent, "nb")
        update_notebook_mounts(
            nb,
            [
                MountSpec(
                    name="raw",
                    uri="s3://lab/raw",
                    credential="lab-s3",
                    options={"endpoint_url": "https://author.example"},
                )
            ],
        )
        source = "x = raw"
        if worker_url:
            update_notebook_workers(
                nb,
                [
                    WorkerSpec(
                        name="gpu", backend=WorkerBackendType.EXECUTOR, config={"url": worker_url}
                    )
                ],
            )
            source = f"# @worker gpu\n{source}"
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", source)
        return NotebookSession(parse_notebook(nb), nb), source

    async def test_the_cell_fails_naming_the_option(self, tmp_path, server):
        from strata.notebook.executor import CellExecutor

        session, source = self._session(tmp_path / "nb")
        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert "mount option 'endpoint_url' cannot be set" in (result.error or "")
        assert server == []

    async def test_a_remote_worker_is_never_sent_it(self, tmp_path, server):
        """Nothing listens at the worker's URL: reaching it would fail differently."""
        from strata.notebook.executor import CellExecutor

        session, source = self._session(tmp_path / "nb", worker_url="http://127.0.0.1:9/v1/execute")
        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert "mount option 'endpoint_url' cannot be set" in (result.error or "")


def test_a_worker_refuses_them_whatever_mode_its_server_runs_in(monkeypatch):
    """A worker's credentials are its operator's, and it cannot tell a personal server's
    dispatch from a service one's."""
    from strata.notebook.remote_executor import _worker_credentials

    monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIALS", json.dumps(REGISTRY))

    with pytest.raises(CredentialError, match="mount option 'endpoint_url' cannot be set"):
        _worker_credentials().storage_options(
            "s3", "lab-s3", {"endpoint_url": "https://author.example"}
        )

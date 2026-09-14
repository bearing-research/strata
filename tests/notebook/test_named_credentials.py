"""Named credentials for mounts and connections. Item 23.

A notebook names a credential and never contains one. The name reaches
provenance, the values never do, and a missing name fails the cell naming it.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.credentials import CredentialError, CredentialResolver
from strata.notebook.models import MountSpec
from strata.notebook.mounts import MountResolver, mount_fingerprint_sync

REGISTRY = {"lab-bucket": {"key": "${LAB_KEY}", "secret": "${LAB_SECRET}", "anon": "false"}}


class TestResolution:
    def test_references_resolve_from_the_notebook_env_before_the_server_env(self, monkeypatch):
        """The notebook's env is where a secret manager puts what it fetched."""
        monkeypatch.setenv("LAB_KEY", "server-key")
        monkeypatch.setenv("LAB_SECRET", "server-secret")
        resolver = CredentialResolver(REGISTRY, env={"LAB_KEY": "vault-key"})

        assert resolver.resolve("lab-bucket") == {
            "key": "vault-key",
            "secret": "server-secret",
            "anon": "false",
        }

    def test_an_unknown_name_is_named(self):
        with pytest.raises(CredentialError, match="'nope'"):
            CredentialResolver(REGISTRY).resolve("nope")

    def test_an_unset_reference_names_the_credential_and_the_variable(self, monkeypatch):
        monkeypatch.delenv("LAB_KEY", raising=False)
        monkeypatch.delenv("LAB_SECRET", raising=False)

        with pytest.raises(CredentialError, match=r"'lab-bucket'.*LAB_KEY"):
            CredentialResolver(REGISTRY).resolve("lab-bucket")

    def test_mount_options_layer_default_then_credential_then_the_mounts_own(self):
        resolver = CredentialResolver(
            {"org": {"endpoint_url": "https://org", "key": "o"}, "lab": {"key": "l"}},
            scheme_defaults={"s3": "org"},
        )

        options = resolver.storage_options("s3", "lab", {"endpoint_url": "https://mine"})

        assert options == {"endpoint_url": "https://mine", "key": "l"}

    def test_config_accepts_the_env_var_json_form(self):
        from strata.config import StrataConfig

        config = StrataConfig(
            notebook_credentials='{"lab-bucket": {"key": "${LAB_KEY}"}}',
            notebook_mount_credentials='{"s3": "lab-bucket"}',
        )

        assert config.notebook_credentials == {"lab-bucket": {"key": "${LAB_KEY}"}}
        assert config.notebook_mount_credentials == {"s3": "lab-bucket"}


class TestNoSecretInTheNotebook:
    def test_a_mount_and_a_connection_round_trip_by_name(self, tmp_path):
        from strata.notebook.models import ConnectionSpec
        from strata.notebook.parser import parse_notebook
        from strata.notebook.writer import (
            create_notebook,
            update_notebook_connections,
            update_notebook_mounts,
        )

        nb = create_notebook(tmp_path, "Creds", initialize_environment=False)
        update_notebook_mounts(
            nb, [MountSpec(name="raw", uri="s3://lab/raw", credential="lab-bucket")]
        )
        update_notebook_connections(
            nb,
            [
                ConnectionSpec(name="wh", driver="postgresql", host="db", credential="wh-ro"),
                ConnectionSpec(name="local", driver="sqlite", path="x.db"),
            ],
        )

        text = (nb / "notebook.toml").read_text()
        state = parse_notebook(nb)

        assert 'credential = "lab-bucket"' in text
        assert 'credential = "wh-ro"' in text
        assert "LAB_KEY" not in text
        assert state.mounts[0].credential == "lab-bucket"
        by_name = {c.name: c for c in state.connections}
        assert by_name["wh"].credential == "wh-ro"
        assert by_name["local"].credential is None

    def test_a_mount_annotation_can_name_a_credential(self):
        from strata.notebook.annotations import parse_annotations

        annotations = parse_annotations("# @mount raw s3://lab/raw ro credential=lab-bucket\nx = 1")

        assert annotations.mounts[0].credential == "lab-bucket"
        assert annotations.mounts[0].mode.value == "ro"


class TestProvenance:
    def _fingerprint(self, tmp_path, credential, env):
        data = tmp_path / "data"
        if not data.exists():
            # Once: the local fingerprint covers mtimes, and rewriting the file
            # between calls would be a change the fingerprint is right to see.
            data.mkdir()
            (data / "f.txt").write_text("rows")
        resolver = MountResolver(
            cache_dir=tmp_path / "cache",
            credential_resolver=CredentialResolver(
                {"a": {"token": "${T}"}, "b": {"token": "${T}"}}, env=env
            ),
        )
        mount = MountSpec(name="data", uri=f"file://{data}", credential=credential)
        return mount_fingerprint_sync(resolver, mount)

    def test_rotating_the_secret_invalidates_nothing(self, tmp_path):
        before = self._fingerprint(tmp_path, "a", {"T": "old"})
        after = self._fingerprint(tmp_path, "a", {"T": "rotated"})

        assert before == after
        assert "old" not in before

    def test_reading_through_another_credential_is_another_identity(self, tmp_path):
        assert self._fingerprint(tmp_path, "a", {"T": "x"}) != self._fingerprint(
            tmp_path, "b", {"T": "x"}
        )

    def test_a_mount_without_a_credential_keeps_its_existing_fingerprint(self, tmp_path):
        """No notebook's cache moves because this feature exists."""
        from strata.notebook.mounts import MountFingerprinter

        data = tmp_path / "data"
        data.mkdir()
        (data / "f.txt").write_text("rows")
        mount = MountSpec(name="data", uri=f"file://{data}")

        assert mount_fingerprint_sync(MountResolver(cache_dir=tmp_path), mount) == (
            f"data:{MountFingerprinter.fingerprint_mount_sync(mount)}"
        )

    def test_staleness_and_execution_agree_on_a_credentialed_mount(self, tmp_path, monkeypatch):
        """Or the cell never matches its own artifacts and sits stale forever."""
        from strata.notebook.executor import CellExecutor
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        data = tmp_path / "data"
        data.mkdir()
        (data / "f.txt").write_text("rows")
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(config=SimpleNamespace(notebook_credentials={"lab": {"k": "${K}"}})),
        )
        monkeypatch.setenv("K", "v")
        nb = create_notebook(tmp_path, "Parity", initialize_environment=False)
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", f"# @mount data file://{data} ro credential=lab\nx = 1")
        session = NotebookSession(parse_notebook(nb), nb)
        cell = session.notebook_state.get_cell("c1")
        from strata.notebook.annotations import parse_annotations

        mounts = parse_annotations(cell.source).mounts

        executed, _ = asyncio.run(CellExecutor(session)._fingerprint_mounts(mounts))
        staleness, _ = session._collect_mount_fingerprints(cell)

        assert executed == staleness
        assert "credential=lab" in executed[0]


def test_a_remote_mount_is_fetched_with_the_credentials_secret(tmp_path, monkeypatch):
    """Resolves with no secret in notebook.toml: fsspec is handed the values."""
    import fsspec

    seen: list[dict] = []

    def _filesystem(protocol, **options):
        seen.append(options)
        raise RuntimeError("stop before any network")

    monkeypatch.setattr(fsspec, "filesystem", _filesystem)
    resolver = MountResolver(
        cache_dir=tmp_path,
        credential_resolver=CredentialResolver(REGISTRY, env={"LAB_KEY": "k", "LAB_SECRET": "s"}),
    )

    with pytest.raises(RuntimeError, match="stop before any network"):
        asyncio.run(
            resolver.prepare_mounts(
                [MountSpec(name="raw", uri="s3://lab/raw", credential="lab-bucket")]
            )
        )

    assert seen and seen[-1] == {"key": "k", "secret": "s", "anon": "false"}


def test_a_missing_credential_fails_the_mount_naming_it(tmp_path):
    resolver = MountResolver(cache_dir=tmp_path, credential_resolver=CredentialResolver({}))

    with pytest.raises(CredentialError, match="'lab-bucket'"):
        asyncio.run(
            resolver.prepare_mounts(
                [MountSpec(name="raw", uri="s3://lab/raw", credential="lab-bucket")]
            )
        )


class TestConnections:
    def test_the_credential_fills_auth_under_the_blocks_own(self, monkeypatch):
        from strata.notebook.models import ConnectionSpec
        from strata.notebook.sql.cell_executor import _resolve_runtime_spec

        spec = ConnectionSpec(
            name="wh", driver="postgresql", credential="wh", auth={"user": "override"}
        )
        resolver = CredentialResolver(
            {"wh": {"user": "svc", "password": "${WH_PW}"}}, env={"WH_PW": "pw"}
        )

        runtime = _resolve_runtime_spec(spec, Path("/nb"), resolver)

        assert runtime.auth == {"user": "override", "password": "pw"}
        assert spec.auth == {"user": "override"}, "the on-disk spec is untouched"

    def test_the_identity_takes_the_name_not_the_values(self):
        from strata.notebook.models import ConnectionSpec
        from strata.notebook.sql.cell_executor import _with_credential

        plain = ConnectionSpec(name="wh", driver="postgresql")
        a = ConnectionSpec(name="wh", driver="postgresql", credential="a")
        b = ConnectionSpec(name="wh", driver="postgresql", credential="b")

        assert _with_credential("id", plain) == "id"
        assert _with_credential("id", a) != _with_credential("id", b)
        assert _with_credential("id", a) == _with_credential("id", a)

    async def test_a_sql_cell_whose_credential_is_missing_fails_naming_it(self, tmp_path):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.sql.cell_executor import execute_sql_cell
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        db = tmp_path / "events.db"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE t (x INTEGER)")
        nb = create_notebook(tmp_path, "sqlcred")
        add_cell_to_notebook(nb, "c1", language="sql")
        source = "# @sql connection=db\n# @cache forever\nSELECT x FROM t\n"
        write_cell(nb, "c1", source)
        toml = nb / "notebook.toml"
        toml.write_text(
            toml.read_text()
            + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db}"\ncredential = "nope"\n'
        )
        session = NotebookSession(parse_notebook(nb), nb)

        result = await execute_sql_cell(session, "c1", source)

        assert result["success"] is False
        assert "'nope'" in result["error"]


def test_a_worker_reads_credentials_from_its_own_environment(monkeypatch):
    from strata.notebook.remote_executor import _worker_credentials

    monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIALS", '{"lab": {"key": "${WK}"}}')
    monkeypatch.setenv("WK", "worker-secret")

    assert _worker_credentials().resolve("lab") == {"key": "worker-secret"}


async def test_a_cell_whose_mount_credential_is_missing_fails_naming_it(tmp_path):
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb = create_notebook(tmp_path, "MissingCred")
    add_cell_to_notebook(nb, "c1", None)
    source = "# @mount raw s3://lab/raw ro credential=lab-bucket\nx = 1"
    write_cell(nb, "c1", source)
    session = NotebookSession(parse_notebook(nb), nb)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success is False
    assert result.error is not None and "'lab-bucket'" in result.error

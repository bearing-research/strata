"""Who may mount a named credential read-write (``notebook_credential_writers``).

A lab's bucket is read-only until an admin lists who writes with it: an ``rw`` mount
through an unlisted name, from a cell or from ``notebook.toml``, fails before the run,
naming the credential; a listed one writes for a matching principal and fails for another.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from strata.auth.principal import principal_context
from strata.notebook.credentials import CredentialError, CredentialResolver
from strata.types import Principal

REGISTRY = {"lab": {"key": "lab-key"}, "shared": {"key": "shared-key"}, "org": {"key": "o"}}
WRITERS = {"lab": ["user_*", "agent:agt_1/*"], "shared": ["*"]}


def _resolver(principal: str | None, writers=WRITERS, defaults=None) -> CredentialResolver:
    return CredentialResolver(
        REGISTRY, writers=writers, principal=principal, scheme_defaults=defaults
    )


class TestWhoWrites:
    @pytest.mark.parametrize("principal", ["user_ana", "agent:agt_1/run-7"])
    def test_a_listed_name_writes_for_a_matching_principal(self, principal):
        _resolver(principal).ensure_writable("s3", "lab")

    @pytest.mark.parametrize("principal", ["agent:agt_2/run-1", "admin", None])
    def test_a_listed_name_is_refused_to_another_principal(self, principal):
        with pytest.raises(CredentialError, match="'lab' is not writable") as excinfo:
            _resolver(principal).ensure_writable("s3", "lab")
        assert "lab-key" not in str(excinfo.value)

    @pytest.mark.parametrize("principal", ["user_ana", None])
    def test_a_star_lets_anyone_write_with_no_principal_included(self, principal):
        _resolver(principal).ensure_writable("s3", "shared")

    def test_an_unlisted_name_is_read_only(self):
        with pytest.raises(CredentialError, match="'org' is read-only"):
            _resolver("user_ana").ensure_writable("s3", "org")

    def test_a_schemes_default_credential_counts(self):
        """A mount naming no credential still writes with its scheme's default."""
        with pytest.raises(CredentialError, match="'org' is read-only"):
            _resolver("user_ana", defaults={"s3": "org"}).ensure_writable("s3", None)

    def test_unset_every_name_writes(self):
        _resolver(None, writers=None).ensure_writable("s3", "org")
        _resolver(None, writers=None).ensure_writable("s3", None)

    def test_set_to_nothing_every_name_is_read_only(self):
        with pytest.raises(CredentialError, match="'shared' is read-only"):
            _resolver("user_ana", writers={}).ensure_writable("s3", "shared")

    @pytest.mark.parametrize("scheme", ["s3", "gs", "az"])
    @pytest.mark.parametrize("writers", [WRITERS, {}])
    def test_a_remote_mount_naming_no_credential_is_refused(self, scheme, writers):
        """It would write as the server's own identity, which no setting lists."""
        with pytest.raises(
            CredentialError, match=f"read-write {scheme} mount must name the source"
        ):
            _resolver("user_ana", writers=writers).ensure_writable(scheme, None)

    def test_a_local_mount_needs_no_credential(self):
        _resolver(None, writers={}).ensure_writable("file", None)

    def test_a_server_that_never_set_it_writes(self):
        from strata.config import StrataConfig

        CredentialResolver.from_config(StrataConfig()).ensure_writable("s3", None)

    def test_config_reads_the_env_var_json_form(self, monkeypatch):
        from strata.config import StrataConfig

        assert StrataConfig().notebook_credential_writers is None
        monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIAL_WRITERS", "{}")
        assert StrataConfig().notebook_credential_writers == {}
        monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIAL_WRITERS", '{"lab": ["user_*"]}')
        assert StrataConfig().notebook_credential_writers == {"lab": ["user_*"]}


class TestARunOnTheServer:
    """The upstream "done when", through real cells."""

    @pytest.fixture
    def fetched(self, monkeypatch):
        import fsspec

        from strata.config import StrataConfig
        from strata.notebook.executor import CellExecutor
        from strata.notebook.session import NotebookSession

        config = StrataConfig(
            notebook_credentials={"lab": {"key": "lab-key"}, "org": {"key": "org-key"}},
            notebook_credential_writers={"lab": ["user_*"]},
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
        seen: list[dict] = []

        def _filesystem(protocol, **options):
            seen.append(options)
            raise RuntimeError("stop before any network")

        monkeypatch.setattr(fsspec, "filesystem", _filesystem)
        return seen

    @staticmethod
    async def _run(
        parent: Path,
        source: str,
        principal: str | None,
        *,
        toml_mount=None,
        worker_url: str | None = None,
    ):
        from strata.notebook.executor import CellExecutor
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

        nb = create_notebook(parent, "nb")
        if toml_mount is not None:
            update_notebook_mounts(nb, [toml_mount])
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
        session = NotebookSession(parse_notebook(nb), nb)
        context = (
            principal_context(Principal(id=principal)) if principal is not None else nullcontext()
        )
        with context:
            return await CellExecutor(session).execute_cell("c1", source)

    async def test_a_cells_rw_mount_through_an_unlisted_name_fails_before_running(
        self, tmp_path, fetched
    ):
        source = "# @mount out s3://lab/out rw credential=org\nx = 1"

        result = await self._run(tmp_path, source, "user_ana")

        assert result.success is False
        assert "'org' is read-only" in (result.error or "")
        assert fetched == []

    async def test_a_cells_rw_mount_naming_no_credential_fails_before_running(
        self, tmp_path, fetched
    ):
        source = "# @mount out s3://lab/out rw\nx = 1"

        result = await self._run(tmp_path, source, "user_ana")

        assert "read-write s3 mount must name the source" in (result.error or "")
        assert fetched == []

    async def test_a_notebook_toml_rw_mount_is_judged_too(self, tmp_path, fetched):
        from strata.notebook.models import MountMode, MountSpec

        mount = MountSpec(
            name="out", uri="s3://lab/out", mode=MountMode.READ_WRITE, credential="org"
        )

        result = await self._run(tmp_path, "x = 1", "user_ana", toml_mount=mount)

        assert "'org' is read-only" in (result.error or "")
        assert fetched == []

    async def test_a_listed_name_writes_for_a_matching_principal(self, tmp_path, fetched):
        source = "# @mount out s3://lab/out rw credential=lab\nx = 1"

        result = await self._run(tmp_path, source, "user_ana")

        # Staging began with the credential's values.
        assert fetched and fetched[-1]["key"] == "lab-key"
        assert "stop before any network" in (result.error or "")

    async def test_a_listed_name_fails_for_another_principal(self, tmp_path, fetched):
        source = "# @mount out s3://lab/out rw credential=lab\nx = 1"

        result = await self._run(tmp_path, source, "agent:agt_2/run-1")

        assert "'lab' is not writable by 'agent:agt_2/run-1'" in (result.error or "")
        assert fetched == []

    async def test_a_read_only_mount_through_an_unlisted_name_still_reads(self, tmp_path, fetched):
        source = "# @mount raw s3://lab/raw ro credential=org\nx = 1"

        result = await self._run(tmp_path, source, None)

        assert fetched and fetched[-1]["key"] == "org-key"
        assert "stop before any network" in (result.error or "")

    async def test_a_remote_worker_is_never_asked(self, tmp_path, fetched):
        """Nothing listens at the worker's URL: reaching it would fail differently."""
        source = "# @mount out s3://lab/out rw credential=org\nx = 1"

        result = await self._run(
            tmp_path, source, "user_ana", worker_url="http://127.0.0.1:9/v1/execute"
        )

        assert result.success is False
        assert "'org' is read-only" in (result.error or "")


class TestASqlWriteCell:
    """A ``write=true`` connection that names a credential writes only for a listed principal."""

    @staticmethod
    async def _run(tmp_path: Path, monkeypatch, principal: str):
        import sqlite3

        from strata.config import StrataConfig
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.sql.cell_executor import execute_sql_cell
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        pytest.importorskip("adbc_driver_sqlite")
        config = StrataConfig(
            notebook_credentials={"lab": {}},
            notebook_credential_writers={"lab": ["user_*"]},
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        db = tmp_path / "out.db"
        source = "# @sql connection=db write=true\nCREATE TABLE t (n INTEGER);\n"
        nb = create_notebook(tmp_path, "nb")
        add_cell_to_notebook(nb, "c1", language="sql")
        write_cell(nb, "c1", source)
        toml = nb / "notebook.toml"
        toml.write_text(
            toml.read_text()
            + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db}"\ncredential = "lab"\n'
        )
        with principal_context(Principal(id=principal)):
            result = await execute_sql_cell(NotebookSession(parse_notebook(nb), nb), "c1", source)
        with sqlite3.connect(db) as conn:
            tables = conn.execute("SELECT name FROM sqlite_master").fetchall()
        return result, tables

    async def test_another_principal_is_refused_before_anything_is_written(
        self, tmp_path, monkeypatch
    ):
        result, tables = await self._run(tmp_path, monkeypatch, "agent:agt_2/run-1")

        assert not result["success"]
        assert "'lab' is not writable by 'agent:agt_2/run-1'" in result["error"]
        assert tables == []

    async def test_a_listed_principal_writes(self, tmp_path, monkeypatch):
        result, tables = await self._run(tmp_path, monkeypatch, "user_ana")

        assert result["success"], result.get("error")
        assert tables == [("t",)]

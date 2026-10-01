"""Service mode confines a SQL cell to its own database and lake.

SQL cells run inside the server process. Unconfined, a cell could read any
file the server can (on Linux ``/proc/self/environ`` holds its secrets) and a
write cell could ``COPY ... TO`` or ``ATTACH`` any path (SQLite: ``ATTACH``,
``VACUUM INTO``). Personal mode is the
user's own machine and stays unconfined. tests/notebook/test_e2e_duckdb_lake.py
checks that a confined cell still reads a real catalog and S3 mount.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from strata.config import StrataConfig
from strata.notebook.executor import CellExecutor
from strata.notebook.models import ConnectionSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.sql.registry import get_adapter
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def outside(tmp_path) -> Path:
    """A file the server can read that no notebook should."""
    path = tmp_path / "server-only" / "secret.txt"
    path.parent.mkdir()
    path.write_text("do not read")
    return path


@pytest.fixture
def database(tmp_path) -> Path:
    path = tmp_path / "nb.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("CREATE TABLE t AS SELECT 1 AS x")
    conn.close()
    return path


def _open(path: Path, confine_to: list[str], *, read_only: bool = True):
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(path), confine_to=confine_to)
    return get_adapter("duckdb").open(spec, read_only=read_only)


def test_a_confined_read_connection_reads_its_own_tables_and_nothing_else(database, outside):
    conn = _open(database, [])

    assert conn.execute("SELECT x FROM t").fetchall() == [(1,)]
    with pytest.raises(duckdb.PermissionException):
        conn.execute(f"SELECT content FROM read_text('{outside}')").fetchall()
    with pytest.raises(duckdb.InvalidInputException):
        conn.execute("SET enable_external_access = true")
    # A cursor, which the executor runs statements on, is confined too.
    with pytest.raises(duckdb.PermissionException):
        conn.cursor().execute(f"SELECT content FROM read_text('{outside}')").fetchall()


def test_a_confined_write_connection_writes_its_database_and_no_other_file(
    database, outside, tmp_path
):
    conn = _open(database, [], read_only=False)

    conn.execute("INSERT INTO t VALUES (2)")
    assert conn.execute("SELECT count(*) FROM t").fetchone() == (2,)
    leak = tmp_path / "leak.csv"
    with pytest.raises(duckdb.PermissionException):
        conn.execute(f"COPY t TO '{leak}'")
    with pytest.raises(duckdb.PermissionException):
        conn.execute(f"ATTACH '{tmp_path / 'other.duckdb'}' AS other")
    assert not leak.exists()


def test_a_location_it_is_confined_to_stays_readable(database, outside, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "data.csv").write_text("a\n1\n")

    conn = _open(database, [f"{allowed}/"])

    assert conn.execute(f"SELECT a FROM read_csv('{allowed / 'data.csv'}')").fetchall() == [(1,)]
    with pytest.raises(duckdb.PermissionException):
        conn.execute(f"SELECT content FROM read_text('{outside}')").fetchall()


def _notebook(tmp_path: Path, source: str) -> Path:
    nb_dir = create_notebook(tmp_path, "confine")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = ":memory:"\n')
    return nb_dir


@pytest.mark.parametrize(
    ("mode", "readable"), [("personal", True), ("service", False)], ids=["personal", "service"]
)
@pytest.mark.asyncio
async def test_only_service_mode_confines_a_sql_cell(
    tmp_path, monkeypatch, outside, mode, readable
):
    config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode=mode)
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    source = f"# @sql connection=db\nSELECT content FROM read_text('{outside}')\n"
    nb_dir = _notebook(tmp_path, source)
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success is readable, result.error
    if not readable:
        assert "Permission" in (result.error or "")


@pytest.mark.parametrize(
    ("sql", "refused"),
    [
        ("ATTACH 'x.db' AS o", "ATTACH"),
        ("/* hidden */ attach database 'x.db' as o", "ATTACH"),
        ("DETACH o", "DETACH"),
        ("VACUUM INTO '/tmp/copy.db'", "VACUUM"),
        ("INSERT INTO t VALUES (1); VACUUM", "VACUUM"),
        ("CREATE TABLE t(x); INSERT INTO t SELECT 1", None),
    ],
)
def test_a_confined_sqlite_write_cell_is_refused_what_reaches_other_files(sql, refused):
    from strata.notebook.sql.analyzer import confined_write_violation

    violation = confined_write_violation(sql, "sqlite")
    if refused is None:
        assert violation is None
    else:
        assert violation is not None and refused in violation


@pytest.mark.parametrize(
    ("mode", "allowed"), [("personal", True), ("service", False)], ids=["personal", "service"]
)
@pytest.mark.asyncio
async def test_only_service_mode_refuses_a_sqlite_write_cell_attaching_another_file(
    tmp_path, monkeypatch, mode, allowed
):
    import sqlite3

    other = tmp_path / "other.sqlite"
    with sqlite3.connect(other) as conn:
        conn.execute("CREATE TABLE secret(x)")
    config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode=mode)
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    nb_dir = create_notebook(tmp_path, "confine_sqlite")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    source = f"# @sql connection=db write=true\nATTACH '{other}' AS o\n"
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text()
        + f'\n[connections.db]\ndriver = "sqlite"\npath = "{tmp_path / "mine.sqlite"}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success is allowed, result.error
    if not allowed:
        assert "ATTACH" in (result.error or "")


# --- A local mount root bounds how much of the server's disk a read cell reads ---


@pytest.mark.parametrize(
    ("mode", "readable"), [("personal", True), ("service", False)], ids=["personal", "service"]
)
@pytest.mark.asyncio
async def test_a_mount_whose_root_holds_server_state_is_refused(
    tmp_path, monkeypatch, mode, readable
):
    """Confinement admits everything under a mount's root, so a notebook that
    mounts a directory holding the server's artifact store reads the store."""
    from strata.notebook.models import MountSpec
    from strata.notebook.writer import update_notebook_mounts

    server_root = tmp_path / "srv"
    secret = server_root / "artifacts" / "secret.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("another tenant's artifact")
    (server_root / "data.csv").write_text("a\n1\n")
    config = StrataConfig(
        cache_dir=tmp_path / "cache", deployment_mode=mode, artifact_dir=secret.parent
    )
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    source = f"# @sql connection=db\nSELECT content FROM read_text('{secret}')\n"
    nb_dir = create_notebook(tmp_path, "mount_root")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    write_cell(nb_dir, "c1", source)
    update_notebook_mounts(nb_dir, [MountSpec(name="srv", uri=server_root.as_uri())])
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = ":memory:"\n'
        'mounts = ["srv"]\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert result.success is readable, result.error
    if not readable:
        assert "mount 'srv'" in (result.error or "")
        assert "artifact" in (result.error or "")


class TestLocalMountRoots:
    """Which local roots a confined SQL cell may mount, checked before anything
    under the root is read (fingerprinting ``/`` would walk the whole disk)."""

    @pytest.fixture
    def config(self, tmp_path):
        return StrataConfig(
            cache_dir=tmp_path / "state" / "cache",
            artifact_dir=tmp_path / "state" / "artifacts",
            metadata_db=tmp_path / "state" / "meta" / "meta.sqlite",
            notebook_storage_dir=tmp_path / "notebooks",
            deployment_mode="service",
        )

    @staticmethod
    def _problem(uri: str, config) -> str | None:
        from strata.notebook.sql.lake import local_mount_root_problem

        return local_mount_root_problem(uri, config)

    @pytest.mark.parametrize("uri", ["file:///", "/", "file:///data", "/usr"])
    def test_the_filesystem_root_and_top_level_directories(self, uri, config):
        assert self._problem(uri, config) is not None

    def test_a_link_to_the_root_is_the_root(self, tmp_path, config):
        link = tmp_path / "data" / "everything"
        link.parent.mkdir()
        link.symlink_to("/")

        assert self._problem(link.as_uri(), config) is not None

    @pytest.mark.parametrize(
        "subpath", ["state", "state/cache", "state/meta", "notebooks", "state/artifacts"]
    )
    def test_a_root_holding_server_state(self, tmp_path, config, subpath):
        assert self._problem((tmp_path / subpath).as_uri(), config) is not None

    def test_the_servers_home_and_what_is_in_it(self, config):
        home = Path.home()

        assert self._problem(home.as_uri(), config) is not None
        assert self._problem((home / ".aws").as_uri(), config) is not None

    def test_the_process_filesystem(self, config):
        assert self._problem("file:///proc/self", config) is not None

    def test_a_notebooks_own_data_and_other_directories_are_fine(self, tmp_path, config):
        assert self._problem((tmp_path / "notebooks" / "nb" / "data").as_uri(), config) is None
        assert self._problem((tmp_path / "lake" / "raw").as_uri(), config) is None
        assert self._problem("s3://bucket/", config) is None

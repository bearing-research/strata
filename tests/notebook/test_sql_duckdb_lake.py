"""DuckDB SQL cells over mounts and a named catalog, without services.

A mount is a view whose files are an input (a new file misses the cache), and a catalog
table read is pinned at the snapshot its provenance folds. Real catalogs:
``test_e2e_duckdb_lake.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.conftest import service_auth

pytest.importorskip("duckdb")

from strata.config import StrataConfig  # noqa: E402
from strata.notebook.executor import CellExecutor  # noqa: E402
from strata.notebook.models import CellStatus, ConnectionSpec, MountSpec  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.sql.drivers.duckdb import DuckDBAdapter  # noqa: E402
from strata.notebook.sql.lake import lake_tables, pin_snapshots  # noqa: E402
from strata.notebook.writer import (  # noqa: E402
    add_cell_to_notebook,
    create_notebook,
    update_notebook_mounts,
    write_cell,
)


def _notebook(tmp_path: Path, cells: dict[str, str], connection: str) -> Path:
    nb_dir = create_notebook(tmp_path, "duckdb_lake")
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb_dir, cell_id, language="sql")
        write_cell(nb_dir, cell_id, source)
    update_notebook_mounts(nb_dir, [MountSpec(name="raw", uri=(tmp_path / "raw").as_uri())])
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + f"\n[connections.lake]\n{connection}\n")
    return nb_dir


def _write_parquet(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"k": values}), path)


def _lake_name(uri: str) -> str:
    import hashlib

    return "lake_" + hashlib.sha256(uri.encode()).hexdigest()[:16]


def _rows(session: NotebookSession, uri: str) -> list[dict[str, Any]]:
    art_id, version = uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
    blob = session.get_artifact_manager().load_artifact_data(art_id, int(version))
    return pa.ipc.open_stream(blob).read_all().to_pylist()


async def _run(nb_dir: Path, session: NotebookSession, cell_id: str) -> Any:
    source = (nb_dir / "cells" / f"{cell_id}.py").read_text()
    result = await CellExecutor(session).execute_cell(cell_id, source)
    # What the route does after a run.
    session.compute_staleness()
    if result.success:
        session.mark_executed_ready(cell_id)
    return result


@pytest.mark.asyncio
async def test_a_mount_is_a_view_and_a_new_file_makes_the_cell_stale(tmp_path):
    _write_parquet(tmp_path / "raw" / "2026" / "a.parquet", [1, 2])
    nb_dir = _notebook(
        tmp_path,
        # No database probe, so the mount's fingerprint is all that tells the
        # two runs apart.
        {"c1": "# @sql connection=lake\n# @cache forever\nSELECT sum(k) AS total FROM raw\n"},
        'driver = "duckdb"\npath = ":memory:"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    first = await _run(nb_dir, session, "c1")
    assert first.success, first.error
    assert _rows(session, first.artifact_uri) == [{"total": 3}]
    assert session.compute_staleness()["c1"].status == CellStatus.READY

    _write_parquet(tmp_path / "raw" / "2026" / "b.parquet", [10])
    assert session.compute_staleness()["c1"].status != CellStatus.READY
    second = await _run(nb_dir, session, "c1")
    assert second.success, second.error
    assert second.cache_hit is False
    assert _rows(session, second.artifact_uri) == [{"total": 13}]

    third = await _run(nb_dir, session, "c1")
    assert third.cache_hit is True


@pytest.mark.asyncio
async def test_a_database_file_and_its_mounts_are_read_together_and_never_written(tmp_path):
    import duckdb

    db = tmp_path / "base.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE labels AS SELECT * FROM (VALUES (1, 'one'), (2, 'two')) t(k, v)")
    _write_parquet(tmp_path / "raw" / "a.parquet", [1, 2])
    nb_dir = _notebook(
        tmp_path,
        {
            "c1": "# @sql connection=lake\nSELECT v FROM raw JOIN labels USING (k) ORDER BY v\n",
            "c2": "# @sql connection=lake\nINSERT INTO labels VALUES (3, 'three')\n",
        },
        f'driver = "duckdb"\npath = "{db}"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    read = await _run(nb_dir, session, "c1")
    assert read.success, read.error
    assert _rows(session, read.artifact_uri) == [{"v": "one"}, {"v": "two"}]

    write = await _run(nb_dir, session, "c2")
    assert write.success is False
    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT count(*) FROM labels").fetchone() == (2,)


@pytest.mark.asyncio
async def test_a_mount_the_cell_does_not_declare_fails_naming_it(tmp_path):
    _write_parquet(tmp_path / "raw" / "a.parquet", [1])
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT * FROM cooked\n"},
        'driver = "duckdb"\npath = ":memory:"\nmounts = ["cooked"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")
    assert result.success is False
    assert "mount 'cooked' is not declared" in result.error


@pytest.mark.asyncio
async def test_a_catalog_the_server_does_not_configure_fails_naming_it(tmp_path, monkeypatch):
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT * FROM lake.taxi.trips\n"},
        'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"',
    )
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: StrataConfig())
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: StrataConfig())
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")
    assert result.success is False
    assert "catalog 'lake' is not configured" in result.error


def test_only_the_catalog_tables_a_query_reads_are_pinned():
    sql = (
        "WITH trips AS (SELECT 1 AS x) "
        "SELECT * FROM lake.taxi.trips t JOIN trips USING (x) "
        "JOIN warehouse.taxi.trips w USING (x) JOIN lake.taxi.zones z USING (x) WHERE t.id > ?"
    )
    pinned = pin_snapshots(sql, "lake", {("taxi", "trips"): 11, ("taxi", "zones"): 22})

    assert "lake.taxi.trips t AT (VERSION => 11)" in pinned
    assert "lake.taxi.zones z AT (VERSION => 22)" in pinned
    assert "warehouse.taxi.trips w USING" in pinned
    assert "JOIN trips USING" in pinned
    assert pinned.endswith("> ?")
    # DuckDB's parser is the judge: an alias after AT (what sqlglot before
    # 30.13 wrote) is a syntax error, and a pin must bind to its own table.
    assert _pinned_versions(pinned) == {("trips", "t"): 11, ("zones", "z"): 22}


def test_a_pinned_query_is_the_cells_own_text():
    """sqlglot's regeneration would read ``MAP {a: b}`` as ``MAP {'a': b}``."""
    sql = "SELECT MAP {a: b} AS m FROM lake.ns.t AS x(a, b) WHERE a > ?"

    pinned = pin_snapshots(sql, "lake", {("ns", "t"): 5})

    assert (
        pinned == "SELECT MAP {a: b} AS m FROM lake.ns.t AS x(a, b) AT (VERSION => 5) WHERE a > ?"
    )
    assert _pinned_versions(pinned) == {("t", "x"): 5}


def _pinned_versions(sql: str) -> dict[tuple[str, str], int]:
    """Each table's AT (VERSION => n), as DuckDB parses *sql*."""
    import json

    import duckdb

    tree = json.loads(duckdb.connect().execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    assert not tree.get("error"), tree.get("error_message")
    versions: dict[tuple[str, str], int] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            at = node.get("at_clause")
            if node.get("type") == "BASE_TABLE" and at:
                versions[(node["table_name"], node["alias"])] = at["expr"]["value"]["value"]
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return versions


def test_lake_tables_are_the_catalog_tables_a_read_cell_reads(tmp_path):
    nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    state = parse_notebook(nb_dir)
    read = "# @sql connection=lake\nSELECT * FROM lake.taxi.trips JOIN lake.taxi.zones USING (id)\n"

    assert sorted(t.uri for t in lake_tables(state, read)) == [
        "lake:taxi.trips",
        "lake:taxi.zones",
    ]
    assert lake_tables(state, "# @sql connection=lake write\nDELETE FROM lake.taxi.trips\n") == []
    state.connections = [ConnectionSpec(name="lake", driver="sqlite", path=":memory:")]
    assert lake_tables(state, read) == []


def test_the_catalog_and_mounts_are_part_of_the_connection_identity():
    adapter = DuckDBAdapter()
    base = {"name": "lake", "driver": "duckdb", "path": ":memory:"}
    identities = {
        adapter.canonicalize_connection_id(ConnectionSpec(**base, **extra))
        for extra in ({}, {"catalog": "lake"}, {"catalog": "other"}, {"mounts": ["raw"]})
    }
    assert len(identities) == 4


@pytest.mark.asyncio
async def test_a_database_file_named_like_a_reserved_database_still_opens(tmp_path):
    import duckdb

    db = tmp_path / "memory.duckdb"
    with duckdb.connect(str(db)) as conn:
        conn.execute("CREATE TABLE labels AS SELECT 1 AS k, 'one' AS v")
    _write_parquet(tmp_path / "raw" / "a.parquet", [1])
    nb_dir = _notebook(
        tmp_path,
        {"c1": "# @sql connection=lake\nSELECT v FROM raw JOIN labels USING (k)\n"},
        f'driver = "duckdb"\npath = "{db}"\nmounts = ["raw"]',
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)

    result = await _run(nb_dir, session, "c1")

    assert result.success, result.error
    assert _rows(session, result.artifact_uri) == [{"v": "one"}]


def test_a_table_the_first_resolution_missed_reads_what_the_retry_found(tmp_path, monkeypatch):
    from strata.notebook import tables
    from strata.notebook.sql.adapter import QualifiedTable
    from strata.notebook.sql.lake import resolve_lake

    nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    config = StrataConfig(catalogs={"lake": {"type": "rest", "uri": "http://catalog"}})
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    monkeypatch.setattr(
        tables,
        "fingerprint_tables",
        lambda specs, cfg, env=None, notebook=None, locations=None: (["x:unresolved"], {}),
    )
    monkeypatch.setattr(
        tables, "resolve_table_snapshot", lambda spec, cfg, env=None, notebook=None: 7
    )

    lake = resolve_lake(
        session,
        "c1",
        "",
        session.notebook_state.connections[0],
        [QualifiedTable("lake", "taxi", "trips")],
    )

    assert lake.snapshots == {("taxi", "trips"): 7}
    # And the cell's key names that snapshot, not the random stand-in fingerprint_tables
    # invents for a table it could not resolve, which no later run would reproduce.
    assert lake.fingerprints == [f"{_lake_name('lake:taxi.trips')}:table:lake:taxi.trips:7"]


def test_a_table_never_written_is_read_unpinned_and_never_cached(tmp_path, monkeypatch):
    from strata.notebook import tables
    from strata.notebook.sql.adapter import QualifiedTable
    from strata.notebook.sql.lake import resolve_lake

    nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    config = StrataConfig(catalogs={"lake": {"type": "rest", "uri": "http://catalog"}})
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    name = _lake_name("lake:taxi.trips")
    monkeypatch.setattr(
        tables,
        "fingerprint_tables",
        lambda specs, cfg, env=None, notebook=None, locations=None: ([], {name: None}),
    )

    def resolve():
        return resolve_lake(
            session,
            "c1",
            "",
            session.notebook_state.connections[0],
            [QualifiedTable("lake", "taxi", "trips")],
        )

    first, second = resolve(), resolve()

    assert first.snapshots == {}
    assert first.fingerprints[0].startswith(f"{name}:table:empty:")
    assert first.fingerprints != second.fingerprints


class TestACatalogCredential:
    """A catalog's named credential resolves against the notebook's env, as a mount's does."""

    @staticmethod
    def _resolve(tmp_path, monkeypatch, credentials):
        from strata.notebook import tables
        from strata.notebook.sql.adapter import QualifiedTable
        from strata.notebook.sql.lake import resolve_lake

        nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        session.notebook_state.env = {"LAKE_TOKEN": "from-the-vault"}
        config = StrataConfig(
            catalogs={
                "lake": {"type": "rest", "uri": "http://catalog", "strata_credential": "lake-ro"}
            },
            notebook_credentials=credentials,
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        monkeypatch.setattr(
            tables,
            "fingerprint_tables",
            lambda specs, cfg, env=None, notebook=None, locations=None: ([], {}),
        )
        monkeypatch.setattr(
            tables, "resolve_table_snapshot", lambda spec, cfg, env=None, notebook=None: 7
        )
        return resolve_lake(
            session,
            "c1",
            "",
            session.notebook_state.connections[0],
            [QualifiedTable("lake", "taxi", "trips")],
        )

    def test_its_fields_become_catalog_properties(self, tmp_path, monkeypatch):
        lake = self._resolve(tmp_path, monkeypatch, {"lake-ro": {"token": "${LAKE_TOKEN}"}})

        assert lake.spec.catalog_properties == {
            "type": "rest",
            "uri": "http://catalog",
            "token": "from-the-vault",
        }

    def test_a_missing_one_fails_naming_it(self, tmp_path, monkeypatch):
        from strata.notebook.sql.lake import LakeError

        with pytest.raises(LakeError, match="credential 'lake-ro' is not defined"):
            self._resolve(tmp_path, monkeypatch, {})


@pytest.mark.asyncio
async def test_a_catalog_credential_set_only_in_the_notebook_env_resolves_its_snapshots(
    tmp_path, monkeypatch
):
    from pyiceberg.catalog.sql import SqlCatalog

    from strata.notebook.models import TableSpec
    from strata.notebook.sql.adapter import QualifiedTable
    from strata.notebook.sql.lake import resolve_lake

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    db_uri = f"sqlite:///{warehouse / 'catalog.db'}"
    catalog = SqlCatalog("lake", uri=db_uri, warehouse=warehouse.as_uri())
    catalog.create_namespace("taxi")
    table = catalog.create_table("taxi.trips", schema=pa.schema([("id", pa.int64())]))
    table.append(pa.table({"id": [1]}))
    snapshot = catalog.load_table("taxi.trips").current_snapshot().snapshot_id
    source = "# @sql connection=lake\nSELECT id FROM lake.taxi.trips\n"
    nb_dir = _notebook(
        tmp_path, {"c1": source}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"'
    )
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.notebook_state.env = {"LAKE_DB": db_uri}
    monkeypatch.delenv("LAKE_DB", raising=False)
    config = StrataConfig(
        cache_dir=tmp_path / "cache",
        catalogs={
            "lake": {"type": "sql", "warehouse": warehouse.as_uri(), "strata_credential": "db"}
        },
        notebook_credentials={"db": {"uri": "${LAKE_DB}"}},
    )
    monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)

    lake = resolve_lake(
        session,
        "c1",
        source,
        session.notebook_state.connections[0],
        [QualifiedTable("lake", "taxi", "trips")],
    )
    staleness = session._collect_table_fingerprints(session.notebook_state.get_cell("c1"))
    _, executed = await CellExecutor(session)._fingerprint_tables(
        [TableSpec(name="trips", uri="lake:taxi.trips")]
    )

    assert lake.snapshots == {("taxi", "trips"): snapshot}
    assert [f.rsplit(":", 1)[1] for f in staleness] == [str(snapshot)]
    assert executed == {"trips": snapshot}


def test_the_catalogs_s3_secret_reaches_only_its_warehouse():
    from strata.notebook.sql.drivers.duckdb import _attach_catalog

    class Recorder:
        def __init__(self):
            self.statements: list[str] = []

        def execute(self, statement):
            self.statements.append(statement)

    keys = {
        "type": "rest",
        "uri": "http://catalog",
        "s3.access-key-id": "k",
        "s3.secret-access-key": "s",
    }
    scoped, unscoped = Recorder(), Recorder()
    _attach_catalog(scoped, "lake", {**keys, "warehouse": "s3://lake/wh"})
    _attach_catalog(unscoped, "lake", {**keys, "warehouse": "prod"})

    (secret,) = [s for s in scoped.statements if "TYPE s3" in s]
    assert "SCOPE 's3://lake/wh/'" in secret
    assert not [s for s in unscoped.statements if "TYPE s3" in s]
    assert all(s.endswith("READ_ONLY)") for s in scoped.statements if s.startswith("ATTACH"))


class _Recorder:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, statement):
        self.statements.append(statement)


class _Globbing(_Recorder):
    """A connection on which every glob finds a file."""

    def execute(self, statement):
        super().execute(statement)
        return self

    def fetchone(self):
        return (1,)


def _which_secret():
    """A DuckDB connection with httpfs, and what answers which secret it reads a path with."""
    import duckdb

    conn = duckdb.connect()
    try:
        conn.execute("INSTALL httpfs; LOAD httpfs")
    except duckdb.Error as exc:
        pytest.skip(f"DuckDB extensions unavailable: {exc}")

    def answering(path):
        row = conn.execute("SELECT name FROM which_secret(?, 's3')", [path]).fetchone()
        return row[0] if row else None

    return conn, answering


class TestAGlueCatalog:
    """Attached through Glue's Iceberg REST endpoint, signed with the catalog's own keys."""

    _GLUE = {"type": "glue", "glue.id": "123456789012", "glue.region": "eu-west-1"}

    @staticmethod
    def _attach(properties):
        from strata.notebook.sql.drivers.duckdb import _attach_catalog

        recorder = _Recorder()
        _attach_catalog(recorder, "lake", properties)
        return recorder.statements

    def test_its_keys_sign_the_catalog_in_its_region(self):
        statements = self._attach(
            {**self._GLUE, "glue.access-key-id": "AKIA1", "glue.secret-access-key": "s1"}
        )

        assert statements == [
            'CREATE OR REPLACE SECRET "strata_catalog_lake" '
            "(TYPE s3, KEY_ID 'AKIA1', SECRET 's1', REGION 'eu-west-1')",
            "ATTACH '123456789012' AS \"lake\" (TYPE iceberg, "
            "ENDPOINT 'glue.eu-west-1.amazonaws.com/iceberg', "
            "AUTHORIZATION_TYPE 'sigv4', SECRET \"strata_catalog_lake\", READ_ONLY)",
        ]

    def test_without_keys_the_aws_credential_chain_signs(self):
        statements = self._attach({**self._GLUE, "glue.profile-name": "lab"})

        assert statements[0] == "INSTALL aws; LOAD aws"
        assert statements[1] == (
            'CREATE OR REPLACE SECRET "strata_catalog_lake" '
            "(TYPE s3, PROVIDER credential_chain, PROFILE 'lab', REGION 'eu-west-1')"
        )

    def test_client_and_s3_keys_stand_in_for_glue_ones(self):
        (secret, _) = self._attach(
            {
                "type": "glue",
                "glue.id": "1",
                "client.region": "us-east-2",
                "s3.access-key-id": "k",
                "s3.secret-access-key": "s",
                "s3.session-token": "t",
            }
        )

        assert "KEY_ID 'k', SECRET 's', SESSION_TOKEN 't', REGION 'us-east-2'" in secret

    @pytest.mark.parametrize(
        ("missing", "message"), [("glue.id", "needs glue.id"), ("glue.region", "needs glue.region")]
    )
    def test_it_is_refused_without_its_account_or_region(self, missing, message):
        properties = {k: v for k, v in self._GLUE.items() if k != missing}

        with pytest.raises(RuntimeError, match=message):
            self._attach(properties)

    def test_duckdb_signs_the_request_for_glue_with_those_keys(self):
        """The statements as DuckDB runs them, against a local stand-in for Glue's host."""
        import socket
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        import duckdb

        host = "glue.eu-west-1.localhost"
        try:
            socket.gethostbyname(host)
        except OSError:
            pytest.skip(f"{host} does not resolve here")
        conn = duckdb.connect()
        try:
            conn.execute("INSTALL httpfs; LOAD httpfs; INSTALL iceberg; LOAD iceberg")
        except duckdb.Error as exc:
            pytest.skip(f"DuckDB extensions unavailable: {exc}")
        seen: list[str] = []

        class Glue(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(self.headers.get("Authorization") or "")
                self.send_response(403)
                self.end_headers()

            def log_message(self, *args):
                del args

        server = HTTPServer(("127.0.0.1", 0), Glue)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            secret, attach = self._attach(
                {**self._GLUE, "glue.access-key-id": "AKIA1", "glue.secret-access-key": "s1"}
            )
            local = f"http://{host}:{server.server_address[1]}/iceberg"
            conn.execute(secret)
            # The stand-in refuses, as Glue does a bad signature.
            with pytest.raises(duckdb.Error, match="403"):
                conn.execute(attach.replace("glue.eu-west-1.amazonaws.com/iceberg", local))
        finally:
            server.shutdown()

        assert seen and seen[0].startswith("AWS4-HMAC-SHA256 Credential=AKIA1/")
        assert "/eu-west-1/glue/aws4_request" in seen[0]


_STORE = {"s3.endpoint": "https://store.example"}


class TestARestCatalogThatVendsNothing:
    """An empty access-delegation header: DuckDB asks the catalog for no credentials and
    reads the tables with the catalog's own keys, where those tables live (a real catalog
    that refuses to vend: test_e2e_duckdb_lake.py)."""

    _LAKE = {
        "type": "rest",
        "uri": "http://catalog",
        "warehouse": "lake",
        "header.X-Iceberg-Access-Delegation": "",
        "s3.access-key-id": "k",
        "s3.secret-access-key": "s",
        "s3.endpoint": "https://store.example:9000",
        "s3.region": "eu-west-1",
        "s3.path-style-access": "true",
    }

    @staticmethod
    def _attach(properties, locations=None):
        from strata.notebook.sql.drivers.duckdb import _attach_catalog

        recorder = _Recorder()
        _attach_catalog(recorder, "lake", properties, locations)
        return recorder.statements

    def test_its_keys_reach_only_the_tables_the_query_reads(self):
        statements = self._attach(self._LAKE, ["s3://lake/l1/b/", "s3://lake/l1/a/"])

        assert statements == [
            'CREATE OR REPLACE SECRET "strata_catalog_lake_s3" (TYPE s3, '
            "KEY_ID 'k', SECRET 's', REGION 'eu-west-1', ENDPOINT 'store.example:9000', "
            "USE_SSL true, URL_STYLE 'path', SCOPE ['s3://lake/l1/a/', 's3://lake/l1/b/'])",
            "ATTACH 'lake' AS \"lake\" (TYPE iceberg, ENDPOINT 'http://catalog', "
            "AUTHORIZATION_TYPE 'none', ACCESS_DELEGATION_MODE 'none', READ_ONLY)",
        ]

    @pytest.mark.parametrize("warehouse", ["s3://lake/wh", "s3://lake/wh/"])
    def test_with_an_s3_warehouse_they_reach_the_warehouse(self, warehouse):
        secret, attach = self._attach({**self._LAKE, "warehouse": warehouse}, ["s3://lake/wh/t/"])

        assert secret.endswith("SCOPE 's3://lake/wh/')")
        assert "ACCESS_DELEGATION_MODE 'none'" in attach

    def test_neither_a_warehouse_nor_a_mount_scope_reaches_a_sibling_prefix(self):
        from strata.notebook.sql.drivers.duckdb import _create_mount_view

        conn, answering = _which_secret()
        secret, _ = self._attach({**self._LAKE, "warehouse": "s3://lake/wh"})
        conn.execute(secret)
        for name, uri in [("events", "s3://raw/events"), ("one", "s3://raw/one.parquet")]:
            mount = _Globbing()
            _create_mount_view(
                mount, {"name": name, "uri": uri, "storage_options": {"key": "m"}}, {}
            )
            conn.execute(mount.statements[0])

        assert answering("s3://lake/wh/ns/t/data/0.parquet") == "strata_catalog_lake_s3"
        assert answering("s3://lake/wh2/ns/t/data/0.parquet") is None
        assert answering("s3://raw/events/part-0.parquet") == "strata_mount_events"
        assert answering("s3://raw/events2/part-0.parquet") is None
        # A mount of one file is read at its root exactly.
        assert answering("s3://raw/one.parquet") == "strata_mount_one"

    @pytest.mark.parametrize(
        "delegation", [None, "vended-credentials", "remote-signing, vended-credentials"]
    )
    def test_a_catalog_asked_to_vend_gets_no_keys_for_a_named_warehouse(self, delegation):
        from strata.notebook.sql.drivers.duckdb import secret_scoped_to_tables

        properties = {k: v for k, v in self._LAKE.items() if not k.startswith("header.")}
        if delegation is not None:
            properties["header.x-iceberg-access-delegation"] = delegation

        assert self._attach(properties, ["s3://lake/l1/a/"]) == [
            "ATTACH 'lake' AS \"lake\" (TYPE iceberg, ENDPOINT 'http://catalog', "
            "AUTHORIZATION_TYPE 'none', READ_ONLY)"
        ]
        assert not secret_scoped_to_tables(properties)

    @pytest.mark.parametrize(
        ("fields", "path_style"),
        [
            (_STORE, True),
            ({}, False),
            *[({**_STORE, "s3.path-style-access": v}, False) for v in ("false", "0", "no", "OFF")],
            *[({"s3.path-style-access": v}, True) for v in ("true", "1", "yes", "On")],
            ({**_STORE, "s3.force-virtual-addressing": "true"}, False),
            ({"s3.path-style-access": "true", "s3.force-virtual-addressing": "1"}, False),
            ({**_STORE, "s3.force-virtual-addressing": "false"}, True),
        ],
    )
    def test_path_style_reads_its_booleans_as_pyiceberg_does(self, fields, path_style):
        from strata.notebook.sql.drivers.duckdb import _s3_secret

        secret = _s3_secret("lake", {"s3.access-key-id": "k", **fields})

        assert secret is not None
        assert ("URL_STYLE 'path'" in secret) is path_style

    def test_a_path_style_that_is_not_a_boolean_fails_naming_it(self):
        from strata.notebook.sql.drivers.duckdb import _s3_secret

        with pytest.raises(ValueError, match="s3.path-style-access"):
            _s3_secret("lake", {**_STORE, "s3.path-style-access": "sometimes"})

    def test_duckdb_answers_with_them_only_under_those_tables(self):
        import duckdb

        from strata.notebook.sql.drivers.duckdb import _s3_secret

        conn = duckdb.connect()
        try:
            conn.execute("INSTALL httpfs; LOAD httpfs")
        except duckdb.Error as exc:
            pytest.skip(f"DuckDB extensions unavailable: {exc}")
        secret, _ = self._attach(self._LAKE, ["s3://lake/l1/a/", "s3://lake/l1/b/"])
        conn.execute(secret)
        conn.execute(_s3_secret("strata_mount_raw", {"key": "m"}, ["s3://raw/events"]))

        def answering(path):
            row = conn.execute("SELECT name FROM which_secret(?, 's3')", [path]).fetchone()
            return row[0] if row else None

        assert answering("s3://lake/l1/a/data/0.parquet") == "strata_catalog_lake_s3"
        assert answering("s3://lake/l1/b/metadata/snap.avro") == "strata_catalog_lake_s3"
        assert answering("s3://lake/l1/ab/data/0.parquet") is None
        assert answering("s3://other/x.parquet") is None
        assert answering("s3://raw/events/part-0.parquet") == "strata_mount_raw"

    def _session(self, tmp_path, monkeypatch, cells=None, mode="personal"):
        nb_dir = _notebook(
            tmp_path, cells or {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"'
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        config = StrataConfig(
            catalogs={"lake": self._LAKE}, deployment_mode=mode, **service_auth(mode)
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        return session

    @staticmethod
    def _catalog(monkeypatch, root="s3://lake"):
        """Each table the catalog loads, and a list that takes the catalog down when not empty.

        A table lives at ``<root>/<namespace>.<table>``.
        """
        from types import SimpleNamespace

        from strata.iceberg.catalog import PyIcebergCatalog

        loads: list[str] = []
        down: list[bool] = []

        def load_table(self, uri):
            loads.append(uri)
            if down:
                raise ConnectionError("the catalog is unreachable")
            snapshot = SimpleNamespace(snapshot_id=7, summary={})
            return SimpleNamespace(
                location=lambda: f"{root}/{uri.partition(':')[2]}",
                current_snapshot=lambda: snapshot,
            )

        monkeypatch.setattr(PyIcebergCatalog, "load_table", load_table)
        return loads, down

    @staticmethod
    def _resolve(session, pinned=None):
        from strata.notebook.sql.adapter import QualifiedTable
        from strata.notebook.sql.lake import resolve_lake

        return resolve_lake(
            session,
            "c1",
            "",
            session.notebook_state.connections[0],
            [QualifiedTable("lake", "taxi", "trips"), QualifiedTable("lake", "taxi", "zones")],
            pinned=pinned,
        )

    def test_each_table_read_is_loaded_once_and_located_by_that_load(self, tmp_path, monkeypatch):
        session = self._session(tmp_path, monkeypatch)
        loads, _ = self._catalog(monkeypatch)

        spec = self._resolve(session).spec

        assert spec.catalog_locations == ["s3://lake/taxi.trips/", "s3://lake/taxi.zones/"]
        assert sorted(loads) == ["lake:taxi.trips", "lake:taxi.zones"]

    def test_a_catalog_that_vends_is_not_located(self, tmp_path, monkeypatch):
        session = self._session(tmp_path, monkeypatch)
        vending = {k: v for k, v in self._LAKE.items() if not k.startswith("header.")}
        config = StrataConfig(catalogs={"lake": vending})
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        self._catalog(monkeypatch)

        assert getattr(self._resolve(session).spec, "catalog_locations", None) is None

    @pytest.mark.parametrize("mode", ["personal", "service"])
    def test_a_pinned_table_is_located_only_when_the_query_runs(self, tmp_path, monkeypatch, mode):
        from strata.notebook.sql.lake import locate_tables

        session = self._session(tmp_path, monkeypatch, mode=mode)
        loads, _ = self._catalog(monkeypatch)
        confined = mode == "service"

        lake = self._resolve(session, pinned={("taxi", "trips"): 3})
        assert loads == ["lake:taxi.zones"]
        assert lake.spec.catalog_locations == ["s3://lake/taxi.zones/"]
        assert lake.locations == (["s3://lake/taxi.zones/"] if confined else [])

        spec = locate_tables(session, lake, lake.spec)
        assert loads == ["lake:taxi.zones", "lake:taxi.trips"]
        assert spec.catalog_locations == ["s3://lake/taxi.zones/", "s3://lake/taxi.trips/"]
        assert lake.locations == (
            ["s3://lake/taxi.zones/", "s3://lake/taxi.trips/"] if confined else []
        )

    def test_a_vending_catalogs_pinned_table_is_located_in_service_mode_for_confinement_only(
        self, tmp_path, monkeypatch
    ):
        from strata.notebook.sql.lake import locate_tables

        session = self._session(tmp_path, monkeypatch, mode="service")
        vending = {k: v for k, v in self._LAKE.items() if not k.startswith("header.")}
        config = StrataConfig(
            catalogs={"lake": vending}, deployment_mode="service", **service_auth()
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        loads, _ = self._catalog(monkeypatch)

        lake = self._resolve(session, pinned={("taxi", "trips"): 3})
        assert loads == ["lake:taxi.zones"]

        spec = locate_tables(session, lake, lake.spec)
        assert loads == ["lake:taxi.zones", "lake:taxi.trips"]
        assert lake.locations == ["s3://lake/taxi.zones/", "s3://lake/taxi.trips/"]
        # The catalog vends its own keys, so its secret stays unscoped.
        assert getattr(spec, "catalog_locations", None) is None

    @pytest.mark.asyncio
    async def test_a_pinned_cache_hit_needs_no_catalog(self, tmp_path, monkeypatch):
        from strata.notebook.sql import cell_executor

        source = "# @sql connection=lake\n# @cache snapshot\nSELECT id FROM lake.taxi.trips\n"
        session = self._session(tmp_path, monkeypatch, {"c1": source})
        loads, down = self._catalog(monkeypatch)
        located: list[list[str]] = []

        def execute(adapter, spec, analysis, params, *, at=None, lake=None):
            located.append(spec.catalog_locations)
            return pa.table({"id": [1, 2]})

        monkeypatch.setattr(cell_executor, "_execute_query", execute)

        first = await cell_executor.execute_sql_cell(session, "c1", source)
        assert first["success"], first["error"]
        assert located == [["s3://lake/taxi.trips/"]]
        assert loads == ["lake:taxi.trips"]

        down.append(True)
        hit = await cell_executor.execute_sql_cell(session, "c1", source)
        assert hit["success"], hit["error"]
        assert hit["cache_hit"] is True
        assert loads == ["lake:taxi.trips"]

        # Past the cache, the pinned table is located before the query, through the catalog.
        monkeypatch.setattr(session.get_artifact_manager(), "find_cached", lambda provenance: None)
        replay = await cell_executor.execute_sql_cell(session, "c1", source)
        assert replay["success"] is False
        assert "the catalog is unreachable" in replay["error"]
        assert len(located) == 1

    @pytest.mark.asyncio
    async def test_the_table_a_query_reads_is_a_leaf_of_its_output(self, tmp_path, monkeypatch):
        from strata.notebook.sql import cell_executor

        source = "# @sql connection=lake\n# @cache snapshot\nSELECT id FROM lake.taxi.trips\n"
        session = self._session(tmp_path, monkeypatch, {"c1": source})
        self._catalog(monkeypatch)
        monkeypatch.setattr(
            cell_executor,
            "_execute_query",
            lambda adapter, spec, analysis, params, *, at=None, lake=None: pa.table({"id": [1]}),
        )

        result = await cell_executor.execute_sql_cell(session, "c1", source)
        assert result["success"], result["error"]

        store = session.get_artifact_manager().artifact_store
        (uri,) = session.notebook_state.get_cell("c1").artifact_uris.values()
        art_id, version = uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
        node = store.definition_of(art_id, int(version))
        leaves = [
            store.graph.get(port.target_node_id)
            for port in store.graph.inputs(node)
            if port.role == "table"
        ]
        # The leaf names the table and where the catalog found it: one leaf per table,
        # not one per bare name across catalogs.
        assert [(leaf.domain, leaf.payload) for leaf in leaves] == [
            ("strata/table", b"lake:taxi.trips\ns3://lake/taxi.trips")
        ]

    @pytest.mark.asyncio
    async def test_a_pinned_cache_hit_in_service_mode_needs_no_catalog(self, tmp_path, monkeypatch):
        from strata.notebook.sql import cell_executor

        source = "# @sql connection=lake\n# @cache snapshot\nSELECT id FROM lake.taxi.trips\n"
        session = self._session(tmp_path, monkeypatch, {"c1": source}, mode="service")
        loads, down = self._catalog(monkeypatch)
        confined: list[list[str]] = []

        def execute(adapter, spec, analysis, params, *, at=None, lake=None):
            confined.append(spec.confine_to)
            return pa.table({"id": [1, 2]})

        monkeypatch.setattr(cell_executor, "_execute_query", execute)

        first = await cell_executor.execute_sql_cell(session, "c1", source)
        assert first["success"], first["error"]
        assert confined == [["s3://lake/taxi.trips/"]]
        assert loads == ["lake:taxi.trips"]

        down.append(True)
        hit = await cell_executor.execute_sql_cell(session, "c1", source)
        assert hit["success"], hit["error"]
        assert hit["cache_hit"] is True
        assert loads == ["lake:taxi.trips"]

        # Past the cache, the pinned table is located before the query, through the catalog.
        monkeypatch.setattr(session.get_artifact_manager(), "find_cached", lambda provenance: None)
        replay = await cell_executor.execute_sql_cell(session, "c1", source)
        assert replay["success"] is False
        assert "the catalog is unreachable" in replay["error"]
        assert len(confined) == 1

    @pytest.mark.asyncio
    async def test_a_service_mode_miss_confines_the_query_to_the_pinned_tables_location(
        self, tmp_path, monkeypatch
    ):
        import duckdb

        from strata.notebook.sql import cell_executor
        from strata.notebook.sql.drivers.duckdb import _confine

        warehouse = tmp_path / "wh"
        _write_parquet(warehouse / "taxi.trips" / "data" / "0.parquet", [1, 2])
        outside = tmp_path / "server-only.parquet"
        _write_parquet(outside, [9])
        source = "# @sql connection=lake\n# @cache snapshot\nSELECT id FROM lake.taxi.trips\n"
        session = self._session(tmp_path, monkeypatch, {"c1": source}, mode="service")
        loads, _ = self._catalog(monkeypatch, root=str(warehouse))
        confined: list[list[str]] = []

        def execute(adapter, spec, analysis, params, *, at=None, lake=None):
            # The driver confines its handle so; read inside and outside the table.
            confined.append(spec.confine_to)
            conn = duckdb.connect(":memory:")
            _confine(conn, spec.confine_to)
            inside = f"{warehouse}/taxi.trips/data/0.parquet"
            table = conn.execute(f"SELECT k AS id FROM read_parquet('{inside}')").to_arrow_table()
            with pytest.raises(duckdb.PermissionException):
                conn.execute(f"SELECT * FROM read_parquet('{outside}')").fetchall()
            return table

        monkeypatch.setattr(cell_executor, "_execute_query", execute)

        first = await cell_executor.execute_sql_cell(session, "c1", source)
        assert first["success"], first["error"]

        # The second run is pinned; a miss locates the table only now, and confines to it.
        monkeypatch.setattr(session.get_artifact_manager(), "find_cached", lambda provenance: None)
        miss = await cell_executor.execute_sql_cell(session, "c1", source)
        assert miss["success"], miss["error"]
        assert miss["cache_hit"] is False
        assert loads == ["lake:taxi.trips", "lake:taxi.trips"]
        assert confined == [[f"{warehouse}/taxi.trips/"]] * 2


class TestAGcsOrAzureMount:
    """Read through the mount's own fsspec filesystem (real stores: test_e2e_duckdb_lake.py)."""

    @staticmethod
    def _open(tmp_path, monkeypatch, mounts):
        import io

        import duckdb
        from fsspec.implementations.memory import MemoryFileSystem

        from strata.notebook import mounts as mounts_module

        class Gs(MemoryFileSystem):
            protocol = ("gs", "gcs")

        built: list[dict] = []

        def filesystem(protocol, storage_options):
            assert protocol == "gcs"
            built.append(storage_options)
            return Gs()

        buffer = io.BytesIO()
        pq.write_table(pa.table({"k": [4, 5]}), buffer)
        Gs().pipe("gs://raw/events/part-0.parquet", buffer.getvalue())
        monkeypatch.setattr(mounts_module, "_mount_filesystem", filesystem)
        conn = DuckDBAdapter(connect_fn=lambda path, read_only: duckdb.connect(path))._open_lake(
            ":memory:", None, None, mounts
        )
        return conn, built

    def test_it_is_a_view_over_its_files(self, tmp_path, monkeypatch):
        options = {"token": "anon"}
        conn, built = self._open(
            tmp_path,
            monkeypatch,
            [
                {"name": "a", "uri": "gs://raw/events", "storage_options": options},
                {"name": "b", "uri": "gcs://raw/events/part-0.parquet", "storage_options": options},
            ],
        )

        assert conn.cursor().execute("SELECT sum(k) FROM a").fetchall() == [(9,)]
        assert conn.cursor().execute("SELECT sum(k) FROM b").fetchall() == [(9,)]
        assert built == [options], "one filesystem per scheme, with the mount's options"

    def test_two_with_different_options_are_refused_naming_both(self, tmp_path, monkeypatch):
        with pytest.raises(RuntimeError, match="mounts 'a' and 'b' read gs with different"):
            self._open(
                tmp_path,
                monkeypatch,
                [
                    {"name": "a", "uri": "gs://raw/events", "storage_options": {"token": "x"}},
                    {"name": "b", "uri": "gs://raw/events", "storage_options": {"token": "y"}},
                ],
            )

    def test_its_confined_location_is_where_duckdb_reads_it(self):
        from strata.notebook.sql.lake import _mount_location

        assert _mount_location("gs://raw/events") == "gs://raw/events/"
        assert _mount_location("az://raw/events/") == "abfs://raw/events/"


class TestNotebookCatalogs:
    """``[catalogs.<name>]`` in notebook.toml: honored in personal mode, refused in service mode."""

    _NOTEBOOK = {"type": "rest", "uri": "http://notebook-catalog"}

    def _session(self, tmp_path, monkeypatch, config):
        from strata.notebook import tables

        nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
        toml = nb_dir / "notebook.toml"
        toml.write_text(
            toml.read_text() + '\n[catalogs.lake]\ntype = "rest"\nuri = "http://notebook-catalog"\n'
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        seen: list[dict] = []

        def fingerprint(specs, cfg, env=None, notebook=None, locations=None):
            seen.append(dict(cfg.catalogs))
            return [], {}

        monkeypatch.setattr(tables, "fingerprint_tables", fingerprint)
        monkeypatch.setattr(
            tables, "resolve_table_snapshot", lambda spec, cfg, env=None, notebook=None: 7
        )
        return session, seen

    @staticmethod
    def _resolve(session):
        from strata.notebook.sql.adapter import QualifiedTable
        from strata.notebook.sql.lake import resolve_lake

        return resolve_lake(
            session,
            "c1",
            "",
            session.notebook_state.connections[0],
            [QualifiedTable("lake", "taxi", "trips")],
        )

    def test_they_are_parsed_from_notebook_toml(self, tmp_path, monkeypatch):
        session, _ = self._session(tmp_path, monkeypatch, StrataConfig())

        assert session.notebook_state.catalogs == {"lake": self._NOTEBOOK}

    def test_one_resolves_in_personal_mode_over_a_server_one_of_its_name(
        self, tmp_path, monkeypatch
    ):
        server = {"type": "rest", "uri": "http://server-catalog"}
        config = StrataConfig(catalogs={"lake": server, "other": server})
        session, seen = self._session(tmp_path, monkeypatch, config)

        lake = self._resolve(session)

        assert lake.spec.catalog_properties == self._NOTEBOOK
        assert seen == [{"lake": self._NOTEBOOK, "other": server}]
        assert config.catalogs["lake"] == server, "the server's own config is untouched"

    def test_staleness_resolves_its_tables_through_it(self, tmp_path, monkeypatch):
        session, seen = self._session(tmp_path, monkeypatch, StrataConfig())
        add_cell_to_notebook(session.path, "c1", language="sql")
        source = "# @sql connection=lake\nSELECT * FROM lake.taxi.trips\n"
        write_cell(session.path, "c1", source)
        session = NotebookSession(parse_notebook(session.path), session.path)

        session._collect_table_fingerprints(session.notebook_state.get_cell("c1"))

        assert seen == [{"lake": self._NOTEBOOK}]

    def test_service_mode_refuses_one_naming_notebook_toml(self, tmp_path, monkeypatch):
        from strata.notebook.sql.lake import LakeError

        session, _ = self._session(
            tmp_path, monkeypatch, StrataConfig(deployment_mode="service", **service_auth())
        )

        with pytest.raises(LakeError, match="defined in notebook.toml"):
            self._resolve(session)


class TestCacheSnapshotOnDuckDB:
    """Only a catalog's tables can be pinned (the pinned run: test_e2e_duckdb_lake.py)."""

    @pytest.mark.asyncio
    async def test_a_connection_without_a_catalog_is_still_refused(self, tmp_path):
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])
        nb_dir = _notebook(
            tmp_path,
            {"c1": "# @sql connection=lake\n# @cache snapshot\nSELECT * FROM raw\n"},
            'driver = "duckdb"\npath = ":memory:"\nmounts = ["raw"]',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await _run(nb_dir, session, "c1")

        assert result.success is False
        assert "supports_snapshot=False" in result.error

    @pytest.mark.asyncio
    async def test_a_query_reading_a_mount_is_refused_naming_it(self, tmp_path):
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])
        nb_dir = _notebook(
            tmp_path,
            {
                "c1": "# @sql connection=lake\n# @cache snapshot\n"
                "SELECT * FROM lake.taxi.trips JOIN raw USING (k)\n"
            },
            'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"\nmounts = ["raw"]',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await _run(nb_dir, session, "c1")

        assert result.success is False
        assert "raw is not one" in result.error

    def test_a_new_snapshot_does_not_make_the_cell_stale(self, tmp_path):
        nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
        state = parse_notebook(nb_dir)
        body = "SELECT * FROM lake.taxi.trips\n"

        assert lake_tables(state, f"# @sql connection=lake\n{body}")
        assert lake_tables(state, f"# @sql connection=lake\n# @cache snapshot\n{body}") == []


def test_a_python_notebook_needs_no_sql_extra(tmp_path):
    """Staleness and provenance of non-SQL cells must not import sqlglot."""
    import subprocess
    import sys
    import textwrap

    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "plain")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")
    script = textwrap.dedent(
        f"""
        import sys
        sys.modules["sqlglot"] = None  # an import of it now fails
        import asyncio
        from pathlib import Path
        from strata.notebook.executor import CellExecutor
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        nb = Path({str(nb_dir)!r})
        session = NotebookSession(parse_notebook(nb), nb)
        session.compute_staleness()
        asyncio.run(CellExecutor(session)._compute_cell_provenance("c1", "x = 1\\n"))
        print("ok")
        """
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.stdout.strip().endswith("ok"), done.stderr


class TestAReadCellReads:
    """A body can end the read-only transaction, so read cells are checked before the driver."""

    @staticmethod
    async def _run(tmp_path, body: str):
        nb_dir = _notebook(
            tmp_path,
            {"c1": f"# @sql connection=lake\n{body}\n"},
            'driver = "duckdb"\npath = ":memory:"',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)
        return await _run(nb_dir, session, "c1")

    @pytest.mark.asyncio
    async def test_a_committed_attach_cannot_write_another_database(self, tmp_path):
        import duckdb

        victim = tmp_path / "victim.duckdb"
        with duckdb.connect(str(victim)) as conn:
            conn.execute("CREATE TABLE t AS SELECT 1 AS x")
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(
            tmp_path,
            f"COMMIT; ATTACH '{victim}' AS w (READ_WRITE); "
            "CREATE TABLE w.main.pwn AS SELECT 42 AS x; SELECT 1 AS ok",
        )

        assert result.success is False
        assert "is not a read" in result.error
        with duckdb.connect(str(victim), read_only=True) as conn:
            tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert tables == ["t"], "a read cell wrote into another database"

    @pytest.mark.asyncio
    async def test_copying_out_of_the_database_is_not_a_read(self, tmp_path):
        target = tmp_path / "leak.csv"
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(tmp_path, f"COPY (SELECT 1 AS x) TO '{target}'")

        assert result.success is False
        assert "COPY is not a read" in result.error
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_reads_still_run(self, tmp_path):
        _write_parquet(tmp_path / "raw" / "a.parquet", [1])

        result = await self._run(
            tmp_path, "WITH a AS (SELECT 1 AS x) SELECT sum(x) AS total FROM a"
        )

        assert result.success, result.error


class TestHowACatalogTableIsWritten:
    """DuckDB resolves database and schema case-insensitively; a two-part name uses the default.

    A missed spelling is read live under a provenance that never goes stale.
    """

    @staticmethod
    def _tables(state, body: str):
        from strata.notebook.sql.lake import lake_tables

        return sorted(t.uri for t in lake_tables(state, f"# @sql connection=lake\n{body}\n"))

    def test_the_spellings_of_one_table_are_that_table(self, tmp_path):
        nb_dir = _notebook(tmp_path, {}, 'driver = "duckdb"\npath = ":memory:"\ncatalog = "lake"')
        state = parse_notebook(nb_dir)

        assert self._tables(state, "SELECT * FROM LAKE.taxi.trips") == ["lake:taxi.trips"]
        assert self._tables(state, 'SELECT * FROM "LAKE".taxi.trips') == ["lake:taxi.trips"]
        assert self._tables(state, "SELECT * FROM lake.trips") == ["lake:main.trips"]
        assert self._tables(state, "SELECT * FROM other.taxi.trips") == []

    def test_every_spelling_is_pinned(self):
        from strata.notebook.sql.lake import pin_snapshots

        snapshots = {("taxi", "trips"): 11, ("main", "zones"): 22}

        pinned = pin_snapshots(
            "SELECT * FROM LAKE.taxi.trips JOIN lake.zones USING (id)", "lake", snapshots
        )

        assert "AT (VERSION => 11)" in pinned
        assert "AT (VERSION => 22)" in pinned


class TestWhatCountsAsAReadStatement:
    """The classifier decides what a read cell may send to the driver. Two
    ways to be wrong: a write it lets through, and a read it refuses."""

    @staticmethod
    def _violation(sql: str, dialect: str = "duckdb"):
        from strata.notebook.sql.analyzer import read_only_violation

        return read_only_violation(sql, dialect)

    @pytest.mark.asyncio
    async def test_explain_analyze_runs_what_it_wraps_so_it_is_not_a_read(self, tmp_path):
        """DuckDB's EXPLAIN ANALYZE executes the statement it describes."""
        target = tmp_path / "leak.csv"
        nb_dir = _notebook(
            tmp_path,
            {"c1": f"# @sql connection=lake\nEXPLAIN ANALYZE COPY (SELECT 1 AS x) TO '{target}'\n"},
            'driver = "duckdb"\npath = ":memory:"',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await _run(nb_dir, session, "c1")

        assert result.success is False
        assert "EXPLAIN ANALYZE" in result.error
        assert not target.exists(), "a read cell wrote a file through EXPLAIN ANALYZE"

    def test_a_plain_explain_is_still_a_read(self):
        assert self._violation("EXPLAIN SELECT 1") is None

    def test_the_parenthesised_form_is_caught_too(self):
        assert "EXPLAIN ANALYZE" in (self._violation("EXPLAIN (ANALYZE) DELETE FROM t") or "")

    @pytest.mark.asyncio
    async def test_a_comment_does_not_hide_the_analyze(self, tmp_path):
        """The parser keeps comments in the argument, so ``/*x*/`` must not hide ANALYZE."""
        target = tmp_path / "leak.csv"
        nb_dir = _notebook(
            tmp_path,
            {
                "c1": "# @sql connection=lake\n"
                f"EXPLAIN /*x*/ ANALYZE COPY (SELECT 1 AS x) TO '{target}'\n"
            },
            'driver = "duckdb"\npath = ":memory:"',
        )
        session = NotebookSession(parse_notebook(nb_dir), nb_dir)

        result = await _run(nb_dir, session, "c1")

        assert result.success is False
        assert not target.exists(), "a comment carried a write past the classifier"

    def test_analyze_anywhere_in_the_option_list_is_caught(self):
        """The options are a set, not a sequence: ANALYZE runs the statement
        wherever in the brackets it is written."""
        for sql in (
            "EXPLAIN (FORMAT JSON, ANALYZE) DELETE FROM t",
            "EXPLAIN --c\nANALYZE DELETE FROM t",
            "EXPLAIN ANALYSE DELETE FROM t",
        ):
            assert "EXPLAIN ANALYZE" in (self._violation(sql) or ""), sql

    def test_a_query_that_merely_mentions_the_word_still_describes_its_plan(self):
        assert self._violation("EXPLAIN SELECT 'ANALYZE' AS w") is None
        assert self._violation("EXPLAIN (FORMAT JSON) SELECT 1") is None

    def test_reads_that_are_not_selects_are_allowed(self):
        for sql in ("VALUES (1), (2)", "SUMMARIZE t", "TABLE t", "DESCRIBE t", "SHOW TABLES"):
            assert self._violation(sql) is None, sql
        assert self._violation("SHOW search_path", "postgres") is None

    def test_the_message_names_the_annotation_that_works(self):
        """`write` alone is ignored by the parser; the flag is `write=true`."""
        from strata.notebook.annotations import parse_annotations

        message = self._violation("INSERT INTO t VALUES (1)") or ""
        assert "write=true" in message
        annotation = parse_annotations(
            "# @sql connection=db write=true\nINSERT INTO t VALUES (1)\n"
        )
        assert annotation.sql is not None and annotation.sql.write is True

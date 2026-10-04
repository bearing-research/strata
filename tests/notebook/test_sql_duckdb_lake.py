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

    assert "lake.taxi.trips AS t AT (VERSION => 11)" in pinned
    assert "lake.taxi.zones AS z AT (VERSION => 22)" in pinned
    assert "warehouse.taxi.trips AS w" in pinned
    assert "JOIN trips USING" in pinned
    assert pinned.endswith("> ?")
    # DuckDB's parser is the judge: an alias after AT (what sqlglot before
    # 30.13 wrote) is a syntax error, and a pin must bind to its own table.
    assert _pinned_versions(pinned) == {("trips", "t"): 11, ("zones", "z"): 22}


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
    monkeypatch.setattr(tables, "fingerprint_tables", lambda specs, cfg: (["x:unresolved"], {}))
    monkeypatch.setattr(tables, "resolve_table_snapshot", lambda spec, cfg: 7)

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
            catalogs={"lake": {"type": "rest", "uri": "http://catalog", "credential": "lake-ro"}},
            notebook_credentials=credentials,
        )
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)
        monkeypatch.setattr(tables, "fingerprint_tables", lambda specs, cfg: ([], {}))
        monkeypatch.setattr(tables, "resolve_table_snapshot", lambda spec, cfg: 7)
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
    assert "SCOPE 's3://lake/wh'" in secret
    assert not [s for s in unscoped.statements if "TYPE s3" in s]
    assert all(s.endswith("READ_ONLY)") for s in scoped.statements if s.startswith("ATTACH"))


class _Recorder:
    def __init__(self):
        self.statements: list[str] = []

    def execute(self, statement):
        self.statements.append(statement)


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

        def fingerprint(specs, cfg):
            seen.append(dict(cfg.catalogs))
            return [], {}

        monkeypatch.setattr(tables, "fingerprint_tables", fingerprint)
        monkeypatch.setattr(tables, "resolve_table_snapshot", lambda spec, cfg: 7)
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

        session, _ = self._session(tmp_path, monkeypatch, StrataConfig(deployment_mode="service"))

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

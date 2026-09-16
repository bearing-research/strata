"""Tests for the BigQuery DriverAdapter — contract, identity hash,
and probe shape with mocked ADBC connections. Real-BigQuery
integration tests are out of scope locally (would need a paid
project + network access)."""

from __future__ import annotations

import json
from contextlib import contextmanager

from strata.notebook.models import ConnectionSpec
from strata.notebook.sql import FreshnessToken, QualifiedTable, SchemaFingerprint
from strata.notebook.sql.drivers.bigquery import BigQueryAdapter

# --- mock-conn helper -------------------------------------------------------


class _FakeCursor:
    """Minimal DBAPI cursor that returns scripted rows.

    ``scripts`` is a list of (query_substring, result) pairs; the
    next ``fetchone`` / ``fetchall`` call uses the result whose
    query substring matches the most recently executed SQL. The
    cursor records every (sql, params) pair so tests can assert
    what the adapter actually issued.
    """

    def __init__(self, scripts: list[tuple[str, object]]):
        self._scripts = scripts
        self._last_match: object = None
        self.executions: list[tuple[str, dict]] = []

    def execute(self, sql, parameters=None):
        self.executions.append((sql, dict(parameters or {})))
        for needle, value in self._scripts:
            if needle in sql:
                self._last_match = value
                return
        self._last_match = None

    def fetchone(self):
        last = self._last_match
        if isinstance(last, list):
            return last.pop(0) if last else None
        if last is None:
            return None
        return last

    def fetchall(self):
        last = self._last_match
        if isinstance(last, list):
            return last
        return [last] if last is not None else []

    def close(self):
        pass


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    @contextmanager
    def cursor(self):
        yield self._cursor

    def close(self):
        pass


# --- capabilities ----------------------------------------------------------


def test_capabilities_match_design():
    a = BigQueryAdapter()
    assert a.name == "bigquery"
    assert a.sqlglot_dialect == "bigquery"
    assert a.capabilities.per_table_freshness is True
    # Time travel: a snapshot is a timestamp (time_travel.py).
    assert a.capabilities.supports_snapshot is True
    assert a.capabilities.needs_separate_probe_conn is False


# --- canonicalize_connection_id -------------------------------------------


def test_connection_id_uses_project_and_dataset(tmp_path):
    """Project + dataset are identity-shaping. Two connections
    that point at different datasets must produce different ids."""
    a = BigQueryAdapter()
    base = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme-prod",
        dataset_id="events",
    )
    other_dataset = base.model_copy(update={"dataset_id": "metrics"})
    other_project = base.model_copy(update={"project_id": "acme-dev"})

    cid = a.canonicalize_connection_id(base)
    assert a.canonicalize_connection_id(other_dataset) != cid
    assert a.canonicalize_connection_id(other_project) != cid


def test_connection_id_includes_credentials_principal(tmp_path):
    """The service account's ``client_email`` is what BigQuery
    actually keys visibility on. Different SAs → different views,
    so the cache must segregate them."""
    sa1 = tmp_path / "sa1.json"
    sa2 = tmp_path / "sa2.json"
    sa1.write_text(json.dumps({"client_email": "reader@acme-prod.iam.gserviceaccount.com"}))
    sa2.write_text(json.dumps({"client_email": "admin@acme-prod.iam.gserviceaccount.com"}))

    a = BigQueryAdapter()
    base = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme-prod",
        dataset_id="events",
        credentials_path=str(sa1),
    )
    swapped = base.model_copy(update={"credentials_path": str(sa2)})
    assert a.canonicalize_connection_id(base) != a.canonicalize_connection_id(swapped)


def test_connection_id_falls_back_to_path_when_principal_unreadable(tmp_path):
    """When the credentials file isn't readable (relative path
    that hasn't been resolved, file moved between machines), the
    path itself folds into the identity so two distinct
    *unread* credentials still produce distinct ids."""
    a = BigQueryAdapter()
    base = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme-prod",
        credentials_path="/missing/sa1.json",
    )
    other = base.model_copy(update={"credentials_path": "/missing/sa2.json"})
    assert a.canonicalize_connection_id(base) != a.canonicalize_connection_id(other)


def test_connection_id_write_credentials_only_in_write_identity(tmp_path):
    """``write_credentials_path`` joins identity only when
    ``read_only=False``.

    Read cells never apply write_credentials_path at open time, so
    changing it must not churn read-cell caches. Write cells *do*
    use those credentials, so the cache identity for write cells
    must distinguish them from the read-side principal.
    """
    sa_ro = tmp_path / "ro.json"
    sa_rw = tmp_path / "rw.json"
    sa_ro.write_text(json.dumps({"client_email": "reader@x.iam"}))
    sa_rw.write_text(json.dumps({"client_email": "writer@x.iam"}))

    a = BigQueryAdapter()
    base = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme",
        credentials_path=str(sa_ro),
    )
    with_write = base.model_copy(update={"write_credentials_path": str(sa_rw)})

    # Read identity: ignores write_credentials_path, so swapping it
    # leaves the read connection_id unchanged.
    assert a.canonicalize_connection_id(base, read_only=True) == a.canonicalize_connection_id(
        with_write, read_only=True
    )

    # Write identity: write_credentials_path joins, so swapping
    # invalidates write-cell caches.
    assert a.canonicalize_connection_id(base, read_only=False) != a.canonicalize_connection_id(
        with_write, read_only=False
    )


def test_connection_id_ambient_adc_sentinel_when_no_credentials():
    """When no credentials are configured for the active side,
    identity carries an ``ambient_adc`` sentinel. Without it, two
    laptops running gcloud auth as different humans would alias
    onto the same connection_id and poison each other's cache."""
    a = BigQueryAdapter()
    spec = ConnectionSpec(name="x", driver="bigquery", project_id="acme")

    cid_ambient = a.canonicalize_connection_id(spec, read_only=True)

    # Same project, but with explicit credentials → must differ
    # from the ambient case.
    with_creds = spec.model_copy(update={"credentials_path": "/tmp/missing.json"})
    cid_explicit = a.canonicalize_connection_id(with_creds, read_only=True)

    assert cid_ambient != cid_explicit


# --- open() credentials selection ------------------------------------------


def test_open_uses_credentials_for_read_cells(tmp_path):
    """``read_only=True`` selects ``credentials_path``."""
    sa = tmp_path / "ro.json"
    sa.write_text("{}")
    captured: list[dict] = []
    a = BigQueryAdapter(connect_fn=lambda kw: captured.append(kw) or _FakeConn(_FakeCursor([])))

    spec = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme",
        dataset_id="events",
        credentials_path=str(sa),
    )
    a.open(spec, read_only=True)

    assert captured[0]["adbc.bigquery.sql.project_id"] == "acme"
    assert captured[0]["adbc.bigquery.sql.dataset_id"] == "events"
    assert captured[0]["adbc.bigquery.sql.auth_credentials"] == str(sa)


def test_open_uses_write_credentials_for_write_cells(tmp_path):
    """``read_only=False`` switches to ``write_credentials_path``
    when configured. This is the BigQuery analogue of Snowflake's
    ``write_role`` — makes ``# @sql write=true`` meaningful."""
    sa_ro = tmp_path / "ro.json"
    sa_rw = tmp_path / "rw.json"
    sa_ro.write_text("{}")
    sa_rw.write_text("{}")
    captured: list[dict] = []
    a = BigQueryAdapter(connect_fn=lambda kw: captured.append(kw) or _FakeConn(_FakeCursor([])))

    spec = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme",
        credentials_path=str(sa_ro),
        write_credentials_path=str(sa_rw),
    )
    a.open(spec, read_only=True)
    a.open(spec, read_only=False)

    assert captured[0]["adbc.bigquery.sql.auth_credentials"] == str(sa_ro)
    assert captured[1]["adbc.bigquery.sql.auth_credentials"] == str(sa_rw)


def test_open_falls_back_to_read_credentials_when_no_write(tmp_path):
    """Without ``write_credentials_path``, write cells inherit
    the read credentials. Documented behavior — the IAM grants
    on that SA decide whether the write succeeds."""
    sa = tmp_path / "ro.json"
    sa.write_text("{}")
    captured: list[dict] = []
    a = BigQueryAdapter(connect_fn=lambda kw: captured.append(kw) or _FakeConn(_FakeCursor([])))

    spec = ConnectionSpec(
        name="x",
        driver="bigquery",
        project_id="acme",
        credentials_path=str(sa),
    )
    a.open(spec, read_only=False)
    assert captured[0]["adbc.bigquery.sql.auth_credentials"] == str(sa)


# --- probe_freshness --------------------------------------------------------


def test_probe_freshness_groups_tables_by_dataset():
    """One ``__TABLES__`` query per (project, dataset). Tables
    grouped by their effective project/dataset get one round-trip
    per group."""
    cur = _FakeCursor(
        [("__TABLES__", ("events", 1714596000000))],
    )
    a = BigQueryAdapter()
    tables = [
        QualifiedTable(catalog="acme-prod", schema="events", name="orders"),
        QualifiedTable(catalog="acme-prod", schema="metrics", name="counters"),
    ]
    token = a.probe_freshness(_FakeConn(cur), tables)
    assert isinstance(token, FreshnessToken)
    assert token.value

    tables_queries = [sql for sql, _ in cur.executions if "__TABLES__" in sql]
    assert len(tables_queries) == 2
    assert any("`acme-prod.events.__TABLES__`" in q for q in tables_queries)
    assert any("`acme-prod.metrics.__TABLES__`" in q for q in tables_queries)


def test_probe_freshness_token_changes_on_last_modified():
    """Same table, different ``last_modified_time`` → different
    token."""
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog="acme", schema="events", name="orders")]

    cur1 = _FakeCursor([("__TABLES__", ("orders", 1714596000000))])
    cur2 = _FakeCursor([("__TABLES__", ("orders", 1714596001000))])
    t1 = a.probe_freshness(_FakeConn(cur1), tables)
    t2 = a.probe_freshness(_FakeConn(cur2), tables)
    assert t1.value != t2.value


def test_probe_freshness_uses_session_defaults_for_unqualified_tables():
    """Tables without a (project, dataset) fall back to
    ``@@project_id`` and ``@@dataset_id``."""
    cur = _FakeCursor(
        [
            ("@@project_id, @@dataset_id", ("acme-prod", "events")),
            ("__TABLES__", ("orders", 1714596000000)),
        ]
    )
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog=None, schema=None, name="orders")]
    a.probe_freshness(_FakeConn(cur), tables)

    queries = [sql for sql, _ in cur.executions if "__TABLES__" in sql]
    assert any("`acme-prod.events.__TABLES__`" in q for q in queries)


def test_probe_freshness_no_dataset_yields_sentinel():
    """If neither the table nor the session has a dataset, fold a
    ``no-dataset`` sentinel rather than crash or run the query
    against an unknown view."""
    cur = _FakeCursor([])
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog=None, schema=None, name="orders")]
    token = a.probe_freshness(_FakeConn(cur), tables)
    assert token.value
    queries = [sql for sql, _ in cur.executions if "__TABLES__" in sql]
    assert queries == []


def test_probe_freshness_missing_table_distinct_from_present():
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog="acme", schema="events", name="orders")]
    cur_present = _FakeCursor([("__TABLES__", ("orders", 1714596000000))])
    cur_missing = _FakeCursor([("__TABLES__", None)])
    assert (
        a.probe_freshness(_FakeConn(cur_present), tables).value
        != a.probe_freshness(_FakeConn(cur_missing), tables).value
    )


def test_probe_freshness_empty_tables():
    """No tables → empty token, no queries issued."""
    cur = _FakeCursor([])
    a = BigQueryAdapter()
    token = a.probe_freshness(_FakeConn(cur), [])
    assert token.value == b""
    assert cur.executions == []


# --- probe_schema -----------------------------------------------------------


def test_probe_schema_walks_columns():
    cur = _FakeCursor(
        [
            (
                "INFORMATION_SCHEMA.COLUMNS",
                [
                    ("id", "INT64", "NO"),
                    ("label", "STRING", "YES"),
                ],
            ),
        ]
    )
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog="acme", schema="events", name="orders")]
    fp = a.probe_schema(_FakeConn(cur), tables)
    assert isinstance(fp, SchemaFingerprint)
    assert fp.value


def test_probe_schema_fingerprint_changes_on_column_set():
    a = BigQueryAdapter()
    tables = [QualifiedTable(catalog="acme", schema="events", name="orders")]
    cur1 = _FakeCursor([("INFORMATION_SCHEMA.COLUMNS", [("id", "INT64", "NO")])])
    cur2 = _FakeCursor(
        [
            (
                "INFORMATION_SCHEMA.COLUMNS",
                [("id", "INT64", "NO"), ("label", "STRING", "YES")],
            )
        ]
    )
    assert (
        a.probe_schema(_FakeConn(cur1), tables).value
        != a.probe_schema(_FakeConn(cur2), tables).value
    )


# --- list_schema ------------------------------------------------------------


def test_list_schema_uses_session_defaults():
    cur = _FakeCursor(
        [
            ("@@project_id, @@dataset_id", ("acme", "events")),
            (
                "INFORMATION_SCHEMA.TABLES",
                [
                    ("acme", "events", "orders", "id", "INT64", "NO"),
                    ("acme", "events", "orders", "amount", "FLOAT64", "YES"),
                    ("acme", "events", "products", "sku", "STRING", "NO"),
                ],
            ),
        ]
    )
    a = BigQueryAdapter()
    schema = a.list_schema(_FakeConn(cur))
    by_name = {t.name: t for t in schema}
    assert set(by_name) == {"orders", "products"}
    orders = by_name["orders"]
    assert [c.name for c in orders.columns] == ["id", "amount"]
    assert orders.columns[0].nullable is False
    assert orders.columns[1].nullable is True


def test_list_schema_empty_when_no_session_defaults():
    cur = _FakeCursor([("@@project_id, @@dataset_id", (None, None))])
    a = BigQueryAdapter()
    assert a.list_schema(_FakeConn(cur)) == []

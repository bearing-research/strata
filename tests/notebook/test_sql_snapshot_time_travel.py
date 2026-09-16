"""``# @cache snapshot`` on warehouses with time travel. Item 25.

No live warehouse: a fake Snowflake connection answers the adapter's queries
and serves a table whose rows carry the time they landed, so a query pinned
``AT (TIMESTAMP => ...)`` sees exactly the rows that existed then.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

from strata.notebook.sql.adapter import QualifiedTable
from strata.notebook.sql.drivers.bigquery import BigQueryAdapter
from strata.notebook.sql.drivers.snowflake import SnowflakeAdapter

T0 = datetime(2026, 9, 15, 10, 0, tzinfo=UTC)
_AT = re.compile(r"AT \(TIMESTAMP => CAST\('([^']+)' AS TIMESTAMPTZ\)\)")


class Warehouse:
    """One ``events`` table whose rows remember when they landed."""

    def __init__(self):
        self.now = T0
        self.rows: list[tuple[datetime, int]] = []
        self.retention_days = 1
        self.queries: list[str] = []

    def insert(self, *ids: int) -> None:
        self.rows.extend((self.now, i) for i in ids)

    def connect(self):
        return _Conn(self)


class _Cursor:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self._row = None
        self._table = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=(), parameters=None):
        if "CURRENT_DATABASE()" in sql:
            self._row = ("DB", "PUBLIC")
        elif "CURRENT_TIMESTAMP()" in sql:
            self._row = (self.warehouse.now,)
        elif "RETENTION_TIME" in sql:
            self._row = (self.warehouse.retention_days,)
        else:
            self.warehouse.queries.append(sql)
            match = _AT.search(sql)
            at = datetime.fromisoformat(match.group(1)) if match else None
            ids = [i for landed, i in self.warehouse.rows if at is None or landed <= at]
            self._table = pa.table({"id": ids})

    def fetchone(self):
        return self._row

    def fetch_arrow_table(self):
        return self._table

    def close(self):
        pass


class _Conn:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse

    def cursor(self):
        return _Cursor(self.warehouse)

    def close(self):
        pass


@pytest.fixture
def warehouse(monkeypatch):
    warehouse = Warehouse()
    monkeypatch.setattr(SnowflakeAdapter, "open", lambda self, spec, read_only: warehouse.connect())
    return warehouse


def _notebook(tmp_path: Path, source: str):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb = create_notebook(tmp_path, "snapshots")
    add_cell_to_notebook(nb, "c1", language="sql")
    write_cell(nb, "c1", source)
    toml = nb / "notebook.toml"
    toml.write_text(
        toml.read_text()
        + '\n[connections.wh]\ndriver = "snowflake"\naccount = "acme"\nuser = "reader"\n'
        'database = "DB"\nschema = "PUBLIC"\n'
    )
    return NotebookSession(parse_notebook(nb), nb), source


def _ids(session, result) -> list[int]:
    ref = result["artifact_uri"].removeprefix("strata://artifact/")
    artifact_id, _, version = ref.partition("@v=")
    blob = session.get_artifact_manager().load_artifact_data(artifact_id, int(version))
    return pa.ipc.open_stream(blob).read_all().column("id").to_pylist()


SOURCE = "# @sql connection=wh\n# @cache snapshot\nSELECT id FROM events\n"


async def test_a_snapshot_cell_hits_on_rerun_and_keeps_its_state_after_new_rows(
    tmp_path, warehouse
):
    from strata.notebook.sql.cell_executor import execute_sql_cell

    warehouse.insert(1, 2)
    warehouse.now = T0 + timedelta(minutes=5)
    session, source = _notebook(tmp_path, SOURCE)

    first = await execute_sql_cell(session, "c1", source)
    assert first["success"], first["error"]
    assert first["cache_hit"] is False
    pinned = warehouse.now.isoformat()
    assert first["snapshot_at"] == pinned
    assert (
        f"State as of {pinned}; queryable until {(warehouse.now + timedelta(days=1)).isoformat()}"
        in first["stdout"]
    )
    assert f"AT (TIMESTAMP => CAST('{pinned}'" in warehouse.queries[-1]

    warehouse.now = T0 + timedelta(hours=1)
    warehouse.insert(3)
    warehouse.now = T0 + timedelta(hours=2)

    again = await execute_sql_cell(session, "c1", source)
    assert again["success"], again["error"]
    assert again["cache_hit"] is True
    assert again["snapshot_at"] == pinned
    assert _ids(session, again) == [1, 2], "the hit is the state as of the pin"
    assert "queryable until" in again["stdout"]
    assert len(warehouse.queries) == 1, "a hit reads nothing from the warehouse"


async def test_the_pinned_query_replays_the_same_rows_after_new_ones_land(tmp_path, warehouse):
    from strata.notebook.sql.analyzer import analyze_sql_cell
    from strata.notebook.sql.cell_executor import _execute_query

    warehouse.insert(1, 2)
    at = (T0 + timedelta(minutes=5)).isoformat()
    warehouse.now = T0 + timedelta(hours=1)
    warehouse.insert(3)
    adapter = SnowflakeAdapter()
    analysis = analyze_sql_cell(SOURCE, dialect=adapter.sqlglot_dialect)

    replayed = _execute_query(adapter, None, analysis, (), at=at)
    current = _execute_query(adapter, None, analysis, ())

    assert replayed.column("id").to_pylist() == [1, 2]
    assert current.column("id").to_pylist() == [1, 2, 3]


async def test_a_rerun_takes_a_new_snapshot(tmp_path, warehouse):
    from strata.notebook.sql.cell_executor import execute_sql_cell

    warehouse.insert(1)
    warehouse.now = T0 + timedelta(minutes=1)
    session, source = _notebook(tmp_path, SOURCE)
    first = await execute_sql_cell(session, "c1", source)
    warehouse.insert(2)
    warehouse.now = T0 + timedelta(minutes=2)

    rerun = await execute_sql_cell(session, "c1", source, use_cache=False)

    assert rerun["snapshot_at"] != first["snapshot_at"]
    assert _ids(session, rerun) == [1, 2]


async def test_changing_the_query_takes_a_new_snapshot(tmp_path, warehouse):
    from strata.notebook.sql.cell_executor import execute_sql_cell

    warehouse.insert(1)
    warehouse.now = T0 + timedelta(minutes=1)
    session, source = _notebook(tmp_path, SOURCE)
    first = await execute_sql_cell(session, "c1", source)
    warehouse.now = T0 + timedelta(minutes=2)

    edited = SOURCE.replace("SELECT id", "SELECT id AS event_id")
    changed = await execute_sql_cell(session, "c1", edited)

    assert changed["cache_hit"] is False
    assert changed["snapshot_at"] != first["snapshot_at"]


class TestTheAdapters:
    def test_snowflake_pins_every_table_but_not_a_cte(self):
        pinned = SnowflakeAdapter().pin_query(
            "WITH recent AS (SELECT * FROM db.s.orders) "
            "SELECT * FROM recent JOIN s.items AS i ON i.id = recent.id WHERE i.x > ?",
            "2026-09-15T10:00:00+00:00",
        )

        assert pinned.count("AT (TIMESTAMP => CAST('2026-09-15T10:00:00+00:00'") == 2
        assert "FROM recent AT" not in pinned
        assert pinned.endswith("> ?")

    def test_snowflake_horizon_is_the_shortest_retention(self):
        class Cursor(_Cursor):
            days = iter([7, 1])

            def execute(self, sql, params=(), parameters=None):
                if "RETENTION_TIME" in sql:
                    self._row = (next(self.days),)
                else:
                    super().execute(sql, params)

        conn = _Conn(Warehouse())
        conn.cursor = lambda: Cursor(conn.warehouse)
        tables = [QualifiedTable("DB", "PUBLIC", "A"), QualifiedTable("DB", "PUBLIC", "B")]

        horizon = SnowflakeAdapter().retention_until(conn, tables, "2026-09-15T10:00:00+00:00")

        assert horizon == "2026-09-16T10:00:00+00:00"

    def test_bigquery_pins_with_system_time_and_states_the_guaranteed_window(self):
        adapter = BigQueryAdapter()

        pinned = adapter.pin_query(
            "SELECT id FROM `p.d.t` WHERE x > ?", "2026-09-15T10:00:00+00:00"
        )

        assert "FOR SYSTEM_TIME AS OF CAST('2026-09-15T10:00:00+00:00' AS TIMESTAMP)" in pinned
        assert adapter.retention_until(None, [], "2026-09-15T10:00:00+00:00") == (
            "2026-09-17T10:00:00+00:00"
        )


class TestWhichTablesAPinCovers:
    """A cell's provenance says it read one moment, so every table it reads
    must carry that moment — and no table the author pinned themselves may be
    moved to another one."""

    _AT = "SELECT * FROM t AT (TIMESTAMP => CAST('2020-01-01T00:00:00+00:00' AS TIMESTAMPTZ))"

    def test_a_base_table_sharing_a_name_with_an_inner_cte_is_pinned(self):
        from strata.notebook.sql.time_travel import pin_tables

        pinned = pin_tables(
            "SELECT * FROM orders WHERE x IN (WITH orders AS (SELECT 1 AS x) SELECT x FROM orders)",
            "snowflake",
            self._AT,
            "when",
        )

        assert pinned.startswith("SELECT * FROM orders AT (TIMESTAMP =>"), pinned

    def test_a_cte_is_not_pinned_but_the_table_it_reads_is(self):
        from strata.notebook.sql.time_travel import pin_tables

        pinned = pin_tables(
            "WITH recent AS (SELECT * FROM base) SELECT * FROM recent",
            "snowflake",
            self._AT,
            "when",
        )

        assert "base AT (TIMESTAMP =>" in pinned
        assert "recent AT" not in pinned

    def test_the_moment_the_author_asked_for_is_kept(self):
        from strata.notebook.sql.time_travel import pin_tables

        pinned = pin_tables("SELECT * FROM a AT (OFFSET => -300)", "snowflake", self._AT, "when")

        assert pinned == "SELECT * FROM a AT (OFFSET => -300)"

"""E2E tests: ``@table`` lake inputs through live notebook execution.

The headline behavior under test: a cell declaring an Iceberg table input
goes stale when new data lands in the table (the snapshot id is part of
the cell's provenance), and re-runs against the new snapshot — while an
unchanged table cache-hits.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)

if sys.platform == "win32":
    pytest.skip(
        "pyiceberg + pyarrow LocalFileSystem path handling broken on Windows",
        allow_module_level=True,
    )


@pytest.fixture
def setup():
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


def _build_warehouse(tmp: Path):
    """One-table Iceberg warehouse; returns (table, table_uri)."""
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    warehouse_path = tmp / "warehouse"
    warehouse_path.mkdir()
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )
    catalog.create_namespace("db")
    schema = Schema(NestedField(1, "id", LongType(), required=False))
    table = catalog.create_table("db.events", schema)
    table.append(pa.table({"id": pa.array([1, 2, 3], type=pa.int64())}))
    return table, f"file://{warehouse_path}#db.events"


def _append_row(table) -> None:
    import pyarrow as pa

    table.append(pa.table({"id": pa.array([99], type=pa.int64())}))


class TestTableInjection:
    def test_table_variables_injected(self, setup):
        client, tmp = setup
        table, uri = _build_warehouse(tmp)
        snapshot_id = table.current_snapshot().snapshot_id

        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                f"# @table events {uri}\nsnap = events_snapshot\nuri_str = events",
            )
            .add_cell("c2", "report = f'{uri_str}:{snap}'", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_output"
                outputs = result["payload"]["outputs"]
                assert str(outputs["snap"]["preview"]) == str(snapshot_id)
                assert outputs["uri_str"]["preview"] == uri


class TestTableStaleness:
    def test_append_makes_cell_recompute(self, setup):
        """New data in the lake → the cell re-runs and sees the new snapshot."""
        client, tmp = setup
        table, uri = _build_warehouse(tmp)
        first_snapshot = table.current_snapshot().snapshot_id

        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                f"# @table events {uri}\nsnap = events_snapshot",
            )
            .add_cell("c2", "downstream = snap + 0", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                first = execute_cell_and_wait(ws, "c1")
                assert str(first["payload"]["outputs"]["snap"]["preview"]) == str(first_snapshot)

                # Unchanged table: re-execute is a cache hit (cached
                # results don't repeat per-variable outputs)
                again = execute_cell_and_wait(ws, "c1")
                assert again["payload"]["cache_hit"] is True

                # New data lands → snapshot moves → provenance changes →
                # the same execute request recomputes against the new snapshot
                _append_row(table)
                new_snapshot = table.current_snapshot().snapshot_id
                assert new_snapshot != first_snapshot

                rerun = execute_cell_and_wait(ws, "c1")
                assert rerun["payload"]["cache_hit"] is False
                assert str(rerun["payload"]["outputs"]["snap"]["preview"]) == str(new_snapshot)

    def test_pinned_table_ignores_append(self, setup):
        """A snapshot pin freezes the cell to that snapshot forever."""
        client, tmp = setup
        table, uri = _build_warehouse(tmp)
        pinned = table.current_snapshot().snapshot_id

        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                f"# @table events {uri} snapshot={pinned}\nsnap = events_snapshot",
            )
            .add_cell("c2", "downstream = snap + 0", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                first = execute_cell_and_wait(ws, "c1")
                assert str(first["payload"]["outputs"]["snap"]["preview"]) == str(pinned)

                _append_row(table)

                again = execute_cell_and_wait(ws, "c1")
                assert again["payload"]["cache_hit"] is True

    def test_table_cell_stays_ready_after_downstream_runs(self, setup):
        """An executed ``@table`` cell must stay READY across a staleness
        recompute, including after a downstream cell runs.

        ``compute_staleness`` omitted the ``@table`` snapshot fingerprint
        from the provenance hash that execution *did* fold in, so the
        cell's stored artifacts were keyed under a hash the staleness
        lookup never reproduced. The cell (and everything downstream)
        resolved to IDLE — in run-all this surfaced as every completed
        cell flipping back to grey, leaving only the last cell green.
        """
        from strata.notebook.models import CellStatus

        client, tmp = setup
        table, uri = _build_warehouse(tmp)

        nb = (
            NotebookBuilder(tmp)
            .add_cell(
                "c1",
                f"# @table events {uri}\nsnap = events_snapshot",
            )
            .add_cell("c2", "downstream = snap + 0", after="c1")
        )

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")

            # A fresh staleness recompute (what every post-execution
            # broadcast triggers) must keep the @table cell green.
            staleness = session.compute_staleness()
            assert staleness["c1"].status == CellStatus.READY
            assert staleness["c2"].status == CellStatus.READY


class TestTableErrors:
    def test_unresolvable_table_fails_with_clear_error(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell(
            "c1",
            f"# @table gone file://{tmp / 'missing'}#db.gone\nx = gone_snapshot",
        )

        with open_notebook_session(client, nb.path) as (sid, _session):
            with ws_connect(client, sid) as ws:
                result = execute_cell_and_wait(ws, "c1")
                assert result["type"] == "cell_error"
                assert "@table gone" in result["payload"]["error"]

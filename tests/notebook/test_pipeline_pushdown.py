"""Phase-0 optimizer: predicate pushdown into the Iceberg scan.

Two proofs. The unit test shows the pass lifts a node's literal ``@table``
filter onto the scan's TableRef, leaving identity (source/provenance) intact.
The integration test shows the lifted predicate, handed to the real
``ReadPlanner``, prunes files/row groups — the payoff is *reading less data*,
asserted as a correctness property (fewer row-group tasks, same rows), not a
timing one.
"""

from __future__ import annotations

import sys

import pyarrow as pa
import pytest

from strata.filters import Filter, FilterOp
from strata.pipeline import (
    PipelineIR,
    PipelineNode,
    ScanFilter,
    TableRef,
    pushdown_scan_filters,
)


def _ir_with_node(source: str, *, table: str = "events") -> PipelineIR:
    node = PipelineNode(
        id="scan",
        name="scan",
        kind="python",
        compute_target="container",
        source=source,
        tables=[TableRef(name=table, uri="file:///wh#test_db.events")],
    )
    return PipelineIR(pipeline_id="p", pipeline_name="P", env_hash="e", nodes=[node])


# ---------------------------------------------------------------------------
# Unit: the pass lifts the filter and preserves identity
# ---------------------------------------------------------------------------


def test_literal_filter_lifted_onto_scan():
    ir = pushdown_scan_filters(_ir_with_node("hot = events[events.value >= 200]\n"))
    table = ir.nodes[0].tables[0]
    assert table.filters == [ScanFilter(column="value", op=">=", value=200)]


@pytest.mark.parametrize(
    ("expr", "column", "op", "value"),
    [
        ("events[events.id == 3]", "id", "=", 3),
        ("events[events.value < 1.5]", "value", "<", 1.5),
        ("events[events.name > 'm']", "name", ">", "m"),
    ],
)
def test_recognizes_each_comparison(expr, column, op, value):
    ir = pushdown_scan_filters(_ir_with_node(f"out = {expr}\n"))
    assert ir.nodes[0].tables[0].filters == [ScanFilter(column=column, op=op, value=value)]


def test_source_and_provenance_are_untouched():
    src = "hot = events[events.value >= 200]\n"
    ir_in = _ir_with_node(src)
    prov_before = ir_in.nodes[0].static_provenance
    ir_out = pushdown_scan_filters(ir_in)
    # Only the scan gained metadata; the code the user wrote is unchanged.
    assert ir_out.nodes[0].source == src
    assert ir_out.nodes[0].static_provenance == prov_before


def test_no_filter_returns_ir_unchanged_identity():
    ir_in = _ir_with_node("total = len(events)\n")
    ir_out = pushdown_scan_filters(ir_in)
    assert ir_out is ir_in  # nothing lifted -> same object, no copy


def test_filter_on_non_table_variable_ignored():
    # `df` is not a @table input, so its mask must not be lifted.
    ir = pushdown_scan_filters(_ir_with_node("hot = df[df.value >= 200]\n"))
    assert ir.nodes[0].tables[0].filters == []


# ---------------------------------------------------------------------------
# Integration: the lifted predicate prunes real row groups
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_group_warehouse(tmp_path):
    """An Iceberg table in three files: value ranges 0-99, 100-199, 200-299."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{warehouse / 'catalog.db'}",
        warehouse=str(warehouse),
    )
    catalog.create_namespace("test_db")
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "name", StringType(), required=False),
    )
    table = catalog.create_table("test_db.events", schema)
    for lo in (0, 100, 200):  # one append per file -> three prunable row groups
        table.append(
            pa.table(
                {
                    "id": pa.array(range(lo, lo + 100), type=pa.int64()),
                    "value": pa.array([float(i) for i in range(lo, lo + 100)], type=pa.float64()),
                    "name": pa.array([f"item_{i}" for i in range(lo, lo + 100)], type=pa.string()),
                }
            )
        )
    return f"file://{warehouse}#test_db.events"


def test_pushed_filter_prunes_row_groups(multi_group_warehouse, strata_config):
    from strata.planner import ReadPlanner

    planner = ReadPlanner(strata_config)

    # Baseline: no predicate -> every file/row-group is a task.
    baseline = planner.plan(multi_group_warehouse)
    assert len(baseline.tasks) == 3

    # The optimizer lifts `events[events.value >= 200]` onto the scan...
    ir = pushdown_scan_filters(_ir_with_node("hot = events[events.value >= 200]\n"))
    lifted = ir.nodes[0].tables[0].filters
    assert lifted == [ScanFilter(column="value", op=">=", value=200)]

    # ...and handed to the real planner, it prunes the two out-of-range files.
    planner_filters = [Filter(column=f.column, op=FilterOp(f.op), value=f.value) for f in lifted]
    pushed = planner.plan(multi_group_warehouse, filters=planner_filters)
    assert len(pushed.tasks) < len(baseline.tasks)
    assert len(pushed.tasks) == 1  # only the values 200-299 file survives

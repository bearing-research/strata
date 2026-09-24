"""Property test: row-group pruning never drops a matching row.

Invariant 2 (conservative pruning) says a row group may be skipped only
when no row in it can match. This checks that directly, for random
columns written through pyarrow's real Parquet writer. It compares
ReadPlanner._should_prune_row_group, fed the file's real statistics,
with an exact evaluation of the filter on the rows. A null never
matches; NaN compares as IEEE says (NaN != x is true).

Needs Hypothesis, which is not a project dependency:

    uv run --with hypothesis pytest formal/test_pruning_properties.py
"""

from __future__ import annotations

import io
import math
import operator
import os
from datetime import UTC, datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.filters import Filter, FilterOp
from strata.planner import ReadPlanner

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_PY_OP = {
    FilterOp.EQ: operator.eq,
    FilterOp.NE: operator.ne,
    FilterOp.LT: operator.lt,
    FilterOp.LE: operator.le,
    FilterOp.GT: operator.gt,
    FilterOp.GE: operator.ge,
}

# Each domain mixes a small pool of boundary values with the full range, so
# that equal values, NaN next to a number, and min == max come up often.
_floats = st.one_of(
    st.sampled_from([0.0, -0.0, 1.0, 5.0, math.nan, math.inf, -math.inf]),
    st.floats(allow_nan=True, allow_infinity=True, width=64),
)
_ints = st.one_of(
    st.sampled_from([0, 1, 5, -1, 2**53, 2**53 + 1, 2**63 - 1, -(2**63)]),
    st.integers(min_value=-(2**63), max_value=2**63 - 1),
)
_texts = st.one_of(
    st.sampled_from(["", "a", "b", "a" * 70, "a" * 70 + "b", "\x00", "é", "\U0010ffff"]),
    st.text(max_size=80),
)
_times = st.one_of(
    st.sampled_from([datetime(2024, 1, 1), datetime(2024, 1, 1, 0, 0, 0, 1)]),
    st.datetimes(min_value=datetime(1970, 1, 1), max_value=datetime(2200, 1, 1)),
)
_decimals = st.one_of(
    st.sampled_from([Decimal("0.00"), Decimal("1.50"), Decimal("-1.50")]),
    st.decimals(
        min_value=Decimal("-99999.99"), max_value=Decimal("99999.99"), places=2, allow_nan=False
    ),
)

# (arrow type, value strategy, filter-value strategy). The filter value is
# drawn from the column's own domain plus a same-kind neighbour, which is
# where boundary bugs live.
_COLUMNS = [
    (pa.float64(), _floats, _floats),
    (pa.int64(), _ints, _ints),
    (pa.int64(), _ints, _floats),
    (pa.string(), _texts, _texts),
    (pa.timestamp("us"), _times, _times),
    (pa.decimal128(7, 2), _decimals, _decimals),
]

_PLANNER = ReadPlanner.__new__(ReadPlanner)  # the method only uses _convert_stats


@st.composite
def cases(draw):
    arrow_type, values, filter_values = draw(st.sampled_from(_COLUMNS))
    # A small per-example pool of values, reused across rows, so that row
    # groups repeat values (min == max) and mix NaN or null with them.
    pool = draw(st.lists(values, min_size=1, max_size=3))
    element = st.one_of(st.none(), st.sampled_from(pool), values)
    column = draw(st.lists(element, min_size=1, max_size=12))
    op = draw(st.sampled_from(list(FilterOp)))
    present = [v for v in column if v is not None]
    # Half the time compare against a value the column holds: boundaries
    # (min, max, min == max) are where pruning goes wrong.
    if present and draw(st.booleans()):
        value = draw(st.sampled_from(present))
    else:
        value = draw(filter_values)
    row_group_size = draw(st.integers(min_value=1, max_value=4))
    return arrow_type, column, op, value, row_group_size


def _matches(rows: pa.Array, op: FilterOp, value) -> bool:
    """Does any row satisfy ``row <op> value``? Exact Python comparison;
    a null never matches, NaN compares as IEEE says."""
    compare = _PY_OP[op]
    for row in rows.to_pylist():
        if row is None:
            continue
        try:
            if compare(row, value):
                return True
        except TypeError:
            return True  # incomparable: pruning must not have relied on it
    return False


def _is_finding_3(rows: pa.Array, op: FilterOp) -> bool:
    """``!=`` on a row group with NaN in it: NaN is left out of the stats."""
    return op is FilterOp.NE and any(
        isinstance(x, float) and math.isnan(x) for x in rows.to_pylist()
    )


@settings(
    max_examples=int(os.environ.get("PRUNING_EXAMPLES", "3000")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(cases())
def test_a_pruned_row_group_has_no_matching_row(case):
    arrow_type, column, op, value, row_group_size = case
    table = pa.table({"v": pa.array(column, type=arrow_type)})
    buf = io.BytesIO()
    pq.write_table(table, buf, row_group_size=row_group_size)
    parquet = pq.ParquetFile(io.BytesIO(buf.getvalue()))

    f = Filter("v", op, value)
    for i in range(parquet.num_row_groups):
        if _PLANNER._should_prune_row_group(parquet.metadata.row_group(i), [(0, f)]):
            rows = parquet.read_row_group(i).column("v").combine_chunks()
            if _is_finding_3(rows, op):
                continue  # known: test_artifact_counterexamples::test_nan_ne_pruning
            assert not _matches(rows, op, value), (
                f"pruned row group {i} holds a row matching v {op.value} {value!r}: "
                f"{rows.to_pylist()}"
            )


def test_timestamps_with_timezones_are_not_pruned_wrongly():
    """A fixed example alongside the random ones: tz-aware filter values."""
    table = pa.table({"v": pa.array([datetime(2024, 1, 1, tzinfo=UTC)], pa.timestamp("us", "UTC"))})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    rg = pq.ParquetFile(io.BytesIO(buf.getvalue())).metadata.row_group(0)
    f = Filter("v", FilterOp.EQ, datetime(2024, 1, 1, tzinfo=UTC))
    assert not _PLANNER._should_prune_row_group(rg, [(0, f)])

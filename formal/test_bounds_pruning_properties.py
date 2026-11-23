"""Property test: pruning by manifest bounds never drops a live row (PR-2, strings and NaN).

A real Iceberg table (local SQL catalog) gets batches of rows with a string and a double
column, each batch sorted by one of them so its row groups have narrow ranges, and
sometimes equality delete files keyed on the string, with the bounds and null counts a
real writer gives them. Each batch is written either by pyiceberg, whose manifests carry
no NaN counts, or as a JVM writer does, with ``nan_value_counts`` for the double, which
pyiceberg's evaluator then prunes by. Iceberg truncates a string bound to 16 characters
(the upper one incremented, or dropped when it cannot be), which is where pruning can go
wrong: data files skipped by their manifest bounds (pyiceberg's evaluator), and a delete
file skipped for a row group whose Parquet range it misses (``iceberg.equality.may_apply``).

The oracle is plain Python over the live rows: an unfiltered scan returns exactly them; a
filtered scan prunes files and row groups only, so it returns every live row that matches
and nothing that is not live. NaN compares as IEEE says; a null never matches.

    uv run --with hypothesis pytest formal/test_bounds_pruning_properties.py
"""

from __future__ import annotations

import math
import operator
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata import planner as planner_module
from strata.cache.disk import CachedFetcher
from strata.config import StrataConfig
from strata.filters import Filter, FilterOp
from strata.planner import ReadPlanner
from tests.conftest import _reset_process_globals  # noqa: F401  (autouse)
from tests.iceberg_fixtures import commit_files

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, event, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)

EXAMPLES = int(os.environ.get("BOUNDS_PRUNING_EXAMPLES", "40"))
_FIELD_ID = b"PARQUET:field_id"
S, F = 2, 3  # field ids

_MAX = "\U0010ffff"  # a bound ending in it cannot be incremented
_STRINGS = [
    "",
    "a",
    "ab",
    "b",
    "a" * 15,
    "a" * 16,
    "a" * 17,
    "a" * 16 + "b",
    "a" * 16 + _MAX,
    "a" * 15 + _MAX + "z",
    _MAX * 17,
    "é" * 17,
    "日本" * 9,
    "z" * 20,
]
_STRING = st.one_of(
    st.sampled_from(_STRINGS),
    st.text(alphabet=st.sampled_from(["a", "b", "é", _MAX]), max_size=20),
)
_FLOATS = [math.nan, math.inf, -math.inf, 0.0, -0.0, 1.0, -1.0, 1e300, 5e-324]
# Weighted to the pool and to NaN: a row group whose finite values all equal the filter's,
# beside a NaN Parquet leaves out of its range, is the case a NaN guard exists for.
_FLOAT = st.one_of(
    st.just(math.nan),
    st.sampled_from(_FLOATS),
    st.sampled_from(_FLOATS),
    st.floats(allow_nan=True, width=64),
)
_ROW = st.tuples(st.one_of(st.none(), _STRING), st.one_of(st.none(), _FLOAT))
_OPS = {
    FilterOp.EQ: operator.eq,
    FilterOp.NE: operator.ne,
    FilterOp.LT: operator.lt,
    FilterOp.LE: operator.le,
    FilterOp.GT: operator.gt,
    FilterOp.GE: operator.ge,
}


def _key(value):
    """Rows compare by value, with NaN equal to itself."""
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    return (type(value).__name__, value)


def _rows_key(rows) -> Counter:
    return Counter(tuple((k, _key(v)) for k, v in sorted(r.items())) for r in rows)


@st.composite
def _case(draw):
    batches = draw(st.lists(st.lists(_ROW, min_size=1, max_size=8), min_size=1, max_size=4))
    order = [draw(st.sampled_from(["s", "f", None])) for _ in batches]
    nan_counts = [draw(st.booleans()) for _ in batches]
    # Values the data holds: a filter or key equal to one is where a range edge matters.
    held = {
        "s": sorted({r[0] for b in batches for r in b if r[0] is not None}),
        "f": [r[1] for b in batches for r in b if r[1] is not None],
    }

    def drawn(column, pool):
        return pool | st.sampled_from(held[column]) if held[column] else pool

    # Equality deletes, each after some batches: (batches before it, its keys).
    deletes = draw(
        st.lists(
            st.tuples(
                st.integers(1, len(batches)),
                st.lists(st.none() | drawn("s", _STRING), min_size=1, max_size=3),
            ),
            max_size=2,
        )
    )
    column = draw(st.sampled_from(["s", "f"]))
    value = draw(drawn(column, _STRING if column == "s" else _FLOAT))
    op = draw(st.sampled_from(list(_OPS)))
    deletes = sorted(deletes, key=lambda d: d[0])
    return batches, order, nan_counts, deletes, Filter(column, op, value)


def _sorted(rows, by):
    if by is None:
        return rows
    index = 0 if by == "s" else 1

    def key(row):
        v = row[index]
        if v is None:
            return (2, 0)
        if isinstance(v, float) and math.isnan(v):
            return (1, 0)
        return (0, v)

    return sorted(rows, key=key)


def _statistics(table, path: Path):
    from pyiceberg.io.pyarrow import (
        compute_statistics_plan,
        data_file_statistics_from_parquet_metadata,
        parquet_path_to_id_mapping,
    )

    schema = table.schema()
    return data_file_statistics_from_parquet_metadata(
        parquet_metadata=pq.read_metadata(path),
        stats_columns=compute_statistics_plan(schema, table.metadata.properties),
        parquet_column_mapping=parquet_path_to_id_mapping(schema),
    ).to_serialized_dict()


def _counted_data_file(table, root: Path, n: int, ids: list[int], rows):
    """A data file as a JVM writer commits it: its manifest entry counts the double's NaNs."""
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    path = root / f"data-{n}.parquet"
    fields = [
        pa.field(name, kind).with_metadata({_FIELD_ID: str(fid).encode()})
        for name, kind, fid in (
            ("id", pa.int64(), 1),
            ("s", pa.string(), S),
            ("f", pa.float64(), F),
        )
    ]
    columns = [ids, [r[0] for r in rows], [r[1] for r in rows]]
    pq.write_table(pa.table(columns, schema=pa.schema(fields)), path, row_group_size=2)
    stats = _statistics(table, path)
    stats["nan_value_counts"] = {
        F: sum(1 for r in rows if isinstance(r[1], float) and math.isnan(r[1]))
    }
    data = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=path.as_uri(),
        file_format=FileFormat.PARQUET,
        partition=Record(),
        file_size_in_bytes=path.stat().st_size,
        **stats,
    )
    data.spec_id = table.metadata.default_spec_id
    commit_files(table, data)


def _equality_delete(table, root: Path, n: int, keys: list[str | None]):
    """An equality delete file on ``s`` with the bounds and counts a writer would give it."""
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record

    path = root / f"eq-{n}.parquet"
    field = pa.field("s", pa.string()).with_metadata({_FIELD_ID: str(S).encode()})
    pq.write_table(pa.table({"s": pa.array(keys, pa.string())}, schema=pa.schema([field])), path)
    delete = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES,
        file_path=path.as_uri(),
        file_format=FileFormat.PARQUET,
        partition=Record(),
        file_size_in_bytes=path.stat().st_size,
        equality_ids=[S],
        **_statistics(table, path),
    )
    delete.spec_id = table.metadata.default_spec_id
    commit_files(table, delete)
    return delete


@settings(
    max_examples=EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(case=_case())
# Each edge random search reaches rarely: a row group [1.0, NaN] whose Parquet range is
# [1.0, 1.0] under f != 1.0; a key of exactly the 16 characters bounds keep; a delete
# file whose bounds miss a row group; a file with NaN counts and finite rows.
@example(case=([[("x", 1.0), ("y", math.nan)]], [None], [False], [], Filter("f", FilterOp.NE, 1.0)))
@example(
    case=([[("a" * 16, None)]], ["s"], [False], [(1, ["a" * 16])], Filter("s", FilterOp.EQ, ""))
)
@example(
    case=(
        [[("a", None), ("b", None), ("b", None)]],
        ["s"],
        [False],
        [(1, ["a"])],
        Filter("s", FilterOp.EQ, "b"),
    )
)
@example(case=([[("x", math.nan), ("y", 2.0)]], [None], [True], [], Filter("f", FilterOp.GT, 1.0)))
def test_pruning_by_bounds_keeps_every_live_row(case):
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    batches, order, nan_counts, deletes, flt = case
    may_apply = planner_module.may_apply
    root = Path(tempfile.mkdtemp(prefix="bounds-"))
    warehouse = root / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
    )
    planner = None
    try:
        catalog.create_namespace("db")
        table = catalog.create_table(
            "db.t",
            schema=Schema(
                NestedField(1, "id", LongType(), required=False),
                NestedField(S, "s", StringType(), required=False),
                NestedField(F, "f", DoubleType(), required=False),
            ),
            properties={"write.parquet.row-group-limit": "2"},
        )
        arrow = table.schema().as_arrow()
        # A first snapshot with no files, which a counted file's commit builds on.
        table.append(pa.table({"id": [], "s": [], "f": []}, schema=arrow))
        table = catalog.load_table("db.t")
        live: dict[int, dict] = {}
        next_id = 0
        pending = list(deletes)
        batched = zip(batches, order, nan_counts, strict=True)
        for done, (batch, by, counted) in enumerate(batched, start=1):
            rows = _sorted(batch, by)
            ids = list(range(next_id, next_id + len(rows)))
            next_id += len(rows)
            if counted:
                _counted_data_file(table, root, next_id, ids, rows)
                event("a file with NaN counts")
            else:
                table.append(
                    pa.table(
                        {
                            "id": pa.array(ids, pa.int64()),
                            "s": pa.array([r[0] for r in rows], pa.string()),
                            "f": pa.array([r[1] for r in rows], pa.float64()),
                        },
                        schema=arrow,
                    )
                )
            table = catalog.load_table("db.t")
            for i, (s, f) in zip(ids, rows, strict=True):
                live[i] = {"id": i, "s": s, "f": f}
            while pending and pending[0][0] == done:
                _, keys = pending.pop(0)
                _equality_delete(table, root, next_id, keys)
                table = catalog.load_table("db.t")
                next_id += 1
                # Null-safe equality: a null key deletes the rows whose key is null.
                doomed = {i for i, r in live.items() if r["s"] in keys}
                for i in doomed:
                    del live[i]
                event("an equality delete removed rows" if doomed else "an equality delete")

        def counted(delete, stats):
            applies = may_apply(delete, stats)
            if not applies:
                event("a delete file skipped for a row group by its bounds")
            return applies

        planner_module.may_apply = counted
        config = StrataConfig(cache_dir=root / "cache")
        planner = ReadPlanner(config)
        fetcher = CachedFetcher(config)
        uri = f"{warehouse.as_uri()}#db.t"

        def scan(filters):
            plan = planner.plan(uri, filters=filters)
            return [row for task in plan.tasks for row in fetcher.fetch(task).to_pylist()]

        # Filtered first: a filtered plan falls back to a cached unfiltered resolution,
        # which would skip pyiceberg's file pruning.
        got = _rows_key(scan([flt]))
        everything = scan(None)
        assert _rows_key(everything) == _rows_key(live.values()), (case, everything)

        assert not got - _rows_key(live.values()), ("not live", case)
        test = _OPS[flt.op]
        matching = [
            r for r in live.values() if r[flt.column] is not None and test(r[flt.column], flt.value)
        ]
        missing = _rows_key(matching) - got
        assert not missing, ("pruned matching rows", case, list(missing.elements()))
        event(f"filter on {flt.column} {flt.op.value}")
        if len(got) < len(live):
            event("the filter pruned something")
    finally:
        planner_module.may_apply = may_apply
        if planner is not None:
            planner.close()
        catalog.close()
        shutil.rmtree(root, ignore_errors=True)

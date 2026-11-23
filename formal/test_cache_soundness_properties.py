"""Stateful property test: a long-lived planner and row-group cache serve what the table holds.

Hypothesis drives a real Iceberg table (local SQL catalog, identity-partitioned on
``region``) through appends, positional and equality deletes, files registered at a
fixed path (a Hive-layout file that omits its partition column) and replaced there,
schema evolution, and drop-and-recreate at the same location. Between them it scans
through one ``ReadPlanner`` and one ``CachedFetcher`` that live for the whole run, and
sometimes restarts both on the same cache directory.

The oracle is a model of the table's live rows, independent of Strata and of
pyiceberg's reader (which refuses equality deletes). An unfiltered scan returns
exactly the live rows; a filtered scan prunes row groups only, so it returns every
live row that matches and nothing that is not live. Checks CK-1, MD-2 and PR-2
(SYSTEMS 3 and 4).

Needs Hypothesis, which is not a project dependency:

    uv run --with hypothesis pytest formal/test_cache_soundness_properties.py
"""

# Fixtures are imported from tests/ and then requested by name (or autouse).
# ruff: noqa: F811

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from strata.cache.disk import CachedFetcher
from strata.config import StrataConfig
from strata.filters import Filter, FilterOp
from strata.planner import ReadPlanner
from tests.conftest import _reset_process_globals  # noqa: F401  (autouse)
from tests.iceberg_fixtures import commit_files

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.stateful import (  # noqa: E402
    RuleBasedStateMachine,
    initialize,
    invariant,
    precondition,
    rule,
)

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="pyiceberg's local FileIO paths are broken on Windows"
)

_FIELD_ID = b"PARQUET:field_id"
REGIONS = ["eu", "us"]
SLOTS = ["a", "b"]  # fixed paths for registered files
ID, REGION = 1, 2  # field ids that never change


class Row:
    """A live row: its values by field id, and where it is (file uri, position)."""

    def __init__(self, values: dict[int, object], file: str, pos: int, seq: int) -> None:
        self.values = values
        self.file = file
        self.pos = pos
        self.seq = seq  # the data sequence number of its file


class CacheMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        self.root = Path(tempfile.mkdtemp(prefix="cache-sound-"))
        warehouse = self.root / "warehouse"
        warehouse.mkdir()
        (self.root / "ext").mkdir()
        from pyiceberg.catalog.sql import SqlCatalog

        self.catalog = SqlCatalog(
            "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=warehouse.as_uri()
        )
        self.catalog.create_namespace("db")
        self.uri = f"{warehouse.as_uri()}#db.t"
        self.config = StrataConfig(cache_dir=self.root / "cache")
        self.planner = ReadPlanner(self.config)
        self.fetcher = CachedFetcher(self.config)
        self.next_id = 0
        self.log: list[str] = []

    def teardown(self) -> None:
        for step in {entry.split()[0] for entry in self.log}:
            hypothesis.event(step)  # which steps ran, for --hypothesis-show-statistics
        self.planner.close()
        self.catalog.close()
        shutil.rmtree(self.root, ignore_errors=True)

    # --- the table -------------------------------------------------------------

    def _create(self) -> None:
        from pyiceberg.partitioning import PartitionField, PartitionSpec
        from pyiceberg.schema import Schema
        from pyiceberg.transforms import IdentityTransform
        from pyiceberg.types import IntegerType, LongType, NestedField, StringType

        schema = Schema(
            NestedField(ID, "id", LongType(), required=False),
            NestedField(REGION, "region", StringType(), required=False),
            NestedField(3, "v", IntegerType(), required=False),
        )
        spec = PartitionSpec(PartitionField(REGION, 1000, IdentityTransform(), "region"))
        self.catalog.create_table(
            "db.t",
            schema=schema,
            partition_spec=spec,
            properties={"write.parquet.row-group-limit": "3"},
        )
        self.rows: dict[int, Row] = {}  # by id
        self.slots: dict[str, object] = {}  # slot -> its registered DataFile
        # snapshot id -> (its live rows, the data files it reads)
        self.history: dict[int, tuple[dict[int, Row], set[str]]] = {}

    @initialize()
    def start(self) -> None:
        self._create()

    def _table(self):
        return self.catalog.load_table("db.t")

    def _fields(self) -> list[tuple[str, int, object]]:
        """The current schema: (name, field id, type)."""
        return [(f.name, f.field_id, f.field_type) for f in self._table().schema().fields]

    def _values(self, ids: list[int], v_base: int) -> list[dict[int, object]]:
        out = []
        for n, i in enumerate(ids):
            values: dict[int, object] = {ID: i}
            for _, field_id, _ in self._fields():
                if field_id not in (ID, REGION):
                    values[field_id] = v_base + n
            out.append(values)
        return out

    def _ids(self, n: int) -> list[int]:
        ids = list(range(self.next_id, self.next_id + n))
        self.next_id += n
        return ids

    # --- writes ----------------------------------------------------------------

    @rule(region=st.sampled_from(REGIONS), n=st.integers(1, 7), v_base=st.integers(-50, 50))
    def append(self, region, n, v_base):
        table = self._table()
        ids = self._ids(n)
        values = self._values(ids, v_base)
        arrow_schema = table.schema().as_arrow()
        columns = {}
        for name, field_id, _ in self._fields():
            if field_id == REGION:
                columns[name] = [region] * n
            else:
                columns[name] = [row.get(field_id) for row in values]
        before = {r.file for r in self.rows.values()}
        table.append(pa.table(columns, schema=arrow_schema))
        table = self._table()
        seq = table.current_snapshot().sequence_number
        for file in self._data_files(table):
            if file.file_path in before or file.file_path.startswith(self._ext_prefix()):
                continue
            stored = pq.read_table(file.file_path.removeprefix("file://"), columns=["id"])
            for pos, i in enumerate(stored.column(0).to_pylist()):
                if i in ids and i not in self.rows:
                    self.rows[i] = Row(
                        {**values[ids.index(i)], REGION: region}, file.file_path, pos, seq
                    )
        self._committed()
        self.log.append(f"append {region} {ids}")

    def _committed(self) -> None:
        table = self._table()
        files = {file.file_path for file in self._data_files(table)}
        self.history[table.current_snapshot().snapshot_id] = (dict(self.rows), files)

    def _ext_prefix(self) -> str:
        return (self.root / "ext").as_uri()

    def _data_files(self, table):
        from pyiceberg.manifest import ManifestContent

        snapshot = table.current_snapshot()
        if snapshot is None:
            return []
        files = []
        for manifest in snapshot.manifests(table.io):
            if manifest.content != ManifestContent.DATA:
                continue
            for entry in manifest.fetch_manifest_entry(table.io, discard_deleted=True):
                files.append(entry.data_file)
        return files

    def _write_slot(self, slot: str, ids: list[int], values: list[dict[int, object]]):
        """A Hive-layout file at the slot's fixed path: no partition column."""
        fields, arrays = [], []
        for name, field_id, field_type in self._fields():
            if field_id == REGION:
                continue
            arrow_type = pa.int64() if field_id == ID else _arrow(field_type)
            fields.append(pa.field(name, arrow_type, metadata={_FIELD_ID: str(field_id).encode()}))
            arrays.append(pa.array([row.get(field_id) for row in values], arrow_type))
        path = self.root / "ext" / f"{slot}.parquet"
        pq.write_table(
            pa.Table.from_arrays(arrays, schema=pa.schema(fields)),
            path,
            compression="none",
            use_dictionary=False,
            row_group_size=3,
        )
        return path

    @rule(
        slot=st.sampled_from(SLOTS),
        region=st.sampled_from(REGIONS),
        n=st.integers(1, 6),
        v_base=st.integers(-50, 50),
    )
    def register(self, slot, region, n, v_base):
        """Write the slot's file (again) and register it, replacing what it held."""
        from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
        from pyiceberg.typedef import Record

        table = self._table()
        ids = self._ids(n)
        values = self._values(ids, v_base)
        # A snapshot that reads the slot's file reads bytes about to be gone: no oracle.
        uri = (self.root / "ext" / f"{slot}.parquet").as_uri()
        self.history = {s: h for s, h in self.history.items() if uri not in h[1]}
        path = self._write_slot(slot, ids, values)
        new = DataFile.from_args(
            content=DataFileContent.DATA,
            file_path=path.as_uri(),
            file_format=FileFormat.PARQUET,
            partition=Record(region),
            record_count=n,
            file_size_in_bytes=path.stat().st_size,
        )
        new.spec_id = table.metadata.default_spec_id
        old = self.slots.get(slot)
        with table.transaction() as tx:
            with tx.update_snapshot().overwrite() as overwrite:
                if old is not None:
                    overwrite.delete_data_file(old)
                overwrite.append_data_file(new)
        seq = self._table().current_snapshot().sequence_number
        uri = path.as_uri()
        if old is not None:
            self.rows = {i: r for i, r in self.rows.items() if r.file != old.file_path}
        for pos, (i, row) in enumerate(zip(ids, values, strict=True)):
            self.rows[i] = Row({**row, REGION: region}, uri, pos, seq)
        self.slots[slot] = new
        self._committed()
        self.log.append(f"register {slot} {region} {ids} size={new.file_size_in_bytes}")

    @precondition(lambda self: self.rows)
    @rule(data=st.data())
    def positional_delete(self, data):
        from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
        from pyiceberg.typedef import Record

        table = self._table()
        live = sorted(self.rows)
        doomed = data.draw(st.lists(st.sampled_from(live), min_size=1, max_size=4, unique=True))
        # One delete file per partition, as writers scope them.
        for region in REGIONS:
            mine = [self.rows[i] for i in doomed if self.rows[i].values[REGION] == region]
            if not mine:
                continue
            deleted = pa.table(
                {
                    "file_path": pa.array([r.file for r in mine]),
                    "pos": pa.array([r.pos for r in mine], pa.int64()),
                },
                schema=pa.schema(
                    [
                        pa.field("file_path", pa.string(), metadata={_FIELD_ID: b"2147483546"}),
                        pa.field("pos", pa.int64(), metadata={_FIELD_ID: b"2147483545"}),
                    ]
                ),
            )
            path = self.root / "warehouse" / f"pos-{self.next_id}-{region}.parquet"
            self.next_id += 1
            pq.write_table(deleted, path)
            delete = DataFile.from_args(
                content=DataFileContent.POSITION_DELETES,
                file_path=path.as_uri(),
                file_format=FileFormat.PARQUET,
                partition=Record(region),
                record_count=deleted.num_rows,
                file_size_in_bytes=path.stat().st_size,
            )
            delete.spec_id = table.metadata.default_spec_id
            commit_files(table, delete)
            table = self._table()
        for i in doomed:
            del self.rows[i]
        self._committed()
        self.log.append(f"positional_delete {doomed}")

    @precondition(lambda self: self.rows)
    @rule(data=st.data(), on_region=st.booleans())
    def equality_delete(self, data, on_region):
        """Delete by id, or every row of a partition by its region (often filled in)."""
        from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
        from pyiceberg.typedef import Record

        table = self._table()
        region = data.draw(st.sampled_from(REGIONS))
        names = {field_id: name for name, field_id, _ in self._fields()}
        if on_region:
            keys = pa.table({names[REGION]: [region]})
            equality_ids = [REGION]
            doomed = [i for i, r in self.rows.items() if r.values[REGION] == region]
        else:
            candidates = sorted(self.rows) + [self.next_id + 1000]  # one that matches nothing
            chosen = data.draw(
                st.lists(st.sampled_from(candidates), min_size=1, max_size=4, unique=True)
            )
            keys = pa.table({names[ID]: pa.array(chosen, pa.int64())})
            equality_ids = [ID]
            doomed = [i for i in chosen if i in self.rows and self.rows[i].values[REGION] == region]
        fields = [keys.schema.field(0).with_metadata({_FIELD_ID: str(equality_ids[0]).encode()})]
        path = self.root / "warehouse" / f"eq-{self.next_id}.parquet"
        self.next_id += 1
        pq.write_table(keys.cast(pa.schema(fields)), path)
        delete = DataFile.from_args(
            content=DataFileContent.EQUALITY_DELETES,
            file_path=path.as_uri(),
            file_format=FileFormat.PARQUET,
            partition=Record(region),
            record_count=keys.num_rows,
            file_size_in_bytes=path.stat().st_size,
            equality_ids=equality_ids,
        )
        delete.spec_id = table.metadata.default_spec_id
        commit_files(table, delete)
        for i in doomed:
            del self.rows[i]
        self._committed()
        self.log.append(f"equality_delete {'region' if on_region else 'id'} {region} {doomed}")

    # --- schema evolution ---------------------------------------------------------

    @rule(name=st.sampled_from(["x", "y", "v"]))
    def add_column(self, name):
        from pyiceberg.types import LongType

        if name in {n for n, _, _ in self._fields()}:
            return
        with self._table().update_schema() as update:
            update.add_column(name, LongType())
        self.log.append(f"add_column {name}")

    @rule(data=st.data())
    def rename_column(self, data):
        movable = [n for n, field_id, _ in self._fields() if field_id not in (ID, REGION)]
        if not movable:
            return
        old = data.draw(st.sampled_from(movable))
        new = data.draw(st.sampled_from(["v", "w", "x", "y"]))
        if new in {n for n, _, _ in self._fields()}:
            return
        with self._table().update_schema() as update:
            update.rename_column(old, new)
        self.log.append(f"rename_column {old} -> {new}")

    @rule(data=st.data())
    def drop_column(self, data):
        droppable = [n for n, field_id, _ in self._fields() if field_id not in (ID, REGION)]
        if not droppable:
            return
        name = data.draw(st.sampled_from(droppable))
        with self._table().update_schema() as update:
            update.delete_column(name)
        self.log.append(f"drop_column {name}")

    @rule()
    def widen(self):
        from pyiceberg.types import IntegerType, LongType

        for name, _, field_type in self._fields():
            if isinstance(field_type, IntegerType):
                with self._table().update_schema() as update:
                    update.update_column(name, field_type=LongType())
                self.log.append(f"widen {name}")
                return

    # --- the process -----------------------------------------------------------

    @rule()
    def recreate(self):
        """Drop the table and create it again at the same location (a new table uuid)."""
        self.catalog.drop_table("db.t")
        self._create()
        self.log.append("recreate")

    @rule()
    def restart(self):
        """A new planner and fetcher on the same cache directory."""
        self.planner.close()
        self.planner = ReadPlanner(self.config)
        self.fetcher = CachedFetcher(self.config)
        self.log.append("restart")

    # --- scans -------------------------------------------------------------------

    def _scan(self, columns=None, filters=None, snapshot_id=None) -> list[dict]:
        plan = self.planner.plan(
            self.uri, columns=columns, filters=filters, snapshot_id=snapshot_id
        )
        return [row for task in plan.tasks for row in self.fetcher.fetch(task).to_pylist()]

    def _expected(self, columns=None, snapshot_id=None) -> list[dict]:
        rows = self.rows
        fields = [(n, field_id) for n, field_id, _ in self._fields()]
        if snapshot_id is not None:
            # An older snapshot reads under its own schema, as pyiceberg does.
            table = self._table()
            schema_id = table.snapshot_by_id(snapshot_id).schema_id
            fields = [(f.name, f.field_id) for f in table.schemas()[schema_id].fields]
            rows = self.history[snapshot_id][0]
        if columns is not None:
            fields = [(n, field_id) for n, field_id in fields if n in columns]
        return [{n: r.values.get(field_id) for n, field_id in fields} for r in rows.values()]

    def _check(self, got, want, what) -> None:
        got_c = Counter(tuple(sorted(r.items())) for r in got)
        want_c = Counter(tuple(sorted(r.items())) for r in want)
        assert got_c == want_c, (
            f"{what}: missing {list((want_c - got_c).elements())}, "
            f"extra {list((got_c - want_c).elements())}\n" + "\n".join(self.log)
        )

    @rule(data=st.data())
    def scan(self, data):
        names = [n for n, _, _ in self._fields()]
        columns = data.draw(
            st.one_of(st.none(), st.lists(st.sampled_from(names), min_size=1, unique=True))
        )
        self._check(self._scan(columns), self._expected(columns), f"scan {columns}")
        self.log.append(f"scan {columns}")

    @precondition(lambda self: getattr(self, "history", None))
    @rule(data=st.data())
    def time_travel(self, data):
        snapshot_id = data.draw(st.sampled_from(sorted(self.history)))
        self._check(
            self._scan(snapshot_id=snapshot_id),
            self._expected(snapshot_id=snapshot_id),
            f"snapshot {snapshot_id}",
        )
        self.log.append(f"time_travel {snapshot_id}")

    @rule(
        data=st.data(),
        op=st.sampled_from([FilterOp.EQ, FilterOp.LT, FilterOp.GE, FilterOp.NE]),
    )
    def filtered_scan(self, data, op):
        fields = [(n, field_id) for n, field_id, _ in self._fields() if field_id != REGION]
        name, field_id = data.draw(st.sampled_from(fields))
        value = data.draw(st.integers(-60, max(self.next_id, 1) + 5))
        got = self._scan(filters=[Filter(name, op, value)])
        live = self._expected()
        got_c = Counter(tuple(sorted(r.items())) for r in got)
        live_c = Counter(tuple(sorted(r.items())) for r in live)
        assert not got_c - live_c, (
            f"filtered {name} {op} {value}: not live {list((got_c - live_c).elements())}\n"
            + "\n".join(self.log)
        )
        test = _PY[op]
        matching = [r for r in live if r[name] is not None and test(r[name], value)]
        missing = Counter(tuple(sorted(r.items())) for r in matching) - got_c
        assert not missing, (
            f"filtered {name} {op} {value}: pruned matching {list(missing.elements())}\n"
            + "\n".join(self.log)
        )
        self.log.append(f"filtered_scan {name} {op} {value}")

    @invariant()
    def a_full_scan_is_the_table(self):
        if not hasattr(self, "rows"):
            return
        self._check(self._scan(), self._expected(), "full scan")


_PY = {
    FilterOp.EQ: lambda a, b: a == b,
    FilterOp.LT: lambda a, b: a < b,
    FilterOp.GE: lambda a, b: a >= b,
    FilterOp.NE: lambda a, b: a != b,
}


def _arrow(field_type) -> pa.DataType:
    from pyiceberg.types import IntegerType

    return pa.int32() if isinstance(field_type, IntegerType) else pa.int64()


CacheMachine.TestCase.settings = settings(
    max_examples=int(os.environ.get("CACHE_SOUNDNESS_EXAMPLES", "25")),
    stateful_step_count=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
test_the_cache_serves_what_the_table_holds = CacheMachine.TestCase

"""Tests for lake table inputs (``@table`` annotation) — tables.py."""

from __future__ import annotations

import sys

import pytest

from strata.config import StrataConfig
from strata.notebook.models import TableSpec
from strata.notebook.tables import fingerprint_tables, resolve_table_snapshot

if sys.platform == "win32":
    pytest.skip(
        "pyiceberg + pyarrow LocalFileSystem path handling broken on Windows",
        allow_module_level=True,
    )


@pytest.fixture
def mini_warehouse(tmp_path):
    """Tiny Iceberg warehouse with one table and one snapshot."""
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    warehouse_path = tmp_path / "warehouse"
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

    return {
        "table": table,
        "uri": f"file://{warehouse_path}#db.events",
        "snapshot_id": table.current_snapshot().snapshot_id,
    }


@pytest.fixture
def config(tmp_path):
    return StrataConfig(cache_dir=tmp_path / "cache")


class TestResolveTableSnapshot:
    def test_resolves_current_snapshot(self, mini_warehouse, config):
        spec = TableSpec(name="events", uri=mini_warehouse["uri"])
        assert resolve_table_snapshot(spec, config) == mini_warehouse["snapshot_id"]

    def test_pin_short_circuits(self, config):
        spec = TableSpec(name="events", uri="file:///nope#db.gone", snapshot_pin=42)
        # Pin wins without touching the (nonexistent) catalog
        assert resolve_table_snapshot(spec, config) == 42

    def test_unreachable_raises_value_error(self, tmp_path, config):
        spec = TableSpec(name="events", uri=f"file://{tmp_path / 'missing'}#db.gone")
        with pytest.raises(ValueError, match="cannot load table"):
            resolve_table_snapshot(spec, config)


class TestFingerprintTables:
    def test_fingerprint_is_stable(self, mini_warehouse, config):
        specs = [TableSpec(name="events", uri=mini_warehouse["uri"])]
        fp1, snaps1 = fingerprint_tables(specs, config)
        fp2, snaps2 = fingerprint_tables(specs, config)
        assert fp1 == fp2
        assert snaps1 == snaps2 == {"events": mini_warehouse["snapshot_id"]}

    def test_append_changes_fingerprint(self, mini_warehouse, config):
        import pyarrow as pa

        specs = [TableSpec(name="events", uri=mini_warehouse["uri"])]
        fp_before, _ = fingerprint_tables(specs, config)

        mini_warehouse["table"].append(pa.table({"id": pa.array([4], type=pa.int64())}))

        fp_after, snaps_after = fingerprint_tables(specs, config)
        assert fp_before != fp_after
        assert snaps_after["events"] != mini_warehouse["snapshot_id"]

    def test_pin_survives_append(self, mini_warehouse, config):
        import pyarrow as pa

        specs = [
            TableSpec(
                name="events",
                uri=mini_warehouse["uri"],
                snapshot_pin=mini_warehouse["snapshot_id"],
            )
        ]
        fp_before, _ = fingerprint_tables(specs, config)
        mini_warehouse["table"].append(pa.table({"id": pa.array([4], type=pa.int64())}))
        fp_after, snaps_after = fingerprint_tables(specs, config)

        assert fp_before == fp_after
        assert snaps_after["events"] == mini_warehouse["snapshot_id"]

    def test_unreachable_yields_random_fingerprint_without_raising(self, tmp_path, config):
        specs = [TableSpec(name="gone", uri=f"file://{tmp_path / 'missing'}#db.gone")]
        fp1, snaps1 = fingerprint_tables(specs, config)
        fp2, _ = fingerprint_tables(specs, config)

        # Never raises; the name is absent from snapshots (execution will
        # hard-error); fingerprints are unique per call so the cell always
        # looks stale rather than serving a possibly-outdated cache hit.
        assert snaps1 == {}
        assert fp1 != fp2
        assert fp1[0].startswith("gone:table:unresolved:")

    def test_sorted_by_name(self, mini_warehouse, config):
        specs = [
            TableSpec(name="zz", uri=mini_warehouse["uri"]),
            TableSpec(name="aa", uri=mini_warehouse["uri"]),
        ]
        fps, _ = fingerprint_tables(specs, config)
        assert fps[0].startswith("aa:") and fps[1].startswith("zz:")

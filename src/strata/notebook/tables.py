"""Iceberg table inputs for notebook cells (``@table`` annotation).

The table's current snapshot id is folded into the cell's provenance hash, so
new data makes the cell stale. At execution the cell namespace gets ``<name>``
(the table URI) and ``<name>_snapshot`` (the resolved snapshot id), so the cell
can scan exactly the snapshot its provenance recorded::

    # @table trips file:///data/warehouse#nyc.trips
    art = client.materialize(
        inputs=[trips],
        transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
    )

``snapshot=<id>`` pins the table, so the cell never goes stale on new data.
A table with no snapshots yet injects ``<name>_snapshot = None`` (a scan reads it
as zero rows) and is never cached, so its first write is seen.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from strata.notebook.models import TableSpec

if TYPE_CHECKING:
    from strata.config import StrataConfig

logger = logging.getLogger(__name__)


def resolve_table_snapshot(
    spec: TableSpec, config: StrataConfig, env: Mapping[str, str] | None = None
) -> int | None:
    """Resolve the snapshot id a cell should read for ``spec``: the pin, else the current one.

    None when the table has no snapshots yet (created, never written).
    A named catalog's credential resolves against *env* (the notebook's) first.

    Raises:
        ValueError: If the catalog/table cannot be reached.
    """
    if spec.snapshot_pin is not None:
        return spec.snapshot_pin

    from strata.iceberg import PyIcebergCatalog

    try:
        # A provider per call: its catalog cache holds this notebook's secrets.
        catalog = PyIcebergCatalog(config, env)
        table = catalog.load_table(spec.uri)
    except Exception as e:
        raise ValueError(f"@table {spec.name}: cannot load table {spec.uri!r}: {e}") from e

    snapshot = table.current_snapshot()
    return None if snapshot is None else snapshot.snapshot_id


def fingerprint_tables(
    specs: list[TableSpec], config: StrataConfig, env: Mapping[str, str] | None = None
) -> tuple[list[str], dict[str, int | None]]:
    """Resolve every table's snapshot for provenance hashing.

    Returns ``(fingerprints, snapshots)``: fingerprints are
    ``"<name>:table:<uri>:<snapshot_id>"``; ``snapshots`` maps table name to
    snapshot id, None for a table with no snapshots yet. Never raises (it runs on
    notebook open too): an unreachable catalog yields a random fingerprint, so the
    cell shows stale and the error surfaces when it runs, instead of serving a
    possibly outdated cache hit. An empty table gets one too: with no snapshot
    to pin, the cell reads whatever is current when it runs.
    """
    fingerprints: list[str] = []
    snapshots: dict[str, int | None] = {}
    for spec in sorted(specs, key=lambda t: t.name):
        try:
            snapshot_id = resolve_table_snapshot(spec, config, env)
        except ValueError as e:
            logger.warning(
                "table fingerprint unresolved for %s (%s): %s",
                spec.name,
                spec.uri,
                e,
            )
            fingerprints.append(f"{spec.name}:table:unresolved:{random_fingerprint()}")
            continue
        snapshots[spec.name] = snapshot_id
        if snapshot_id is None:
            fingerprints.append(f"{spec.name}:table:empty:{random_fingerprint()}")
            continue
        fingerprints.append(f"{spec.name}:table:{spec.uri}:{snapshot_id}")
    return fingerprints, snapshots


def without_empty_table_nonce(fingerprint: str) -> str:
    """``fingerprint`` with an empty table's random part dropped.

    For asking whether a cell's inputs changed (its test result), not whether a cached
    value may be reused: an empty table is the same input until it gets a snapshot.
    """
    name, sep, rest = fingerprint.partition(":")
    if sep and rest.startswith("table:empty:"):
        return f"{name}:table:empty"
    return fingerprint


def random_fingerprint() -> str:
    """A fingerprint no other run shares, for a table whose state cannot be keyed."""
    return hashlib.sha256(os.urandom(32)).hexdigest()

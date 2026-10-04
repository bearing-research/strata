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
) -> int:
    """Resolve the snapshot id a cell should read for ``spec``: the pin, else the current one.

    A named catalog's credential resolves against *env* (the notebook's) first.

    Raises:
        ValueError: If the table has no snapshots, or the catalog/table
            cannot be reached.
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
    if snapshot is None:
        raise ValueError(f"@table {spec.name}: table {spec.uri!r} has no snapshots yet")
    return snapshot.snapshot_id


def fingerprint_tables(
    specs: list[TableSpec], config: StrataConfig, env: Mapping[str, str] | None = None
) -> tuple[list[str], dict[str, int]]:
    """Resolve every table's snapshot for provenance hashing.

    Returns ``(fingerprints, snapshots)``: fingerprints are
    ``"<name>:table:<uri>:<snapshot_id>"``; ``snapshots`` maps table name to
    snapshot id. Never raises (it runs on notebook open too): an unreachable
    catalog yields a random fingerprint, so the cell shows stale and the error
    surfaces when it runs, instead of serving a possibly outdated cache hit.
    """
    fingerprints: list[str] = []
    snapshots: dict[str, int] = {}
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
            random_fp = hashlib.sha256(os.urandom(32)).hexdigest()
            fingerprints.append(f"{spec.name}:table:unresolved:{random_fp}")
            continue
        fingerprints.append(f"{spec.name}:table:{spec.uri}:{snapshot_id}")
        snapshots[spec.name] = snapshot_id
    return fingerprints, snapshots

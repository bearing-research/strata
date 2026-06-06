"""Iceberg table inputs for notebook cells (``@table`` annotation).

A ``@table`` declaration connects a cell to the lake with snapshot-level
staleness: the table's current snapshot id is resolved at provenance time
and folded into the cell's provenance hash (alongside mount fingerprints),
so new data landing in the table makes the cell stale and the normal
cascade machinery re-runs it.

At execution time the executor injects two variables into the cell
namespace: ``<name>`` — the table URI string — and ``<name>_snapshot`` —
the resolved snapshot id — so the cell can scan deterministically at
exactly the snapshot its provenance recorded::

    # @table trips file:///data/warehouse#nyc.trips
    art = client.materialize(
        inputs=[trips],
        transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
    )

``snapshot=<id>`` pins the table: the fingerprint is the pin, so the cell
never goes stale on new data (the lake-side analog of a mount ``pin``).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import TYPE_CHECKING

from strata.notebook.models import TableSpec

if TYPE_CHECKING:
    from strata.config import StrataConfig

logger = logging.getLogger(__name__)


def resolve_table_snapshot(spec: TableSpec, config: StrataConfig) -> int:
    """Resolve the snapshot id a cell should read for ``spec``.

    Returns the pin when set; otherwise loads the table and returns its
    current snapshot id.

    Raises:
        ValueError: If the table has no snapshots, or the catalog/table
            cannot be reached (the cell needs a concrete snapshot to run).
    """
    if spec.snapshot_pin is not None:
        return spec.snapshot_pin

    from strata.iceberg import PyIcebergCatalog

    try:
        catalog = PyIcebergCatalog(config)
        table = catalog.load_table(spec.uri)
    except Exception as e:
        raise ValueError(
            f"@table {spec.name}: cannot load table {spec.uri!r}: {e}"
        ) from e

    snapshot = table.current_snapshot()
    if snapshot is None:
        raise ValueError(
            f"@table {spec.name}: table {spec.uri!r} has no snapshots yet"
        )
    return snapshot.snapshot_id


def fingerprint_tables(
    specs: list[TableSpec], config: StrataConfig
) -> tuple[list[str], dict[str, int]]:
    """Resolve every table's snapshot for provenance hashing.

    Returns ``(fingerprints, snapshots)`` where each fingerprint is
    ``"<name>:table:<uri>:<snapshot_id>"`` and ``snapshots`` maps the
    table name to its resolved snapshot id (for namespace injection).

    Mirrors mount fingerprinting's failure stance: an unreachable catalog
    yields a random fingerprint — the cell shows stale and re-executes
    (where the resolution error surfaces properly) rather than serving a
    possibly-outdated cache hit. Provenance computation must never raise:
    it also runs on notebook open, when the lake may be unreachable.
    """
    fingerprints: list[str] = []
    snapshots: dict[str, int] = {}
    for spec in sorted(specs, key=lambda t: t.name):
        try:
            snapshot_id = resolve_table_snapshot(spec, config)
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

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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.notebook.models import TableSpec

if TYPE_CHECKING:
    from strata.config import StrataConfig

logger = logging.getLogger(__name__)


def resolve_table_snapshot(
    spec: TableSpec,
    config: StrataConfig,
    env: Mapping[str, str] | None = None,
    *,
    notebook: Path | None = None,
) -> int | None:
    """Resolve the snapshot id a cell should read for ``spec``: the pin, else the current one.

    None when the table has no snapshots yet (created, never written).
    A named catalog's credential resolves against *env* (the notebook's) first, and only if
    the notebook at *notebook* may use it.

    Raises:
        ValueError: If the catalog/table cannot be reached.
    """
    return resolve_table_read(spec, config, env, notebook=notebook)[0]


def resolve_table_read(
    spec: TableSpec,
    config: StrataConfig,
    env: Mapping[str, str] | None,
    *,
    notebook: Path | None,
) -> tuple[int | None, int | None, str | None]:
    """The snapshot to read, its data files' bytes and the table's location.

    A pin's size and location are not looked up.
    """
    if spec.snapshot_pin is not None:
        return spec.snapshot_pin, None, None

    from strata.iceberg.catalog import PyIcebergCatalog

    try:
        # A provider per call: its catalog cache holds this notebook's secrets.
        catalog = PyIcebergCatalog(config, env, notebook=notebook)
        table = catalog.load_table(spec.uri)
    except Exception as e:
        raise ValueError(f"@table {spec.name}: cannot load table {spec.uri!r}: {e}") from e

    location = table.location()
    snapshot = table.current_snapshot()
    if snapshot is None:
        return None, 0, location
    size = snapshot.summary.get("total-files-size") if snapshot.summary else None
    return snapshot.snapshot_id, int(size) if size is not None else None, location


def fingerprint_tables(
    specs: list[TableSpec],
    config: StrataConfig,
    env: Mapping[str, str] | None = None,
    *,
    notebook: Path | None = None,
    sizes: dict[str, int | None] | None = None,
    locations: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, int | None]]:
    """Resolve every table's snapshot for provenance hashing.

    *sizes*, when given, is filled with each resolved table's data-file bytes (from its
    snapshot summary, ``None`` if unknown), an upper bound on what a scan reads.
    *locations*, when given, is filled with each loaded table's location.

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
            snapshot_id, size, location = resolve_table_read(spec, config, env, notebook=notebook)
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
        if sizes is not None:
            sizes[spec.name] = size
        if locations is not None and location is not None:
            locations[spec.name] = location
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


def without_table_nonce(fingerprint: str) -> str:
    """``fingerprint`` with an empty or unresolved table's random part dropped.

    What a yes to a large read is kept for (``read_confirm``): with the nonce in it, the
    yes would never match the next run's question.
    """
    name, sep, rest = fingerprint.partition(":")
    for state in ("empty", "unresolved"):
        if sep and rest.startswith(f"table:{state}:"):
            return f"{name}:table:{state}"
    return fingerprint


def random_fingerprint() -> str:
    """A fingerprint no other run shares, for a table whose state cannot be keyed."""
    return hashlib.sha256(os.urandom(32)).hexdigest()


def table_credential(table_uri: str, config: Any) -> str | None:
    """The named credential a table's catalog reads through, if it names one."""
    from strata.iceberg.catalog import named_catalog
    from strata.notebook.credentials import CATALOG_CREDENTIAL_KEY

    catalog, _table = named_catalog(table_uri, config)
    if catalog is None:
        return None
    return config.catalogs[catalog].get(CATALOG_CREDENTIAL_KEY) or None

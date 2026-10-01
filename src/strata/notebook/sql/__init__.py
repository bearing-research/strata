"""SQL cell support for Strata notebooks.

Public surface:
- ``DriverAdapter`` — protocol every backend implements
- ``AdapterCapabilities`` — capability flags (per-table fingerprint,
  snapshot support, separate probe-conn requirement)
- ``QualifiedTable`` — fully qualified table reference (catalog.schema.name)
- ``FreshnessToken`` / ``SchemaFingerprint`` — opaque equality tokens
  folded into the SQL cell provenance hash
- ``register_adapter`` / ``get_adapter`` — driver registry

Per-driver implementations live in ``strata.notebook.sql.drivers.*``.
Each driver module registers its adapter at import time.
"""

from strata.notebook.sql.adapter import (
    AdapterCapabilities,
    ColumnInfo,
    DriverAdapter,
    FreshnessToken,
    QualifiedTable,
    SchemaFingerprint,
    TableSchema,
    hash_connection_identity,
)

# Drivers whose optional ADBC package isn't installed are skipped; they surface as
# ``connection_driver_unknown`` only when a SQL cell references them.
from strata.notebook.sql.drivers import register_default_adapters as _register
from strata.notebook.sql.registry import (
    get_adapter,
    known_drivers,
    register_adapter,
)

_register()

__all__ = [
    "AdapterCapabilities",
    "ColumnInfo",
    "DriverAdapter",
    "FreshnessToken",
    "QualifiedTable",
    "SchemaFingerprint",
    "TableSchema",
    "get_adapter",
    "hash_connection_identity",
    "known_drivers",
    "register_adapter",
]

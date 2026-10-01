"""SQL cell support for Strata notebooks.

Exports the ``DriverAdapter`` protocol, capability flags, table and freshness
types folded into SQL cell provenance, and the driver registry. Each module in
``strata.notebook.sql.drivers`` registers its adapter at import time.
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

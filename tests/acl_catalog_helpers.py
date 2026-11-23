"""A SQL catalog that resolves names case-insensitively, as AWS Glue does."""

from __future__ import annotations

from pyiceberg.catalog.sql import SqlCatalog


class LowercasingSqlCatalog(SqlCatalog):
    """Glue stores names lowercase and answers ``DB.Events`` with ``db.events``."""

    def load_table(self, identifier):
        parts = identifier if isinstance(identifier, tuple) else tuple(identifier.split("."))
        return super().load_table(tuple(part.lower() for part in parts))

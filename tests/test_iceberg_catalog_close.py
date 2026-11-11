"""A catalog provider closes the catalogs it built.

A ``SqlCatalog`` holds a pool of database connections; one left open is closed only when the
collector finds it, with a ``ResourceWarning``.
"""

from __future__ import annotations

import gc
import warnings

from strata.config import StrataConfig
from strata.iceberg.catalog import PyIcebergCatalog


def _load_and_drop(config: StrataConfig, table_uri: str) -> None:
    # Its own frame, so nothing of the provider outlives the call.
    with PyIcebergCatalog(config) as catalogs:
        catalogs.load_table(table_uri)


def test_a_provider_closes_the_connections_its_catalogs_opened(temp_warehouse, tmp_path):
    config = StrataConfig(cache_dir=tmp_path / "cache")
    gc.collect()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        _load_and_drop(config, temp_warehouse["table_uri"])
        gc.collect()

    unclosed = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ResourceWarning) and "sqlite3.Connection" in str(w.message)
    ]
    assert unclosed == []

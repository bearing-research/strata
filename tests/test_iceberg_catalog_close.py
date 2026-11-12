"""A catalog provider closes the catalogs it built, and the ones it drops.

A ``SqlCatalog`` holds a pool of database connections; one left open is closed only when the
collector finds it, with a ``ResourceWarning``.
"""

from __future__ import annotations

import gc
import warnings

import pytest
from pyiceberg.exceptions import NoSuchTableError

from strata.config import StrataConfig
from strata.iceberg.catalog import PyIcebergCatalog


def _unclosed_connections(action) -> list[str]:
    """The SQLite connections the collector finds open once *action* has run."""
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        action()
        gc.collect()
    return [
        str(w.message)
        for w in caught
        if issubclass(w.category, ResourceWarning) and "sqlite3.Connection" in str(w.message)
    ]


def test_a_provider_closes_the_connections_its_catalogs_opened(temp_warehouse, tmp_path):
    config = StrataConfig(cache_dir=tmp_path / "cache")

    def load_and_drop():
        # Its own frame, so nothing of the provider outlives the call.
        with PyIcebergCatalog(config) as catalogs:
            catalogs.load_table(temp_warehouse["table_uri"])

    assert _unclosed_connections(load_and_drop) == []


def test_a_named_catalog_dropped_for_new_settings_is_closed(tmp_path):
    """The admin credentials route drops named catalogs so the next use reads the new ones."""
    warehouse = tmp_path / "north"
    warehouse.mkdir()
    north = {"type": "sql", "uri": f"sqlite:///{warehouse / 'catalog.db'}"}
    config = StrataConfig(cache_dir=tmp_path / "cache", catalogs={"north": north})

    with PyIcebergCatalog(config) as catalogs:

        def build_and_forget():
            with pytest.raises(NoSuchTableError):
                catalogs.load_table("north:taxi.trips")
            catalogs.forget_named_catalogs()

        assert _unclosed_connections(build_and_forget) == []


def test_a_catalog_rebuilt_after_its_connection_failed_is_closed(temp_warehouse, tmp_path):
    """Asked directly, not of the collector: the failure's traceback keeps the catalog alive."""
    from sqlalchemy.exc import OperationalError

    config = StrataConfig(cache_dir=tmp_path / "cache")
    warehouse, _ = PyIcebergCatalog.parse_table_uri(temp_warehouse["table_uri"])

    with PyIcebergCatalog(config) as catalogs:
        poisoned = catalogs._get_catalog(warehouse)
        closed: list[bool] = []
        close = poisoned.close

        def disk_error(table_id):
            raise OperationalError("SELECT", {}, Exception("disk I/O error"))

        def recorded_close():
            closed.append(True)
            close()

        poisoned.load_table = disk_error
        poisoned.close = recorded_close

        catalogs.load_table(temp_warehouse["table_uri"])

        assert closed == [True]

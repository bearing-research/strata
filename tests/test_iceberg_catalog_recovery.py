"""A catalog whose backing connection dies must not stay cached.

``PyIcebergCatalog`` caches one ``Catalog`` per warehouse. When that catalog's
connection goes bad — the observed case is a SqlCatalog on SQLite returning
``disk I/O error`` (SQLITE_IOERR) — the object lived on in the cache, so every
later read of that warehouse failed the same way until the process restarted.
Retrying at the call site could never help, which is why a bounded retry loop
in the smoke test still lost all four attempts.
"""

from __future__ import annotations

import pytest

from strata.config import StrataConfig
from strata.iceberg import PyIcebergCatalog, _is_connection_io_error


class _Catalog:
    """Stand-in catalog whose ``load_table`` outcome is scripted."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def load_table(self, table_id):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def catalog(tmp_path):
    return PyIcebergCatalog(StrataConfig(artifact_dir=str(tmp_path / "artifacts")))


class TestConnectionIoErrorMatcher:
    def test_matches_sqlite_io_error(self):
        assert _is_connection_io_error(Exception("(sqlite3.OperationalError) disk I/O error"))

    def test_matches_adbc_finalizer_fallout(self):
        assert _is_connection_io_error(Exception("Underflow in closing this AdbcStatement"))

    def test_does_not_match_a_missing_table(self):
        """A real error must surface on the first attempt, not be retried."""
        assert not _is_connection_io_error(Exception("Table does not exist: db.events"))

    def test_does_not_match_a_bad_uri(self):
        assert not _is_connection_io_error(ValueError("Unsupported mount URI scheme"))


class TestCatalogRebuiltAfterIoError:
    def test_poisoned_catalog_is_replaced_and_the_read_succeeds(self, catalog, monkeypatch):
        sentinel = object()
        broken = _Catalog([Exception("(sqlite3.OperationalError) disk I/O error")])
        healthy = _Catalog([sentinel])
        built = iter([broken, healthy])

        monkeypatch.setattr(catalog, "_build_catalog", lambda _wh: next(built))

        assert catalog.load_table("db.events") is sentinel
        assert broken.calls == 1
        assert healthy.calls == 1

    def test_a_real_error_is_not_retried(self, catalog, monkeypatch):
        only = _Catalog([Exception("Table does not exist: db.events")])
        monkeypatch.setattr(catalog, "_build_catalog", lambda _wh: only)

        with pytest.raises(Exception, match="does not exist"):
            catalog.load_table("db.events")
        assert only.calls == 1, "a genuine error must not trigger a rebuild"

    def test_a_second_io_error_propagates(self, catalog, monkeypatch):
        """One rebuild, not an unbounded retry loop."""
        err = Exception("(sqlite3.OperationalError) disk I/O error")
        catalogs = [_Catalog([err]), _Catalog([err])]
        built = iter(catalogs)
        monkeypatch.setattr(catalog, "_build_catalog", lambda _wh: next(built))

        with pytest.raises(Exception, match="disk I/O error"):
            catalog.load_table("db.events")
        assert all(c.calls == 1 for c in catalogs)

    def test_healthy_reads_do_not_rebuild(self, catalog, monkeypatch):
        sentinel = object()
        healthy = _Catalog([sentinel, sentinel])
        monkeypatch.setattr(catalog, "_build_catalog", lambda _wh: healthy)

        assert catalog.load_table("db.events") is sentinel
        assert catalog.load_table("db.events") is sentinel
        assert healthy.calls == 2

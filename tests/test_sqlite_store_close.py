"""Strata's own SQLite stores close a dialect they made, never one passed in, and a
reset closes the singleton before dropping it."""

from __future__ import annotations

import pytest
from vestigo.sql_backend import SqliteDialect

from strata.auth import api_keys
from strata.streaming import ownership


def _recording(monkeypatch, dialect: SqliteDialect, closes: list[str], name: str) -> None:
    close = dialect.close

    def recorded():
        closes.append(name)
        close()

    monkeypatch.setattr(dialect, "close", recorded)


@pytest.mark.parametrize("store_cls", [ownership.StreamOwnershipStore, api_keys.ApiKeyStore])
def test_a_store_closes_the_dialect_it_made_and_not_one_passed_in(store_cls, tmp_path, monkeypatch):
    closes: list[str] = []
    own = store_cls(tmp_path / "own.sqlite")
    _recording(monkeypatch, own._dialect, closes, "own")
    own.close()
    assert closes == ["own"]

    shared = SqliteDialect(tmp_path / "shared.sqlite")
    _recording(monkeypatch, shared, closes, "shared")
    try:
        store_cls(tmp_path / "shared.sqlite", dialect=shared).close()
        assert closes == ["own"]
    finally:
        shared.close()


@pytest.mark.parametrize(
    ("module", "singleton", "getter", "reset"),
    [
        (
            ownership,
            "_ownership_store",
            "get_stream_ownership_store",
            "reset_stream_ownership_store",
        ),
        (api_keys, "_api_key_store", "get_api_key_store", "reset_api_key_store"),
    ],
)
def test_a_reset_closes_the_singleton(module, singleton, getter, reset, tmp_path, monkeypatch):
    monkeypatch.setattr(module, singleton, None)
    store = getattr(module, getter)(tmp_path / "db.sqlite")
    closes: list[str] = []
    _recording(monkeypatch, store._dialect, closes, "store")
    getattr(module, reset)()
    assert closes == ["store"]
    assert getattr(module, singleton) is None

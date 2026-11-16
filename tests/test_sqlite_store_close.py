"""Strata's own SQLite stores close a dialect they made, never one passed in, and a
reset closes the singleton before dropping it."""

from __future__ import annotations

import pytest
from vestigo.sql_backend import SqliteDialect

from strata.auth import api_keys
from strata.notebook import worker_store
from strata.streaming import ownership

STORES = [
    ownership.StreamOwnershipStore,
    worker_store.NotebookWorkerStore,
    api_keys.ApiKeyStore,
]


@pytest.mark.parametrize("store_cls", STORES)
def test_a_store_closes_the_dialect_it_made_and_not_one_passed_in(store_cls, tmp_path, monkeypatch):
    closes = []
    own = store_cls(tmp_path / "own.sqlite")
    monkeypatch.setattr(own._dialect, "close", lambda: closes.append("own"))
    own.close()
    assert closes == ["own"]

    shared = SqliteDialect(tmp_path / "shared.sqlite")
    monkeypatch.setattr(shared, "close", lambda: closes.append("shared"))
    store_cls(tmp_path / "shared.sqlite", dialect=shared).close()
    assert closes == ["own"]


@pytest.mark.parametrize(
    ("module", "singleton", "store_cls", "reset"),
    [
        (
            ownership,
            "_ownership_store",
            ownership.StreamOwnershipStore,
            "reset_stream_ownership_store",
        ),
        (
            worker_store,
            "_worker_store",
            worker_store.NotebookWorkerStore,
            "reset_notebook_worker_store",
        ),
        (api_keys, "_api_key_store", api_keys.ApiKeyStore, "reset_api_key_store"),
    ],
)
def test_a_reset_closes_the_singleton(module, singleton, store_cls, reset, tmp_path, monkeypatch):
    store = store_cls(tmp_path / "db.sqlite")
    closes = []
    monkeypatch.setattr(store, "close", lambda: closes.append(True))
    monkeypatch.setattr(module, singleton, store)
    getattr(module, reset)()
    assert closes == [True]
    assert getattr(module, singleton) is None

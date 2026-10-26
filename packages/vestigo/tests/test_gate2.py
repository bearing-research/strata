"""Gate 2 on SQLite; the Postgres module runs the same class against a live server."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from gate2 import Gate2, Ledger, make_ledger

from vestigo.store import ArtifactStore


@pytest.fixture
def ledger(tmp_path) -> Iterator[Ledger]:
    store = ArtifactStore(tmp_path / "artifacts")
    yield make_ledger(store)
    store.close()


class TestGate2(Gate2):
    pass

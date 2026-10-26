"""Gate 2 on SQLite; the Postgres module runs the same class against a live server."""

from __future__ import annotations

import pytest
from gate2 import Gate2, Ledger, make_ledger
from vestigo.store import ArtifactStore


@pytest.fixture
def ledger(tmp_path) -> Ledger:
    return make_ledger(ArtifactStore(tmp_path / "artifacts"))


class TestGate2(Gate2):
    pass

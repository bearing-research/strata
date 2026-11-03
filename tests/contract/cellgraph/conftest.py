"""Run the traces with the session behind a host that remembers what it was asked."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.contract.cellgraph.recording_host import RecordingHost, host_members


@pytest.fixture
def executor_host_only(monkeypatch) -> Iterator[None]:
    """Every ``CellExecutor`` gets the session through ``RecordingHost``; a stray use fails."""
    from strata.notebook.executor import CellExecutor

    members, strays = host_members(), set[str]()
    init = CellExecutor.__init__

    def recording(self, session, *args, **kwargs):
        init(self, RecordingHost(session, members, strays), *args, **kwargs)

    monkeypatch.setattr(CellExecutor, "__init__", recording)
    yield
    assert not strays, f"not in the ExecutorHost protocol: {sorted(strays)}"

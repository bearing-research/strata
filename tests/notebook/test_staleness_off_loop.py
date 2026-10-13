"""Staleness after an environment change runs off the event loop.

It reads the outside world (``@fetch``, ``@table``) and, for a worker-annotated cell, the
server store's registry: on the loop, one slow read stalls every socket and route.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import create_notebook
from tests.conftest import hold, ran_while_held


@pytest.mark.asyncio
async def test_a_dependency_change_computes_staleness_off_the_loop(tmp_path, monkeypatch):
    notebook = create_notebook(tmp_path, "Off loop", initialize_environment=False)
    session = NotebookSession(parse_notebook(notebook), notebook)

    def added(path, package, **kwargs):
        return SimpleNamespace(success=True, lockfile_changed=True, operation_log=None)

    async def environment_refreshed(**kwargs):
        return None

    # No real uv: the change succeeds and moves the lockfile, so staleness is recomputed.
    monkeypatch.setattr("strata.notebook.dependencies.add_dependency", added)
    monkeypatch.setattr(session, "on_dependencies_changed", environment_refreshed)
    gate = hold(monkeypatch, session, "compute_staleness")

    ran, outcome = await ran_while_held(gate, session.mutate_dependency("six", action="add"))

    assert ran, "staleness ran on the event loop"
    assert outcome.result.success

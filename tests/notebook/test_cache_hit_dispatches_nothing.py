"""A provenance hit dispatches zero work.

A hit resolved by asking something (re-reading an input, probing a worker, checking the
shared store first) would keep every other test green while making hits cost money. So
these tests assert absence, not speed: no process spawned, no request sent. The
instruments record rather than raise, since an executor that catches broadly would turn a
raised spawn into a failed cell.
"""

from __future__ import annotations

import asyncio

import httpx

from strata.config import StrataConfig
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

SOURCE = "value = sum(range(2000))"


def build_notebook(parent, name: str) -> NotebookSession:
    """A two-cell notebook, so the upstream's output is stored; a leaf cell could never hit."""
    notebook_dir = create_notebook(parent / name, name)
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", SOURCE)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def record_spawns(monkeypatch) -> list[tuple]:
    """Every process the executor starts from here on, still really started."""
    spawned: list[tuple] = []
    real = asyncio.create_subprocess_exec

    async def spy(*args, **kwargs):
        spawned.append(args)
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    return spawned


def record_requests(monkeypatch) -> list[str]:
    """Every request the executor sends from here on."""
    sent: list[str] = []
    real = httpx.AsyncClient.send

    async def spy(self, request, *args, **kwargs):
        sent.append(f"{request.method} {request.url}")
        return await real(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", spy)
    return sent


async def test_a_local_hit_spawns_nothing(tmp_path, monkeypatch):
    session = build_notebook(tmp_path, "solo")
    executor = CellExecutor(session)

    first = await executor.execute_cell("up", SOURCE)
    assert first.success, first.error
    assert first.cache_hit is False

    spawned = record_spawns(monkeypatch)

    second = await executor.execute_cell("up", SOURCE)
    assert second.success, second.error
    assert second.cache_hit is True
    assert spawned == []


async def test_a_local_hit_asks_the_team_store_nothing(tmp_path, monkeypatch):
    """With a team cache configured, a local hit never consults the shared store.

    A lookup before the local check would put a network round-trip on every cell run.
    """
    session = build_notebook(tmp_path, "with-team-store")
    monkeypatch.setattr(
        CellExecutor,
        "_lake_config",
        lambda self: StrataConfig(
            cache_dir=tmp_path / "cache",
            notebook_remote_store_url="http://store.example",
            notebook_team_cache_enabled=True,
        ),
    )
    executor = CellExecutor(session)

    first = await executor.execute_cell("up", SOURCE)
    assert first.success, first.error

    spawned = record_spawns(monkeypatch)
    sent = record_requests(monkeypatch)

    second = await executor.execute_cell("up", SOURCE)
    assert second.success, second.error
    assert second.cache_hit is True
    assert spawned == []
    assert sent == []


async def test_the_instruments_fire_when_a_cell_actually_runs(tmp_path, monkeypatch):
    """The control: a bug reporting every run as a cache hit would pass the tests above by never
    running anything.
    """
    session = build_notebook(tmp_path, "edited")
    executor = CellExecutor(session)

    first = await executor.execute_cell("up", SOURCE)
    assert first.success, first.error

    edited = SOURCE + "\nvalue += 1"
    write_cell(session.path, "up", edited)
    session.reload()

    spawned = record_spawns(monkeypatch)

    changed = await executor.execute_cell("up", edited)
    assert changed.success, changed.error
    assert changed.cache_hit is False
    assert spawned, "an edited cell must actually run"

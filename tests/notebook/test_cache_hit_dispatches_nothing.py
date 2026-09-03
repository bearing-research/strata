"""A provenance hit dispatches zero work.

"You never recompute" is the product claim and the unit economics at once: a
hit costs nothing to serve, which is why hit rate is margin rather than a
feature. It has been structurally true for a long time and guarded by nothing —
any future change that resolved a hit by *asking* something (re-reading an
input, probing a worker, checking the shared store before the local one) would
keep every existing test green while quietly making cache hits cost money.

So these tests do not assert that a hit is fast. They assert it does not
happen: no process spawned, no request sent. Timing would be a threshold to
tune; absence is a fact.

The instruments record rather than raise. An executor that catches broadly
would swallow an exception thrown from a spawn and report a failed cell, which
looks the same as "nothing was spawned" — and the last test here exists to
prove the instrument fires at all, which it could not do if the mechanism were
an exception the executor might eat.
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
    """A two-cell notebook, so the upstream's output is actually stored.

    A leaf cell keeps nothing in the artifact store, so it could never
    demonstrate a provenance hit in the first place.
    """
    notebook_dir = create_notebook(parent / name, name)
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", SOURCE)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "doubled = value * 2")
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


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
    """Configured team cache, local hit — the shared store is never consulted.

    The ordering is the whole point. A lookup before the local check would put
    a network round-trip on the hot path of every cell run and, in a hosted
    deployment, would bill for answering a question already answered locally.
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
    """The control. Without it, a bug that reported *every* run as a cache hit
    would satisfy both tests above — they would pass by never running anything,
    which is the exact outcome they exist to distinguish from."""
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

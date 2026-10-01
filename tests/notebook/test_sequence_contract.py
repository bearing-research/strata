"""One sequence per outbound message, as the protocol reference promises.

Several paths sent a batch of frames under one ``seq``; a client deduping on it kept the
first frame and lost the rest. Asserted over a whole stream, since each frame looks fine alone.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# Print to both streams: a run emits a console frame per stream plus its result,
# and a cell writing to only one stream hides the grouping.
PRINTS_AND_SUCCEEDS = (
    "import sys\nprint('from a')\nprint('a warning', file=sys.stderr)\nrows = [1, 2, 3]\nrows\n"
)
PRINTS_AND_FAILS = "print('about to fail')\nraise ValueError('boom')\nrows = [1]\n"


class Observer:
    """A connection that records only what the server sends it."""

    def __init__(self) -> None:
        self.raw: list[str] = []

    async def send_text(self, text: str) -> None:
        self.raw.append(text)

    @property
    def sent(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self.raw]


def _assert_one_sequence_each(observer: Observer, label: str) -> None:
    frames = observer.sent
    assert frames, f"{label}: nothing was sent, so nothing is proven"
    seqs = [frame["seq"] for frame in frames]

    repeated = {seq: count for seq, count in Counter(seqs).items() if count > 1}
    if repeated:
        shared = [
            f"seq {f['seq']} {f['type']} {(f.get('payload') or {}).get('cell_id')}"
            for f in frames
            if f["seq"] in repeated
        ]
        raise AssertionError(f"{label}: frames share a sequence: {shared}")
    assert seqs == sorted(seqs), f"{label}: frames went out of order: {seqs}"

    # No number without a frame behind it: an unsent sequence reads as a gap, which
    # clients resync on, so burning one per run would reset their state each time.
    missing = sorted(set(range(seqs[0], seqs[-1] + 1)) - set(seqs))
    assert not missing, f"{label}: numbers drawn with no frame sent: {missing}"


@pytest.fixture
def chain(tmp_path) -> Path:
    nb = create_notebook(tmp_path / "nb", "sequence")
    after = None
    for cell_id, source in (
        ("a", PRINTS_AND_SUCCEEDS),
        ("b", "print('from b')\ntotal = sum(rows)\ntotal\n"),
        ("c", "doubled = total * 2\n{'doubled': doubled}\n"),
    ):
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language="python")
        write_cell(nb, cell_id, source)
        after = cell_id
    return nb


def _session(nb: Path) -> Any:
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    return session


@contextmanager
def _watching(session: Any) -> Iterator[Observer]:
    """Register a connection for the session and remove it after (the map is module global)."""
    from strata.notebook.ws import _notebook_connections

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        yield observer
    finally:
        _notebook_connections.get(session.id, []).remove(observer)


@pytest.mark.asyncio
async def test_a_run_that_prints_numbers_its_console_and_its_result_apart(chain):
    """stdout, stderr and the result each get a number; a client deduping on one lost the result."""
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    session = _session(chain)
    with _watching(session) as observer:
        # The cell that is asked for has to be the one that prints: an upstream
        # rebuilt on the way sends no frames of its own.
        await execute_cell_and_broadcast(
            session, "a", _ensure_execution_state(session.id), session.id, mode="normal"
        )

    _assert_one_sequence_each(observer, "run with stdout")
    consoles = [f for f in observer.sent if f["type"] == "cell_console"]
    streams = {(f.get("payload") or {}).get("stream") for f in consoles}
    assert streams == {"stdout", "stderr"}, f"both streams have to be in the stream: {streams}"


@pytest.mark.asyncio
async def test_a_failure_numbers_its_console_its_error_and_its_status_apart(chain):
    from strata.notebook.models import CellStatus
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    session = _session(chain)
    state = _ensure_execution_state(session.id)
    await execute_cell_and_broadcast(session, "c", state, session.id, mode="normal")

    write_cell(chain, "a", PRINTS_AND_FAILS)
    session.reload()
    session._analyze_and_build_dag()
    with _watching(session) as observer:
        await execute_cell_and_broadcast(session, "a", state, session.id, mode="normal")

    _assert_one_sequence_each(observer, "failure")
    assert session.notebook_state.get_cell("a").status == CellStatus.ERROR
    assert [f["type"] for f in observer.sent].count("cell_error") == 1


@pytest.mark.asyncio
async def test_requests_through_the_handlers_leave_no_gaps(chain):
    """Driven through the real handlers, with one observer across every request.

    The handlers' reservation is where a sequence could be drawn without a frame following,
    leaving a gap the reference treats as a reason to resync. A fresh observer per request
    would hide it.
    """
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_cell_execute,
        _handle_cell_execute_force,
        _handle_cell_execute_rerun,
        _handle_notebook_rerun_all,
    )

    session = _session(chain)
    state = _ensure_execution_state(session.id)

    with _watching(session) as observer:
        for handler in (
            _handle_cell_execute,
            _handle_cell_execute_rerun,
            _handle_cell_execute_force,
        ):
            await handler(observer, session, {"cell_id": "b"}, state, session.id)
            task = state.execution_task
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)

        await _handle_notebook_rerun_all(observer, session, state, session.id, {})
        task = state.execution_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    _assert_one_sequence_each(observer, "four requests, one client")


@pytest.mark.asyncio
async def test_run_all_numbers_every_cell_it_starts_apart(chain):
    """Each cell's running frame gets its own sequence, not the one the run began with."""
    from strata.notebook.ws import _ensure_execution_state, _handle_notebook_run_all

    session = _session(chain)
    state = _ensure_execution_state(session.id)
    with _watching(session) as observer:
        await _handle_notebook_run_all(observer, session, state, session.id, {})
        task = state.execution_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    _assert_one_sequence_each(observer, "run all")
    started = [
        f
        for f in observer.sent
        if f["type"] == "cell_status" and (f.get("payload") or {}).get("status") == "running"
    ]
    assert len(started) >= 2, f"too few cells ran to prove anything: {started}"

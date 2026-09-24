"""One sequence per outbound message, as the protocol reference promises.

The reference says the counter "increments on every outbound message" and tells
client authors to "key dedupe on ``seq``". Several places sent a batch of
frames under one number instead: an execution's stdout, stderr and result; one
number for every cell a failure made stale; one for every cell a Run All
started; the error and status of a failed cell. A client following that advice
kept the first of each batch and dropped the rest, which meant losing the frame
that said a cell had finished, or the error text behind a red cell.

Asserted over a whole stream rather than one call, because that is how the
grouping showed up: each individual frame looked fine.
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

# Both streams: an execution emits a console frame per stream plus its result,
# and those three are what used to share one number. A cell writing to only one
# of them leaves the grouping invisible.
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

    # And no number without a frame behind it. A sequence drawn and not sent
    # reads as a gap, which the reference tells a client to resync on, so
    # burning one on every run would replace its whole state each time.
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
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    return session


@contextmanager
def _watching(session: Any) -> Iterator[Observer]:
    """Register a connection for the session, and take it out again.

    The connection map is module global and nothing resets it between tests.
    """
    from strata.notebook.ws import _notebook_connections

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        yield observer
    finally:
        _notebook_connections.get(session.id, []).remove(observer)


@pytest.mark.asyncio
async def test_a_run_that_prints_numbers_its_console_and_its_result_apart(chain):
    """stdout, stderr and the result describe one execution and used to share
    one number. A client deduping on it kept the console and lost the result."""
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
async def test_run_all_numbers_every_cell_it_starts_apart(chain):
    """Every cell's running frame drew the one sequence the run began with."""
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

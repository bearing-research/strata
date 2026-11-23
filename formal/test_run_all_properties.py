"""Property test: Run All leaves what running each cell in order leaves (SYSTEMS 9, BAT-1).

Hypothesis builds a small notebook of Python cells over ints and lists: constants,
arithmetic, lists of upstream values, a cell that appends to an upstream list in place,
and a cell that fails. A random dependency order is laid out in a random display order,
so a consumer can sit above its producer. A sink reads every value, so each is stored.

Two copies of the notebook run through the real WebSocket: one with Run All, one cell
by cell in display order (cascades accepted). Then one cell is edited in both and
each runs again the same way, so the second pass mixes cache hits with runs. After
each pass both copies must report the same status for every cell and hold the same
stored value for every ready one.

Needs Hypothesis, which is not a project dependency. Slow (real cells):

    uv run --with hypothesis pytest formal/test_run_all_properties.py --timeout=0
"""

# Fixtures are imported from tests/ and then requested by name (or autouse).
# ruff: noqa: F811

from __future__ import annotations

import json
import os
import queue
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import _reset_process_globals  # noqa: F401  (autouse)
from tests.notebook.conftest import fast_notebook_env  # noqa: F401  (autouse)
from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    _reset_ws_globals,
    create_test_app,
    open_notebook_session,
    ws_connect,
)

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, event, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

SINK = "sink"


TEMPLATES = {
    # kind: (what it makes, what it reads, source given its variable, its input and a constant)
    "const": ("int", None, "{v} = {k}"),
    "list_const": ("list", None, "{v} = [{k}]"),
    "add": ("int", "int", "{v} = {d} + {k}"),
    "wrap": ("list", "int", "{v} = [{d}, {k}]"),
    "fail": ("int", "int", "{v} = {d} // ({k} % 2)"),  # fails for an even constant
    "length": ("int", "list", "{v} = len({d}) + {k}"),
    "append": ("int", "list", "{d}.append({k})\n{v} = len({d})"),  # mutates its input
}


@st.composite
def notebooks(draw):
    """``(sources, edit)``: cell id -> source in display order, and one (cell, source) edit."""
    n = draw(st.integers(2, 5))
    rank = draw(st.permutations(range(n)))  # rank[p]: position p's place in the dependency order
    made: list[str] = []  # by rank
    cells: list[tuple[str, str | None]] = []  # by rank: (template, input variable)
    chosen: set[str] = set()
    for r in range(n):
        kinds = [
            kind for kind, (_, reads, _) in TEMPLATES.items() if reads is None or reads in made
        ]
        # The leads (an in-place mutation, a failure) are drawn twice as often.
        kind = draw(st.sampled_from(kinds + [k for k in kinds if k in ("append", "fail")]))
        chosen.add(kind)
        makes, reads, template = TEMPLATES[kind]
        dep = None
        if reads is not None:
            dep = f"v{draw(st.sampled_from([d for d in range(r) if made[d] == reads]))}"
        made.append(makes)
        cells.append((template, dep))

    def source(r: int, k: int) -> str:
        template, dep = cells[r]
        return template.format(v=f"v{r}", d=dep, k=k)

    sources = {f"c{p}": source(rank[p], draw(st.integers(0, 3))) for p in range(n)}
    for kind in chosen:
        event(f"cell: {kind}")
    if any(rank[p] > rank[q] for p in range(n) for q in range(p + 1, n)):
        event("a consumer above its producer")
    sources[SINK] = f"{SINK} = [{', '.join(f'v{r}' for r in range(n))}]"
    p = draw(st.integers(0, n - 1))
    return sources, (f"c{p}", source(rank[p], draw(st.integers(0, 3))))


class Runs:
    """Each scheduled run's end and each cascade prompt, without reading the socket.

    The test socket's receive blocks with no timeout, so waiting on frames hangs once
    a run sends its last one; these hooks are where the server itself finishes.
    """

    def __init__(self) -> None:
        self.finished: queue.Queue[str] = queue.Queue()
        self.prompts: queue.Queue[str] = queue.Queue()

    def wait(self, ws, cell_id: str | None = None) -> None:
        """Wait for the run just requested to end, accepting the cascade it asks for."""
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            with suppress(queue.Empty):
                self.finished.get(timeout=0.05)
                return
            with suppress(queue.Empty):
                plan_id = self.prompts.get_nowait()
                assert cell_id is not None, "Run All asked for a cascade"
                ws.execute_cascade(cell_id, plan_id)
        raise AssertionError(f"run of {cell_id or 'all cells'} did not end")


@contextmanager
def hooked_runs():
    import strata.notebook.ws as ws_module

    runs = Runs()
    run_task, send = ws_module._run_execution_task, ws_module._send_message

    async def run_execution_task(*args, **kwargs):
        try:
            return await run_task(*args, **kwargs)
        finally:
            runs.finished.put("done")

    async def send_message(websocket, message):
        if message["type"] == "cascade_prompt":
            runs.prompts.put(message["payload"]["plan_id"])
        return await send(websocket, message)

    with (
        mock.patch.object(ws_module, "_run_execution_task", run_execution_task),
        mock.patch.object(ws_module, "_send_message", send_message),
    ):
        yield runs


def settle(session, cell_id: str, source: str) -> None:
    """Wait for the server to take an edit (it is applied off the socket's read loop)."""
    deadline = time.monotonic() + 60
    while session.notebook_state.get_cell(cell_id).source != source:
        assert time.monotonic() < deadline, f"edit of {cell_id} never applied"
        time.sleep(0.05)


def outcome(session, sources: dict[str, str]) -> dict[str, tuple[str, object]]:
    """Each cell's status, and for a ready one the stored values of what it defines."""
    store = session.get_artifact_manager().artifact_store
    result: dict[str, tuple[str, object]] = {}
    for cell_id in sources:
        cell = session.notebook_state.get_cell(cell_id)
        status = cell.status.value
        values = None
        if status == "ready" and cell.artifact_uris:
            values = {}
            for var, uri in sorted(cell.artifact_uris.items()):
                artifact_id, _, version = uri.removeprefix("strata://artifact/").partition("@v=")
                values[var] = json.loads(store.blob_store.read_blob(artifact_id, int(version)))
        result[cell_id] = (status, values)
    return result


def build(root: Path, sources: dict[str, str]) -> Path:
    nb = NotebookBuilder(root)
    previous = None
    for cell_id, src in sources.items():
        nb.add_cell(cell_id, src, previous)
        previous = cell_id
    return nb.path


@pytest.mark.timeout(3600)  # real cells: well past the suite's 180 s default
@settings(
    max_examples=int(os.environ.get("RUN_ALL_EXAMPLES", "15")),
    deadline=None,
    suppress_health_check=list(HealthCheck),
)
@given(notebooks())
def test_run_all_leaves_what_single_cell_runs_leave(case):
    sources, (edit_cell, edit_source) = case
    _reset_ws_globals()
    client = TestClient(create_test_app())
    with tempfile.TemporaryDirectory() as tmpdir, hooked_runs() as runs:
        batched = build(Path(tmpdir) / "batched", sources)
        single = build(Path(tmpdir) / "single", sources)
        with (
            open_notebook_session(client, batched) as (b_sid, b_session),
            open_notebook_session(client, single) as (s_sid, s_session),
            ws_connect(client, b_sid) as b_ws,
            ws_connect(client, s_sid) as s_ws,
        ):
            for step in ("fresh", f"after editing {edit_cell}"):
                if step != "fresh":
                    for ws, session in ((b_ws, b_session), (s_ws, s_session)):
                        ws.update_source(edit_cell, edit_source)
                        settle(session, edit_cell, edit_source)
                b_ws.run_all()
                runs.wait(b_ws)
                for cell_id in sources:
                    s_ws.execute_cell(cell_id)
                    runs.wait(s_ws, cell_id)

                got, want = outcome(b_session, sources), outcome(s_session, sources)
                assert got == want, (
                    f"{step}: Run All left {got}, single-cell runs left {want}; sources {sources}"
                )

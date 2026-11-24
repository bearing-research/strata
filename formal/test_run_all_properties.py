"""Property test: Run All leaves what running each cell in order leaves (SYSTEMS 9, BAT-1).

Hypothesis builds a small notebook of Python cells over ints and lists: constants,
arithmetic, lists of upstream values, a cell that appends to an upstream list in place,
and a cell that fails. Module state too: a function, one that appends to an upstream
list through its globals, a call of either, an imported module and a use of it. And the
cells Run All treats apart: a ``@nocache`` constant, a ``@loop`` over an int it redefines
(seeded by its upstream), and any cell under ``@timeout``, which Run All runs on its
own between batches, as it does a worker's. A random dependency order is laid out in a
random display order, so a consumer can sit above its producer. A sink reads every
value, so each is stored.

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

import hashlib
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
from hypothesis import HealthCheck, event, example, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

SINK = "sink"
# Reads the sink, so the sink's own view of every value is stored and compared: a batched
# cell can change another's upstream that only a later reader sees.
READER = ("reader", f"seen = {SINK}")


TEMPLATES = {
    # kind: (what it makes, what it reads, source given its variable, its input and a constant)
    "const": ("int", None, "{v} = {k}"),
    "list_const": ("list", None, "{v} = [{k}]"),
    "add": ("int", "int", "{v} = {d} + {k}"),
    "wrap": ("list", "int", "{v} = [{d}, {k}]"),
    "fail": ("int", "int", "{v} = {d} // ({k} % 2)"),  # fails for an even constant
    "length": ("int", "list", "{v} = len({d}) + {k}"),
    "append": ("int", "list", "{d}.append({k})\n{v} = len({d})"),  # mutates its input
    "fresh": ("int", None, "# @nocache\n{v} = {k}"),
    # Counts its runs in a file of its work dir; its value stays a constant, since a
    # cell-by-cell pass runs it again for each consumer's cascade.
    "counted": (
        "int",
        None,
        "# @nocache\n(lambda p: p.write_text(str(int(p.read_text()) + 1 if p.exists() else 1)))"
        "(__import__('pathlib').Path('runs_{v}'))\n{v} = {k}",
    ),
    "func": ("func", None, "def {v}(x):\n    return x + {k}"),
    "mutator": ("func", "list", "def {v}(x):\n    {d}.append(x)\n    return len({d})"),
    "call": ("int", "func", "{v} = {d}({k})"),
    "module": ("module", None, "import math as {v}"),
    "use": ("int", "module", "{v} = {d}.floor({k} / 2)"),
    # Redefines its input: the upstream seeds the carry, the loop's value replaces it.
    "loop": ("int", "int", "# @loop max_iter=2 carry={d}\n{d} = {d} + {k}"),
}
ALONE = "# @timeout 60\n"  # Run All runs the cell on its own, as it does a worker's


@st.composite
def notebooks(draw):
    """``(sources, edit, traits, counted)``: cell id -> source in display order, one
    (cell, source) edit, what the notebook exercises (recorded as events by the test:
    Hypothesis keeps few of a strategy's own), and the variables of the counted cells."""
    n = draw(st.integers(2, 5))
    # rank[p]: position p's place in the dependency order. Half are laid out in it, so a
    # chain resolves; the rest put some consumer above its producer.
    rank = draw(st.just(list(range(n))) | st.permutations(range(n)))
    made: list[tuple[str, str]] = []  # by rank: (what it makes, under what name)
    cells: list[tuple[str, str | None, bool]] = []  # by rank: (template, input, alone)
    traits: set[str] = set()
    for r in range(n):
        kinds = [
            kind
            for kind, (_, reads, _) in TEMPLATES.items()
            if reads is None or any(m == reads for m, _ in made)
        ]
        # The leads (an in-place mutation, a failure, module state, a loop) and what reads
        # an upstream are drawn twice as often.
        leads = ("append", "fail", "mutator", "call", "use", "loop")
        kind = draw(st.sampled_from(kinds + [k for k in kinds if k in leads]))
        traits.add(f"cell: {kind}")
        makes, reads, template = TEMPLATES[kind]
        dep = None
        if reads is not None:
            dep = draw(st.sampled_from([name for m, name in made if m == reads]))
        alone = draw(st.integers(0, 3)) == 0
        if alone:
            traits.add("a cell Run All runs on its own")
        made.append((makes, dep if kind == "loop" else f"v{r}"))
        cells.append((template, dep, alone))

    def source(r: int, k: int) -> str:
        template, dep, alone = cells[r]
        return (ALONE if alone else "") + template.format(v=f"v{r}", d=dep, k=k)

    sources = {f"c{p}": source(rank[p], draw(st.integers(0, 3))) for p in range(n)}
    if any(rank[p] > rank[q] for p in range(n) for q in range(p + 1, n)):
        traits.add("a consumer above its producer")
    # Values only (a function or module reaches it through its callers); a loop redefines
    # its input's name.
    names = dict.fromkeys(name for m, name in made if m in ("int", "list"))
    sources[SINK] = f"{SINK} = [{', '.join(names)}]"
    p = draw(st.integers(0, n - 1))
    counted = [f"v{rank[q]}" for q in range(n) if cells[rank[q]][0] == TEMPLATES["counted"][2]]
    edit = (f"c{p}", source(rank[p], draw(st.integers(0, 3))))
    return sources, edit, sorted(traits), counted


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
    notebook_id = session.notebook_state.id.encode()
    result: dict[str, tuple[str, object]] = {}
    for cell_id in sources:
        cell = session.notebook_state.get_cell(cell_id)
        status = cell.status.value
        values = None
        if status == "ready" and cell.artifact_uris:
            values = {}
            for var, uri in sorted(cell.artifact_uris.items()):
                artifact_id, _, version = uri.removeprefix("strata://artifact/").partition("@v=")
                # A function's record names its module, and the upstream values it uses
                # by artifact id, after the notebook; the two copies' ids differ.
                blob = store.blob_store.read_blob(artifact_id, int(version))
                blob = blob.replace(notebook_id, b"<notebook>")
                try:
                    value = json.loads(blob)
                except ValueError:
                    value = ("bytes", hashlib.sha256(blob).hexdigest())
                if isinstance(value, dict) and "symbol_name" in value:
                    # A cell-defined function's module also ends in a per-copy suffix.
                    value.pop("module_name", None)
                values[var] = value
        result[cell_id] = (status, values)
    return result


def build(root: Path, sources: dict[str, str]) -> Path:
    nb = NotebookBuilder(root)
    previous = None
    for cell_id, src in [*sources.items(), READER]:
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
# Finding 24, which random search reaches rarely: a @nocache producer in a batch (of two
# or more; one cell runs as a single run) read by a cell Run All runs on its own after it.
@example(
    (
        {
            "c0": TEMPLATES["counted"][2].format(v="v0", k=1),
            "c1": "v1 = v0 + 1",
            "c2": ALONE + "v2 = v0 + 2",
            SINK: f"{SINK} = [v0, v1, v2]",
        },
        ("c2", ALONE + "v2 = v0 + 3"),
        [],
        ["v0"],
    )
)
# A function appending to an upstream list through its globals, called by a later cell:
# in a batch the globals are the shared namespace.
@example(
    (
        {
            "c0": "v0 = [1]",
            "c1": TEMPLATES["mutator"][2].format(v="v1", d="v0", k=0),
            "c2": "v2 = v1(2)",
            SINK: f"{SINK} = [v0, v2]",
        },
        ("c2", "v2 = v1(3)"),
        [],
        [],
    )
)
# The same function from a cell Run All runs on its own, called twice in the batch after.
@example(
    (
        {
            "c0": "v0 = [1]",
            "c1": ALONE + TEMPLATES["mutator"][2].format(v="v1", d="v0", k=0),
            "c2": "v2 = v1(2)",
            "c3": "v3 = v1(3)",
            SINK: f"{SINK} = [v0, v2, v3]",
        },
        ("c3", "v3 = v1(4)"),
        [],
        [],
    )
)
def test_run_all_leaves_what_single_cell_runs_leave(case):
    sources, (edit_cell, edit_source), traits, counted = case
    for trait in traits:
        event(trait)
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
            for passes, step in enumerate(("fresh", f"after editing {edit_cell}"), start=1):
                if step != "fresh":
                    for ws, session in ((b_ws, b_session), (s_ws, s_session)):
                        ws.update_source(edit_cell, edit_source)
                        settle(session, edit_cell, edit_source)
                b_ws.run_all()
                runs.wait(b_ws)
                # BAT-2: Run All runs each @nocache cell once.
                for var in counted:
                    runs_file = batched / "work" / f"runs_{var}"
                    ran = runs_file.read_text() if runs_file.exists() else "0"
                    assert ran == str(passes), f"{step}: {var} ran {ran} times in {passes} Run Alls"
                for cell_id in sources:
                    s_ws.execute_cell(cell_id)
                    runs.wait(s_ws, cell_id)

                got, want = outcome(b_session, sources), outcome(s_session, sources)
                assert got == want, (
                    f"{step}: Run All left {got}, single-cell runs left {want}; sources {sources}"
                )

"""The harness user's sweep and hand-over run off the event loop.

``sweep_leftovers`` reads every process on the host, up to twenty times, and
``hand_over`` re-owns a whole run directory: on the loop, every other notebook and
socket the server holds waits for them on each run.
"""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

from strata.notebook.run_scope import in_thread

SRC = Path(__file__).resolve().parents[2] / "src" / "strata"

# Each one sweeps, re-owns a tree, or both (``write_manifest`` and ``harness_fields``
# make the work dir). By name: a closure over every function that calls one takes in
# names as common as ``start``. A new sync caller of one belongs here.
SWEEPS = {
    "hand_over",
    "prepare_work_dir",
    "sweep_leftovers",
    "write_manifest",
    "harness_fields",
}

OFFLOADS = ("to_thread", "run_sync", "run_in_executor")


def _called(func: ast.expr) -> str | None:
    """The called name without its leading underscores: ``self._harness_fields`` is
    passed on as ``context.harness_fields``."""
    if isinstance(func, ast.Name):
        return func.id.lstrip("_")
    if isinstance(func, ast.Attribute):
        return func.attr.lstrip("_")
    return None


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef):
    """The coroutine's own nodes: not those of a function defined in it."""
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.append(child)


def _calls(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.Call]:
    return [node for node in _own_nodes(fn) if isinstance(node, ast.Call)]


def _hands_a_sweep_to_a_thread(call: ast.Call) -> bool:
    """Without in_thread, which a cancel does not wait for: named, or inside a partial or
    a lambda."""
    return _called(call.func) in OFFLOADS and any(
        isinstance(node, (ast.Name, ast.Attribute)) and _called(node) in SWEEPS
        for arg in call.args
        for node in ast.walk(arg)
    )


def test_no_coroutine_sweeps_or_hands_over_on_the_loop():
    trees = {
        path: ast.parse(path.read_text(encoding="utf-8")) for path in sorted(SRC.rglob("*.py"))
    }
    nodes = [(path, node) for path, tree in trees.items() for node in ast.walk(tree)]
    on_loop = [
        f"{path.relative_to(SRC)}:{call.lineno} {fn.name}"
        for path, fn in nodes
        if isinstance(fn, ast.AsyncFunctionDef)
        for call in _calls(fn)
        if _called(call.func) in SWEEPS
    ]
    abandoned = [
        f"{path.relative_to(SRC)}:{call.lineno}"
        for path, call in nodes
        if isinstance(call, ast.Call) and _hands_a_sweep_to_a_thread(call)
    ]

    assert on_loop == []
    assert abandoned == []


def test_the_check_sees_a_call_on_the_loop():
    tree = ast.parse("async def run(path, user):\n    hand_over(path, user)\n")
    fn = tree.body[0]
    assert isinstance(fn, ast.AsyncFunctionDef)

    assert [_called(c.func) for c in _calls(fn)] == ["hand_over"]


@pytest.mark.asyncio
async def test_a_cancelled_run_waits_for_its_thread():
    """Else the run's cleanup (removing the directory it walks) runs beside it."""
    entered, release = threading.Event(), threading.Event()
    finished: list[bool] = []

    def walk() -> None:
        entered.set()
        release.wait(30)
        finished.append(True)

    run = asyncio.create_task(in_thread(walk))
    await asyncio.to_thread(entered.wait, 30)
    run.cancel()
    # Released from outside the loop, late enough that a cancel let through has landed.
    threading.Timer(1.0, release.set).start()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert finished == [True]


@pytest.mark.asyncio
async def test_a_second_cancel_waits_for_the_thread_too():
    entered, release = threading.Event(), threading.Event()
    finished: list[bool] = []

    def walk() -> None:
        entered.set()
        release.wait(30)
        finished.append(True)

    run = asyncio.create_task(in_thread(walk))
    await asyncio.to_thread(entered.wait, 30)
    run.cancel()
    await asyncio.sleep(0)
    run.cancel()
    threading.Timer(1.0, release.set).start()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert finished == [True]


@pytest.mark.asyncio
async def test_a_cancel_is_raised_whatever_the_thread_raised():
    entered, release = threading.Event(), threading.Event()

    def walk() -> None:
        entered.set()
        release.wait(30)
        raise OSError("work/ is a link")

    run = asyncio.create_task(in_thread(walk))
    await asyncio.to_thread(entered.wait, 30)
    run.cancel()
    threading.Timer(1.0, release.set).start()

    with pytest.raises(asyncio.CancelledError):
        await run


@pytest.mark.asyncio
async def test_a_cancel_from_an_anyio_scope_waits_for_the_thread():
    """It waits; that it does not spin meanwhile (the shield scope) is not pinned here."""
    import anyio

    release = threading.Event()
    finished: list[bool] = []

    def walk() -> None:
        release.wait(30)
        finished.append(True)

    with anyio.CancelScope() as scope:
        threading.Timer(1.0, release.set).start()
        scope.cancel()
        await in_thread(walk)

    assert finished == [True]


@pytest.mark.parametrize(
    "offload",
    [
        "asyncio.to_thread(hand_over, path)",
        "anyio.to_thread.run_sync(functools.partial(context.write_manifest, path))",
        "anyio.to_thread.run_sync(lambda: hand_over(path, user))",
    ],
)
def test_the_check_sees_a_sweep_handed_to_a_bare_thread(offload):
    tree = ast.parse(f"async def run(path, user):\n    await {offload}\n")
    (call,) = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and _called(n.func) in OFFLOADS]

    assert _hands_a_sweep_to_a_thread(call)

"""The harness user's sweep and hand-over run off the event loop.

``sweep_leftovers`` reads every process on the host, up to twenty times, and
``hand_over`` re-owns a whole run directory: on the loop, every other notebook and
socket the server holds waits for them on each run.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "strata"

# Each one sweeps, re-owns a tree, or both (``write_manifest`` makes the work dir).
BLOCKING = {"hand_over", "prepare_work_dir", "sweep_leftovers", "write_manifest"}


def _called(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _own_nodes(fn: ast.AsyncFunctionDef):
    """The coroutine's own nodes: not those of a function defined in it."""
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stack.append(child)


def test_no_coroutine_sweeps_or_hands_over_on_the_loop():
    on_loop = []
    for path in sorted(SRC.rglob("*.py")):
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for node in _own_nodes(fn):
                if isinstance(node, ast.Call) and _called(node.func) in BLOCKING:
                    on_loop.append(f"{path.relative_to(SRC)}:{node.lineno} {fn.name}")

    assert on_loop == []


def test_the_check_sees_a_call_on_the_loop():
    tree = ast.parse("async def run(path, user):\n    hand_over(path, user)\n")
    fn = tree.body[0]
    assert isinstance(fn, ast.AsyncFunctionDef)

    assert [_called(n.func) for n in _own_nodes(fn) if isinstance(n, ast.Call)] == ["hand_over"]

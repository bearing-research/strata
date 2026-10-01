"""Every artifact write records the source that produced it.

``store_cell_output`` takes ``source`` as optional, so any call site can omit it, and a
lineage view with code for some cells and blanks for others misleads. Static rather than
behavioural, so it also fails on call sites that do not exist yet.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "strata"


def _store_cell_output_calls() -> list[tuple[Path, ast.Call]]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "store_cell_output":
                calls.append((path, node))
    return calls


def test_every_store_cell_output_call_passes_source():
    missing = [
        f"{path.relative_to(SRC)}:{call.lineno}"
        for path, call in _store_cell_output_calls()
        if not any(kw.arg == "source" for kw in call.keywords)
    ]

    assert not missing, (
        "these artifact writes record no source, so anything reading their "
        "lineage sees a blank where the code should be: " + ", ".join(missing)
    )


def test_the_guard_is_looking_at_something():
    """A rglob that matches nothing would make the test above vacuous."""
    calls = _store_cell_output_calls()
    assert len(calls) >= 8, f"expected the known call sites, found {len(calls)}"

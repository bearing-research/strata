"""Import bindings of a cell; its defines and references come from ``cellgraph.python``."""

from __future__ import annotations

import ast


def imported_names(source: str) -> set[str]:
    """Top-level names bound by ``import`` statements in *source*.

    A consumer missing one can re-import it, unlike a missing data artifact, so
    this picks the log level for an absent upstream artifact. Binding rule
    matches cellgraph's Python analyzer; star imports contribute nothing.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name.split(".")[0])
    return names

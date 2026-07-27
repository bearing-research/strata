"""IR -> IR optimizer passes over a compiled pipeline.

Passes here rewrite the *physical* plan while preserving *logical* identity:
a node's provenance is a function of its source + inputs + env, and these
passes never touch those, so a cache hit stays a cache hit and experiment ->
production data parity is unchanged. All they change is how much work the
backend does to produce the byte-identical result.

First pass: predicate pushdown into the Iceberg scan. When a node filters a
``@table`` input with a literal comparison, lift that predicate onto the scan's
:class:`~strata.pipeline.ir.TableRef` so ``planner.py`` can prune files and row
groups — reusing Strata's core pruning, which no rented runtime can do. This is
the narrow first slice: it recognizes ``table[table.col <op> literal]`` where
``table`` is a table-input variable and ``<op>`` is a single comparison against
a constant. Widening the recognizer (``&``/``|``, string masks, projection) is
the relational-lift phase; the wire it rides on is here.
"""

from __future__ import annotations

import ast

from strata.pipeline.ir import PipelineIR, PipelineNode, ScanFilter

# ast comparison node -> the FilterOp string the planner understands.
_AST_OP_TO_STR: dict[type[ast.cmpop], str] = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}


def pushdown_scan_filters(ir: PipelineIR) -> PipelineIR:
    """Return *ir* with recognizable ``@table`` filters lifted onto their scans.

    Pure and semantics-preserving: returns *ir* unchanged when nothing lifts, or
    a copy whose affected nodes carry the pushed-down predicates on their
    :class:`TableRef`. Node sources are left intact — the predicate is additive
    metadata the renderer/planner honors, not a rewrite of the user's code.
    """
    new_nodes: list[PipelineNode] = []
    changed = False
    for node in ir.nodes:
        lifted = _lift_node(node)
        changed = changed or lifted is not node
        new_nodes.append(lifted)
    if not changed:
        return ir
    return ir.model_copy(update={"nodes": new_nodes})


def _lift_node(node: PipelineNode) -> PipelineNode:
    """Lift filters on *node*'s table inputs; return *node* unchanged if none."""
    if not node.tables:
        return node
    table_names = {t.name for t in node.tables}
    try:
        tree = ast.parse(node.source)
    except SyntaxError:
        return node

    lifted: dict[str, list[ScanFilter]] = {}
    for sub in ast.walk(tree):
        match = _match_table_filter(sub, table_names)
        if match is not None:
            name, scan_filter = match
            lifted.setdefault(name, []).append(scan_filter)
    if not lifted:
        return node

    new_tables = [
        table.model_copy(update={"filters": table.filters + lifted[table.name]})
        if table.name in lifted
        else table
        for table in node.tables
    ]
    return node.model_copy(update={"tables": new_tables})


def _match_table_filter(subscript: ast.AST, table_names: set[str]) -> tuple[str, ScanFilter] | None:
    """Match ``table[table.col <op> literal]`` -> ``(table, ScanFilter)`` or None."""
    if not isinstance(subscript, ast.Subscript):
        return None
    base = subscript.value
    if not isinstance(base, ast.Name) or base.id not in table_names:
        return None

    compare = subscript.slice
    if not isinstance(compare, ast.Compare) or len(compare.ops) != 1:
        return None
    left, right = compare.left, compare.comparators[0]
    op_str = _AST_OP_TO_STR.get(type(compare.ops[0]))
    if op_str is None:
        return None

    # Left side must be `table.column` on the same table variable.
    if not (
        isinstance(left, ast.Attribute)
        and isinstance(left.value, ast.Name)
        and left.value.id == base.id
    ):
        return None
    # Right side must be a scalar literal.
    if not isinstance(right, ast.Constant) or not isinstance(right.value, (int, float, str)):
        return None

    return base.id, ScanFilter(column=left.attr, op=op_str, value=right.value)

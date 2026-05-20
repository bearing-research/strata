"""Source-backed export of reusable top-level notebook code.

Decides whether a cell can be treated as a synthetic Python module for
cross-cell reuse of top-level ``def``/``class`` definitions.

The cell source is sliced before validation: nodes that aren't safe to
re-execute in a clean namespace (calls, control flow, runtime
assignments) are dropped, and only the kept slice is what we propose to
re-execute. A def/class still needs all the names it references to
resolve inside the slice (or in builtins), so we run a free-variable
pass with the stdlib ``symtable`` module to detect closures over
runtime-only names.

This lifts the older "the *whole cell* must be pure" rule to "the
*defs/classes you want to share* must be self-contained." Cells that
mix runtime work and library code can now export the library code
cleanly; runtime values that downstream cells consume continue through
the regular artifact path.

Limitations
-----------

The slice has *single-cell scope*. The synthetic module is built from
exactly one cell's source — there is no transitive composition across
cells. Concretely:

* A def can't reference a name imported in a different cell.
  ``import math`` in cell 1 doesn't make ``math`` visible to a def
  exported from cell 2 — the def has to live in the same cell as the
  import. (Tested:
  ``test_def_referencing_cross_cell_import_blocks_export``.)

* A def can't call a helper function defined in a different cell.
  Move the helper into the same cell as its caller, or duplicate it.
  (Tested: ``test_def_referencing_cross_cell_helper_blocks_export``.)

* Type annotations *do* participate in the free-variable check on
  Python ≤ 3.13 by default — ``def f(x: SomeType): ...`` blocks
  export when ``SomeType`` isn't bound in the slice. Adding
  ``from __future__ import annotations`` to the cell relaxes this:
  PEP 563 stringifies annotations and ``symtable`` correctly drops
  them from the reference set.

  On Python 3.14+ this check is naturally relaxed for every cell:
  PEP 749 makes annotations lazily evaluated by default, so
  ``symtable`` no longer reports annotation references as free
  variables and no future-import is needed. ``transform.__annotations__``
  access can still NameError lazily if downstream code reaches for
  it; the export check covers module-load safety, not annotation
  access. (Tested:
  ``test_annotation_reference_blocks_without_future_import`` and
  ``test_future_annotations_relaxes_annotation_check``.)

* Slicing reformats the source via ``ast.unparse``. Sliced cells lose
  comments and exact whitespace in the *synthetic module*; the cell's
  source on disk is untouched. Pure module cells — those that pass
  through unsliced — keep their bytes verbatim. (Tested:
  ``test_pure_cell_keeps_original_source_bytes`` and
  ``test_sliced_source_loses_comments_in_synthetic_module``.)

* Lambda assignments are blocked even though ``cloudpickle`` could
  serialize a lambda value. The synthetic-module path is for
  source-backed library code; lambdas express runtime behavior and
  would surprise downstream consumers if they rode that path
  silently. (Tested: ``test_top_level_lambda_assignment_is_blocking``.)

* Star imports (``from foo import *``) are dropped from the slice and
  surfaced as a reason. The slice can't validate the names they would
  bind. (Tested: ``test_star_import_is_blocked``.)
"""

from __future__ import annotations

import ast
import builtins as _python_builtins
import symtable
from dataclasses import dataclass, field
from typing import TypedDict


class ModuleExportEntry(TypedDict):
    """Wire shape for one entry in a cell's ``module_exports`` list."""

    name: str
    kind: str


_BUILTIN_NAMES: frozenset[str] = frozenset(dir(_python_builtins))


@dataclass(frozen=True)
class ExportedSymbol:
    """One exportable top-level symbol."""

    name: str
    kind: str


@dataclass(frozen=True)
class ModuleExportPlan:
    """Validated module-export plan for a cell source string."""

    module_source: str
    exported_symbols: dict[str, ExportedSymbol] = field(default_factory=dict)
    unsupported_symbols: set[str] = field(default_factory=set)
    blocking_symbols: set[str] = field(default_factory=set)
    unsupported_reasons: list[str] = field(default_factory=list)
    # True when the slicer dropped any node from the cell. Pure module
    # cells (no drops) keep this False so callers that want the strict
    # "this is library code, nothing else" signal — like the UI pill —
    # can gate on ``not sliced``.
    sliced: bool = False

    @property
    def is_exportable(self) -> bool:
        return not self.unsupported_reasons

    def format_error(self) -> str:
        """Return a user-facing reason string for unsupported module export."""
        if not self.unsupported_reasons:
            return ""
        return "; ".join(self.unsupported_reasons)


def build_module_export_plan(source: str) -> ModuleExportPlan:
    """Validate a cell source and produce an export plan.

    Slices ``source`` to keep only nodes that re-execute safely in a
    clean module namespace (docstring, imports, defs, async defs,
    classes, literal-constant assignments). Everything else is dropped.

    The slice is then validated with ``symtable``: any def/class whose
    body, decorators, default values, base classes, or class-body
    statements reference names not bound in the slice (and not Python
    builtins) is moved out of ``exported_symbols`` into
    ``blocking_symbols`` with a precise reason. Lambda assignments to
    names also act as hard blockers — cloudpickle can serialize a
    lambda value, but treating ``f = lambda x: ...`` as cross-cell
    library code would be misleading.

    Cells whose only "drop" is benign runtime state (``df = load()``)
    keep ``is_exportable = True``; the runtime variable just flows
    through the regular artifact path and the slice carries the
    library code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return ModuleExportPlan(
            module_source=source,
            unsupported_reasons=[f"invalid syntax: {exc.msg}"],
        )

    keep_nodes: list[ast.stmt] = []
    drop_nodes: list[ast.stmt] = []
    star_import_dropped = False
    blocking_lambda_names: set[str] = set()

    for index, node in enumerate(tree.body):
        if _is_module_docstring(node, index):
            keep_nodes.append(node)
            continue

        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            # Star imports bind unknown names; the slice can't validate
            # against them, so we never keep them. Surface a reason so
            # the UI explains why.
            star_import_dropped = True
            drop_nodes.append(node)
            continue

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_nodes.append(node)
            continue

        if _is_literal_constant_assignment(node):
            keep_nodes.append(node)
            continue

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep_nodes.append(node)
            continue

        # Drop the node. Track lambda-assignments specifically — those
        # bind names that look like library code, so downstream
        # consumption should fail loudly.
        drop_nodes.append(node)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            for target in node.targets:
                blocking_lambda_names.update(_target_names(target))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Lambda):
            blocking_lambda_names.update(_target_names(node.target))

    sliced = len(drop_nodes) > 0
    slice_source = _emit_slice_source(keep_nodes, original=source)

    exported_symbols: dict[str, ExportedSymbol] = {}
    unsupported_symbols: set[str] = set(blocking_lambda_names)
    blocking_symbols: set[str] = set(blocking_lambda_names)
    unsupported_reasons: list[str] = []
    module_load_unresolved: set[str] = set()

    kind_map: dict[str, str] = {}
    for node in keep_nodes:
        if isinstance(node, ast.FunctionDef):
            kind_map[node.name] = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            kind_map[node.name] = "async function"
        elif isinstance(node, ast.ClassDef):
            kind_map[node.name] = "class"

    # Divergence check: if a name bound by the slice is *also* rebound
    # by a dropped statement at module scope, the synthetic module's
    # value diverges from the cell's final state. Common case:
    # ``def f(): ...; f = wrap(f)`` — slice exports the unwrapped ``f``
    # while the cell's runtime ``f`` is wrapped.
    kept_bindings = _kept_bindings(keep_nodes)
    dropped_bindings: set[str] = set()
    for node in drop_nodes:
        dropped_bindings.update(_module_bindings_in(node))
    divergent = kept_bindings & dropped_bindings
    if divergent:
        for name in divergent:
            unsupported_symbols.add(name)
            if name in kind_map:
                blocking_symbols.add(name)
        unsupported_reasons.append(
            "names reassigned at runtime would diverge from the slice's value: "
            f"{', '.join(sorted(divergent))}"
        )

    # Slice-level free-variable analysis. Only needed when the slice
    # contains at least one def/class — literal-only slices have no
    # free-var concern.
    has_def_or_class = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in keep_nodes
    )
    if has_def_or_class:
        try:
            module_table = symtable.symtable(slice_source, "<slice>", "exec")
        except SyntaxError as exc:
            # ``ast.unparse`` should always produce parseable text; if
            # it ever doesn't, surface the failure rather than silently
            # exporting broken source.
            return ModuleExportPlan(
                module_source=slice_source,
                unsupported_reasons=[f"sliced source did not re-parse: {exc.msg}"],
                sliced=sliced,
            )

        module_locals = {sym.get_name() for sym in module_table.get_symbols() if sym.is_local()}

        # Names referenced at module scope but not bound there:
        # decorators, default values, base classes — all evaluated at
        # module load.
        for sym in module_table.get_symbols():
            name = sym.get_name()
            if sym.is_referenced() and not sym.is_local() and name not in _BUILTIN_NAMES:
                module_load_unresolved.add(name)

        if module_load_unresolved:
            unsupported_reasons.append(
                "top-level expressions reference names not defined or imported in "
                f"this cell: {', '.join(sorted(module_load_unresolved))}"
            )

        for child in module_table.get_children():
            if child.get_type() not in ("function", "class"):
                continue
            symbol_name = child.get_name()
            unresolved = _scope_unresolved(child, module_locals)
            if unresolved:
                unsupported_symbols.add(symbol_name)
                blocking_symbols.add(symbol_name)
                kind_word = kind_map.get(symbol_name, child.get_type())
                unsupported_reasons.append(
                    f"{kind_word} `{symbol_name}` references names not defined or imported in "
                    f"this cell: {', '.join(sorted(unresolved))}"
                )
                continue
            if module_load_unresolved:
                # Module-load failure poisons every symbol — the synthetic
                # module's ``exec`` would raise before binding any of them.
                unsupported_symbols.add(symbol_name)
                blocking_symbols.add(symbol_name)
                continue
            if symbol_name in divergent:
                # Name diverges from runtime; don't export.
                continue
            exported_symbols[symbol_name] = ExportedSymbol(
                symbol_name, kind_map.get(symbol_name, child.get_type())
            )

    # Literal-constant assignments ride alongside any kept defs/classes.
    # Skip them when the slice itself can't import (module-load free
    # vars unresolved), and skip names that diverge with runtime drops.
    if not module_load_unresolved:
        for node in keep_nodes:
            if _is_literal_constant_assignment(node):
                for name in _target_names_for_assignment(node):
                    if name in divergent:
                        continue
                    exported_symbols.setdefault(name, ExportedSymbol(name, "constant"))

    if blocking_lambda_names:
        unsupported_reasons.append("top-level lambdas are not shareable across cells")

    if star_import_dropped:
        unsupported_reasons.append("star imports are not supported for cross-cell code export")

    return ModuleExportPlan(
        module_source=slice_source,
        exported_symbols=exported_symbols,
        unsupported_symbols=unsupported_symbols,
        blocking_symbols=blocking_symbols,
        unsupported_reasons=unsupported_reasons,
        sliced=sliced,
    )


def _emit_slice_source(keep_nodes: list[ast.stmt], *, original: str) -> str:
    """Return the slice as runnable Python source.

    Falls back to the original source when no slicing happened — this
    preserves the user's exact bytes (comments, formatting) for the
    common pure-cell case where ``ast.unparse`` would otherwise reformat
    them.
    """
    if not keep_nodes:
        return ""
    try:
        tree = ast.parse(original)
    except SyntaxError:
        # Shouldn't happen — caller already parsed once — but keep this
        # safe.
        body = ast.unparse(ast.Module(body=keep_nodes, type_ignores=[]))
        return body if body.endswith("\n") else f"{body}\n"

    if len(keep_nodes) == len(tree.body):
        return original if original.endswith("\n") else f"{original}\n"

    body = ast.unparse(ast.Module(body=keep_nodes, type_ignores=[]))
    return body if body.endswith("\n") else f"{body}\n"


def _scope_unresolved(scope: symtable.SymbolTable, module_locals: set[str]) -> set[str]:
    """Return names referenced in *scope* that resolve via module
    globals but aren't bound in the slice's module locals.

    For function scopes this represents call-time NameErrors; for class
    scopes it represents module-load-time NameErrors when the class
    body executes.

    Skips ``is_free()`` symbols — those are closure variables that
    Python has already resolved to an enclosing scope's binding, so
    they're guaranteed to exist at runtime even though they're not in
    module_locals.
    """
    missing: set[str] = set()
    for sym in scope.get_symbols():
        name = sym.get_name()
        if (
            sym.is_referenced()
            and not sym.is_local()
            and not sym.is_parameter()
            and not sym.is_free()
            and name not in module_locals
            and name not in _BUILTIN_NAMES
        ):
            missing.add(name)
    for inner in scope.get_children():
        if inner.get_type() in ("function", "class"):
            missing.update(_scope_unresolved(inner, module_locals))
    return missing


def _is_literal_constant_assignment(node: ast.AST) -> bool:
    """True when *node* is a top-level assignment of a literal value."""
    if isinstance(node, ast.Assign):
        if not all(isinstance(t, (ast.Name, ast.Tuple, ast.List)) for t in node.targets):
            return False
        return _is_literal_value(node.value)
    if isinstance(node, ast.AnnAssign):
        if node.value is None or not isinstance(node.target, ast.Name):
            return False
        return _is_literal_value(node.value)
    return False


def _is_literal_value(node: ast.expr) -> bool:
    """Return True for compile-time-constant expressions."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.USub, ast.UAdd, ast.Invert, ast.Not)
    ):
        return _is_literal_value(node.operand)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_value(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            key is not None and _is_literal_value(key) and _is_literal_value(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    return False


def _target_names_for_assignment(node: ast.AST) -> list[str]:
    """Collect names bound by a literal-constant assignment."""
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            for name in sorted(_target_names(target)):
                names.append(name)
        return names
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _is_module_docstring(node: ast.stmt, index: int) -> bool:
    """Return whether *node* is the module docstring expression."""
    if index != 0 or not isinstance(node, ast.Expr):
        return False
    value = node.value
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _target_names(target: ast.expr) -> set[str]:
    """Extract assigned names from an assignment target."""
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_target_names(item))
        return names
    return set()


def _kept_bindings(keep_nodes: list[ast.stmt]) -> set[str]:
    """Names the slice binds at module scope."""
    bindings: set[str] = set()
    for node in keep_nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bindings.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            bindings.update(_target_names_for_assignment(node))
    return bindings


def _module_bindings_in(node: ast.stmt) -> set[str]:
    """Names bound at module scope by *node* (a dropped top-level
    statement). Recurses into control-flow bodies but stops at function
    and class scopes — those bind locally, not at module scope.
    """
    bindings: set[str] = set()
    if isinstance(node, ast.Assign):
        for target in node.targets:
            bindings.update(_target_names(target))
    elif isinstance(node, ast.AnnAssign):
        bindings.update(_target_names(node.target))
    elif isinstance(node, ast.AugAssign):
        bindings.update(_target_names(node.target))
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        bindings.update(_target_names(node.target))
        for sub in node.body:
            bindings.update(_module_bindings_in(sub))
        for sub in node.orelse:
            bindings.update(_module_bindings_in(sub))
    elif isinstance(node, (ast.With, ast.AsyncWith)):
        for item in node.items:
            if item.optional_vars is not None:
                bindings.update(_target_names(item.optional_vars))
        for sub in node.body:
            bindings.update(_module_bindings_in(sub))
    elif isinstance(node, (ast.If, ast.While)):
        for sub in node.body:
            bindings.update(_module_bindings_in(sub))
        for sub in node.orelse:
            bindings.update(_module_bindings_in(sub))
    elif isinstance(node, ast.Try):
        for sub in node.body + node.orelse + node.finalbody:
            bindings.update(_module_bindings_in(sub))
        for handler in node.handlers:
            if handler.name is not None:
                bindings.add(handler.name)
            for sub in handler.body:
                bindings.update(_module_bindings_in(sub))
    elif isinstance(node, ast.Match):
        for case in node.cases:
            for sub in case.body:
                bindings.update(_module_bindings_in(sub))
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            bindings.update(_target_names(target))
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        # Nested inside a dropped block — its name still binds at
        # module scope when the block executes.
        bindings.add(node.name)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            bindings.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                continue
            bindings.add(alias.asname or alias.name)
    return bindings

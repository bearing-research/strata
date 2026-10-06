"""Source-backed export of reusable top-level notebook code.

Decides whether a cell's top-level ``def``/``class`` definitions can be shared
across cells as a synthetic module. The source is sliced to nodes that are safe
to re-execute in a clean namespace, and a ``symtable`` free-variable pass checks
that each shared definition resolves inside the slice.

Limitations:

* Single-cell scope: a def cannot use a name imported, or a helper defined,
  in another cell.
* Annotations count as references unless the cell has
  ``from __future__ import annotations``. This is an explicit AST walk, since
  symtable stops reporting annotation refs on 3.14 (PEP 749).
* Sliced cells go through ``ast.unparse`` and lose comments in the synthetic
  module; unsliced cells keep their bytes.
* Lambda assignments block export: they are runtime behavior, not library code.
* Star imports are dropped and reported, since their names cannot be validated.
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


# Python 3.14 symtable reports the compiler-generated ``__conditional_annotations__``
# as a referenced global in any scope with annotated assignments; it always resolves.
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(_python_builtins)) | {"__conditional_annotations__"}


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
    # Free names of the exported code that are produced upstream: resolved from the
    # store and injected into the module namespace before exec instead of blocking.
    # Empty unless the caller passes ``injectable``.
    injected_inputs: set[str] = field(default_factory=set)
    unsupported_reasons: list[str] = field(default_factory=list)
    # False for pure module cells, so the UI pill can gate on ``not sliced``.
    sliced: bool = False

    @property
    def is_exportable(self) -> bool:
        return not self.unsupported_reasons

    def format_error(self) -> str:
        """Return a user-facing reason string for unsupported module export."""
        if not self.unsupported_reasons:
            return ""
        return "; ".join(self.unsupported_reasons)


def build_module_export_plan(
    source: str, *, injectable: frozenset[str] = frozenset()
) -> ModuleExportPlan:
    """Validate a cell source and produce an export plan.

    The slice keeps the docstring, imports, defs, classes and literal-constant
    assignments. A def/class referencing a name not bound there (or builtin)
    moves to ``blocking_symbols`` with a reason, as do lambda assignments.
    Benign dropped runtime state (``df = load()``) keeps ``is_exportable``.

    ``injectable`` names (upstream variables) do not block; they are recorded in
    ``injected_inputs`` and hydrated from the artifact store later. Empty, every
    unresolved free name blocks.
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
            # Star imports bind unknown names the slice can't validate; never keep them.
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

        # Lambda assignments look like library code, so consuming them must fail loudly.
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
    injected_inputs: set[str] = set()

    kind_map: dict[str, str] = {}
    for node in keep_nodes:
        if isinstance(node, ast.FunctionDef):
            kind_map[node.name] = "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            kind_map[node.name] = "async function"
        elif isinstance(node, ast.ClassDef):
            kind_map[node.name] = "class"

    # A slice-bound name also rebound by a dropped statement diverges from the
    # cell's final state, e.g. ``def f(): ...; f = wrap(f)``.
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

    # Literal-only slices have no free-variable concern.
    has_def_or_class = any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for n in keep_nodes
    )
    if has_def_or_class:
        try:
            module_table = symtable.symtable(slice_source, "<slice>", "exec")
        except SyntaxError as exc:
            # If ``ast.unparse`` ever yields unparseable text, surface it rather than
            # exporting broken source.
            return ModuleExportPlan(
                module_source=slice_source,
                unsupported_reasons=[f"sliced source did not re-parse: {exc.msg}"],
                sliced=sliced,
            )

        module_locals = {sym.get_name() for sym in module_table.get_symbols() if sym.is_local()}

        # Unbound at module scope: decorators, defaults, base classes (all evaluated at load).
        for sym in module_table.get_symbols():
            name = sym.get_name()
            if sym.is_referenced() and not sym.is_local() and name not in _BUILTIN_NAMES:
                module_load_unresolved.add(name)

        # Walk annotations explicitly: symtable reports them as free vars before 3.14
        # but not under PEP 749, so this keeps the check version-independent. Under
        # ``from __future__ import annotations`` they are never evaluated, so skip.
        if not _has_future_annotations(tree):
            annotation_refs = _collect_annotation_names(keep_nodes)
            annotation_unresolved = {
                name
                for name in annotation_refs
                if name not in module_locals and name not in _BUILTIN_NAMES
            }
            module_load_unresolved |= annotation_unresolved

        # Upstream-produced names get hydrated before exec; only the rest block.
        module_load_hard = module_load_unresolved - injectable
        injected_inputs |= module_load_unresolved & injectable

        if module_load_hard:
            unsupported_reasons.append(
                "top-level expressions reference names not defined or imported in "
                f"this cell: {', '.join(sorted(module_load_hard))}"
            )

        for child in module_table.get_children():
            if child.get_type() not in ("function", "class"):
                continue
            symbol_name = child.get_name()
            unresolved = _scope_unresolved(child, module_locals)
            unresolved_hard = unresolved - injectable
            if unresolved_hard:
                unsupported_symbols.add(symbol_name)
                blocking_symbols.add(symbol_name)
                kind_word = kind_map.get(symbol_name, child.get_type())
                unsupported_reasons.append(
                    f"{kind_word} `{symbol_name}` references names not defined or imported in "
                    f"this cell: {', '.join(sorted(unresolved_hard))}"
                )
                continue
            if module_load_hard:
                # The module's ``exec`` would raise before binding any symbol.
                unsupported_symbols.add(symbol_name)
                blocking_symbols.add(symbol_name)
                continue
            if symbol_name in divergent:
                continue
            # Hard set is empty here, so the rest are injectable.
            injected_inputs |= unresolved
            exported_symbols[symbol_name] = ExportedSymbol(
                symbol_name, kind_map.get(symbol_name, child.get_type())
            )

    # Literal constants ride along with kept defs/classes, unless the slice can't
    # import or the name diverges with runtime drops.
    if not (module_load_unresolved - injectable):
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
        injected_inputs=injected_inputs,
        sliced=sliced,
    )


def runtime_binding_names(source: str) -> frozenset[str]:
    """Names a cell binds at module scope via runtime code (what the slicer drops).

    Candidates for same-cell hydration: callers pass them (with cross-cell
    producers) as ``injectable`` to :func:`build_module_export_plan`.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return frozenset()
    kept = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, kept) or _is_literal_constant_assignment(node):
            continue
        names |= _module_bindings_in(node)
    return frozenset(names)


def _emit_slice_source(keep_nodes: list[ast.stmt], *, original: str) -> str:
    """Return the slice as runnable Python source; the original bytes when nothing was sliced."""
    if not keep_nodes:
        return ""
    try:
        tree = ast.parse(original)
    except SyntaxError:
        # The caller already parsed once; stay safe anyway.
        body = ast.unparse(ast.Module(body=keep_nodes, type_ignores=[]))
        return body if body.endswith("\n") else f"{body}\n"

    if len(keep_nodes) == len(tree.body):
        return original if original.endswith("\n") else f"{original}\n"

    body = ast.unparse(ast.Module(body=keep_nodes, type_ignores=[]))
    return body if body.endswith("\n") else f"{body}\n"


def _scope_unresolved(scope: symtable.SymbolTable, module_locals: set[str]) -> set[str]:
    """Names in *scope* that resolve via module globals but are not bound in the slice.

    These would raise NameError at call time (functions) or load time (classes).
    ``is_free()`` closure variables are skipped.
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


def _has_future_annotations(tree: ast.Module) -> bool:
    """True if the source has ``from __future__ import annotations`` in effective position.

    Walking stops at the first non-``__future__`` statement after any docstring,
    matching PEP 563's rule.
    """
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            # A module docstring may precede __future__ imports.
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                if alias.name == "annotations":
                    return True
            continue
        # Any other statement terminates the future-import block.
        break
    return False


def _collect_annotation_names(nodes: list[ast.stmt]) -> set[str]:
    """Names referenced inside type-annotation expressions in *nodes*, including nested defs.

    Not attributed per symbol: an unresolved annotation is a module-load
    failure, which blocks every exported symbol. The caller filters against
    module locals and builtins.
    """
    names: set[str] = set()

    def _visit_annotation(annotation: ast.expr | None) -> None:
        if annotation is None:
            return
        for sub in ast.walk(annotation):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                names.add(sub.id)

    def _walk(stmt: ast.AST) -> None:
        # Iterate annotation slots by hand: ast.walk alone can't tell an annotation
        # from a body expression.
        for sub in ast.walk(stmt):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = sub.args
                for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
                    _visit_annotation(arg.annotation)
                if args.vararg is not None:
                    _visit_annotation(args.vararg.annotation)
                if args.kwarg is not None:
                    _visit_annotation(args.kwarg.annotation)
                _visit_annotation(sub.returns)
            elif isinstance(sub, ast.AnnAssign):
                _visit_annotation(sub.annotation)

    for node in nodes:
        _walk(node)

    return names


def _module_bindings_in(node: ast.stmt) -> set[str]:
    """Names a dropped top-level statement binds at module scope.

    Recurses into control-flow bodies but not into function or class scopes.
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
        # Nested in a dropped block, it still binds at module scope when the block runs.
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

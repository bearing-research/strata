"""AST-based variable analysis for notebook cells."""

from __future__ import annotations

import ast
import builtins
import symtable
from dataclasses import dataclass, field


@dataclass
class CellAnalysis:
    """Result of analyzing a single cell.

    Attributes:
        defines: List of top-level variable names defined by this cell
        references: List of free variable names referenced but not defined in this cell
        error: Error message if analysis failed (None if successful)
    """

    defines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    # Defines from in-place mutation (``df["col"] = ...``). The harness forces
    # serialization of these even when the mutation preserved ``id()``.
    mutation_defines: list[str] = field(default_factory=list)
    # Free names that are builtins, kept out of ``references`` so every
    # ``print``/``len`` caller doesn't list them. A cell can still define
    # ``input = ...``, so the DAG resolves these against producers too: an
    # unshadowed builtin wires nothing, a shadowed one gets its edge.
    builtin_references: list[str] = field(default_factory=list)
    error: str | None = None


def imported_names(source: str) -> set[str]:
    """Top-level names bound by ``import`` statements in *source*.

    These bindings are re-importable by name in any cell sharing the venv, so
    a consuming cell that's missing one can simply re-import it — unlike a data
    artifact, whose absence is a real materialisation gap. Used to pick the log
    level when an upstream variable's artifact is unexpectedly absent. Matches
    the binding rule in :meth:`VariableAnalyzer.visit_Import` (``asname`` or the
    imported name); star imports contribute nothing.
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
                names.add(alias.asname or alias.name)
    return names


def _collect_name_targets(target: ast.expr, out: set[str]) -> None:
    """Recursively collect Name ids from an assignment target.

    Handles plain ``x``, tuple/list unpacking (``a, b`` or ``[a, b]``),
    and starred targets (``*rest``). Subscript / attribute targets are
    ignored — they're mutations, not pure binds, and travel through a
    separate code path.
    """
    if isinstance(target, ast.Name):
        out.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _collect_name_targets(elt, out)
    elif isinstance(target, ast.Starred):
        _collect_name_targets(target.value, out)


def _has_inplace_true(keywords: list[ast.keyword]) -> bool:
    """Return whether a call's keywords contain a literal ``inplace=True``."""
    return any(
        kw.arg == "inplace" and isinstance(kw.value, ast.Constant) and kw.value.value is True
        for kw in keywords
    )


class VariableAnalyzer(ast.NodeVisitor):
    """AST visitor that collects defined and referenced variables."""

    def __init__(self):
        """Initialize the analyzer."""
        self.defines: set[str] = set()
        self.references: set[str] = set()
        # Pure ``x = ...`` targets. ``df = ...`` then ``df["col"] = ...`` in
        # one cell is a local define, not a mutation of an upstream.
        self.pure_defines: set[str] = set()
        # Subscript/attribute mutations. Unlike pure rebinds these stay in
        # references: the cell depends on an upstream producer.
        self.mutation_defines: set[str] = set()
        # ``df = df.dropna()`` style: the RHS read is a genuine upstream
        # reference and survives the pure-define filter.
        self.rebind_with_self_read: set[str] = set()
        # Pure defines so far in source order: tells ``x = 0\nx += 1``
        # (local) from a bare ``x += 1`` (upstream read).
        self._defined_so_far: set[str] = set()
        self._in_nested_scope = False
        self._local_vars: set[str] = set()  # Track local scope variables
        # PEP 563 stringifies annotations, so under it they are not
        # runtime references.
        self._future_annotations: bool = False

    def visit_Module(self, node: ast.Module) -> None:
        """Visit module — process top-level statements."""
        self._future_annotations = any(
            isinstance(s, ast.ImportFrom)
            and s.module == "__future__"
            and any(alias.name == "annotations" for alias in s.names)
            for s in node.body
        )
        for child in node.body:
            self.visit(child)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handle: x = ... or x, y = ..."""
        # Needed before visiting the RHS to catch read-before-write, including
        # swaps like ``a, b = b, a``.
        pure_target_names: set[str] = set()
        for target in node.targets:
            _collect_name_targets(target, pure_target_names)
        all_underscore = all(self._is_pure_underscore(target) for target in node.targets)
        if not all_underscore:
            # A target read on its own RHS is an upstream reference unless
            # the cell pure-defined it earlier (``x = 0; x = x + 1``).
            for child in ast.walk(node.value):
                if (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in pure_target_names
                    and child.id not in self._defined_so_far
                ):
                    self.rebind_with_self_read.add(child.id)
            self.visit(node.value)
        for target in node.targets:
            self._add_assign_target(target)
            _collect_name_targets(target, self._defined_so_far)

    def _is_pure_underscore(self, target: ast.expr) -> bool:
        """Check if a target is a pure _ (not part of unpacking)."""
        if isinstance(target, ast.Name):
            return target.id == "_"
        return False

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handle: x += ... or df["col"] += ...

        Augmented assignment is ``x = x + value`` desugared: the LHS is
        implicitly read before being written. Same class of bug as the
        pure-rebind case in ``visit_Assign`` — without explicit tracking
        the DAG misses the upstream edge to whoever produced ``x``
        first, and the cell hits NameError at runtime.

        For a Name target the implicit read isn't a visible AST node
        (the target sits in Store context), so we add the name to
        ``self.references`` and ``rebind_with_self_read`` directly —
        but only if the cell hasn't already pure-defined that name
        earlier in source order. Subscript / attribute targets
        (``df["col"] += 1``) flow through ``_add_assign_target`` →
        ``mutation_defines`` and stay in references via that path.
        """
        if isinstance(node.target, ast.Name):
            if node.target.id not in self._defined_so_far:
                self.references.add(node.target.id)
                self.rebind_with_self_read.add(node.target.id)
        self._add_assign_target(node.target)
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._defined_so_far.add(node.target.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle: x: int = ... or x: int (without value)."""
        self._add_assign_target(node.target)
        # Module-scope annotations are evaluated at runtime unless PEP 563.
        if not self._future_annotations:
            self.visit(node.annotation)
        if node.value:
            self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Detect in-place mutation expressed as ``X.method(..., inplace=True)``.

        ``inplace=True`` is the unambiguous pandas idiom for "mutate the
        receiver" (``df.drop`` / ``fillna`` / ``sort_values`` / ``rename`` …).
        Unlike ``df["col"] = …`` it isn't an assignment target, so the analyzer
        would otherwise treat ``df`` as read-only. In the shared-namespace
        batch path that divergence is a correctness bug: a downstream cell
        observes the mutated object while the stored artifact still holds the
        pre-mutation value, so its provenance references stale inputs.

        Flagging the receiver root as a mutation-define routes it through the
        same machinery as subscript mutation — the cell becomes a (re)producer
        of ``df`` in the DAG and serializes the post-mutation value, so
        downstream reads resolve to it in both single-cell and batch execution.
        Only a literal ``inplace=True`` triggers this (not ``inplace=False`` or
        a variable), and only an attribute call on a name/attribute receiver
        (mutating a call result has no cross-cell effect).
        """
        if isinstance(node.func, ast.Attribute) and _has_inplace_true(node.keywords):
            self._add_reference_target(node.func.value)
            self._add_mutation_define(node.func.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handle: def f(): ... — function name is defined, body is nested scope.

        Decorators, default arg values, return annotation, and arg
        annotations all evaluate at module load (the latter only when
        ``from __future__ import annotations`` is *not* set), so any
        free variables in those positions are real module-scope
        references. Body free vars are picked up by the symtable pass.
        """
        self.defines.add(node.name)
        self._visit_function_signature(node)
        # Don't recurse into function body (it's a nested scope)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle: async def f(): ..."""
        self.defines.add(node.name)
        self._visit_function_signature(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle: class C: ... — class name is defined, body is nested scope.

        Decorators, base classes, and class keyword arguments
        (``metaclass=`` etc.) evaluate at module load and are walked
        here. The class body itself is a nested scope; the symtable
        pass picks up its free variables.
        """
        self.defines.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for kw in node.keywords:
            self.visit(kw.value)
        # Don't recurse into class body

    def _visit_function_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Visit decorators, default values, and (when not under PEP 563)
        type annotations of a function definition.
        """
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        if not self._future_annotations:
            if node.returns is not None:
                self.visit(node.returns)
            for arg_list in (node.args.posonlyargs, node.args.args, node.args.kwonlyargs):
                for arg in arg_list:
                    if arg.annotation is not None:
                        self.visit(arg.annotation)
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                self.visit(node.args.kwarg.annotation)

    def visit_Import(self, node: ast.Import) -> None:
        """Handle: import foo or import foo as bar."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.defines.add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from foo import bar or from foo import *."""
        for alias in node.names:
            if alias.name == "*":
                # Star imports: defines are unknowable, skip.
                pass
            else:
                name = alias.asname if alias.asname else alias.name
                self.defines.add(name)

    def visit_For(self, node: ast.For) -> None:
        """Handle: for x in ... — x is defined at top level."""
        self._add_assign_target(node.target)
        self.visit(node.iter)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        """Handle: with ... as x: — x is defined at top level."""
        for item in node.items:
            if item.optional_vars:
                self._add_assign_target(item.optional_vars)
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        """Handle: async with ... as x:"""
        for item in node.items:
            if item.optional_vars:
                self._add_assign_target(item.optional_vars)
            self.visit(item.context_expr)
        for stmt in node.body:
            self.visit(stmt)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        """Handle: except E as e: — e is defined at top level."""
        if node.name:
            self.defines.add(node.name)
        if node.type:
            self.visit(node.type)
        for stmt in node.body:
            self.visit(stmt)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Handle: walrus operator := at top level."""
        if isinstance(node.target, ast.Name):
            self.defines.add(node.target.id)
        self.visit(node.value)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Handle: del x — x is referenced but not defined."""
        for target in node.targets:
            self._add_delete_target(target)

    def _add_delete_target(self, target: ast.expr) -> None:
        """Extract variable names from a del statement target.

        del x → x is referenced
        del x.attr → x is referenced
        del x[key] → x is referenced
        """
        if isinstance(target, ast.Name):
            self.references.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_delete_target(elt)
        elif isinstance(target, ast.Subscript):
            self._add_delete_target(target.value)
            self.visit(target.slice)
        elif isinstance(target, ast.Attribute):
            self._add_delete_target(target.value)

    def visit_Name(self, node: ast.Name) -> None:
        """Collect referenced names (Load context)."""
        if isinstance(node.ctx, ast.Load):
            # Skip names local to the current scope (e.g. lambda params)
            if node.id not in self._local_vars:
                self.references.add(node.id)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """``[elt for x in iter if cond]`` — visit element/conditions with
        loop targets locally scoped, visit iters in outer scope."""
        self._visit_comprehension(node, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """``{elt for x in iter}``."""
        self._visit_comprehension(node, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """``{k: v for x in iter}`` — both key and value are scoped."""
        self._visit_comprehension(node, [node.key, node.value])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """``(elt for x in iter)``."""
        self._visit_comprehension(node, [node.elt])

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        elements: list[ast.expr],
    ) -> None:
        """Walk comp parts with loop variables locally scoped.

        The first generator's iterable runs in the OUTER scope, so it's
        visited without any local additions. Subsequent generators'
        iterables run inside the comprehension's scope (they can see
        prior loop variables), so they're visited under the local
        binding. Element(s) and ``if`` clauses always see all loop
        variables.

        Python 3.13 inlines comprehensions into the enclosing scope
        (PEP 709), so ``symtable`` no longer creates a child scope for
        them — the AST-level scope tracking here is what actually picks
        up free variables in comp elements.
        """
        if not node.generators:
            return

        # First generator's iter runs in the OUTER scope.
        self.visit(node.generators[0].iter)

        old_local_vars = self._local_vars
        try:
            local_vars = set(self._local_vars)
            for gen in node.generators:
                local_vars |= self._extract_target_names(gen.target)
            self._local_vars = local_vars

            # Element(s) and if-clauses see all loop targets.
            for elt in elements:
                self.visit(elt)
            # Subsequent generators' iter expressions can see prior
            # loop targets, so they get visited under the local scope.
            for gen in node.generators[1:]:
                self.visit(gen.iter)
            for gen in node.generators:
                for cond in gen.ifs:
                    self.visit(cond)
        finally:
            self._local_vars = old_local_vars

    def _extract_target_names(self, target: ast.expr) -> set[str]:
        """Names bound by a comprehension or for-loop target."""
        names: set[str] = set()
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                names |= self._extract_target_names(elt)
        elif isinstance(target, ast.Starred):
            names |= self._extract_target_names(target.value)
        return names

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Lambda expression — arguments are local scope."""
        # Defaults evaluate in the enclosing scope, before params shadow.
        for default in node.args.defaults:
            self.visit(default)
        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)
        old_local_vars = self._local_vars
        self._local_vars = self._local_vars | self._get_lambda_params(node.args)
        self.visit(node.body)
        self._local_vars = old_local_vars

    def _get_lambda_params(self, args: ast.arguments) -> set[str]:
        """Extract parameter names from lambda arguments."""
        params = set()
        for arg in args.posonlyargs:
            params.add(arg.arg)
        for arg in args.args:
            params.add(arg.arg)
        if args.vararg:
            params.add(args.vararg.arg)
        for arg in args.kwonlyargs:
            params.add(arg.arg)
        if args.kwarg:
            params.add(args.kwarg.arg)
        return params

    def _add_assign_target(self, target: ast.expr) -> None:
        """Extract variable names from an assignment target.

        Handles:
        - Name: x
        - Tuple/List: (x, y) or [x, y]
        - Subscript: df["col"] → defines the root name df
        - Attribute: obj.attr → defines the root name obj
        """
        if isinstance(target, ast.Name):
            self.defines.add(target.id)
            self.pure_defines.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_assign_target(elt)
        elif isinstance(target, ast.Subscript):
            # df["col"] = ...: reads df and produces the mutated df, so
            # downstream reads route through this cell (else they can run
            # before the mutation and KeyError).
            self._add_reference_target(target.value)
            self._add_mutation_define(target.value)
        elif isinstance(target, ast.Attribute):
            # obj.attr = ...: same as subscript mutation.
            self._add_reference_target(target.value)
            self._add_mutation_define(target.value)
        elif isinstance(target, ast.Starred):
            self._add_assign_target(target.value)

    def _add_reference_target(self, node: ast.expr) -> None:
        """Extract root name from expression and add to references.

        Used for attribute/subscript mutations (e.g. obj.attr = ..., df["col"] = ...)
        which reference the root object but don't define it.
        """
        if isinstance(node, ast.Name):
            self.references.add(node.id)
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            self._add_reference_target(node.value)

    def _add_mutation_define(self, node: ast.expr) -> None:
        """Record the root name of a mutated target as a define.

        ``df["col"] = ...`` or ``obj.attr = ...`` mutates an existing
        object. For DAG purposes the mutating cell is the producer of
        the *post-mutation* view that downstream cells observe, so we
        also treat the root name as a define. The name also stays in
        references via ``_add_reference_target`` — the mutation reads
        the prior value. Tracking it in ``mutation_defines`` tells the
        caller not to strip it from the final references set.
        """
        if isinstance(node, ast.Name):
            self.defines.add(node.id)
            self.mutation_defines.add(node.id)
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            self._add_mutation_define(node.value)


def _collect_body_refs(source: str) -> set[str]:
    """Find names referenced inside function/class bodies that resolve
    via module globals at runtime.

    The AST visitor walks module-scope expressions (including, after
    the recent extension, decorators / defaults / bases / annotations)
    but deliberately stops at the boundary of a function or class
    *body*. Bodies are a nested scope and need real scope analysis to
    tell a free variable apart from a parameter or a closure
    reference. ``symtable`` does that analysis exactly the way the
    Python compiler does, so we delegate.

    Returns names that should be added to the cell's references — let
    the caller filter for builtins / privates / defines.
    """
    try:
        root = symtable.symtable(source, "<cell>", "exec")
    except SyntaxError:
        return set()

    refs: set[str] = set()
    for child in root.get_children():
        if child.get_type() in ("function", "class"):
            _walk_body(child, refs)
    return refs


def _walk_body(scope: symtable.SymbolTable, refs: set[str]) -> None:
    """Recursively collect names that fall through to module globals.

    Within a function or class scope, a symbol falls through to module
    globals iff it's referenced AND ``is_global()`` AND not locally
    bound, not a parameter, not a closure variable. Closures
    (``is_free()``) resolve via the enclosing scope chain, not module
    globals — Python's compiler has already wired them up correctly.
    """
    for sym in scope.get_symbols():
        if (
            sym.is_referenced()
            and sym.is_global()
            and not sym.is_local()
            and not sym.is_parameter()
            and not sym.is_free()
        ):
            refs.add(sym.get_name())
    for inner in scope.get_children():
        if inner.get_type() in ("function", "class"):
            _walk_body(inner, refs)


def _collect_global_writes(source: str) -> tuple[set[str], set[str]]:
    """Find names assigned at module scope from inside a function via
    an explicit ``global`` declaration.

    A pattern like::

        def lazy_init():
            global STATE
            STATE = compute()
        lazy_init()

    binds ``STATE`` at module scope at runtime. The AST visitor only
    walks module-level statements for ``defines``, so it would miss
    this. Symtable flags such names as ``is_assigned() and
    is_declared_global()`` at the function's scope, which is enough
    to detect the binding without simulating runtime.

    Returns ``(writes, writes_with_reads)``:

    * ``writes`` — every name that's a global-declared assign target,
      including those that are also read in the same function.
    * ``writes_with_reads`` — subset that's *also* referenced. These
      need to stay in the cell's references (mutation_defines path)
      so the DAG records this cell as both a consumer and producer of
      the name, parallel to ``df["col"] = df["col"] * 2``.

    Bare ``global X`` declarations without a matching assign are
    skipped (no binding occurs). ``nonlocal`` is skipped — those
    write to the enclosing function's scope, not module scope.
    """
    try:
        root = symtable.symtable(source, "<cell>", "exec")
    except SyntaxError:
        return set(), set()
    writes: set[str] = set()
    read_writes: set[str] = set()
    for child in root.get_children():
        if child.get_type() in ("function", "class"):
            _walk_global_writes(child, writes, read_writes)
    return writes, read_writes


def _walk_global_writes(
    scope: symtable.SymbolTable,
    writes: set[str],
    read_writes: set[str],
) -> None:
    for sym in scope.get_symbols():
        if sym.is_assigned() and sym.is_declared_global():
            writes.add(sym.get_name())
            if sym.is_referenced():
                read_writes.add(sym.get_name())
    for inner in scope.get_children():
        if inner.get_type() in ("function", "class"):
            _walk_global_writes(inner, writes, read_writes)


def analyze_cell(source: str) -> CellAnalysis:
    """Analyze a cell's source code and extract defines/references.

    Args:
        source: Cell source code as a string

    Returns:
        CellAnalysis with defines, references, and optional error message
    """
    if not source or not source.strip():
        return CellAnalysis(defines=[], references=[])

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return CellAnalysis(
            defines=[],
            references=[],
            error=f"Syntax error: {e.msg}",
        )

    # The visitor stops at function/class bodies; symtable covers those.
    analyzer = VariableAnalyzer()
    analyzer.visit(tree)

    nested_refs = _collect_body_refs(source)

    # ``global`` writes inside functions are module defines too.
    # ``global_read_writes`` are also read there, so they stay in references.
    global_writes, global_read_writes = _collect_global_writes(source)

    builtin_names = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__"}
    defines = [v for v in (analyzer.defines | global_writes) if not v.startswith("_")]
    # A pure assignment in the same cell supersedes a mutation-define.
    effective_mutation_defines = {
        v
        for v in (analyzer.mutation_defines | global_read_writes)
        if not v.startswith("_") and v in set(defines) and v not in analyzer.pure_defines
    }
    # Pure defines are filtered from references (intra-cell rebinds);
    # mutation-defines and ``df = df.dropna()`` self-reads are kept.
    pure_defined_names = set(defines) - effective_mutation_defines - analyzer.rebind_with_self_read
    combined_refs = set(analyzer.references) | nested_refs
    references = [
        v
        for v in combined_refs
        if not v.startswith("_") and v not in builtin_names and v not in pure_defined_names
    ]
    builtin_references = [
        v
        for v in combined_refs
        if not v.startswith("_") and v in builtin_names and v not in pure_defined_names
    ]

    defines = sorted(set(defines))
    references = sorted(set(references))
    mutation_defines = sorted(effective_mutation_defines)

    return CellAnalysis(
        defines=defines,
        references=references,
        mutation_defines=mutation_defines,
        builtin_references=sorted(set(builtin_references)),
        error=None,
    )

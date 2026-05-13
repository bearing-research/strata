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
    # Subset of ``defines`` that came from in-place mutations
    # (``df["col"] = ...``, ``obj.attr = ...``). The cell both reads
    # and re-produces these names; the harness uses this to force
    # serialization even when the mutation preserved ``id()``.
    mutation_defines: list[str] = field(default_factory=list)
    error: str | None = None


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


class VariableAnalyzer(ast.NodeVisitor):
    """AST visitor that collects defined and referenced variables."""

    def __init__(self):
        """Initialize the analyzer."""
        self.defines: set[str] = set()
        self.references: set[str] = set()
        # Names that got into ``defines`` via a pure ``x = ...`` target
        # (not a subscript/attribute mutation). Used to demote a
        # variable out of ``mutation_defines`` when the same cell does
        # both — ``df = ...`` followed by ``df["col"] = ...`` is
        # locally-defined, not a mutation of an upstream.
        self.pure_defines: set[str] = set()
        # Subset of `defines` that came from subscript/attribute
        # mutations (``df["col"] = ...``) without a sibling pure
        # assignment. These still need to appear in references so the
        # DAG knows we depend on an upstream producer — unlike pure
        # rebinds (``x = x + 1``) where the reference is intra-cell
        # and should be filtered out.
        self.mutation_defines: set[str] = set()
        # Names whose value is read on the RHS of an assignment that
        # also defines them at module scope — ``df = df.dropna()`` style.
        # The reference is genuine (Python evaluates RHS before binding
        # LHS, so the upstream producer must exist) and survives the
        # pure-define filter below. Common pattern in Jupyter notebooks.
        self.rebind_with_self_read: set[str] = set()
        # Names pure-defined earlier in source order in this cell.
        # Used to distinguish ``x = 0\nx += 1`` (intra-cell rebind, no
        # upstream needed) from bare ``x += 1`` (genuine upstream read).
        # Walked incrementally in source order via visit_Module.
        self._defined_so_far: set[str] = set()
        self._in_nested_scope = False
        self._local_vars: set[str] = set()  # Track local scope variables
        # Set in visit_Module if the cell carries
        # ``from __future__ import annotations``. PEP 563 stringifies
        # annotations, so a name in a parameter or return annotation
        # shouldn't count as a runtime reference under that flag.
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
        # Names being pure-assigned by this statement, e.g. ``df`` in
        # ``df = df.dropna()`` or ``a, b`` in ``a, b = b, a``. We need
        # them before visiting the RHS so we can record genuine
        # read-before-write references — including the tuple/list
        # unpacking case, which is how Python does a swap.
        pure_target_names: set[str] = set()
        for target in node.targets:
            _collect_name_targets(target, pure_target_names)
        all_underscore = all(self._is_pure_underscore(target) for target in node.targets)
        if not all_underscore:
            # Walk the RHS looking for Loads of any target name being
            # bound here. A name read on the RHS of its own assignment
            # is an upstream reference UNLESS the cell already pure-
            # defined that name earlier in source order (``x = 0; x =
            # x + 1`` reads the local ``x``, not an upstream one).
            for child in ast.walk(node.value):
                if (
                    isinstance(child, ast.Name)
                    and isinstance(child.ctx, ast.Load)
                    and child.id in pure_target_names
                    and child.id not in self._defined_so_far
                ):
                    self.rebind_with_self_read.add(child.id)
            self.visit(node.value)
        # Collect targets (defines), and remember Name-target binds for
        # subsequent statements in this cell.
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
        # Augmented assignment defines the target
        self._add_assign_target(node.target)
        # Visit the value
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._defined_so_far.add(node.target.id)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle: x: int = ... or x: int (without value)."""
        self._add_assign_target(node.target)
        if node.value:
            self.visit(node.value)

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
            # import foo → defines 'foo'
            # import foo as bar → defines 'bar'
            name = alias.asname if alias.asname else alias.name
            self.defines.add(name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle: from foo import bar or from foo import *."""
        for alias in node.names:
            if alias.name == "*":
                # Star imports — we can't determine what's defined
                # Log would go here, but we just skip for v1
                pass
            else:
                # from foo import bar → defines 'bar'
                # from foo import bar as baz → defines 'baz'
                name = alias.asname if alias.asname else alias.name
                self.defines.add(name)

    def visit_For(self, node: ast.For) -> None:
        """Handle: for x in ... — x is defined at top level."""
        self._add_assign_target(node.target)
        # Visit iterable (may have references)
        self.visit(node.iter)
        # Visit body (nested scope, but loop variable is top-level)
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
        # Visit body (nested scope)
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
        # if (x := value) — x is defined
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
            # Don't add names that are local to current scope (e.g., lambda parameters)
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
        # Save current local vars
        old_local_vars = self._local_vars
        # Add lambda parameters to local scope
        self._local_vars = self._local_vars | self._get_lambda_params(node.args)
        # Visit body with parameters in local scope
        self.visit(node.body)
        # Restore local vars
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
            # Simple assignment: x = ...
            self.defines.add(target.id)
            self.pure_defines.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            # Tuple/list unpacking: (a, b) = ... or [a, b] = ...
            for elt in target.elts:
                self._add_assign_target(elt)
        elif isinstance(target, ast.Subscript):
            # Subscript mutation: df["col"] = ... — the cell both reads
            # the existing df AND produces the mutated version that
            # downstream cells will observe. Record both roles so the
            # DAG routes downstream reads through the mutating cell
            # (otherwise downstream cells can run before the mutation
            # and KeyError on the new column).
            self._add_reference_target(target.value)
            self._add_mutation_define(target.value)
        elif isinstance(target, ast.Attribute):
            # Attribute mutation: obj.attr = ... — same reasoning as
            # subscript mutation above.
            self._add_reference_target(target.value)
            self._add_mutation_define(target.value)
        elif isinstance(target, ast.Starred):
            # Starred assignment: *rest = ...
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
        # Empty cell
        return CellAnalysis(defines=[], references=[])

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        # Syntax error — return empty analysis with error message
        return CellAnalysis(
            defines=[],
            references=[],
            error=f"Syntax error: {e.msg}",
        )

    # Analyze the tree for defines, mutation_defines, and module-scope
    # references. The AST visitor handles the bookkeeping that's
    # specific to Strata (mutation_defines, pure_defines for the
    # demote-on-rebind logic) but it deliberately doesn't recurse into
    # function/class bodies.
    analyzer = VariableAnalyzer()
    analyzer.visit(tree)

    # Augment references with names referenced inside function/class
    # *bodies* — those are nested scopes and the AST visitor stops at
    # the boundary. ``symtable`` does correct scope analysis: closure
    # variables (``is_free()``) and parameters don't surface as module
    # references; only names that fall through to module globals do.
    nested_refs = _collect_body_refs(source)

    # Augment defines with module-scope bindings produced from inside a
    # function via explicit ``global`` declarations (e.g. lazy-init
    # patterns). Without this, a downstream cell that consumes such a
    # name would see no upstream producer. ``global_read_writes`` is
    # the subset that's also read in the same function — those stay in
    # references (parallel to ``df["col"] = df["col"] * 2``).
    global_writes, global_read_writes = _collect_global_writes(source)

    # Filter: exclude private variables and builtins
    builtin_names = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "__package__"}
    defines = [v for v in (analyzer.defines | global_writes) if not v.startswith("_")]
    # A name only counts as a mutation-define if the cell didn't also
    # pure-assign it. If a cell does ``df = ...`` and later
    # ``df["col"] = ...``, the pure assignment supersedes — ``df`` is
    # locally produced and shouldn't drag in a phantom upstream.
    effective_mutation_defines = {
        v
        for v in (analyzer.mutation_defines | global_read_writes)
        if not v.startswith("_") and v in set(defines) and v not in analyzer.pure_defines
    }
    # Mutation-defines stay in references (the cell depends on an
    # upstream producer of the pre-mutation object). Pure defines are
    # filtered from references — that handles intra-cell rebinds like
    # ``x = 5\ny = x + 1`` where ``x`` is locally produced before being
    # read.
    #
    # Exception: ``df = df.dropna()``-style rebinds where the RHS
    # reads the same name being bound. The reference is genuine (RHS
    # evaluates before LHS binding) and must survive the filter so the
    # DAG draws the upstream edge.
    pure_defined_names = set(defines) - effective_mutation_defines - analyzer.rebind_with_self_read
    combined_refs = set(analyzer.references) | nested_refs
    references = [
        v
        for v in combined_refs
        if not v.startswith("_") and v not in builtin_names and v not in pure_defined_names
    ]

    # Remove duplicates and sort for consistency
    defines = sorted(set(defines))
    references = sorted(set(references))
    mutation_defines = sorted(effective_mutation_defines)

    return CellAnalysis(
        defines=defines,
        references=references,
        mutation_defines=mutation_defines,
        error=None,
    )

"""Tests for AST-based variable analysis."""

from strata.notebook.analyzer import analyze_cell


class TestAnalyzerBasics:
    """Test basic variable analysis."""

    def test_empty_cell(self):
        """Empty cell has no defines or references."""
        result = analyze_cell("")
        assert result.defines == []
        assert result.references == []
        assert result.error is None

    def test_comment_only_cell(self):
        """Cell with only comments has no variables."""
        result = analyze_cell("# This is a comment\n# Another comment")
        assert result.defines == []
        assert result.references == []

    def test_simple_assignment(self):
        """Simple assignment: x = ..."""
        result = analyze_cell("x = 1")
        assert result.defines == ["x"]
        assert result.references == []

    def test_multiple_assignments(self):
        """Multiple assignments in same cell."""
        result = analyze_cell("x = 1\ny = 2\nz = 3")
        assert set(result.defines) == {"x", "y", "z"}
        assert result.references == []

    def test_tuple_unpacking(self):
        """Tuple unpacking: a, b = ..."""
        result = analyze_cell("a, b = (1, 2)")
        assert set(result.defines) == {"a", "b"}
        assert result.references == []

    def test_list_unpacking(self):
        """List unpacking: [a, b] = ..."""
        result = analyze_cell("[a, b] = [1, 2]")
        assert set(result.defines) == {"a", "b"}
        assert result.references == []

    def test_nested_unpacking(self):
        """Nested unpacking: (a, (b, c)) = ..."""
        result = analyze_cell("(a, (b, c)) = (1, (2, 3))")
        assert set(result.defines) == {"a", "b", "c"}

    def test_starred_unpacking(self):
        """Starred unpacking: a, *rest, b = ..."""
        result = analyze_cell("a, *rest, b = [1, 2, 3, 4]")
        assert set(result.defines) == {"a", "rest", "b"}


class TestAnalyzerAssignmentTypes:
    """Test different assignment types."""

    def test_augmented_assignment(self):
        """Augmented assignment: x += ..."""
        result = analyze_cell("x += 1")
        assert result.defines == ["x"]

    def test_subscript_assignment(self):
        """Subscript mutation defines AND references the root (see analyzer docs)."""
        result = analyze_cell('df["col"] = 1')
        assert result.defines == ["df"]
        assert "df" in result.references
        assert result.mutation_defines == ["df"]

    def test_attribute_assignment(self):
        """Attribute mutation defines AND references the root."""
        result = analyze_cell("obj.attr = 1")
        assert result.defines == ["obj"]
        assert "obj" in result.references
        assert result.mutation_defines == ["obj"]

    def test_nested_attribute_assignment(self):
        """Nested attribute mutation defines AND references the root name."""
        result = analyze_cell("obj.inner.attr = 1")
        assert result.defines == ["obj"]
        assert "obj" in result.references
        assert result.mutation_defines == ["obj"]

    def test_inplace_method_call_is_mutation_define(self):
        """df.drop(..., inplace=True) mutates df → define + reference + mutation."""
        result = analyze_cell("df.drop(columns=['a'], inplace=True)")
        assert result.defines == ["df"]
        assert "df" in result.references
        assert result.mutation_defines == ["df"]

    def test_inplace_false_is_not_a_mutation(self):
        """inplace=False reads df but does not mutate it."""
        result = analyze_cell("df.drop(columns=['a'], inplace=False)")
        assert "df" in result.references
        assert result.mutation_defines == []
        assert result.defines == []

    def test_inplace_non_literal_is_not_a_mutation(self):
        """A non-literal inplace= flag is too ambiguous to treat as a mutation."""
        result = analyze_cell("df.sort_values('a', inplace=flag)")
        assert result.mutation_defines == []

    def test_inplace_on_local_assignment_does_not_drag_phantom_upstream(self):
        """df = ...; df.fillna(inplace=True): df is locally produced, not upstream."""
        result = analyze_cell("df = make()\ndf.fillna(0, inplace=True)")
        assert "df" in result.defines
        assert result.mutation_defines == []
        assert "df" not in result.references

    def test_inplace_on_call_result_has_no_receiver_define(self):
        """get_df().drop(inplace=True): mutating a temporary has no cross-cell effect."""
        result = analyze_cell("get_df().drop(columns=['a'], inplace=True)")
        assert result.mutation_defines == []
        assert result.defines == []

    def test_annotated_assignment(self):
        """Annotated assignment: x: int = ..."""
        result = analyze_cell("x: int = 1")
        assert result.defines == ["x"]

    def test_annotated_assignment_no_value(self):
        """Annotated assignment without value: x: int."""
        result = analyze_cell("x: int")
        assert result.defines == ["x"]


class TestAnalyzerDefinitions:
    """Test definition extraction."""

    def test_function_definition(self):
        """Function definition: def f(): ..."""
        result = analyze_cell("def f():\n    x = 1\n    return x")
        assert result.defines == ["f"]
        assert result.references == []

    def test_nested_function(self):
        """Nested function definition — inner function not a top-level define."""
        result = analyze_cell("def outer():\n    def inner():\n        pass")
        assert result.defines == ["outer"]

    def test_class_definition(self):
        """Class definition: class C: ..."""
        result = analyze_cell("class C:\n    x = 1")
        assert result.defines == ["C"]

    def test_async_function(self):
        """Async function: async def f(): ..."""
        result = analyze_cell("async def f():\n    pass")
        assert result.defines == ["f"]


class TestAnalyzerImports:
    """Test import statement handling."""

    def test_simple_import(self):
        """import foo → defines foo."""
        result = analyze_cell("import pandas")
        assert result.defines == ["pandas"]

    def test_import_alias(self):
        """import foo as bar → defines bar."""
        result = analyze_cell("import pandas as pd")
        assert result.defines == ["pd"]

    def test_multiple_imports(self):
        """import foo, bar → defines foo and bar."""
        result = analyze_cell("import os, sys")
        assert set(result.defines) == {"os", "sys"}

    def test_from_import(self):
        """from foo import bar → defines bar."""
        result = analyze_cell("from pandas import DataFrame")
        assert result.defines == ["DataFrame"]

    def test_from_import_alias(self):
        """from foo import bar as baz → defines baz."""
        result = analyze_cell("from pandas import DataFrame as DF")
        assert result.defines == ["DF"]

    def test_from_import_multiple(self):
        """from foo import bar, baz → defines bar and baz."""
        result = analyze_cell("from pandas import DataFrame, Series")
        assert set(result.defines) == {"DataFrame", "Series"}


class TestAnalyzerReferences:
    """Test reference extraction."""

    def test_simple_reference(self):
        """Reference to undefined variable."""
        result = analyze_cell("y = x + 1")
        assert result.defines == ["y"]
        assert result.references == ["x"]

    def test_multiple_references(self):
        """Multiple references."""
        result = analyze_cell("z = x + y")
        assert result.defines == ["z"]
        assert set(result.references) == {"x", "y"}

    def test_function_call(self):
        """Function call: f() → f is referenced."""
        result = analyze_cell("result = len(mylist)")
        assert result.defines == ["result"]
        assert set(result.references) == {"mylist"}  # len is builtin

    def test_method_call(self):
        """Method call: obj.method()."""
        result = analyze_cell("result = df.sum()")
        assert result.defines == ["result"]
        assert result.references == ["df"]

    def test_subscript_reference(self):
        """Subscript: df["col"] → df is referenced."""
        result = analyze_cell('x = df["col"]')
        assert result.defines == ["x"]
        assert result.references == ["df"]

    def test_attribute_reference(self):
        """Attribute access: obj.attr → obj is referenced."""
        result = analyze_cell("x = obj.attr")
        assert result.defines == ["x"]
        assert result.references == ["obj"]

    def test_builtin_excluded(self):
        """Builtin functions like len, print are excluded."""
        result = analyze_cell("print(len([1, 2, 3]))")
        assert result.defines == []
        assert result.references == []

    def test_rebind_with_self_read_keeps_reference(self):
        """``df = df.dropna()`` and ``x = x + 1`` style rebinds: the RHS
        reads the same name being bound on the LHS, so the read is a
        genuine upstream reference. Without this, the DAG would skip
        the edge to the cell that produced the original ``df`` and the
        runtime would crash with NameError.

        Common pattern in Jupyter notebooks (caught by the corpus runner
        on titanic_classifier.ipynb)."""
        result = analyze_cell("df = df.dropna()")
        assert "df" in result.defines
        assert "df" in result.references

        result = analyze_cell("x = x + 1")
        assert "x" in result.defines
        assert "x" in result.references

        # Nested target — ``df.head()`` after the rebind reads the
        # local df, but the original read still surfaces.
        result = analyze_cell("df = df.dropna()\ndf.head()\n")
        assert "df" in result.defines
        assert "df" in result.references

    def test_pure_define_then_read_not_a_reference(self):
        """Counterpoint to the rebind-with-self-read case: when a cell
        defines a name THEN reads it (``x = 5; y = x + 1``), the read
        is intra-cell — no upstream needed. The earlier read must not
        accidentally drag the name into references."""
        result = analyze_cell("x = 5\ny = x + 1\n")
        assert "x" not in result.references
        assert "x" in result.defines
        assert "y" in result.defines

    def test_augassign_bare_target_is_a_reference(self):
        """``x += 1`` reads ``x`` before writing it. Same class as the
        Assign rebind — the implicit read needs to surface as an
        upstream reference, otherwise the DAG drops the edge and the
        cell hits NameError. The AugAssign target has no visible Name
        in Load context, so the analyzer has to inject the read."""
        result = analyze_cell("x += 1")
        assert "x" in result.defines
        assert "x" in result.references

    def test_augassign_after_local_define_not_a_reference(self):
        """``x = 0\\nx += 1`` reads the local ``x`` set on the prior
        line — no upstream dep. The source-order check has to suppress
        the augassign's implicit read."""
        result = analyze_cell("x = 0\nx += 1\n")
        assert "x" not in result.references
        assert "x" in result.defines

    def test_tuple_swap_keeps_both_references(self):
        """``a, b = b, a`` reads both ``a`` and ``b`` on the RHS while
        binding both on the LHS. Both names are genuine upstream
        references; the tuple-unpack handler has to collect Name
        targets nested in Tuple."""
        result = analyze_cell("a, b = b, a")
        assert "a" in result.defines and "b" in result.defines
        assert "a" in result.references and "b" in result.references

    def test_tuple_unpacking_after_local_define_not_references(self):
        """And the source-order suppression flows through tuple unpacking
        too — locally bound names don't drag in spurious upstream deps."""
        result = analyze_cell("a, b = 1, 2\na, b = b, a\n")
        assert "a" not in result.references
        assert "b" not in result.references


class TestAnalyzerPrivateVariables:
    """Test private variable (starting with _) handling."""

    def test_private_define_excluded(self):
        """Variables starting with _ are excluded from defines."""
        result = analyze_cell("_private = 1")
        assert result.defines == []

    def test_private_reference_excluded(self):
        """References to _private variables are excluded."""
        result = analyze_cell("x = _private + 1")
        assert result.defines == ["x"]
        assert result.references == []

    def test_dunder_excluded(self):
        """__dunder__ variables are excluded."""
        result = analyze_cell("__name__ = 'main'")
        assert result.defines == []

    def test_single_underscore_excluded(self):
        """Single _ is excluded."""
        result = analyze_cell("_ = unused")
        assert result.defines == []
        assert result.references == []  # unused is not defined


class TestAnalyzerLoopsAndContextManagers:
    """Test loop variables and context manager variables."""

    def test_for_loop_variable(self):
        """For loop variable: for x in ... → x is defined."""
        result = analyze_cell("for x in items:\n    print(x)")
        assert result.defines == ["x"]
        assert result.references == ["items"]

    def test_for_loop_nested_vars(self):
        """Nested for loop: for (a, b) in items."""
        result = analyze_cell("for (a, b) in items:\n    pass")
        assert set(result.defines) == {"a", "b"}
        assert result.references == ["items"]

    def test_with_statement_variable(self):
        """With statement: with ... as x → x is defined."""
        result = analyze_cell("with open('file') as f:\n    pass")
        assert result.defines == ["f"]
        assert result.references == []

    def test_except_handler_variable(self):
        """Exception handler: except E as e → e is defined."""
        result = analyze_cell("try:\n    pass\nexcept Exception as e:\n    pass")
        assert result.defines == ["e"]

    def test_async_with_statement(self):
        """Async with statement: async with ... as x."""
        result = analyze_cell("async with async_ctx() as x:\n    pass")
        assert result.defines == ["x"]
        assert result.references == ["async_ctx"]


class TestAnalyzerComprehensions:
    """Test comprehension handling (loop vars are NOT top-level)."""

    def test_list_comprehension(self):
        """List comprehension: [x for x in items] → x is NOT a top-level define."""
        result = analyze_cell("[x for x in items]")
        assert result.defines == []
        assert result.references == ["items"]

    def test_list_comprehension_with_condition(self):
        """List comp with condition: [x for x in items if x > 0]."""
        result = analyze_cell("[x for x in items if x > 0]")
        assert result.defines == []
        assert result.references == ["items"]

    def test_list_comprehension_with_outer_var(self):
        """List comp using outer variable: free names in the element are
        picked up; the loop variable stays comp-local."""
        result = analyze_cell("[x * factor for x in items]")
        assert result.defines == []
        assert set(result.references) == {"items", "factor"}

    def test_list_comprehension_function_call_in_element(self):
        """``[helper(x) for x in items]`` — ``helper`` is a free var in
        the element position, must be picked up so the DAG can load
        the synthetic module that exports it. This is the seed-cell
        case that motivated the fix."""
        result = analyze_cell("out = [helper(x) for x in items]")
        assert "out" in result.defines
        assert set(result.references) == {"items", "helper"}

    def test_list_comprehension_with_condition_outer_var(self):
        """``[x for x in items if predicate(x)]`` — ``predicate`` is a
        free var in the condition position."""
        result = analyze_cell("[x for x in items if predicate(x)]")
        assert result.defines == []
        assert set(result.references) == {"items", "predicate"}

    def test_dict_comprehension_with_outer_vars(self):
        """``{f(k): g(v) for k, v in items}`` — ``f`` and ``g`` are
        outer-scope free vars; ``k`` and ``v`` are comp-local."""
        result = analyze_cell("{f(k): g(v) for k, v in items}")
        assert result.defines == []
        assert set(result.references) == {"items", "f", "g"}

    def test_nested_comprehension(self):
        """``[a + b for a in xs for b in ys]`` — outer iter ``xs`` runs
        in outer scope, inner iter ``ys`` runs in comp scope but ``ys``
        is still a free var; ``a`` and ``b`` stay comp-local."""
        result = analyze_cell("[a + b for a in xs for b in ys]")
        assert result.defines == []
        assert set(result.references) == {"xs", "ys"}

    def test_comprehension_target_does_not_leak(self):
        """The loop variable ``x`` in a comprehension is comp-scoped and
        must not surface as a module reference even though Python 3.13
        inlines comp scopes (PEP 709)."""
        result = analyze_cell("out = [x * 2 for x in items]")
        # ``x`` is bound by the comp, not a free var. Should not appear
        # in references regardless of PEP 709 inlining.
        assert "x" not in result.references

    def test_dict_comprehension(self):
        """Dict comprehension: {k: v for k, v in items}."""
        result = analyze_cell("{k: v for k, v in items}")
        assert result.defines == []
        assert result.references == ["items"]

    def test_set_comprehension(self):
        """Set comprehension: {x for x in items}."""
        result = analyze_cell("{x for x in items}")
        assert result.defines == []
        assert result.references == ["items"]

    def test_generator_expression(self):
        """Generator expression: (x for x in items)."""
        result = analyze_cell("(x for x in items)")
        assert result.defines == []
        assert result.references == ["items"]


class TestAnalyzerLambda:
    """Test lambda expressions."""

    def test_lambda_simple(self):
        """Lambda: lambda x: x + 1 — x is parameter, not cell-level define."""
        result = analyze_cell("f = lambda x: x + 1")
        assert result.defines == ["f"]
        assert result.references == []

    def test_lambda_with_outer_ref(self):
        """Lambda with outer reference: lambda x: x + y."""
        result = analyze_cell("f = lambda x: x + y")
        assert result.defines == ["f"]
        assert result.references == ["y"]


class TestAnalyzerWalrusOperator:
    """Test walrus operator (:=) at top level."""

    def test_walrus_in_if(self):
        """Walrus in if: if (x := value): — x is defined."""
        result = analyze_cell("if (x := value):\n    pass")
        assert result.defines == ["x"]
        assert result.references == ["value"]


class TestAnalyzerRealWorldExamples:
    """Test real-world notebook cells."""

    def test_pandas_cell(self):
        """Real-world: pandas DataFrame operations."""
        source = """
import pandas as pd
df = pd.read_csv('data.csv')
df['new_col'] = df['old_col'] * 2
cleaned = df[df['new_col'] > 100]
"""
        result = analyze_cell(source)
        assert set(result.defines) == {"pd", "df", "cleaned"}
        assert result.references == []

    def test_data_transformation_cell(self):
        """Real-world: transform input dataframe."""
        source = """
cleaned = df[df.value > 50]
summary = {
    'rows': len(cleaned),
    'mean': cleaned.value.mean(),
}
"""
        result = analyze_cell(source)
        assert set(result.defines) == {"cleaned", "summary"}
        assert result.references == ["df"]

    def test_plot_cell(self):
        """Real-world: plotting with external inputs."""
        source = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(data.x, data.y)
ax.set_title(title)
plt.show()
"""
        result = analyze_cell(source)
        assert set(result.defines) == {"plt", "fig", "ax"}
        assert set(result.references) == {"data", "title"}

    def test_model_training_cell(self):
        """Real-world: ML model training."""
        source = """
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
"""
        result = analyze_cell(source)
        assert set(result.defines) == {"RandomForestClassifier", "model", "score"}
        assert set(result.references) == {"X_train", "y_train", "X_test", "y_test"}


class TestAnalyzerSyntaxErrors:
    """Test handling of syntax errors."""

    def test_syntax_error_returns_error(self):
        """Syntax error returns error message."""
        result = analyze_cell("x = ")
        assert result.defines == []
        assert result.references == []
        assert result.error is not None
        assert "Syntax error" in result.error

    def test_syntax_error_unclosed_paren(self):
        """Unclosed parenthesis."""
        result = analyze_cell("x = sum([1, 2, 3")
        assert result.error is not None


class TestAnalyzerEdgeCases:
    """Test edge cases and corner cases."""

    def test_variable_defined_then_used(self):
        """Variable defined then used in same cell."""
        result = analyze_cell("x = 1\ny = x + 1")
        assert set(result.defines) == {"x", "y"}
        # x is not a reference because it's defined in the cell
        assert result.references == []

    def test_global_statement_ignored(self):
        """Global statement: global x — x is not a top-level define."""
        result = analyze_cell("global x\nx = 1")
        assert result.defines == ["x"]

    def test_nonlocal_statement_ignored(self):
        """Nonlocal statement: nonlocal x."""
        result = analyze_cell("def outer():\n    x = 1\n    def inner():\n        nonlocal x")
        assert result.defines == ["outer"]

    def test_del_statement(self):
        """del statement does not define variables."""
        result = analyze_cell("del x")
        assert result.defines == []
        assert result.references == ["x"]

    def test_assert_statement(self):
        """assert statement references variables."""
        result = analyze_cell("assert x > 0")
        assert result.defines == []
        assert result.references == ["x"]

    def test_raise_statement(self):
        """raise with exception expression."""
        result = analyze_cell("raise ValueError(msg)")
        assert result.defines == []
        assert result.references == ["msg"]


class TestAnalyzerNestedScopes:
    """Test that references inside nested scopes (function/class
    bodies, decorators, default arg values, base classes, type
    annotations) are picked up correctly.

    Without this, a cell like ``def f(): return upstream_var`` would
    show no references, the DAG wouldn't add the upstream cell as a
    parent, and the synthetic module wouldn't be loaded — leading to a
    NameError at call time.
    """

    def test_function_body_reference(self):
        """Free variable inside a function body becomes a reference."""
        result = analyze_cell("def f():\n    return upstream_var")
        assert "f" in result.defines
        assert "upstream_var" in result.references

    def test_function_body_local_assign_not_a_reference(self):
        """Names bound by an assignment inside the function don't bubble out."""
        result = analyze_cell("def f():\n    x = 1\n    return x")
        assert "f" in result.defines
        assert "x" not in result.references

    def test_method_body_reference(self):
        """Free variable inside a method body becomes a reference."""
        result = analyze_cell("class C:\n    def m(self):\n        return upstream_var")
        assert "C" in result.defines
        assert "upstream_var" in result.references

    def test_method_self_attribute_not_a_reference(self):
        """``self.x`` is attribute access, not a free-variable lookup."""
        result = analyze_cell("class C:\n    def m(self):\n        return self.x")
        assert "C" in result.defines
        assert result.references == []

    def test_decorator_reference(self):
        """``@upstream_decorator`` evaluates at module load — picked up."""
        result = analyze_cell("@upstream_decorator\ndef f():\n    pass")
        assert "f" in result.defines
        assert "upstream_decorator" in result.references

    def test_default_arg_reference(self):
        """Default arg value evaluates at module load — picked up."""
        result = analyze_cell("def f(x=upstream_default):\n    pass")
        assert "f" in result.defines
        assert "upstream_default" in result.references

    def test_class_base_reference(self):
        """Class base evaluates at module load — picked up."""
        result = analyze_cell("class C(UpstreamBase):\n    pass")
        assert "C" in result.defines
        assert "UpstreamBase" in result.references

    def test_class_body_reference(self):
        """Class body assignments at the class scope reference module globals."""
        result = analyze_cell("class C:\n    value = upstream_helper(0)")
        assert "C" in result.defines
        assert "upstream_helper" in result.references

    def test_annotation_reference_without_future_import(self):
        """Type annotations (without ``from __future__ import annotations``)
        evaluate at function-definition time, so they reference module globals."""
        result = analyze_cell("def f(x: UpstreamType) -> UpstreamType:\n    return x")
        assert "f" in result.defines
        assert "UpstreamType" in result.references

    def test_annotation_reference_with_future_annotations_is_skipped(self):
        """With ``from __future__ import annotations`` (PEP 563), annotations
        are stringified and never evaluated. ``symtable`` correctly drops
        them from the reference set."""
        result = analyze_cell(
            "from __future__ import annotations\n"
            "def f(x: UpstreamType) -> UpstreamType:\n"
            "    return x"
        )
        assert "f" in result.defines
        assert "UpstreamType" not in result.references

    def test_closure_over_outer_parameter_is_not_a_reference(self):
        """A nested function closing over its outer function's parameter
        resolves via the closure chain, not module globals — should NOT
        be flagged."""
        result = analyze_cell(
            "def outer(items):\n    def inner():\n        return items\n    return inner"
        )
        assert "outer" in result.defines
        assert result.references == []

    def test_lambda_inside_function_closes_over_param(self):
        """Lambda inside a function, closing over the function's parameter,
        is a closure — not a module-globals lookup."""
        result = analyze_cell(
            "def sort_by_score(items):\n    return sorted(items, key=lambda i: items[i])"
        )
        assert "sort_by_score" in result.defines
        assert result.references == []

    def test_function_referencing_cross_cell_helper_picks_it_up(self):
        """The motivating case: a cell defines a function that calls a
        helper from another cell. The reference must surface so the
        DAG adds an upstream edge and the synthetic module is loaded."""
        result = analyze_cell("def use_helper():\n    return cross_cell_helper(42)")
        assert "use_helper" in result.defines
        assert "cross_cell_helper" in result.references

    def test_existing_module_scope_reference_still_works(self):
        """The existing AST visitor's module-scope refs are preserved
        unchanged — nothing in the new symtable pass should break the
        common case."""
        result = analyze_cell("y = x + 1")
        assert result.defines == ["y"]
        assert result.references == ["x"]


class TestAnalyzerGlobalWrites:
    """``def f(): global X; X = ...`` binds ``X`` at module scope at
    runtime. The analyzer must register ``X`` as a define so downstream
    cells that read ``X`` correctly see this cell as their upstream
    producer."""

    def test_global_write_registered_as_define(self):
        """The lazy-init pattern: function declares + writes a global."""
        result = analyze_cell(
            "def lazy_init():\n    global STATE\n    STATE = compute()\nlazy_init()"
        )
        assert "STATE" in result.defines
        assert "lazy_init" in result.defines
        # ``STATE`` is only written inside the function (no read), so
        # it shouldn't appear as a reference.
        assert "STATE" not in result.references
        # ``compute`` is referenced but not bound — upstream dependency.
        assert "compute" in result.references

    def test_global_read_and_write_keeps_name_in_references(self):
        """``STATE = compute(STATE)`` reads the prior value and writes
        a new one. The cell is both consumer and producer of STATE,
        same shape as ``df["col"] = df["col"] * 2``. The DAG needs
        STATE in both defines AND references so it routes the
        downstream read through this cell, not the original producer."""
        result = analyze_cell("def lazy_init():\n    global STATE\n    STATE = compute(STATE)\n")
        assert "STATE" in result.defines
        assert "STATE" in result.references
        assert "STATE" in result.mutation_defines

    def test_multiple_globals_in_one_declaration(self):
        """``global STATE, FLAG`` declares both — the analyzer should
        register every name that's actually written."""
        result = analyze_cell(
            "def init():\n    global STATE, FLAG\n    STATE = compute()\n    FLAG = True\n"
        )
        assert "STATE" in result.defines
        assert "FLAG" in result.defines

    def test_bare_global_declaration_without_assign_is_not_a_define(self):
        """``global Y`` without an assignment doesn't bind Y at module
        scope. Symtable flags it as ``declared_global`` but not
        ``assigned``, and we filter on the assigned flag."""
        result = analyze_cell("def f():\n    global Y\n    return 1\n")
        assert "Y" not in result.defines
        # Y should also not be in references — it's just a declaration,
        # no actual access.
        assert "Y" not in result.references

    def test_nonlocal_does_not_register_as_module_define(self):
        """``nonlocal`` writes to the enclosing function's scope, not
        module scope. Symtable flags those as ``is_local()`` (not
        ``is_declared_global()``), so they don't surface as defines
        from this cell."""
        result = analyze_cell(
            "def outer():\n    x = 1\n    def inner():\n        nonlocal x\n        x = 2\n"
        )
        assert "x" not in result.defines
        assert result.defines == ["outer"]

    def test_global_write_in_method_body(self):
        """A method inside a class can also write a module-level
        global. Symtable's recursive scope walk catches it."""
        result = analyze_cell(
            "class Service:\n    def init(self):\n        global READY\n        READY = True\n"
        )
        assert "READY" in result.defines
        assert "Service" in result.defines

    def test_local_assignment_inside_function_is_not_a_define(self):
        """A function that assigns to a local name (no ``global``) does
        NOT define that name at module scope, even though the AST
        contains ``Name(Store)`` nodes inside the function body."""
        result = analyze_cell("def f():\n    x = 1\n    return x\n")
        assert "x" not in result.defines
        assert result.defines == ["f"]

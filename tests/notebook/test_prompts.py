"""Tests for prompt variable rendering and the ``{{ var }}`` template evaluator.

The evaluator is a deliberately tiny AST-walker (no ``eval``) that surfaces
Python values to an LLM. Its security contract — no private access, no
arbitrary calls, only a whitelist of zero-arg pandas helpers — is the most
important thing to pin down, alongside the per-variable rendering/trimming.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from strata.notebook.llm.prompts import (
    _resolve_prompt_expression,
    render_prompt_template,
    variable_to_text,
)


class TestVariableToText:
    def test_str_passes_through(self):
        assert variable_to_text("hello") == "hello"

    def test_dict_renders_as_json(self):
        out = variable_to_text({"a": 1, "b": [2, 3]})
        assert '"a": 1' in out
        assert '"b"' in out

    def test_list_renders_as_json(self):
        out = variable_to_text([1, 2, 3])
        assert out.startswith("[")
        assert "1" in out and "3" in out

    def test_non_json_values_fall_back_to_str(self):
        out = variable_to_text({"d": datetime.date(2020, 1, 2)})
        assert "2020-01-02" in out

    def test_long_value_is_truncated_with_marker(self):
        out = variable_to_text("x" * 1000, max_tokens=5)  # max_chars = 20
        assert out.endswith("... (truncated)")
        assert out.startswith("x" * 20)

    def test_dataframe_shows_shape_columns_and_all_columns(self):
        df = pd.DataFrame({f"c{i}": [i] for i in range(30)})
        out = variable_to_text(df)
        assert "shape=(1, 30)" in out
        # No mid-column ellipsis: every column must survive rendering.
        for i in range(30):
            assert f"c{i}" in out

    def test_dataframe_tall_drops_rows_and_marks_count(self):
        df = pd.DataFrame({"a": list(range(1000))})
        out = variable_to_text(df, max_tokens=20)  # tiny budget forces trimming
        assert "more rows)" in out
        # The column-header line is pinned even when rows are dropped.
        assert out.splitlines()[0].startswith("DataFrame shape=(1000, 1)")

    def test_dataframe_keeps_some_rows_then_drops_the_rest(self):
        # A mid-size budget keeps a prefix of rows and drops the tail (exercises
        # both the keep and the break branches of the row-fitting loop).
        df = pd.DataFrame({"a": list(range(100))})
        out = variable_to_text(df, max_tokens=30)
        assert "more rows)" in out
        body = out.splitlines()
        # Some data rows survived between the header and the marker.
        assert len(body) > 3
        dropped = int(body[-1].split("(")[1].split(" ")[0])
        assert 0 < dropped < 100

    def test_series_render(self):
        out = variable_to_text(pd.Series([1, 2, 3], name="nums"))
        assert "name='nums'" in out
        assert "length=3" in out

    def test_ndarray_render(self):
        out = variable_to_text(np.arange(20))
        assert "ndarray shape=(20,)" in out


class TestRenderTemplate:
    def test_basic_substitution(self):
        assert render_prompt_template("Hi {{ name }}!", {"name": "Bob"}) == "Hi Bob!"

    def test_multiple_variables(self):
        assert render_prompt_template("{{ a }}-{{ b }}", {"a": "x", "b": "y"}) == "x-y"

    def test_whitespace_inside_braces_is_tolerated(self):
        assert render_prompt_template("{{name}}", {"name": "Z"}) == "Z"

    def test_missing_variable_leaves_literal(self):
        assert render_prompt_template("{{ gone }}", {}) == "{{ gone }}"

    def test_attribute_access_is_substituted(self):
        out = render_prompt_template("{{ df.shape }}", {"df": pd.DataFrame({"a": [1, 2]})})
        assert "{{" not in out
        assert "2" in out

    def test_whitelisted_method_call_renders_result(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = render_prompt_template("{{ df.head() }}", {"df": df})
        assert "DataFrame shape=" in out


class TestTemplateSecurity:
    """Every unsafe expression must be rejected, leaving the literal in place."""

    def test_private_attribute_blocked(self):
        class Obj:
            def __init__(self):
                self._secret = "nope"

        assert render_prompt_template("{{ o._secret }}", {"o": Obj()}) == "{{ o._secret }}"

    def test_uncalled_callable_attribute_blocked(self):
        assert render_prompt_template("{{ s.upper }}", {"s": "hi"}) == "{{ s.upper }}"

    def test_method_call_with_arguments_blocked(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert render_prompt_template("{{ df.head(2) }}", {"df": df}) == "{{ df.head(2) }}"

    def test_non_whitelisted_method_blocked(self):
        df = pd.DataFrame({"a": [1]})
        assert render_prompt_template("{{ df.to_csv() }}", {"df": df}) == "{{ df.to_csv() }}"

    def test_call_on_plain_name_blocked(self):
        assert render_prompt_template("{{ foo() }}", {"foo": lambda: "x"}) == "{{ foo() }}"

    def test_arbitrary_expression_blocked(self):
        assert render_prompt_template("{{ 1 + 1 }}", {}) == "{{ 1 + 1 }}"


class TestEvaluatorRaises:
    """The evaluator raises precise errors (``render_prompt_template`` swallows
    them; these assert the contract directly)."""

    def test_missing_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            _resolve_prompt_expression("missing", {})

    def test_private_attribute_raises(self):
        obj = type("O", (), {"_x": 1})()
        with pytest.raises(ValueError, match="[Pp]rivate"):
            _resolve_prompt_expression("o._x", {"o": obj})

    def test_callable_attribute_raises(self):
        with pytest.raises(ValueError, match="[Cc]allable"):
            _resolve_prompt_expression("s.upper", {"s": "hi"})

    def test_method_with_arguments_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="do not accept arguments"):
            _resolve_prompt_expression("df.head(1)", {"df": df})

    def test_call_on_non_attribute_raises(self):
        with pytest.raises(ValueError, match="attribute method"):
            _resolve_prompt_expression("foo()", {"foo": lambda: 1})

    def test_unsafe_method_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="[Uu]nsafe"):
            _resolve_prompt_expression("df.to_dict()", {"df": df})

    def test_private_method_call_raises(self):
        with pytest.raises(ValueError, match="[Pp]rivate"):
            _resolve_prompt_expression("o._hidden()", {"o": object()})

    def test_calling_a_non_callable_attribute_is_unsafe(self):
        # ``df.shape`` is a tuple, not a method — calling it must be rejected.
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="[Uu]nsafe"):
            _resolve_prompt_expression("df.shape()", {"df": df})

    def test_unsupported_expression_raises(self):
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            _resolve_prompt_expression("1 + 1", {})

    @pytest.mark.parametrize("method_name", ["describe", "head", "tail"])
    def test_whitelisted_pandas_methods_resolve(self, method_name):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _resolve_prompt_expression(f"df.{method_name}()", {"df": df})
        assert isinstance(result, pd.DataFrame)

    def test_safe_method_on_non_pandas_object_is_unsafe(self):
        with pytest.raises(ValueError, match="[Uu]nsafe"):
            _resolve_prompt_expression("s.strip()", {"s": "  hi  "})

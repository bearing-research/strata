"""Variable-to-text rendering and ``{{ var }}`` template expansion.

Used by prompt cells (``prompt_executor``) to surface Python values to
the model. The template language is
deliberately tiny — only attribute access plus a handful of zero-arg
methods — so we can evaluate it via AST without ``eval``.
"""

from __future__ import annotations

import ast
from typing import Any


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _pd_display_ctx():
    """Disable every pandas truncation knob for LLM rendering.

    pandas' default ``str()`` / ``to_string()`` output collapses the middle
    of a wide frame to ``...`` once the column count exceeds
    ``display.max_columns``. For prompt injection that is catastrophic —
    the LLM literally cannot see the hidden columns and responds with
    empty fields. We lift every limit, then trim rows ourselves to fit
    the token budget.
    """
    import pandas as pd

    return pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.max_colwidth",
        None,
        "display.width",
        10_000,
    )


def _fit_rendered_lines(
    preamble: str,
    pinned_lines: list[str],
    data_lines: list[str],
    max_chars: int,
) -> str:
    """Assemble ``preamble`` + pinned lines + as many data lines as fit.

    ``pinned_lines`` (e.g. a DataFrame's column header row) are always
    kept; data rows are dropped from the tail until the total length
    is at or below ``max_chars``. A ``... (N more rows)`` marker is
    appended when anything is dropped.
    """
    full = "\n".join([preamble, *pinned_lines, *data_lines])
    if len(full) <= max_chars:
        return full

    overhead = len(preamble) + sum(len(line) + 1 for line in pinned_lines) + 1 + 40
    budget = max(0, max_chars - overhead)
    kept: list[str] = []
    running = 0
    for line in data_lines:
        if running + len(line) + 1 > budget:
            break
        kept.append(line)
        running += len(line) + 1

    dropped = len(data_lines) - len(kept)
    tail = f"\n... ({dropped} more rows)" if dropped > 0 else ""
    return "\n".join([preamble, *pinned_lines, *kept]) + tail


def _dataframe_to_text(df: Any, max_chars: int) -> str:
    """Render a DataFrame for prompt injection without column ellipsis.

    ``df.to_markdown()`` would give nicer output but requires ``tabulate``
    (not a dep). We use ``to_string(index=False)`` inside a context that
    forces every column to render, then preserve the column-header line
    while trimming data rows to fit the budget.
    """
    preamble = f"DataFrame shape={df.shape} columns={list(df.columns)}"
    with _pd_display_ctx():
        lines = df.to_string(index=False).split("\n")
    return _fit_rendered_lines(preamble, lines[:1], lines[1:], max_chars)


def _series_to_text(s: Any, max_chars: int) -> str:
    """Render a Series without truncating values, capped at ``max_chars``."""
    preamble = f"Series name={s.name!r} length={len(s)} dtype={s.dtype}"
    with _pd_display_ctx():
        body = s.to_string()
    data_lines = body.split("\n") if body else []
    return _fit_rendered_lines(preamble, [], data_lines, max_chars)


def variable_to_text(value: Any, max_tokens: int = 2000) -> str:
    """Convert a Python value to a text representation for prompt injection.

    Applies type-specific formatting with a per-variable token budget.
    """
    max_chars = max_tokens * 4

    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            return _dataframe_to_text(value, max_chars)
        if isinstance(value, pd.Series):
            return _series_to_text(value, max_chars)
    except ImportError:
        pass

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            header = f"ndarray shape={value.shape} dtype={value.dtype}"
            preview = str(value.flat[:10])
            text = f"{header}\n{preview}"
            return text[:max_chars]
    except ImportError:
        pass

    if isinstance(value, dict):
        import json as _json

        text = _json.dumps(value, indent=2, default=str)
    elif isinstance(value, (list, tuple)):
        import json as _json

        text = _json.dumps(value, indent=2, default=str)
    else:
        text = str(value)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n... (truncated)"
    return text


def render_prompt_template(
    template: str,
    variables: dict[str, Any],
    max_tokens_per_var: int = 2000,
) -> str:
    """Render a prompt template by replacing ``{{ var }}`` with text values."""
    import re

    def _replace(match: re.Match) -> str:
        expr = match.group(1).strip()
        try:
            value = _resolve_prompt_expression(expr, variables)
        except Exception:
            return match.group(0)
        else:
            return variable_to_text(value, max_tokens=max_tokens_per_var)

    return re.sub(r"\{\{\s*([^}]+)\s*\}\}", _replace, template)


def _resolve_prompt_expression(expr: str, variables: dict[str, Any]) -> Any:
    """Resolve a prompt template expression without executing arbitrary code."""
    parsed = ast.parse(expr, mode="eval")
    return _evaluate_prompt_node(parsed.body, variables)


def _evaluate_prompt_node(node: ast.AST, variables: dict[str, Any]) -> Any:
    """Evaluate a restricted AST node for prompt templating."""
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise KeyError(node.id)
        return variables[node.id]

    if isinstance(node, ast.Attribute):
        value = _evaluate_prompt_node(node.value, variables)
        if node.attr.startswith("_"):
            raise ValueError("Private attributes are not allowed")
        resolved = getattr(value, node.attr)
        if callable(resolved):
            raise ValueError("Callable attributes must be explicitly allowed")
        return resolved

    if isinstance(node, ast.Call):
        if node.args or node.keywords:
            raise ValueError("Prompt template calls do not accept arguments")
        if not isinstance(node.func, ast.Attribute):
            raise ValueError("Only attribute method calls are allowed")
        obj = _evaluate_prompt_node(node.func.value, variables)
        method_name = node.func.attr
        if method_name.startswith("_"):
            raise ValueError("Private methods are not allowed")
        method = getattr(obj, method_name)
        if not _is_safe_prompt_method(obj, method_name, method):
            raise ValueError(f"Unsafe prompt method: {method_name}")
        return method()

    raise ValueError("Unsupported prompt expression")


def _is_safe_prompt_method(obj: Any, method_name: str, method: Any) -> bool:
    """Allow a very small set of known-safe zero-arg prompt helpers."""
    if not callable(method):
        return False

    try:
        import pandas as pd

        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return method_name in {"describe", "head", "tail"}
    except ImportError:
        pass

    return False

"""Analyzer for widget-type notebook cells.

A widget cell is a declarative control panel — one ``name = control(...)``
line per control:

    alpha     = slider(0, 1, step=0.01, default=0.5)
    optimizer = dropdown(["adam", "sgd"], default="adam")
    epochs    = number(default=10, min=1, max=200)

The cell is **never executed**. It is parsed with ``ast`` to extract, per
control, the target variable name (a DAG ``defines``) and a
``WidgetDescriptor`` (kind + params + resolved default). Widget cells have no
upstream, so ``references`` is always empty.

Only structural analysis lives here (what defines exist, what the descriptors
are, and structural errors). Semantic validation — slider ranges, defaults in
bounds — is surfaced as advisory diagnostics by
``annotation_validation._validate_widget_cell_annotations``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

# Control kind → its positional parameter names, in order. Any keyword argument
# is accepted too; unknown keywords are captured in ``params`` and flagged by
# validation, not here. Keep the set small + explicit (JSON-scalar values only).
_POSITIONAL_PARAMS: dict[str, list[str]] = {
    "slider": ["min", "max"],
    "number": ["default"],
    "dropdown": ["options"],
    "text": ["default"],
    "checkbox": ["default"],
}
WIDGET_KINDS = frozenset(_POSITIONAL_PARAMS)


@dataclass
class WidgetDescriptor:
    """One control declared by a widget cell."""

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    default: Any = None


@dataclass
class WidgetAnalysis:
    """Structural analysis of a widget cell."""

    defines: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)  # always empty
    descriptors: list[WidgetDescriptor] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _resolve_default(kind: str, params: dict[str, Any]) -> Any:
    """Fill in a control's default when the source omits ``default=``."""
    if "default" in params:
        return params["default"]
    if kind == "slider":
        return params.get("min")
    if kind == "dropdown":
        options = params.get("options")
        return options[0] if isinstance(options, list) and options else None
    if kind == "checkbox":
        return False
    if kind == "text":
        return ""
    return None  # number with no default


def _build_descriptor(name: str, kind: str, call: ast.Call) -> WidgetDescriptor | str:
    """Extract a descriptor from a ``name = kind(...)`` call, or an error string."""
    params: dict[str, Any] = {}
    positional_names = _POSITIONAL_PARAMS[kind]

    for index, arg in enumerate(call.args):
        if index >= len(positional_names):
            return f"`{name} = {kind}(...)` takes at most {len(positional_names)} positional argument(s)"
        try:
            params[positional_names[index]] = ast.literal_eval(arg)
        except (ValueError, SyntaxError):
            return f"`{name} = {kind}(...)` arguments must be literal values"

    for keyword in call.keywords:
        if keyword.arg is None:
            return f"`{name} = {kind}(...)` does not accept ``**kwargs``"
        try:
            params[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return f"`{name} = {kind}(...)` arguments must be literal values"

    return WidgetDescriptor(
        name=name, kind=kind, params=params, default=_resolve_default(kind, params)
    )


def analyze_widget_cell(source: str) -> WidgetAnalysis:
    """Parse a widget cell into its controls (defines + descriptors)."""
    result = WidgetAnalysis()
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        result.errors.append(f"Widget cell has invalid syntax: {exc.msg}")
        return result

    seen: set[str] = set()
    for node in tree.body:
        # Only ``name = control(...)`` statements declare controls. Blank lines,
        # comments (stripped by the parser), and anything else are ignored so a
        # stray line doesn't abort the whole panel — but a malformed *control*
        # (unknown kind, non-literal arg) is reported.
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name):
            continue

        name = node.targets[0].id
        kind = node.value.func.id
        if kind not in WIDGET_KINDS:
            result.errors.append(
                f"Unknown widget control `{kind}` for `{name}` "
                f"(expected one of {', '.join(sorted(WIDGET_KINDS))})"
            )
            continue
        if name in seen:
            result.errors.append(f"Duplicate widget variable `{name}`")
            continue

        descriptor = _build_descriptor(name, kind, node.value)
        if isinstance(descriptor, str):
            result.errors.append(descriptor)
            continue

        seen.add(name)
        result.descriptors.append(descriptor)
        result.defines.append(name)

    return result

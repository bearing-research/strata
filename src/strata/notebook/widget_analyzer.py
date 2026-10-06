"""Analyzer for widget cells.

A widget cell is one ``name = control(...)`` line per control, e.g.
``alpha = slider(0, 1, step=0.01, default=0.5)``. It is never executed: ``ast``
yields each control's target name (a ``defines``) and a ``WidgetDescriptor``.
``references`` is always empty. Semantic checks (ranges, defaults in bounds) are
advisory diagnostics in ``annotation_validation``.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any

# Control kind to positional parameter names. Unknown keywords land in
# ``params`` and are flagged by validation. JSON-scalar values only.
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


def _nice_slider_step(low: float, high: float) -> float | int:
    """Pick a readable slider increment for a ``[low, high]`` range.

    About 100 steps, snapped to a 1/2/5 x 10^n value; integer bounds with a whole
    result step by ints. ``slider(0, 1)`` gives ``0.01``.
    """
    span = high - low
    raw = span / 100.0
    exponent = math.floor(math.log10(raw))
    base = 10.0**exponent
    fraction = raw / base
    if fraction < 1.5:
        nice = 1.0
    elif fraction < 3.5:
        nice = 2.0
    elif fraction < 7.5:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * base
    if exponent < 0:  # trim binary-float noise on sub-integer steps
        step = round(step, -exponent + 1)
    if isinstance(low, int) and isinstance(high, int) and step.is_integer():
        return int(step)
    return step


def _default_slider_step(params: dict[str, Any]) -> None:
    """Fill in ``step`` for a slider when the source omits it (in place).

    No-op unless ``min`` and ``max`` are numeric and ``max > min``; bad ranges are
    left for advisory validation to flag.
    """
    if "step" in params:
        return
    low, high = params.get("min"), params.get("max")
    if not isinstance(low, int | float) or isinstance(low, bool):
        return
    if not isinstance(high, int | float) or isinstance(high, bool):
        return
    if high <= low:
        return
    params["step"] = _nice_slider_step(low, high)


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
            allowed = len(positional_names)
            return f"`{name} = {kind}(...)` takes at most {allowed} positional argument(s)"
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

    if kind == "slider":
        _default_slider_step(params)

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
        # Only ``name = control(...)`` declares a control; other lines are ignored so a
        # stray line doesn't abort the panel, but a malformed control is reported.
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


def _coerce_one(descriptor: WidgetDescriptor, value: Any) -> Any:
    """Coerce/clamp one incoming value to a control's type and bounds.

    ``None`` means unrepresentable (non-numeric slider value, unknown dropdown
    option); the caller drops it and keeps the prior value.
    """
    kind = descriptor.kind
    if kind in ("slider", "number"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        # NaN/inf have no JSON form: persisted, they make the notebook unopenable.
        if not math.isfinite(num):
            return None
        low, high = descriptor.params.get("min"), descriptor.params.get("max")
        if isinstance(low, int | float) and num < low:
            num = float(low)
        if isinstance(high, int | float) and num > high:
            num = float(high)
        # Keep int-ness so an int control at its default is a cache hit (10, not 10.0).
        if isinstance(descriptor.default, int) and not isinstance(descriptor.default, bool):
            if num.is_integer():
                return int(num)
        return num
    if kind == "checkbox":
        return bool(value)
    if kind == "dropdown":
        options = descriptor.params.get("options") or []
        return value if value in options else None
    if kind == "text":
        return str(value)
    return None


def coerce_widget_values(
    descriptors: list[WidgetDescriptor], values: dict[str, Any]
) -> dict[str, Any]:
    """Validate and coerce incoming control values against the cell's descriptors.

    Unknown names and uncoercible values are dropped; sliders and numbers are
    clamped to their range.
    """
    by_name = {d.name: d for d in descriptors}
    clean: dict[str, Any] = {}
    for name, value in values.items():
        descriptor = by_name.get(name)
        if descriptor is None:
            continue
        coerced = _coerce_one(descriptor, value)
        if coerced is not None:
            clean[name] = coerced
    return clean


def descriptor_provenance(descriptor: WidgetDescriptor, value: object) -> str:
    """Content hash of one control's declaration (kind + params) and current value.

    Returning a control to a prior value reproduces the prior hash.
    """
    import hashlib
    import json

    descriptor_json = json.dumps(
        {"kind": descriptor.kind, "params": descriptor.params}, sort_keys=True, default=str
    )
    value_json = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(f"{descriptor_json}\x00{value_json}".encode()).hexdigest()

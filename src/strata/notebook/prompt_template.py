"""A prompt cell's template: text with ``{{ expression }}`` placeholders.

The one reading of the syntax, shared by analysis (what the cell reads, what keeps it
from running) and rendering (what the model receives), so the two cannot disagree.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

# The escape first, so ``\{{`` never opens a placeholder.
_TOKEN_RE = re.compile(r"\\(\{[{%])|\{\{(.*?)\}\}|\{\{|\{%", re.DOTALL)


class PromptTemplateError(ValueError):
    """A template that cannot be filled in; nothing should reach the model."""


@dataclass(frozen=True)
class Placeholder:
    """One ``{{ expression }}`` the renderer can evaluate."""

    text: str
    expression: str
    root: str


@dataclass(frozen=True)
class TemplateProblem:
    """Why part of a template cannot be read, and where in the template it is."""

    message: str
    offset: int


def _root_name(node: ast.AST) -> str | None:
    """The name *node* reads, if it is a name, ``.attribute`` or no-argument ``.method()``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return None if node.attr.startswith("_") else _root_name(node.value)
    if isinstance(node, ast.Call) and not node.args and not node.keywords:
        return _root_name(node.func) if isinstance(node.func, ast.Attribute) else None
    return None


def _placeholder(text: str, expression: str) -> Placeholder | None:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    root = _root_name(tree.body)
    return None if root is None else Placeholder(text, expression.strip(), root)


def parse_template(template: str) -> tuple[list[str | Placeholder], list[TemplateProblem]]:
    """Split *template* into literal text and placeholders, with what could not be read."""
    parts: list[str | Placeholder] = []
    problems: list[TemplateProblem] = []
    position = 0
    for match in _TOKEN_RE.finditer(template):
        parts.append(template[position : match.start()])
        position = match.end()
        token = match.group(0)
        if match.group(1) is not None:
            parts.append(match.group(1))
        elif match.group(2) is not None:
            placeholder = _placeholder(token, match.group(2))
            if placeholder is None:
                problems.append(
                    TemplateProblem(
                        f"`{token}` is not a template expression: use `name`, "
                        "`name.attribute` or `name.method()`. "
                        r"Write `\{{` for a literal `{{`.",
                        match.start(),
                    )
                )
            else:
                parts.append(placeholder)
        elif token == "{{":
            problems.append(
                TemplateProblem(
                    r"`{{` has no closing `}}`. Write `\{{` for a literal `{{`.", match.start()
                )
            )
        else:
            problems.append(
                TemplateProblem(
                    "`{%` is not template syntax: conditions and loops belong in a Python "
                    r"cell upstream. Write `\{%` for a literal `{%`.",
                    match.start(),
                )
            )
    parts.append(template[position:])
    # ``{% if %}...{% endif %}`` is one mistake, not two; the first stays where it is.
    first_of_each: dict[str, TemplateProblem] = {}
    for problem in problems:
        first_of_each.setdefault(problem.message, problem)
    return parts, list(first_of_each.values())

"""Analyzer for prompt-type notebook cells.

Extracts ``{{ expr }}`` references for DAG building and determines
the output variable name from the ``@name`` annotation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from strata.notebook.annotations import (
    iter_annotation_block,
    parse_annotation_directive,
    strip_leading_annotations,
)

_TEMPLATE_VAR_RE = re.compile(r"\{\{\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*(?:\([^)]*\))?)*)\s*\}\}")

# Python builtins that should not be treated as upstream references
_BUILTINS = frozenset(
    {
        "True",
        "False",
        "None",
        "print",
        "len",
        "range",
        "str",
        "int",
        "float",
        "list",
        "dict",
        "set",
        "tuple",
        "type",
        "isinstance",
        "sorted",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sum",
        "min",
        "max",
        "abs",
        "round",
        "any",
        "all",
        "open",
        "input",
        "format",
        "repr",
    }
)


@dataclass
class PromptAnalysis:
    """Analysis result for a prompt cell."""

    name: str = "result"
    defines: list[str] = field(default_factory=lambda: ["result"])
    references: list[str] = field(default_factory=list)
    model: str | None = None
    temperature: float | None = None
    output_type: str | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None
    template_body: str = ""
    # ``@output_schema`` — parsed JSON schema object. When set, we send
    # provider-native structured output (OpenAI's ``json_schema``, or
    # ``json_object`` fallback for providers that don't support schemas)
    # so the response comes back as validated JSON instead of free-form
    # text. ``output_schema_raw`` keeps the original annotation string so
    # validators can distinguish "user wrote a bad schema" from "user
    # didn't write one at all".
    output_schema: dict[str, Any] | None = None
    output_schema_raw: str | None = None
    output_schema_error: str | None = None
    # ``@validate_retries N`` — total attempts for the validate-and-retry
    # loop (1 initial call + N-1 retries). ``None`` means "use the
    # executor default". Only has effect when ``output_schema`` is set;
    # without a schema there's nothing to validate against.
    validate_retries: int | None = None


def analyze_prompt_cell(source: str) -> PromptAnalysis:
    """Analyze a prompt cell's source to extract references and config.

    The source format is::

        # @name summary
        # @model claude-sonnet-4-20250514
        # @temperature 0.0
        # @output json
        # @system You are a data analyst.
        Summarize {{ df }} by category and list {{ metrics }}.

    Returns:
        PromptAnalysis with defines, references, and prompt config.
    """
    result = PromptAnalysis()

    # Walk the leading annotation block via the shared helper so this
    # path agrees with parse_annotations / annotation_validation on what
    # counts as a directive line.
    for _lineno, line in iter_annotation_block(source):
        parsed = parse_annotation_directive(line)
        if parsed is None:
            continue
        key, value = parsed

        if key == "name":
            if value and value.isidentifier():
                result.name = value
        elif key == "model":
            result.model = value or None
        elif key == "temperature":
            try:
                result.temperature = float(value)
            except ValueError:
                pass
        elif key == "output":
            result.output_type = value or None
        elif key == "max_tokens":
            try:
                result.max_tokens = int(value)
            except ValueError:
                pass
        elif key == "system":
            result.system_prompt = value or None
        elif key == "validate_retries":
            try:
                parsed_retries = int(value)
            except ValueError:
                pass
            else:
                if parsed_retries >= 1:
                    result.validate_retries = parsed_retries
        elif key == "output_schema":
            if value:
                result.output_schema_raw = value
                try:
                    parsed_schema = json.loads(value)
                except json.JSONDecodeError as exc:
                    result.output_schema_error = f"Invalid JSON in @output_schema: {exc.msg}"
                else:
                    if isinstance(parsed_schema, dict):
                        result.output_schema = parsed_schema
                    else:
                        result.output_schema_error = "@output_schema must be a JSON object"

    result.template_body = strip_leading_annotations(source).strip()

    # Extract {{ var }} references from template body
    refs: list[str] = []
    for match in _TEMPLATE_VAR_RE.finditer(result.template_body):
        expr = match.group(1)
        # Extract the root variable name (before any . or ())
        root_var = expr.split(".")[0].split("(")[0]
        if root_var and root_var not in _BUILTINS and root_var not in refs:
            refs.append(root_var)

    result.references = refs
    result.defines = [result.name]

    return result

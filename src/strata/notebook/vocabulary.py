"""Strata's annotations, declared on cellgraph (design 6).

Nothing at run time imports this module yet: the conformance replay does
(``tests/contract/cellgraph/test_replay.py``), and the facade that adopts cellgraph
is the next step. Each annotation reuses its parser from ``annotations.py`` and
the value checks of ``annotation_validation._value_problem``, so a directive the
validator refuses is a parse error here, with the same code.

Mounts, tables, fetches and datasets bind names only in Python and R cells, and a
mount is set in three layers, so their effects come from ``cell_effects`` rather
than from the annotations: the hook merges the layers by name as
``resolve_cell_mounts`` does. The host passes, per cell, ``meta = {"language": ...,
"mounts": [...], "notebook_mounts": [...], "connections": [...]}``: ``mounts`` with
the notebook-level ones folded in, as ``parse_notebook`` does, and
``notebook_mounts`` the notebook-level ones alone.

Strata reports every directive line it cannot read and keeps the lines it can, so
the parsers here return ``cg.Partial`` for a mix of the two.
"""

from __future__ import annotations

import ast
import dataclasses
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

import cellgraph as cg
from cellgraph.python import analyze_cell

from strata.notebook.annotation_validation import (
    _SUPPORTED_MOUNT_SCHEMES,
    _find_annotation_line,
    _table_uri_malformed,
    _value_problem,
)
from strata.notebook.annotations import (
    MARKDOWN_DIRECTIVE_KEYS,
    CellAnnotations,
    LoopAnnotation,
    _merge_loop_annotation,
    _parse_cache_annotation,
    _parse_dataset_annotation,
    _parse_fetch_annotation,
    _parse_mount_annotation,
    _parse_sql_annotation,
    _parse_table_annotation,
    _parse_write_table_annotation,
    parse_annotations,
)
from strata.notebook.models import CellLanguage, CellState, MountMode, MountSpec
from strata.notebook.mounts import resolve_cell_mounts
from strata.notebook.prompt_analyzer import analyze_prompt_cell
from strata.notebook.widget_analyzer import analyze_widget_cell

# Languages whose bound names (mounts, tables, fetches, datasets) leave references.
_BINDING_LANGUAGES = frozenset({CellLanguage.PYTHON.value, CellLanguage.R.value})


def _problem(directive: cg.Directive) -> cg.AnnotationError | None:
    """The error ``_value_problem`` gives the directive, if any."""
    problem = _value_problem(directive.key, directive.value, CellLanguage.PYTHON)
    if problem is None:
        return None
    code, why = problem
    return cg.AnnotationError(
        f"`@{directive.key} {directive.value}` could not be read ({why})",
        code=code,
        line=directive.line,
    )


def _partial(value: Any, errors: list[cg.AnnotationError]) -> Any:
    return cg.Partial(value, errors) if errors else value


def _read(inner: cg.Parser) -> cg.Parser:
    """*inner* over the lines ``_value_problem`` accepts; every other line is an error."""

    def parse(directives: list[cg.Directive]) -> Any:
        errors: list[cg.AnnotationError] = []
        good: list[cg.Directive] = []
        for directive in directives:
            problem = _problem(directive)
            if problem is None:
                good.append(directive)
            else:
                errors.append(problem)
        if not good:
            raise ExceptionGroup("unreadable directives", errors)
        try:
            value = inner(good)
        except cg.AnnotationError as exc:
            raise ExceptionGroup("unreadable directives", [*errors, exc]) from None
        if isinstance(value, cg.Partial):
            errors.extend(value.errors)
            value = value.value
        return _partial(value, errors)

    return cg.mark_stacked(parse) if cg.is_stacked(inner) else parse


def _last(convert: Callable[[str], Any]) -> cg.Parser:
    """The last readable line's value, converted."""
    return _read(lambda directives: convert(directives[-1].value))


_OFF = frozenset({"off", "false", "no", "0"})


def _flag(directives: list[cg.Directive]) -> bool:
    return directives[-1].value.strip().lower() not in _OFF


def _lines(parse_one: Callable[[str], Any | None], code: str, hint: str) -> cg.Parser:
    """One value per line, each read by *parse_one*; a line it cannot read is *code*."""

    def parse(directives: list[cg.Directive]) -> Any:
        values: list[Any] = []
        errors: list[cg.AnnotationError] = []
        for directive in directives:
            value = parse_one(directive.value)
            if value is None:
                errors.append(
                    cg.AnnotationError(
                        f"`@{directive.key} {directive.value}` could not be read ({hint})",
                        code=code,
                        line=directive.line,
                    )
                )
            else:
                values.append(value)
        return _partial(values, errors)

    return cg.mark_stacked(parse)


def _parse_env(directives: list[cg.Directive]) -> dict[str, str]:
    env: dict[str, str] = {}
    for directive in directives:
        eq = directive.value.find("=")
        env[directive.value[:eq].strip()] = directive.value[eq + 1 :].strip()
    return env


def _parse_output_dirs(directives: list[cg.Directive]) -> Any:
    names: list[str] = []
    errors: list[cg.AnnotationError] = []
    for directive in directives:
        if not directive.value.isidentifier():
            errors.append(
                cg.AnnotationError(
                    f"`@output_dir {directive.value}` could not be read "
                    "(expected `@output_dir <name>`, a Python identifier)",
                    code="output_dir_unreadable",
                    line=directive.line,
                )
            )
        elif directive.value not in names:
            names.append(directive.value)
    return _partial(names, errors)


def _with_result(apply: Callable[[CellAnnotations, str], None], field: str) -> cg.Parser:
    """Fold every line into one ``CellAnnotations`` field, as ``parse_annotations`` does."""

    def parse(directives: list[cg.Directive]) -> Any:
        result = CellAnnotations()
        for directive in directives:
            apply(result, directive.value)
        return getattr(result, field)

    return _read(parse)


def _parse_loop(directives: list[cg.Directive]) -> Any:
    """``@loop`` and ``@loop_until`` in source order, then the checks that block a loop.

    Every line is folded, readable or not, as ``parse_annotations`` does, so a bad
    token does not hide a good ``carry=``; each problem is reported.
    """
    errors = [e for e in map(_problem, directives) if e is not None]
    result = CellAnnotations()
    for directive in directives:
        if directive.key == "loop":
            _merge_loop_annotation(result, directive.value)
        elif result.loop is None:
            result.loop = LoopAnnotation(max_iter=0, carry="", until_expr=directive.value)
        else:
            result.loop.until_expr = directive.value
    loop = result.loop
    assert loop is not None
    # ``_find_annotation_line(source, "loop")``: the first line whose key starts with it.
    line = directives[0].line
    until_line = next((d.line for d in directives if d.key == "loop_until"), line)
    if loop.max_iter <= 0:
        errors.append(
            cg.AnnotationError(
                "`@loop` needs max_iter=<N>", code="loop_missing_max_iter", line=line
            )
        )
    if not loop.carry:
        errors.append(
            cg.AnnotationError("`@loop` needs carry=<var>", code="loop_missing_carry", line=line)
        )
    if loop.until_expr:
        try:
            ast.parse(loop.until_expr, mode="eval")
        except SyntaxError as exc:
            errors.append(
                cg.AnnotationError(
                    f"the loop condition does not parse: {exc.msg}",
                    code="loop_until_syntax_error",
                    line=until_line,
                )
            )
    return _partial(loop, errors)


def _loop_effects(loop: Any) -> list[cg.Effect]:
    effects: list[cg.Effect] = []
    if loop.carry and loop.start_from_cell is None:
        # Iteration 0 reads the carry from upstream even though the body assigns it.
        effects.append(cg.Reads([loop.carry], even_if_defined=True))
    elif loop.carry:
        effects.append(cg.Seeds(loop.start_from_cell, loop.carry))
    if loop.until_expr:
        # The condition runs in the cell's namespace: a name the body doesn't bind is
        # an input like any other.
        names = analyze_cell(loop.until_expr).references
        effects.append(cg.Reads(names, even_if_defined=False))
    return effects


def _validate_loop(loop: Any, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    if loop.carry and context.defines and loop.carry not in context.defines:
        return [
            cg.Diagnostic(
                "loop_carry_unknown",
                cg.Severity.WARN,
                f"the carry `{loop.carry}` is not defined by the cell",
                context.cell.id,
                "loop",
                _find_annotation_line(context.cell.source, "loop"),
            )
        ]
    return []


# The cells ``validate_cell_annotations`` runs no mount or table checks on.
_NO_INPUT_CHECKS = {
    CellLanguage.MARKDOWN.value,
    CellLanguage.PROMPT.value,
    CellLanguage.SQL.value,
    CellLanguage.WIDGET.value,
}


def _validate_mounts(mounts: Any, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    """Strata's mount checks: the URI scheme, and a name a notebook-level mount has."""
    cell = context.cell
    if cell.language in _NO_INPUT_CHECKS:
        return []
    notebook_mount_names = {m.name for m in cell.meta.get("notebook_mounts", [])}
    diagnostics: list[cg.Diagnostic] = []
    for mount in mounts:
        line = _find_annotation_line(cell.source, "mount", mount.name)
        scheme = mount.uri.split("://")[0].lower() if "://" in mount.uri else ""
        if not scheme or scheme not in _SUPPORTED_MOUNT_SCHEMES:
            diagnostics.append(
                cg.Diagnostic(
                    "mount_uri_unsupported",
                    cg.Severity.WARN,
                    f"`@mount {mount.name}` uses unsupported URI scheme '{scheme or mount.uri}'",
                    cell.id,
                    "mount",
                    line,
                )
            )
        if mount.name in notebook_mount_names:
            diagnostics.append(
                cg.Diagnostic(
                    "mount_shadows_notebook",
                    cg.Severity.INFO,
                    f"`@mount {mount.name}` overrides the notebook-level mount with the same name",
                    cell.id,
                    "mount",
                    line,
                )
            )
    return diagnostics


def _validate_tables(tables: Any, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    """Strata's table checks: the URI, a name the cell defines, and a repeated name."""
    cell = context.cell
    if cell.language in _NO_INPUT_CHECKS:
        return []
    diagnostics: list[cg.Diagnostic] = []
    for table in tables:
        line = _find_annotation_line(cell.source, "table", table.name)
        if _table_uri_malformed(table.uri):
            diagnostics.append(
                cg.Diagnostic(
                    "table_uri_malformed",
                    cg.Severity.WARN,
                    f"`@table {table.name}` URI should be `<warehouse>#<namespace>.<table>`, "
                    "`<catalog>:<namespace>.<table>` or `<namespace>.<table>`",
                    cell.id,
                    "table",
                    line,
                )
            )
        if table.name in context.defines:
            diagnostics.append(
                cg.Diagnostic(
                    "table_shadows_define",
                    cg.Severity.WARN,
                    f"`@table {table.name}` injects a variable that this cell also defines",
                    cell.id,
                    "table",
                    line,
                )
            )
    names = [table.name for table in tables]
    for dup in sorted({name for name in names if names.count(name) > 1}):
        diagnostics.append(
            cg.Diagnostic(
                "table_duplicate_name",
                cg.Severity.ERROR,
                f"`@table {dup}` is declared more than once",
                cell.id,
                "table",
                _find_annotation_line(cell.source, "table", dup),
            )
        )
    return diagnostics


def _validate_output_schema(schema: Any, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    """A prompt cell's ``@output_schema`` must be a JSON Schema (``prompt_analyzer``)."""
    cell = context.cell
    if cell.language != CellLanguage.PROMPT.value:
        return []
    error = analyze_prompt_cell(cell.source).output_schema_error
    if not error:
        return []
    line = _find_annotation_line(cell.source, "output_schema")
    return [
        cg.Diagnostic(
            "prompt_output_schema_invalid", cg.Severity.ERROR, error, cell.id, "output_schema", line
        )
    ]


def _validate_name(name: str, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    """A prompt cell's name is its output variable, so every ``@name`` line must be one."""
    cell = context.cell
    if cell.language != CellLanguage.PROMPT.value:
        return []
    return [
        cg.Diagnostic(
            "name_unreadable",
            cg.Severity.ERROR,
            f"`@name {d.value}`: a prompt cell's name is its output variable, "
            "so a Python identifier",
            cell.id,
            "name",
            d.line,
        )
        for d in cg.GRAMMAR.directives(cell.source)
        if d.key == "name" and d.value and not d.value.isidentifier()
    ]


def _validate_app(value: str, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    """Every ``@app`` line must be ``hide``, except in markdown, where nothing is checked."""
    cell = context.cell
    if cell.language == CellLanguage.MARKDOWN.value:
        return []
    directives = [d for d in cg.GRAMMAR.directives(cell.source) if d.key == "app"]
    return [
        cg.Diagnostic(
            error.code or "app_unreadable",
            cg.Severity.ERROR,
            str(error),
            cell.id,
            "app",
            error.line,
        )
        for error in filter(None, map(_problem, directives))
    ]


def _validate_per_variant(group: Any, context: cg.ValidationContext) -> list[cg.Diagnostic]:
    source = context.cell.source
    if not any(write.mode == "overwrite" for write in parse_annotations(source).table_writes):
        return []
    return [
        cg.Diagnostic(
            "per_variant_write_table_overwrite",
            cg.Severity.ERROR,
            "every variant writes the same table, so `overwrite` keeps only the last one's rows",
            context.cell.id,
            "per_variant",
            _find_annotation_line(source, "write_table", "overwrite"),
        )
    ]


def cell_effects(meta: Mapping[str, Any], parsed: Mapping[str, Any]) -> list[cg.Effect]:
    """The effects of the three mount layers and of the inputs a cell binds."""
    language = meta.get("language")
    mounts: list[MountSpec] = resolve_cell_mounts(
        [], list(meta.get("mounts", [])), parsed.get("mount", [])
    )
    effects: list[cg.Effect] = []
    for mount in mounts:
        if mount.mode == MountMode.READ_WRITE:
            # No input: the mount has no fingerprint, and the cell's value outlives its key.
            effects.append(cg.NoCache())
    if language not in _BINDING_LANGUAGES:
        return effects
    for mount in mounts:
        effects.append(cg.Binds([mount.name]))
        if mount.mode != MountMode.READ_WRITE:
            effects.append(cg.Input("mount", mount.name, spec=mount))
    # A repeated table name is ``table_duplicate_name``; bind it once.
    tables = {table.name: table for table in reversed(parsed.get("table", []))}
    for table in reversed(tables.values()):
        effects.append(cg.Binds([table.name, f"{table.name}_snapshot"]))
        effects.append(cg.Input("table", table.name, spec=table))
    for fetch in parsed.get("fetch", []):
        effects.append(cg.Binds([fetch.name]))
        effects.append(cg.Input("fetch", fetch.name, spec=fetch))
    for dataset in parsed.get("dataset", []):
        effects.append(cg.Binds([dataset.name]))
        effects.append(cg.Input("dataset", dataset.name, spec=dataset))
    return effects


def _strata_annotations() -> list[cg.Annotation]:
    return [
        cg.Annotation("worker", parse=_last(str), in_key=False),
        cg.Annotation("timeout", parse=_last(float), in_key=False),
        cg.Annotation("env", parse=_read(cg.mark_stacked(_parse_env)), in_key=False),
        cg.Annotation(
            "mount",
            parse=_lines(
                _parse_mount_annotation,
                "mount_unreadable",
                "expected `@mount <name> <uri> [ro|rw] [credential=<name>]`",
            ),
            in_key=False,
            validate=_validate_mounts,
        ),
        cg.Annotation(
            "table",
            parse=_lines(
                _parse_table_annotation,
                "table_unreadable",
                "expected `@table <name> <uri> [snapshot=<id>]`",
            ),
            in_key=False,
            validate=_validate_tables,
        ),
        cg.Annotation(
            "fetch",
            parse=_lines(
                _parse_fetch_annotation,
                "fetch_unreadable",
                "expected `@fetch <name> <url> [sha256=<hex>] [refetch=never|stale|always]`",
            ),
            in_key=False,
        ),
        cg.Annotation(
            "dataset",
            parse=_lines(
                _parse_dataset_annotation,
                "dataset_unreadable",
                "expected `@dataset <var> <name>[@<alias>|@v=<n>]`",
            ),
            in_key=False,
        ),
        # Where and how a result is written is part of what the run does.
        cg.Annotation(
            "write_table",
            parse=_lines(
                _parse_write_table_annotation,
                "write_table_unreadable",
                "expected `@write_table <var> <catalog>:<namespace>.<table> "
                "append|overwrite|merge`, with `on=<col>,...` after merge",
            ),
            in_key=True,
            encode=lambda writes: [write.model_dump() for write in writes],
            effects=lambda writes: [cg.Keeps([write.variable for write in writes])],
        ),
        cg.Annotation(
            "output_dir",
            parse=cg.mark_stacked(_parse_output_dirs),
            in_key=True,
            effects=lambda names: [cg.Binds(names), cg.Defines(names), cg.Keeps(names)],
        ),
        cg.Annotation(
            "stochastic",
            parse=_read(_flag),
            effects=lambda on: [cg.NoCache()] if on else [],
            in_key=False,
        ),
        cg.Annotation("model", parse=_last(str), in_key=False),
        cg.Annotation("temperature", parse=_last(float), in_key=False),
        cg.Annotation("output", parse=_last(str), in_key=False),
        cg.Annotation("max_tokens", parse=_last(int), in_key=False),
        cg.Annotation("system", parse=_last(str), in_key=False),
        cg.Annotation("validate_retries", parse=_last(int), in_key=False),
        cg.Annotation(
            "output_schema", parse=_last(str), in_key=False, validate=_validate_output_schema
        ),
        cg.Annotation("sql", parse=_with_result(_parse_sql_annotation, "sql"), in_key=False),
        cg.Annotation("cache", parse=_with_result(_parse_cache_annotation, "cache"), in_key=False),
        cg.Annotation(
            "loop",
            also_reads=("loop_until",),
            parse=_parse_loop,
            in_key=True,
            encode=dataclasses.asdict,
            effects=_loop_effects,
            validate=_validate_loop,
        ),
        cg.Annotation("live", parse=_read(_flag), in_key=False),
        cg.Annotation("app", parse=lambda ds: ds[-1].value, in_key=False, validate=_validate_app),
    ]


def strata_vocabulary() -> cg.Vocabulary:
    """Core plus Strata's own annotations, with Strata's checks on the core ones."""
    core = cg.Vocabulary.core()
    # ``_value_problem`` checks every line of these, where core's parsers read one.
    for key in ("name", "after", "nocache", "per_variant"):
        core = core.replace(key, parse=_read(core[key].parse))
    core = core.replace("name", validate=_validate_name).replace(
        "per_variant", validate=_validate_per_variant
    )
    return core | cg.Vocabulary(_strata_annotations(), cell_effects=cell_effects)


# --- language analyzers -------------------------------------------------------------


class RAnalyzer:
    """Strata's R analyzer (``Rscript``), on cellgraph's protocol."""

    owns_key = True

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        from strata.notebook.languages.r.analyzer import _RAnalyzer

        state = CellState(id=cell.id, source=cell.source, language=CellLanguage.R)
        # The R analyzer discards its session.
        result = _RAnalyzer().analyze(state, None)  # ty: ignore[invalid-argument-type]
        return cg.Analyzed(
            defines=list(result.defines),
            references=list(result.references),
            builtin_references=list(result.builtin_references),
            mutation_defines=list(result.mutation_defines),
        )

    def normalize(self, body: str) -> str:
        return cg.strip_lines(body)


class SqlAnalyzer:
    """Strata's SQL analyzer: the output name and the ``:name`` placeholders. Host-keyed."""

    owns_key = False

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        from strata.notebook.languages.analyzer import _SqlAnalyzer

        state = CellState(id=cell.id, source=cell.source, language=CellLanguage.SQL)
        # It reads only the session's connections, for the dialect.
        session = SimpleNamespace(
            notebook_state=SimpleNamespace(connections=list(cell.meta.get("connections", [])))
        )
        result = _SqlAnalyzer().analyze(state, session)  # ty: ignore[invalid-argument-type]
        return cg.Analyzed(defines=list(result.defines), references=list(result.references))

    def normalize(self, body: str) -> str:
        return cg.strip_lines(body)


class PromptAnalyzer:
    """Strata's prompt analyzer: ``@name`` and the template's variables. Host-keyed."""

    owns_key = False

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        result = analyze_prompt_cell(cell.source)
        return cg.Analyzed(defines=list(result.defines), references=list(result.references))

    def normalize(self, body: str) -> str:
        return cg.strip_lines(body)


class NoNamesAnalyzer:
    """Markdown: no names, no edges, and its ``# @`` lines are headings, except ``@app``."""

    owns_key = True
    directives = MARKDOWN_DIRECTIVE_KEYS

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        return cg.Analyzed()

    def normalize(self, body: str) -> str:
        return cg.strip_lines(body)


class WidgetAnalyzer:
    """A widget cell: each control is a define."""

    owns_key = True

    def analyze(self, cell: cg.Cell) -> cg.Analyzed:
        return cg.Analyzed(defines=list(analyze_widget_cell(cell.source).defines))

    def normalize(self, body: str) -> str:
        return cg.strip_lines(body)


def register_analyzers() -> None:
    """Register Strata's analyzers with cellgraph; Python is cellgraph's own."""
    cg.register_language_analyzer(CellLanguage.R.value, RAnalyzer())
    cg.register_language_analyzer(CellLanguage.SQL.value, SqlAnalyzer())
    cg.register_language_analyzer(CellLanguage.PROMPT.value, PromptAnalyzer())
    cg.register_language_analyzer(CellLanguage.MARKDOWN.value, NoNamesAnalyzer())
    cg.register_language_analyzer(CellLanguage.WIDGET.value, WidgetAnalyzer())


register_analyzers()

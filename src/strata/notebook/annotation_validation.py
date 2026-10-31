"""Cross-reference validation for cell source annotations.

Runs on notebook open, reload and after a WS source flush, never while typing, and
again before each run: an ERROR keeps the cell from running (``blocking_diagnostic``).
"""

from __future__ import annotations

import ast
import difflib
import logging
from typing import TYPE_CHECKING

from strata.notebook.annotations import (
    _LOOP_START_FROM_RE,
    ANNOTATION_KEYS,
    iter_annotation_block,
    parse_annotation_directive,
    parse_annotations,
    unreadable_input_directives,
)
from strata.notebook.models import (
    AnnotationDiagnostic,
    CellLanguage,
    CellState,
    DiagnosticSeverity,
    NotebookState,
)
from strata.notebook.workers import notebook_worker_definitions_editable, resolve_worker_spec

if TYPE_CHECKING:
    from strata.notebook.dag import NotebookDag

logger = logging.getLogger(__name__)

_SUPPORTED_MOUNT_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "az", "azure"})


def _table_uri_malformed(uri: str) -> bool:
    """True when ``@table`` resolution could not name the table: no namespace or no table.

    Resolution hands the id to the Iceberg catalog, which takes nested namespaces
    (``db.sch.events``). The named-catalog split is skipped: a catalog name holds
    no dot, so the check passes or fails the same either way.
    """
    from strata.iceberg.catalog import PyIcebergCatalog

    _warehouse, table_id = PyIcebergCatalog.parse_table_uri(uri)
    parts = table_id.split(".")
    return len(parts) < 2 or not all(parts)


def blocking_diagnostic(
    cell: CellState,
    notebook_state: NotebookState,
    dag: NotebookDag | None = None,
) -> AnnotationDiagnostic | None:
    """The first ERROR diagnostic, which keeps the cell from running, or ``None``."""
    return next(
        (
            d
            for d in validate_cell_annotations(cell, notebook_state, dag)
            if d.severity == DiagnosticSeverity.ERROR
        ),
        None,
    )


def validate_cell_annotations(
    cell: CellState,
    notebook_state: NotebookState,
    dag: NotebookDag | None = None,
) -> list[AnnotationDiagnostic]:
    """Validate a cell's annotations against notebook-wide context.

    *dag* is the notebook's; without one (its build failed, which is reported on its
    own) the ``@per_variant`` group checks are skipped.
    """
    # Language-agnostic: any language can be a variant member, so these run before
    # language dispatch.
    _cell_annotations = parse_annotations(cell.source)
    variant_diagnostics = _validate_variant_annotation(cell, _cell_annotations, notebook_state)
    variant_diagnostics = variant_diagnostics + _validate_per_variant_annotation(
        cell, _cell_annotations, dag
    )
    if cell.language == CellLanguage.MARKDOWN:
        # Pure prose: ``# @worker`` would be a markdown heading, not an annotation.
        return variant_diagnostics
    variant_diagnostics = variant_diagnostics + _validate_recorded_inputs(cell)
    variant_diagnostics = variant_diagnostics + _validate_unreadable_values(cell)
    variant_diagnostics = variant_diagnostics + _validate_unknown_annotations(cell)
    if cell.language == CellLanguage.PROMPT:
        return variant_diagnostics + _validate_prompt_cell_annotations(cell)
    if cell.language == CellLanguage.SQL:
        return variant_diagnostics + _validate_sql_cell_annotations(cell, notebook_state)
    if cell.language == CellLanguage.WIDGET:
        return variant_diagnostics + _validate_widget_cell_annotations(cell)
    diagnostics: list[AnnotationDiagnostic] = list(variant_diagnostics)
    diagnostics.extend(_validate_module_export(cell, notebook_state))
    annotations = parse_annotations(cell.source)

    # --- worker_unknown ---
    # The catalog execution resolves against: in service mode the server's registry,
    # where members cannot declare workers at all.
    if annotations.worker and resolve_worker_spec(notebook_state, annotations.worker) is None:
        # A [[workers]] entry the server ignores reads as "defined" to the author.
        ignored = not notebook_worker_definitions_editable(notebook_state) and any(
            w.name == annotations.worker for w in notebook_state.workers
        )
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="worker_unknown",
                message=(
                    f"`@worker {annotations.worker}` is defined in notebook.toml, but this "
                    "server runs cells only on its registry's workers."
                    if ignored
                    else f"`@worker {annotations.worker}` is not a worker this notebook can use. "
                    "Execution will fail until the worker is added."
                ),
                line=_find_annotation_line(cell.source, "worker"),
            )
        )

    # --- mount checks ---
    notebook_mount_names = {m.name for m in notebook_state.mounts}
    for mount in annotations.mounts:
        line = _find_annotation_line(cell.source, "mount", mount.name)

        scheme = mount.uri.split("://")[0].lower() if "://" in mount.uri else ""
        if not scheme or scheme not in _SUPPORTED_MOUNT_SCHEMES:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="mount_uri_unsupported",
                    message=(
                        f"`@mount {mount.name}` uses unsupported URI scheme "
                        f"'{scheme or mount.uri}'. "
                        f"Supported: {', '.join(sorted(_SUPPORTED_MOUNT_SCHEMES))}."
                    ),
                    line=line,
                )
            )

        if mount.name in notebook_mount_names:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="mount_shadows_notebook",
                    message=(
                        f"`@mount {mount.name}` overrides the notebook-level "
                        f"mount with the same name."
                    ),
                    line=line,
                )
            )

    # --- table checks ---
    for table in annotations.tables:
        line = _find_annotation_line(cell.source, "table", table.name)

        if _table_uri_malformed(table.uri):
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="table_uri_malformed",
                    message=(
                        f"`@table {table.name}` URI should be "
                        "`<warehouse>#<namespace>.<table>`, "
                        "`<catalog>:<namespace>.<table>` or `<namespace>.<table>` "
                        "(e.g. file:///data/warehouse#nyc.trips)."
                    ),
                    line=line,
                )
            )

        # table_shadows_define: the injected variable hides a real definition
        if table.name in (cell.defines or []):
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="table_shadows_define",
                    message=(
                        f"`@table {table.name}` injects a variable that this "
                        "cell also defines; the definition wins and the "
                        "table URI is shadowed."
                    ),
                    line=line,
                )
            )

    # table_duplicate_name: snapshots are keyed by table name, so a duplicate makes one
    # @table win injection while both feed provenance. Execution rejects it; flag early.
    table_names = [t.name for t in annotations.tables]
    for dup in sorted({n for n in table_names if table_names.count(n) > 1}):
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="table_duplicate_name",
                message=(
                    f"`@table {dup}` is declared more than once; each @table must "
                    "have a unique name; the cell will fail to run."
                ),
                line=_find_annotation_line(cell.source, "table", dup),
            )
        )

    diagnostics.extend(_validate_loop_annotation(cell, annotations, notebook_state))

    return diagnostics


def _validate_recorded_inputs(cell: CellState) -> list[AnnotationDiagnostic]:
    """Report an input or output directive (``@mount``, ``@fetch``, ...) that did not parse."""
    hints = {
        "mount": "expected `@mount <name> <uri> [ro|rw] [credential=<name>]`",
        "table": "expected `@table <name> <uri> [snapshot=<id>]`",
        "fetch": "expected `@fetch <name> <url> [sha256=<hex>] [refetch=never|stale|always]`",
        "dataset": "expected `@dataset <var> <name>[@<alias>|@v=<n>]`",
        "write_table": (
            "expected `@write_table <var> <catalog>:<namespace>.<table> "
            "append|overwrite|merge`, with `on=<col>,...` after merge"
        ),
        "output_dir": "expected `@output_dir <name>`, a Python identifier",
    }
    return [
        _unreadable(f"{directive}_unreadable", directive, value, hints[directive], lineno)
        for lineno, directive, value in unreadable_input_directives(cell.source)
    ]


def _unreadable(code: str, key: str, value: str, why: str, line: int) -> AnnotationDiagnostic:
    shown = f"@{key} {value}" if value else f"@{key}"
    return AnnotationDiagnostic(
        severity=DiagnosticSeverity.ERROR,
        code=code,
        message=f"`{shown}` could not be read ({why}), so the cell does not run.",
        line=line,
    )


_FLAG_VALUES = frozenset({"", "on", "off", "true", "false", "yes", "no", "1", "0"})
# What ``@sql write=`` tells apart; ``on`` would read as off.
_SQL_WRITE_VALUES = frozenset({"true", "false", "yes", "no", "1", "0"})
_NEEDS_VALUE = frozenset(
    {"worker", "after", "name", "model", "system", "loop_until", "output_schema"}
)


def _value_problem(key: str, value: str, language: CellLanguage) -> tuple[str, str] | None:
    """``(code, why)`` when ``# @key value`` cannot be read as written, else ``None``.

    The parsers drop or default such a value, so the cell would run as a computation
    other than the one its source declares.
    """
    code = f"{key}_unreadable"
    if key in _NEEDS_VALUE and not value:
        return code, "it needs a value"
    if key == "name" and language == CellLanguage.PROMPT and not value.isidentifier():
        return code, "a prompt cell's name is its output variable, so a Python identifier"
    if key in ("nocache", "stochastic", "live") and value.lower() not in _FLAG_VALUES:
        return code, "expected nothing, or one of on, off, true, false, yes, no, 1, 0"
    if key == "per_variant":
        tokens = value.split()
        if len(tokens) > 1 or (tokens and not tokens[0].isidentifier()):
            return code, "expected at most one group name, a Python identifier"
    if key == "timeout" and not _positive(value, float):
        return "timeout_not_numeric", "expected a positive number of seconds"
    if key == "env" and value.find("=") <= 0:
        return "env_malformed", "expected `@env KEY=value`"
    if key == "temperature" and not _positive(value, float, allow_zero=True):
        return code, "expected a number, 0 or more"
    if key in ("max_tokens", "validate_retries") and not _positive(value, int):
        return code, "expected a positive integer"
    if key == "output" and value not in ("json", "text"):
        return code, "expected json or text"
    if key == "app" and value != "hide":
        return code, "expected `@app hide`"
    if key == "sql":
        for token in value.split():
            k, _, v = token.partition("=")
            if not (k == "connection" and v) and not (
                k == "write" and v.lower() in _SQL_WRITE_VALUES
            ):
                return code, "expected `@sql connection=<name> [write=true|false]`"
    if key == "loop":
        for token in value.split():
            k, _, v = token.partition("=")
            readable = bool(v) and (
                k in ("carry", "until")
                or (k == "max_iter" and _positive(v, int))
                or (k == "start_from" and _LOOP_START_FROM_RE.match(v) is not None)
            )
            if not readable:
                return code, (
                    "expected `@loop max_iter=<N> carry=<var> [until=<expr>] "
                    "[start_from=<cell>@iter=<k>]`, without spaces in a value"
                )
    if key == "cache":
        head, *rest = value.split() or [""]
        if head.startswith("ttl="):
            if rest or not _positive(head.removeprefix("ttl="), int):
                return "cache_ttl_invalid", "`ttl=` takes a positive integer (seconds)"
        elif rest or head not in ("fingerprint", "forever", "session", "snapshot"):
            return "cache_policy_unknown", (
                "expected fingerprint, forever, session, snapshot or ttl=<seconds>"
            )
    return None


def _positive(value: str, kind: type, *, allow_zero: bool = False) -> bool:
    try:
        number = kind(value)
    except ValueError:
        return False
    # ``not number > 0`` also refuses a NaN.
    return number >= 0 if allow_zero else number > 0


def _validate_unreadable_values(cell: CellState) -> list[AnnotationDiagnostic]:
    """Report each directive whose value cannot be read (``_value_problem``)."""
    diagnostics: list[AnnotationDiagnostic] = []
    for lineno, line_text in iter_annotation_block(cell.source):
        parsed = parse_annotation_directive(line_text)
        if parsed is None:
            continue
        key, value = parsed
        problem = _value_problem(key, value, cell.language)
        if problem is not None:
            diagnostics.append(_unreadable(problem[0], key, value, problem[1], lineno))
    return diagnostics


def _validate_module_export(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Warn when a cell's defs and classes are not safe to re-execute as a module.

    The slicer keeps imports, defs, classes and literal constants. Warns when a kept
    def/class references a name not imported or defined as a literal here, when a
    kept name is rebound by dropped runtime code, or when a lambda is assigned to a
    downstream-consumed name. Silent unless another cell references the affected
    names: private single-cell helpers are a common, safe pattern.
    """
    from strata.notebook.module_export import build_module_export_plan, runtime_binding_names

    # Upstream defines and this cell's runtime values can be hydrated into the synthetic
    # module, so a def closing over them isn't blocked. Mirrors
    # _write_module_export_outputs.
    cross_cell = frozenset(
        v
        for v in (*cell.references, *cell.builtin_references)
        for other in notebook_state.cells
        if other.id != cell.id and v in other.defines
    )
    injectable = cross_cell | runtime_binding_names(cell.source)
    plan = build_module_export_plan(cell.source, injectable=injectable)
    if plan.is_exportable:
        return []

    exported_code = [
        name
        for name, symbol in plan.exported_symbols.items()
        if symbol.kind in ("function", "async function", "class")
    ]
    blocked = sorted(set(exported_code) | plan.blocking_symbols)
    if not blocked:
        return []

    referenced_elsewhere: set[str] = set()
    for other in notebook_state.cells:
        if other.id == cell.id:
            continue
        referenced_elsewhere.update(other.references)
        referenced_elsewhere.update(other.builtin_references)
    if not referenced_elsewhere.intersection(blocked):
        return []

    names = ", ".join(f"`{n}`" for n in blocked)
    return [
        AnnotationDiagnostic(
            severity=DiagnosticSeverity.WARN,
            code="module_export_blocked",
            message=(
                f"This cell defines reusable code ({names}) that downstream cells "
                f"reference, but it can't be shared across cells: "
                f"{plan.format_error()}."
            ),
            line=None,
        )
    ]


def _validate_prompt_cell_annotations(cell: CellState) -> list[AnnotationDiagnostic]:
    """Surface prompt-cell annotation errors (e.g. malformed ``@output_schema``)."""
    from strata.notebook.prompt_analyzer import analyze_prompt_cell

    analysis = analyze_prompt_cell(cell.source)
    diagnostics: list[AnnotationDiagnostic] = []
    if analysis.output_schema_error:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="prompt_output_schema_invalid",
                message=analysis.output_schema_error,
                line=_find_annotation_line(cell.source, "output_schema"),
            )
        )
    return diagnostics


def _validate_widget_cell_annotations(cell: CellState) -> list[AnnotationDiagnostic]:
    """Surface structural and semantic widget-cell errors.

    Structural: unknown control, non-literal argument, duplicate variable.
    Semantic: slider range, default out of bounds.
    """
    from strata.notebook.widget_analyzer import analyze_widget_cell

    analysis = analyze_widget_cell(cell.source)
    diagnostics: list[AnnotationDiagnostic] = []

    for message in analysis.errors:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="widget_invalid",
                message=message,
                line=None,
            )
        )

    for descriptor in analysis.descriptors:
        line = _find_widget_line(cell.source, descriptor.name)
        diagnostics.extend(_validate_widget_descriptor(descriptor, line))

    return diagnostics


def _validate_widget_descriptor(descriptor, line: int | None) -> list[AnnotationDiagnostic]:
    """Semantic checks for one control: ranges + defaults in bounds."""
    diagnostics: list[AnnotationDiagnostic] = []
    params = descriptor.params
    low, high = params.get("min"), params.get("max")

    if (
        descriptor.kind == "slider"
        and isinstance(low, int | float)
        and isinstance(high, int | float)
    ):
        if low >= high:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="widget_bad_range",
                    message=(
                        f"`{descriptor.name}`: slider min ({low}) must be less than max ({high})."
                    ),
                    line=line,
                )
            )

    default = descriptor.default
    if descriptor.kind in ("slider", "number") and isinstance(default, int | float):
        if isinstance(low, int | float) and default < low:
            diagnostics.append(_default_oob(descriptor.name, default, line))
        elif isinstance(high, int | float) and default > high:
            diagnostics.append(_default_oob(descriptor.name, default, line))

    if descriptor.kind == "dropdown":
        options = params.get("options")
        if isinstance(options, list) and default is not None and default not in options:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="widget_default_not_an_option",
                    message=f"`{descriptor.name}`: default {default!r} is not one of the options.",
                    line=line,
                )
            )

    return diagnostics


def _default_oob(name: str, default, line: int | None) -> AnnotationDiagnostic:
    return AnnotationDiagnostic(
        severity=DiagnosticSeverity.WARN,
        code="widget_default_out_of_range",
        message=f"`{name}`: default {default} is outside the control's min/max range.",
        line=line,
    )


def _find_widget_line(source: str, name: str) -> int | None:
    """1-based line of the ``name = control(...)`` declaration, if found."""
    for index, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.lstrip()
        if stripped.startswith(f"{name} ") or stripped.startswith(f"{name}="):
            return index
    return None


def _validate_referenced_connection(
    conn,
    line: int | None,
) -> list[AnnotationDiagnostic]:
    """Diagnose the connection a SQL cell references.

    Reports an unknown ``driver`` (``connection_driver_unknown``) and an ``auth.*``
    literal instead of a ``${VAR}`` (``connection_auth_literal_secret``); the writer
    blanks such literals on the next save, silently breaking the connection.
    """
    diagnostics: list[AnnotationDiagnostic] = []

    # Lazy import: non-SQL paths skip the cost, and missing optional ADBC packages
    # don't crash validation.
    try:
        from strata.notebook.sql.registry import known_drivers

        registered = set(known_drivers())
    except ImportError:
        registered = set()

    if registered and conn.driver not in registered:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="connection_driver_unknown",
                message=(
                    f"Connection {conn.name!r} declares driver "
                    f"{conn.driver!r}, which is not registered. Known "
                    f"drivers: {', '.join(sorted(registered))}."
                ),
                line=line,
            )
        )

    # Keeps what counts as a ${VAR} indirection defined in one place.
    from strata.notebook.writer import is_auth_indirection

    for key, value in (conn.auth or {}).items():
        if not value:
            continue
        if is_auth_indirection(value):
            continue
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="connection_auth_literal_secret",
                message=(
                    f"Connection {conn.name!r} `auth.{key}` contains a "
                    "literal value, not a `${VAR}` reference. The "
                    "literal will be blanked on next save to keep "
                    "secrets off disk; switch to ${VAR} form to "
                    "preserve the binding."
                ),
                line=line,
            )
        )

    return diagnostics


def _validate_sql_cell_annotations(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Surface SQL-cell directive issues and problems with its referenced connection."""
    diagnostics: list[AnnotationDiagnostic] = []
    annotations = parse_annotations(cell.source)

    sql = annotations.sql
    if sql is None or not sql.connection:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="sql_connection_missing",
                message=(
                    "SQL cells require `# @sql connection=<name>`. The connection "
                    "must match a `[connections.<name>]` block in notebook.toml."
                ),
                line=_find_annotation_line(cell.source, "sql"),
            )
        )
    else:
        valid_by_name = {c.name: c for c in notebook_state.connections}
        malformed_by_name = {m.name: m for m in notebook_state.malformed_connections}
        sql_line = _find_annotation_line(cell.source, "sql")
        target = sql.connection
        if target in valid_by_name:
            diagnostics.extend(_validate_referenced_connection(valid_by_name[target], sql_line))
        elif target in malformed_by_name:
            mal = malformed_by_name[target]
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="connection_malformed",
                    message=(
                        f"Connection {target!r} is declared but failed to "
                        f"parse: {mal.error}. Fix the `[connections."
                        f"{target}]` block in notebook.toml."
                    ),
                    line=sql_line,
                )
            )
        else:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="sql_connection_unknown",
                    message=(
                        f"`@sql connection={target}` is not declared in this "
                        f"notebook. Add a `[connections.{target}]` block to "
                        "notebook.toml."
                    ),
                    line=sql_line,
                )
            )

    # Surface the analyzer's sqlglot ``parse_error`` as a diagnostic so it survives the
    # session boundary; otherwise parse failures never reach the UI.
    if (
        sql is not None
        and sql.connection
        and sql.connection in valid_by_name
        and notebook_state.cells is not None
    ):
        try:
            from strata.notebook.sql.analyzer import analyze_sql_cell
            from strata.notebook.sql.registry import get_adapter

            connection = valid_by_name[sql.connection]
            adapter = get_adapter(connection.driver)
            sql_analysis = analyze_sql_cell(cell.source, dialect=adapter.sqlglot_dialect)
            # Under the default fingerprint cache, a table the analyzer cannot
            # name makes the executor re-run the query every time.
            reruns = sql_analysis.cache_policy.kind == "fingerprint"
            if sql_analysis.parse_error:
                # The executor refuses unparseable SQL before connecting, whatever the cache policy.
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="sql_parse_error",
                        message=(
                            f"sqlglot couldn't parse this SQL cell: "
                            f"{sql_analysis.parse_error}. The cell will not run "
                            "until its SQL parses."
                        ),
                        line=None,
                    )
                )
            elif sql_analysis.unresolved_tables and reruns:
                named = ", ".join(sql_analysis.unresolved_tables)
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="sql_dynamic_table",
                        message=(
                            f"{named} is resolved only when the query runs (a "
                            "table named at run time, a table function or a "
                            "file), so the cache can't tell when it changes: "
                            "this cell re-runs every time. Add `# @cache session` "
                            "or `# @cache ttl=...` to reuse results."
                        ),
                        line=None,
                    )
                )
        except (KeyError, ImportError):
            # Driver not registered (connection_driver_unknown covers it) or sqlglot missing:
            # neither is a parse error.
            pass
        except Exception as exc:
            # A bug in table extraction, not the cell. Validation runs on open, so raising would
            # keep the notebook from opening.
            logger.exception("SQL analysis failed for cell %s", cell.id)
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="sql_analysis_failed",
                    message=(
                        f"Strata couldn't analyze this SQL cell ({type(exc).__name__}: "
                        f"{exc}). This is a Strata bug; running the cell may fail."
                    ),
                    line=None,
                )
            )

    return diagnostics


def _validate_loop_annotation(
    cell: CellState,
    annotations,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Validate ``@loop`` / ``@loop_until`` directives."""
    diagnostics: list[AnnotationDiagnostic] = []
    loop = annotations.loop
    if loop is None:
        return diagnostics

    loop_line = _find_annotation_line(cell.source, "loop")
    until_line = _find_annotation_line(cell.source, "loop_until") or loop_line

    if loop.max_iter <= 0:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="loop_missing_max_iter",
                message=(
                    "`@loop` requires a positive `max_iter=<N>`. "
                    "The loop cell must declare a safety bound on the iteration count."
                ),
                line=loop_line,
            )
        )

    if not loop.carry:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="loop_missing_carry",
                message=(
                    "`@loop` requires `carry=<variable>`. "
                    "The carry variable is threaded between iterations."
                ),
                line=loop_line,
            )
        )
    elif cell.defines and loop.carry not in cell.defines:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="loop_carry_unknown",
                message=(
                    f"`@loop carry={loop.carry}` does not match any top-level "
                    f"assignment in the cell. The cell must rebind "
                    f"`{loop.carry}` each iteration for the loop to make progress."
                ),
                line=loop_line,
            )
        )

    if loop.until_expr:
        try:
            ast.parse(loop.until_expr, mode="eval")
        except SyntaxError as exc:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="loop_until_syntax_error",
                    message=(
                        f"`@loop_until` expression is not a valid Python expression: {exc.msg}."
                    ),
                    line=until_line,
                )
            )

    if loop.start_from_cell is not None:
        known_cells = {c.id for c in notebook_state.cells}
        if loop.start_from_cell not in known_cells:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="loop_start_from_unknown",
                    message=(
                        f"`@loop start_from={loop.start_from_cell}@iter="
                        f"{loop.start_from_iter}` references a cell that does "
                        f"not exist in this notebook."
                    ),
                    line=loop_line,
                )
            )
        elif loop.start_from_cell == cell.id:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.ERROR,
                    code="loop_start_from_unknown",
                    message=(
                        "`@loop start_from` must reference a different cell: "
                        "a loop cell cannot seed itself from its own iterations."
                    ),
                    line=loop_line,
                )
            )

    return diagnostics


def _validate_variant_annotation(
    cell: CellState,
    annotations,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Validate ``# @variant`` membership against siblings and notebook.toml.

    - ``variant_contract_mismatch``: defines diverge from the union of the active
      siblings, so the diagnostic lands on the outlier.
    - ``variant_active_unknown``: notebook.toml selects a variant no cell provides.
    - ``variant_malformed``: the line did not parse into a (group, name) pair.
    """
    diagnostics: list[AnnotationDiagnostic] = []
    variant_line = _find_annotation_line(cell.source, "variant")

    # Malformed: a ``@variant`` line is present but parse_annotations rejected it.
    if variant_line is not None and annotations.variant is None:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="variant_malformed",
                message=(
                    "`@variant` requires `group` and `name`, both Python "
                    "identifiers: `# @variant <group> <name>`."
                ),
                line=variant_line,
            )
        )
        return diagnostics

    if annotations.variant is None:
        return diagnostics

    group_id = annotations.variant.group
    siblings = [
        c
        for c in notebook_state.cells
        if c.id != cell.id and c.variant_group == group_id and c.variant_name is not None
    ]

    # variant_contract_mismatch: siblings disagree on defines. Compare value defines only;
    # imports are a means, not an interface, and downstream never references them.
    if siblings:
        own_imports = _collect_top_level_imports(cell.source)
        own_defines = set(cell.defines) - own_imports
        for sibling in siblings:
            sibling_imports = _collect_top_level_imports(sibling.source)
            sibling_defines = set(sibling.defines) - sibling_imports
            missing = sibling_defines - own_defines
            extra = own_defines - sibling_defines
            if missing or extra:
                diff_parts = []
                if missing:
                    diff_parts.append(f"missing {sorted(missing)}")
                if extra:
                    diff_parts.append(f"extra {sorted(extra)}")
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="variant_contract_mismatch",
                        message=(
                            f"Variant `{annotations.variant.name}` defines a different "
                            f"set of names than sibling `{sibling.variant_name}` in "
                            f"group `{group_id}` ({', '.join(diff_parts)}). All variants "
                            "in a group must share the same defines contract."
                        ),
                        line=variant_line,
                    )
                )
                break

    # variant_mode_invalid: execution treats an unknown mode as switch; flag the surprise.
    mode = notebook_state.variant_modes.get(group_id, "switch")
    if mode not in ("switch", "sweep"):
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="variant_mode_invalid",
                message=(
                    f"notebook.toml sets mode `{mode}` for variant group `{group_id}`, "
                    "but only `switch` and `sweep` are valid. Treating it as `switch`."
                ),
                line=variant_line,
            )
        )

    # variant_active_redundant: `active` is ignored in sweep mode. Harmless, confusing.
    if mode == "sweep":
        active = notebook_state.variant_active_selections.get(group_id)
        if active:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.INFO,
                    code="variant_active_redundant",
                    message=(
                        f'Group `{group_id}` is in sweep mode, so `active = "{active}"` '
                        "is ignored: every variant runs and downstream consumes a "
                        "{variant: value} dict."
                    ),
                    line=variant_line,
                )
            )

    # variant_active_unknown: toml selects a missing variant. Surfaced on every member;
    # skipped in sweep mode, where the active pointer is ignored.
    selected = notebook_state.variant_active_selections.get(group_id) if mode != "sweep" else None
    # Truthy check: an empty ``active = ""`` means "first in source order", not unknown.
    if selected:
        all_members = [annotations.variant.name] + [s.variant_name for s in siblings]
        if selected not in all_members:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="variant_active_unknown",
                    message=(
                        f"notebook.toml selects variant `{selected}` for group "
                        f"`{group_id}`, but no cell in the group has that name. "
                        f"Falling back to the first variant in source order."
                    ),
                    line=variant_line,
                )
            )

    return diagnostics


def _sweep_groups_read_by(cell_id: str, dag: NotebookDag) -> dict[str, int]:
    """Return ``{group: member_count}`` for the sweep groups the DAG wired this cell to."""
    sizes = {res.group: len(res.members) for res in dag.variant_groups if res.mode == "sweep"}
    return {group: sizes[group] for group in dag.sweep_groups_read(cell_id)}


def _validate_unknown_annotations(cell: CellState) -> list[AnnotationDiagnostic]:
    """Warn on a ``# @key`` no parser reads: a typo like ``@wroker`` is otherwise ignored."""
    diagnostics: list[AnnotationDiagnostic] = []
    for lineno, line_text in iter_annotation_block(cell.source):
        parsed = parse_annotation_directive(line_text)
        if parsed is None or parsed[0] in ANNOTATION_KEYS:
            continue
        key = parsed[0]
        close = difflib.get_close_matches(key, ANNOTATION_KEYS, n=1, cutoff=0.75)
        hint = f" Did you mean `@{close[0]}`?" if close else ""
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="annotation_unknown",
                message=f"`@{key}` is not an annotation, so it is ignored.{hint}",
                line=lineno,
            )
        )
    return diagnostics


def _validate_per_variant_annotation(
    cell: CellState,
    annotations,
    dag: NotebookDag | None,
) -> list[AnnotationDiagnostic]:
    """Validate ``# @per_variant [group]`` fan-out membership.

    - ``per_variant_on_variant_member``: the cell is itself a variant.
    - ``per_variant_no_sweep_source``: it reads no sweep-sourced variable.
    - ``per_variant_ambiguous_group``: bare ``@per_variant`` but 2+ sweep groups.
    - ``per_variant_unknown_group``: the named group is not a sweep source it reads.
    - ``per_variant_group_of_one``: the group has one variant (info only).
    """
    if not annotations.per_variant:
        return []

    diagnostics: list[AnnotationDiagnostic] = []
    line = _find_annotation_line(cell.source, "per_variant")

    # Every variant writes the same table, so an overwrite keeps only the last one's rows.
    if any(spec.mode == "overwrite" for spec in annotations.table_writes):
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="per_variant_write_table_overwrite",
                message=(
                    "`@write_table ... overwrite` can't be combined with `@per_variant`: "
                    "each variant would replace the one before it. Append, merge on a "
                    "column that tells the variants apart, or overwrite from a cell that "
                    "reads the {variant: value} dict."
                ),
                line=_find_annotation_line(cell.source, "write_table", "overwrite"),
            )
        )

    # Mutually exclusive with @variant membership.
    if annotations.variant is not None:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="per_variant_on_variant_member",
                message=(
                    "`@per_variant` can't be combined with `@variant`: a cell "
                    "can't both be a variant and fan out over a sweep group."
                ),
                line=line,
            )
        )
        return diagnostics
    if dag is None:
        return diagnostics

    sweep_groups = _sweep_groups_read_by(cell.id, dag)
    named = annotations.per_variant_group

    if named is not None:
        if named not in sweep_groups:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="per_variant_unknown_group",
                    message=(
                        f"`@per_variant {named}` names a group this cell doesn't "
                        "read from in sweep mode. Reference a variable produced "
                        "by a sweep-mode variant group."
                    ),
                    line=line,
                )
            )
            return diagnostics
        effective_group = named
    else:
        if not sweep_groups:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="per_variant_no_sweep_source",
                    message=(
                        "`@per_variant` requires the cell to reference a variable "
                        "from a sweep-mode variant group; none were found."
                    ),
                    line=line,
                )
            )
            return diagnostics
        if len(sweep_groups) > 1:
            names = ", ".join(sorted(sweep_groups))
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="per_variant_ambiguous_group",
                    message=(
                        f"`@per_variant` is ambiguous: this cell reads from "
                        f"multiple sweep groups ({names}). Name one explicitly: "
                        "`# @per_variant <group>`."
                    ),
                    line=line,
                )
            )
            return diagnostics
        effective_group = next(iter(sweep_groups))

    if sweep_groups.get(effective_group) == 1:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.INFO,
                code="per_variant_group_of_one",
                message=(
                    f"Group `{effective_group}` has a single variant, so "
                    "`@per_variant` runs this cell once. The annotation is "
                    "harmless but unnecessary here."
                ),
                line=line,
            )
        )

    return diagnostics


def _collect_top_level_imports(source: str) -> set[str]:
    """Return module-scope names bound by ``import`` / ``from import`` statements."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _find_annotation_line(source: str, directive: str, needle: str | None = None) -> int | None:
    """Return 1-based line number of the first matching annotation."""
    for lineno, line_text in iter_annotation_block(source):
        lowered = line_text.strip().lstrip("#").strip().lower()
        if not lowered.startswith(f"@{directive.lower()}"):
            continue
        if needle is None or needle.lower() in lowered:
            return lineno
    return None

"""Cross-reference validation for cell source annotations.

Runs on notebook open, reload and after a WS source flush, never while typing, and
again before each run: an ERROR keeps the cell from running (``blocking_diagnostic``).
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING

from strata.notebook.annotations import (
    _LOOP_START_FROM_RE,
    iter_annotation_block,
    parse_annotations,
)
from strata.notebook.models import (
    AnnotationDiagnostic,
    CellLanguage,
    CellState,
    ConnectionSpec,
    DiagnosticSeverity,
    MalformedConnection,
    NotebookState,
)
from strata.notebook.workers import notebook_worker_definitions_editable, resolve_worker_spec

if TYPE_CHECKING:
    import cellgraph as cg


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
    analysis: cg.Analysis | None = None,
) -> AnnotationDiagnostic | None:
    """The first ERROR diagnostic, which keeps the cell from running, or ``None``."""
    return next(
        (
            d
            for d in validate_cell_annotations(cell, notebook_state, analysis)
            if d.severity == DiagnosticSeverity.ERROR
        ),
        None,
    )


def validate_cell_annotations(
    cell: CellState,
    notebook_state: NotebookState,
    analysis: cg.Analysis | None = None,
) -> list[AnnotationDiagnostic]:
    """Validate a cell's annotations against notebook-wide context.

    cellgraph's checks (``strata.notebook.vocabulary``), with Strata's own that read
    the server or the other cells' code. *analysis* is the notebook's, used when it
    read *cell* as it is.
    """
    from strata.notebook.graph import analysis_with
    from strata.notebook.vocabulary import strata_diagnostics

    if cell.language == CellLanguage.MARKDOWN:
        # Pure prose: ``# @worker`` is a heading, and only ``@app`` is read (unchecked).
        return []
    language_checks: list[AnnotationDiagnostic] = []
    if cell.language == CellLanguage.WIDGET:
        language_checks = _validate_widget_cell_annotations(cell)
    elif cell.language not in (CellLanguage.PROMPT, CellLanguage.SQL):
        language_checks = _validate_module_export(cell, notebook_state) + _validate_worker(
            cell, notebook_state
        )
    analysis = analysis_with(cell, notebook_state, analysis)
    diagnostics = strata_diagnostics(
        analysis.cells[cell.id], language_checks, without_dag=bool(analysis.structural_errors)
    )
    if cell.language == CellLanguage.SQL:
        # After the connection's, which the SQL analyzer reports.
        diagnostics += _validate_sql_tables(cell, notebook_state)
    return diagnostics


def _validate_worker(cell: CellState, notebook_state: NotebookState) -> list[AnnotationDiagnostic]:
    """``worker_unknown``: a worker execution cannot resolve."""
    annotations = parse_annotations(cell.source)
    diagnostics: list[AnnotationDiagnostic] = []
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

    return diagnostics


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


@functools.lru_cache(maxsize=4096)
def _export_blockers(source: str, cross_cell: frozenset[str]) -> tuple[tuple[str, ...], str]:
    """The defs, classes and names that keep *source* from exporting, and why.

    Memoized: validation runs on every source flush, mostly over unchanged cells.
    """
    from strata.notebook.module_export import build_module_export_plan, runtime_binding_names

    injectable = cross_cell | runtime_binding_names(source)
    plan = build_module_export_plan(source, injectable=injectable)
    if plan.is_exportable:
        return (), ""
    exported_code = [
        name
        for name, symbol in plan.exported_symbols.items()
        if symbol.kind in ("function", "async function", "class")
    ]
    blocked = tuple(sorted(set(exported_code) | plan.blocking_symbols))
    return blocked, plan.format_error() if blocked else ""


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
    # Upstream defines and this cell's runtime values can be hydrated into the synthetic
    # module, so a def closing over them isn't blocked. Mirrors
    # _write_module_export_outputs.
    others = [other for other in notebook_state.cells if other.id != cell.id]
    defined_elsewhere = {name for other in others for name in other.defines}
    cross_cell = frozenset(
        v for v in (*cell.references, *cell.builtin_references) if v in defined_elsewhere
    )
    blocked, error = _export_blockers(cell.source, cross_cell)
    if not blocked:
        return []

    referenced_elsewhere = {
        name for other in others for name in (*other.references, *other.builtin_references)
    }
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
                f"{error}."
            ),
            line=None,
        )
    ]


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


def sql_connection_diagnostics(
    source: str,
    connections: list[ConnectionSpec],
    malformed_connections: list[MalformedConnection],
) -> list[AnnotationDiagnostic]:
    """A SQL cell's ``@sql`` connection: missing, malformed, unknown, or its own problems."""
    diagnostics: list[AnnotationDiagnostic] = []
    sql = parse_annotations(source).sql
    if sql is None or not sql.connection:
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.ERROR,
                code="sql_connection_missing",
                message=(
                    "SQL cells require `# @sql connection=<name>`. The connection "
                    "must match a `[connections.<name>]` block in notebook.toml."
                ),
                line=_find_annotation_line(source, "sql"),
            )
        )
    else:
        valid_by_name = {c.name: c for c in connections}
        malformed_by_name = {m.name: m for m in malformed_connections}
        target = sql.connection
        sql_line = _sql_connection_line(source, target)
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

    return diagnostics


def _validate_sql_tables(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """SQL that does not parse, or names a table only when it runs."""
    diagnostics: list[AnnotationDiagnostic] = []
    sql = parse_annotations(cell.source).sql
    valid_by_name = {c.name: c for c in notebook_state.connections}

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


def _sql_connection_line(source: str, target: str) -> int | None:
    """The last ``@sql`` line naming ``target``: later lines win, so it is the one in force."""
    found = None
    for lineno, line_text in iter_annotation_block(source):
        tokens = line_text.strip().lstrip("#").split()
        if tokens and tokens[0].lower() == "@sql" and f"connection={target}" in tokens:
            found = lineno
    return found


def _find_annotation_line(source: str, directive: str, needle: str | None = None) -> int | None:
    """Return 1-based line number of the first matching annotation."""
    for lineno, line_text in iter_annotation_block(source):
        lowered = line_text.strip().lstrip("#").strip().lower()
        if not lowered.startswith(f"@{directive.lower()}"):
            continue
        if needle is None or needle.lower() in lowered:
            return lineno
    return None

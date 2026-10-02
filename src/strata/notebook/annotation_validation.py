"""Cross-reference validation for cell source annotations.

Runs on notebook open, reload and after a WS source flush, never while typing.
Diagnostics are advisory and never block execution.
"""

from __future__ import annotations

import ast
import logging

from strata.notebook.annotations import (
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

logger = logging.getLogger(__name__)

_BUILTIN_WORKER_NAMES = frozenset({"local"})
_SUPPORTED_MOUNT_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "az", "azure"})


def _table_uri_malformed(uri: str) -> bool:
    """True when the scan could not name the table, by the parse ``table_identity_for`` uses.

    The named-catalog split is skipped: a catalog name holds no dot, so the
    ``<namespace>.<table>`` check passes or fails the same either way.
    """
    from strata.iceberg import PyIcebergCatalog
    from strata.types import TableIdentity

    _warehouse, table_id = PyIcebergCatalog.parse_table_uri(uri)
    try:
        TableIdentity.from_table_id(table_id)
    except ValueError:
        return True
    return False


def validate_cell_annotations(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Validate a cell's annotations against notebook-wide context."""
    # Language-agnostic: any language can be a variant member, so these run before
    # language dispatch.
    _cell_annotations = parse_annotations(cell.source)
    variant_diagnostics = _validate_variant_annotation(cell, _cell_annotations, notebook_state)
    variant_diagnostics = variant_diagnostics + _validate_per_variant_annotation(
        cell, _cell_annotations, notebook_state
    )
    if cell.language == CellLanguage.MARKDOWN:
        # Pure prose: ``# @worker`` would be a markdown heading, not an annotation.
        return variant_diagnostics
    # A dropped ``@fetch`` or ``@dataset`` is a variable the cell will reach for and miss.
    variant_diagnostics = variant_diagnostics + _validate_recorded_inputs(cell)
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
    if annotations.worker:
        known = {w.name for w in notebook_state.workers} | _BUILTIN_WORKER_NAMES
        if annotations.worker not in known:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="worker_unknown",
                    message=(
                        f"`@worker {annotations.worker}` is not declared in this notebook. "
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
                        "cell also defines — the definition wins and the "
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
                    f"`@table {dup}` is declared more than once — each @table must "
                    "have a unique name; the cell will fail to run."
                ),
                line=_find_annotation_line(cell.source, "table", dup),
            )
        )

    # --- timeout_not_numeric / env_malformed ---
    # The parser silently swallows these, so re-scan raw lines.
    for lineno, line_text in iter_annotation_block(cell.source):
        parsed = parse_annotation_directive(line_text)
        if parsed is None:
            continue
        key, value = parsed

        if key == "timeout":
            if not value:
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="timeout_not_numeric",
                        message="`@timeout` requires a numeric value (seconds).",
                        line=lineno,
                    )
                )
            else:
                try:
                    t = float(value)
                    if t <= 0:
                        diagnostics.append(
                            AnnotationDiagnostic(
                                severity=DiagnosticSeverity.WARN,
                                code="timeout_not_numeric",
                                message=f"`@timeout {value}` must be a positive number.",
                                line=lineno,
                            )
                        )
                except ValueError:
                    diagnostics.append(
                        AnnotationDiagnostic(
                            severity=DiagnosticSeverity.WARN,
                            code="timeout_not_numeric",
                            message=f"`@timeout {value}` is not a valid number.",
                            line=lineno,
                        )
                    )

        elif key == "env":
            eq_idx = value.find("=")
            if eq_idx <= 0:
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="env_malformed",
                        message=(
                            f"`@env {value}` is malformed. Expected format: `@env KEY=value`."
                        ),
                        line=lineno,
                    )
                )

    diagnostics.extend(_validate_loop_annotation(cell, annotations, notebook_state))

    return diagnostics


def _validate_recorded_inputs(cell: CellState) -> list[AnnotationDiagnostic]:
    """Report a ``@fetch`` or ``@dataset`` line that did not parse.

    Both are dropped when unparseable, so the cell would otherwise die on a
    ``NameError`` with nothing pointing at the ignored line.
    """
    hints = {
        "fetch": "expected `@fetch <name> <url> [sha256=<hex>] [refetch=never|stale|always]`",
        "dataset": "expected `@dataset <var> <name>[@<alias>|@v=<n>]`",
    }
    return [
        AnnotationDiagnostic(
            severity=DiagnosticSeverity.WARN,
            code=f"{directive}_unreadable",
            message=(
                f"`@{directive} {value}` could not be read and was ignored, so the "
                f"variable it declares will not exist when the cell runs. "
                f"{hints[directive]}."
            ),
            line=lineno,
        )
        for lineno, directive, value in unreadable_input_directives(cell.source)
    ]


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
                severity=DiagnosticSeverity.WARN,
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

    # Re-scan raw lines for malformed @cache values the permissive parser dropped.
    for lineno, line_text in iter_annotation_block(cell.source):
        parsed = parse_annotation_directive(line_text)
        if parsed is None or parsed[0] != "cache":
            continue
        _, value = parsed
        if not value:
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="cache_policy_unknown",
                    message=(
                        "`@cache` requires a policy: fingerprint | forever | "
                        "session | snapshot | ttl=<seconds>."
                    ),
                    line=lineno,
                )
            )
            continue
        head = value.split()[0]
        if head in {"fingerprint", "forever", "session", "snapshot"}:
            continue
        if head.startswith("ttl="):
            raw = head.removeprefix("ttl=")
            try:
                if int(raw) > 0:
                    continue
            except ValueError:
                pass
            diagnostics.append(
                AnnotationDiagnostic(
                    severity=DiagnosticSeverity.WARN,
                    code="cache_ttl_invalid",
                    message=(f"`@cache ttl={raw}` requires a positive integer (seconds)."),
                    line=lineno,
                )
            )
            continue
        diagnostics.append(
            AnnotationDiagnostic(
                severity=DiagnosticSeverity.WARN,
                code="cache_policy_unknown",
                message=(
                    f"`@cache {head}` is not a recognized policy. Use "
                    "fingerprint | forever | session | snapshot | ttl=<seconds>."
                ),
                line=lineno,
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
                        "`@loop start_from` must reference a different cell — "
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
                severity=DiagnosticSeverity.WARN,
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
                        "is ignored — every variant runs and downstream consumes a "
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


def _sweep_groups_read_by(
    cell: CellState,
    notebook_state: NotebookState,
) -> dict[str, int]:
    """Return ``{group: member_count}`` for the sweep groups this cell reads from.

    Resolves producers from ``notebook_state`` alone, so validation does not need a
    rebuilt DAG.
    """
    refs = set(cell.references) | set(cell.builtin_references)
    groups: dict[str, int] = {}
    for group_id, mode in notebook_state.variant_modes.items():
        if mode != "sweep":
            continue
        members = [
            c
            for c in notebook_state.cells
            if c.variant_group == group_id and c.variant_name is not None
        ]
        if any(refs & set(member.defines) for member in members):
            groups[group_id] = len(members)
    return groups


def _validate_per_variant_annotation(
    cell: CellState,
    annotations,
    notebook_state: NotebookState,
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

    sweep_groups = _sweep_groups_read_by(cell, notebook_state)
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

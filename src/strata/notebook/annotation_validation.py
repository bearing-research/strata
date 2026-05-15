"""Cross-reference validation for cell source annotations.

Runs on notebook open, reload, and after WS source flush — never
during active typing.  Diagnostics are advisory and never block
execution.
"""

from __future__ import annotations

import ast

from strata.notebook.annotations import (
    iter_annotation_block,
    parse_annotation_directive,
    parse_annotations,
)
from strata.notebook.models import (
    AnnotationDiagnostic,
    CellLanguage,
    CellState,
    DiagnosticSeverity,
    NotebookState,
)

_BUILTIN_WORKER_NAMES = frozenset({"local"})
_SUPPORTED_MOUNT_SCHEMES = frozenset({"file", "s3", "gs", "gcs", "az", "azure"})


def validate_cell_annotations(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Validate a cell's annotations against notebook-wide context."""
    # Variant validation is language-agnostic: any cell language can be
    # a variant member, so the cross-sibling checks run before language
    # dispatch.
    variant_diagnostics = _validate_variant_annotation(
        cell, parse_annotations(cell.source), notebook_state
    )
    if cell.language == CellLanguage.PROMPT:
        return variant_diagnostics + _validate_prompt_cell_annotations(cell)
    if cell.language == CellLanguage.SQL:
        return variant_diagnostics + _validate_sql_cell_annotations(cell, notebook_state)
    if cell.language == CellLanguage.MARKDOWN:
        # Markdown cells are pure prose; ``# @worker`` etc. would be a
        # markdown heading, not an annotation. No validation applies.
        return variant_diagnostics
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

        # mount_uri_unsupported
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

        # mount_shadows_notebook
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

    # --- timeout_not_numeric / env_malformed ---
    # The parser silently swallows these, so we re-scan raw lines.
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


def _validate_module_export(
    cell: CellState,
    notebook_state: NotebookState,
) -> list[AnnotationDiagnostic]:
    """Warn when a Python cell defines reusable code (def / class) but
    the slice we'd re-execute in the synthetic module isn't safe.

    The slicer keeps imports, defs, classes, and literal constants and
    drops everything else. After slicing we still block when:
      - a kept def/class references a name that isn't imported or
        defined as a literal in this cell (the synthetic module would
        NameError on import or call);
      - a kept name is also rebound by dropped runtime code (the
        synthetic module's value would diverge from the cell's runtime
        value);
      - the cell has a lambda assignment to a downstream-consumed name
        (lambdas don't ride the source-backed module path).

    Suppressed when no other cell in the notebook references the
    affected names — small private helpers used only inside a single
    cell are a common, safe pattern. The warning is for users who are
    *trying* to share defs across cells and hit one of the failure
    modes above.
    """
    from strata.notebook.module_export import build_module_export_plan

    plan = build_module_export_plan(cell.source)
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

    # No downstream cell wants any of these names — silence the warning.
    referenced_elsewhere: set[str] = set()
    for other in notebook_state.cells:
        if other.id == cell.id:
            continue
        referenced_elsewhere.update(other.references)
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
    """Surface prompt-cell annotation errors (e.g. malformed ``@output_schema``).

    Called only for ``language == "prompt"`` cells. Python-cell validators
    (worker/mount/timeout/env/loop) don't apply here.
    """
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


def _validate_referenced_connection(
    conn,
    line: int | None,
) -> list[AnnotationDiagnostic]:
    """Emit diagnostics for a connection that a SQL cell references.

    Two checks:

    1. ``connection_driver_unknown`` — the chosen ``driver`` isn't in
       the SQL-adapter registry. The runtime would fail later with a
       harder-to-diagnose error.
    2. ``connection_auth_literal_secret`` — an ``auth.*`` value is a
       literal string instead of a ``${VAR}`` indirection. The writer
       will blank it on next save, breaking the connection silently
       unless the user is told.
    """
    diagnostics: list[AnnotationDiagnostic] = []

    # Driver registry check. Imported lazily so notebook code paths
    # that don't touch SQL don't pay the import cost, and so missing
    # optional ADBC packages don't crash validation.
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

    # Auth literal-secret check. Importing from writer keeps the
    # contract about what counts as a ${VAR} indirection in one place.
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
    """Surface SQL-cell annotation issues.

    Checks the cell's ``# @sql`` / ``# @cache`` directives, plus the
    referenced connection itself: malformed body, unknown driver, or
    auth values written as literals (which the writer will scrub on
    next save).
    """
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

    # SQL parse errors when a dialect can be resolved. The analyzer
    # captures sqlglot's syntax / token / optimize errors as
    # ``parse_error``; surfacing them as a diagnostic is what makes
    # them survive the session boundary — without this, parse failures
    # would be silently invisible to the UI.
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
            if sql_analysis.parse_error:
                diagnostics.append(
                    AnnotationDiagnostic(
                        severity=DiagnosticSeverity.WARN,
                        code="sql_parse_error",
                        message=(
                            f"sqlglot couldn't parse this SQL cell: "
                            f"{sql_analysis.parse_error}. The cell may "
                            "still execute, but cache invalidation by "
                            "touched-table fingerprint will fall back to "
                            "session-only."
                        ),
                        line=None,
                    )
                )
        except (KeyError, ImportError):
            # Driver not registered (handled by connection_driver_unknown
            # above) or sqlglot not installed — neither is a parse error.
            pass

    # Re-scan raw lines to catch malformed @cache values that the
    # permissive parser silently dropped.
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
    """Validate ``# @variant`` membership against siblings and toml.

    - ``variant_contract_mismatch`` — this variant's defines diverge from
      the union of its siblings (computed against active members so the
      diagnostic surfaces on the *outlier*, not on the rest).
    - ``variant_active_unknown`` — notebook.toml selects a variant name
      that no cell in the group provides.
    - ``variant_malformed`` — the ``@variant`` line is present but didn't
      parse into a (group, name) pair.
    """
    diagnostics: list[AnnotationDiagnostic] = []
    variant_line = _find_annotation_line(cell.source, "variant")

    # Detect the malformed case: a ``@variant`` line is present but
    # parse_annotations rejected it (wrong number of tokens, or non-
    # identifier values).
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

    # variant_contract_mismatch — siblings disagree on defines.
    # Compare *value defines* only: imports are scaffolding (a variant
    # using sklearn.linear_model vs sklearn.ensemble is a means, not an
    # interface). Downstream cells reference produced values, never the
    # variant's import names, so excluding imports gives the same
    # correctness guarantee with much less friction.
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

    # variant_active_unknown — toml selects a variant that doesn't exist
    # in this group. Surface on every member so the user sees it on
    # whichever variant they're looking at.
    selected = notebook_state.variant_active_selections.get(group_id)
    if selected is not None:
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

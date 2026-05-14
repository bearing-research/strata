"""Analyzer for SQL-type notebook cells.

Extracts the cell's metadata (``# @sql connection=``, ``# @cache``,
``# @name``), the SQL body, ``:name`` bind placeholders, and — when a
dialect is supplied — the set of tables the query touches via sqlglot.

Bind placeholders use ``:name`` syntax universally, regardless of the
backend dialect. Strata maps them to the correct ADBC binding form at
execute time (``?`` for SQLite, ``$1`` for Postgres, etc.). The
analyzer doesn't need to know the dialect to find them.

Table extraction does need a dialect — same SQL parses to different
trees in different dialects. When the connection isn't declared yet
(or the chosen driver isn't registered), table extraction is skipped
and ``parse_error`` stays None; the executor will pick this up later
with the resolved dialect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlglot.errors import SqlglotError as _SqlglotError

from strata.notebook.annotations import CachePolicy, parse_annotations, strip_leading_annotations
from strata.notebook.sql.adapter import QualifiedTable

# Bind-placeholder pattern. ``(?<![:\w])`` rules out ``::cast`` (Postgres
# type-cast operator) and identifiers like ``schema:foo`` where the
# colon is part of a larger token.
_BIND_PLACEHOLDER_RE = re.compile(r"(?<![:\w]):([a-zA-Z_]\w*)")


@dataclass
class SqlAnalysis:
    """Analysis result for a SQL cell.

    ``parse_error`` carries any sqlglot exception message; ``tables``
    is empty when ``parse_error`` is set. Both stay empty when no
    dialect was supplied (the analyzer can't pick the right grammar
    without one).

    Two placeholder views, populated together:

    - ``references`` — deduplicated, source-order. The DAG layer
      consumes this; one cell shouldn't claim the same upstream
      variable twice in its edges.
    - ``placeholder_positions`` — every ``:name`` occurrence in
      source order, duplicates included. The executor rewrites
      ``:name`` to the driver's positional syntax (``?`` for SQLite,
      ``$1`` / ``$2`` for Postgres) in this exact order, and the
      bind layer's tuple lines up position-for-position.
    """

    name: str = "result"
    defines: list[str] = field(default_factory=lambda: ["result"])
    references: list[str] = field(default_factory=list)
    placeholder_positions: list[str] = field(default_factory=list)
    connection: str | None = None
    cache_policy: CachePolicy = field(default_factory=lambda: CachePolicy(kind="fingerprint"))
    sql_body: str = ""
    tables: list[QualifiedTable] = field(default_factory=list)
    parse_error: str | None = None


def analyze_sql_cell(source: str, *, dialect: str | None = None) -> SqlAnalysis:
    """Analyze a SQL cell.

    Pipeline:

    1. Parse the leading comment block via the shared
       ``parse_annotations`` helper. Pulls out ``# @sql``, ``# @cache``,
       and any ``# @name`` override.
    2. Strip annotations from the source to leave the SQL body.
    3. Find ``:name`` placeholders in the body — dialect-independent
       since this is Strata's binding surface, not the backend's. The
       extractor strips strings and comments first so ``'foo :bar'``
       and ``-- :bar`` don't false-match.
    4. If ``dialect`` is provided, parse the body with sqlglot and
       walk ``find_all_in_scope(parsed, exp.Table)`` — the scope-aware
       walker that excludes CTE / derived-table references (the naive
       ``find_all`` would surface those as if they were base tables).
    """
    annotations = parse_annotations(source)
    sql_body = strip_leading_annotations(source).strip()

    output_name = annotations.name if annotations.name else "result"
    if not output_name.isidentifier():
        output_name = "result"

    positions = _extract_placeholder_positions(sql_body)
    references = _dedupe_preserve_order(positions)

    cache_policy = annotations.cache or CachePolicy(kind="fingerprint")
    connection = annotations.sql.connection if annotations.sql else None

    tables: list[QualifiedTable] = []
    parse_error: str | None = None
    if dialect is not None and sql_body:
        try:
            tables = _extract_tables(sql_body, dialect)
        except _SqlglotError as exc:
            # User-authored SQL syntax / token / optimize errors. Caller
            # surfaces this as a ``sql_parse_error`` diagnostic. Other
            # exception classes (TypeError, AttributeError, etc.) are
            # analyzer bugs and propagate normally — masking them as
            # parse errors would hide real regressions.
            parse_error = str(exc)

    return SqlAnalysis(
        name=output_name,
        defines=[output_name],
        references=references,
        placeholder_positions=positions,
        connection=connection,
        cache_policy=cache_policy,
        sql_body=sql_body,
        tables=tables,
        parse_error=parse_error,
    )


def _extract_placeholder_positions(sql: str) -> list[str]:
    """Return ``:name`` placeholders in source order, duplicates kept.

    Strings and comments are blanked before the regex runs so embedded
    ``:foo`` inside ``'literal :foo'`` or ``-- :foo`` doesn't surface
    as a bind reference. Duplicates are preserved here because the
    executor rewrites ``:name`` to positional binds (``?`` for SQLite,
    ``$1`` for Postgres) in this exact order; the bind layer must
    produce one tuple slot per occurrence to keep positions aligned.

    Use ``_extract_placeholders`` (or ``SqlAnalysis.references``)
    when you want the deduplicated DAG-facing view.
    """
    cleaned = _blank_strings_and_comments(sql)
    return [m.group(1) for m in _BIND_PLACEHOLDER_RE.finditer(cleaned)]


def _extract_placeholders(sql: str) -> list[str]:
    """Return ``:name`` placeholders in source order, deduplicated.

    Thin wrapper over ``_extract_placeholder_positions`` for callers
    that only need the DAG-facing (deduplicated) view. The cell's DAG
    references list shouldn't carry duplicates — one cell can't claim
    the same upstream variable twice.
    """
    return _dedupe_preserve_order(_extract_placeholder_positions(sql))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _blank_strings_and_comments(sql: str) -> str:
    """Replace string literals and comments with spaces in-place.

    Length-preserving so byte offsets in error messages stay aligned
    with the original source. Recognizes:

    - ``'string'`` with ``''`` doubled-quote escaping
    - ``-- line comment`` (to end of line)
    - ``/* block comment */``
    - ``$tag$ ... $tag$`` Postgres dollar-quoted strings (including
      empty-tag ``$$ ... $$``)

    Dollar-quoting matters for placeholder extraction because the body
    is treated literally — including any ``:name`` patterns. Without
    blanking, ``SELECT $$:foo$$ AS x`` produces a bogus ``foo``
    reference. ``$1`` / ``$2`` (Postgres positional bind syntax) is
    NOT a dollar quote and falls through unchanged: my recognizer
    only triggers when ``$`` is followed by either another ``$`` or an
    identifier-start character followed by a closing ``$``.

    Doesn't try to parse double-quoted identifiers or
    backtick-quoted MySQL identifiers — those are dialect-specific
    edge cases. False positives there surface as bogus placeholder
    references that the executor rejects with a clear "no upstream
    variable :foo" error.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        c = sql[i]

        # ``-- line comment``
        if c == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                out.append(" ")
                i += 1
            continue

        # ``/* block comment */``
        if c == "/" and i + 1 < n and sql[i + 1] == "*":
            out.append(" ")
            out.append(" ")
            i += 2
            while i < n - 1 and not (sql[i] == "*" and sql[i + 1] == "/"):
                out.append(" ")
                i += 1
            if i < n - 1:
                out.append(" ")
                out.append(" ")
                i += 2
            else:
                # unterminated; consume the rest as blanks
                while i < n:
                    out.append(" ")
                    i += 1
            continue

        # ``'string'``
        if c == "'":
            out.append(" ")
            i += 1
            while i < n:
                if sql[i] == "'":
                    if i + 1 < n and sql[i + 1] == "'":
                        out.append(" ")
                        out.append(" ")
                        i += 2
                        continue
                    out.append(" ")
                    i += 1
                    break
                out.append(" ")
                i += 1
            continue

        # ``$tag$...$tag$`` dollar-quoted Postgres string. Allow
        # empty tags (``$$``) and tags matching ``[a-zA-Z_][a-zA-Z0-9_]*``.
        # ``$1`` / ``$2`` positional binds fall through here because
        # the char after ``$`` isn't ``$`` or an identifier-start.
        if c == "$":
            tag_end = _scan_dollar_quote_open(sql, i)
            if tag_end is not None:
                opening = sql[i : tag_end + 1]
                end_idx = sql.find(opening, tag_end + 1)
                if end_idx == -1:
                    # unterminated — consume the rest as blanks
                    while i < n:
                        out.append(" ")
                        i += 1
                else:
                    stop = end_idx + len(opening)
                    while i < stop:
                        out.append(" ")
                        i += 1
                continue

        out.append(c)
        i += 1

    return "".join(out)


def rewrite_named_to_positional(sql: str, dialect: str | None) -> str:
    """Rewrite ``:name`` placeholders to the dialect's positional form.

    ADBC's parameter-binding API takes positional values, and each
    backend uses its own placeholder syntax — Postgres expects
    ``$1`` / ``$2`` / ..., SQLite expects ``?``, and other ADBC
    drivers default to the qmark form. Strata's surface is uniform
    ``:name``; this function bridges to the wire form just before
    we hand the SQL to the cursor.

    The rewriter uses ``_blank_strings_and_comments`` to find
    placeholder positions on a string-and-comment-blanked copy, so
    a ``:foo`` inside ``'literal :foo'`` or ``-- :foo`` survives
    untouched in the output. Position-by-position substitution then
    emits the positional form into the *original* text — keeping
    comments and string contents byte-exact for any backend that
    cares (Postgres' query log, SQLite's prepared-statement cache).

    The caller is responsible for keeping the bind tuple in
    ``placeholder_positions`` order (the analyzer's
    duplicate-preserving view), so position N in the rewritten SQL
    matches position N in the bind tuple.
    """
    if dialect == "postgres":

        def emit(i: int) -> str:
            return f"${i + 1}"
    else:
        # SQLite ADBC + most others use qmark.

        def emit(i: int) -> str:
            return "?"

    cleaned = _blank_strings_and_comments(sql)
    out: list[str] = []
    last = 0
    for i, match in enumerate(_BIND_PLACEHOLDER_RE.finditer(cleaned)):
        start, end = match.start(), match.end()
        out.append(sql[last:start])
        out.append(emit(i))
        last = end
    out.append(sql[last:])
    return "".join(out)


def _scan_dollar_quote_open(sql: str, start: int) -> int | None:
    """If ``sql[start:]`` opens a dollar-quote, return the index of
    its closing ``$`` (so ``sql[start : ret + 1]`` is the full
    ``$tag$`` opening). Otherwise return None.

    Treats ``$$`` (empty tag) and ``$<ident>$`` (tag matches
    ``[a-zA-Z_][a-zA-Z0-9_]*``) as opens. Anything else — ``$1``,
    ``$ word``, end-of-string — returns None so the caller emits the
    ``$`` verbatim.
    """
    n = len(sql)
    if start >= n or sql[start] != "$":
        return None
    j = start + 1
    if j < n and sql[j] == "$":
        return j  # $$ — empty tag
    # $tag$ — tag must start with letter/underscore.
    if j < n and (sql[j].isalpha() or sql[j] == "_"):
        while j < n and (sql[j].isalnum() or sql[j] == "_"):
            j += 1
        if j < n and sql[j] == "$":
            return j
    return None


def _extract_tables(sql: str, dialect: str) -> list[QualifiedTable]:
    """Walk parsed SQL for base-table references, deduplicated and ordered.

    Uses ``sqlglot.optimizer.scope.traverse_scope`` to visit every
    scope (root, CTEs, derived tables, subqueries) and collects
    ``exp.Table`` nodes whose ``scope.sources[name]`` is an
    ``exp.Table`` — the source-of-truth signal for "this is a base
    table reference, not a binding to another scope."

    Filtering by ``scope.sources`` correctly drops CTE references
    in the outer scope (where ``WITH foo AS (...) SELECT * FROM foo``
    binds ``foo`` to a Scope, not a Table) while still surfacing the
    base tables inside the CTE body.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.scope import Scope, traverse_scope

    parsed_list = sqlglot.parse(sql, dialect=dialect)
    seen: set[tuple[str | None, str | None, str]] = set()
    out: list[QualifiedTable] = []
    for parsed in parsed_list:
        if parsed is None:
            continue
        for scope in traverse_scope(parsed):
            for table_node in scope.find_all(exp.Table):
                source = scope.sources.get(table_node.alias_or_name)
                if isinstance(source, Scope):
                    # Reference to a CTE / derived table, not a
                    # base table.
                    continue
                qt = QualifiedTable(
                    catalog=table_node.catalog or None,
                    schema=table_node.db or None,
                    name=table_node.name,
                )
                key = (qt.catalog, qt.schema, qt.name)
                if key in seen:
                    continue
                seen.add(key)
                out.append(qt)
    return out

"""Analyzer for SQL-type notebook cells.

Extracts the cell's directives, SQL body, ``:name`` bind placeholders and, given
a dialect, the tables the query reads (via sqlglot). Placeholders are ``:name``
on every backend and found without a dialect; table extraction needs one, and is
skipped (``parse_error`` stays None) until the connection's dialect is known.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sqlglot.errors import SqlglotError as _SqlglotError

from strata.notebook.annotations import CachePolicy, parse_annotations, strip_leading_annotations
from strata.notebook.sql.adapter import QualifiedTable

# ``(?<![:\w])`` rules out ``::cast`` and tokens like ``schema:foo``.
_BIND_PLACEHOLDER_RE = re.compile(r"(?<![:\w]):([a-zA-Z_]\w*)")
# Dialects whose ordinary ``'...'`` strings take backslash escapes (``'it\'s'``).
_BACKSLASH_STRING_DIALECTS = frozenset({"bigquery", "mysql", "snowflake"})


@dataclass
class SqlAnalysis:
    """Analysis result for a SQL cell.

    ``tables`` is empty when ``parse_error`` is set, and both stay empty without a
    dialect. ``unresolved_tables`` holds the SQL text of references named only at
    run time (Snowflake ``IDENTIFIER($var)``), kept out of ``tables`` because a
    freshness probe cannot ask about them. ``references`` is the deduplicated
    placeholder list for the DAG; ``placeholder_positions`` keeps every occurrence in
    source order, which the executor's positional binds must match.
    """

    name: str = "result"
    defines: list[str] = field(default_factory=lambda: ["result"])
    references: list[str] = field(default_factory=list)
    placeholder_positions: list[str] = field(default_factory=list)
    connection: str | None = None
    cache_policy: CachePolicy = field(default_factory=lambda: CachePolicy(kind="fingerprint"))
    sql_body: str = ""
    tables: list[QualifiedTable] = field(default_factory=list)
    unresolved_tables: list[str] = field(default_factory=list)
    parse_error: str | None = None


def analyze_sql_cell(source: str, *, dialect: str | None = None) -> SqlAnalysis:
    """Analyze a SQL cell.

    Placeholders are found after blanking strings and comments, so ``'foo :bar'``
    and ``-- :bar`` do not match. Tables come from a scope-aware walk that excludes
    CTE and derived-table references.
    """
    annotations = parse_annotations(source)
    sql_body = strip_leading_annotations(source).strip()

    output_name = annotations.name if annotations.name else "result"
    if not output_name.isidentifier():
        output_name = "result"

    positions = _extract_placeholder_positions(sql_body, dialect)
    references = _dedupe_preserve_order(positions)

    cache_policy = annotations.cache or CachePolicy(kind="fingerprint")
    connection = annotations.sql.connection if annotations.sql else None

    tables: list[QualifiedTable] = []
    unresolved_tables: list[str] = []
    parse_error: str | None = None
    if dialect is not None and sql_body:
        try:
            try:
                tables, unresolved_tables = _extract_tables(sql_body, dialect)
            except _SqlglotError:
                # sqlglot reads DuckDB ``SELECT :x`` as an alias; the text that runs
                # has positional binds instead.
                tables, unresolved_tables = _extract_tables(
                    rewrite_named_to_positional(sql_body, dialect), dialect
                )
        except _SqlglotError as exc:
            # User SQL errors only; anything else is an analyzer bug and
            # must propagate.
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
        unresolved_tables=unresolved_tables,
        parse_error=parse_error,
    )


# What a read cell may run; the rest (DDL, DML, COPY, ATTACH, transaction
# statements that would end the driver's read-only one) needs a write cell.
# Names, not classes, so sqlglot is imported only for SQL cells.
_READ_STATEMENT_NAMES = (
    "Select",
    "SetOperation",
    "Union",
    "Except",
    "Intersect",
    "Subquery",
    "Describe",
    "Show",
    "Values",
    "Summarize",
    "Pivot",
    "Unpivot",
)
# sqlglot parses ungrammared statements as ``Command`` (EXPLAIN, some SHOW).
# Only these are read-only; ATTACH, CALL, INSTALL stay refused.
_READ_COMMANDS = ("EXPLAIN", "SHOW", "DESC", "DESCRIBE")
# EXPLAIN ANALYZE runs the wrapped statement, so it needs that statement's
# privilege. PRAGMA is always refused: sqlglot can't tell reporting from setting.
# Strip comments first, or ``EXPLAIN /*x*/ ANALYZE`` slips a write through.
_COMMENTS = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
# ``ANALYZE`` leading the argument or anywhere in a leading option list
# (``EXPLAIN (FORMAT JSON, ANALYZE)``), but not elsewhere in the query.
_ANALYZE = re.compile(
    r"^\s*(?:\(\s*[^)]*\banaly[sz]e\b|\(?\s*analy[sz]e\b)",
    re.IGNORECASE,
)


def read_only_violation(sql: str, dialect: str | None) -> str | None:
    """The first statement in *sql* a read cell may not run, or None.

    The connection's read-only transaction is one the body can end
    (``COMMIT; ATTACH '/db' AS w (READ_WRITE); ...``), so the statements are checked
    before anything reaches the driver.
    """
    if dialect is None or not sql.strip():
        return None
    import sqlglot

    try:
        statements = [statement for statement in sqlglot.parse(sql, dialect=dialect) if statement]
    except _SqlglotError:
        # The caller surfaces the parse error; nothing runs.
        return None
    for statement in statements:
        name = type(statement).__name__
        if name in _READ_STATEMENT_NAMES:
            continue
        if name == "Alias" and type(statement.this).__name__ in ("Select", "Table", "Column"):
            # ``TABLE t``.
            continue
        if name == "Command":
            head = str(statement.this or "").upper()
            argument = statement.args.get("expression")
            argument_text = _COMMENTS.sub(" ", str(getattr(argument, "this", argument) or ""))
            if head in _READ_COMMANDS and not (head == "EXPLAIN" and _ANALYZE.match(argument_text)):
                continue
            if head == "EXPLAIN":
                return (
                    "a SQL cell reads, and EXPLAIN ANALYZE runs the statement it describes. "
                    "Use `# @sql connection=<name> write=true` for a cell that changes a "
                    "database, or EXPLAIN without ANALYZE."
                )
        kind = str(statement.this or "").upper() if name == "Command" else name.upper()
        return (
            f"a SQL cell reads, and {kind} is not a read. Use "
            "`# @sql connection=<name> write=true` for a cell that changes a database."
        )
    return None


def confined_write_violation(sql: str, dialect: str | None) -> str | None:
    """The first statement in *sql* a confined write cell may not run, or None.

    In service mode a SQLite cell runs in the server process, where ``ATTACH`` and
    ``VACUUM INTO`` reach other files; ADBC SQLite has no engine-level confinement
    (DuckDB does, ``duckdb._confine``). SQL that does not parse is refused too.
    """
    if dialect is None or not sql.strip():
        return None
    import sqlglot

    try:
        statements = [statement for statement in sqlglot.parse(sql, dialect=dialect) if statement]
    except _SqlglotError:
        return (
            "the SQL could not be parsed, and on this server a SQL cell runs only SQL it can check"
        )
    for statement in statements:
        name = type(statement).__name__
        head = str(statement.this or "").upper() if name == "Command" else name.upper()
        if head in ("ATTACH", "DETACH", "VACUUM"):
            return (
                f"{head} reaches files outside the connection's database, which this "
                "server does not allow"
            )
    return None


def _extract_placeholder_positions(sql: str, dialect: str | None = None) -> list[str]:
    """Return ``:name`` placeholders in source order, duplicates kept.

    Strings and comments are blanked first. Duplicates stay because the executor
    rewrites each occurrence to a positional bind in this order.
    """
    cleaned = _blank_strings_and_comments(sql, dialect)
    return [m.group(1) for m in _BIND_PLACEHOLDER_RE.finditer(cleaned)]


def _extract_placeholders(sql: str, dialect: str | None = None) -> list[str]:
    """Return ``:name`` placeholders in source order, deduplicated, for the DAG."""
    return _dedupe_preserve_order(_extract_placeholder_positions(sql, dialect))


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _blank_strings_and_comments(sql: str, dialect: str | None = None) -> str:
    """Replace string literals and comments with spaces, preserving length.

    Recognizes ``'...'`` (``''`` escapes, and backslash escapes in BigQuery,
    MySQL and Snowflake), Postgres ``E'...'`` (backslash escapes too), ``-- ...``,
    ``/* ... */`` and Postgres ``$tag$ ... $tag$`` / ``$$ ... $$``; ``$1`` is not a
    dollar quote. Double-quoted and backtick identifiers are not handled; a false
    placeholder there is rejected by the executor as an unknown upstream variable.
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
            # ``E'...'`` and ``e'...'``, but not the tail of an identifier like ``name'``.
            escape_string = dialect in _BACKSLASH_STRING_DIALECTS or (
                i > 0
                and sql[i - 1] in "eE"
                and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] in "_$"))
            )
            out.append(" ")
            i += 1
            while i < n:
                if escape_string and sql[i] == "\\" and i + 1 < n:
                    out.append(" ")
                    out.append(" ")
                    i += 2
                    continue
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

        # ``$tag$...$tag$`` dollar quote. ``$1`` positional binds fall
        # through: no ``$`` or identifier-start follows.
        if c == "$":
            tag_end = _scan_dollar_quote_open(sql, i)
            if tag_end is not None:
                opening = sql[i : tag_end + 1]
                end_idx = sql.find(opening, tag_end + 1)
                if end_idx == -1:
                    # unterminated; consume the rest as blanks
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

    ``$1``, ``$2``, ... for Postgres; ``?`` for SQLite and other ADBC drivers.
    Positions come from a blanked copy, so placeholders in strings and comments are
    left alone, and the original text is otherwise byte-exact. The caller binds
    values in ``placeholder_positions`` order.
    """
    if dialect == "postgres":

        def emit(i: int) -> str:
            return f"${i + 1}"
    else:
        # SQLite ADBC + most others use qmark.

        def emit(i: int) -> str:
            return "?"

    cleaned = _blank_strings_and_comments(sql, dialect)
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
    """Return the index of the closing ``$`` if ``sql[start:]`` opens a dollar-quote.

    ``$$`` and ``$<ident>$`` open one; anything else (``$1``, ``$ word``, end of
    string) returns None so the caller emits the ``$`` verbatim.
    """
    n = len(sql)
    if start >= n or sql[start] != "$":
        return None
    j = start + 1
    if j < n and sql[j] == "$":
        return j  # $$: empty tag
    # $tag$: tag must start with a letter or underscore.
    if j < n and (sql[j].isalpha() or sql[j] == "_"):
        while j < n and (sql[j].isalnum() or sql[j] == "_"):
            j += 1
        if j < n and sql[j] == "$":
            return j
    return None


def base_table_nodes(tree: Any, dialect: str) -> list[Any]:
    """The ``exp.Table`` nodes in *tree* that are base tables, scope by scope.

    Same rule as ``_extract_tables``, returning nodes so a caller can rewrite them.
    """
    from sqlglot import exp
    from sqlglot.optimizer.scope import Scope, traverse_scope

    nodes: list[Any] = []
    for scope in traverse_scope(tree):
        for node in scope.find_all(exp.Table):
            source = scope.sources.get(node.alias_or_name)
            if isinstance(source, Scope):
                continue
            nodes.append(node)
    return nodes


def _stored_name(identifier: Any, dialect: str) -> str | None:
    """*identifier*'s name as the database stores it, or ``None`` when absent.

    Snowflake uppercases unquoted identifiers and compares exactly in
    ``INFORMATION_SCHEMA`` (``events`` is ``EVENTS``, ``"events"`` stays);
    Postgres folds to lowercase.
    """
    from sqlglot import exp

    if not isinstance(identifier, exp.Identifier):
        return identifier or None  # absent: ``None``, or ``""`` for ``db..t``
    if not identifier.name:
        return None
    if identifier.args.get("quoted"):
        return identifier.name
    if dialect == "snowflake":
        return identifier.name.upper()
    if dialect == "postgres":
        return identifier.name.lower()
    return identifier.name


def _qualified(table_node: Any, dialect: str) -> QualifiedTable | None:
    """The table *table_node* names, or ``None`` when a part of it is not a name.

    ``$tbl``, ``$sch.events``, ``getvariable('db').events`` and a table function
    such as ``read_parquet(...)`` all sit where a name goes, and each is
    resolved only when the query runs.
    """
    from sqlglot import exp

    this = table_node.this
    if not isinstance(this, exp.Identifier) or not this.name:
        return None
    for qualifier in (table_node.args.get("catalog"), table_node.args.get("db")):
        if not (qualifier is None or isinstance(qualifier, (str, exp.Identifier))):
            return None
    return QualifiedTable(
        catalog=_stored_name(table_node.args.get("catalog"), dialect),
        schema=_stored_name(table_node.args.get("db"), dialect),
        name=_stored_name(this, dialect) or "",
    )


def _named_by_literal(node: Any, dialect: str) -> QualifiedTable | None:
    """The table a string literal names, as ``IDENTIFIER('db.sch.t')`` or
    ``TABLE('t')`` do, or ``None`` for anything that is not a string literal."""
    from sqlglot import exp

    if isinstance(node, exp.DynamicIdentifier):
        node = node.this
    if not (isinstance(node, exp.Literal) and node.is_string and node.name):
        return None
    return _qualified(exp.to_table(node.name, dialect=dialect), dialect)


def _table_reference(table_node: Any, sql: str, dialect: str) -> QualifiedTable | None:
    """The table *table_node* names, or ``None`` when no probe can name it.

    ``None`` for anything resolved only at run time: ``IDENTIFIER($var)`` or
    ``IDENTIFIER(?)`` (a string literal inside is a static name), a session variable
    (``$tbl``), and table functions (``read_parquet(...)``, set-returning
    functions). Also for DuckDB file reads (``FROM 'events.parquet'``, or a name
    that looks like a file, quoted or not): nothing fingerprints the file, so a
    mount on a lake connection is the way to read files the cache can see. BigQuery
    wildcard tables and ``INFORMATION_SCHEMA`` views are names its probe cannot find.
    """
    from sqlglot import exp

    this = table_node.this
    if isinstance(this, exp.DynamicIdentifier):
        return _named_by_literal(this, dialect)
    if not isinstance(this, exp.Identifier):
        return None
    start = this.meta.get("start")
    if start is not None and sql[start] == "'":
        return None
    name = ".".join(part.name for part in table_node.parts)
    if dialect == "duckdb" and _names_a_file(name):
        return None
    if dialect == "bigquery" and ("*" in name or "INFORMATION_SCHEMA" in name.upper().split(".")):
        return None
    return _qualified(table_node, dialect)


# Names DuckDB replacement-scans as files in table position.
_FILE_NAME_RE = re.compile(
    r"\.(parquet|csv|tsv|json|jsonl|ndjson|arrow|orc)(\.(gz|zst|zstd|bz2|xz|lz4))?$",
    re.IGNORECASE,
)


def _names_a_file(name: str) -> bool:
    """Whether DuckDB reads *name*, a table reference's dotted text, as a file."""
    return "/" in name or "*" in name or _FILE_NAME_RE.search(name) is not None


def _unresolved_text(table_node: Any, sql: str, dialect: str) -> str:
    """How a reference no probe can name reads in the cell, for a diagnostic."""
    from sqlglot import exp

    this = table_node.this
    if isinstance(this, exp.Identifier):
        start, end = this.meta.get("start"), this.meta.get("end")
        if start is not None and end is not None and sql[start] == "'":
            return sql[start : end + 1]
    return ".".join(part.sql(dialect=dialect) for part in table_node.parts)


def _extract_tables(sql: str, dialect: str) -> tuple[list[QualifiedTable], list[str]]:
    """Walk parsed SQL for base-table references, deduplicated and ordered.

    Keeps ``exp.Table`` nodes whose ``scope.sources`` entry is a table, so a CTE
    name in an outer scope is dropped while the base tables in its body surface.
    Also returns the SQL text of references named only at run time.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.scope import Scope, traverse_scope

    parsed_list = sqlglot.parse(sql, dialect=dialect)
    seen: set[tuple[str | None, str | None, str]] = set()
    out: list[QualifiedTable] = []
    unresolved: list[str] = []

    def record(qt: QualifiedTable | None, text: str) -> None:
        if qt is None:
            if text not in unresolved:
                unresolved.append(text)
            return
        key = (qt.catalog, qt.schema, qt.name)
        if key not in seen:
            seen.add(key)
            out.append(qt)

    for parsed in parsed_list:
        if parsed is None:
            continue
        for scope in traverse_scope(parsed):
            for table_node in scope.find_all(exp.Table):
                source = scope.sources.get(table_node.alias_or_name)
                if isinstance(source, Scope):
                    # A CTE or derived table, not a base table.
                    continue
                if table_node.args.get("rows_from"):
                    # ``ROWS FROM (f(), g())`` names nothing itself; each
                    # function in it is a table node of its own, walked here.
                    continue
                qt = _table_reference(table_node, sql, dialect)
                record(qt, "" if qt else _unresolved_text(table_node, sql, dialect))
            # Snowflake ``TABLE(...)`` is not an ``exp.Table``. A string literal
            # names a table; ``$var`` or a function resolves only at run time.
            for rows_node in scope.find_all(exp.TableFromRows):
                bare = rows_node.copy()
                bare.set("alias", None)
                record(_named_by_literal(rows_node.this, dialect), bare.sql(dialect=dialect))
    return out, unresolved

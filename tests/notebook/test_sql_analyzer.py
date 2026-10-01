"""Tests for the SQL cell analyzer."""

from __future__ import annotations

import pytest

from strata.notebook.annotations import strip_leading_annotations as _strip_leading_annotations
from strata.notebook.sql.adapter import QualifiedTable
from strata.notebook.sql.analyzer import (
    SqlAnalysis,
    _blank_strings_and_comments,
    _extract_placeholder_positions,
    _extract_placeholders,
    analyze_sql_cell,
)

# --- annotations & body extraction ----------------------------------------


def test_strip_leading_annotations_returns_only_sql():
    src = "# @sql connection=warehouse\n# @cache forever\n\nSELECT 1\n"
    assert _strip_leading_annotations(src).strip() == "SELECT 1"


def test_strip_leading_annotations_handles_no_annotations():
    src = "SELECT 1\nFROM t"
    assert _strip_leading_annotations(src).strip() == "SELECT 1\nFROM t"


def test_strip_leading_annotations_blank_when_only_comments():
    src = "# @sql connection=db\n# @cache forever"
    assert _strip_leading_annotations(src) == ""


def test_analyze_extracts_connection_and_cache_policy():
    src = "# @sql connection=warehouse\n# @cache forever\nSELECT * FROM events\n"
    result = analyze_sql_cell(src)
    assert result.connection == "warehouse"
    assert result.cache_policy.kind == "forever"


def test_analyze_default_cache_policy_is_fingerprint():
    """No `# @cache` → fingerprint default. The provenance layer
    folds this into the hash so users get correct invalidation
    without opting in."""
    src = "# @sql connection=db\nSELECT 1"
    assert analyze_sql_cell(src).cache_policy.kind == "fingerprint"


def test_analyze_name_annotation_overrides_default():
    src = "# @sql connection=db\n# @name events_count\nSELECT 42 AS n"
    assert analyze_sql_cell(src).name == "events_count"
    assert analyze_sql_cell(src).defines == ["events_count"]


def test_analyze_invalid_name_falls_back_to_result():
    """Non-identifier names (with spaces, hyphens, leading digits)
    fall back to ``result`` rather than producing an unusable
    output variable name."""
    src = "# @sql connection=db\n# @name 123-not-ok\nSELECT 1"
    assert analyze_sql_cell(src).name == "result"


def test_analyze_default_name_is_result():
    src = "SELECT 1"
    assert analyze_sql_cell(src).name == "result"
    assert analyze_sql_cell(src).defines == ["result"]


# --- bind placeholder extraction -----------------------------------------


def test_placeholders_simple_named_refs():
    sql = "SELECT * FROM users WHERE id = :user_id AND tenant = :tenant_id"
    refs = _extract_placeholders(sql)
    assert refs == ["user_id", "tenant_id"]


def test_placeholders_dedupe_repeated_names():
    """The DAG references list shouldn't carry duplicates."""
    sql = "SELECT :foo + :foo AS doubled, :bar AS single"
    assert _extract_placeholders(sql) == ["foo", "bar"]


def test_placeholder_positions_preserve_duplicates():
    """Codex review fix: the deduped ``references`` list is right
    for the DAG, but the executor needs every ``:name`` occurrence
    in source order to rewrite ``:foo + :foo`` into the driver's
    positional binds (``? + ?`` for SQLite, ``$1 + $2`` for
    Postgres). ``_extract_placeholder_positions`` is the
    duplicate-preserving counterpart."""
    sql = "SELECT :foo + :foo AS doubled, :bar AS single"
    assert _extract_placeholder_positions(sql) == ["foo", "foo", "bar"]


def test_analyze_exposes_both_references_and_positions():
    """End-to-end: ``analyze_sql_cell`` populates the deduped
    ``references`` field and the duplicate-preserving
    ``placeholder_positions`` field together so DAG and executor
    consumers each get the view they need."""
    src = "# @sql connection=db\nSELECT :foo + :foo + :bar"
    result = analyze_sql_cell(src)
    assert result.references == ["foo", "bar"]
    assert result.placeholder_positions == ["foo", "foo", "bar"]


def test_placeholders_preserve_first_appearance_order():
    sql = "SELECT * FROM t WHERE a = :first AND b = :second AND c = :first"
    assert _extract_placeholders(sql) == ["first", "second"]


def test_placeholders_skip_postgres_cast_operator():
    """``::cast`` is Postgres' type-cast operator. The leading colon
    is part of an existing token and shouldn't trigger a placeholder
    match."""
    sql = "SELECT id::int, value::text FROM t WHERE x = :real_param"
    assert _extract_placeholders(sql) == ["real_param"]


def test_placeholders_skip_strings():
    """``:foo`` inside a string literal is data, not a binding."""
    sql = "SELECT 'literal :foo' AS s, :real AS x FROM t"
    assert _extract_placeholders(sql) == ["real"]


def test_placeholders_skip_escaped_single_quotes_in_strings():
    """``'a''b :foo'`` is one string with an escaped quote — the
    ``:foo`` is still inside it and shouldn't surface."""
    sql = "SELECT 'a''b :foo c' FROM t WHERE x = :real"
    assert _extract_placeholders(sql) == ["real"]


def test_placeholders_skip_line_comments():
    sql = "SELECT 1 -- :ignored\nWHERE x = :real"
    assert _extract_placeholders(sql) == ["real"]


def test_placeholders_skip_block_comments():
    sql = "SELECT 1 /* :ignored multi\nline */ WHERE x = :real"
    assert _extract_placeholders(sql) == ["real"]


def test_placeholders_handle_unterminated_block_comment_gracefully():
    """An unterminated ``/*`` shouldn't crash; the rest of the
    source becomes blanks and any earlier placeholders stay
    visible."""
    sql = "SELECT :real FROM t /* unterminated :ignored"
    refs = _extract_placeholders(sql)
    assert "real" in refs
    assert "ignored" not in refs


def test_blank_strings_preserves_length():
    """Length-preserving so byte offsets in error messages stay
    aligned with the original source."""
    sql = "SELECT 'hello :x' FROM t"
    cleaned = _blank_strings_and_comments(sql)
    assert len(cleaned) == len(sql)


def test_placeholders_skip_dollar_quoted_strings_empty_tag():
    """Codex review fix: ``$$ ... $$`` is a Postgres dollar-quoted
    string. The body is literal — including ``:foo`` — and must
    not surface as a bind reference."""
    sql = "SELECT $$:foo$$ AS x, :real AS y FROM t"
    refs = _extract_placeholders(sql)
    assert refs == ["real"]


def test_placeholders_skip_dollar_quoted_strings_with_tag():
    """``$body$ ... $body$`` is also dollar-quoting — the named tag
    just lets the body itself contain ``$`` characters."""
    sql = "SELECT $body$:ignored and $$ inside$body$ AS s, :real FROM t"
    refs = _extract_placeholders(sql)
    assert refs == ["real"]


def test_placeholders_handle_unterminated_dollar_quote():
    """An unterminated ``$$`` shouldn't crash; the rest of the source
    becomes blanks. Earlier placeholders stay visible."""
    sql = "SELECT :real FROM t WHERE x = $$unterminated :ignored"
    refs = _extract_placeholders(sql)
    assert "real" in refs
    assert "ignored" not in refs


def test_placeholders_do_not_treat_positional_dollar_as_quote():
    """``$1`` / ``$2`` are Postgres positional-bind syntax, not
    dollar-quote opens. They fall through and any ``:name``
    placeholders elsewhere still surface."""
    sql = "SELECT $1, $2, :real FROM t"
    refs = _extract_placeholders(sql)
    assert refs == ["real"]


def test_blank_dollar_quote_preserves_length():
    sql = "SELECT $$:foo$$ FROM t"
    cleaned = _blank_strings_and_comments(sql)
    assert len(cleaned) == len(sql)


# --- analyze_sql_cell wiring ---------------------------------------------


def test_analyze_no_dialect_skips_table_extraction():
    """Without a dialect we can't pick the right grammar. Skip table
    extraction; bind placeholders still work via the dialect-
    independent regex path."""
    src = "# @sql connection=db\nSELECT * FROM events WHERE id = :user_id"
    result = analyze_sql_cell(src)
    assert result.tables == []
    assert result.parse_error is None
    assert result.references == ["user_id"]


def test_analyze_with_dialect_extracts_simple_table():
    src = "# @sql connection=db\nSELECT * FROM events"
    result = analyze_sql_cell(src, dialect="postgres")
    assert result.tables == [QualifiedTable(catalog=None, schema=None, name="events")]


def test_analyze_with_dialect_extracts_qualified_tables():
    src = (
        "# @sql connection=db\n"
        "SELECT * FROM analytics.events e "
        "JOIN public.users u ON u.id = e.user_id"
    )
    result = analyze_sql_cell(src, dialect="postgres")
    by_name = {t.name: t for t in result.tables}
    assert "events" in by_name
    assert by_name["events"].schema == "analytics"
    assert "users" in by_name
    assert by_name["users"].schema == "public"


def test_analyze_with_dialect_filters_cte_references():
    """A SQL parser walking ``find_all(exp.Table)`` would surface
    CTE references as if they were base tables. The scope-aware
    walker drops them — verify here by writing a CTE alias and
    checking it doesn't leak into ``tables``."""
    src = (
        "# @sql connection=db\n"
        "WITH summary AS (SELECT user_id, COUNT(*) FROM events GROUP BY user_id)\n"
        "SELECT * FROM summary"
    )
    result = analyze_sql_cell(src, dialect="postgres")
    names = {t.name for t in result.tables}
    assert "events" in names
    assert "summary" not in names, (
        "CTE name leaked as a base-table reference — analyzer must "
        "use find_all_in_scope, not find_all"
    )


def test_analyze_with_dialect_dedupes_table_references():
    src = (
        "# @sql connection=db\n"
        "SELECT * FROM events WHERE EXISTS (SELECT 1 FROM events WHERE id < 10)"
    )
    result = analyze_sql_cell(src, dialect="postgres")
    names = [t.name for t in result.tables]
    assert names.count("events") == 1


def test_identifier_with_a_literal_names_its_table():
    """Snowflake's ``IDENTIFIER('...')`` with a string is a static name,
    qualified or not, so the analyzer can fingerprint the table. Snowflake
    reads the string as it reads an identifier, so unquoted parts are the
    uppercased names it stores."""
    src = "# @sql connection=db\nSELECT * FROM IDENTIFIER('events')"
    result = analyze_sql_cell(src, dialect="snowflake")
    assert result.tables == [QualifiedTable(catalog=None, schema=None, name="EVENTS")]
    assert result.unresolved_tables == []

    src = "# @sql connection=db\nSELECT * FROM IDENTIFIER('db.sch.events')"
    result = analyze_sql_cell(src, dialect="snowflake")
    assert result.tables == [QualifiedTable(catalog="DB", schema="SCH", name="EVENTS")]


@pytest.mark.parametrize("reference", ["IDENTIFIER($tbl)", "IDENTIFIER(:tbl)", "IDENTIFIER(?)"])
def test_identifier_named_at_run_time_is_reported_not_guessed(reference):
    """A session variable or a bind parameter names the table only when the
    query runs. sqlglot 30.13+ reads ``$tbl`` as a table called ``tbl``;
    fingerprinting that would track a table the query never reads."""
    src = f"# @sql connection=db\nSELECT * FROM {reference} JOIN orders USING (id)"
    result = analyze_sql_cell(src, dialect="snowflake")
    assert result.tables == [QualifiedTable(catalog=None, schema=None, name="ORDERS")]
    assert result.unresolved_tables == [reference]


@pytest.mark.parametrize(
    ("dialect", "reference"),
    [
        ("snowflake", "TABLE($tbl)"),
        ("snowflake", "TABLE(IDENTIFIER($tbl))"),
        ("snowflake", "TABLE(MY_UDTF(1))"),
        ("snowflake", "$tbl"),
        ("snowflake", "$sch.events"),
        ("duckdb", "QUERY_TABLE(GETVARIABLE('t'))"),
        ("duckdb", "READ_PARQUET('events.parquet')"),
        ("duckdb", "'events.parquet'"),
        ("duckdb", '"events.parquet"'),
        ("duckdb", "events.parquet"),
        ("duckdb", "events.csv.gz"),
        ("duckdb", '"data/events.json"'),
        ("duckdb", '"data/*.parquet"'),
        ("duckdb", '"s3://bucket/events"'),
        ("postgres", "MY_FUNC()"),
    ],
)
def test_a_table_no_probe_can_name_is_reported_not_tracked(dialect, reference):
    """A table function, a session variable or a file path sits where a table
    name goes, and a freshness probe cannot ask about any of them. Each used to
    come back as a table with an empty or invented name, or as nothing at all,
    and the cell was then served from its cache however the data changed."""
    src = f"# @sql connection=db\nSELECT * FROM {reference} AS x JOIN orders USING (id)"
    result = analyze_sql_cell(src, dialect=dialect)
    assert [t.name.lower() for t in result.tables] == ["orders"]
    assert result.unresolved_tables == [reference]


@pytest.mark.parametrize(
    ("reference", "unresolved"),
    [
        ("`proj.ds.events_*`", "`proj`.`ds`.`events_*`"),
        ("proj.ds.INFORMATION_SCHEMA.TABLES", "proj.ds.`INFORMATION_SCHEMA.TABLES`"),
        ("`region-us`.INFORMATION_SCHEMA.JOBS", "`region-us`.`INFORMATION_SCHEMA.JOBS`"),
    ],
)
def test_a_bigquery_wildcard_or_metadata_view_is_reported_not_tracked(reference, unresolved):
    """The BigQuery probe looks each table up in its dataset's ``__TABLES__``,
    which has no row for a wildcard table or an ``INFORMATION_SCHEMA`` view.
    "Missing" is the same answer on every run, so the cell was served from its
    cache however the data changed, and a region-qualified view failed the
    probe outright."""
    src = f"# @sql connection=db\nSELECT * FROM {reference} AS x JOIN ds.orders USING (id)"
    result = analyze_sql_cell(src, dialect="bigquery")
    assert result.tables == [QualifiedTable(catalog=None, schema="ds", name="orders")]
    assert result.unresolved_tables == [unresolved]


def test_a_file_looking_name_is_a_table_outside_duckdb():
    """Only DuckDB reads a file where a table name goes. Elsewhere
    ``events.parquet`` is the table ``parquet`` in the schema ``events``."""
    src = "# @sql connection=db\nSELECT * FROM events.parquet"
    result = analyze_sql_cell(src, dialect="postgres")
    assert result.tables == [QualifiedTable(catalog=None, schema="events", name="parquet")]
    assert result.unresolved_tables == []


@pytest.mark.parametrize(
    ("dialect", "reference", "unresolved"),
    [
        ("postgres", "ROWS FROM (generate_series(1, 3))", ["GENERATE_SERIES(1, 3)"]),
        (
            "postgres",
            "ROWS FROM (generate_series(1, 3), my_func())",
            ["GENERATE_SERIES(1, 3)", "MY_FUNC()"],
        ),
        ("duckdb", "ROWS FROM (range(3))", ["RANGE(0, 3)"]),
        ("postgres", "json_to_recordset('[]')", ["JSON_TO_RECORDSET('[]')"]),
    ],
)
def test_a_set_returning_function_list_is_reported_not_a_crash(dialect, reference, unresolved):
    """``ROWS FROM (...)`` is a table with no name of its own, only the
    functions it calls. The analyzer read the name it does not have and raised,
    and a notebook holding such a cell did not open."""
    src = f"# @sql connection=db\nSELECT * FROM {reference} AS x(a) JOIN orders USING (id)"
    result = analyze_sql_cell(src, dialect=dialect)
    assert result.parse_error is None
    assert [t.name for t in result.tables] == ["orders"]
    assert result.unresolved_tables == unresolved


def test_a_notebook_with_a_rows_from_cell_opens(tmp_path):
    """The server session and the CLI's ops both analyze every cell on open."""
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb_dir = create_notebook(tmp_path, "rows_from")
    add_cell_to_notebook(nb_dir, "sql", language="sql")
    write_cell(
        nb_dir,
        "sql",
        "# @sql connection=pg\n# @name q\nSELECT * FROM ROWS FROM (generate_series(1, 3))\n",
    )
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.pg]\ndriver = "postgresql"\nhost = "localhost"\n'
    )

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    cell = session.notebook_state.get_cell("sql")
    assert cell.defines == ["q"]
    assert [d.code for d in cell.annotation_diagnostics] == ["sql_dynamic_table"]

    ops = LocalNotebookOps(nb_dir)
    assert [c.id for c in ops.list_cells()] == ["sql"]


def test_an_analyzer_failure_does_not_stop_a_notebook_opening(tmp_path, monkeypatch):
    """A bug in table extraction is the analyzer's, not the notebook's: the
    notebook opens, the cell keeps its defines, and its header says what went
    wrong."""
    import strata.notebook.sql.analyzer as analyzer_mod
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    def boom(_sql, _dialect):
        raise RuntimeError("analyzer bug")

    monkeypatch.setattr(analyzer_mod, "_extract_tables", boom)
    nb_dir = create_notebook(tmp_path, "analyzer_bug")
    add_cell_to_notebook(nb_dir, "sql", language="sql")
    write_cell(nb_dir, "sql", "# @sql connection=pg\n# @name q\nSELECT * FROM events\n")
    toml = nb_dir / "notebook.toml"
    toml.write_text(
        toml.read_text() + '\n[connections.pg]\ndriver = "postgresql"\nhost = "localhost"\n'
    )

    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    cell = session.notebook_state.get_cell("sql")
    assert cell.defines == ["q"]
    messages = {d.code: d.message for d in cell.annotation_diagnostics}
    assert "analyzer bug" in messages["sql_analysis_failed"]


def test_table_with_a_literal_names_its_table():
    """Snowflake's ``TABLE('...')`` with a string names a table, as
    ``IDENTIFIER('...')`` does."""
    src = "# @sql connection=db\nSELECT * FROM TABLE('db.sch.events')"
    result = analyze_sql_cell(src, dialect="snowflake")
    assert result.tables == [QualifiedTable(catalog="DB", schema="SCH", name="EVENTS")]
    assert result.unresolved_tables == []


def test_snowflake_names_are_the_ones_it_stores():
    """Snowflake stores an unquoted identifier uppercased and Postgres
    lowercased; a quoted one is stored as written. Other dialects keep the name
    as typed."""
    src = '# @sql connection=db\nSELECT * FROM mydb.public.events JOIN "MixedCase" USING (id)'
    result = analyze_sql_cell(src, dialect="snowflake")
    assert result.tables == [
        QualifiedTable(catalog="MYDB", schema="PUBLIC", name="EVENTS"),
        QualifiedTable(catalog=None, schema=None, name="MixedCase"),
    ]

    src = '# @sql connection=db\nSELECT * FROM MyDb.Public.Events JOIN "MixedCase" USING (id)'
    result = analyze_sql_cell(src, dialect="postgres")
    assert result.tables == [
        QualifiedTable(catalog="mydb", schema="public", name="events"),
        QualifiedTable(catalog=None, schema=None, name="MixedCase"),
    ]

    for dialect in ("duckdb", "bigquery", "sqlite"):
        result = analyze_sql_cell(src, dialect=dialect)
        assert [t.name for t in result.tables] == ["Events", "MixedCase"], dialect


# --- result type ----------------------------------------------------------


def test_returns_sqlanalysis_dataclass():
    result = analyze_sql_cell("SELECT 1")
    assert isinstance(result, SqlAnalysis)
    assert result.defines == ["result"]
    assert result.references == []
    assert result.connection is None


def test_internal_errors_are_not_swallowed_as_parse_errors(monkeypatch):
    """Codex review fix: only sqlglot-class errors get re-labeled as
    ``parse_error``. Internal bugs (TypeError, AttributeError,
    import failures from a sibling module) propagate unchanged so
    real regressions don't masquerade as user-authored SQL syntax
    errors."""
    import strata.notebook.sql.analyzer as analyzer_mod

    def boom(_sql, _dialect):
        raise TypeError("not a sqlglot error — internal bug")

    monkeypatch.setattr(analyzer_mod, "_extract_tables", boom)

    src = "# @sql connection=db\nSELECT 1"
    import pytest

    with pytest.raises(TypeError, match="internal bug"):
        analyze_sql_cell(src, dialect="postgres")


def test_genuine_sqlglot_parse_errors_become_parse_error_field():
    """Sanity check on the narrowed catch: real sqlglot parse errors
    still land in ``parse_error`` and produce empty tables — no
    propagation."""
    src = "# @sql connection=db\nSELECT * FROM"  # truncated
    result = analyze_sql_cell(src, dialect="postgres")
    assert result.parse_error is not None
    assert result.tables == []

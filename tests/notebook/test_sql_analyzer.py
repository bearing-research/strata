"""Tests for the SQL cell analyzer."""

from __future__ import annotations

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


def test_analyze_dynamic_sql_does_not_resolve():
    """``IDENTIFIER('events')`` is Snowflake's dynamic-name form. A
    static parser can't resolve it, so the analyzer surfaces no
    table — the executor's freshness probe will fall back to
    session-only for queries it can't fingerprint."""
    src = "# @sql connection=db\nSELECT * FROM IDENTIFIER('events')"
    result = analyze_sql_cell(src, dialect="snowflake")
    # The IDENTIFIER call doesn't surface as an exp.Table reference,
    # so tables stays empty.
    assert all(t.name != "events" for t in result.tables)


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

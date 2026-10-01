"""Tests for `# @sql` and `# @cache` annotations and their validation."""

from __future__ import annotations

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.annotations import CachePolicy, parse_annotations
from strata.notebook.models import CellState, ConnectionSpec, NotebookState

# --- annotation parsing ---------------------------------------------------


def test_parses_sql_connection():
    src = "# @sql connection=warehouse\nSELECT 1"
    a = parse_annotations(src)
    assert a.sql is not None
    assert a.sql.connection == "warehouse"


def test_sql_without_connection_keyword_yields_none_connection():
    """``@sql`` without ``connection=`` parses to a None connection so validation can flag it."""
    src = "# @sql warehouse\nSELECT 1"
    a = parse_annotations(src)
    assert a.sql is not None
    assert a.sql.connection is None


def test_sql_later_lines_override_earlier():
    src = "# @sql connection=foo\n# @sql connection=bar\nSELECT 1"
    a = parse_annotations(src)
    assert a.sql is not None
    assert a.sql.connection == "bar"


def test_no_sql_annotation_yields_none():
    a = parse_annotations("SELECT 1")
    assert a.sql is None


def test_parses_cache_policies():
    cases = [
        ("# @cache fingerprint\nSELECT 1", CachePolicy(kind="fingerprint")),
        ("# @cache forever\nSELECT 1", CachePolicy(kind="forever")),
        ("# @cache session\nSELECT 1", CachePolicy(kind="session")),
        ("# @cache snapshot\nSELECT 1", CachePolicy(kind="snapshot")),
        ("# @cache ttl=300\nSELECT 1", CachePolicy(kind="ttl", ttl_seconds=300)),
    ]
    for src, expected in cases:
        got = parse_annotations(src).cache
        assert got == expected, f"{src!r} → {got!r}, expected {expected!r}"


def test_invalid_cache_policy_drops_to_none():
    """Unknown policies and malformed ttl parse to None; validation reports them."""
    cases = [
        "# @cache\nSELECT 1",
        "# @cache bogus\nSELECT 1",
        "# @cache ttl=abc\nSELECT 1",
        "# @cache ttl=-5\nSELECT 1",
        "# @cache ttl=0\nSELECT 1",
    ]
    for src in cases:
        assert parse_annotations(src).cache is None, src


# --- validation diagnostics -----------------------------------------------


def _sql_cell(source: str) -> CellState:
    return CellState(id="c1", source=source, language="sql")


def _state_with(connections: list[ConnectionSpec]) -> NotebookState:
    return NotebookState(id="nb1", connections=connections)


def test_validation_flags_missing_sql_annotation():
    cell = _sql_cell("SELECT 1")
    diags = validate_cell_annotations(cell, _state_with([]))
    codes = [d.code for d in diags]
    assert "sql_connection_missing" in codes
    err = next(d for d in diags if d.code == "sql_connection_missing")
    assert err.severity == "error"


def test_validation_flags_sql_without_connection_keyword():
    cell = _sql_cell("# @sql warehouse\nSELECT 1")
    diags = validate_cell_annotations(cell, _state_with([]))
    assert any(d.code == "sql_connection_missing" for d in diags)


def test_validation_flags_unknown_connection():
    cell = _sql_cell("# @sql connection=warehouse\nSELECT 1")
    state = _state_with([ConnectionSpec(name="other", driver="sqlite")])
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "sql_connection_unknown" in codes
    assert "sql_connection_missing" not in codes


def test_validation_clean_when_connection_declared():
    cell = _sql_cell("# @sql connection=warehouse\nSELECT 1")
    state = _state_with([ConnectionSpec(name="warehouse", driver="postgresql")])
    diags = validate_cell_annotations(cell, state)
    assert diags == []


def test_validation_flags_unknown_cache_policy():
    cell = _sql_cell(
        "# @sql connection=db\n# @cache bogus\nSELECT 1",
    )
    state = _state_with([ConnectionSpec(name="db", driver="sqlite")])
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "cache_policy_unknown" in codes


def test_validation_flags_invalid_ttl():
    cell = _sql_cell(
        "# @sql connection=db\n# @cache ttl=abc\nSELECT 1",
    )
    state = _state_with([ConnectionSpec(name="db", driver="sqlite")])
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "cache_ttl_invalid" in codes


def test_validation_clean_for_valid_cache_policies():
    # A registered driver keeps connection_driver_unknown out of this assertion, which
    # is about cache_policy_unknown / cache_ttl_invalid.
    state = _state_with([ConnectionSpec(name="db", driver="postgresql")])
    for policy in ("fingerprint", "forever", "session", "snapshot", "ttl=600"):
        cell = _sql_cell(f"# @sql connection=db\n# @cache {policy}\nSELECT 1")
        diags = validate_cell_annotations(cell, state)
        assert diags == [], f"{policy}: {diags}"


# --- connection-level diagnostics ---


def test_validation_flags_malformed_connection():
    """A declared but malformed connection gets ``connection_malformed``, not ``unknown``."""
    from strata.notebook.models import MalformedConnection

    cell = _sql_cell("# @sql connection=warehouse\nSELECT 1")
    state = NotebookState(
        id="nb1",
        connections=[],
        malformed_connections=[
            MalformedConnection(
                name="warehouse",
                body={"host": "localhost"},
                error="connection is missing required 'driver' key",
            )
        ],
    )
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "connection_malformed" in codes
    assert "sql_connection_unknown" not in codes
    err = next(d for d in diags if d.code == "connection_malformed")
    assert err.severity == "error"
    assert "driver" in err.message.lower()


def test_validation_flags_unknown_driver():
    """A driver missing from the adapter registry is flagged at validation, not at run time."""
    from strata.notebook.sql import AdapterCapabilities, FreshnessToken, SchemaFingerprint
    from strata.notebook.sql.registry import (
        _reset_for_tests,
        _restore_defaults_for_tests,
        register_adapter,
    )

    class _Stub:
        name = "postgresql"
        sqlglot_dialect = "postgres"
        capabilities = AdapterCapabilities(
            per_table_freshness=True,
            supports_snapshot=False,
            needs_separate_probe_conn=True,
        )

        def canonicalize_connection_id(self, spec):
            return ""

        def open(self, spec, *, read_only):
            return None

        def probe_freshness(self, probe_conn, tables):
            return FreshnessToken(value=b"")

        def probe_schema(self, probe_conn, tables):
            return SchemaFingerprint(value=b"")

    _reset_for_tests()
    register_adapter(_Stub())
    try:
        cell = _sql_cell("# @sql connection=warehouse\nSELECT 1")
        state = _state_with([ConnectionSpec(name="warehouse", driver="snowflakez")])
        diags = validate_cell_annotations(cell, state)
        codes = [d.code for d in diags]
        assert "connection_driver_unknown" in codes
        err = next(d for d in diags if d.code == "connection_driver_unknown")
        assert "snowflakez" in err.message
        assert "postgresql" in err.message  # known-driver hint
    finally:
        _restore_defaults_for_tests()


def test_validation_skips_driver_check_when_registry_empty():
    """With no adapters registered (ADBC extras absent), driver checks would be pure noise."""
    from strata.notebook.sql.registry import _reset_for_tests, _restore_defaults_for_tests

    _reset_for_tests()
    try:
        cell = _sql_cell("# @sql connection=warehouse\nSELECT 1")
        state = _state_with([ConnectionSpec(name="warehouse", driver="madeup")])
        diags = validate_cell_annotations(cell, state)
        assert all(d.code != "connection_driver_unknown" for d in diags)
    finally:
        _restore_defaults_for_tests()


def test_validation_flags_literal_auth_values():
    """``auth.password = "hunter2"`` is flagged: the writer blanks literal secrets on save.

    Without a diagnostic the connection would break after an unrelated rewrite.
    """
    cell = _sql_cell("# @sql connection=db\nSELECT 1")
    state = _state_with(
        [
            ConnectionSpec(
                name="db",
                driver="postgresql",
                auth={
                    "user": "${PGUSER}",
                    "password": "hunter2",  # literal
                    "extra_secret": "literal_too",
                },
            )
        ]
    )
    diags = validate_cell_annotations(cell, state)
    literal_diags = [d for d in diags if d.code == "connection_auth_literal_secret"]
    # One per literal; the ${VAR} indirection is silent.
    assert len(literal_diags) == 2
    keys_flagged = {("password" in d.message, "extra_secret" in d.message) for d in literal_diags}
    assert keys_flagged == {(True, False), (False, True)}


def test_validation_clean_when_all_auth_uses_indirection():
    cell = _sql_cell("# @sql connection=db\nSELECT 1")
    state = _state_with(
        [
            ConnectionSpec(
                name="db",
                driver="postgresql",
                auth={"user": "${PGUSER}", "password": "${PGPASS}"},
            )
        ]
    )
    diags = validate_cell_annotations(cell, state)
    assert all(d.code != "connection_auth_literal_secret" for d in diags)


# --- SQL parse errors surface as diagnostics ---


def test_validation_surfaces_sql_parse_error_diagnostic():
    """A SQL syntax error reaches the cell as a ``sql_parse_error`` diagnostic.

    Otherwise only ``defines`` / ``references`` cross the session boundary and the
    parse failure is invisible in the UI.
    """
    cell = _sql_cell("# @sql connection=db\nSELECT * FROM")  # truncated
    state = _state_with([ConnectionSpec(name="db", driver="postgresql")])
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "sql_parse_error" in codes
    err = next(d for d in diags if d.code == "sql_parse_error")
    assert err.severity == "warn"
    # Body should mention what went wrong, not just say "error".
    assert "parse" in err.message.lower() or "sql" in err.message.lower()


def test_validation_no_sql_parse_error_for_valid_sql():
    cell = _sql_cell("# @sql connection=db\nSELECT * FROM events")
    state = _state_with([ConnectionSpec(name="db", driver="postgresql")])
    diags = validate_cell_annotations(cell, state)
    assert all(d.code != "sql_parse_error" for d in diags)


def test_validation_skips_sql_parse_check_when_connection_unknown():
    """No resolved connection means no dialect, so no ``sql_parse_error`` is reported."""
    cell = _sql_cell("# @sql connection=missing\nSELECT * FROM")
    state = _state_with([])
    diags = validate_cell_annotations(cell, state)
    codes = [d.code for d in diags]
    assert "sql_connection_unknown" in codes
    assert "sql_parse_error" not in codes

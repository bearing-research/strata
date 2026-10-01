"""Tests for the DriverAdapter protocol surface and registry plumbing."""

from __future__ import annotations

import pytest

from strata.notebook.sql import (
    AdapterCapabilities,
    FreshnessToken,
    QualifiedTable,
    SchemaFingerprint,
    get_adapter,
    hash_connection_identity,
    known_drivers,
    register_adapter,
)
from strata.notebook.sql.registry import _reset_for_tests, _restore_defaults_for_tests

# --- QualifiedTable -----------------------------------------------------


def test_qualified_table_renders_dotted():
    assert QualifiedTable(catalog="db", schema="public", name="t").render() == ("db.public.t")


def test_qualified_table_drops_missing_layers():
    assert QualifiedTable(catalog=None, schema=None, name="t").render() == "t"
    assert QualifiedTable(catalog=None, schema="public", name="t").render() == ("public.t")


# --- hash_connection_identity ------------------------------------------


def test_identity_hash_stable_across_key_order():
    a = hash_connection_identity("postgresql", {"host": "localhost", "port": 5432})
    b = hash_connection_identity("postgresql", {"port": 5432, "host": "localhost"})
    assert a == b


def test_identity_hash_changes_on_driver_change():
    a = hash_connection_identity("postgresql", {"host": "h"})
    b = hash_connection_identity("sqlite", {"host": "h"})
    assert a != b


def test_identity_hash_changes_on_value_change():
    a = hash_connection_identity("postgresql", {"role": "reader"})
    b = hash_connection_identity("postgresql", {"role": "admin"})
    assert a != b


# --- FreshnessToken / SchemaFingerprint -------------------------------


def test_freshness_token_equality_by_value():
    a = FreshnessToken(value=b"abc")
    b = FreshnessToken(value=b"abc")
    c = FreshnessToken(value=b"xyz")
    assert a == b
    assert a != c


def test_freshness_token_distinguishes_session_only():
    a = FreshnessToken(value=b"abc")
    b = FreshnessToken(value=b"abc", is_session_only=True)
    assert a != b


def test_freshness_token_distinguishes_snapshot():
    a = FreshnessToken(value=b"abc")
    b = FreshnessToken(value=b"abc", is_snapshot=True)
    assert a != b


def test_schema_fingerprint_equality_by_value():
    assert SchemaFingerprint(value=b"x") == SchemaFingerprint(value=b"x")
    assert SchemaFingerprint(value=b"x") != SchemaFingerprint(value=b"y")


# --- registry ----------------------------------------------------------


@pytest.fixture
def clean_registry():
    """Empty the registry; restore the default adapters at teardown."""
    _reset_for_tests()
    yield
    _restore_defaults_for_tests()


class _StubAdapter:
    name = "stub"
    sqlglot_dialect = ""
    capabilities = AdapterCapabilities(
        per_table_freshness=True,
        supports_snapshot=False,
        needs_separate_probe_conn=False,
    )

    def canonicalize_connection_id(self, spec):
        return "x"

    def open(self, spec, *, read_only):
        return None

    def probe_freshness(self, probe_conn, tables):
        return FreshnessToken(value=b"")

    def probe_schema(self, probe_conn, tables):
        return SchemaFingerprint(value=b"")


def test_register_and_lookup(clean_registry):
    register_adapter(_StubAdapter())
    assert known_drivers() == ["stub"]
    assert get_adapter("stub").name == "stub"


def test_get_adapter_unknown_raises_with_known_list(clean_registry):
    register_adapter(_StubAdapter())
    with pytest.raises(KeyError) as exc_info:
        get_adapter("nope")
    assert "nope" in str(exc_info.value)
    assert "stub" in str(exc_info.value)


def test_register_replaces_previous(clean_registry):
    """Re-registration is intentional: tests swap adapters through it."""

    class _Replacement(_StubAdapter):
        name = "stub"

        def canonicalize_connection_id(self, spec):
            return "y"

    register_adapter(_StubAdapter())
    register_adapter(_Replacement())
    assert get_adapter("stub").canonicalize_connection_id(None) == "y"


def test_known_drivers_sorted(clean_registry):
    class _A(_StubAdapter):
        name = "alpha"

    class _Z(_StubAdapter):
        name = "zulu"

    class _M(_StubAdapter):
        name = "mike"

    register_adapter(_Z())
    register_adapter(_A())
    register_adapter(_M())
    assert known_drivers() == ["alpha", "mike", "zulu"]

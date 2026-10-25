"""Ledger-side contract fixtures: output keys, table ownership, provenance classes (ADR 002 E0)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest
from vestigo.build_store import BuildStore
from vestigo.provenance import derive_subkey
from vestigo.store import ArtifactStore
from vestigo.tables import LEDGER_TABLES, TRANSIENT_TABLES

from strata.migrate import MIGRATED_TABLES

FIXTURES = Path(__file__).parent / "fixtures"
OUTPUT_KEYS = json.loads((FIXTURES / "output_keys.json").read_text(encoding="utf-8"))
TABLES = json.loads((FIXTURES / "tables.json").read_text(encoding="utf-8"))
CLASSES = json.loads((FIXTURES / "provenance_classes.json").read_text(encoding="utf-8"))
ADR001 = json.loads((FIXTURES / "adr001_tests.json").read_text(encoding="utf-8"))


def _derive(vector: dict) -> str:
    key = OUTPUT_KEYS["parent"]
    for labels in vector.get("nested") or [vector["labels"]]:
        key = derive_subkey(key, *labels)
    return key


@pytest.mark.parametrize("vector", OUTPUT_KEYS["vectors"], ids=lambda v: v["name"])
def test_output_key_vector_reproduces(vector: dict) -> None:
    assert _derive(vector) == vector["key"]


def test_output_key_rule_is_the_documented_wire_format() -> None:
    """Pin the byte format independently of derive_subkey: parent and labels joined by ':'."""
    parent = OUTPUT_KEYS["parent"]
    for vector in OUTPUT_KEYS["vectors"]:
        if "labels" in vector:
            raw = ":".join([parent, *vector["labels"]]).encode()
            assert hashlib.sha256(raw).hexdigest() == vector["key"], vector["name"]


def test_build_labelled_keys_never_match_the_cache_lookup() -> None:
    lookup = derive_subkey(OUTPUT_KEYS["parent"], *OUTPUT_KEYS["lookup_labels"])
    labelled = [v for v in OUTPUT_KEYS["vectors"] if v["name"] != "cached_output"]
    assert lookup == next(v["key"] for v in OUTPUT_KEYS["vectors"] if v["name"] == "cached_output")
    assert all(v["key"] != lookup for v in labelled)


def _fresh_store_tables(tmp_path: Path) -> set[str]:
    store = ArtifactStore(tmp_path)
    BuildStore(store.db_path, dialect=store.dialect)
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    return {name for (name,) in rows if name != "sqlite_sequence"}


def test_fresh_store_creates_exactly_the_ledger_tables(tmp_path: Path) -> None:
    assert _fresh_store_tables(tmp_path) == {t["name"] for t in TABLES["ledger"]}


def test_strata_owned_tables_are_not_created_by_the_ledger(tmp_path: Path) -> None:
    strata_owned = {t["name"] for t in TABLES["strata_owned"]}
    assert strata_owned
    assert not (_fresh_store_tables(tmp_path) & strata_owned)


def test_fixture_matches_the_packages_table_list() -> None:
    """The package publishes its own list; the fixture is what a second implementation reads."""
    migrated = [(t["name"], tuple(t["primary_key"])) for t in TABLES["ledger"] if t["migrated"]]
    assert migrated == list(LEDGER_TABLES)
    transient = {t["name"] for t in TABLES["ledger"] if not t["migrated"]}
    assert transient == set(TRANSIENT_TABLES)


def test_migrate_copies_the_ledger_first_then_stratas_tables() -> None:
    assert MIGRATED_TABLES[: len(LEDGER_TABLES)] == LEDGER_TABLES
    strata_owned = {t["name"] for t in TABLES["strata_owned"]}
    assert {name for name, _ in MIGRATED_TABLES[len(LEDGER_TABLES) :]} <= strata_owned


def test_provenance_domains_fit_the_charset() -> None:
    charset = re.compile(CLASSES["domain_charset"])
    for entry in CLASSES["domains"]:
        assert charset.match(entry["domain"]), entry["domain"]
        assert entry["class"] in CLASSES["classes"], entry["domain"]


def test_reconstructed_nodes_have_their_own_domain() -> None:
    by_class: dict[str, set[str]] = {}
    for entry in CLASSES["domains"]:
        by_class.setdefault(entry["class"], set()).add(entry["domain"])
    assert by_class["complete"].isdisjoint(by_class["reconstructed"])
    link = next(x for x in CLASSES["links"] if x["kind"] == "reconstructed_from")
    assert link["from_domain"] in by_class["complete"]
    assert link["to_domain"] in by_class["reconstructed"]


def test_adr001_matrix_assigns_every_row_to_a_step() -> None:
    steps = set(ADR001["steps"])
    assert ADR001["rows"]
    for row in ADR001["rows"]:
        assert row["step"] in steps, row["test"]
    fixtures_row = next(r for r in ADR001["rows"] if r["test"].startswith("CGN1"))
    assert fixtures_row["step"] == "E0"

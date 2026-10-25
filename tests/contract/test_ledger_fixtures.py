"""Ledger-side contract fixtures: output keys, table ownership, provenance classes (ADR 002 E0)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pytest

from strata.artifact.store import ArtifactStore
from strata.migrate import MIGRATED_TABLES
from strata.notebook.provenance import derive_subkey
from strata.transforms.build_store import BuildStore

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


def test_fresh_store_creates_exactly_the_ledger_tables_plus_the_pending_split(
    tmp_path: Path,
) -> None:
    """E0 baseline: the store's tables are the ledger list plus what E1 splits out of store.py."""
    ledger = {t["name"] for t in TABLES["ledger"]}
    pending_split = {t["name"] for t in TABLES["strata_owned"] if t.get("split_out_in") == "E1"}
    assert _fresh_store_tables(tmp_path) == ledger | pending_split


def test_other_strata_owned_tables_are_not_created_by_the_store(tmp_path: Path) -> None:
    elsewhere = {t["name"] for t in TABLES["strata_owned"] if "split_out_in" not in t}
    assert elsewhere
    assert not (_fresh_store_tables(tmp_path) & elsewhere)


def test_copy_order_and_primary_keys_match_migrate() -> None:
    """migrate.py composes the ledger list with Strata's own; the ledger part must agree."""
    ledger = {t["name"] for t in TABLES["ledger"]}
    fixture_order = [t["name"] for t in TABLES["ledger"] if t["migrated"]]
    migrate_order = [name for name, _ in MIGRATED_TABLES if name in ledger]
    assert migrate_order == fixture_order
    migrate_pks = dict(MIGRATED_TABLES)
    for table in TABLES["ledger"]:
        if table["migrated"]:
            assert tuple(table["primary_key"]) == migrate_pks[table["name"]], table["name"]
        else:
            assert table["name"] not in migrate_pks, table["name"]


def test_non_ledger_migrated_tables_are_all_declared_strata_owned() -> None:
    ledger = {t["name"] for t in TABLES["ledger"]}
    strata_owned = {t["name"] for t in TABLES["strata_owned"]}
    assert {name for name, _ in MIGRATED_TABLES} - ledger <= strata_owned


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

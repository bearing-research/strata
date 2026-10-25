"""The CGN1 and CGD1 fixtures reproduce byte for byte (ADR 002 E0).

A second implementation, in any language, loads the same JSON and must
produce the same digests; these tests pin the Python reference encoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.contract.hashing import (
    DOMAIN_RE,
    canonical_domain,
    cgd1_envelope,
    definition_id,
    identity_scope,
    node_id,
)

FIXTURES = Path(__file__).parent / "fixtures"
CGN1 = json.loads((FIXTURES / "cgn1.json").read_text(encoding="utf-8"))
CGD1 = json.loads((FIXTURES / "cgd1.json").read_text(encoding="utf-8"))
CLASSES = json.loads((FIXTURES / "provenance_classes.json").read_text(encoding="utf-8"))
SCOPE = bytes.fromhex(CGN1["scope"])
_BY_NAME = {v["name"]: v for v in CGN1["vectors"]}
_CGD1_BY_NAME = {v["name"]: v for v in CGD1["vectors"]}


def _node_id(vector: dict, domain: str | None = None) -> bytes:
    edges = [(e["role"], e["position"], _node_id(_BY_NAME[e["target"]])) for e in vector["edges"]]
    return node_id(
        SCOPE,
        CGN1["domain"] if domain is None else domain,
        CGN1["payload_version"],
        vector["payload_utf8"].encode("utf-8"),
        edges,
    )


@pytest.mark.parametrize("vector", CGN1["vectors"], ids=lambda v: v["name"])
def test_cgn1_vector_reproduces(vector: dict) -> None:
    assert _node_id(vector).hex() == vector["node_id"]


def test_cgn1_edge_insertion_order_is_irrelevant() -> None:
    assert _node_id(_BY_NAME["pair"]) == _node_id(_BY_NAME["pair_reversed_insertion"])


def test_cgn1_nfkc_folds_ligature_roles() -> None:
    assert _node_id(_BY_NAME["nfkc_ligature_role"]) == _node_id(_BY_NAME["nfkc_plain_role"])


def test_cgn1_roles_are_case_sensitive() -> None:
    upper = _BY_NAME["uppercase_role"]
    lower = {**upper, "edges": [{**upper["edges"][0], "role": "weights"}]}
    assert _node_id(lower) != _node_id(upper)


@pytest.mark.parametrize("case", CGN1["invalid"], ids=lambda c: c["name"])
def test_cgn1_rejects_invalid_envelopes(case: dict) -> None:
    with pytest.raises(ValueError):
        _node_id(case, domain=case["domain"])


_A = b"\x01" * 32


@pytest.mark.parametrize(
    ("scope", "payload_version", "edges"),
    [
        (b"\x00" * 15, 1, []),
        (SCOPE, 2**32, []),
        (SCOPE, 1, [("left", 0, b"\x00" * 31)]),
        (SCOPE, 1, [("left", -1, _A)]),
        (SCOPE, 1, [("left", 2**32, _A)]),
        (SCOPE, 1, [("r" * 65536, 0, _A)]),
    ],
    ids=[
        "short_scope",
        "payload_version_u32",
        "short_target",
        "negative_position",
        "position_u32",
        "role_u16",
    ],
)
def test_cgn1_rejects_out_of_range_fields_with_value_error(scope, payload_version, edges) -> None:
    with pytest.raises(ValueError):
        node_id(scope, "test", payload_version, b"", edges)


@pytest.mark.parametrize("vector", CGN1["scope_rule"]["vectors"], ids=lambda v: repr(v["tenant"]))
def test_scope_rule_reproduces(vector: dict) -> None:
    assert identity_scope(vector["tenant"]).hex() == vector["scope"]


def test_scope_separates_tenants() -> None:
    a = node_id(identity_scope("a"), "test", 1, b"", [])
    b = node_id(identity_scope("b"), "test", 1, b"", [])
    assert a != b


def test_fixture_charset_is_the_encoders_charset() -> None:
    assert CLASSES["domain_charset"] == DOMAIN_RE.pattern


@pytest.mark.parametrize("domain", [d["domain"] for d in CLASSES["domains"]])
def test_strata_domains_fit_the_restricted_charset(domain: str) -> None:
    assert canonical_domain(domain) == domain.encode("ascii")


def _definition_id(vector: dict) -> bytes:
    inputs = [
        (
            i["name"],
            i["type_contract_utf8"].encode(),
            bytes.fromhex(i["placeholder_hex"]),
            i["required"],
        )
        for i in vector["inputs"]
    ]
    outputs = [
        (o["name"], o["type_contract_utf8"].encode(), bytes.fromhex(o["producer_hex"]))
        for o in vector["outputs"]
    ]
    return definition_id(
        bytes.fromhex(CGD1["scope"]),
        CGD1["domain"],
        CGD1["interface_version"],
        inputs,
        outputs,
        [bytes.fromhex(r) for r in vector.get("roots_hex", [])],
        vector["effect_contract_utf8"].encode(),
        vector["execution_contract_utf8"].encode(),
    )


@pytest.mark.parametrize(
    "vector",
    [v for v in CGD1["vectors"] if v["status"] != "unverified"],
    ids=lambda v: v["name"],
)
def test_cgd1_vector_reproduces(vector: dict) -> None:
    assert _definition_id(vector).hex() == vector["definition_id"]


def test_cgd1_g1_is_recorded_as_unverified() -> None:
    """G1's placeholder bytes are unpublished; the fixture must say so rather than guess."""
    g1 = _CGD1_BY_NAME["G1"]
    assert g1["status"] == "unverified"
    assert "placeholder" in g1["reason"]


def test_cgd1_s1_pins_the_ports_layout() -> None:
    """S1 exists to catch a wrong field width in the ports section; G0 has no ports."""
    s1 = _CGD1_BY_NAME["S1"]
    assert s1["status"] == "strata_computed"
    assert s1["inputs"] and s1["outputs"] and s1["roots_hex"]


@pytest.mark.parametrize("case", CGD1["invalid"], ids=lambda c: c["name"])
def test_cgd1_rejects_invalid_definitions(case: dict) -> None:
    inputs = [(n, b"none:v1", _A, True) for n in case["inputs"]]
    outputs = [(n, b"finite:v1", _A) for n in case["outputs"]]
    with pytest.raises(ValueError):
        cgd1_envelope(SCOPE, "test", 1, inputs, outputs, [_A] * case["roots"])


def test_cgd1_changed_interface_changes_the_hash() -> None:
    one = definition_id(SCOPE, "test", 1, [("x", b"none:v1", _A, True)], [], [_A])
    two = definition_id(SCOPE, "test", 1, [("x", b"none:v1", _A, False)], [], [_A])
    assert one != two

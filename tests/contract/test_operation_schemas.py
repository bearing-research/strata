"""Typed transaction operations are data with a closed schema (ADR 002 §7.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).parent / "fixtures" / "operations.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)
OPERATIONS = [ref["$ref"].removeprefix("#/$defs/") for ref in SCHEMA["$defs"]["operation"]["oneOf"]]


def test_schema_is_a_valid_draft_2020_12_schema() -> None:
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("example", SCHEMA["examples"], ids=lambda e: f"{len(e)} ops")
def test_examples_validate(example: list) -> None:
    VALIDATOR.validate(example)


def test_every_operation_has_an_example() -> None:
    exercised = {op["op"] for example in SCHEMA["examples"] for op in example}
    assert exercised == set(OPERATIONS)


def test_every_def_is_an_operation() -> None:
    """No helper definitions: the schema holds the operation list and nothing speculative."""
    assert set(SCHEMA["$defs"]) == {"operation", *OPERATIONS}


@pytest.mark.parametrize("case", SCHEMA["counterexamples"], ids=lambda c: c["name"])
def test_counterexamples_are_rejected(case: dict) -> None:
    assert not VALIDATOR.is_valid(case["value"])


def _objects(schema: dict):
    """Every object schema reachable through ``properties``, including nested ones."""
    if schema.get("type") == "object":
        yield schema
    for child in schema.get("properties", {}).values():
        yield from _objects(child)


@pytest.mark.parametrize("name", OPERATIONS)
def test_no_operation_admits_free_text_sql(name: str) -> None:
    """The escape hatch is closed: every object in an operation is closed and has no SQL field.

    ``details`` on an audit event is a free object by design: it is recorded, never interpreted.
    """
    for obj in _objects(SCHEMA["$defs"][name]):
        if obj is SCHEMA["$defs"][name]["properties"].get("event", {}).get("properties", {}).get(
            "details"
        ):
            continue
        assert obj["additionalProperties"] is False, name
        assert "sql" not in obj.get("properties", {}), name

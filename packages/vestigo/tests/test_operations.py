"""Typed operations are data with a closed schema (ADR 002 §7.3); the package parses them."""

from __future__ import annotations

import pytest
from vestigo.operations import (
    AdvanceReference,
    AssertReferenceRevision,
    AssignAuthority,
    AuditEvent,
    CheckBuildLease,
    InsertAuditEvent,
    from_json,
    to_json,
)

# The contract fixture's example, verbatim (tests/contract/fixtures/operations.json).
EXAMPLE = [
    {"op": "CheckBuildLease", "build_id": "b1", "attempt": "a1"},
    {"op": "AssertReferenceRevision", "name": "model/latest", "expected_revision": 7},
    {"op": "AssignAuthority", "target": "slot:report/q3", "release_id": "r9"},
    {"op": "AdvanceReference", "name": "model/latest", "release_id": "r9"},
    {"op": "InsertAuditEvent", "event": {"action": "release.accept", "principal": "alice"}},
]


def test_the_example_round_trips():
    parsed = from_json(EXAMPLE)
    assert parsed == [
        CheckBuildLease("b1", "a1"),
        AssertReferenceRevision("model/latest", 7),
        AssignAuthority("slot:report/q3", "r9"),
        AdvanceReference("model/latest", "r9"),
        InsertAuditEvent(AuditEvent("release.accept", "alice")),
    ]
    assert to_json(parsed) == EXAMPLE
    assert from_json([]) == []


def test_details_are_recorded_not_interpreted():
    (op,) = from_json([{"op": "InsertAuditEvent", "event": {"action": "x", "details": {"k": 1}}}])
    assert isinstance(op, InsertAuditEvent)
    assert op.event.details == {"k": 1}
    assert to_json([op]) == [
        {"op": "InsertAuditEvent", "event": {"action": "x", "details": {"k": 1}}}
    ]


@pytest.mark.parametrize(
    "value",
    [
        [{"op": "RunSql", "sql": "DELETE FROM artifact_versions"}],
        [{"op": "CheckBuildLease", "build_id": "b1", "attempt": "a1", "sql": "x"}],
        [{"op": "CheckBuildLease", "build_id": "b1"}],
        [{"op": "CheckBuildLease", "build_id": "", "attempt": "a1"}],
        [{"op": "AssertReferenceRevision", "name": "n", "expected_revision": -1}],
        [{"op": "AssertReferenceRevision", "name": "n", "expected_revision": "7"}],
        [{"op": "AssertReferenceRevision", "name": "n", "expected_revision": True}],
        [{"op": "InsertAuditEvent", "event": {"principal": "alice"}}],
        [{"op": "InsertAuditEvent", "event": {"action": "x", "sql": "y"}}],
        [{"op": "InsertAuditEvent", "event": {"action": "x", "details": "free text"}}],
        [{"build_id": "b1", "attempt": "a1"}],
    ],
    ids=lambda v: str(v[0].get("op")) + ":" + ",".join(sorted(v[0])),
)
def test_strangers_extras_and_missing_fields_are_refused(value):
    with pytest.raises(ValueError):
        from_json(value)


def test_only_operations_serialize():
    with pytest.raises(ValueError, match="not an operation"):
        to_json([object()])  # type: ignore[list-item]

"""Typed transaction operations (ADR 002 §7.3): what a caller asks for with a release.

The ledger interprets them inside the publication transaction, on the same
connection, under the same write key, so a reference moves or an authority is
assigned exactly when the outputs become current, or not at all. They are data:
each serializes to the JSON object of the ``operations.json`` contract and back.
Nothing here is SQL, and there is no extension mechanism; a need the set does not
cover is a new operation, reviewed as a contract change.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CheckBuildLease:
    """Fail unless *attempt* still holds the build's lease and its write window is open."""

    build_id: str
    attempt: str


@dataclass(frozen=True)
class AssertReferenceRevision:
    """Fail unless the reference is at *expected_revision* (0 when it does not exist).

    An :class:`AdvanceReference` of the same name later in the transaction is guarded
    on it: it moves the reference from that revision only, whatever lands meanwhile.
    """

    name: str
    expected_revision: int


@dataclass(frozen=True)
class AssignAuthority:
    """Make the release the single winner of an output slot; fail if the slot has one."""

    target: str
    release_id: str


@dataclass(frozen=True)
class AdvanceReference:
    """Move a mutable reference to the release's outputs, which it then retains."""

    name: str
    release_id: str


@dataclass(frozen=True)
class AuditEvent:
    action: str
    principal: str | None = None
    details: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class InsertAuditEvent:
    """Append one audit row in the same transaction."""

    event: AuditEvent


Operation = (
    CheckBuildLease
    | AssertReferenceRevision
    | AssignAuthority
    | AdvanceReference
    | InsertAuditEvent
)

_BY_NAME: dict[str, type] = {
    "CheckBuildLease": CheckBuildLease,
    "AssertReferenceRevision": AssertReferenceRevision,
    "AssignAuthority": AssignAuthority,
    "AdvanceReference": AdvanceReference,
    "InsertAuditEvent": InsertAuditEvent,
}
_FIELDS: dict[str, dict[str, type]] = {
    "CheckBuildLease": {"build_id": str, "attempt": str},
    "AssertReferenceRevision": {"name": str, "expected_revision": int},
    "AssignAuthority": {"target": str, "release_id": str},
    "AdvanceReference": {"name": str, "release_id": str},
    "InsertAuditEvent": {"event": dict},
}


def to_json(operations: Iterable[Operation]) -> list[dict[str, Any]]:
    """The operations as the contract's JSON objects, in order."""
    out: list[dict[str, Any]] = []
    for op in operations:
        name = type(op).__name__
        if name not in _BY_NAME:
            raise ValueError(f"{name} is not an operation")
        if isinstance(op, InsertAuditEvent):
            event: dict[str, Any] = {"action": op.event.action}
            if op.event.principal is not None:
                event["principal"] = op.event.principal
            if op.event.details is not None:
                event["details"] = dict(op.event.details)
            out.append({"op": name, "event": event})
        else:
            out.append({"op": name, **op.__dict__})
    return out


def from_json(items: Iterable[Mapping[str, Any]]) -> list[Operation]:
    """Parse the contract's JSON objects; a stranger, an extra or a missing field is refused."""
    parsed: list[Operation] = []
    for item in items:
        name = item.get("op")
        if name not in _BY_NAME:
            raise ValueError(f"unknown operation {name!r}")
        fields = _FIELDS[name]
        given = {k: v for k, v in item.items() if k != "op"}
        if set(given) != set(fields):
            raise ValueError(f"{name} takes {sorted(fields)}, got {sorted(given)}")
        for key, kind in fields.items():
            value = given[key]
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ValueError(f"{name}.{key} must be {kind.__name__}")
            if kind is str and not value:
                raise ValueError(f"{name}.{key} must be nonempty")
            if kind is int and value < 0:
                raise ValueError(f"{name}.{key} must not be negative")
        if name == "InsertAuditEvent":
            parsed.append(InsertAuditEvent(_parse_event(given["event"])))
        else:
            parsed.append(_BY_NAME[name](**given))
    return parsed


def _parse_event(event: Mapping[str, Any]) -> AuditEvent:
    if set(event) - {"action", "principal", "details"}:
        raise ValueError("an audit event has action, principal and details only")
    action = event.get("action")
    if not isinstance(action, str) or not action:
        raise ValueError("an audit event's action must be a nonempty string")
    principal = event.get("principal")
    if principal is not None and not isinstance(principal, str):
        raise ValueError("an audit event's principal must be a string")
    details = event.get("details")
    if details is not None and not isinstance(details, Mapping):
        raise ValueError("an audit event's details must be an object")
    return AuditEvent(action=action, principal=principal, details=details)

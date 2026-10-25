"""Reference encoders for the ``CGN1`` and ``CGD1`` identity protocols.

RFC 003 §4 and §6.1 define the byte envelopes; ADR 002 §6.3 adds two rules on
top: the identity scope is derived from the tenant, and edge roles are
NFKC-normalized UTF-8 rather than restricted ASCII. This module is the seed
for the package's graph module (ADR 002 E3); until then the fixture tests in
this directory are its only caller. Every rejection is a ``ValueError``.
"""

from __future__ import annotations

import hashlib
import re
import struct
import unicodedata
from collections.abc import Iterable

DOMAIN_RE = re.compile(r"^[a-z0-9_\-./]+$")
NODE_ID_BYTES = 32
SCOPE_BYTES = 16
_U16 = 0xFFFF
_U32 = 0xFFFF_FFFF


def identity_scope(tenant: str) -> bytes:
    """``sha256("scope:" + tenant)[:16]``; the tenant is free text, ``''`` when there is none."""
    return hashlib.sha256(f"scope:{tenant}".encode()).digest()[:SCOPE_BYTES]


def canonical_role(role: str) -> bytes:
    """A role is the identifier's NFKC form, the normalization Python applies to identifiers."""
    if not role:
        raise ValueError("role must be nonempty")
    encoded = unicodedata.normalize("NFKC", role).encode("utf-8")
    _check_range(len(encoded), _U16, "role length")
    return encoded


def canonical_domain(domain: str) -> bytes:
    if not DOMAIN_RE.match(domain):
        raise ValueError(f"domain {domain!r} is not lowercase ASCII [a-z0-9_-./]")
    return domain.encode("ascii")


def _check_range(value: int, upper: int, what: str) -> None:
    if not 0 <= value <= upper:
        raise ValueError(f"{what} {value} is outside 0..{upper}")


def _check_scope(scope: bytes) -> None:
    if len(scope) != SCOPE_BYTES:
        raise ValueError(f"identity scope must be {SCOPE_BYTES} bytes, got {len(scope)}")


def _check_node_id(node_id: bytes, what: str) -> None:
    if len(node_id) != NODE_ID_BYTES:
        raise ValueError(f"{what} must be {NODE_ID_BYTES} bytes, got {len(node_id)}")


def _check_unique(items: list, what: str) -> None:
    if len(set(items)) != len(items):
        raise ValueError(f"duplicate {what}")


def cgn1_envelope(
    scope: bytes,
    domain: str,
    payload_version: int,
    payload: bytes,
    edges: Iterable[tuple[str, int, bytes]],
) -> bytes:
    """The canonical ``CGN1`` bytes of a node; ``edges`` are ``(role, position, target)``.

    Insertion order never matters: edges are sorted by ``(role bytes, position)``.
    A repeated ``(role, position)`` port is invalid; a repeated target is not.
    """
    _check_scope(scope)
    _check_range(payload_version, _U32, "payload version")
    canonical = []
    for role, position, target in edges:
        _check_range(position, _U32, "position")
        _check_node_id(target, "edge target")
        canonical.append((canonical_role(role), position, target))
    canonical.sort(key=lambda e: (e[0], e[1]))
    _check_unique([(r, p) for r, p, _ in canonical], "(role, position) port")
    d = canonical_domain(domain)
    out = bytearray(b"CGN1")
    out += struct.pack("<H", 1)
    out += scope
    out += struct.pack("<I", len(d)) + d
    out += struct.pack("<I", payload_version)
    out += struct.pack("<Q", len(payload)) + payload
    out += struct.pack("<I", len(canonical))
    for role, position, target in canonical:
        out += struct.pack("<H", len(role)) + role
        out += struct.pack("<I", position)
        out += target
    return bytes(out)


def node_id(
    scope: bytes,
    domain: str,
    payload_version: int,
    payload: bytes,
    edges: Iterable[tuple[str, int, bytes]],
) -> bytes:
    return hashlib.sha256(cgn1_envelope(scope, domain, payload_version, payload, edges)).digest()


def _canonical_port_name(name: str) -> bytes:
    encoded = canonical_domain(name)
    _check_range(len(encoded), _U16, "port name length")
    return encoded


def cgd1_envelope(
    scope: bytes,
    domain: str,
    interface_version: int,
    inputs: Iterable[tuple[str, bytes, bytes, bool]],
    outputs: Iterable[tuple[str, bytes, bytes]],
    roots: Iterable[bytes],
    effect_contract: bytes = b"",
    execution_contract: bytes = b"",
) -> bytes:
    """The canonical ``CGD1`` bytes of a graph definition (RFC 003 §6.1).

    ``inputs`` are ``(name, type_contract, placeholder_node_id, required)``;
    ``outputs`` are ``(name, type_contract, producer_node_id)``. Ports sort by
    name bytes, roots lexicographically. A port name used twice, on either side
    or across the two, is invalid, and so is a repeated root: both would let two
    encoders disagree on the bytes of one graph.
    """
    _check_scope(scope)
    _check_range(interface_version, _U32, "interface version")
    d = canonical_domain(domain)
    ins = sorted(inputs, key=lambda i: i[0].encode("ascii"))
    outs = sorted(outputs, key=lambda o: o[0].encode("ascii"))
    _check_unique([i[0] for i in ins] + [o[0] for o in outs], "port name")
    root_ids = sorted(roots)
    _check_unique(root_ids, "root")
    out = bytearray(b"CGD1")
    out += struct.pack("<H", 1)
    out += scope
    out += struct.pack("<I", len(d)) + d
    out += struct.pack("<I", interface_version)
    out += struct.pack("<I", len(ins))
    for name, type_contract, placeholder, required in ins:
        _check_node_id(placeholder, "placeholder node id")
        n = _canonical_port_name(name)
        out += struct.pack("<H", len(n)) + n
        out += struct.pack("<I", len(type_contract)) + type_contract
        out += placeholder
        out += bytes([1 if required else 0])
    out += struct.pack("<I", len(outs))
    for name, type_contract, producer in outs:
        _check_node_id(producer, "producer node id")
        n = _canonical_port_name(name)
        out += struct.pack("<H", len(n)) + n
        out += struct.pack("<I", len(type_contract)) + type_contract
        out += producer
    out += struct.pack("<I", len(root_ids))
    for root in root_ids:
        _check_node_id(root, "root node id")
        out += root
    out += struct.pack("<I", len(effect_contract)) + effect_contract
    out += struct.pack("<I", len(execution_contract)) + execution_contract
    return bytes(out)


def definition_id(
    scope: bytes,
    domain: str,
    interface_version: int,
    inputs: Iterable[tuple[str, bytes, bytes, bool]],
    outputs: Iterable[tuple[str, bytes, bytes]],
    roots: Iterable[bytes],
    effect_contract: bytes = b"",
    execution_contract: bytes = b"",
) -> bytes:
    return hashlib.sha256(
        cgd1_envelope(
            scope,
            domain,
            interface_version,
            inputs,
            outputs,
            roots,
            effect_contract,
            execution_contract,
        )
    ).digest()

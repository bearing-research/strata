"""The structural graph (RFC 003 §4 and §5, ADR 002 §6).

A node is structure without execution: a domain the application names, an opaque
payload, and edges to the nodes it reads, each under a role and a position. Its
identity is the SHA-256 of the canonical ``CGN1`` envelope, so two writers that
describe the same structure under the same identity scope get the same ``NodeId``,
and insertion order never matters. The ledger never parses a payload.

Nodes live in ``nodes`` and ``node_inputs`` and are never garbage collected here.
Traversal is depth-bounded and sized: a recursive query over a large store is a
cost the caller names.
"""

from __future__ import annotations

import hashlib
import heapq
import re
import struct
import time
import unicodedata
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vestigo.sql_backend import SqlDialect, StoreConnection

DOMAIN_RE = re.compile(r"^[a-z0-9_\-./]+$")
NODE_ID_BYTES = 32
SCOPE_BYTES = 16
_U16 = 0xFFFF
_U32 = 0xFFFF_FFFF

NodeId = str
"""A node's identity: 64 lowercase hex characters, the SHA-256 of its CGN1 envelope."""

MAX_DEPTH = 64
MAX_NODES = 10_000


# --- Canonical identities ------------------------------------------------------


def identity_scope(tenant: str | None) -> bytes:
    """``sha256("scope:" + tenant)[:16]``; the tenant is free text, ``''`` when there is none."""
    return hashlib.sha256(f"scope:{tenant or ''}".encode()).digest()[:SCOPE_BYTES]


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
    encoders disagree on the bytes of one graph. Composition itself is deferred
    (ADR 002 §6.4); the encoder is fixed now because it is inside every later hash.
    """
    _check_scope(scope)
    _check_range(interface_version, _U32, "interface version")
    d = canonical_domain(domain)
    ins = sorted(inputs, key=lambda i: i[0].encode("ascii"))
    outs = sorted(outputs, key=lambda o: o[0].encode("ascii"))
    _check_unique([i[0] for i in ins] + [o[0] for o in outs], "port name")
    root_list = sorted(roots)
    _check_unique(root_list, "root")
    for root in root_list:
        _check_node_id(root, "root")
    out = bytearray(b"CGD1")
    out += struct.pack("<H", 1)
    out += scope
    out += struct.pack("<I", len(d)) + d
    out += struct.pack("<I", interface_version)
    out += struct.pack("<I", len(ins))
    for name, type_contract, placeholder, required in ins:
        n = _canonical_port_name(name)
        _check_node_id(placeholder, "input placeholder")
        out += struct.pack("<H", len(n)) + n
        out += struct.pack("<I", len(type_contract)) + type_contract
        out += placeholder
        out += struct.pack("<B", 1 if required else 0)
    out += struct.pack("<I", len(outs))
    for name, type_contract, producer in outs:
        n = _canonical_port_name(name)
        _check_node_id(producer, "output producer")
        out += struct.pack("<H", len(n)) + n
        out += struct.pack("<I", len(type_contract)) + type_contract
        out += producer
    out += struct.pack("<I", len(root_list))
    for root in root_list:
        out += root
    out += struct.pack("<I", len(effect_contract)) + effect_contract
    out += struct.pack("<I", len(execution_contract)) + execution_contract
    return bytes(out)


def graph_definition_id(
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


# --- Nodes -----------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """One input port of a node: what it reads under *role* at *position*.

    *target* is a stored node's id, or the index of an earlier node in the same
    batch (RFC 003 §4.2), which the store resolves before hashing.
    """

    role: str
    position: int
    target: NodeId | int


@dataclass(frozen=True)
class NodeSpec:
    """What a writer says about a node; the store hashes it and never parses *payload*."""

    domain: str
    payload: bytes
    edges: tuple[Edge, ...] = ()
    payload_version: int = 1


@dataclass(frozen=True)
class NodeRecord:
    scope: str
    node_id: NodeId
    domain: str
    payload_version: int
    payload: bytes
    created_at: float


@dataclass(frozen=True)
class InputEdge:
    role: str
    position: int
    target_node_id: NodeId


@dataclass(frozen=True)
class Traversal:
    """How far a walk may go; both bounds are the caller's statement of cost."""

    max_depth: int = MAX_DEPTH
    max_nodes: int = MAX_NODES


DEFAULT_TRAVERSAL = Traversal()


GRAPH_DDL: tuple[str, ...] = (
    """CREATE TABLE IF NOT EXISTS nodes (
        scope TEXT NOT NULL,
        node_id TEXT NOT NULL,
        domain TEXT NOT NULL,
        payload_version INTEGER NOT NULL,
        payload BLOB NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (scope, node_id)
    )""",
    """CREATE TABLE IF NOT EXISTS node_inputs (
        scope TEXT NOT NULL,
        node_id TEXT NOT NULL,
        role TEXT NOT NULL,
        position INTEGER NOT NULL,
        target_node_id TEXT NOT NULL,
        PRIMARY KEY (scope, node_id, role, position)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_node_inputs_target ON node_inputs(scope, target_node_id)",
)
"""The graph's tables; also the tail of a fresh store's schema (``store._SCHEMA_SQL``)."""


@dataclass(frozen=True)
class _Resolved:
    """A batch entry with its edges' targets as stored ids and its own id computed."""

    spec: NodeSpec
    node_id: NodeId
    edges: tuple[tuple[str, int, NodeId], ...] = field(default_factory=tuple)


class GraphStore:
    """The structural graph of one ledger: content-addressed nodes and their inputs.

    Independent of artifacts: a planner writes candidate structures nothing ran, a
    notebook writes a cell's definition with its output (``definition_id`` on the
    version). Reached as ``ArtifactStore.graph``.
    """

    def __init__(self, connect: Callable[[], StoreConnection], dialect: SqlDialect) -> None:
        self._connect = connect
        self._dialect = dialect

    # --- Writing ---

    def put_batch(self, specs: Sequence[NodeSpec], *, tenant: str | None = None) -> list[NodeId]:
        """Store *specs* under the tenant's identity scope, atomically; the ids, in order.

        Idempotent: a node that exists is left as it is. An edge may name an earlier
        entry of the batch by index.

        Raises:
            ValueError: An edge names a later entry, itself, or a node the store does
                not hold; or a spec fails the ``CGN1`` rules. Nothing is written then.
        """
        conn = self._connect()
        try:
            ids = self.put_batch_on(conn, specs, tenant=tenant)
            conn.commit()
            return ids
        finally:
            conn.close()

    def put_batch_on(
        self, conn: StoreConnection, specs: Sequence[NodeSpec], *, tenant: str | None = None
    ) -> list[NodeId]:
        """As :meth:`put_batch`, on the caller's transaction (a publication's, say)."""
        scope = identity_scope(tenant)
        scope_hex = scope.hex()
        resolved = self._resolve(conn, scope, scope_hex, specs)
        now = time.time()
        for entry in resolved:
            conn.execute(
                "INSERT INTO nodes (scope, node_id, domain, payload_version, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (scope, node_id) DO NOTHING",
                (
                    scope_hex,
                    entry.node_id,
                    entry.spec.domain,
                    entry.spec.payload_version,
                    entry.spec.payload,
                    now,
                ),
            )
            for role, position, target in entry.edges:
                conn.execute(
                    "INSERT INTO node_inputs (scope, node_id, role, position, target_node_id) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (scope, node_id, role, position) DO NOTHING",
                    (scope_hex, entry.node_id, role, position, target),
                )
        return [entry.node_id for entry in resolved]

    def _resolve(
        self, conn: StoreConnection, scope: bytes, scope_hex: str, specs: Sequence[NodeSpec]
    ) -> list[_Resolved]:
        """Batch-local references to ids, every stored reference checked, every id computed."""
        stored = {
            _checked_id(e.target)
            for spec in specs
            for e in spec.edges
            if not isinstance(e.target, int)
        }
        held = self._held(conn, scope_hex, stored)
        resolved: list[_Resolved] = []
        known: set[NodeId] = set()
        for index, spec in enumerate(specs):
            edges: list[tuple[str, int, NodeId]] = []
            for edge in spec.edges:
                if isinstance(edge.target, int):
                    if not 0 <= edge.target < index:
                        raise ValueError(
                            f"batch entry {index} names entry {edge.target}; "
                            "an edge may name only an earlier entry"
                        )
                    target = resolved[edge.target].node_id
                else:
                    target = edge.target
                    if target not in known and target not in held:
                        raise ValueError(f"node {target} is not in the store")
                edges.append((canonical_role(edge.role).decode("utf-8"), edge.position, target))
            digest = node_id(
                scope,
                spec.domain,
                spec.payload_version,
                spec.payload,
                [(role, position, bytes.fromhex(target)) for role, position, target in edges],
            ).hex()
            known.add(digest)
            resolved.append(_Resolved(spec=spec, node_id=digest, edges=tuple(edges)))
        return resolved

    @staticmethod
    def _held(conn: StoreConnection, scope_hex: str, ids: Collection[NodeId]) -> set[NodeId]:
        """Which of *ids* the scope holds, in one query."""
        if not ids:
            return set()
        placeholders = ", ".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT node_id FROM nodes WHERE scope = ? AND node_id IN ({placeholders})",
            (scope_hex, *ids),
        ).fetchall()
        return {r["node_id"] for r in rows}

    # --- Reading ---

    def get(self, node_id: NodeId, *, tenant: str | None = None) -> NodeRecord | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT scope, node_id, domain, payload_version, payload, created_at "
                "FROM nodes WHERE scope = ? AND node_id = ?",
                (identity_scope(tenant).hex(), _checked_id(node_id)),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return NodeRecord(
            scope=row["scope"],
            node_id=row["node_id"],
            domain=row["domain"],
            payload_version=row["payload_version"],
            payload=bytes(row["payload"]),
            created_at=row["created_at"],
        )

    def inputs(self, node_id: NodeId, *, tenant: str | None = None) -> list[InputEdge]:
        """A node's input ports in canonical order (role, then position)."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT role, position, target_node_id FROM node_inputs "
                "WHERE scope = ? AND node_id = ? ORDER BY role, position",
                (identity_scope(tenant).hex(), _checked_id(node_id)),
            ).fetchall()
        finally:
            conn.close()
        return [
            InputEdge(role=r["role"], position=r["position"], target_node_id=r["target_node_id"])
            for r in rows
        ]

    def ancestors(
        self,
        roots: Collection[NodeId],
        *,
        tenant: str | None = None,
        traversal: Traversal = DEFAULT_TRAVERSAL,
    ) -> list[NodeId]:
        """Every node the *roots* read, directly or through others; the roots excluded.

        Nearest first (by depth), then by id; cut at the traversal's bounds.
        """
        return self._walk(roots, "node_id", "target_node_id", tenant, traversal)

    def dependents(
        self,
        node_id: NodeId,
        *,
        tenant: str | None = None,
        traversal: Traversal = DEFAULT_TRAVERSAL,
    ) -> list[NodeId]:
        """Every node that reads *node_id*, directly or through others; itself excluded."""
        return self._walk([node_id], "target_node_id", "node_id", tenant, traversal)

    def topological_order(
        self,
        roots: Collection[NodeId],
        *,
        tenant: str | None = None,
        traversal: Traversal = DEFAULT_TRAVERSAL,
    ) -> list[NodeId]:
        """The roots and their ancestors, inputs before the nodes that read them.

        Among nodes nothing orders, by id, so the order is the same on every call.
        """
        scope_hex = identity_scope(tenant).hex()
        members = set(roots) | set(self.ancestors(roots, tenant=tenant, traversal=traversal))
        if not members:
            return []
        placeholders = ", ".join("?" for _ in members)
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT node_id, target_node_id FROM node_inputs "
                f"WHERE scope = ? AND node_id IN ({placeholders})",
                (scope_hex, *sorted(members)),
            ).fetchall()
        finally:
            conn.close()
        unread: dict[NodeId, int] = dict.fromkeys(members, 0)
        readers: dict[NodeId, list[NodeId]] = {m: [] for m in members}
        for row in rows:
            reader, target = row["node_id"], row["target_node_id"]
            if target not in members:
                raise ValueError("the traversal bound cut the graph before its inputs")
            unread[reader] += 1
            readers[target].append(reader)
        # Kahn's algorithm, taking the smallest ready id first.
        order: list[NodeId] = []
        ready = [m for m, count in unread.items() if count == 0]
        heapq.heapify(ready)
        while ready:
            node = heapq.heappop(ready)
            order.append(node)
            for reader in readers[node]:
                unread[reader] -= 1
                if unread[reader] == 0:
                    heapq.heappush(ready, reader)
        # Content addressing makes a cycle impossible to store, so nothing stays pending.
        return order

    def _walk(
        self,
        starts: Collection[NodeId],
        from_column: str,
        to_column: str,
        tenant: str | None,
        traversal: Traversal,
    ) -> list[NodeId]:
        start_ids = sorted({_checked_id(s) for s in starts})
        if not start_ids:
            return []
        scope_hex = identity_scope(tenant).hex()
        placeholders = ", ".join("?" for _ in start_ids)
        # One recursive query on both dialects; the depth bound ends it on any store.
        sql = f"""
            WITH RECURSIVE walk(node_id, depth) AS (
                SELECT {to_column}, 1 FROM node_inputs
                WHERE scope = ? AND {from_column} IN ({placeholders})
                UNION
                SELECT i.{to_column}, w.depth + 1 FROM node_inputs i
                JOIN walk w ON i.{from_column} = w.node_id
                WHERE i.scope = ? AND w.depth < ?
            )
            SELECT node_id, MIN(depth) AS depth FROM walk
            WHERE node_id NOT IN ({placeholders})
            GROUP BY node_id ORDER BY depth, node_id LIMIT ?
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                sql,
                (
                    scope_hex,
                    *start_ids,
                    scope_hex,
                    traversal.max_depth,
                    *start_ids,
                    traversal.max_nodes + 1,
                ),
            ).fetchall()
        finally:
            conn.close()
        if len(rows) > traversal.max_nodes:
            raise ValueError(f"more than {traversal.max_nodes} nodes reached; raise the bound")
        return [r["node_id"] for r in rows]


def _checked_id(node_id: NodeId) -> NodeId:
    if len(node_id) != 2 * NODE_ID_BYTES or any(c not in "0123456789abcdef" for c in node_id):
        raise ValueError(f"{node_id!r} is not a node id (64 lowercase hex characters)")
    return node_id

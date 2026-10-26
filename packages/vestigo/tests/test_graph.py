"""The structural graph (ADR 002 §6): content-addressed nodes, independent of artifacts.

Gate 2's graph steps: ``put_batch`` a three-node structure, read it back, walk it. The
ids are the SHA-256 of the canonical ``CGN1`` envelope, so what a writer describes is
what any other writer gets, in any order.
"""

from __future__ import annotations

import random
import sqlite3

import pytest

from vestigo.graph import (
    Edge,
    NodeSpec,
    Traversal,
    canonical_role,
    cgd1_envelope,
    identity_scope,
    node_id,
)
from vestigo.provenance import TransformSpec
from vestigo.store import ArtifactStore


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


LEAF = NodeSpec(domain="test/table", payload=b"s3://bucket/events")
FIRST = NodeSpec(domain="test/cell", payload=b"a", edges=(Edge("rows", 0, 0),))
SECOND = NodeSpec(domain="test/cell", payload=b"b", edges=(Edge("left", 0, 0), Edge("right", 0, 1)))
THREE = [LEAF, FIRST, SECOND]


def _hex(n: int) -> str:
    return f"{n:064x}"


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Recording:
    """A connection that records the node ids each ``INSERT INTO nodes`` carries, and
    hands back a node's ports in a foreign collation's order (glibc's en_US.utf8)."""

    def __init__(self, inner, inserted: list[str], port_order: list[str]):
        self._inner = inner
        self._inserted = inserted
        self._port_order = port_order

    def __getattr__(self, attr):
        return getattr(self._inner, attr)

    def execute(self, sql, params=()):
        if sql.startswith("INSERT INTO nodes"):
            self._inserted.append(params[1])
        cursor = self._inner.execute(sql, params)
        if sql.startswith("SELECT role, position, target_node_id FROM node_inputs"):
            rows = sorted(cursor.fetchall(), key=lambda r: self._port_order.index(r["role"]))
            return _Rows(rows)
        return cursor


@pytest.fixture
def recording(store):
    """The store's graph on recording connections; yields the recorded insert ids."""
    inserted: list[str] = []
    connect = store.graph._connect
    store.graph._connect = lambda: _Recording(connect(), inserted, GLIBC_ORDER)
    yield inserted
    store.graph._connect = connect


GLIBC_ORDER = ["a", "b", "\u00e9", "_x", "Z"]


class TestPutBatch:
    def test_stores_three_nodes_and_their_ports(self, store):
        leaf, first, second = store.graph.put_batch(THREE)

        for ident in (leaf, first, second):
            assert len(ident) == 64 and int(ident, 16) >= 0
        record = store.graph.get(second)
        assert record is not None
        assert (record.domain, record.payload, record.payload_version) == ("test/cell", b"b", 1)
        assert record.scope == identity_scope(None).hex()
        ports = store.graph.inputs(second)
        assert [(p.role, p.position, p.target_node_id) for p in ports] == [
            ("left", 0, leaf),
            ("right", 0, first),
        ]
        assert store.graph.inputs(leaf) == []

    def test_ids_are_content_addressed_and_the_write_idempotent(self, store, tmp_path):
        once = store.graph.put_batch(THREE)
        again = store.graph.put_batch(THREE)
        elsewhere = ArtifactStore(tmp_path / "other").graph.put_batch(THREE)

        assert once == again == elsewhere
        conn = sqlite3.connect(store.db_path)
        assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM node_inputs").fetchone()[0] == 3

    def test_an_edge_may_name_a_stored_node_by_id(self, store):
        (leaf,) = store.graph.put_batch([LEAF])
        (first,) = store.graph.put_batch([NodeSpec("test/cell", b"a", (Edge("rows", 0, leaf),))])

        assert store.graph.put_batch([LEAF, FIRST]) == [leaf, first]

    def test_a_batch_reference_may_name_only_an_earlier_entry(self, store):
        forward = [NodeSpec("test/cell", b"a", (Edge("rows", 0, 1),)), LEAF]
        with pytest.raises(ValueError, match="earlier entry"):
            store.graph.put_batch(forward)
        itself = [NodeSpec("test/cell", b"a", (Edge("rows", 0, 0),))]
        with pytest.raises(ValueError, match="earlier entry"):
            store.graph.put_batch(itself)
        # Nothing of a refused batch is written.
        assert (
            store.graph.get(node_id(identity_scope(None), "test/table", 1, LEAF.payload, []).hex())
            is None
        )

    def test_an_edge_to_a_node_the_store_lacks_is_refused(self, store):
        with pytest.raises(ValueError, match="not in the store"):
            store.graph.put_batch([NodeSpec("test/cell", b"a", (Edge("rows", 0, _hex(7)),))])

    def test_a_port_used_twice_is_refused(self, store):
        twice = NodeSpec("test/cell", b"a", (Edge("rows", 0, 0), Edge("rows", 0, 0)))
        with pytest.raises(ValueError, match="duplicate"):
            store.graph.put_batch([LEAF, twice])

    def test_a_malformed_id_is_refused(self, store):
        with pytest.raises(ValueError, match="not a node id"):
            store.graph.get("abc")
        with pytest.raises(ValueError, match="not a node id"):
            store.graph.put_batch([NodeSpec("test/cell", b"a", (Edge("rows", 0, "X" * 64),))])

    def test_tenants_have_separate_scopes(self, store):
        mine = store.graph.put_batch(THREE, tenant="alice")
        theirs = store.graph.put_batch(THREE, tenant="bob")

        assert set(mine).isdisjoint(theirs)
        assert store.graph.get(mine[0], tenant="alice") is not None
        assert store.graph.get(mine[0], tenant="bob") is None
        assert store.graph.get(mine[0]) is None
        # An edge cannot cross scopes: the other tenant's node is not in this one's store.
        with pytest.raises(ValueError, match="not in the store"):
            store.graph.put_batch(
                [NodeSpec("test/cell", b"c", (Edge("rows", 0, mine[0]),))], tenant="bob"
            )

    def test_roles_are_stored_in_their_nfkc_form(self, store):
        ligature = NodeSpec("test/cell", b"a", (Edge("ﬁle", 0, 0),))
        plain = NodeSpec("test/cell", b"a", (Edge("file", 0, 0),))

        _, one = store.graph.put_batch([LEAF, ligature])
        _, two = store.graph.put_batch([LEAF, plain])

        assert one == two
        assert [p.role for p in store.graph.inputs(one)] == ["file"]

    def test_a_role_is_an_identifier(self, store):
        for role in ("a b", "1x", "x\n", "left-table", "a.b"):
            with pytest.raises(ValueError, match="not an identifier"):
                store.graph.put_batch([LEAF, NodeSpec("test/cell", b"a", (Edge(role, 0, 0),))])
        assert canonical_role("\ufb01le") == b"file"
        assert canonical_role("donn\u00e9es") == "données".encode()

    def test_ports_come_in_the_encoders_order_whatever_the_backend_collates(self, store, recording):
        # glibc's en_US.utf8 orders these a, b, é, _x, Z; the encoder's byte order does not.
        edges = tuple(Edge(role, 0, 0) for role in GLIBC_ORDER)
        _, mixed = store.graph.put_batch([LEAF, NodeSpec("test/cell", b"m", edges)])

        assert [p.role for p in store.graph.inputs(mixed)] == ["Z", "_x", "a", "b", "\u00e9"]

    def test_nodes_are_inserted_in_id_order(self, store, recording):
        leaves = [NodeSpec("test/table", bytes([i])) for i in range(6)]
        ids = store.graph.put_batch(leaves)
        assert ids != sorted(ids), "the batch happens to be in id order; pick other payloads"

        descending = [leaves[ids.index(i)] for i in sorted(ids, reverse=True)]
        recording.clear()
        assert store.graph.put_batch(descending) == sorted(ids, reverse=True)
        assert recording == sorted(ids)


class TestEncoders:
    SCOPE = identity_scope(None)
    A = b"\x01" * 32

    def test_a_non_ascii_port_name_is_refused_as_a_port_name(self):
        with pytest.raises(ValueError, match="not lowercase ASCII"):
            cgd1_envelope(self.SCOPE, "test", 1, [("\u00e9", b"", self.A, True)], [], [self.A])
        with pytest.raises(ValueError, match="not lowercase ASCII"):
            cgd1_envelope(self.SCOPE, "test", 1, [], [("\u00e9", b"", self.A)], [self.A])


class TestTraversal:
    def test_ancestors_dependents_and_topological_order(self, store):
        leaf, first, second = store.graph.put_batch(THREE)

        # Both are direct inputs of second, so depth ties and the id decides.
        assert store.graph.ancestors([second]) == sorted([leaf, first])
        assert store.graph.ancestors([leaf]) == []
        assert set(store.graph.dependents(leaf)) == {first, second}
        assert store.graph.dependents(second) == []
        assert store.graph.topological_order([second]) == [leaf, first, second]
        assert store.graph.topological_order([first, second]) == [leaf, first, second]
        assert store.graph.topological_order([]) == []

    def test_ancestors_come_nearest_first(self, store):
        leaf, first, second = store.graph.put_batch(THREE)
        assert store.graph.ancestors([second])[0] in (leaf, first)
        chain = [LEAF] + [
            NodeSpec("test/cell", bytes([i]), (Edge("prev", 0, i),)) for i in range(4)
        ]
        ids = store.graph.put_batch(chain)
        # A chain has one node per depth, so depth order is the chain reversed.
        assert store.graph.ancestors([ids[-1]]) == ids[-2::-1]

    def test_a_depth_bound_below_one_is_refused(self):
        for depth in (0, -5):
            with pytest.raises(ValueError, match="below 1"):
                Traversal(max_depth=depth)
        assert Traversal(max_depth=1).max_depth == 1

    def test_a_root_the_store_lacks_is_refused(self, store):
        leaf, first, second = store.graph.put_batch(THREE)
        with pytest.raises(ValueError, match="not in the store"):
            store.graph.topological_order(["e" * 64])
        with pytest.raises(ValueError, match="not in the store"):
            store.graph.topological_order([second, "e" * 64])
        # Walks from an unknown node answer with what they reach: nothing.
        assert store.graph.ancestors(["e" * 64]) == []

    def test_a_walk_stops_at_its_depth_bound(self, store):
        chain = [LEAF] + [
            NodeSpec("test/cell", bytes([i]), (Edge("prev", 0, i),)) for i in range(5)
        ]
        ids = store.graph.put_batch(chain)
        near = store.graph.ancestors([ids[-1]], traversal=Traversal(max_depth=2))

        assert near == [ids[-2], ids[-3]]
        with pytest.raises(ValueError, match="cut the graph"):
            store.graph.topological_order([ids[-1]], traversal=Traversal(max_depth=2))
        assert store.graph.topological_order([ids[-1]], traversal=Traversal(max_depth=5)) == ids

    def test_a_start_node_reached_again_does_not_count_against_the_bound(self, store):
        leaf, first, second = store.graph.put_batch(THREE)
        # first is an ancestor of second: starting from both must not count it twice.
        assert store.graph.ancestors([first, second], traversal=Traversal(max_nodes=1)) == [leaf]

    def test_a_walk_refuses_to_exceed_its_size_bound(self, store):
        _, _, second = store.graph.put_batch(THREE)
        with pytest.raises(ValueError, match="more than 1 nodes"):
            store.graph.ancestors([second], traversal=Traversal(max_nodes=1))
        assert len(store.graph.ancestors([second], traversal=Traversal(max_nodes=2))) == 2

    def test_a_walk_reaches_nodes_shared_by_two_lineages(self, store):
        leaf, first, second = store.graph.put_batch(THREE)
        (other,) = store.graph.put_batch([NodeSpec("test/cell", b"z", (Edge("rows", 0, leaf),))])

        assert set(store.graph.dependents(leaf)) == {first, second, other}
        assert store.graph.topological_order([second, other])[0] == leaf


class TestDefinitions:
    """A version names the node that produced it (``definition_id``)."""

    def test_a_version_created_with_its_definition_records_the_last_node(self, store):
        version = store.create_artifact(
            "out", "prov-1", TransformSpec("x@v1", {}, []), definition=THREE
        )
        leaf, first, second = store.graph.put_batch(THREE)

        assert store.definition_of("out", version) == second
        assert store.graph.get(leaf) is not None
        assert store.definition_of("out", 99) is None

    def test_a_refused_definition_creates_no_version(self, store):
        bad = [NodeSpec("test/cell", b"a", (Edge("rows", 0, _hex(1)),))]
        with pytest.raises(ValueError):
            store.create_artifact("out", "prov-1", TransformSpec("x@v1", {}, []), definition=bad)
        assert store.get_latest_version("out") is None

    def test_a_definition_is_written_under_the_versions_tenant(self, store):
        version = store.create_artifact(
            "out", "prov-1", TransformSpec("x@v1", {}, []), tenant="t1", definition=THREE
        )
        assert store.definition_of("out", version) is None
        node = store.definition_of("out", version, tenant="t1")
        assert node is not None
        assert store.graph.get(node, tenant="t1") is not None
        assert store.graph.get(node) is None

    def test_versions_without_a_definition_can_be_given_one_once(self, store):
        spec = TransformSpec("x@v1", {}, [])
        v1 = store.create_artifact("old", "prov-1", spec)
        store.finalize_artifact("old", v1, "{}", 1, 1)
        v2 = store.create_artifact("old", "prov-2", spec)
        store.finalize_artifact("old", v2, "{}", 1, 1)
        unfinished = store.create_artifact("building", "prov-3", spec)
        leaf, first, second = store.graph.put_batch(THREE)

        lacking = [(v.id, v.version) for v in store.versions_lacking_definition()]
        assert lacking == [("old", 1), ("old", 2)]
        assert ("building", unfinished) not in lacking

        assert store.versions_lacking_definition(executor="x@v1") != []
        assert store.versions_lacking_definition(executor="x_v1") == []
        assert store.versions_lacking_definition(executor="x@v") == []

        store.set_definitions({("old", 1): second, ("old", 2): first})
        store.set_definitions({("old", 1): leaf})

        assert store.definition_of("old", 1) == second
        assert store.definition_of("old", 2) == first
        assert store.versions_lacking_definition() == []

    def test_a_definition_must_be_a_node_the_store_holds(self, store):
        """A string that is not a node of the version's scope is refused, not recorded."""
        spec = TransformSpec("x@v1", {}, [])
        v1 = store.create_artifact("old", "prov-1", spec)
        store.finalize_artifact("old", v1, "{}", 1, 1)
        with pytest.raises(ValueError):
            store.set_definitions({("old", 1): _hex(5)})
        assert store.definition_of("old", 1) is None


class TestMigration:
    def test_an_old_store_gains_the_graph_on_open(self, tmp_path):
        store = ArtifactStore(tmp_path / "old")
        conn = sqlite3.connect(store.db_path)
        conn.executescript(
            "DROP TABLE nodes; DROP TABLE node_inputs; DROP INDEX idx_versions_definition; "
            "ALTER TABLE artifact_versions DROP COLUMN definition_id; "
            "DELETE FROM schema_version WHERE version >= 13;"
        )
        conn.commit()
        conn.close()

        reopened = ArtifactStore(tmp_path / "old")
        version = reopened.create_artifact(
            "a", "p", TransformSpec("x@v1", {}, []), definition=THREE
        )

        assert reopened.definition_of("a", version) is not None


class TestCanonicalization:
    """Properties of the CGN1 identity under a seeded generator."""

    SCOPE = identity_scope("prop")

    @staticmethod
    def _random_node(rng: random.Random, pool: list[bytes]):
        domain = "".join(rng.choice("abc/_-.") for _ in range(rng.randint(1, 8)))
        payload = rng.randbytes(rng.randint(0, 16))
        edges = []
        used = set()
        for _ in range(rng.randint(0, 5)):
            role = "".join(rng.choice("abＡéﬁx_") for _ in range(rng.randint(1, 4)))
            position = rng.randint(0, 3)
            port = (canonical_role(role), position)
            if port in used:
                continue
            used.add(port)
            edges.append((role, position, rng.choice(pool)))
        return domain, rng.randint(0, 3), payload, edges

    def test_edge_order_never_changes_the_id(self):
        rng = random.Random(1234)
        pool = [rng.randbytes(32) for _ in range(4)]
        for _ in range(300):
            domain, pv, payload, edges = self._random_node(rng, pool)
            shuffled = list(edges)
            rng.shuffle(shuffled)
            assert node_id(self.SCOPE, domain, pv, payload, edges) == node_id(
                self.SCOPE, domain, pv, payload, shuffled
            )

    def test_every_field_is_inside_the_id(self):
        rng = random.Random(99)
        pool = [rng.randbytes(32) for _ in range(4)]
        for _ in range(300):
            domain, pv, payload, edges = self._random_node(rng, pool)
            base = node_id(self.SCOPE, domain, pv, payload, edges)
            assert node_id(self.SCOPE, domain + "x", pv, payload, edges) != base
            assert node_id(self.SCOPE, domain, pv + 1, payload, edges) != base
            assert node_id(self.SCOPE, domain, pv, payload + b"\0", edges) != base
            assert node_id(identity_scope("other"), domain, pv, payload, edges) != base
            if edges:
                role, position, target = edges[0]
                moved = [(role, position + 4, target)] + edges[1:]
                assert node_id(self.SCOPE, domain, pv, payload, moved) != base
                other = next(t for t in pool if t != target)
                retargeted = [(role, position, other)] + edges[1:]
                assert node_id(self.SCOPE, domain, pv, payload, retargeted) != base

    def test_roles_equal_under_nfkc_are_one_role(self):
        rng = random.Random(7)
        pool = [rng.randbytes(32)]
        pairs = [("ﬁle", "file"), ("Ａ", "A"), ("é", "é"), ("x", "x")]
        for _ in range(100):
            composed, plain = rng.choice(pairs)
            edges = [(composed, 0, pool[0])]
            assert node_id(self.SCOPE, "d", 1, b"", edges) == node_id(
                self.SCOPE, "d", 1, b"", [(plain, 0, pool[0])]
            )
        assert canonical_role("A") != canonical_role("a")

"""Gate 2 (ADR 002 §8): the lifecycle, proven in the package before any harness exists.

One test class, run against SQLite and PostgreSQL by the modules that import it; each
test stands alone on a ``ledger`` fixture those modules provide. Nothing here imports
Strata.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

import pytest
from vestigo.build_store import BuildStore
from vestigo.graph import Edge, NodeSpec
from vestigo.operations import (
    AdvanceReference,
    AssertReferenceRevision,
    AssignAuthority,
    AuditEvent,
    CheckBuildLease,
    InsertAuditEvent,
)
from vestigo.provenance import TransformSpec, checkpoint_output_key, derive_subkey, run_output_key
from vestigo.releases import (
    OperationFailed,
    Release,
    ReleaseIncomplete,
    ReleaseOutOfOrder,
    ReleaseRejected,
    ReleaseStore,
)
from vestigo.store import ArtifactStore, StagedVersion

PARENT = hashlib.sha256(b"train").hexdigest()
SPEC = TransformSpec(executor="train@v1", params={"epochs": "3"}, inputs=[])


class Clock:
    """A clock the tests advance, so a write window can close without sleeping."""

    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def __call__(self) -> float:
        return self.now


@dataclass
class Ledger:
    store: ArtifactStore
    builds: BuildStore
    releases: ReleaseStore
    clock: Clock
    build_ids: list[str] = field(default_factory=list)


def make_ledger(store: ArtifactStore) -> Ledger:
    clock = Clock()
    builds = BuildStore(store.db_path, dialect=store.dialect, clock=clock)
    return Ledger(store, builds, ReleaseStore(store, builds, clock=clock), clock)


def _build(
    ledger: Ledger,
    attempt: str = "a1",
    *,
    occurrence: str | None = None,
    computation_key: str = PARENT,
) -> str:
    """A building build under *attempt*, whose target row is the final output."""
    build_id = uuid.uuid4().hex
    target = f"final-{build_id}"
    version = ledger.store.create_artifact(
        target,
        run_output_key(computation_key, build_id, "model"),
        SPEC,
        computation_key=computation_key,
        minted=True,
    )
    ledger.builds.create_build(
        build_id,
        target,
        version,
        executor_ref="train@v1",
        occurrence=occurrence,
        computation_key=computation_key,
    )
    assert ledger.builds.open_attempt(
        build_id, attempt, writable_until=ledger.clock() + 3600, lease_owner="runner-1"
    )
    return build_id


def _staged_row(artifact_id: str, version: int, data: bytes) -> StagedVersion:
    return StagedVersion(
        artifact_id=artifact_id,
        version=version,
        schema_json="{}",
        row_count=None,
        byte_size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )


def _state(ledger: Ledger, row) -> str:
    """The stored state of a staged or stored version."""
    artifact_id = getattr(row, "artifact_id", None) or row.id
    return ledger.store.get_artifact(artifact_id, row.version).state


def _stage(
    ledger: Ledger, release: Release, name: str, data: bytes, *, write: bool = True
) -> StagedVersion:
    """One output of a checkpoint release: its own key, its bytes written unless told not to."""
    artifact_id = f"{release.build_id}-{release.kind}-{release.sequence}-{name}"
    key = (
        run_output_key(PARENT, release.build_id, name)
        if release.kind == "final"
        else checkpoint_output_key(PARENT, release.build_id, release.sequence, name)
    )
    version = ledger.store.create_artifact(
        artifact_id, key, SPEC, computation_key=PARENT, minted=True, release_id=release.release_id
    )
    if write:
        ledger.store.write_blob(artifact_id, version, data)
    return _staged_row(artifact_id, version, data)


def _checkpoint(ledger: Ledger, build_id: str, attempt: str = "a1", *, operations=()):
    """Prepare, stage and accept one checkpoint with a model and its metrics."""
    release = ledger.releases.prepare(build_id, attempt, "checkpoint")
    staged = [
        _stage(ledger, release, "model", b"weights" + bytes([release.sequence])),
        _stage(ledger, release, "metrics", b"{}"),
    ]
    return ledger.releases.accept(release.release_id, staged, list(operations))


class Gate2:
    """The gate's steps; a ``ledger`` fixture is the importing module's."""

    def test_a_three_node_structure_is_put_and_walked(self, ledger: Ledger):
        leaf = NodeSpec("strata/table", b"s3://lake/events")
        features = NodeSpec("strata/cell", b"features", (Edge("events", 0, 0),))
        train = NodeSpec("strata/cell", b"train", (Edge("features", 0, 1),))
        leaf_id, features_id, train_id = ledger.store.graph.put_batch([leaf, features, train])

        assert ledger.store.graph.topological_order([train_id]) == [leaf_id, features_id, train_id]
        assert set(ledger.store.graph.dependents(leaf_id)) == {features_id, train_id}

    def test_two_checkpoints_and_a_final_with_typed_operations(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [
            _stage(ledger, release, "model", b"w1"),
            _stage(ledger, release, "metrics", b"m1"),
        ]
        first = ledger.releases.accept(
            release.release_id,
            staged,
            [
                CheckBuildLease(build_id, "a1"),
                AssertReferenceRevision("train/latest", 0),
                AdvanceReference("train/latest", release.release_id),
                InsertAuditEvent(AuditEvent("release.accept", "alice", {"sequence": 1})),
            ],
        )
        assert first.release.state == "accepted"
        # Two outputs of one invocation: the same release id, both ready.
        assert {v.state for v in first.outputs} == {"ready"}
        assert {_state(ledger, v) for v in first.outputs} == {"ready"}
        assert ledger.releases.reference("train/latest") == (release.release_id, 1)
        assert ledger.builds.get_build(build_id).state == "building"

        second = ledger.releases.prepare(build_id, "a1", "checkpoint")
        assert second.sequence == 2
        staged = [_stage(ledger, second, "model", b"w2"), _stage(ledger, second, "metrics", b"m2")]
        bundle = ledger.releases.accept(
            second.release_id,
            staged,
            [
                AssertReferenceRevision("train/latest", 1),
                AdvanceReference("train/latest", second.release_id),
            ],
        )
        assert ledger.releases.reference("train/latest") == (second.release_id, 2)
        assert [v.provenance_hash for v in bundle.outputs] == [
            checkpoint_output_key(PARENT, build_id, 2, "metrics"),
            checkpoint_output_key(PARENT, build_id, 2, "model"),
        ] or {v.provenance_hash for v in bundle.outputs} == {
            checkpoint_output_key(PARENT, build_id, 2, "metrics"),
            checkpoint_output_key(PARENT, build_id, 2, "model"),
        }

        final = ledger.releases.prepare(build_id, "a1", "final")
        build = ledger.builds.get_build(build_id)
        assert ledger.releases.attach(final.release_id, [(build.artifact_id, build.version)]) == 1
        ledger.store.write_blob(build.artifact_id, build.version, b"final-weights")
        staged = [_staged_row(build.artifact_id, build.version, b"final-weights")]
        done = ledger.releases.accept(
            final.release_id, staged, [AssignAuthority(f"slot:{build_id}", final.release_id)]
        )
        assert done.release.kind == "final"
        assert ledger.builds.get_build(build_id).state == "ready"
        assert ledger.releases.authority(f"slot:{build_id}") == final.release_id
        # A checkpoint's outputs never answer the computation's own cache lookup.
        assert ledger.store.find_by_provenance(derive_subkey(PARENT, "model")) is None

    def test_a_checkpoint_bundle_resolves_in_one_read(self, ledger: Ledger):
        build_id = _build(ledger)
        _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)

        bundle = ledger.releases.bundle(second.release.release_id)
        assert bundle is not None
        assert bundle.release.sequence == 2
        assert sorted(v.id.rsplit("-", 1)[1] for v in bundle.outputs) == ["metrics", "model"]
        assert all(v.state == "ready" for v in bundle.outputs)
        assert all(ledger.store.read_blob(v.id, v.version) is not None for v in bundle.outputs)

    def test_recovery_after_a_crash_between_prepare_and_accept(self, ledger: Ledger):
        build_id = _build(ledger)
        _checkpoint(ledger, build_id)
        crashed = ledger.releases.prepare(build_id, "a1", "checkpoint")
        _stage(ledger, crashed, "model", b"w3")
        # The process dies here. A new attempt resumes the same execution.
        assert (
            ledger.releases.resume(
                build_id, "a2", writable_until=ledger.clock() + 3600, lease_owner="runner-2"
            )
            == 1
        )
        assert ledger.releases.get(crashed.release_id).state == "abandoned"
        assert ledger.builds.get_build(build_id).attempt == "a2"

        # The old attempt is fenced: it can neither prepare nor accept.
        with pytest.raises(ReleaseRejected):
            ledger.releases.prepare(build_id, "a1", "checkpoint")
        with pytest.raises(ValueError, match="abandoned"):
            ledger.releases.accept(crashed.release_id, [])
        # The new attempt continues the sequence; the earlier checkpoint stays valid.
        resumed = _checkpoint(ledger, build_id, "a2")
        assert resumed.release.sequence == 3
        (first,) = [r for r in ledger.releases.releases_of(build_id) if r.sequence == 1]
        assert first.state == "accepted"
        assert all(v.state == "ready" for v in ledger.releases.bundle(first.release_id).outputs)

    def test_a_stale_attempt_publishing_after_a_resume_is_rejected(self, ledger: Ledger):
        build_id = _build(ledger)
        stale = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, stale, "model", b"w1")]
        ledger.releases.resume(build_id, "a2", writable_until=ledger.clock() + 3600)
        # resume abandoned it; an attempt whose window merely closed is rejected at accept.
        assert ledger.releases.get(stale.release_id).state == "abandoned"

        late = ledger.releases.prepare(build_id, "a2", "checkpoint")
        staged = [_stage(ledger, late, "model", b"w2")]
        ledger.clock.now += 7200  # a2's write window closes
        with pytest.raises(ReleaseRejected):
            ledger.releases.accept(late.release_id, staged)
        assert ledger.releases.get(late.release_id).state == "rejected"
        assert _state(ledger, staged[0]) == "building"

    def test_gc_under_a_root_keeps_the_head_and_collects_the_rest(self, ledger: Ledger):
        build_id = _build(ledger)
        old = _checkpoint(ledger, build_id)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"best")]
        head = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/best", release.release_id)]
        )
        candidates = [(v.id, v.version) for v in old.outputs + head.outputs]

        result = ledger.store.garbage_collect(collectable=candidates, collect_latest=True)

        assert result["deleted_count"] == len(old.outputs)
        assert all(ledger.store.get_artifact(v.id, v.version) is None for v in old.outputs)
        assert all(_state(ledger, v) == "ready" for v in head.outputs)

    def test_a_crash_before_acceptance_leaves_an_orphan_and_no_authority(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, release, "model", b"all-written")]
        # Every byte is written; the process dies before accept.
        row = ledger.store.get_artifact(staged[0].artifact_id, staged[0].version)
        assert row.state == "building"
        assert ledger.store.read_blob(row.id, row.version) == b"all-written"
        assert ledger.releases.authority(f"slot:{build_id}") is None
        assert ledger.releases.get(release.release_id).state == "preparing"
        assert ledger.builds.get_build(build_id).state == "building"
        assert ledger.releases.abandon(release.release_id)

    def test_overlapping_releases_are_accepted_in_sequence_order(self, ledger: Ledger):
        build_id = _build(ledger)
        first = ledger.releases.prepare(build_id, "a1", "checkpoint")
        second = ledger.releases.prepare(build_id, "a1", "checkpoint")
        first_staged = [_stage(ledger, first, "model", b"w1")]
        second_staged = [_stage(ledger, second, "model", b"w2")]  # finishes uploading first

        with pytest.raises(ReleaseOutOfOrder):
            ledger.releases.accept(second.release_id, second_staged)
        assert ledger.releases.get(second.release_id).state == "preparing"
        ledger.releases.accept(first.release_id, first_staged)
        ledger.releases.accept(second.release_id, second_staged)
        assert [r.state for r in ledger.releases.releases_of(build_id, "checkpoint")] == [
            "accepted",
            "accepted",
        ]

    def test_head_updates_out_of_order_resolve_by_revision(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            release.release_id,
            [_stage(ledger, release, "model", b"w1")],
            [AdvanceReference("train/head", release.release_id)],
        )
        # Two updaters read revision 1; the slower one's assertion is stale.
        fast = ledger.releases.prepare(build_id, "a1", "checkpoint")
        slow = ledger.releases.prepare(build_id, "a1", "checkpoint")
        fast_staged = [_stage(ledger, fast, "model", b"w2")]
        slow_staged = [_stage(ledger, slow, "model", b"w3")]
        ledger.releases.accept(
            fast.release_id,
            fast_staged,
            [
                AssertReferenceRevision("train/head", 1),
                AdvanceReference("train/head", fast.release_id),
            ],
        )
        with pytest.raises(OperationFailed, match="revision 2, not 1"):
            ledger.releases.accept(
                slow.release_id,
                slow_staged,
                [
                    AssertReferenceRevision("train/head", 1),
                    AdvanceReference("train/head", slow.release_id),
                ],
            )
        assert ledger.releases.reference("train/head") == (fast.release_id, 2)
        assert ledger.releases.get(slow.release_id).state == "preparing"
        assert _state(ledger, slow_staged[0]) == "building"

    def test_the_object_store_path_accepts_only_after_a_manifest_check(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        unwritten = _stage(ledger, release, "model", b"not yet", write=False)
        with pytest.raises(ReleaseIncomplete, match="none stored"):
            ledger.releases.accept(release.release_id, [unwritten])
        assert ledger.releases.get(release.release_id).state == "preparing"

        ledger.store.write_blob(unwritten.artifact_id, unwritten.version, b"short")
        with pytest.raises(ReleaseIncomplete, match="7 bytes declared, 5 stored"):
            ledger.releases.accept(release.release_id, [unwritten])

        ledger.store.write_blob(unwritten.artifact_id, unwritten.version, b"not yet")
        assert ledger.releases.accept(release.release_id, [unwritten]).release.state == "accepted"

    # --- ADR 001's mandatory tests assigned to E4a ---

    def test_repeated_invocation_site_gives_distinct_occurrences(self, ledger: Ledger):
        a = _build(ledger, occurrence="variant=a")
        b = _build(ledger, occurrence="variant=b")
        assert a != b
        assert {ledger.builds.get_build(x).occurrence for x in (a, b)} == {"variant=a", "variant=b"}

    def test_multi_slot_authority_is_all_or_none(self, ledger: Ledger):
        taken = _build(ledger)
        winner = ledger.releases.prepare(taken, "a1", "final")
        ledger.releases.accept(
            winner.release_id,
            [_stage(ledger, winner, "model", b"w")],
            [AssignAuthority("slot:b", winner.release_id)],
        )
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, release, "model", b"w")]
        with pytest.raises(OperationFailed, match="slot:b already belongs"):
            ledger.releases.accept(
                release.release_id,
                staged,
                [
                    AssignAuthority("slot:a", release.release_id),
                    AssignAuthority("slot:b", release.release_id),
                ],
            )
        assert ledger.releases.authority("slot:a") is None
        assert ledger.releases.authority("slot:b") == winner.release_id
        assert ledger.releases.get(release.release_id).state == "preparing"
        assert ledger.builds.get_build(build_id).state == "building"

    def test_two_attempts_racing_for_a_slot_have_one_winner(self, ledger: Ledger):
        winners = []
        for _ in range(2):
            build_id = _build(ledger)
            release = ledger.releases.prepare(build_id, "a1", "final")
            staged = [_stage(ledger, release, "model", b"w")]
            try:
                ledger.releases.accept(
                    release.release_id, staged, [AssignAuthority("slot:race", release.release_id)]
                )
                winners.append(release.release_id)
            except OperationFailed:
                pass
        assert len(winners) == 1
        assert ledger.releases.authority("slot:race") == winners[0]

    def test_a_retry_after_a_lost_response_returns_the_original_release(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        staged = [_stage(ledger, release, "model", b"w1")]
        first = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/latest", release.release_id)]
        )
        # The reference moves on before the lost response is retried.
        later = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            later.release_id,
            [_stage(ledger, later, "model", b"w2")],
            [AdvanceReference("train/latest", later.release_id)],
        )

        retried = ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("train/latest", release.release_id)]
        )

        assert retried.release == first.release
        assert [v.id for v in retried.outputs] == [v.id for v in first.outputs]
        assert ledger.releases.reference("train/latest") == (later.release_id, 2)

    def test_an_alias_move_leaves_a_bound_run_on_its_exact_inputs(self, ledger: Ledger):
        build_id = _build(ledger)
        release = ledger.releases.prepare(build_id, "a1", "checkpoint")
        bound = {"strata://artifact/features@v=3": "features@v=3"}
        version = ledger.store.create_artifact(
            "bound-output",
            checkpoint_output_key(PARENT, build_id, 1, "model"),
            SPEC,
            input_versions=bound,
            release_id=release.release_id,
            minted=True,
        )
        ledger.store.write_blob("bound-output", version, b"w")
        staged = [_staged_row("bound-output", version, b"w")]
        ledger.releases.accept(
            release.release_id, staged, [AdvanceReference("features/latest", release.release_id)]
        )
        other = ledger.releases.prepare(build_id, "a1", "checkpoint")
        ledger.releases.accept(
            other.release_id,
            [_stage(ledger, other, "model", b"w2")],
            [AdvanceReference("features/latest", other.release_id)],
        )
        (output,) = ledger.releases.bundle(release.release_id).outputs
        assert output.input_versions is not None and "features@v=3" in output.input_versions

    def test_a_resume_binds_the_attempt_to_an_explicit_accepted_release(self, ledger: Ledger):
        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        unsettled = ledger.releases.prepare(build_id, "a1", "checkpoint")

        with pytest.raises(ValueError, match="not an accepted release"):
            ledger.releases.resume(
                build_id,
                "a2",
                writable_until=ledger.clock() + 3600,
                from_release=unsettled.release_id,
            )
        ledger.releases.resume(
            build_id,
            "a2",
            writable_until=ledger.clock() + 3600,
            from_release=first.release.release_id,
        )

        attempt = ledger.builds.get_attempt(build_id, "a2")
        assert attempt is not None and attempt.resumed_from == first.release.release_id
        assert ledger.builds.get_attempt(build_id, "a1").resumed_from is None
        assert ledger.releases.latest_accepted(build_id).release_id == first.release.release_id

    def test_a_computation_finds_its_open_build_until_the_final_release(self, ledger: Ledger):
        # Other tests leave builds of PARENT open in a shared store; this computation is its own.
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = _build(ledger, occurrence="variant=a", computation_key=key)
        assert ledger.builds.find_open_build(key) is None
        assert ledger.builds.find_open_build(key, occurrence="variant=a").build_id == build_id

        # A lapsed lease is still the computation's open build: what a resume continues.
        ledger.clock.now += 120
        assert ledger.builds.find_open_build(key, occurrence="variant=a").build_id == build_id

        ledger.releases.resume(build_id, "a2", writable_until=ledger.clock() + 3600)
        final = ledger.releases.prepare(build_id, "a2", "final")
        staged = [_stage(ledger, final, "model", b"done")]
        ledger.releases.accept(final.release_id, staged)
        assert ledger.builds.find_open_build(key, occurrence="variant=a") is None

    def test_the_final_release_retargets_the_build_and_fails_its_placeholder(self, ledger: Ledger):
        build_id = _build(ledger)
        placeholder = ledger.builds.get_build(build_id)
        assert _state(ledger, placeholder) == "building"

        final = ledger.releases.prepare(build_id, "a1", "final")
        staged = [_stage(ledger, final, "model", b"done"), _stage(ledger, final, "metrics", b"{}")]
        ledger.releases.accept(final.release_id, staged)

        build = ledger.builds.get_build(build_id)
        assert (build.artifact_id, build.version) == (staged[0].artifact_id, staged[0].version)
        assert build.state == "ready"
        assert (
            ledger.store.get_artifact(placeholder.artifact_id, placeholder.version).state
            == "failed"
        )

    def test_an_empty_final_release_ends_the_build_and_retires_its_placeholder(
        self, ledger: Ledger
    ):
        """A run that stores nothing still completes: nothing is left building."""
        key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
        build_id = _build(ledger, computation_key=key)
        placeholder = ledger.builds.get_build(build_id)

        final = ledger.releases.prepare(build_id, "a1", "final")
        ledger.releases.accept(final.release_id, [])

        build = ledger.builds.get_build(build_id)
        assert build.state == "ready"
        assert (build.artifact_id, build.version) == (placeholder.artifact_id, placeholder.version)
        assert _state(ledger, placeholder) == "failed"
        assert ledger.builds.find_open_build(key) is None

    def test_a_head_set_between_releases_is_fenced_on_the_attempt(self, ledger: Ledger):
        build_id = _build(ledger)
        best = _checkpoint(ledger, build_id)
        _checkpoint(ledger, build_id)

        ledger.releases.apply(
            build_id, "a1", [AdvanceReference("train/best", best.release.release_id)]
        )
        assert ledger.releases.reference("train/best") == (best.release.release_id, 1)

        # The new attempt may point a head at the old attempt's checkpoint; the old one may not.
        ledger.releases.resume(build_id, "a2", writable_until=ledger.clock() + 3600)
        with pytest.raises(ReleaseRejected):
            ledger.releases.apply(
                build_id, "a1", [AdvanceReference("train/best", best.release.release_id)]
            )
        ledger.releases.apply(
            build_id, "a2", [AdvanceReference("train/best", best.release.release_id)]
        )
        assert ledger.releases.reference("train/best") == (best.release.release_id, 2)
        with pytest.raises(OperationFailed):
            ledger.releases.apply(build_id, "a2", [AssertReferenceRevision("train/best", 1)])

    def test_collectable_checkpoints_spare_the_newest_of_an_open_build(self, ledger: Ledger):
        build_id = _build(ledger)
        first = _checkpoint(ledger, build_id)
        second = _checkpoint(ledger, build_id)

        def rows(bundle):
            return sorted((v.id, v.version) for v in bundle.outputs)

        assert ledger.releases.collectable_checkpoints() == rows(first)

        # Collectable is not collected: a root still holds what it names.
        ledger.releases.apply(
            build_id, "a1", [AdvanceReference("train/best", first.release.release_id)]
        )
        final = ledger.releases.prepare(build_id, "a1", "final")
        ledger.releases.accept(final.release_id, [_stage(ledger, final, "model", b"done")])
        collectable = ledger.releases.collectable_checkpoints()
        assert collectable == sorted(rows(first) + rows(second))

        result = ledger.store.garbage_collect(collectable=collectable, collect_latest=True)
        assert result["deleted_count"] == len(second.outputs)
        assert all(_state(ledger, v) == "ready" for v in first.outputs)
        assert all(ledger.store.get_artifact(v.id, v.version) is None for v in second.outputs)

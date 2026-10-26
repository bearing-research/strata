# Formal models

TLA+ models of the two lifecycles this package implements. The lifecycle
model describes the code as it is: a finalize that finds its provenance held
under another id supersedes its own row with a pointer at the holder, and a
sweep takes a canonical only once nothing points at it. The lease model keeps
a `Patched` constant: `FALSE` is the behaviour the model found wrong in the
first consumer, `TRUE` the code as it is. They moved here with the code they
model (ADR 002).

| File | What it checks |
| --- | --- |
| `tla/ArtifactLifecycle.tla` | `artifact_versions`: create, write blob, fail, finalize (own id, refresh, foreign id through a pointer), finalize together, retain, release, one row of a sweep |
| `tla/Artifact_GCRebuild.cfg` | One id with GC on: a sweep during a rebuild leaves the current value in place (1,868 states) |
| `tla/Artifact_CrossIdDedup.cfg` | Two ids sharing a provenance, GC off: the loser reads the winner's bytes through its pointer (1,495 states) |
| `tla/Artifact_Patched.cfg` | Both ids, GC on, pointers and sweeps together. All invariants, `PointersResolve` among them |
| `tla/Artifact_Retention.cfg` | With retention roots and required edges (`MaxVer = 2`). All invariants and `HeldIsKept` |
| `tla/BuildLease.tla` | One build and its lease: claim, reclaim, open an attempt, heartbeat, checkpoint, publish, finalize, complete, fail, and the pull-path routes |
| `tla/Build_RunnerPath.cfg` | Two runners, lease can expire mid-build, unpatched. Violation: a lapsed runner finalizes |
| `tla/Build_PullPath.cfg` | Two executors fetch the same build's manifest, unpatched. Violation: both upload |
| `tla/Build_Patched.cfg` | Runners and executors as the code is: attempts rotate the fence, a claim or reclaim clears it, a lapsed lease fences its attempt (`OnlyLeaseHolderReleases`). All invariants hold |

The state counts above are from the last TLC run of that configuration; the
other configurations have not been re-run since the models were rewritten to
match the code (2026-10-26) and should be before the next release.

```bash
curl -sSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
TLC="java -cp $PWD/tla2tools.jar tlc2.TLC -deadlock -workers auto"
cd formal/tla
$TLC -config Artifact_GCRebuild.cfg    ArtifactLifecycle.tla   # passes
$TLC -config Artifact_CrossIdDedup.cfg ArtifactLifecycle.tla   # passes
$TLC -config Artifact_Patched.cfg      ArtifactLifecycle.tla   # passes
$TLC -config Artifact_Retention.cfg    ArtifactLifecycle.tla   # passes
$TLC -config Build_RunnerPath.cfg      BuildLease.tla          # violation
$TLC -config Build_PullPath.cfg        BuildLease.tla          # violation
$TLC -config Build_Patched.cfg         BuildLease.tla          # passes
```

`-deadlock` disables deadlock checking: a bounded model that has used every
version or lease epoch has no enabled action, which is expected. TLC stops at
the first violated invariant; to see a particular one, list only it in the
config's `INVARIANTS` line. TLC writes `states/` beside the models; it is
ignored, and must not be deleted while a check runs.

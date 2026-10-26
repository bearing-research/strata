# Formal models

TLA+ models of the two lifecycles this package implements, with the
configurations that found their bugs (in Strata, before the extraction) and
the patched configurations that now pass. They moved here with the code they
model (ADR 002).

| File | What it checks |
| --- | --- |
| `tla/ArtifactLifecycle.tla` | `artifact_versions`: create, write blob, fail, finalize, force-finalize, garbage collect |
| `tla/Artifact_GCRebuild.cfg` | One id with GC on. Violation: GC collected a version being rebuilt |
| `tla/Artifact_CrossIdDedup.cfg` | Two ids sharing a provenance, GC off. Violation: cross-id dedup served a failed row |
| `tla/Artifact_Patched.cfg` | Both ids, GC on, with the fix. All invariants hold (5,997,420 states, about 2 min) |
| `tla/Artifact_Retention.cfg` | Patched, with retention roots and required edges (`MaxVer = 2`). All invariants and `HeldIsKept` hold (3,597,345 states, about 4 min) |
| `tla/BuildLease.tla` | One build and its lease: claim, reclaim, heartbeat, publish, finalize, complete, fail, and the pull-path routes |
| `tla/Build_RunnerPath.cfg` | Two runners, lease can expire mid-build. Violation: a lapsed runner finalizes |
| `tla/Build_PullPath.cfg` | Two executors fetch the same build's manifest. Violation: both upload |
| `tla/Build_Patched.cfg` | Runners and executors with the fix, and checkpoint releases fenced on the lease (`OnlyLeaseHolderReleases`). All invariants hold |

```bash
curl -sSLO https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar
TLC="java -cp $PWD/tla2tools.jar tlc2.TLC -deadlock -workers auto"
cd formal/tla
$TLC -config Artifact_GCRebuild.cfg    ArtifactLifecycle.tla   # violation
$TLC -config Artifact_CrossIdDedup.cfg ArtifactLifecycle.tla   # violation
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

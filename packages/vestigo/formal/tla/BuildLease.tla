------------------------------- MODULE BuildLease -------------------------------
(***************************************************************************)
(* Model of one build (BuildStore) and the lease that fences it.           *)
(*                                                                         *)
(* Two ways a build gets executed, each with its own actors:              *)
(*                                                                         *)
(*   Runners   : a consumer's in-process runner. Claims with claim_build   *)
(*               or reclaim_expired_build, renews with renew_lease, writes *)
(*               the blob, finalizes the artifact fenced on its exact      *)
(*               claim (a BuildFence), then complete_build(lease_owner).   *)
(*               On error: fail_build plus fail_artifact. A runner also    *)
(*               stands for a release's attempt: open_attempt (or          *)
(*               ReleaseStore.resume) makes it the build's current one.    *)
(*                                                                         *)
(*   Executors : a consumer's pull protocol for external executors. The    *)
(*               manifest claims or renews the lease as the shared owner   *)
(*               "ext" and hands out a signed upload URL plus a lease      *)
(*               token; upload writes the blob (signature checked, build   *)
(*               still active); finalize checks the token, promotes the    *)
(*               artifact, then complete_build(lease_owner = "ext").       *)
(*                                                                         *)
(* A lease token is modelled as the lease epoch: every claim, reclaim,     *)
(* opened attempt and manifest renewal changes the lease, so it bumps the  *)
(* epoch. The attempt column is modelled by the epoch too: open_attempt    *)
(* rotates it, and claim_build and reclaim_expired_build clear it, which   *)
(* fences the lapsed attempt exactly as a bump does. Blob bytes are        *)
(* modelled as "who wrote them". Two attempts of a nondeterministic        *)
(* transform (or one reading a moving input) write different bytes; for a  *)
(* byte-deterministic transform some violations below are harmless.       *)
(*                                                                         *)
(* Time is abstracted: a runner's lease may expire at any moment (a GC     *)
(* pause, a blocked event loop, a DB outage longer than the lease), and a  *)
(* signed URL stays valid for an arbitrary time after it is minted.        *)
(* Executor lease expiry is not modelled.                                  *)
(*                                                                         *)
(* Releases (ADR 002 section 7.2): while a build runs, its attempt may     *)
(* accept any number of checkpoint releases (ReleaseStore.accept).         *)
(* Patched, every acceptance is fenced as lease_held is: the attempt must  *)
(* be the build's current one and its lease must not have lapsed; a        *)
(* release by any other attempt is rejected. Unpatched, nothing fences a   *)
(* checkpoint, and a fenced attempt publishes over the attempt that took   *)
(* over.                                                                   *)
(*                                                                         *)
(* Patched = FALSE is the lifecycle before the fixes the violating         *)
(* configurations (Build_RunnerPath, Build_PullPath) found; TRUE is the    *)
(* code as it is (Build_Patched).                                          *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    Runners,      \* in-process BuildRunner instances (e.g. two server nodes)
    Executors,    \* external pull-model executors
    MaxEpoch,     \* bound on lease generations
    Patched       \* FALSE = before the fixes; TRUE = the code as it is

None   == "none"
Ext    == "ext"                \* _EXTERNAL_LEASE_OWNER: shared by every executor
Actors == Runners \union Executors
Active == {"pending", "building"}

\* Unpatched, every attempt writes the one blob key (artifact_id, version).
\* Patched, each lease epoch writes its own key.
Slot(t) == IF Patched THEN t ELSE 0
Slots   == 0..MaxEpoch

VARIABLES
    build,      \* artifact_builds.state
    owner,      \* artifact_builds.lease_owner
    epoch,      \* lease generation (stands in for lease_expires_at)
    expired,    \* the current runner lease has expired
    art,        \* artifact_versions.state of the build's (artifact_id, version)
    blobs,      \* blobs[s] : who wrote the bytes in slot s (None = absent)
    chosen,     \* slot the ready artifact reads from
    digest,     \* whose bytes finalize recorded (content_sha256)
    pc,         \* per-actor program counter
    tok,        \* tok[a] : the epoch actor a was granted
    url,        \* url[e] : executor e still holds a valid signed upload URL
    staleFail,  \* history: a build/artifact was failed by an actor without the lease
    staleBytes, \* history: finalize published bytes written by a different attempt
    staleRelease \* history: a checkpoint was accepted from an attempt without the lease

vars == <<build, owner, epoch, expired, art, blobs, chosen, digest,
          pc, tok, url, staleFail, staleBytes, staleRelease>>

TypeOK ==
    /\ build \in {"pending", "building", "ready", "failed"}
    /\ owner \in Runners \union {Ext, None}
    /\ epoch \in 0..MaxEpoch
    /\ expired \in BOOLEAN
    /\ art \in {"building", "ready", "failed"}
    /\ blobs \in [Slots -> Actors \union {None}]
    /\ chosen \in Slots
    /\ digest \in Actors \union {None}
    /\ pc \in [Actors -> {"idle", "exec", "published", "finalized", "failing", "done"}]
    /\ tok \in [Actors -> 0..MaxEpoch]
    /\ url \in [Executors -> BOOLEAN]
    /\ staleFail \in BOOLEAN
    /\ staleBytes \in BOOLEAN
    /\ staleRelease \in BOOLEAN

Init ==
    /\ build = "pending"
    /\ owner = None
    /\ epoch = 0
    /\ expired = FALSE
    /\ art = "building"          \* materialize creates the artifact row with the build
    /\ blobs = [s \in Slots |-> None]
    /\ chosen = 0
    /\ digest = None
    /\ pc = [a \in Actors |-> "idle"]
    /\ tok = [a \in Actors |-> 0]
    /\ url = [e \in Executors |-> FALSE]
    /\ staleFail = FALSE
    /\ staleBytes = FALSE
    /\ staleRelease = FALSE

HoldsLease(a, o) == owner = o /\ tok[a] = epoch

\* lease_held: a is the build's current attempt and its lease has not lapsed.
Fenced(a) == tok[a] = epoch /\ ~expired

\* finalize_artifact / finalize_and_set_name on the artifact row, as seen by
\* attempt a: promote with a's slot, or no-op if already ready.
PromoteArtifact(a) ==
    IF art = "building"
    THEN /\ art' = "ready"
         /\ chosen' = Slot(tok[a])
         /\ digest' = blobs[Slot(tok[a])]
         /\ staleBytes' = (staleBytes \/ blobs[Slot(tok[a])] # a)
    ELSE UNCHANGED <<art, chosen, digest, staleBytes, staleRelease>>

-----------------------------------------------------------------------------
(* Runners                                                                 *)

Claim(r) ==                                   \* claim_build
    /\ pc[r] = "idle" /\ build = "pending" /\ epoch < MaxEpoch
    /\ build' = "building" /\ owner' = r /\ epoch' = epoch + 1 /\ expired' = FALSE
    /\ tok' = [tok EXCEPT ![r] = epoch + 1]
    /\ pc' = [pc EXCEPT ![r] = "exec"]
    /\ UNCHANGED <<art, blobs, chosen, digest, url, staleFail, staleBytes, staleRelease>>

\* open_attempt / ReleaseStore.resume: r becomes the build's attempt, fencing
\* every earlier one. Patched, a live lease is taken over by no other attempt
\* (open_attempt_within refuses it); unpatched, nothing checks the lease.
OpenAttempt(r) ==
    /\ pc[r] = "idle" /\ build \in Active /\ epoch < MaxEpoch
    /\ Patched => (owner = None \/ expired)
    /\ build' = "building" /\ owner' = r /\ epoch' = epoch + 1 /\ expired' = FALSE
    /\ tok' = [tok EXCEPT ![r] = epoch + 1]
    /\ pc' = [pc EXCEPT ![r] = "exec"]
    /\ UNCHANGED <<art, blobs, chosen, digest, url, staleFail, staleBytes, staleRelease>>

Expire ==                                     \* heartbeat missed for a whole lease
    /\ build = "building" /\ owner \in Runners /\ ~expired
    /\ expired' = TRUE
    /\ UNCHANGED <<build, owner, epoch, art, blobs, chosen, digest, pc, tok, url,
                   staleFail, staleBytes, staleRelease>>

Renew(r) ==                                   \* renew_lease: no expiry check in the WHERE
    /\ pc[r] \in {"exec", "published", "finalized"}
    /\ build = "building" /\ owner = r /\ expired
    /\ expired' = FALSE
    /\ UNCHANGED <<build, owner, epoch, art, blobs, chosen, digest, pc, tok, url,
                   staleFail, staleBytes, staleRelease>>

Reclaim(r) ==                                 \* list_expired_leases + reclaim_expired_build
    /\ pc[r] = "idle" /\ build = "building" /\ expired
    /\ owner \in Runners /\ owner # r /\ epoch < MaxEpoch
    /\ owner' = r /\ epoch' = epoch + 1 /\ expired' = FALSE
    /\ tok' = [tok EXCEPT ![r] = epoch + 1]
    /\ pc' = [pc EXCEPT ![r] = "exec"]
    /\ UNCHANGED <<build, art, blobs, chosen, digest, url, staleFail, staleBytes, staleRelease>>

Publish(r) ==                                 \* publish_blob_from_path: no lease check
    /\ pc[r] = "exec"
    /\ blobs' = [blobs EXCEPT ![Slot(tok[r])] = r]
    /\ pc' = [pc EXCEPT ![r] = "published"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, chosen, digest, tok, url,
                   staleFail, staleBytes, staleRelease>>

\* Unpatched: finalize_artifact (unfenced), later complete_build (fenced).
\* Patched: one transaction, fenced on the lease, that promotes the artifact
\* from this attempt's slot and completes the build, or discards.
RunnerFinalize(r) ==
    /\ pc[r] = "published"
    /\ IF Patched
       THEN IF HoldsLease(r, r) /\ build = "building"
            THEN /\ PromoteArtifact(r)
                 /\ build' = "ready"
                 /\ pc' = [pc EXCEPT ![r] = "done"]
            ELSE /\ pc' = [pc EXCEPT ![r] = "done"]
                 /\ UNCHANGED <<build, art, chosen, digest, staleBytes, staleRelease>>
       ELSE IF art = "failed"                 \* ValueError: not in building state
            THEN /\ pc' = [pc EXCEPT ![r] = "failing"]
                 /\ UNCHANGED <<build, art, chosen, digest, staleBytes, staleRelease>>
            ELSE /\ PromoteArtifact(r)
                 /\ pc' = [pc EXCEPT ![r] = "finalized"]
                 /\ UNCHANGED build
    /\ UNCHANGED <<owner, epoch, expired, blobs, tok, url, staleFail, staleRelease>>

RunnerComplete(r) ==                          \* complete_build(lease_owner=r)
    /\ pc[r] = "finalized"
    /\ build' = IF build = "building" /\ owner = r THEN "ready" ELSE build
    /\ pc' = [pc EXCEPT ![r] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes, staleRelease>>

RunnerError(r) ==                             \* executor error / timeout / bad output
    /\ pc[r] \in {"exec", "published"}
    /\ pc' = [pc EXCEPT ![r] = "failing"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes, staleRelease>>

\* fail_build + fail_artifact. Unpatched, neither checks the lease.
RunnerFail(r) ==
    /\ pc[r] = "failing"
    /\ LET may == ~Patched \/ HoldsLease(r, r) IN
       /\ build' = IF may /\ build \in Active THEN "failed" ELSE build
       /\ art' = IF may /\ art = "building" THEN "failed" ELSE art
       /\ staleFail' = (staleFail \/
                        ((build' # build \/ art' # art) /\ ~HoldsLease(r, r)))
    /\ pc' = [pc EXCEPT ![r] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, blobs, chosen, digest, tok, url, staleBytes, staleRelease>>

-----------------------------------------------------------------------------
(* Pull-model executors                                                    *)

Manifest(e) ==                                \* GET /v1/builds/{id}/manifest
    /\ pc[e] = "idle" /\ build \in Active /\ owner \in {None, Ext} /\ epoch < MaxEpoch
    /\ build' = "building" /\ owner' = Ext /\ epoch' = epoch + 1
    /\ tok' = [tok EXCEPT ![e] = epoch + 1]
    /\ url' = [url EXCEPT ![e] = TRUE]
    /\ pc' = [pc EXCEPT ![e] = "exec"]
    /\ UNCHANGED <<expired, art, blobs, chosen, digest, staleFail, staleBytes, staleRelease>>

UrlExpire(e) ==
    /\ url[e]
    /\ url' = [url EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<build, owner, epoch, expired, art, blobs, chosen, digest, pc, tok,
                   staleFail, staleBytes, staleRelease>>

Upload(e) ==                                  \* POST /v1/artifacts/upload: no lease check
    /\ pc[e] = "exec" /\ url[e] /\ build \in Active
    /\ blobs' = [blobs EXCEPT ![Slot(tok[e])] = e]
    /\ pc' = [pc EXCEPT ![e] = "published"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, chosen, digest, tok, url,
                   staleFail, staleBytes, staleRelease>>

ExecFinalize(e) ==                            \* POST finalize: token check, then finalize_and_set_name
    /\ pc[e] = "published" /\ build \in Active
    /\ IF tok[e] = epoch /\ blobs[Slot(tok[e])] # None
       THEN /\ PromoteArtifact(e)
            /\ pc' = [pc EXCEPT ![e] = "finalized"]
       ELSE /\ pc' = [pc EXCEPT ![e] = "done"]    \* 409, nothing published
            /\ UNCHANGED <<art, chosen, digest, staleBytes, staleRelease>>
    /\ UNCHANGED <<build, owner, epoch, expired, blobs, tok, url, staleFail, staleRelease>>

ExecComplete(e) ==                            \* complete_build(lease_owner="ext")
    /\ pc[e] = "finalized"
    /\ build' = IF build = "building" /\ owner = Ext THEN "ready" ELSE build
    /\ pc' = [pc EXCEPT ![e] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes, staleRelease>>

-----------------------------------------------------------------------------
(* Releases                                                                *)

\* ReleaseStore.accept of a checkpoint by attempt a, owner o, while the build
\* runs. Patched, acceptance requires the fence (lease_held); unpatched, the
\* checkpoint is accepted whoever holds it. The build stays building either way.
Checkpoint(a, o) ==
    /\ pc[a] \in {"exec", "published"} /\ build = "building"
    /\ LET accepted == ~Patched \/ Fenced(a) IN
       staleRelease' = (staleRelease \/ (accepted /\ ~HoldsLease(a, o)))
    /\ UNCHANGED <<build, owner, epoch, expired, art, blobs, chosen, digest, pc, tok, url,
                   staleFail, staleBytes>>

-----------------------------------------------------------------------------
Next ==
    \/ Expire
    \/ \E r \in Runners :
        \/ Claim(r) \/ OpenAttempt(r) \/ Renew(r) \/ Reclaim(r) \/ Publish(r) \/ Checkpoint(r, r)
        \/ RunnerFinalize(r) \/ RunnerComplete(r) \/ RunnerError(r) \/ RunnerFail(r)
    \/ \E e \in Executors :
        \/ Manifest(e) \/ UrlExpire(e) \/ Upload(e) \/ ExecFinalize(e) \/ ExecComplete(e)
        \/ Checkpoint(e, Ext)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)

\* A ready artifact is immutable: the bytes readers get are the bytes whose
\* digest finalize recorded (verify_artifacts' digest_mismatch otherwise).
ReadyBytesStable ==
    art = "ready" => blobs[chosen] = digest

\* Only the attempt holding the lease may fail the build or its artifact.
\* Otherwise a stale attempt's timeout kills the attempt that took over.
OnlyLeaseHolderFails == ~staleFail

\* The bytes published are the bytes of the attempt that passed the fence.
NoStaleBytesPublished == ~staleBytes

\* Only the attempt holding the lease may accept a checkpoint release.
OnlyLeaseHolderReleases == ~staleRelease

\* Sanity (holds): a completed build has a ready artifact.
ReadyBuildHasArtifact == build = "ready" => art = "ready"
=============================================================================

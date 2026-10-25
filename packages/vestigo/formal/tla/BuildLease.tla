------------------------------- MODULE BuildLease -------------------------------
(***************************************************************************)
(* Model of one transform build and the lease that fences it.             *)
(*                                                                         *)
(* Two ways a build gets executed, each with its own actors:              *)
(*                                                                         *)
(*   Runners   : BuildRunner._execute_build (transforms/runner.py). Claims *)
(*               with claim_build / reclaim_expired_build, renews through  *)
(*               the heartbeat loop, writes the blob with                  *)
(*               publish_blob_from_path, then finalize_artifact, then      *)
(*               complete_build(lease_owner=me). On error: fail_build +    *)
(*               fail_artifact.                                            *)
(*                                                                         *)
(*   Executors : the v2 pull protocol (api/routers/builds.py). GET         *)
(*               manifest claims or renews the lease as owner "ext" and    *)
(*               mints signed URLs plus a lease token; POST upload writes  *)
(*               the blob (checked: signature, build still active); POST   *)
(*               finalize checks the lease token, runs                     *)
(*               finalize_and_set_name, then complete_build(owner="ext").  *)
(*                                                                         *)
(* A lease token is modelled as the lease epoch: every claim, reclaim and  *)
(* manifest renewal changes (lease_owner, lease_expires_at), so it bumps   *)
(* the epoch. Blob bytes are modelled as "who wrote them". Two attempts of *)
(* a nondeterministic transform (or one reading a moving input) write      *)
(* different bytes; for a byte-deterministic transform some violations     *)
(* below are harmless (see formal/README.md).                              *)
(*                                                                         *)
(* Time is abstracted: a runner's lease may expire at any moment (a GC     *)
(* pause, a blocked event loop, a DB outage longer than the lease), and a  *)
(* signed URL stays valid for an arbitrary time after it is minted.        *)
(* Executor lease expiry is not modelled.                                  *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    Runners,      \* in-process BuildRunner instances (e.g. two server nodes)
    Executors,    \* external pull-model executors
    MaxEpoch,     \* bound on lease generations
    Patched       \* FALSE = the code as it is; TRUE = the proposed fix

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
    staleBytes  \* history: finalize published bytes written by a different attempt

vars == <<build, owner, epoch, expired, art, blobs, chosen, digest,
          pc, tok, url, staleFail, staleBytes>>

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

HoldsLease(a, o) == owner = o /\ tok[a] = epoch

\* finalize_artifact / finalize_and_set_name on the artifact row, as seen by
\* attempt a: promote with a's slot, or no-op if already ready.
PromoteArtifact(a) ==
    IF art = "building"
    THEN /\ art' = "ready"
         /\ chosen' = Slot(tok[a])
         /\ digest' = blobs[Slot(tok[a])]
         /\ staleBytes' = (staleBytes \/ blobs[Slot(tok[a])] # a)
    ELSE UNCHANGED <<art, chosen, digest, staleBytes>>

-----------------------------------------------------------------------------
(* Runners                                                                 *)

Claim(r) ==                                   \* claim_build
    /\ pc[r] = "idle" /\ build = "pending" /\ epoch < MaxEpoch
    /\ build' = "building" /\ owner' = r /\ epoch' = epoch + 1 /\ expired' = FALSE
    /\ tok' = [tok EXCEPT ![r] = epoch + 1]
    /\ pc' = [pc EXCEPT ![r] = "exec"]
    /\ UNCHANGED <<art, blobs, chosen, digest, url, staleFail, staleBytes>>

Expire ==                                     \* heartbeat missed for a whole lease
    /\ build = "building" /\ owner \in Runners /\ ~expired
    /\ expired' = TRUE
    /\ UNCHANGED <<build, owner, epoch, art, blobs, chosen, digest, pc, tok, url,
                   staleFail, staleBytes>>

Renew(r) ==                                   \* renew_lease: no expiry check in the WHERE
    /\ pc[r] \in {"exec", "published", "finalized"}
    /\ build = "building" /\ owner = r /\ expired
    /\ expired' = FALSE
    /\ UNCHANGED <<build, owner, epoch, art, blobs, chosen, digest, pc, tok, url,
                   staleFail, staleBytes>>

Reclaim(r) ==                                 \* list_expired_leases + reclaim_expired_build
    /\ pc[r] = "idle" /\ build = "building" /\ expired
    /\ owner \in Runners /\ owner # r /\ epoch < MaxEpoch
    /\ owner' = r /\ epoch' = epoch + 1 /\ expired' = FALSE
    /\ tok' = [tok EXCEPT ![r] = epoch + 1]
    /\ pc' = [pc EXCEPT ![r] = "exec"]
    /\ UNCHANGED <<build, art, blobs, chosen, digest, url, staleFail, staleBytes>>

Publish(r) ==                                 \* publish_blob_from_path: no lease check
    /\ pc[r] = "exec"
    /\ blobs' = [blobs EXCEPT ![Slot(tok[r])] = r]
    /\ pc' = [pc EXCEPT ![r] = "published"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, chosen, digest, tok, url,
                   staleFail, staleBytes>>

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
                 /\ UNCHANGED <<build, art, chosen, digest, staleBytes>>
       ELSE IF art = "failed"                 \* ValueError: not in building state
            THEN /\ pc' = [pc EXCEPT ![r] = "failing"]
                 /\ UNCHANGED <<build, art, chosen, digest, staleBytes>>
            ELSE /\ PromoteArtifact(r)
                 /\ pc' = [pc EXCEPT ![r] = "finalized"]
                 /\ UNCHANGED build
    /\ UNCHANGED <<owner, epoch, expired, blobs, tok, url, staleFail>>

RunnerComplete(r) ==                          \* complete_build(lease_owner=r)
    /\ pc[r] = "finalized"
    /\ build' = IF build = "building" /\ owner = r THEN "ready" ELSE build
    /\ pc' = [pc EXCEPT ![r] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes>>

RunnerError(r) ==                             \* executor error / timeout / bad output
    /\ pc[r] \in {"exec", "published"}
    /\ pc' = [pc EXCEPT ![r] = "failing"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes>>

\* fail_build + fail_artifact. Unpatched, neither checks the lease.
RunnerFail(r) ==
    /\ pc[r] = "failing"
    /\ LET may == ~Patched \/ HoldsLease(r, r) IN
       /\ build' = IF may /\ build \in Active THEN "failed" ELSE build
       /\ art' = IF may /\ art = "building" THEN "failed" ELSE art
       /\ staleFail' = (staleFail \/
                        ((build' # build \/ art' # art) /\ ~HoldsLease(r, r)))
    /\ pc' = [pc EXCEPT ![r] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, blobs, chosen, digest, tok, url, staleBytes>>

-----------------------------------------------------------------------------
(* Pull-model executors                                                    *)

Manifest(e) ==                                \* GET /v1/builds/{id}/manifest
    /\ pc[e] = "idle" /\ build \in Active /\ owner \in {None, Ext} /\ epoch < MaxEpoch
    /\ build' = "building" /\ owner' = Ext /\ epoch' = epoch + 1
    /\ tok' = [tok EXCEPT ![e] = epoch + 1]
    /\ url' = [url EXCEPT ![e] = TRUE]
    /\ pc' = [pc EXCEPT ![e] = "exec"]
    /\ UNCHANGED <<expired, art, blobs, chosen, digest, staleFail, staleBytes>>

UrlExpire(e) ==
    /\ url[e]
    /\ url' = [url EXCEPT ![e] = FALSE]
    /\ UNCHANGED <<build, owner, epoch, expired, art, blobs, chosen, digest, pc, tok,
                   staleFail, staleBytes>>

Upload(e) ==                                  \* POST /v1/artifacts/upload: no lease check
    /\ pc[e] = "exec" /\ url[e] /\ build \in Active
    /\ blobs' = [blobs EXCEPT ![Slot(tok[e])] = e]
    /\ pc' = [pc EXCEPT ![e] = "published"]
    /\ UNCHANGED <<build, owner, epoch, expired, art, chosen, digest, tok, url,
                   staleFail, staleBytes>>

ExecFinalize(e) ==                            \* POST finalize: token check, then finalize_and_set_name
    /\ pc[e] = "published" /\ build \in Active
    /\ IF tok[e] = epoch /\ blobs[Slot(tok[e])] # None
       THEN /\ PromoteArtifact(e)
            /\ pc' = [pc EXCEPT ![e] = "finalized"]
       ELSE /\ pc' = [pc EXCEPT ![e] = "done"]    \* 409, nothing published
            /\ UNCHANGED <<art, chosen, digest, staleBytes>>
    /\ UNCHANGED <<build, owner, epoch, expired, blobs, tok, url, staleFail>>

ExecComplete(e) ==                            \* complete_build(lease_owner="ext")
    /\ pc[e] = "finalized"
    /\ build' = IF build = "building" /\ owner = Ext THEN "ready" ELSE build
    /\ pc' = [pc EXCEPT ![e] = "done"]
    /\ UNCHANGED <<owner, epoch, expired, art, blobs, chosen, digest, tok, url,
                   staleFail, staleBytes>>

-----------------------------------------------------------------------------
Next ==
    \/ Expire
    \/ \E r \in Runners :
        \/ Claim(r) \/ Renew(r) \/ Reclaim(r) \/ Publish(r)
        \/ RunnerFinalize(r) \/ RunnerComplete(r) \/ RunnerError(r) \/ RunnerFail(r)
    \/ \E e \in Executors :
        \/ Manifest(e) \/ UrlExpire(e) \/ Upload(e) \/ ExecFinalize(e) \/ ExecComplete(e)

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

\* Sanity (holds): a completed build has a ready artifact.
ReadyBuildHasArtifact == build = "ready" => art = "ready"
=============================================================================

-------------------------------- MODULE Streams --------------------------------
(***************************************************************************)
(* Model of one stream-mode materialize miss and the GETs that read it.   *)
(*                                                                         *)
(* The miss registers the stream, its placeholder artifact (building) and *)
(* a cleanup timer. A GET cancels the timer, waits for admission (a slot  *)
(* that other scans hold until Free, plus the per-client cap and the      *)
(* queue timeout, abstracted to Reject firing whenever a GET waits), and  *)
(* once admitted starts the background build if none has started. The     *)
(* build finalizes the artifact and re-arms the timer; the GET then       *)
(* serves the artifact and re-arms it again. Every GET that leaves the    *)
(* queue without admission (429, cancel) re-arms it too.                  *)
(*                                                                         *)
(* The timer fires after the TTL, which a config can set to anything, so  *)
(* Fire may happen whenever the timer is armed. Expiry drops the stream   *)
(* and fails the placeholder if no build has started. A finalize of a     *)
(* failed artifact fails, and the GET answers 500.                         *)
(*                                                                         *)
(* Patched = TRUE models the fix: a GET counts itself in the stream's     *)
(* `waiting` from cancelling the timer until admission settles, and Fire  *)
(* leaves a stream with a waiting GET alone.                              *)
(*                                                                         *)
(* Left out: the passthrough path (no artifact store), where expiry under *)
(* a serving GET only drops what the GET already holds; the prefetch;     *)
(* multi-node ownership. Stream ids are fresh uuids, so a second miss is  *)
(* a second, independent instance of this model (STR-2).                  *)
(*                                                                         *)
(* Mirrors:                                                                *)
(*   strata/api/routers/streams.py::get_stream                             *)
(*   strata/api/routers/materialize.py::_handle_identity_materialize       *)
(*   strata/streaming/registry.py::StreamRegistry.schedule_cleanup         *)
(*   strata/streaming/registry.py::StreamRegistry.cancel_cleanup           *)
(*   strata/streaming/scan_builds.py::ScanBuildManager.build_identity_artifact *)
(*   strata/server.py::ServerState._fail_unbuilt_stream                    *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    Gets,     \* concurrent GETs of the one stream
    Patched   \* TRUE = expiry skips a stream with a GET waiting for admission

VARIABLES
    live,     \* the stream is registered (and its scan with it)
    timer,    \* "armed" | "off": the pending cleanup task
    art,      \* the placeholder artifact: "building" | "failed" | "ready"
    build,    \* "none" | "running" | "done": the background build task
    slots,    \* free admission slots
    freed,    \* the other scans have released their slot
    waiting,  \* StreamState.waiting
    pc,       \* pc[g] : "idle" | "waiting" | "building" | "done"
    result    \* result[g] : "none" | "404" | "429" | "ok" | "500"

vars == <<live, timer, art, build, slots, freed, waiting, pc, result>>

TypeOK ==
    /\ live \in BOOLEAN
    /\ timer \in {"armed", "off"}
    /\ art \in {"building", "failed", "ready"}
    /\ build \in {"none", "running", "done"}
    /\ slots \in 0..1
    /\ freed \in BOOLEAN
    /\ waiting \in Nat
    /\ pc \in [Gets -> {"idle", "waiting", "building", "done"}]
    /\ result \in [Gets -> {"none", "404", "429", "ok", "500"}]

\* The materialize miss has returned the stream URL.
Init ==
    /\ live = TRUE
    /\ timer = "armed"
    /\ art = "building"
    /\ build = "none"
    /\ slots = 0
    /\ freed = FALSE
    /\ waiting = 0
    /\ pc = [g \in Gets |-> "idle"]
    /\ result = [g \in Gets |-> "none"]

\* get_stream up to the admit: look the stream up, cancel its cleanup.
Start(g) ==
    /\ pc[g] = "idle"
    /\ IF live
       THEN /\ timer' = "off"
            /\ waiting' = waiting + 1
            /\ pc' = [pc EXCEPT ![g] = "waiting"]
            /\ UNCHANGED result
       ELSE /\ pc' = [pc EXCEPT ![g] = "done"]
            /\ result' = [result EXCEPT ![g] = "404"]
            /\ UNCHANGED <<timer, waiting>>
    /\ UNCHANGED <<live, art, build, slots, freed>>

\* QoSRejected (per-client cap, queue timeout) or a cancel while queued.
Reject(g) ==
    /\ pc[g] = "waiting"
    /\ waiting' = waiting - 1
    /\ timer' = "armed"
    /\ pc' = [pc EXCEPT ![g] = "done"]
    /\ result' = [result EXCEPT ![g] = "429"]
    /\ UNCHANGED <<live, art, build, slots, freed>>

\* Admitted; the build starts with no await in between.
Admit(g) ==
    /\ pc[g] = "waiting"
    /\ slots > 0
    /\ slots' = slots - 1
    /\ waiting' = waiting - 1
    /\ build' = IF build = "none" THEN "running" ELSE build
    /\ pc' = [pc EXCEPT ![g] = "building"]
    /\ UNCHANGED <<live, timer, art, freed, result>>

\* build_identity_artifact: finalize (refused unless building), then re-arm.
Finish ==
    /\ build = "running"
    /\ build' = "done"
    /\ art' = IF art = "building" THEN "ready" ELSE art
    /\ timer' = "armed"
    /\ UNCHANGED <<live, slots, freed, waiting, pc, result>>

\* The GET's finally after the shielded build: release the slot, re-arm, serve.
Serve(g) ==
    /\ pc[g] = "building"
    /\ build = "done"
    /\ slots' = slots + 1
    /\ timer' = "armed"
    /\ pc' = [pc EXCEPT ![g] = "done"]
    /\ result' = [result EXCEPT ![g] = IF art = "ready" THEN "ok" ELSE "500"]
    /\ UNCHANGED <<live, art, build, freed, waiting>>

\* Another scan ends and frees the slot.
Free ==
    /\ ~freed
    /\ freed' = TRUE
    /\ slots' = slots + 1
    /\ UNCHANGED <<live, timer, art, build, waiting, pc, result>>

\* The cleanup task wakes after the TTL.
Fire ==
    /\ timer = "armed"
    /\ timer' = "off"
    /\ IF Patched /\ live /\ waiting > 0
       THEN UNCHANGED <<live, art>>
       ELSE /\ live' = FALSE
            /\ art' = IF live /\ build = "none" THEN "failed" ELSE art
    /\ UNCHANGED <<build, slots, freed, waiting, pc, result>>

Next ==
    \/ \E g \in Gets : Start(g) \/ Reject(g) \/ Admit(g) \/ Serve(g)
    \/ Finish
    \/ Free
    \/ Fire

Spec == Init /\ [][Next]_vars

\* STR-1: a GET that was admitted serves the artifact, not a 500.
NoFailedServe == \A g \in Gets : result[g] # "500"

\* STR-2: once nothing is in flight, the stream is expired or about to be.
Quiet == /\ \A g \in Gets : pc[g] \in {"idle", "done"}
         /\ build # "running"
NeverStranded == Quiet => (~live \/ timer = "armed")

=============================================================================

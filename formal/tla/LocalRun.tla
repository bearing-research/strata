-------------------------------- MODULE LocalRun --------------------------------
(***************************************************************************)
(* Model of one local cell run holding its build's lease.                 *)
(*                                                                         *)
(* A run (process p1) is a ledger build whose current attempt it opened.  *)
(* The cell's checkpoints and its final outputs are releases, each fenced  *)
(* on the attempt: the build is building, the attempt is its current one,  *)
(* and the lease is live (vestigo's lease_held). Two threads of p1 touch   *)
(* the lease: the event thread, which releases (LocalRuns.handle, and      *)
(* finalize from the executor), and the renewal task (LocalRuns._renew).   *)
(*                                                                         *)
(* The lease is wall-clock. It can lapse while nothing is wrong: a laptop  *)
(* asleep, or an event loop blocked past the lease (the outputs are stored *)
(* synchronously on the loop). Lapse may fire at any time, except from a   *)
(* hold to its release: the model assumes those few statements and one     *)
(* transaction take far less than a lease. A lapsed lease stays lapsed    *)
(* (renew_lease); the build goes on only in a new attempt (open_attempt),  *)
(* which is refused while another attempt's lease is live.                 *)
(*                                                                         *)
(* A rival process p2 runs the same computation: it finds the build with   *)
(* a lapsed lease and resumes it in its own attempt (LocalRuns.open).      *)
(*                                                                         *)
(* Patched = FALSE: the code before the fix. A release goes straight to    *)
(* the fence, and the renewal task stops at the first failed renewal.     *)
(* Patched = TRUE: before each release the run holds its lease: it renews, *)
(* and if the lease lapsed and nobody took the build, continues in a new   *)
(* attempt; the renewal task does the same. Locked serializes the two      *)
(* threads' holds (a lock on the run).                                     *)
(*                                                                         *)
(* Mirrors:                                                                *)
(*   strata/notebook/local_runs.py::LocalRuns.open                         *)
(*   strata/notebook/local_runs.py::LocalRuns.handle                       *)
(*   strata/notebook/local_runs.py::LocalRuns.finalize                     *)
(*   strata/notebook/local_runs.py::LocalRuns._renew                       *)
(*   strata/notebook/local_runs.py::LocalRuns._hold                        *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    Releases,   \* the run's releases: checkpoints, then the final one
    MaxAtt,     \* bound on attempts opened
    MaxLapse,   \* bound on lapses
    Rival,      \* TRUE: a second process may resume the build
    Patched,    \* TRUE: hold the lease before each release (the fix)
    Locked      \* TRUE: the two threads' holds are serialized

None == "none"
Atts == 1..MaxAtt

VARIABLES
    build,    \* "building" | "ready" | "failed"
    cur,      \* the build's current attempt
    live,     \* its lease is live
    who,      \* who[a] : the process that opened attempt a
    made,     \* attempts opened so far
    runAtt,   \* p1's run.attempt
    ev,       \* event thread: "work" | "renew" | "open" | "assign" | "release" | "done" | "killed"
    evNew,    \* the attempt the event thread opened, not yet assigned
    done,     \* releases p1 has made
    ren,      \* renewal task: "sleep" | "open" | "assign" | "stopped"
    renNew,   \* the attempt the renewal task opened, not yet assigned
    lock,     \* "none" | "ev" | "ren"
    p2,       \* rival: "idle" | "found" | "done"
    p2Took,   \* the rival opened an attempt of the build
    lapses

vars == <<build, cur, live, who, made, runAtt, ev, evNew, done, ren, renNew, lock,
          p2, p2Took, lapses>>

TypeOK ==
    /\ build \in {"building", "ready", "failed"}
    /\ cur \in Atts
    /\ live \in BOOLEAN
    /\ who \in [Atts -> {"p1", "p2", None}]
    /\ made \in Atts
    /\ runAtt \in Atts
    /\ ev \in {"work", "renew", "open", "assign", "release", "done", "killed"}
    /\ evNew \in 0..MaxAtt /\ renNew \in 0..MaxAtt
    /\ done \in 0..Releases
    /\ ren \in {"sleep", "open", "assign", "stopped"}
    /\ lock \in {"none", "ev", "ren"}
    /\ p2 \in {"idle", "found", "done"}
    /\ p2Took \in BOOLEAN
    /\ lapses \in 0..MaxLapse

\* LocalRuns.open has made attempt 1 the build's, with a live lease.
Init ==
    /\ build = "building" /\ cur = 1 /\ live = TRUE
    /\ who = [a \in Atts |-> IF a = 1 THEN "p1" ELSE None]
    /\ made = 1 /\ runAtt = 1
    /\ ev = "work" /\ evNew = 0 /\ done = 0
    /\ ren = "sleep" /\ renNew = 0 /\ lock = "none"
    /\ p2 = "idle" /\ p2Took = FALSE /\ lapses = 0

Running == ev \notin {"done", "killed"}

\* renew_lease: only the owner (the process, not the attempt), only while live.
Renews(p) == build = "building" /\ live /\ who[cur] = p

\* open_attempt with a new attempt id: refused while a lease is live.
OpenFor(p) ==
    /\ build = "building" /\ ~live /\ made < MaxAtt
    /\ made' = made + 1
    /\ cur' = made + 1
    /\ who' = [who EXCEPT ![made + 1] = p]
    /\ live' = TRUE

Free(t) == ~Locked \/ lock = "none" \/ lock = t
Take(t) == lock' = IF Locked THEN t ELSE "none"
Give == lock' = "none"

\* --- Event thread: work, then a release (a checkpoint, or the final one) ---

Work ==
    /\ ev = "work" /\ done < Releases
    /\ IF Patched
       THEN /\ Free("ev") /\ Take("ev") /\ ev' = "renew"
       ELSE /\ ev' = "release" /\ UNCHANGED lock
    /\ UNCHANGED <<build, cur, live, who, made, runAtt, evNew, done, ren, renNew,
                   p2, p2Took, lapses>>

EvRenew ==                                     \* _hold: renew_lease
    /\ ev = "renew"
    /\ IF Renews("p1")
       THEN ev' = "release" /\ Give
       ELSE ev' = "open" /\ UNCHANGED lock
    /\ UNCHANGED <<build, cur, live, who, made, runAtt, evNew, done, ren, renNew,
                   p2, p2Took, lapses>>

EvOpen ==                                      \* _hold: open_attempt, a new attempt
    /\ ev = "open"
    /\ \/ /\ OpenFor("p1") /\ evNew' = made + 1 /\ ev' = "assign" /\ UNCHANGED lock
       \/ /\ ~(build = "building" /\ ~live)     \* refused: the release will be fenced
          /\ ev' = "release" /\ Give
          /\ UNCHANGED <<cur, live, who, made, evNew>>
    /\ UNCHANGED <<build, runAtt, done, ren, renNew, p2, p2Took, lapses>>

EvAssign ==                                    \* _hold: run.attempt = the new attempt
    /\ ev = "assign"
    /\ runAtt' = evNew /\ evNew' = 0 /\ ev' = "release" /\ Give
    /\ UNCHANGED <<build, cur, live, who, made, done, ren, renNew, p2, p2Took, lapses>>

EvRelease ==                                   \* prepare .. accept, fenced on lease_held
    /\ ev = "release"
    /\ IF build = "building" /\ cur = runAtt /\ live
       THEN /\ done' = done + 1
            /\ IF done + 1 = Releases
               THEN build' = "ready" /\ ev' = "done"
               ELSE ev' = "work" /\ UNCHANGED build
       ELSE /\ ev' = "killed"                  \* RunFenced: the harness is killed
            /\ UNCHANGED <<build, done>>
    /\ UNCHANGED <<cur, live, who, made, runAtt, evNew, ren, renNew, lock, p2, p2Took,
                   lapses>>

\* --- Renewal task ---

RenTick ==                                     \* _renew: every 20 s
    /\ ren = "sleep" /\ Running /\ Free("ren")
    /\ IF Renews("p1")
       THEN UNCHANGED <<ren, lock>>
       ELSE IF Patched
            THEN ren' = "open" /\ Take("ren")
            ELSE ren' = "stopped" /\ UNCHANGED lock
    /\ UNCHANGED <<build, cur, live, who, made, runAtt, ev, evNew, done, renNew,
                   p2, p2Took, lapses>>

RenOpen ==
    /\ ren = "open"
    /\ \/ /\ OpenFor("p1") /\ renNew' = made + 1 /\ ren' = "assign" /\ UNCHANGED lock
       \/ /\ ~(build = "building" /\ ~live)
          /\ ren' = "sleep" /\ Give
          /\ UNCHANGED <<cur, live, who, made, renNew>>
    /\ UNCHANGED <<build, runAtt, ev, evNew, done, p2, p2Took, lapses>>

RenAssign ==
    /\ ren = "assign"
    /\ runAtt' = renNew /\ renNew' = 0 /\ ren' = "sleep" /\ Give
    /\ UNCHANGED <<build, cur, live, who, made, ev, evNew, done, p2, p2Took, lapses>>

\* --- The world ---

Lapse ==                                       \* asleep, or a blocked loop, past the lease
    /\ build = "building" /\ live /\ lapses < MaxLapse
    \* From a hold's renewal or new attempt to its release is a few statements and
    \* one transaction, far shorter than a lease.
    /\ ev \notin {"open", "assign", "release"} /\ ren # "assign"
    /\ live' = FALSE /\ lapses' = lapses + 1
    /\ UNCHANGED <<build, cur, who, made, runAtt, ev, evNew, done, ren, renNew, lock,
                   p2, p2Took>>

RivalFind ==                                   \* open: find_open_build, lease not live
    /\ Rival /\ p2 = "idle" /\ build = "building" /\ ~live
    /\ p2' = "found"
    /\ UNCHANGED <<build, cur, live, who, made, runAtt, ev, evNew, done, ren, renNew,
                   lock, p2Took, lapses>>

RivalResume ==                                 \* releases.resume: under the build's key
    /\ p2 = "found"
    /\ \/ /\ OpenFor("p2") /\ p2Took' = TRUE
       \/ /\ ~(build = "building" /\ ~live)     \* ValueError: a lease is live again
          /\ UNCHANGED <<cur, live, who, made, p2Took>>
    /\ p2' = "done"
    /\ UNCHANGED <<build, runAtt, ev, evNew, done, ren, renNew, lock, lapses>>

Next ==
    \/ Work \/ EvRenew \/ EvOpen \/ EvAssign \/ EvRelease
    \/ RenTick \/ RenOpen \/ RenAssign
    \/ Lapse \/ RivalFind \/ RivalResume

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
\* The run is fenced only when another process took its build. A lease that
\* lapsed with nobody else wanting the build is no reason to lose the run.
NoNeedlessFence == ev = "killed" => p2Took

\* The run never names an attempt that is not its own.
RunAttemptIsOwn == who[runAtt] = "p1"

\* Only one process has an attempt with a live lease: the build's current one.
OneLiveHolder == live => who[cur] # None
=============================================================================

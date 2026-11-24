------------------------------- MODULE LocalOpen -------------------------------
(***************************************************************************)
(* Model of LocalRuns.open in two processes on one store (LR-2).           *)
(*                                                                         *)
(* Each process runs the same computation. open reads the computation's   *)
(* newest open build (find_open_build) and acts on what it read, in later  *)
(* transactions: a build whose lease was live at the read refuses the run; *)
(* a lapsed one is resumed (a normal run with a checkpoint) or failed (a   *)
(* rerun), each write pinned to the claim read, and on a miss the build is *)
(* read again. With nothing open, the process opens a build of its own.    *)
(*                                                                         *)
(* Atomic = FALSE: the code before the fix. The open is create_build, a    *)
(* pending build no read finds, then open_attempt: two transactions, and   *)
(* neither looks for a build opened since the read. Atomic = TRUE:         *)
(* vestigo's open_build, one transaction under the computation's lock that *)
(* creates and opens the build unless one is open, and then returns that   *)
(* one; open reads it again as one found.                                  *)
(*                                                                         *)
(* A lease may lapse at any time (a laptop asleep, a loop blocked).        *)
(*                                                                         *)
(* Mirrors:                                                                *)
(*   strata/notebook/local_runs.py::LocalRuns.open                         *)
(*   strata/notebook/local_runs.py::LocalRuns._new_build                   *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Atomic,     \* TRUE: the create and the open are one locked transaction (the fix)
    MaxB,       \* bound on builds created
    MaxLapse    \* bound on lapses

Procs == {"p1", "p2"}
None == "none"
Builds == 1..MaxB

VARIABLES
    state,    \* state[b]: "none" | "pending" | "building" | "failed"
    live,     \* live[b]: its lease is live
    holder,   \* holder[b]: the process whose attempt is current
    made,     \* builds created so far
    pc,       \* pc[p]: "read" | "act" | "open" | "running" | "refused"
    seen,     \* seen[p]: the build p's read found, 0 for none
    seenHolder, seenLive,  \* the claim p read
    mine,     \* mine[p]: the pending build p created and has not opened
    lapses

vars == <<state, live, holder, made, pc, seen, seenHolder, seenLive, mine, lapses>>

TypeOK ==
    /\ state \in [Builds -> {"none", "pending", "building", "failed"}]
    /\ live \in [Builds -> BOOLEAN]
    /\ holder \in [Builds -> Procs \cup {None}]
    /\ made \in 0..MaxB
    /\ pc \in [Procs -> {"read", "act", "open", "running", "refused"}]
    /\ seen \in [Procs -> 0..MaxB]
    /\ seenHolder \in [Procs -> Procs \cup {None}]
    /\ seenLive \in [Procs -> BOOLEAN]
    /\ mine \in [Procs -> 0..MaxB]
    /\ lapses \in 0..MaxLapse

Init ==
    /\ state = [b \in Builds |-> "none"]
    /\ live = [b \in Builds |-> FALSE]
    /\ holder = [b \in Builds |-> None]
    /\ made = 0
    /\ pc = [p \in Procs |-> "read"]
    /\ seen = [p \in Procs |-> 0]
    /\ seenHolder = [p \in Procs |-> None]
    /\ seenLive = [p \in Procs |-> FALSE]
    /\ mine = [p \in Procs |-> 0]
    /\ lapses = 0

Open == {b \in Builds : state[b] = "building"}
\* find_open_build: the newest building build, 0 for none.
Newest == IF Open = {} THEN 0 ELSE CHOOSE b \in Open : \A c \in Open : c <= b

Read(p) ==
    /\ pc[p] = "read"
    /\ seen' = [seen EXCEPT ![p] = Newest]
    /\ seenHolder' = [seenHolder EXCEPT ![p] = IF Newest = 0 THEN None ELSE holder[Newest]]
    /\ seenLive' = [seenLive EXCEPT ![p] = Newest # 0 /\ live[Newest]]
    /\ pc' = [pc EXCEPT ![p] = "act"]
    /\ UNCHANGED <<state, live, holder, made, mine, lapses>>

\* The write is pinned to the claim read: still open, the same holder, still lapsed.
Pinned(p, b) == state[b] = "building" /\ holder[b] = seenHolder[p] /\ ~live[b]

Refuse(p) ==
    /\ pc[p] = "act" /\ seen[p] # 0 /\ seenLive[p]
    /\ pc' = [pc EXCEPT ![p] = "refused"]
    /\ UNCHANGED <<state, live, holder, made, seen, seenHolder, seenLive, mine, lapses>>

\* releases.resume: a normal run continues the lapsed build in its own attempt.
Resume(p) ==
    LET b == seen[p] IN
    /\ pc[p] = "act" /\ b # 0 /\ ~seenLive[p]
    /\ IF Pinned(p, b)
       THEN /\ live' = [live EXCEPT ![b] = TRUE]
            /\ holder' = [holder EXCEPT ![b] = p]
            /\ pc' = [pc EXCEPT ![p] = "running"]
       ELSE /\ pc' = [pc EXCEPT ![p] = "read"]
            /\ UNCHANGED <<live, holder>>
    /\ UNCHANGED <<state, made, seen, seenHolder, seenLive, mine, lapses>>

\* fail_build: a rerun fails the lapsed build, then reads again.
Supersede(p) ==
    LET b == seen[p] IN
    /\ pc[p] = "act" /\ b # 0 /\ ~seenLive[p]
    /\ state' = IF Pinned(p, b) THEN [state EXCEPT ![b] = "failed"] ELSE state
    /\ pc' = [pc EXCEPT ![p] = "read"]
    /\ UNCHANGED <<live, holder, made, seen, seenHolder, seenLive, mine, lapses>>

\* Nothing was open at the read: open a build of this process's own.
Create(p) ==
    /\ pc[p] = "act" /\ seen[p] = 0
    /\ IF Atomic /\ Newest # 0
       THEN \* open_build returned the other's build: read it as one found.
            /\ pc' = [pc EXCEPT ![p] = "read"]
            /\ UNCHANGED <<state, live, holder, made, mine>>
       ELSE /\ made < MaxB
            /\ made' = made + 1
            /\ IF Atomic
               THEN /\ state' = [state EXCEPT ![made + 1] = "building"]
                    /\ live' = [live EXCEPT ![made + 1] = TRUE]
                    /\ holder' = [holder EXCEPT ![made + 1] = p]
                    /\ pc' = [pc EXCEPT ![p] = "running"]
                    /\ UNCHANGED mine
               ELSE \* create_build: pending, which no read finds.
                    /\ state' = [state EXCEPT ![made + 1] = "pending"]
                    /\ mine' = [mine EXCEPT ![p] = made + 1]
                    /\ pc' = [pc EXCEPT ![p] = "open"]
                    /\ UNCHANGED <<live, holder>>
    /\ UNCHANGED <<seen, seenHolder, seenLive, lapses>>

\* open_attempt on the pending build: its first attempt, live.
OpenAttempt(p) ==
    LET b == mine[p] IN
    /\ pc[p] = "open"
    /\ state' = [state EXCEPT ![b] = "building"]
    /\ live' = [live EXCEPT ![b] = TRUE]
    /\ holder' = [holder EXCEPT ![b] = p]
    /\ mine' = [mine EXCEPT ![p] = 0]
    /\ pc' = [pc EXCEPT ![p] = "running"]
    /\ UNCHANGED <<made, seen, seenHolder, seenLive, lapses>>

Lapse(b) ==
    /\ state[b] = "building" /\ live[b] /\ lapses < MaxLapse
    /\ live' = [live EXCEPT ![b] = FALSE]
    /\ lapses' = lapses + 1
    /\ UNCHANGED <<state, holder, made, pc, seen, seenHolder, seenLive, mine>>

Next ==
    \/ \E p \in Procs :
        Read(p) \/ Refuse(p) \/ Resume(p) \/ Supersede(p) \/ Create(p) \/ OpenAttempt(p)
    \/ \E b \in Builds : Lapse(b)

Spec == Init /\ [][Next]_vars

\* LR-2: a computation has at most one open build, so one run of it, not two.
OneOpenBuild == Cardinality({b \in Builds : state[b] \in {"pending", "building"}}) <= 1
================================================================================

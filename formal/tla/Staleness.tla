-------------------------------- MODULE Staleness --------------------------------
(***************************************************************************)
(* Model of notebook cell status under edits and runs.                    *)
(*                                                                         *)
(* A chain of cells, each reading its upstream's latest result. Mirrors    *)
(* notebook/ws.py and session.py:                                          *)
(*                                                                         *)
(*   Edit   : cell_source_update. Refused for the cell that is executing   *)
(*            (running_cell / requested_cell), allowed for any other cell; *)
(*            then compute_staleness_async() applies the walk to every     *)
(*            cell's status (_apply_staleness_map).                        *)
(*   Start  : one execution at a time. A cell whose upstream is not ready  *)
(*            gets a cascade, which runs the upstream first, so a run      *)
(*            starts only on a ready upstream. The cell reads its          *)
(*            upstream's current result when it starts (_load_input_blobs). *)
(*   Finish : the run stores its result, keyed on what it read. Then       *)
(*            _refresh_and_broadcast_changed_staleness runs the walk and,  *)
(*            through preserve_ready_cell_id, marks the cell READY.        *)
(*                                                                         *)
(* The walk (_compute_staleness_locked) is modelled by what it decides: a  *)
(* cell is ready when its latest result was computed from its current      *)
(* source and from its upstream's latest result, and that upstream is      *)
(* ready too. Source versions only grow, so an edit never reverts to a     *)
(* source that already has a cached result.                                *)
(***************************************************************************)
EXTENDS Naturals

CONSTANTS
    MaxSrc,     \* bound on edits per cell
    MaxRuns,    \* bound on runs per cell
    Patched     \* TRUE = the proposed fix

None == "none"

\* The chain a -> b -> c (b reads a, c reads b).
Cells == {"a", "b", "c"}
Up == [x \in Cells |-> CASE x = "a" -> None [] x = "b" -> "a" [] x = "c" -> "b"]
Depth == 3
NoArt == [s |-> 0, up |-> 0, ok |-> FALSE]

VARIABLES
    src,        \* src[x] : source version
    gen,        \* gen[x] : successful runs so far (x's artifact version)
    art,        \* art[x] : what x's latest result was computed from
    status,     \* status[x] : what the session reports
    running,    \* the cell being executed, or None
    snap        \* what the running cell read when it started

vars == <<src, gen, art, status, running, snap>>

TypeOK ==
    /\ src \in [Cells -> 0..MaxSrc]
    /\ gen \in [Cells -> 0..MaxRuns]
    /\ status \in [Cells -> {"idle", "ready", "stale", "running"}]
    /\ running \in Cells \union {None}

\* Is x's latest result current, given source versions S, run counts G and
\* results A? Unrolled Depth times, which is exact for chains that long.
RECURSIVE FreshIn(_, _, _, _, _)
FreshIn(S, G, A, x, n) ==
    /\ A[x].ok
    /\ A[x].s = S[x]
    /\ \/ Up[x] = None
       \/ /\ n > 0
          /\ A[x].up = G[Up[x]]
          /\ FreshIn(S, G, A, Up[x], n - 1)

\* The walk's verdict for x: ready if current, stale if it holds an older
\* result, idle if it never ran.
WalkIn(S, G, A, x) ==
    IF FreshIn(S, G, A, x, Depth) THEN "ready"
    ELSE IF G[x] > 0 THEN "stale" ELSE "idle"

Fresh(x) == FreshIn(src, gen, art, x, Depth)

\* _apply_staleness_map writes every cell, the running one included.
\* Patched, a running cell keeps "running".
Applied(S, G, A, run) ==
    [x \in Cells |-> IF Patched /\ x = run THEN "running" ELSE WalkIn(S, G, A, x)]

Init ==
    /\ src = [x \in Cells |-> 0]
    /\ gen = [x \in Cells |-> 0]
    /\ art = [x \in Cells |-> NoArt]
    /\ status = [x \in Cells |-> "idle"]
    /\ running = None
    /\ snap = NoArt

Edit(x) ==
    LET S == [src EXCEPT ![x] = @ + 1] IN
    /\ x # running
    /\ src[x] < MaxSrc
    /\ src' = S
    /\ status' = Applied(S, gen, art, running)
    /\ UNCHANGED <<gen, art, running, snap>>

Start(x) ==
    /\ running = None
    /\ gen[x] < MaxRuns
    /\ Up[x] # None => status[Up[x]] = "ready"
    /\ running' = x
    /\ snap' = [s |-> src[x], up |-> IF Up[x] = None THEN 0 ELSE gen[Up[x]], ok |-> TRUE]
    /\ status' = [status EXCEPT ![x] = "running"]
    /\ UNCHANGED <<src, gen, art>>

\* The run stores its result, the walk runs, and the cell is preserved as
\* READY. Patched, the walk's verdict for the cell stands.
Finish ==
    LET x == running
        G == [gen EXCEPT ![x] = @ + 1]
        A == [art EXCEPT ![x] = snap]
        walked == Applied(src, G, A, None)
    IN
    /\ running # None
    /\ gen' = G
    /\ art' = A
    /\ running' = None
    /\ snap' = NoArt
    /\ status' = IF Patched THEN walked ELSE [walked EXCEPT ![x] = "ready"]
    /\ UNCHANGED src

Next ==
    \/ Finish
    \/ \E x \in Cells : Edit(x) \/ Start(x)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)

\* A cell shown READY holds a result computed from its current source and
\* its upstream's current result. This is what the UI, GET /cells, agents
\* and the impact preview report. (Execution itself does not rely on it:
\* the executor re-checks provenance and rebuilds stale upstreams.)
ReadyMeansCurrent == \A x \in Cells : status[x] = "ready" => Fresh(x)

\* The cell being executed is shown as running.
RunningShown == running # None => status[running] = "running"
=============================================================================

-------------------------------- MODULE Staleness --------------------------------
(***************************************************************************)
(* Model of notebook cell status under edits and runs.                    *)
(*                                                                         *)
(* A chain of cells, each reading its upstream's latest result. Mirrors    *)
(* notebook/ws.py, routes.py, ops.py and session.py:                       *)
(*                                                                         *)
(*   WsEdit  : cell_source_update. Refused for the cell that is executing  *)
(*             or requested (running_cell / requested_cell); then          *)
(*             compute_staleness_async() applies the walk to every cell's  *)
(*             status (_apply_staleness_map), leaving the running cell     *)
(*             alone when KeepRunning.                                     *)
(*   ApiEdit : PUT /cells/{id} and MCP edit_cell, when ApiEdits. Guarded   *)
(*             like WsEdit only when ApiGuarded; otherwise any cell, and   *)
(*             the walk is applied to every cell, the running one too.     *)
(*   Start   : one execution at a time. A cell whose upstream is not ready *)
(*             gets a cascade, which runs the upstream first, so a run     *)
(*             starts only on a ready upstream (the code also runs stale   *)
(*             upstreams inline; the model leaves that out). The cell      *)
(*             reads its source and its upstream's current result when it  *)
(*             starts (_load_input_blobs).                                 *)
(*   Plan / BatchStart : Run All (Batch). Planning freezes every cell's    *)
(*             source; the cells then run in order, each on its frozen     *)
(*             source and its upstream's result. Only the first cell is    *)
(*             requested, so the others stay editable while they wait.     *)
(*   Finish  : the run stores its result, keyed on what it read. Then      *)
(*             _refresh_and_broadcast_changed_staleness runs the walk and  *)
(*             sets the cell's status by Rule:                             *)
(*               "ready"    : ready (before finding 11's fix);             *)
(*               "upstream" : ready unless an upstream is not ready;       *)
(*               "walk"     : the walk's verdict.                          *)
(*                                                                         *)
(* The walk (_compute_staleness_locked) is modelled by what it decides: a  *)
(* cell is ready when its latest result was computed from its current      *)
(* source and from its upstream's latest result, and that upstream is      *)
(* ready too. Source versions only grow, so an edit never reverts to a     *)
(* source that already has a cached result. The model has no uncached      *)
(* leaves, for which the walk says idle after a run and the code keeps     *)
(* the cell ready.                                                         *)
(*                                                                         *)
(* Mirrors:                                                                *)
(*   strata/notebook/ws.py::_handle_cell_source_update                     *)
(*   strata/notebook/routes.py::update_cell_source                         *)
(*   strata/notebook/ops.py::LocalNotebookOps.edit_cell                    *)
(*   strata/notebook/ws.py::_execute_run_all                               *)
(*   strata/notebook/ws.py::_refresh_and_broadcast_changed_staleness       *)
(*   strata/notebook/session.py::NotebookSession.compute_staleness_async   *)
(*   strata/notebook/session.py::NotebookSession._compute_staleness_locked *)
(*   strata/notebook/session.py::NotebookSession._apply_staleness_map      *)
(*   strata/notebook/executor.py::CellExecutor._load_input_blobs           *)
(***************************************************************************)
EXTENDS Naturals, Sequences

CONSTANTS
    MaxSrc,       \* bound on edits per cell
    MaxRuns,      \* bound on runs per cell
    MaxBatch,     \* bound on Run Alls
    KeepRunning,  \* a guarded edit's recompute leaves the running cell alone
    ApiEdits,     \* REST / MCP edits happen
    ApiGuarded,   \* ... and are refused for the running or requested cell
    Batch,        \* Run All happens
    Rule          \* how a finished run's status is set: "ready" | "upstream" | "walk"

None == "none"

\* The chain a -> b -> c (b reads a, c reads b).
Cells == {"a", "b", "c"}
Order == <<"a", "b", "c">>
Up == [x \in Cells |-> CASE x = "a" -> None [] x = "b" -> "a" [] x = "c" -> "b"]
Depth == 3
NoArt == [s |-> 0, up |-> 0, ok |-> FALSE]

VARIABLES
    src,        \* src[x] : source version
    gen,        \* gen[x] : successful runs so far (x's artifact version)
    art,        \* art[x] : what x's latest result was computed from
    status,     \* status[x] : what the session reports
    running,    \* the cell being executed, or None
    snap,       \* what the running cell read when it started
    queue,      \* Run All's cells still to run, in order
    frozen,     \* the sources Run All planned with
    requested,  \* Run All's requested cell, or None
    batches     \* Run Alls so far

vars == <<src, gen, art, status, running, snap, queue, frozen, requested, batches>>

TypeOK ==
    /\ src \in [Cells -> 0..MaxSrc]
    /\ gen \in [Cells -> 0..MaxRuns]
    /\ status \in [Cells -> {"idle", "ready", "stale", "running"}]
    /\ running \in Cells \union {None}
    /\ requested \in Cells \union {None}
    /\ batches \in 0..MaxBatch

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

\* _apply_staleness_map: the walk's verdict for every cell not in Keep.
Applied(S, G, A, Keep) ==
    [x \in Cells |-> IF x \in Keep THEN status[x] ELSE WalkIn(S, G, A, x)]

Busy == {running, requested} \ {None}

Init ==
    /\ src = [x \in Cells |-> 0]
    /\ gen = [x \in Cells |-> 0]
    /\ art = [x \in Cells |-> NoArt]
    /\ status = [x \in Cells |-> "idle"]
    /\ running = None
    /\ snap = NoArt
    /\ queue = <<>>
    /\ frozen = src
    /\ requested = None
    /\ batches = 0

EditWith(x, keep) ==
    LET S == [src EXCEPT ![x] = @ + 1] IN
    /\ src[x] < MaxSrc
    /\ src' = S
    /\ status' = Applied(S, gen, art, keep)
    /\ UNCHANGED <<gen, art, running, snap, queue, frozen, requested, batches>>

Running == IF running = None THEN {} ELSE {running}

WsEdit(x) ==
    /\ x \notin Busy
    /\ EditWith(x, IF KeepRunning THEN Running ELSE {})

ApiEdit(x) ==
    /\ ApiEdits
    /\ ApiGuarded => x \notin Busy
    /\ EditWith(x, IF ApiGuarded THEN Running ELSE {})

Reads(x, s) == [s |-> s, up |-> IF Up[x] = None THEN 0 ELSE gen[Up[x]], ok |-> TRUE]

Start(x) ==
    /\ running = None /\ queue = <<>>
    /\ gen[x] < MaxRuns
    /\ Up[x] # None => status[Up[x]] = "ready"
    /\ running' = x
    /\ snap' = Reads(x, src[x])
    /\ status' = [status EXCEPT ![x] = "running"]
    /\ UNCHANGED <<src, gen, art, queue, frozen, requested, batches>>

Plan ==                                        \* notebook_run_all plans one batch
    /\ Batch /\ running = None /\ queue = <<>> /\ batches < MaxBatch
    /\ \A x \in Cells : gen[x] < MaxRuns
    /\ queue' = Order /\ frozen' = src /\ requested' = Head(Order)
    /\ batches' = batches + 1
    /\ UNCHANGED <<src, gen, art, status, running, snap>>

BatchStart ==
    /\ running = None /\ queue # <<>>
    /\ LET x == Head(queue) IN
       /\ running' = x
       /\ snap' = Reads(x, frozen[x])
       /\ status' = [status EXCEPT ![x] = "running"]
    /\ UNCHANGED <<src, gen, art, queue, frozen, requested, batches>>

Finish ==
    LET x == running
        G == [gen EXCEPT ![x] = @ + 1]
        A == [art EXCEPT ![x] = snap]
        walked == Applied(src, G, A, {})
        upstreamOff == Up[x] # None /\ walked[Up[x]] # "ready"
    IN
    /\ running # None
    /\ gen' = G
    /\ art' = A
    /\ running' = None
    /\ snap' = NoArt
    /\ status' = CASE Rule = "ready" -> [walked EXCEPT ![x] = "ready"]
                   [] Rule = "upstream" ->
                        [walked EXCEPT ![x] = IF upstreamOff THEN "stale" ELSE "ready"]
                   [] Rule = "walk" -> walked
    /\ IF queue # <<>> /\ Head(queue) = x
       THEN /\ queue' = Tail(queue)
            /\ requested' = IF Tail(queue) = <<>> THEN None ELSE requested
       ELSE UNCHANGED <<queue, requested>>
    /\ UNCHANGED <<src, frozen, batches>>

Next ==
    \/ Finish \/ Plan \/ BatchStart
    \/ \E x \in Cells : WsEdit(x) \/ ApiEdit(x) \/ Start(x)

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

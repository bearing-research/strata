-------------------------------- MODULE Admission --------------------------------
(***************************************************************************)
(* Model of one tenant's stream admission limiter.                        *)
(*                                                                         *)
(* ResizableLimiter (adaptive_concurrency.py) is an asyncio.Condition     *)
(* guarding (capacity, in_use). acquire() waits on the condition while    *)
(* full, with a timeout whose handler re-checks in_use; release()         *)
(* decrements in_use and notify(1)s one waiter.                           *)
(*                                                                         *)
(* A waiting request can also be cancelled outright: a client disconnect   *)
(* or shutdown while it is queued (QoSAdmission.admit, the #238 path).    *)
(* CPython 3.13+ Condition.wait() re-notifies another waiter when a        *)
(* notified waiter is cancelled; 3.12 does not (NotifyOnCancel).          *)
(*                                                                         *)
(* TenantRegistry.get_or_create_quotas LRU-evicts a tenant's quotas, and    *)
(* the limiters with them, once more than MAX_TRACKED_TENANTS (1000) are   *)
(* tracked. Pressure from the other 1000 tenants is abstracted to an       *)
(* Evict action that may fire at any time. The next admission then builds  *)
(* a new limiter: a new generation here.                                  *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Requests,        \* concurrent scan requests from one tenant
    Capacity,        \* the tenant's slots for this tier
    MaxGen,          \* bound on limiter generations (evictions + 1)
    NotifyOnCancel,  \* TRUE = CPython >= 3.13 Condition semantics
    SafeEviction     \* TRUE = proposed fix: never evict a limiter in use

Gens == 1..MaxGen

VARIABLES
    current,   \* generation the registry hands out (0 = evicted / not created)
    made,      \* generations created so far
    inUse,     \* inUse[g] : ResizableLimiter._in_use of generation g
    st,        \* st[r] : "idle" | "waiting" | "notified" | "holding" | "gone"
    gen        \* gen[r] : generation request r acquired or waits on

vars == <<current, made, inUse, st, gen>>

TypeOK ==
    /\ current \in 0..MaxGen
    /\ made \in 0..MaxGen
    /\ inUse \in [Gens -> 0..Capacity]
    /\ st \in [Requests -> {"idle", "waiting", "notified", "holding", "gone"}]
    /\ gen \in [Requests -> 0..MaxGen]

Init ==
    /\ current = 0
    /\ made = 0
    /\ inUse = [g \in Gens |-> 0]
    /\ st = [r \in Requests |-> "idle"]
    /\ gen = [r \in Requests |-> 0]

WaitingOn(g) == {r \in Requests : st[r] = "waiting" /\ gen[r] = g}
NotifiedOn(g) == {r \in Requests : st[r] = "notified" /\ gen[r] = g}
HeldTotal == LET Sum[S \in SUBSET Gens] ==
                    IF S = {} THEN 0
                    ELSE LET g == CHOOSE x \in S : TRUE IN inUse[g] + Sum[S \ {g}]
             IN Sum[Gens]

\* notify(1): wake one waiter, if any. Which one is up to asyncio (FIFO in
\* practice); the model lets it be any of them.
Notify1(g, stNow) ==
    IF \E w \in Requests : stNow[w] = "waiting" /\ gen[w] = g
    THEN \E w \in {x \in Requests : stNow[x] = "waiting" /\ gen[x] = g} :
            st' = [stNow EXCEPT ![w] = "notified"]
    ELSE st' = stNow

-----------------------------------------------------------------------------
(* get_or_create_limiters + acquire(timeout): take a free slot at once,    *)
(* otherwise wait on the condition.                                        *)
Admit(r) ==
    /\ st[r] = "idle"
    /\ \/ /\ current # 0
          /\ UNCHANGED <<current, made>>
       \/ /\ current = 0 /\ made < MaxGen
          /\ made' = made + 1 /\ current' = made + 1
    /\ LET g == IF current # 0 THEN current ELSE made + 1 IN
       /\ gen' = [gen EXCEPT ![r] = g]
       /\ IF inUse[g] < Capacity
          THEN /\ inUse' = [inUse EXCEPT ![g] = @ + 1]
               /\ st' = [st EXCEPT ![r] = "holding"]
          ELSE /\ st' = [st EXCEPT ![r] = "waiting"]
               /\ UNCHANGED inUse

(* A notified waiter runs: re-acquire the lock and re-check.               *)
Wake(r) ==
    LET g == gen[r] IN
    /\ st[r] = "notified"
    /\ IF inUse[g] < Capacity
       THEN /\ inUse' = [inUse EXCEPT ![g] = @ + 1]
            /\ st' = [st EXCEPT ![r] = "holding"]
       ELSE /\ st' = [st EXCEPT ![r] = "waiting"]
            /\ UNCHANGED inUse
    /\ UNCHANGED <<current, made, gen>>

(* The wait_for deadline passes. The handler re-checks in_use, so a slot   *)
(* that is free now is still taken; otherwise the request gets a 429.      *)
Timeout(r) ==
    LET g == gen[r] IN
    /\ st[r] \in {"waiting", "notified"}
    /\ IF inUse[g] < Capacity
       THEN /\ inUse' = [inUse EXCEPT ![g] = @ + 1]
            /\ st' = [st EXCEPT ![r] = "holding"]
       ELSE /\ st' = [st EXCEPT ![r] = "gone"]
            /\ UNCHANGED inUse
    /\ UNCHANGED <<current, made, gen>>

(* The request task is cancelled while queued (client disconnect).        *)
Cancel(r) ==
    LET g == gen[r]
        st1 == [st EXCEPT ![r] = "gone"]
    IN
    /\ st[r] \in {"waiting", "notified"}
    /\ IF st[r] = "notified" /\ NotifyOnCancel
       THEN Notify1(g, st1)
       ELSE st' = st1
    /\ UNCHANGED <<current, made, inUse, gen>>

(* Admission.release -> limiter.release(): on the limiter it acquired.     *)
Release(r) ==
    LET g == gen[r] IN
    /\ st[r] = "holding"
    /\ inUse' = [inUse EXCEPT ![g] = @ - 1]
    /\ Notify1(g, [st EXCEPT ![r] = "gone"])
    /\ UNCHANGED <<current, made, gen>>

(* LRU eviction of the tenant's quotas, limiters included.                 *)
Evict ==
    /\ current # 0
    /\ SafeEviction => (inUse[current] = 0 /\ WaitingOn(current) \union NotifiedOn(current) = {})
    /\ current' = 0
    /\ UNCHANGED <<made, inUse, st, gen>>

Next ==
    \/ Evict
    \/ \E r \in Requests :
        Admit(r) \/ Wake(r) \/ Timeout(r) \/ Cancel(r) \/ Release(r)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)

\* No lost wakeup: whenever a slot is free and someone is queued for it,
\* at least one queued request has been notified and will re-check.
\* Otherwise every queued request sleeps until its deadline and gets a 429
\* while the slot sits idle.
NoLostWakeup ==
    \A g \in Gens :
        (inUse[g] < Capacity /\ WaitingOn(g) # {}) => NotifiedOn(g) # {}

\* The tenant never holds more slots than its quota.
WithinQuota == HeldTotal <= Capacity

\* active_scan_count() (graceful-shutdown drain) sums only limiters the
\* registry still tracks. It must see every live stream.
DrainSeesAll == HeldTotal = IF current = 0 THEN 0 ELSE inUse[current]
=============================================================================

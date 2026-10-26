--------------------------- MODULE ArtifactLifecycle ---------------------------
(***************************************************************************)
(* Model of the artifact_versions state machine in vestigo/store.py,      *)
(* with the retention roots and required edges of ADR 002 E2.             *)
(*                                                                         *)
(* Each SQL transaction is one atomic action. The notebook write path      *)
(* (artifact_integration.store_cell_output) is two transactions,           *)
(* finalize_artifact and then force_finalize_canonical, so it is two       *)
(* actions and other actions can run between them.                         *)
(*                                                                         *)
(* Abstractions:                                                           *)
(*   - Tenants are ignored (a single tenant '').                           *)
(*   - Names and aliases are ignored, so every version is "unnamed". That  *)
(*     is exactly the case of notebook cell outputs (nb_..._var_...).       *)
(*   - Time is abstracted away: garbage_collect may treat any row as older *)
(*     than max_age_days. This over-approximates; counterexamples need the *)
(*     affected row to be old enough in practice.                          *)
(*   - Blob bytes are a boolean "present" flag.                            *)
(*   - With Retention, a version may be retained (retain/release, any      *)
(*     kind and holder collapse to one root) and may declare at creation   *)
(*     the stored versions it requires (an output folder's files). GC      *)
(*     protects the closure of its roots over required edges.              *)
(***************************************************************************)
EXTENDS Naturals, FiniteSets

CONSTANTS
    Ids,        \* artifact ids, e.g. two notebook cells in two notebooks
    Provs,      \* provenance hashes
    MaxVer,     \* bound on versions per id (keeps the state space finite)
    EnableGC,   \* include garbage_collect in the model
    Patched,    \* FALSE = the code as it is; TRUE = the proposed fix
                \* (see "Proposed fix" in formal/README.md)
    Retention   \* include retain/release and required edges (E2)

Versions == 1..MaxVer
States   == {"absent", "building", "ready", "superseded", "failed"}

VARIABLES
    st,         \* st[i][v]   : row state ("absent" = no row / deleted)
    prov,       \* prov[i][v] : provenance hash of the row
    blob,       \* blob[i][v] : blob bytes present
    pending,    \* set of <<i, v>> whose notebook caller still has to run
                \* force_finalize_canonical (finalize was deduped to a foreign id)
    everReady,  \* everReady[i] : id i has held a ready version at some point
    roots,      \* retention_roots: set of <<i, v>> something retains
    req         \* retention_edges: set of <<<<i, v>>, <<j, w>>>>, i@v requires j@w

vars == <<st, prov, blob, pending, everReady, roots, req>>

Refs == Ids \X Versions

TypeOK ==
    /\ st \in [Ids -> [Versions -> States]]
    /\ prov \in [Ids -> [Versions -> Provs]]
    /\ blob \in [Ids -> [Versions -> BOOLEAN]]
    /\ pending \subseteq (Ids \X Versions)
    /\ everReady \in [Ids -> BOOLEAN]
    /\ roots \subseteq Refs
    /\ req \subseteq Refs \X Refs

Stored == {r \in Refs : st[r[1]][r[2]] # "absent"}

\* Highest version number with a row in any state (0 when none).
\* garbage_collect's "latest version" rule uses MAX(version) over all states.
MaxRowVer(i) ==
    IF \E v \in Versions : st[i][v] # "absent"
    THEN CHOOSE v \in Versions : st[i][v] # "absent" /\
                                \A w \in Versions : st[i][w] # "absent" => w <= v
    ELSE 0

\* States get_latest_version(i) accepts. Today only 'ready'; the patch also
\* accepts 'superseded', which by design stays fetchable by id+version.
Current == IF Patched THEN {"ready", "superseded"} ELSE {"ready"}

\* get_latest_version(i): the highest current version, or 0 for None.
LatestReady(i) ==
    IF \E v \in Versions : st[i][v] \in Current
    THEN CHOOSE v \in Versions : st[i][v] \in Current /\
                                \A w \in Versions : st[i][w] \in Current => w <= v
    ELSE 0

ReadyWithProv(p) == {<<i, v>> \in Refs : st[i][v] = "ready" /\ prov[i][v] = p}

\* garbage_collect's roots (_protected_reachable): retention roots, rows still
\* building, and the current value of every id; a held version holds whatever
\* it requires, so the closure over required edges is protected.
RECURSIVE Reach(_)
Reach(S) ==
    LET next == S \union {e[2] : e \in {e \in req : e[1] \in S}} IN
    IF next = S THEN S ELSE Reach(next)

Held == Reach(roots
              \union {r \in Refs : st[r[1]][r[2]] = "building"}
              \union {<<i, LatestReady(i)>> : i \in {j \in Ids : LatestReady(j) # 0}})

Init ==
    /\ st = [i \in Ids |-> [v \in Versions |-> "absent"]]
    /\ prov \in [Ids -> [Versions -> Provs]]   \* value irrelevant until created
    /\ blob = [i \in Ids |-> [v \in Versions |-> FALSE]]
    /\ pending = {}
    /\ everReady = [i \in Ids |-> FALSE]
    /\ roots = {}
    /\ req = {}

-----------------------------------------------------------------------------
(* create_artifact: next version in 'building', requiring R (stored      *)
(* versions, written as edges in the row's transaction). A consumer may  *)
(* store a folder's files before the manifest that requires them.         *)
Create(i, p, R) ==
    LET v == MaxRowVer(i) + 1 IN
    /\ v \in Versions
    /\ R \subseteq Stored
    /\ Retention \/ R = {}
    /\ st' = [st EXCEPT ![i][v] = "building"]
    /\ prov' = [prov EXCEPT ![i][v] = p]
    /\ req' = req \union {<<<<i, v>>, r>> : r \in R}
    /\ UNCHANGED <<blob, pending, everReady, roots>>

(* retain: a root on a stored version, idempotent per holder.             *)
Retain(i, v) ==
    /\ Retention
    /\ st[i][v] # "absent"
    /\ roots' = roots \union {<<i, v>>}
    /\ UNCHANGED <<st, prov, blob, pending, everReady, req>>

(* release: the holder lets go.                                            *)
Release(i, v) ==
    /\ <<i, v>> \in roots
    /\ roots' = roots \ {<<i, v>>}
    /\ UNCHANGED <<st, prov, blob, pending, everReady, req>>

(* blob_store.write_blob                                                  *)
WriteBlob(i, v) ==
    /\ st[i][v] = "building"
    /\ ~blob[i][v]
    /\ blob' = [blob EXCEPT ![i][v] = TRUE]
    /\ UNCHANGED <<st, prov, pending, everReady, roots, req>>

(* Build error: fail_artifact.                                            *)
Fail(i, v) ==
    /\ st[i][v] = "building"
    /\ st' = [st EXCEPT ![i][v] = "failed"]
    /\ UNCHANGED <<prov, blob, pending, everReady, roots, req>>

(* finalize_artifact, one of three outcomes:                              *)
(*  - another id already holds a ready row with this provenance: mark ours *)
(*    'failed', return the foreign one. The notebook caller then owes a    *)
(*    force_finalize_canonical (recorded in `pending`);                    *)
(*  - an older ready version of the same id has this provenance: supersede *)
(*    it and promote ours (refresh rebuild);                               *)
(*  - otherwise promote ours.                                             *)
Finalize(i, v) ==
    LET p == prov[i][v]
        foreign == {x \in ReadyWithProv(p) : x[1] # i}
        own     == {x \in ReadyWithProv(p) : x[1] = i}
    IN
    /\ st[i][v] = "building"
    /\ blob[i][v]
    /\ IF foreign # {}
       THEN /\ st' = [st EXCEPT ![i][v] = "failed"]
            /\ pending' = pending \union {<<i, v>>}
            /\ UNCHANGED everReady
       ELSE /\ st' = [x \in Ids |-> [w \in Versions |->
                        IF x = i /\ w = v THEN "ready"
                        ELSE IF <<x, w>> \in own THEN "superseded"
                        ELSE st[x][w]]]
            /\ everReady' = [everReady EXCEPT ![i] = TRUE]
            /\ UNCHANGED pending
    /\ UNCHANGED <<prov, blob, roots, req>>

(* force_finalize_canonical: supersede every other ready row with the     *)
(* same provenance, then flip ours failed -> ready.                       *)
ForcePromote(i, v) ==
    LET p == prov[i][v] IN
    /\ <<i, v>> \in pending
    /\ pending' = pending \ {<<i, v>>}
    /\ IF st[i][v] = "failed"
       THEN /\ st' = [x \in Ids |-> [w \in Versions |->
                        IF x = i /\ w = v THEN "ready"
                        ELSE IF st[x][w] = "ready" /\ prov[x][w] = p THEN "superseded"
                        ELSE st[x][w]]]
            /\ everReady' = [everReady EXCEPT ![i] = TRUE]
       ELSE UNCHANGED <<st, everReady>>   \* UPDATE ... WHERE state='failed' hits 0 rows
    /\ UNCHANGED <<prov, blob, roots, req>>

(* garbage_collect (collect_latest=False, no names/aliases): delete any   *)
(* ready/superseded/failed row that is not the MAX(version) of its id,    *)
(* then its blob. The patch additionally protects the latest *current*  *)
(* version. It must keep the MAX(version) rule too: deleting the highest  *)
(* row lets create_artifact (MAX(version)+1) reuse its version number, and *)
(* a still-pending force_finalize_canonical then promotes the new, blobless *)
(* row (TLC finds this if the MAX rule is dropped). With Retention it   *)
(* skips every held version, and deleting a version drops its roots and   *)
(* its outgoing edges (_delete_version_children); edges into it stay, so  *)
(* a requirer that outlived it reads as broken (requirements_met).        *)
GC(i, v) ==
    /\ st[i][v] \in {"ready", "superseded", "failed"}
    /\ v < MaxRowVer(i)
    /\ Patched => v # LatestReady(i)
    /\ Retention => <<i, v>> \notin Held
    /\ st' = [st EXCEPT ![i][v] = "absent"]
    /\ blob' = [blob EXCEPT ![i][v] = FALSE]
    /\ roots' = roots \ {<<i, v>>}
    /\ req' = {e \in req : e[1] # <<i, v>>}
    /\ UNCHANGED <<prov, pending, everReady>>

Next ==
    \/ \E i \in Ids, p \in Provs, R \in (IF Retention THEN SUBSET Refs ELSE {{}}) :
        Create(i, p, R)
    \/ \E i \in Ids, v \in Versions :
        \/ WriteBlob(i, v) \/ Fail(i, v) \/ Finalize(i, v)
        \/ ForcePromote(i, v) \/ (EnableGC /\ GC(i, v))
        \/ Retain(i, v) \/ Release(i, v)

Spec == Init /\ [][Next]_vars

-----------------------------------------------------------------------------
(* Invariants                                                              *)

\* The partial unique index idx_tenant_provenance_unique.
UniqueReadyPerProv ==
    \A p \in Provs : Cardinality(ReadyWithProv(p)) <= 1

\* A servable row always has its bytes.
ServableHasBlob ==
    \A i \in Ids, v \in Versions :
        st[i][v] \in {"ready", "superseded"} => blob[i][v]

\* Once an id has had a current value, get_latest_version(id) keeps
\* returning one. Notebook cells resolve their inputs this way
\* (executor._load_input_blobs), so a violation is an upstream that
\* silently disappears from a downstream cell's namespace. A caller that
\* is still between finalize and force_finalize_canonical is exempt,
\* since it has not returned yet.
CurrentValueDurable ==
    \A i \in Ids :
        (everReady[i] /\ ~\E v \in Versions : <<i, v>> \in pending)
            => LatestReady(i) # 0

\* A root points at a stored version: deleting a version drops its roots.
RootsStored ==
    roots \subseteq Stored

\* No step deletes a held version, so a held output folder never loses a
\* file: what it requires is held with it. Stated over steps rather than
\* states because a caller may retain, or require, a version whose own
\* requirements GC already took while nothing held it (TLC finds that in
\* seven steps); the ledger keeps what is held, and whether a version is
\* whole when a caller holds it is the caller's to check (requirements_met).
HeldIsKept ==
    [][\A r \in Held \intersect Stored : r \in Stored']_vars
=============================================================================
